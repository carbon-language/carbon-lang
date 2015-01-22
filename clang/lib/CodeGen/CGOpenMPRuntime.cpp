//===----- CGOpenMPRuntime.cpp - Interface to OpenMP Runtimes -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides a class for OpenMP runtime code generation.
//
//===----------------------------------------------------------------------===//

#include "CGOpenMPRuntime.h"
#include "CodeGenFunction.h"
#include "clang/AST/Decl.h"
#include "clang/AST/StmtOpenMP.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace clang;
using namespace CodeGen;

namespace {
/// \brief API for captured statement code generation in OpenMP constructs.
class CGOpenMPRegionInfo : public CodeGenFunction::CGCapturedStmtInfo {
public:
  CGOpenMPRegionInfo(const OMPExecutableDirective &D, const CapturedStmt &CS,
                     const VarDecl *ThreadIDVar)
      : CGCapturedStmtInfo(CS, CR_OpenMP), ThreadIDVar(ThreadIDVar),
        Directive(D) {
    assert(ThreadIDVar != nullptr && "No ThreadID in OpenMP region.");
  }

  /// \brief Gets a variable or parameter for storing global thread id
  /// inside OpenMP construct.
  const VarDecl *getThreadIDVariable() const { return ThreadIDVar; }

  /// \brief Gets an LValue for the current ThreadID variable.
  LValue getThreadIDVariableLValue(CodeGenFunction &CGF);

  static bool classof(const CGCapturedStmtInfo *Info) {
    return Info->getKind() == CR_OpenMP;
  }

  /// \brief Emit the captured statement body.
  void EmitBody(CodeGenFunction &CGF, Stmt *S) override;

  /// \brief Get the name of the capture helper.
  StringRef getHelperName() const override { return ".omp_outlined."; }

private:
  /// \brief A variable or parameter storing global thread id for OpenMP
  /// constructs.
  const VarDecl *ThreadIDVar;
  /// \brief OpenMP executable directive associated with the region.
  const OMPExecutableDirective &Directive;
};
} // namespace

LValue CGOpenMPRegionInfo::getThreadIDVariableLValue(CodeGenFunction &CGF) {
  return CGF.MakeNaturalAlignAddrLValue(
      CGF.GetAddrOfLocalVar(ThreadIDVar),
      CGF.getContext().getPointerType(ThreadIDVar->getType()));
}

void CGOpenMPRegionInfo::EmitBody(CodeGenFunction &CGF, Stmt *S) {
  CodeGenFunction::OMPPrivateScope PrivateScope(CGF);
  CGF.EmitOMPPrivateClause(Directive, PrivateScope);
  CGF.EmitOMPFirstprivateClause(Directive, PrivateScope);
  if (PrivateScope.Privatize())
    // Emit implicit barrier to synchronize threads and avoid data races.
    CGF.CGM.getOpenMPRuntime().EmitOMPBarrierCall(CGF, Directive.getLocStart(),
                                                  /*IsExplicit=*/false);
  CGCapturedStmtInfo::EmitBody(CGF, S);
}

CGOpenMPRuntime::CGOpenMPRuntime(CodeGenModule &CGM)
    : CGM(CGM), DefaultOpenMPPSource(nullptr) {
  IdentTy = llvm::StructType::create(
      "ident_t", CGM.Int32Ty /* reserved_1 */, CGM.Int32Ty /* flags */,
      CGM.Int32Ty /* reserved_2 */, CGM.Int32Ty /* reserved_3 */,
      CGM.Int8PtrTy /* psource */, nullptr);
  // Build void (*kmpc_micro)(kmp_int32 *global_tid, kmp_int32 *bound_tid,...)
  llvm::Type *MicroParams[] = {llvm::PointerType::getUnqual(CGM.Int32Ty),
                               llvm::PointerType::getUnqual(CGM.Int32Ty)};
  Kmpc_MicroTy = llvm::FunctionType::get(CGM.VoidTy, MicroParams, true);
  KmpCriticalNameTy = llvm::ArrayType::get(CGM.Int32Ty, /*NumElements*/ 8);
}

llvm::Value *
CGOpenMPRuntime::EmitOpenMPOutlinedFunction(const OMPExecutableDirective &D,
                                            const VarDecl *ThreadIDVar) {
  const CapturedStmt *CS = cast<CapturedStmt>(D.getAssociatedStmt());
  CodeGenFunction CGF(CGM, true);
  CGOpenMPRegionInfo CGInfo(D, *CS, ThreadIDVar);
  CGF.CapturedStmtInfo = &CGInfo;
  return CGF.GenerateCapturedStmtFunction(*CS);
}

llvm::Value *
CGOpenMPRuntime::GetOrCreateDefaultOpenMPLocation(OpenMPLocationFlags Flags) {
  llvm::Value *Entry = OpenMPDefaultLocMap.lookup(Flags);
  if (!Entry) {
    if (!DefaultOpenMPPSource) {
      // Initialize default location for psource field of ident_t structure of
      // all ident_t objects. Format is ";file;function;line;column;;".
      // Taken from
      // http://llvm.org/svn/llvm-project/openmp/trunk/runtime/src/kmp_str.c
      DefaultOpenMPPSource =
          CGM.GetAddrOfConstantCString(";unknown;unknown;0;0;;");
      DefaultOpenMPPSource =
          llvm::ConstantExpr::getBitCast(DefaultOpenMPPSource, CGM.Int8PtrTy);
    }
    auto DefaultOpenMPLocation = new llvm::GlobalVariable(
        CGM.getModule(), IdentTy, /*isConstant*/ true,
        llvm::GlobalValue::PrivateLinkage, /*Initializer*/ nullptr);
    DefaultOpenMPLocation->setUnnamedAddr(true);

    llvm::Constant *Zero = llvm::ConstantInt::get(CGM.Int32Ty, 0, true);
    llvm::Constant *Values[] = {Zero,
                                llvm::ConstantInt::get(CGM.Int32Ty, Flags),
                                Zero, Zero, DefaultOpenMPPSource};
    llvm::Constant *Init = llvm::ConstantStruct::get(IdentTy, Values);
    DefaultOpenMPLocation->setInitializer(Init);
    OpenMPDefaultLocMap[Flags] = DefaultOpenMPLocation;
    return DefaultOpenMPLocation;
  }
  return Entry;
}

llvm::Value *CGOpenMPRuntime::EmitOpenMPUpdateLocation(
    CodeGenFunction &CGF, SourceLocation Loc, OpenMPLocationFlags Flags) {
  // If no debug info is generated - return global default location.
  if (CGM.getCodeGenOpts().getDebugInfo() == CodeGenOptions::NoDebugInfo ||
      Loc.isInvalid())
    return GetOrCreateDefaultOpenMPLocation(Flags);

  assert(CGF.CurFn && "No function in current CodeGenFunction.");

  llvm::Value *LocValue = nullptr;
  auto I = OpenMPLocThreadIDMap.find(CGF.CurFn);
  if (I != OpenMPLocThreadIDMap.end())
    LocValue = I->second.DebugLoc;
  // OpenMPLocThreadIDMap may have null DebugLoc and non-null ThreadID, if
  // GetOpenMPThreadID was called before this routine.
  if (LocValue == nullptr) {
    // Generate "ident_t .kmpc_loc.addr;"
    llvm::AllocaInst *AI = CGF.CreateTempAlloca(IdentTy, ".kmpc_loc.addr");
    AI->setAlignment(CGM.getDataLayout().getPrefTypeAlignment(IdentTy));
    auto &Elem = OpenMPLocThreadIDMap.FindAndConstruct(CGF.CurFn);
    Elem.second.DebugLoc = AI;
    LocValue = AI;

    CGBuilderTy::InsertPointGuard IPG(CGF.Builder);
    CGF.Builder.SetInsertPoint(CGF.AllocaInsertPt);
    CGF.Builder.CreateMemCpy(LocValue, GetOrCreateDefaultOpenMPLocation(Flags),
                             llvm::ConstantExpr::getSizeOf(IdentTy),
                             CGM.PointerAlignInBytes);
  }

  // char **psource = &.kmpc_loc_<flags>.addr.psource;
  auto *PSource =
      CGF.Builder.CreateConstInBoundsGEP2_32(LocValue, 0, IdentField_PSource);

  auto OMPDebugLoc = OpenMPDebugLocMap.lookup(Loc.getRawEncoding());
  if (OMPDebugLoc == nullptr) {
    SmallString<128> Buffer2;
    llvm::raw_svector_ostream OS2(Buffer2);
    // Build debug location
    PresumedLoc PLoc = CGF.getContext().getSourceManager().getPresumedLoc(Loc);
    OS2 << ";" << PLoc.getFilename() << ";";
    if (const FunctionDecl *FD =
            dyn_cast_or_null<FunctionDecl>(CGF.CurFuncDecl)) {
      OS2 << FD->getQualifiedNameAsString();
    }
    OS2 << ";" << PLoc.getLine() << ";" << PLoc.getColumn() << ";;";
    OMPDebugLoc = CGF.Builder.CreateGlobalStringPtr(OS2.str());
    OpenMPDebugLocMap[Loc.getRawEncoding()] = OMPDebugLoc;
  }
  // *psource = ";<File>;<Function>;<Line>;<Column>;;";
  CGF.Builder.CreateStore(OMPDebugLoc, PSource);

  return LocValue;
}

llvm::Value *CGOpenMPRuntime::GetOpenMPThreadID(CodeGenFunction &CGF,
                                                SourceLocation Loc) {
  assert(CGF.CurFn && "No function in current CodeGenFunction.");

  llvm::Value *ThreadID = nullptr;
  // Check whether we've already cached a load of the thread id in this
  // function.
  auto I = OpenMPLocThreadIDMap.find(CGF.CurFn);
  if (I != OpenMPLocThreadIDMap.end()) {
    ThreadID = I->second.ThreadID;
    if (ThreadID != nullptr)
      return ThreadID;
  }
  if (auto OMPRegionInfo =
          dyn_cast_or_null<CGOpenMPRegionInfo>(CGF.CapturedStmtInfo)) {
    // Check if this an outlined function with thread id passed as argument.
    auto ThreadIDVar = OMPRegionInfo->getThreadIDVariable();
    auto LVal = OMPRegionInfo->getThreadIDVariableLValue(CGF);
    auto RVal = CGF.EmitLoadOfLValue(LVal, Loc);
    LVal = CGF.MakeNaturalAlignAddrLValue(RVal.getScalarVal(),
                                          ThreadIDVar->getType());
    ThreadID = CGF.EmitLoadOfLValue(LVal, Loc).getScalarVal();
    // If value loaded in entry block, cache it and use it everywhere in
    // function.
    if (CGF.Builder.GetInsertBlock() == CGF.AllocaInsertPt->getParent()) {
      auto &Elem = OpenMPLocThreadIDMap.FindAndConstruct(CGF.CurFn);
      Elem.second.ThreadID = ThreadID;
    }
  } else {
    // This is not an outlined function region - need to call __kmpc_int32
    // kmpc_global_thread_num(ident_t *loc).
    // Generate thread id value and cache this value for use across the
    // function.
    CGBuilderTy::InsertPointGuard IPG(CGF.Builder);
    CGF.Builder.SetInsertPoint(CGF.AllocaInsertPt);
    llvm::Value *Args[] = {EmitOpenMPUpdateLocation(CGF, Loc)};
    ThreadID = CGF.EmitRuntimeCall(
        CreateRuntimeFunction(OMPRTL__kmpc_global_thread_num), Args);
    auto &Elem = OpenMPLocThreadIDMap.FindAndConstruct(CGF.CurFn);
    Elem.second.ThreadID = ThreadID;
  }
  return ThreadID;
}

void CGOpenMPRuntime::FunctionFinished(CodeGenFunction &CGF) {
  assert(CGF.CurFn && "No function in current CodeGenFunction.");
  if (OpenMPLocThreadIDMap.count(CGF.CurFn))
    OpenMPLocThreadIDMap.erase(CGF.CurFn);
}

llvm::Type *CGOpenMPRuntime::getIdentTyPointerTy() {
  return llvm::PointerType::getUnqual(IdentTy);
}

llvm::Type *CGOpenMPRuntime::getKmpc_MicroPointerTy() {
  return llvm::PointerType::getUnqual(Kmpc_MicroTy);
}

llvm::Constant *
CGOpenMPRuntime::CreateRuntimeFunction(OpenMPRTLFunction Function) {
  llvm::Constant *RTLFn = nullptr;
  switch (Function) {
  case OMPRTL__kmpc_fork_call: {
    // Build void __kmpc_fork_call(ident_t *loc, kmp_int32 argc, kmpc_micro
    // microtask, ...);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty,
                                getKmpc_MicroPointerTy()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ true);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_fork_call");
    break;
  }
  case OMPRTL__kmpc_global_thread_num: {
    // Build kmp_int32 __kmpc_global_thread_num(ident_t *loc);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_global_thread_num");
    break;
  }
  case OMPRTL__kmpc_threadprivate_cached: {
    // Build void *__kmpc_threadprivate_cached(ident_t *loc,
    // kmp_int32 global_tid, void *data, size_t size, void ***cache);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty,
                                CGM.VoidPtrTy, CGM.SizeTy,
                                CGM.VoidPtrTy->getPointerTo()->getPointerTo()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidPtrTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_threadprivate_cached");
    break;
  }
  case OMPRTL__kmpc_critical: {
    // Build void __kmpc_critical(ident_t *loc, kmp_int32 global_tid,
    // kmp_critical_name *crit);
    llvm::Type *TypeParams[] = {
        getIdentTyPointerTy(), CGM.Int32Ty,
        llvm::PointerType::getUnqual(KmpCriticalNameTy)};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_critical");
    break;
  }
  case OMPRTL__kmpc_threadprivate_register: {
    // Build void __kmpc_threadprivate_register(ident_t *, void *data,
    // kmpc_ctor ctor, kmpc_cctor cctor, kmpc_dtor dtor);
    // typedef void *(*kmpc_ctor)(void *);
    auto KmpcCtorTy =
        llvm::FunctionType::get(CGM.VoidPtrTy, CGM.VoidPtrTy,
                                /*isVarArg*/ false)->getPointerTo();
    // typedef void *(*kmpc_cctor)(void *, void *);
    llvm::Type *KmpcCopyCtorTyArgs[] = {CGM.VoidPtrTy, CGM.VoidPtrTy};
    auto KmpcCopyCtorTy =
        llvm::FunctionType::get(CGM.VoidPtrTy, KmpcCopyCtorTyArgs,
                                /*isVarArg*/ false)->getPointerTo();
    // typedef void (*kmpc_dtor)(void *);
    auto KmpcDtorTy =
        llvm::FunctionType::get(CGM.VoidTy, CGM.VoidPtrTy, /*isVarArg*/ false)
            ->getPointerTo();
    llvm::Type *FnTyArgs[] = {getIdentTyPointerTy(), CGM.VoidPtrTy, KmpcCtorTy,
                              KmpcCopyCtorTy, KmpcDtorTy};
    auto FnTy = llvm::FunctionType::get(CGM.VoidTy, FnTyArgs,
                                        /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_threadprivate_register");
    break;
  }
  case OMPRTL__kmpc_end_critical: {
    // Build void __kmpc_end_critical(ident_t *loc, kmp_int32 global_tid,
    // kmp_critical_name *crit);
    llvm::Type *TypeParams[] = {
        getIdentTyPointerTy(), CGM.Int32Ty,
        llvm::PointerType::getUnqual(KmpCriticalNameTy)};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_end_critical");
    break;
  }
  case OMPRTL__kmpc_cancel_barrier: {
    // Build kmp_int32 __kmpc_cancel_barrier(ident_t *loc, kmp_int32
    // global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name*/ "__kmpc_cancel_barrier");
    break;
  }
  // Build __kmpc_for_static_init*(
  //               ident_t *loc, kmp_int32 tid, kmp_int32 schedtype,
  //               kmp_int32 *p_lastiter, kmp_int[32|64] *p_lower,
  //               kmp_int[32|64] *p_upper, kmp_int[32|64] *p_stride,
  //               kmp_int[32|64] incr, kmp_int[32|64] chunk);
  case OMPRTL__kmpc_for_static_init_4: {
    auto ITy = CGM.Int32Ty;
    auto PtrTy = llvm::PointerType::getUnqual(ITy);
    llvm::Type *TypeParams[] = {
        getIdentTyPointerTy(),                     // loc
        CGM.Int32Ty,                               // tid
        CGM.Int32Ty,                               // schedtype
        llvm::PointerType::getUnqual(CGM.Int32Ty), // p_lastiter
        PtrTy,                                     // p_lower
        PtrTy,                                     // p_upper
        PtrTy,                                     // p_stride
        ITy,                                       // incr
        ITy                                        // chunk
    };
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_for_static_init_4");
    break;
  }
  case OMPRTL__kmpc_for_static_init_4u: {
    auto ITy = CGM.Int32Ty;
    auto PtrTy = llvm::PointerType::getUnqual(ITy);
    llvm::Type *TypeParams[] = {
        getIdentTyPointerTy(),                     // loc
        CGM.Int32Ty,                               // tid
        CGM.Int32Ty,                               // schedtype
        llvm::PointerType::getUnqual(CGM.Int32Ty), // p_lastiter
        PtrTy,                                     // p_lower
        PtrTy,                                     // p_upper
        PtrTy,                                     // p_stride
        ITy,                                       // incr
        ITy                                        // chunk
    };
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_for_static_init_4u");
    break;
  }
  case OMPRTL__kmpc_for_static_init_8: {
    auto ITy = CGM.Int64Ty;
    auto PtrTy = llvm::PointerType::getUnqual(ITy);
    llvm::Type *TypeParams[] = {
        getIdentTyPointerTy(),                     // loc
        CGM.Int32Ty,                               // tid
        CGM.Int32Ty,                               // schedtype
        llvm::PointerType::getUnqual(CGM.Int32Ty), // p_lastiter
        PtrTy,                                     // p_lower
        PtrTy,                                     // p_upper
        PtrTy,                                     // p_stride
        ITy,                                       // incr
        ITy                                        // chunk
    };
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_for_static_init_8");
    break;
  }
  case OMPRTL__kmpc_for_static_init_8u: {
    auto ITy = CGM.Int64Ty;
    auto PtrTy = llvm::PointerType::getUnqual(ITy);
    llvm::Type *TypeParams[] = {
        getIdentTyPointerTy(),                     // loc
        CGM.Int32Ty,                               // tid
        CGM.Int32Ty,                               // schedtype
        llvm::PointerType::getUnqual(CGM.Int32Ty), // p_lastiter
        PtrTy,                                     // p_lower
        PtrTy,                                     // p_upper
        PtrTy,                                     // p_stride
        ITy,                                       // incr
        ITy                                        // chunk
    };
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_for_static_init_8u");
    break;
  }
  case OMPRTL__kmpc_for_static_fini: {
    // Build void __kmpc_for_static_fini(ident_t *loc, kmp_int32 global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_for_static_fini");
    break;
  }
  case OMPRTL__kmpc_push_num_threads: {
    // Build void __kmpc_push_num_threads(ident_t *loc, kmp_int32 global_tid,
    // kmp_int32 num_threads)
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty,
                                CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_push_num_threads");
    break;
  }
  case OMPRTL__kmpc_serialized_parallel: {
    // Build void __kmpc_serialized_parallel(ident_t *loc, kmp_int32
    // global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_serialized_parallel");
    break;
  }
  case OMPRTL__kmpc_end_serialized_parallel: {
    // Build void __kmpc_end_serialized_parallel(ident_t *loc, kmp_int32
    // global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_end_serialized_parallel");
    break;
  }
  case OMPRTL__kmpc_flush: {
    // Build void __kmpc_flush(ident_t *loc, ...);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ true);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_flush");
    break;
  }
  case OMPRTL__kmpc_master: {
    // Build kmp_int32 __kmpc_master(ident_t *loc, kmp_int32 global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_master");
    break;
  }
  case OMPRTL__kmpc_end_master: {
    // Build void __kmpc_end_master(ident_t *loc, kmp_int32 global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_end_master");
    break;
  }
  }
  return RTLFn;
}

llvm::Constant *
CGOpenMPRuntime::getOrCreateThreadPrivateCache(const VarDecl *VD) {
  // Lookup the entry, lazily creating it if necessary.
  return GetOrCreateInternalVariable(CGM.Int8PtrPtrTy,
                                     Twine(CGM.getMangledName(VD)) + ".cache.");
}

llvm::Value *CGOpenMPRuntime::getOMPAddrOfThreadPrivate(CodeGenFunction &CGF,
                                                        const VarDecl *VD,
                                                        llvm::Value *VDAddr,
                                                        SourceLocation Loc) {
  auto VarTy = VDAddr->getType()->getPointerElementType();
  llvm::Value *Args[] = {EmitOpenMPUpdateLocation(CGF, Loc),
                         GetOpenMPThreadID(CGF, Loc),
                         CGF.Builder.CreatePointerCast(VDAddr, CGM.Int8PtrTy),
                         CGM.getSize(CGM.GetTargetTypeStoreSize(VarTy)),
                         getOrCreateThreadPrivateCache(VD)};
  return CGF.EmitRuntimeCall(
      CreateRuntimeFunction(OMPRTL__kmpc_threadprivate_cached), Args);
}

void CGOpenMPRuntime::EmitOMPThreadPrivateVarInit(
    CodeGenFunction &CGF, llvm::Value *VDAddr, llvm::Value *Ctor,
    llvm::Value *CopyCtor, llvm::Value *Dtor, SourceLocation Loc) {
  // Call kmp_int32 __kmpc_global_thread_num(&loc) to init OpenMP runtime
  // library.
  auto OMPLoc = EmitOpenMPUpdateLocation(CGF, Loc);
  CGF.EmitRuntimeCall(CreateRuntimeFunction(OMPRTL__kmpc_global_thread_num),
                      OMPLoc);
  // Call __kmpc_threadprivate_register(&loc, &var, ctor, cctor/*NULL*/, dtor)
  // to register constructor/destructor for variable.
  llvm::Value *Args[] = {OMPLoc,
                         CGF.Builder.CreatePointerCast(VDAddr, CGM.VoidPtrTy),
                         Ctor, CopyCtor, Dtor};
  CGF.EmitRuntimeCall(
      CreateRuntimeFunction(OMPRTL__kmpc_threadprivate_register), Args);
}

llvm::Function *CGOpenMPRuntime::EmitOMPThreadPrivateVarDefinition(
    const VarDecl *VD, llvm::Value *VDAddr, SourceLocation Loc,
    bool PerformInit, CodeGenFunction *CGF) {
  VD = VD->getDefinition(CGM.getContext());
  if (VD && ThreadPrivateWithDefinition.count(VD) == 0) {
    ThreadPrivateWithDefinition.insert(VD);
    QualType ASTTy = VD->getType();

    llvm::Value *Ctor = nullptr, *CopyCtor = nullptr, *Dtor = nullptr;
    auto Init = VD->getAnyInitializer();
    if (CGM.getLangOpts().CPlusPlus && PerformInit) {
      // Generate function that re-emits the declaration's initializer into the
      // threadprivate copy of the variable VD
      CodeGenFunction CtorCGF(CGM);
      FunctionArgList Args;
      ImplicitParamDecl Dst(CGM.getContext(), /*DC=*/nullptr, SourceLocation(),
                            /*Id=*/nullptr, CGM.getContext().VoidPtrTy);
      Args.push_back(&Dst);

      auto &FI = CGM.getTypes().arrangeFreeFunctionDeclaration(
          CGM.getContext().VoidPtrTy, Args, FunctionType::ExtInfo(),
          /*isVariadic=*/false);
      auto FTy = CGM.getTypes().GetFunctionType(FI);
      auto Fn = CGM.CreateGlobalInitOrDestructFunction(
          FTy, ".__kmpc_global_ctor_.", Loc);
      CtorCGF.StartFunction(GlobalDecl(), CGM.getContext().VoidPtrTy, Fn, FI,
                            Args, SourceLocation());
      auto ArgVal = CtorCGF.EmitLoadOfScalar(
          CtorCGF.GetAddrOfLocalVar(&Dst),
          /*Volatile=*/false, CGM.PointerAlignInBytes,
          CGM.getContext().VoidPtrTy, Dst.getLocation());
      auto Arg = CtorCGF.Builder.CreatePointerCast(
          ArgVal,
          CtorCGF.ConvertTypeForMem(CGM.getContext().getPointerType(ASTTy)));
      CtorCGF.EmitAnyExprToMem(Init, Arg, Init->getType().getQualifiers(),
                               /*IsInitializer=*/true);
      ArgVal = CtorCGF.EmitLoadOfScalar(
          CtorCGF.GetAddrOfLocalVar(&Dst),
          /*Volatile=*/false, CGM.PointerAlignInBytes,
          CGM.getContext().VoidPtrTy, Dst.getLocation());
      CtorCGF.Builder.CreateStore(ArgVal, CtorCGF.ReturnValue);
      CtorCGF.FinishFunction();
      Ctor = Fn;
    }
    if (VD->getType().isDestructedType() != QualType::DK_none) {
      // Generate function that emits destructor call for the threadprivate copy
      // of the variable VD
      CodeGenFunction DtorCGF(CGM);
      FunctionArgList Args;
      ImplicitParamDecl Dst(CGM.getContext(), /*DC=*/nullptr, SourceLocation(),
                            /*Id=*/nullptr, CGM.getContext().VoidPtrTy);
      Args.push_back(&Dst);

      auto &FI = CGM.getTypes().arrangeFreeFunctionDeclaration(
          CGM.getContext().VoidTy, Args, FunctionType::ExtInfo(),
          /*isVariadic=*/false);
      auto FTy = CGM.getTypes().GetFunctionType(FI);
      auto Fn = CGM.CreateGlobalInitOrDestructFunction(
          FTy, ".__kmpc_global_dtor_.", Loc);
      DtorCGF.StartFunction(GlobalDecl(), CGM.getContext().VoidTy, Fn, FI, Args,
                            SourceLocation());
      auto ArgVal = DtorCGF.EmitLoadOfScalar(
          DtorCGF.GetAddrOfLocalVar(&Dst),
          /*Volatile=*/false, CGM.PointerAlignInBytes,
          CGM.getContext().VoidPtrTy, Dst.getLocation());
      DtorCGF.emitDestroy(ArgVal, ASTTy,
                          DtorCGF.getDestroyer(ASTTy.isDestructedType()),
                          DtorCGF.needsEHCleanup(ASTTy.isDestructedType()));
      DtorCGF.FinishFunction();
      Dtor = Fn;
    }
    // Do not emit init function if it is not required.
    if (!Ctor && !Dtor)
      return nullptr;

    llvm::Type *CopyCtorTyArgs[] = {CGM.VoidPtrTy, CGM.VoidPtrTy};
    auto CopyCtorTy =
        llvm::FunctionType::get(CGM.VoidPtrTy, CopyCtorTyArgs,
                                /*isVarArg=*/false)->getPointerTo();
    // Copying constructor for the threadprivate variable.
    // Must be NULL - reserved by runtime, but currently it requires that this
    // parameter is always NULL. Otherwise it fires assertion.
    CopyCtor = llvm::Constant::getNullValue(CopyCtorTy);
    if (Ctor == nullptr) {
      auto CtorTy = llvm::FunctionType::get(CGM.VoidPtrTy, CGM.VoidPtrTy,
                                            /*isVarArg=*/false)->getPointerTo();
      Ctor = llvm::Constant::getNullValue(CtorTy);
    }
    if (Dtor == nullptr) {
      auto DtorTy = llvm::FunctionType::get(CGM.VoidTy, CGM.VoidPtrTy,
                                            /*isVarArg=*/false)->getPointerTo();
      Dtor = llvm::Constant::getNullValue(DtorTy);
    }
    if (!CGF) {
      auto InitFunctionTy =
          llvm::FunctionType::get(CGM.VoidTy, /*isVarArg*/ false);
      auto InitFunction = CGM.CreateGlobalInitOrDestructFunction(
          InitFunctionTy, ".__omp_threadprivate_init_.");
      CodeGenFunction InitCGF(CGM);
      FunctionArgList ArgList;
      InitCGF.StartFunction(GlobalDecl(), CGM.getContext().VoidTy, InitFunction,
                            CGM.getTypes().arrangeNullaryFunction(), ArgList,
                            Loc);
      EmitOMPThreadPrivateVarInit(InitCGF, VDAddr, Ctor, CopyCtor, Dtor, Loc);
      InitCGF.FinishFunction();
      return InitFunction;
    }
    EmitOMPThreadPrivateVarInit(*CGF, VDAddr, Ctor, CopyCtor, Dtor, Loc);
  }
  return nullptr;
}

void CGOpenMPRuntime::EmitOMPParallelCall(CodeGenFunction &CGF,
                                          SourceLocation Loc,
                                          llvm::Value *OutlinedFn,
                                          llvm::Value *CapturedStruct) {
  // Build call __kmpc_fork_call(loc, 1, microtask, captured_struct/*context*/)
  llvm::Value *Args[] = {
      EmitOpenMPUpdateLocation(CGF, Loc),
      CGF.Builder.getInt32(1), // Number of arguments after 'microtask' argument
      // (there is only one additional argument - 'context')
      CGF.Builder.CreateBitCast(OutlinedFn, getKmpc_MicroPointerTy()),
      CGF.EmitCastToVoidPtr(CapturedStruct)};
  auto RTLFn = CreateRuntimeFunction(OMPRTL__kmpc_fork_call);
  CGF.EmitRuntimeCall(RTLFn, Args);
}

void CGOpenMPRuntime::EmitOMPSerialCall(CodeGenFunction &CGF,
                                        SourceLocation Loc,
                                        llvm::Value *OutlinedFn,
                                        llvm::Value *CapturedStruct) {
  auto ThreadID = GetOpenMPThreadID(CGF, Loc);
  // Build calls:
  // __kmpc_serialized_parallel(&Loc, GTid);
  llvm::Value *SerArgs[] = {EmitOpenMPUpdateLocation(CGF, Loc), ThreadID};
  auto RTLFn = CreateRuntimeFunction(OMPRTL__kmpc_serialized_parallel);
  CGF.EmitRuntimeCall(RTLFn, SerArgs);

  // OutlinedFn(&GTid, &zero, CapturedStruct);
  auto ThreadIDAddr = EmitThreadIDAddress(CGF, Loc);
  auto Int32Ty =
      CGF.getContext().getIntTypeForBitwidth(/*DestWidth*/ 32, /*Signed*/ true);
  auto ZeroAddr = CGF.CreateMemTemp(Int32Ty, /*Name*/ ".zero.addr");
  CGF.InitTempAlloca(ZeroAddr, CGF.Builder.getInt32(/*C*/ 0));
  llvm::Value *OutlinedFnArgs[] = {ThreadIDAddr, ZeroAddr, CapturedStruct};
  CGF.EmitCallOrInvoke(OutlinedFn, OutlinedFnArgs);

  // __kmpc_end_serialized_parallel(&Loc, GTid);
  llvm::Value *EndSerArgs[] = {EmitOpenMPUpdateLocation(CGF, Loc), ThreadID};
  RTLFn = CreateRuntimeFunction(OMPRTL__kmpc_end_serialized_parallel);
  CGF.EmitRuntimeCall(RTLFn, EndSerArgs);
}

// If we're inside an (outlined) parallel region, use the region info's
// thread-ID variable (it is passed in a first argument of the outlined function
// as "kmp_int32 *gtid"). Otherwise, if we're not inside parallel region, but in
// regular serial code region, get thread ID by calling kmp_int32
// kmpc_global_thread_num(ident_t *loc), stash this thread ID in a temporary and
// return the address of that temp.
llvm::Value *CGOpenMPRuntime::EmitThreadIDAddress(CodeGenFunction &CGF,
                                                  SourceLocation Loc) {
  if (auto OMPRegionInfo =
          dyn_cast_or_null<CGOpenMPRegionInfo>(CGF.CapturedStmtInfo))
    return CGF.EmitLoadOfLValue(OMPRegionInfo->getThreadIDVariableLValue(CGF),
                                SourceLocation()).getScalarVal();
  auto ThreadID = GetOpenMPThreadID(CGF, Loc);
  auto Int32Ty =
      CGF.getContext().getIntTypeForBitwidth(/*DestWidth*/ 32, /*Signed*/ true);
  auto ThreadIDTemp = CGF.CreateMemTemp(Int32Ty, /*Name*/ ".threadid_temp.");
  CGF.EmitStoreOfScalar(ThreadID,
                        CGF.MakeNaturalAlignAddrLValue(ThreadIDTemp, Int32Ty));

  return ThreadIDTemp;
}

llvm::Constant *
CGOpenMPRuntime::GetOrCreateInternalVariable(llvm::Type *Ty,
                                             const llvm::Twine &Name) {
  SmallString<256> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  Out << Name;
  auto RuntimeName = Out.str();
  auto &Elem = *InternalVars.insert(std::make_pair(RuntimeName, nullptr)).first;
  if (Elem.second) {
    assert(Elem.second->getType()->getPointerElementType() == Ty &&
           "OMP internal variable has different type than requested");
    return &*Elem.second;
  }

  return Elem.second = new llvm::GlobalVariable(
             CGM.getModule(), Ty, /*IsConstant*/ false,
             llvm::GlobalValue::CommonLinkage, llvm::Constant::getNullValue(Ty),
             Elem.first());
}

llvm::Value *CGOpenMPRuntime::GetCriticalRegionLock(StringRef CriticalName) {
  llvm::Twine Name(".gomp_critical_user_", CriticalName);
  return GetOrCreateInternalVariable(KmpCriticalNameTy, Name.concat(".var"));
}

void CGOpenMPRuntime::EmitOMPCriticalRegion(
    CodeGenFunction &CGF, StringRef CriticalName,
    const std::function<void()> &CriticalOpGen, SourceLocation Loc) {
  auto RegionLock = GetCriticalRegionLock(CriticalName);
  // __kmpc_critical(ident_t *, gtid, Lock);
  // CriticalOpGen();
  // __kmpc_end_critical(ident_t *, gtid, Lock);
  // Prepare arguments and build a call to __kmpc_critical
  llvm::Value *Args[] = {EmitOpenMPUpdateLocation(CGF, Loc),
                         GetOpenMPThreadID(CGF, Loc), RegionLock};
  auto RTLFn = CreateRuntimeFunction(OMPRTL__kmpc_critical);
  CGF.EmitRuntimeCall(RTLFn, Args);
  CriticalOpGen();
  // Build a call to __kmpc_end_critical
  RTLFn = CreateRuntimeFunction(OMPRTL__kmpc_end_critical);
  CGF.EmitRuntimeCall(RTLFn, Args);
}

static void EmitOMPIfStmt(CodeGenFunction &CGF, llvm::Value *IfCond,
                          const std::function<void()> &BodyOpGen) {
  llvm::Value *CallBool = CGF.EmitScalarConversion(
      IfCond,
      CGF.getContext().getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/true),
      CGF.getContext().BoolTy);

  auto *ThenBlock = CGF.createBasicBlock("omp_if.then");
  auto *ContBlock = CGF.createBasicBlock("omp_if.end");
  // Generate the branch (If-stmt)
  CGF.Builder.CreateCondBr(CallBool, ThenBlock, ContBlock);
  CGF.EmitBlock(ThenBlock);
  BodyOpGen();
  // Emit the rest of bblocks/branches
  CGF.EmitBranch(ContBlock);
  CGF.EmitBlock(ContBlock, true);
}

void CGOpenMPRuntime::EmitOMPMasterRegion(
    CodeGenFunction &CGF, const std::function<void()> &MasterOpGen,
    SourceLocation Loc) {
  // if(__kmpc_master(ident_t *, gtid)) {
  //   MasterOpGen();
  //   __kmpc_end_master(ident_t *, gtid);
  // }
  // Prepare arguments and build a call to __kmpc_master
  llvm::Value *Args[] = {EmitOpenMPUpdateLocation(CGF, Loc),
                         GetOpenMPThreadID(CGF, Loc)};
  auto RTLFn = CreateRuntimeFunction(OMPRTL__kmpc_master);
  auto *IsMaster = CGF.EmitRuntimeCall(RTLFn, Args);
  EmitOMPIfStmt(CGF, IsMaster, [&]() -> void {
    MasterOpGen();
    // Build a call to __kmpc_end_master.
    // OpenMP [1.2.2 OpenMP Language Terminology]
    // For C/C++, an executable statement, possibly compound, with a single
    // entry at the top and a single exit at the bottom, or an OpenMP construct.
    // * Access to the structured block must not be the result of a branch.
    // * The point of exit cannot be a branch out of the structured block.
    // * The point of entry must not be a call to setjmp().
    // * longjmp() and throw() must not violate the entry/exit criteria.
    // * An expression statement, iteration statement, selection statement, or
    // try block is considered to be a structured block if the corresponding
    // compound statement obtained by enclosing it in { and } would be a
    // structured block.
    // It is analyzed in Sema, so we can just call __kmpc_end_master() on
    // fallthrough rather than pushing a normal cleanup for it.
    RTLFn = CreateRuntimeFunction(OMPRTL__kmpc_end_master);
    CGF.EmitRuntimeCall(RTLFn, Args);
  });
}

void CGOpenMPRuntime::EmitOMPBarrierCall(CodeGenFunction &CGF,
                                         SourceLocation Loc, bool IsExplicit) {
  // Build call __kmpc_cancel_barrier(loc, thread_id);
  auto Flags = static_cast<OpenMPLocationFlags>(
      OMP_IDENT_KMPC |
      (IsExplicit ? OMP_IDENT_BARRIER_EXPL : OMP_IDENT_BARRIER_IMPL));
  // Build call __kmpc_cancel_barrier(loc, thread_id);
  // Replace __kmpc_barrier() function by __kmpc_cancel_barrier() because this
  // one provides the same functionality and adds initial support for
  // cancellation constructs introduced in OpenMP 4.0. __kmpc_cancel_barrier()
  // is provided default by the runtime library so it safe to make such
  // replacement.
  llvm::Value *Args[] = {EmitOpenMPUpdateLocation(CGF, Loc, Flags),
                         GetOpenMPThreadID(CGF, Loc)};
  auto RTLFn = CreateRuntimeFunction(OMPRTL__kmpc_cancel_barrier);
  CGF.EmitRuntimeCall(RTLFn, Args);
}

/// \brief Schedule types for 'omp for' loops (these enumerators are taken from
/// the enum sched_type in kmp.h).
enum OpenMPSchedType {
  /// \brief Lower bound for default (unordered) versions.
  OMP_sch_lower = 32,
  OMP_sch_static_chunked = 33,
  OMP_sch_static = 34,
  OMP_sch_dynamic_chunked = 35,
  OMP_sch_guided_chunked = 36,
  OMP_sch_runtime = 37,
  OMP_sch_auto = 38,
  /// \brief Lower bound for 'ordered' versions.
  OMP_ord_lower = 64,
  /// \brief Lower bound for 'nomerge' versions.
  OMP_nm_lower = 160,
};

/// \brief Map the OpenMP loop schedule to the runtime enumeration.
static OpenMPSchedType getRuntimeSchedule(OpenMPScheduleClauseKind ScheduleKind,
                                          bool Chunked) {
  switch (ScheduleKind) {
  case OMPC_SCHEDULE_static:
    return Chunked ? OMP_sch_static_chunked : OMP_sch_static;
  case OMPC_SCHEDULE_dynamic:
    return OMP_sch_dynamic_chunked;
  case OMPC_SCHEDULE_guided:
    return OMP_sch_guided_chunked;
  case OMPC_SCHEDULE_auto:
    return OMP_sch_auto;
  case OMPC_SCHEDULE_runtime:
    return OMP_sch_runtime;
  case OMPC_SCHEDULE_unknown:
    assert(!Chunked && "chunk was specified but schedule kind not known");
    return OMP_sch_static;
  }
  llvm_unreachable("Unexpected runtime schedule");
}

bool CGOpenMPRuntime::isStaticNonchunked(OpenMPScheduleClauseKind ScheduleKind,
                                         bool Chunked) const {
  auto Schedule = getRuntimeSchedule(ScheduleKind, Chunked);
  return Schedule == OMP_sch_static;
}

bool CGOpenMPRuntime::isDynamic(OpenMPScheduleClauseKind ScheduleKind) const {
  auto Schedule = getRuntimeSchedule(ScheduleKind, /* Chunked */ false);
  assert(Schedule != OMP_sch_static_chunked && "cannot be chunked here");
  return Schedule != OMP_sch_static;
}

void CGOpenMPRuntime::EmitOMPForInit(CodeGenFunction &CGF, SourceLocation Loc,
                                     OpenMPScheduleClauseKind ScheduleKind,
                                     unsigned IVSize, bool IVSigned,
                                     llvm::Value *IL, llvm::Value *LB,
                                     llvm::Value *UB, llvm::Value *ST,
                                     llvm::Value *Chunk) {
  OpenMPSchedType Schedule = getRuntimeSchedule(ScheduleKind, Chunk != nullptr);
  // Call __kmpc_for_static_init(
  //          ident_t *loc, kmp_int32 tid, kmp_int32 schedtype,
  //          kmp_int32 *p_lastiter, kmp_int[32|64] *p_lower,
  //          kmp_int[32|64] *p_upper, kmp_int[32|64] *p_stride,
  //          kmp_int[32|64] incr, kmp_int[32|64] chunk);
  // TODO: Implement dynamic schedule.

  // If the Chunk was not specified in the clause - use default value 1.
  if (Chunk == nullptr)
    Chunk = CGF.Builder.getIntN(IVSize, /*C*/ 1);

  llvm::Value *Args[] = {
      EmitOpenMPUpdateLocation(CGF, Loc, OMP_IDENT_KMPC),
      GetOpenMPThreadID(CGF, Loc),
      CGF.Builder.getInt32(Schedule), // Schedule type
      IL,                             // &isLastIter
      LB,                             // &LB
      UB,                             // &UB
      ST,                             // &Stride
      CGF.Builder.getIntN(IVSize, 1), // Incr
      Chunk                           // Chunk
  };
  assert((IVSize == 32 || IVSize == 64) &&
         "Index size is not compatible with the omp runtime");
  auto F = IVSize == 32 ? (IVSigned ? OMPRTL__kmpc_for_static_init_4
                                    : OMPRTL__kmpc_for_static_init_4u)
                        : (IVSigned ? OMPRTL__kmpc_for_static_init_8
                                    : OMPRTL__kmpc_for_static_init_8u);
  auto RTLFn = CreateRuntimeFunction(F);
  CGF.EmitRuntimeCall(RTLFn, Args);
}

void CGOpenMPRuntime::EmitOMPForFinish(CodeGenFunction &CGF, SourceLocation Loc,
                                       OpenMPScheduleClauseKind ScheduleKind) {
  assert((ScheduleKind == OMPC_SCHEDULE_static ||
          ScheduleKind == OMPC_SCHEDULE_unknown) &&
         "Non-static schedule kinds are not yet implemented");
  // Call __kmpc_for_static_fini(ident_t *loc, kmp_int32 tid);
  llvm::Value *Args[] = {EmitOpenMPUpdateLocation(CGF, Loc, OMP_IDENT_KMPC),
                         GetOpenMPThreadID(CGF, Loc)};
  auto RTLFn = CreateRuntimeFunction(OMPRTL__kmpc_for_static_fini);
  CGF.EmitRuntimeCall(RTLFn, Args);
}

void CGOpenMPRuntime::EmitOMPNumThreadsClause(CodeGenFunction &CGF,
                                              llvm::Value *NumThreads,
                                              SourceLocation Loc) {
  // Build call __kmpc_push_num_threads(&loc, global_tid, num_threads)
  llvm::Value *Args[] = {
      EmitOpenMPUpdateLocation(CGF, Loc), GetOpenMPThreadID(CGF, Loc),
      CGF.Builder.CreateIntCast(NumThreads, CGF.Int32Ty, /*isSigned*/ true)};
  llvm::Constant *RTLFn = CreateRuntimeFunction(OMPRTL__kmpc_push_num_threads);
  CGF.EmitRuntimeCall(RTLFn, Args);
}

void CGOpenMPRuntime::EmitOMPFlush(CodeGenFunction &CGF, ArrayRef<const Expr *>,
                                   SourceLocation Loc) {
  // Build call void __kmpc_flush(ident_t *loc, ...)
  // FIXME: List of variables is ignored by libiomp5 runtime, no need to
  // generate it, just request full memory fence.
  llvm::Value *Args[] = {EmitOpenMPUpdateLocation(CGF, Loc),
                         llvm::ConstantInt::get(CGM.Int32Ty, 0)};
  auto *RTLFn = CreateRuntimeFunction(OMPRTL__kmpc_flush);
  CGF.EmitRuntimeCall(RTLFn, Args);
}
