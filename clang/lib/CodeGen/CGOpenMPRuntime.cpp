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
#include "CGCleanup.h"
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
/// \brief Base class for handling code generation inside OpenMP regions.
class CGOpenMPRegionInfo : public CodeGenFunction::CGCapturedStmtInfo {
public:
  CGOpenMPRegionInfo(const OMPExecutableDirective &D, const CapturedStmt &CS)
      : CGCapturedStmtInfo(CS, CR_OpenMP), Directive(D) {}

  CGOpenMPRegionInfo(const OMPExecutableDirective &D)
      : CGCapturedStmtInfo(CR_OpenMP), Directive(D) {}

  /// \brief Get a variable or parameter for storing global thread id
  /// inside OpenMP construct.
  virtual const VarDecl *getThreadIDVariable() const = 0;

  /// \brief Get an LValue for the current ThreadID variable.
  /// \return LValue for thread id variable. This LValue always has type int32*.
  virtual LValue getThreadIDVariableLValue(CodeGenFunction &CGF);

    /// \brief Emit the captured statement body.
  virtual void EmitBody(CodeGenFunction &CGF, const Stmt *S) override;

  static bool classof(const CGCapturedStmtInfo *Info) {
    return Info->getKind() == CR_OpenMP;
  }
protected:
  /// \brief OpenMP executable directive associated with the region.
  const OMPExecutableDirective &Directive;
};

/// \brief API for captured statement code generation in OpenMP constructs.
class CGOpenMPOutlinedRegionInfo : public CGOpenMPRegionInfo {
public:
  CGOpenMPOutlinedRegionInfo(const OMPExecutableDirective &D,
                             const CapturedStmt &CS, const VarDecl *ThreadIDVar)
      : CGOpenMPRegionInfo(D, CS), ThreadIDVar(ThreadIDVar) {
    assert(ThreadIDVar != nullptr && "No ThreadID in OpenMP region.");
  }
  /// \brief Get a variable or parameter for storing global thread id
  /// inside OpenMP construct.
  virtual const VarDecl *getThreadIDVariable() const override {
    return ThreadIDVar;
  }
  /// \brief Get the name of the capture helper.
  StringRef getHelperName() const override { return ".omp_outlined."; }

private:
  /// \brief A variable or parameter storing global thread id for OpenMP
  /// constructs.
  const VarDecl *ThreadIDVar;
};

/// \brief API for captured statement code generation in OpenMP constructs.
class CGOpenMPTaskOutlinedRegionInfo : public CGOpenMPRegionInfo {
public:
  CGOpenMPTaskOutlinedRegionInfo(const OMPExecutableDirective &D,
                                 const CapturedStmt &CS,
                                 const VarDecl *ThreadIDVar,
                                 const VarDecl *PartIDVar)
      : CGOpenMPRegionInfo(D, CS), ThreadIDVar(ThreadIDVar),
        PartIDVar(PartIDVar) {
    assert(ThreadIDVar != nullptr && "No ThreadID in OpenMP region.");
  }
  /// \brief Get a variable or parameter for storing global thread id
  /// inside OpenMP construct.
  virtual const VarDecl *getThreadIDVariable() const override {
    return ThreadIDVar;
  }

  /// \brief Get an LValue for the current ThreadID variable.
  virtual LValue getThreadIDVariableLValue(CodeGenFunction &CGF) override;

  /// \brief Emit the captured statement body.
  virtual void EmitBody(CodeGenFunction &CGF, const Stmt *S) override;

  /// \brief Get the name of the capture helper.
  StringRef getHelperName() const override { return ".omp_outlined."; }

private:
  /// \brief A variable or parameter storing global thread id for OpenMP
  /// constructs.
  const VarDecl *ThreadIDVar;
  /// \brief A variable or parameter storing part id for OpenMP tasking
  /// constructs.
  const VarDecl *PartIDVar;
};

/// \brief API for inlined captured statement code generation in OpenMP
/// constructs.
class CGOpenMPInlinedRegionInfo : public CGOpenMPRegionInfo {
public:
  CGOpenMPInlinedRegionInfo(const OMPExecutableDirective &D,
                            CodeGenFunction::CGCapturedStmtInfo *OldCSI)
      : CGOpenMPRegionInfo(D), OldCSI(OldCSI),
        OuterRegionInfo(dyn_cast_or_null<CGOpenMPRegionInfo>(OldCSI)) {}
  // \brief Retrieve the value of the context parameter.
  virtual llvm::Value *getContextValue() const override {
    if (OuterRegionInfo)
      return OuterRegionInfo->getContextValue();
    llvm_unreachable("No context value for inlined OpenMP region");
  }
  /// \brief Lookup the captured field decl for a variable.
  virtual const FieldDecl *lookup(const VarDecl *VD) const override {
    if (OuterRegionInfo)
      return OuterRegionInfo->lookup(VD);
    llvm_unreachable("Trying to reference VarDecl that is neither local nor "
                     "captured in outer OpenMP region");
  }
  virtual FieldDecl *getThisFieldDecl() const override {
    if (OuterRegionInfo)
      return OuterRegionInfo->getThisFieldDecl();
    return nullptr;
  }
  /// \brief Get a variable or parameter for storing global thread id
  /// inside OpenMP construct.
  virtual const VarDecl *getThreadIDVariable() const override {
    if (OuterRegionInfo)
      return OuterRegionInfo->getThreadIDVariable();
    return nullptr;
  }

  /// \brief Get the name of the capture helper.
  virtual StringRef getHelperName() const override {
    llvm_unreachable("No helper name for inlined OpenMP construct");
  }

  CodeGenFunction::CGCapturedStmtInfo *getOldCSI() const { return OldCSI; }

private:
  /// \brief CodeGen info about outer OpenMP region.
  CodeGenFunction::CGCapturedStmtInfo *OldCSI;
  CGOpenMPRegionInfo *OuterRegionInfo;
};
} // namespace

LValue CGOpenMPRegionInfo::getThreadIDVariableLValue(CodeGenFunction &CGF) {
  return CGF.MakeNaturalAlignAddrLValue(
      CGF.Builder.CreateAlignedLoad(
          CGF.GetAddrOfLocalVar(getThreadIDVariable()),
          CGF.PointerAlignInBytes),
      getThreadIDVariable()
          ->getType()
          ->castAs<PointerType>()
          ->getPointeeType());
}

void CGOpenMPRegionInfo::EmitBody(CodeGenFunction &CGF, const Stmt *S) {
  CodeGenFunction::OMPPrivateScope PrivateScope(CGF);
  CGF.EmitOMPPrivateClause(Directive, PrivateScope);
  CGF.EmitOMPFirstprivateClause(Directive, PrivateScope);
  if (PrivateScope.Privatize())
    // Emit implicit barrier to synchronize threads and avoid data races.
    CGF.CGM.getOpenMPRuntime().emitBarrierCall(CGF, Directive.getLocStart(),
                                               /*IsExplicit=*/false);
  CGCapturedStmtInfo::EmitBody(CGF, S);
}

LValue CGOpenMPTaskOutlinedRegionInfo::getThreadIDVariableLValue(
    CodeGenFunction &CGF) {
  return CGF.MakeNaturalAlignAddrLValue(
      CGF.GetAddrOfLocalVar(getThreadIDVariable()),
      getThreadIDVariable()->getType());
}

void CGOpenMPTaskOutlinedRegionInfo::EmitBody(CodeGenFunction &CGF,
                                              const Stmt *S) {
  if (PartIDVar) {
    // TODO: emit code for untied tasks.
  }
  CGCapturedStmtInfo::EmitBody(CGF, S);
}

CGOpenMPRuntime::CGOpenMPRuntime(CodeGenModule &CGM)
    : CGM(CGM), DefaultOpenMPPSource(nullptr), KmpRoutineEntryPtrTy(nullptr) {
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

void CGOpenMPRuntime::clear() {
  InternalVars.clear();
}

llvm::Value *
CGOpenMPRuntime::emitOutlinedFunction(const OMPExecutableDirective &D,
                                      const VarDecl *ThreadIDVar) {
  assert(ThreadIDVar->getType()->isPointerType() &&
         "thread id variable must be of type kmp_int32 *");
  const CapturedStmt *CS = cast<CapturedStmt>(D.getAssociatedStmt());
  CodeGenFunction CGF(CGM, true);
  CGOpenMPOutlinedRegionInfo CGInfo(D, *CS, ThreadIDVar);
  CGF.CapturedStmtInfo = &CGInfo;
  return CGF.GenerateCapturedStmtFunction(*CS);
}

llvm::Value *
CGOpenMPRuntime::emitTaskOutlinedFunction(const OMPExecutableDirective &D,
                                          const VarDecl *ThreadIDVar,
                                          const VarDecl *PartIDVar) {
  assert(!ThreadIDVar->getType()->isPointerType() &&
         "thread id variable must be of type kmp_int32 for tasks");
  auto *CS = cast<CapturedStmt>(D.getAssociatedStmt());
  CodeGenFunction CGF(CGM, true);
  CGOpenMPTaskOutlinedRegionInfo CGInfo(D, *CS, ThreadIDVar, PartIDVar);
  CGF.CapturedStmtInfo = &CGInfo;
  return CGF.GenerateCapturedStmtFunction(*CS);
}

llvm::Value *
CGOpenMPRuntime::getOrCreateDefaultLocation(OpenMPLocationFlags Flags) {
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

llvm::Value *CGOpenMPRuntime::emitUpdateLocation(CodeGenFunction &CGF,
                                                 SourceLocation Loc,
                                                 OpenMPLocationFlags Flags) {
  // If no debug info is generated - return global default location.
  if (CGM.getCodeGenOpts().getDebugInfo() == CodeGenOptions::NoDebugInfo ||
      Loc.isInvalid())
    return getOrCreateDefaultLocation(Flags);

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
    CGF.Builder.CreateMemCpy(LocValue, getOrCreateDefaultLocation(Flags),
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

llvm::Value *CGOpenMPRuntime::getThreadID(CodeGenFunction &CGF,
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
    if (OMPRegionInfo->getThreadIDVariable()) {
      // Check if this an outlined function with thread id passed as argument.
      auto LVal = OMPRegionInfo->getThreadIDVariableLValue(CGF);
      ThreadID = CGF.EmitLoadOfLValue(LVal, Loc).getScalarVal();
      // If value loaded in entry block, cache it and use it everywhere in
      // function.
      if (CGF.Builder.GetInsertBlock() == CGF.AllocaInsertPt->getParent()) {
        auto &Elem = OpenMPLocThreadIDMap.FindAndConstruct(CGF.CurFn);
        Elem.second.ThreadID = ThreadID;
      }
      return ThreadID;
    }
  }

  // This is not an outlined function region - need to call __kmpc_int32
  // kmpc_global_thread_num(ident_t *loc).
  // Generate thread id value and cache this value for use across the
  // function.
  CGBuilderTy::InsertPointGuard IPG(CGF.Builder);
  CGF.Builder.SetInsertPoint(CGF.AllocaInsertPt);
  ThreadID =
      CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_global_thread_num),
                          emitUpdateLocation(CGF, Loc));
  auto &Elem = OpenMPLocThreadIDMap.FindAndConstruct(CGF.CurFn);
  Elem.second.ThreadID = ThreadID;
  return ThreadID;
}

void CGOpenMPRuntime::functionFinished(CodeGenFunction &CGF) {
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
CGOpenMPRuntime::createRuntimeFunction(OpenMPRTLFunction Function) {
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
    // Build void __kmpc_flush(ident_t *loc);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
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
  case OMPRTL__kmpc_omp_taskyield: {
    // Build kmp_int32 __kmpc_omp_taskyield(ident_t *, kmp_int32 global_tid,
    // int end_part);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty, CGM.IntTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_omp_taskyield");
    break;
  }
  case OMPRTL__kmpc_single: {
    // Build kmp_int32 __kmpc_single(ident_t *loc, kmp_int32 global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_single");
    break;
  }
  case OMPRTL__kmpc_end_single: {
    // Build void __kmpc_end_single(ident_t *loc, kmp_int32 global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_end_single");
    break;
  }
  case OMPRTL__kmpc_omp_task_alloc: {
    // Build kmp_task_t *__kmpc_omp_task_alloc(ident_t *, kmp_int32 gtid,
    // kmp_int32 flags, size_t sizeof_kmp_task_t, size_t sizeof_shareds,
    // kmp_routine_entry_t *task_entry);
    assert(KmpRoutineEntryPtrTy != nullptr &&
           "Type kmp_routine_entry_t must be created.");
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty, CGM.Int32Ty,
                                CGM.SizeTy, CGM.SizeTy, KmpRoutineEntryPtrTy};
    // Return void * and then cast to particular kmp_task_t type.
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidPtrTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_omp_task_alloc");
    break;
  }
  case OMPRTL__kmpc_omp_task: {
    // Build kmp_int32 __kmpc_omp_task(ident_t *, kmp_int32 gtid, kmp_task_t
    // *new_task);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty,
                                CGM.VoidPtrTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_omp_task");
    break;
  }
  case OMPRTL__kmpc_copyprivate: {
    // Build void __kmpc_copyprivate(ident_t *loc, kmp_int32 global_tid,
    // kmp_int32 cpy_size, void *cpy_data, void(*cpy_func)(void *, void *),
    // kmp_int32 didit);
    llvm::Type *CpyTypeParams[] = {CGM.VoidPtrTy, CGM.VoidPtrTy};
    auto *CpyFnTy =
        llvm::FunctionType::get(CGM.VoidTy, CpyTypeParams, /*isVarArg=*/false);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty, CGM.Int32Ty,
                                CGM.VoidPtrTy, CpyFnTy->getPointerTo(),
                                CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_copyprivate");
    break;
  }
  }
  return RTLFn;
}

llvm::Constant *CGOpenMPRuntime::createForStaticInitFunction(unsigned IVSize,
                                                             bool IVSigned) {
  assert((IVSize == 32 || IVSize == 64) &&
         "IV size is not compatible with the omp runtime");
  auto Name = IVSize == 32 ? (IVSigned ? "__kmpc_for_static_init_4"
                                       : "__kmpc_for_static_init_4u")
                           : (IVSigned ? "__kmpc_for_static_init_8"
                                       : "__kmpc_for_static_init_8u");
  auto ITy = IVSize == 32 ? CGM.Int32Ty : CGM.Int64Ty;
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
  return CGM.CreateRuntimeFunction(FnTy, Name);
}

llvm::Constant *CGOpenMPRuntime::createDispatchInitFunction(unsigned IVSize,
                                                            bool IVSigned) {
  assert((IVSize == 32 || IVSize == 64) &&
         "IV size is not compatible with the omp runtime");
  auto Name =
      IVSize == 32
          ? (IVSigned ? "__kmpc_dispatch_init_4" : "__kmpc_dispatch_init_4u")
          : (IVSigned ? "__kmpc_dispatch_init_8" : "__kmpc_dispatch_init_8u");
  auto ITy = IVSize == 32 ? CGM.Int32Ty : CGM.Int64Ty;
  llvm::Type *TypeParams[] = { getIdentTyPointerTy(), // loc
                               CGM.Int32Ty,           // tid
                               CGM.Int32Ty,           // schedtype
                               ITy,                   // lower
                               ITy,                   // upper
                               ITy,                   // stride
                               ITy                    // chunk
  };
  llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
  return CGM.CreateRuntimeFunction(FnTy, Name);
}

llvm::Constant *CGOpenMPRuntime::createDispatchNextFunction(unsigned IVSize,
                                                            bool IVSigned) {
  assert((IVSize == 32 || IVSize == 64) &&
         "IV size is not compatible with the omp runtime");
  auto Name =
      IVSize == 32
          ? (IVSigned ? "__kmpc_dispatch_next_4" : "__kmpc_dispatch_next_4u")
          : (IVSigned ? "__kmpc_dispatch_next_8" : "__kmpc_dispatch_next_8u");
  auto ITy = IVSize == 32 ? CGM.Int32Ty : CGM.Int64Ty;
  auto PtrTy = llvm::PointerType::getUnqual(ITy);
  llvm::Type *TypeParams[] = {
    getIdentTyPointerTy(),                     // loc
    CGM.Int32Ty,                               // tid
    llvm::PointerType::getUnqual(CGM.Int32Ty), // p_lastiter
    PtrTy,                                     // p_lower
    PtrTy,                                     // p_upper
    PtrTy                                      // p_stride
  };
  llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg*/ false);
  return CGM.CreateRuntimeFunction(FnTy, Name);
}

llvm::Constant *
CGOpenMPRuntime::getOrCreateThreadPrivateCache(const VarDecl *VD) {
  // Lookup the entry, lazily creating it if necessary.
  return getOrCreateInternalVariable(CGM.Int8PtrPtrTy,
                                     Twine(CGM.getMangledName(VD)) + ".cache.");
}

llvm::Value *CGOpenMPRuntime::getAddrOfThreadPrivate(CodeGenFunction &CGF,
                                                     const VarDecl *VD,
                                                     llvm::Value *VDAddr,
                                                     SourceLocation Loc) {
  auto VarTy = VDAddr->getType()->getPointerElementType();
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc),
                         CGF.Builder.CreatePointerCast(VDAddr, CGM.Int8PtrTy),
                         CGM.getSize(CGM.GetTargetTypeStoreSize(VarTy)),
                         getOrCreateThreadPrivateCache(VD)};
  return CGF.EmitRuntimeCall(
      createRuntimeFunction(OMPRTL__kmpc_threadprivate_cached), Args);
}

void CGOpenMPRuntime::emitThreadPrivateVarInit(
    CodeGenFunction &CGF, llvm::Value *VDAddr, llvm::Value *Ctor,
    llvm::Value *CopyCtor, llvm::Value *Dtor, SourceLocation Loc) {
  // Call kmp_int32 __kmpc_global_thread_num(&loc) to init OpenMP runtime
  // library.
  auto OMPLoc = emitUpdateLocation(CGF, Loc);
  CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_global_thread_num),
                      OMPLoc);
  // Call __kmpc_threadprivate_register(&loc, &var, ctor, cctor/*NULL*/, dtor)
  // to register constructor/destructor for variable.
  llvm::Value *Args[] = {OMPLoc,
                         CGF.Builder.CreatePointerCast(VDAddr, CGM.VoidPtrTy),
                         Ctor, CopyCtor, Dtor};
  CGF.EmitRuntimeCall(
      createRuntimeFunction(OMPRTL__kmpc_threadprivate_register), Args);
}

llvm::Function *CGOpenMPRuntime::emitThreadPrivateVarDefinition(
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
      emitThreadPrivateVarInit(InitCGF, VDAddr, Ctor, CopyCtor, Dtor, Loc);
      InitCGF.FinishFunction();
      return InitFunction;
    }
    emitThreadPrivateVarInit(*CGF, VDAddr, Ctor, CopyCtor, Dtor, Loc);
  }
  return nullptr;
}

void CGOpenMPRuntime::emitParallelCall(CodeGenFunction &CGF, SourceLocation Loc,
                                       llvm::Value *OutlinedFn,
                                       llvm::Value *CapturedStruct) {
  // Build call __kmpc_fork_call(loc, 1, microtask, captured_struct/*context*/)
  llvm::Value *Args[] = {
      emitUpdateLocation(CGF, Loc),
      CGF.Builder.getInt32(1), // Number of arguments after 'microtask' argument
      // (there is only one additional argument - 'context')
      CGF.Builder.CreateBitCast(OutlinedFn, getKmpc_MicroPointerTy()),
      CGF.EmitCastToVoidPtr(CapturedStruct)};
  auto RTLFn = createRuntimeFunction(OMPRTL__kmpc_fork_call);
  CGF.EmitRuntimeCall(RTLFn, Args);
}

void CGOpenMPRuntime::emitSerialCall(CodeGenFunction &CGF, SourceLocation Loc,
                                     llvm::Value *OutlinedFn,
                                     llvm::Value *CapturedStruct) {
  auto ThreadID = getThreadID(CGF, Loc);
  // Build calls:
  // __kmpc_serialized_parallel(&Loc, GTid);
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc), ThreadID};
  CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_serialized_parallel),
                      Args);

  // OutlinedFn(&GTid, &zero, CapturedStruct);
  auto ThreadIDAddr = emitThreadIDAddress(CGF, Loc);
  auto Int32Ty =
      CGF.getContext().getIntTypeForBitwidth(/*DestWidth*/ 32, /*Signed*/ true);
  auto ZeroAddr = CGF.CreateMemTemp(Int32Ty, /*Name*/ ".zero.addr");
  CGF.InitTempAlloca(ZeroAddr, CGF.Builder.getInt32(/*C*/ 0));
  llvm::Value *OutlinedFnArgs[] = {ThreadIDAddr, ZeroAddr, CapturedStruct};
  CGF.EmitCallOrInvoke(OutlinedFn, OutlinedFnArgs);

  // __kmpc_end_serialized_parallel(&Loc, GTid);
  llvm::Value *EndArgs[] = {emitUpdateLocation(CGF, Loc), ThreadID};
  CGF.EmitRuntimeCall(
      createRuntimeFunction(OMPRTL__kmpc_end_serialized_parallel), EndArgs);
}

// If we're inside an (outlined) parallel region, use the region info's
// thread-ID variable (it is passed in a first argument of the outlined function
// as "kmp_int32 *gtid"). Otherwise, if we're not inside parallel region, but in
// regular serial code region, get thread ID by calling kmp_int32
// kmpc_global_thread_num(ident_t *loc), stash this thread ID in a temporary and
// return the address of that temp.
llvm::Value *CGOpenMPRuntime::emitThreadIDAddress(CodeGenFunction &CGF,
                                                  SourceLocation Loc) {
  if (auto OMPRegionInfo =
          dyn_cast_or_null<CGOpenMPRegionInfo>(CGF.CapturedStmtInfo))
    if (OMPRegionInfo->getThreadIDVariable())
      return OMPRegionInfo->getThreadIDVariableLValue(CGF).getAddress();

  auto ThreadID = getThreadID(CGF, Loc);
  auto Int32Ty =
      CGF.getContext().getIntTypeForBitwidth(/*DestWidth*/ 32, /*Signed*/ true);
  auto ThreadIDTemp = CGF.CreateMemTemp(Int32Ty, /*Name*/ ".threadid_temp.");
  CGF.EmitStoreOfScalar(ThreadID,
                        CGF.MakeNaturalAlignAddrLValue(ThreadIDTemp, Int32Ty));

  return ThreadIDTemp;
}

llvm::Constant *
CGOpenMPRuntime::getOrCreateInternalVariable(llvm::Type *Ty,
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

llvm::Value *CGOpenMPRuntime::getCriticalRegionLock(StringRef CriticalName) {
  llvm::Twine Name(".gomp_critical_user_", CriticalName);
  return getOrCreateInternalVariable(KmpCriticalNameTy, Name.concat(".var"));
}

void CGOpenMPRuntime::emitCriticalRegion(
    CodeGenFunction &CGF, StringRef CriticalName,
    const std::function<void()> &CriticalOpGen, SourceLocation Loc) {
  auto RegionLock = getCriticalRegionLock(CriticalName);
  // __kmpc_critical(ident_t *, gtid, Lock);
  // CriticalOpGen();
  // __kmpc_end_critical(ident_t *, gtid, Lock);
  // Prepare arguments and build a call to __kmpc_critical
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc),
                         RegionLock};
  CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_critical), Args);
  CriticalOpGen();
  // Build a call to __kmpc_end_critical
  CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_end_critical), Args);
}

static void emitIfStmt(CodeGenFunction &CGF, llvm::Value *IfCond,
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

void CGOpenMPRuntime::emitMasterRegion(CodeGenFunction &CGF,
                                       const std::function<void()> &MasterOpGen,
                                       SourceLocation Loc) {
  // if(__kmpc_master(ident_t *, gtid)) {
  //   MasterOpGen();
  //   __kmpc_end_master(ident_t *, gtid);
  // }
  // Prepare arguments and build a call to __kmpc_master
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc)};
  auto *IsMaster =
      CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_master), Args);
  emitIfStmt(CGF, IsMaster, [&]() -> void {
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
    CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_end_master), Args);
  });
}

void CGOpenMPRuntime::emitTaskyieldCall(CodeGenFunction &CGF,
                                        SourceLocation Loc) {
  // Build call __kmpc_omp_taskyield(loc, thread_id, 0);
  llvm::Value *Args[] = {
      emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc),
      llvm::ConstantInt::get(CGM.IntTy, /*V=*/0, /*isSigned=*/true)};
  CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_omp_taskyield), Args);
}

static llvm::Value *emitCopyprivateCopyFunction(
    CodeGenModule &CGM, llvm::Type *ArgsType, ArrayRef<const Expr *> SrcExprs,
    ArrayRef<const Expr *> DstExprs, ArrayRef<const Expr *> AssignmentOps) {
  auto &C = CGM.getContext();
  // void copy_func(void *LHSArg, void *RHSArg);
  FunctionArgList Args;
  ImplicitParamDecl LHSArg(C, /*DC=*/nullptr, SourceLocation(), /*Id=*/nullptr,
                           C.VoidPtrTy);
  ImplicitParamDecl RHSArg(C, /*DC=*/nullptr, SourceLocation(), /*Id=*/nullptr,
                           C.VoidPtrTy);
  Args.push_back(&LHSArg);
  Args.push_back(&RHSArg);
  FunctionType::ExtInfo EI;
  auto &CGFI = CGM.getTypes().arrangeFreeFunctionDeclaration(
      C.VoidTy, Args, EI, /*isVariadic=*/false);
  auto *Fn = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(CGFI), llvm::GlobalValue::InternalLinkage,
      ".omp.copyprivate.copy_func", &CGM.getModule());
  CGM.SetLLVMFunctionAttributes(/*D=*/nullptr, CGFI, Fn);
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, CGFI, Args);
  // Dst = (void*[n])(LHSArg);
  // Src = (void*[n])(RHSArg);
  auto *LHS = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      CGF.Builder.CreateAlignedLoad(CGF.GetAddrOfLocalVar(&LHSArg),
                                    CGF.PointerAlignInBytes),
      ArgsType);
  auto *RHS = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      CGF.Builder.CreateAlignedLoad(CGF.GetAddrOfLocalVar(&RHSArg),
                                    CGF.PointerAlignInBytes),
      ArgsType);
  // *(Type0*)Dst[0] = *(Type0*)Src[0];
  // *(Type1*)Dst[1] = *(Type1*)Src[1];
  // ...
  // *(Typen*)Dst[n] = *(Typen*)Src[n];
  CodeGenFunction::OMPPrivateScope Scope(CGF);
  for (unsigned I = 0, E = AssignmentOps.size(); I < E; ++I) {
    Scope.addPrivate(
        cast<VarDecl>(cast<DeclRefExpr>(SrcExprs[I])->getDecl()),
        [&]() -> llvm::Value *{
          return CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
              CGF.Builder.CreateAlignedLoad(CGF.Builder.CreateStructGEP(RHS, I),
                                            CGM.PointerAlignInBytes),
              CGF.ConvertTypeForMem(C.getPointerType(SrcExprs[I]->getType())));
        });
    Scope.addPrivate(
        cast<VarDecl>(cast<DeclRefExpr>(DstExprs[I])->getDecl()),
        [&]() -> llvm::Value *{
          return CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
              CGF.Builder.CreateAlignedLoad(CGF.Builder.CreateStructGEP(LHS, I),
                                            CGM.PointerAlignInBytes),
              CGF.ConvertTypeForMem(C.getPointerType(SrcExprs[I]->getType())));
        });
  }
  Scope.Privatize();
  for (auto *E : AssignmentOps) {
    CGF.EmitIgnoredExpr(E);
  }
  Scope.ForceCleanup();
  CGF.FinishFunction();
  return Fn;
}

void CGOpenMPRuntime::emitSingleRegion(CodeGenFunction &CGF,
                                       const std::function<void()> &SingleOpGen,
                                       SourceLocation Loc,
                                       ArrayRef<const Expr *> CopyprivateVars,
                                       ArrayRef<const Expr *> SrcExprs,
                                       ArrayRef<const Expr *> DstExprs,
                                       ArrayRef<const Expr *> AssignmentOps) {
  assert(CopyprivateVars.size() == SrcExprs.size() &&
         CopyprivateVars.size() == DstExprs.size() &&
         CopyprivateVars.size() == AssignmentOps.size());
  auto &C = CGM.getContext();
  // int32 did_it = 0;
  // if(__kmpc_single(ident_t *, gtid)) {
  //   SingleOpGen();
  //   __kmpc_end_single(ident_t *, gtid);
  //   did_it = 1;
  // }
  // call __kmpc_copyprivate(ident_t *, gtid, <buf_size>, <copyprivate list>,
  // <copy_func>, did_it);

  llvm::AllocaInst *DidIt = nullptr;
  if (!CopyprivateVars.empty()) {
    // int32 did_it = 0;
    auto KmpInt32Ty = C.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/1);
    DidIt = CGF.CreateMemTemp(KmpInt32Ty, ".omp.copyprivate.did_it");
    CGF.InitTempAlloca(DidIt, CGF.Builder.getInt32(0));
  }
  // Prepare arguments and build a call to __kmpc_single
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc)};
  auto *IsSingle =
      CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_single), Args);
  emitIfStmt(CGF, IsSingle, [&]() -> void {
    SingleOpGen();
    if (DidIt) {
      // did_it = 1;
      CGF.Builder.CreateAlignedStore(CGF.Builder.getInt32(1), DidIt,
                                     DidIt->getAlignment());
    }
    // Build a call to __kmpc_end_single.
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
    // It is analyzed in Sema, so we can just call __kmpc_end_single() on
    // fallthrough rather than pushing a normal cleanup for it.
    CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_end_single), Args);
  });
  // call __kmpc_copyprivate(ident_t *, gtid, <buf_size>, <copyprivate list>,
  // <copy_func>, did_it);
  if (DidIt) {
    llvm::APInt ArraySize(/*unsigned int numBits=*/32, CopyprivateVars.size());
    auto CopyprivateArrayTy =
        C.getConstantArrayType(C.VoidPtrTy, ArraySize, ArrayType::Normal,
                               /*IndexTypeQuals=*/0);
    // Create a list of all private variables for copyprivate.
    auto *CopyprivateList =
        CGF.CreateMemTemp(CopyprivateArrayTy, ".omp.copyprivate.cpr_list");
    for (unsigned I = 0, E = CopyprivateVars.size(); I < E; ++I) {
      auto *Elem = CGF.Builder.CreateStructGEP(CopyprivateList, I);
      CGF.Builder.CreateAlignedStore(
          CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
              CGF.EmitLValue(CopyprivateVars[I]).getAddress(), CGF.VoidPtrTy),
          Elem, CGM.PointerAlignInBytes);
    }
    // Build function that copies private values from single region to all other
    // threads in the corresponding parallel region.
    auto *CpyFn = emitCopyprivateCopyFunction(
        CGM, CGF.ConvertTypeForMem(CopyprivateArrayTy)->getPointerTo(),
        SrcExprs, DstExprs, AssignmentOps);
    auto *BufSize = CGF.Builder.getInt32(
        C.getTypeSizeInChars(CopyprivateArrayTy).getQuantity());
    auto *CL = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(CopyprivateList,
                                                               CGF.VoidPtrTy);
    auto *DidItVal =
        CGF.Builder.CreateAlignedLoad(DidIt, CGF.PointerAlignInBytes);
    llvm::Value *Args[] = {
        emitUpdateLocation(CGF, Loc), // ident_t *<loc>
        getThreadID(CGF, Loc),        // i32 <gtid>
        BufSize,                      // i32 <buf_size>
        CL,                           // void *<copyprivate list>
        CpyFn,                        // void (*) (void *, void *) <copy_func>
        DidItVal                      // i32 did_it
    };
    CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_copyprivate), Args);
  }
}

void CGOpenMPRuntime::emitBarrierCall(CodeGenFunction &CGF, SourceLocation Loc,
                                      bool IsExplicit) {
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
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc, Flags),
                         getThreadID(CGF, Loc)};
  CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_cancel_barrier), Args);
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

void CGOpenMPRuntime::emitForInit(CodeGenFunction &CGF, SourceLocation Loc,
                                  OpenMPScheduleClauseKind ScheduleKind,
                                  unsigned IVSize, bool IVSigned,
                                  llvm::Value *IL, llvm::Value *LB,
                                  llvm::Value *UB, llvm::Value *ST,
                                  llvm::Value *Chunk) {
  OpenMPSchedType Schedule = getRuntimeSchedule(ScheduleKind, Chunk != nullptr);
  if (Schedule != OMP_sch_static && Schedule != OMP_sch_static_chunked) {
    // Call __kmpc_dispatch_init(
    //          ident_t *loc, kmp_int32 tid, kmp_int32 schedule,
    //          kmp_int[32|64] lower, kmp_int[32|64] upper,
    //          kmp_int[32|64] stride, kmp_int[32|64] chunk);

    // If the Chunk was not specified in the clause - use default value 1.
    if (Chunk == nullptr)
      Chunk = CGF.Builder.getIntN(IVSize, 1);
    llvm::Value *Args[] = { emitUpdateLocation(CGF, Loc, OMP_IDENT_KMPC),
                            getThreadID(CGF, Loc),
                            CGF.Builder.getInt32(Schedule), // Schedule type
                            CGF.Builder.getIntN(IVSize, 0), // Lower
                            UB,                             // Upper
                            CGF.Builder.getIntN(IVSize, 1), // Stride
                            Chunk                           // Chunk
    };
    CGF.EmitRuntimeCall(createDispatchInitFunction(IVSize, IVSigned), Args);
  } else {
    // Call __kmpc_for_static_init(
    //          ident_t *loc, kmp_int32 tid, kmp_int32 schedtype,
    //          kmp_int32 *p_lastiter, kmp_int[32|64] *p_lower,
    //          kmp_int[32|64] *p_upper, kmp_int[32|64] *p_stride,
    //          kmp_int[32|64] incr, kmp_int[32|64] chunk);
    if (Chunk == nullptr) {
      assert(Schedule == OMP_sch_static &&
             "expected static non-chunked schedule");
      // If the Chunk was not specified in the clause - use default value 1.
      Chunk = CGF.Builder.getIntN(IVSize, 1);
    } else
      assert(Schedule == OMP_sch_static_chunked &&
             "expected static chunked schedule");
    llvm::Value *Args[] = { emitUpdateLocation(CGF, Loc, OMP_IDENT_KMPC),
                            getThreadID(CGF, Loc),
                            CGF.Builder.getInt32(Schedule), // Schedule type
                            IL,                             // &isLastIter
                            LB,                             // &LB
                            UB,                             // &UB
                            ST,                             // &Stride
                            CGF.Builder.getIntN(IVSize, 1), // Incr
                            Chunk                           // Chunk
    };
    CGF.EmitRuntimeCall(createForStaticInitFunction(IVSize, IVSigned), Args);
  }
}

void CGOpenMPRuntime::emitForFinish(CodeGenFunction &CGF, SourceLocation Loc,
                                    OpenMPScheduleClauseKind ScheduleKind) {
  assert((ScheduleKind == OMPC_SCHEDULE_static ||
          ScheduleKind == OMPC_SCHEDULE_unknown) &&
         "Non-static schedule kinds are not yet implemented");
  // Call __kmpc_for_static_fini(ident_t *loc, kmp_int32 tid);
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc, OMP_IDENT_KMPC),
                         getThreadID(CGF, Loc)};
  CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_for_static_fini),
                      Args);
}

llvm::Value *CGOpenMPRuntime::emitForNext(CodeGenFunction &CGF,
                                          SourceLocation Loc, unsigned IVSize,
                                          bool IVSigned, llvm::Value *IL,
                                          llvm::Value *LB, llvm::Value *UB,
                                          llvm::Value *ST) {
  // Call __kmpc_dispatch_next(
  //          ident_t *loc, kmp_int32 tid, kmp_int32 *p_lastiter,
  //          kmp_int[32|64] *p_lower, kmp_int[32|64] *p_upper,
  //          kmp_int[32|64] *p_stride);
  llvm::Value *Args[] = {
      emitUpdateLocation(CGF, Loc, OMP_IDENT_KMPC), getThreadID(CGF, Loc),
      IL, // &isLastIter
      LB, // &Lower
      UB, // &Upper
      ST  // &Stride
  };
  llvm::Value *Call =
      CGF.EmitRuntimeCall(createDispatchNextFunction(IVSize, IVSigned), Args);
  return CGF.EmitScalarConversion(
      Call, CGF.getContext().getIntTypeForBitwidth(32, /* Signed */ true),
      CGF.getContext().BoolTy);
}

void CGOpenMPRuntime::emitNumThreadsClause(CodeGenFunction &CGF,
                                           llvm::Value *NumThreads,
                                           SourceLocation Loc) {
  // Build call __kmpc_push_num_threads(&loc, global_tid, num_threads)
  llvm::Value *Args[] = {
      emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc),
      CGF.Builder.CreateIntCast(NumThreads, CGF.Int32Ty, /*isSigned*/ true)};
  CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_push_num_threads),
                      Args);
}

void CGOpenMPRuntime::emitFlush(CodeGenFunction &CGF, ArrayRef<const Expr *>,
                                SourceLocation Loc) {
  // Build call void __kmpc_flush(ident_t *loc)
  CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_flush),
                      emitUpdateLocation(CGF, Loc));
}

namespace {
/// \brief Indexes of fields for type kmp_task_t.
enum KmpTaskTFields {
  /// \brief List of shared variables.
  KmpTaskTShareds,
  /// \brief Task routine.
  KmpTaskTRoutine,
  /// \brief Partition id for the untied tasks.
  KmpTaskTPartId,
  /// \brief Function with call of destructors for private variables.
  KmpTaskTDestructors,
};
} // namespace

void CGOpenMPRuntime::emitKmpRoutineEntryT(QualType KmpInt32Ty) {
  if (!KmpRoutineEntryPtrTy) {
    // Build typedef kmp_int32 (* kmp_routine_entry_t)(kmp_int32, void *); type.
    auto &C = CGM.getContext();
    QualType KmpRoutineEntryTyArgs[] = {KmpInt32Ty, C.VoidPtrTy};
    FunctionProtoType::ExtProtoInfo EPI;
    KmpRoutineEntryPtrQTy = C.getPointerType(
        C.getFunctionType(KmpInt32Ty, KmpRoutineEntryTyArgs, EPI));
    KmpRoutineEntryPtrTy = CGM.getTypes().ConvertType(KmpRoutineEntryPtrQTy);
  }
}

static void addFieldToRecordDecl(ASTContext &C, DeclContext *DC,
                                 QualType FieldTy) {
  auto *Field = FieldDecl::Create(
      C, DC, SourceLocation(), SourceLocation(), /*Id=*/nullptr, FieldTy,
      C.getTrivialTypeSourceInfo(FieldTy, SourceLocation()),
      /*BW=*/nullptr, /*Mutable=*/false, /*InitStyle=*/ICIS_NoInit);
  Field->setAccess(AS_public);
  DC->addDecl(Field);
}

static QualType createKmpTaskTRecordDecl(CodeGenModule &CGM,
                                         QualType KmpInt32Ty,
                                         QualType KmpRoutineEntryPointerQTy) {
  auto &C = CGM.getContext();
  // Build struct kmp_task_t {
  //         void *              shareds;
  //         kmp_routine_entry_t routine;
  //         kmp_int32           part_id;
  //         kmp_routine_entry_t destructors;
  //         /*  private vars  */
  //       };
  auto *RD = C.buildImplicitRecord("kmp_task_t");
  RD->startDefinition();
  addFieldToRecordDecl(C, RD, C.VoidPtrTy);
  addFieldToRecordDecl(C, RD, KmpRoutineEntryPointerQTy);
  addFieldToRecordDecl(C, RD, KmpInt32Ty);
  addFieldToRecordDecl(C, RD, KmpRoutineEntryPointerQTy);
  // TODO: add private fields.
  RD->completeDefinition();
  return C.getRecordType(RD);
}

/// \brief Emit a proxy function which accepts kmp_task_t as the second
/// argument.
/// \code
/// kmp_int32 .omp_task_entry.(kmp_int32 gtid, kmp_task_t *tt) {
///   TaskFunction(gtid, tt->part_id, tt->shareds);
///   return 0;
/// }
/// \endcode
static llvm::Value *
emitProxyTaskFunction(CodeGenModule &CGM, SourceLocation Loc,
                      QualType KmpInt32Ty, QualType KmpTaskTPtrQTy,
                      QualType SharedsPtrTy, llvm::Value *TaskFunction) {
  auto &C = CGM.getContext();
  FunctionArgList Args;
  ImplicitParamDecl GtidArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, KmpInt32Ty);
  ImplicitParamDecl TaskTypeArg(C, /*DC=*/nullptr, Loc,
                                /*Id=*/nullptr, KmpTaskTPtrQTy);
  Args.push_back(&GtidArg);
  Args.push_back(&TaskTypeArg);
  FunctionType::ExtInfo Info;
  auto &TaskEntryFnInfo =
      CGM.getTypes().arrangeFreeFunctionDeclaration(KmpInt32Ty, Args, Info,
                                                    /*isVariadic=*/false);
  auto *TaskEntryTy = CGM.getTypes().GetFunctionType(TaskEntryFnInfo);
  auto *TaskEntry =
      llvm::Function::Create(TaskEntryTy, llvm::GlobalValue::InternalLinkage,
                             ".omp_task_entry.", &CGM.getModule());
  CGM.SetLLVMFunctionAttributes(/*D=*/nullptr, TaskEntryFnInfo, TaskEntry);
  CodeGenFunction CGF(CGM);
  CGF.disableDebugInfo();
  CGF.StartFunction(GlobalDecl(), KmpInt32Ty, TaskEntry, TaskEntryFnInfo, Args);

  // TaskFunction(gtid, tt->part_id, tt->shareds);
  auto *GtidParam = CGF.EmitLoadOfScalar(
      CGF.GetAddrOfLocalVar(&GtidArg), /*Volatile=*/false,
      C.getTypeAlignInChars(KmpInt32Ty).getQuantity(), KmpInt32Ty, Loc);
  auto TaskTypeArgAddr = CGF.EmitLoadOfScalar(
      CGF.GetAddrOfLocalVar(&TaskTypeArg), /*Volatile=*/false,
      CGM.PointerAlignInBytes, KmpTaskTPtrQTy, Loc);
  auto *PartidPtr = CGF.Builder.CreateStructGEP(TaskTypeArgAddr,
                                                /*Idx=*/KmpTaskTPartId);
  auto *PartidParam = CGF.EmitLoadOfScalar(
      PartidPtr, /*Volatile=*/false,
      C.getTypeAlignInChars(KmpInt32Ty).getQuantity(), KmpInt32Ty, Loc);
  auto *SharedsPtr = CGF.Builder.CreateStructGEP(TaskTypeArgAddr,
                                                 /*Idx=*/KmpTaskTShareds);
  auto *SharedsParam =
      CGF.EmitLoadOfScalar(SharedsPtr, /*Volatile=*/false,
                           CGM.PointerAlignInBytes, C.VoidPtrTy, Loc);
  llvm::Value *CallArgs[] = {
      GtidParam, PartidParam,
      CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
          SharedsParam, CGF.ConvertTypeForMem(SharedsPtrTy))};
  CGF.EmitCallOrInvoke(TaskFunction, CallArgs);
  CGF.EmitStoreThroughLValue(
      RValue::get(CGF.Builder.getInt32(/*C=*/0)),
      CGF.MakeNaturalAlignAddrLValue(CGF.ReturnValue, KmpInt32Ty));
  CGF.FinishFunction();
  return TaskEntry;
}

void CGOpenMPRuntime::emitTaskCall(
    CodeGenFunction &CGF, SourceLocation Loc, bool Tied,
    llvm::PointerIntPair<llvm::Value *, 1, bool> Final,
    llvm::Value *TaskFunction, QualType SharedsTy, llvm::Value *Shareds) {
  auto &C = CGM.getContext();
  auto KmpInt32Ty = C.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/1);
  // Build type kmp_routine_entry_t (if not built yet).
  emitKmpRoutineEntryT(KmpInt32Ty);
  // Build particular struct kmp_task_t for the given task.
  auto KmpTaskQTy =
      createKmpTaskTRecordDecl(CGM, KmpInt32Ty, KmpRoutineEntryPtrQTy);
  QualType KmpTaskTPtrQTy = C.getPointerType(KmpTaskQTy);
  auto KmpTaskTPtrTy = CGF.ConvertType(KmpTaskQTy)->getPointerTo();
  auto KmpTaskTySize = CGM.getSize(C.getTypeSizeInChars(KmpTaskQTy));
  QualType SharedsPtrTy = C.getPointerType(SharedsTy);

  // Build a proxy function kmp_int32 .omp_task_entry.(kmp_int32 gtid,
  // kmp_task_t *tt);
  auto *TaskEntry = emitProxyTaskFunction(CGM, Loc, KmpInt32Ty, KmpTaskTPtrQTy,
                                          SharedsPtrTy, TaskFunction);

  // Build call kmp_task_t * __kmpc_omp_task_alloc(ident_t *, kmp_int32 gtid,
  // kmp_int32 flags, size_t sizeof_kmp_task_t, size_t sizeof_shareds,
  // kmp_routine_entry_t *task_entry);
  // Task flags. Format is taken from
  // http://llvm.org/svn/llvm-project/openmp/trunk/runtime/src/kmp.h,
  // description of kmp_tasking_flags struct.
  const unsigned TiedFlag = 0x1;
  const unsigned FinalFlag = 0x2;
  unsigned Flags = Tied ? TiedFlag : 0;
  auto *TaskFlags =
      Final.getPointer()
          ? CGF.Builder.CreateSelect(Final.getPointer(),
                                     CGF.Builder.getInt32(FinalFlag),
                                     CGF.Builder.getInt32(/*C=*/0))
          : CGF.Builder.getInt32(Final.getInt() ? FinalFlag : 0);
  TaskFlags = CGF.Builder.CreateOr(TaskFlags, CGF.Builder.getInt32(Flags));
  auto SharedsSize = C.getTypeSizeInChars(SharedsTy);
  llvm::Value *AllocArgs[] = {emitUpdateLocation(CGF, Loc),
                              getThreadID(CGF, Loc), TaskFlags, KmpTaskTySize,
                              CGM.getSize(SharedsSize),
                              CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
                                  TaskEntry, KmpRoutineEntryPtrTy)};
  auto *NewTask = CGF.EmitRuntimeCall(
      createRuntimeFunction(OMPRTL__kmpc_omp_task_alloc), AllocArgs);
  auto *NewTaskNewTaskTTy =
      CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(NewTask, KmpTaskTPtrTy);
  // Fill the data in the resulting kmp_task_t record.
  // Copy shareds if there are any.
  if (!SharedsTy->getAsStructureType()->getDecl()->field_empty())
    CGF.EmitAggregateCopy(
        CGF.EmitLoadOfScalar(
            CGF.Builder.CreateStructGEP(NewTaskNewTaskTTy,
                                        /*Idx=*/KmpTaskTShareds),
            /*Volatile=*/false, CGM.PointerAlignInBytes, SharedsPtrTy, Loc),
        Shareds, SharedsTy);
  // TODO: generate function with destructors for privates.
  // Provide pointer to function with destructors for privates.
  CGF.Builder.CreateAlignedStore(
      llvm::ConstantPointerNull::get(
          cast<llvm::PointerType>(KmpRoutineEntryPtrTy)),
      CGF.Builder.CreateStructGEP(NewTaskNewTaskTTy,
                                  /*Idx=*/KmpTaskTDestructors),
      CGM.PointerAlignInBytes);

  // NOTE: routine and part_id fields are intialized by __kmpc_omp_task_alloc()
  // libcall.
  // Build kmp_int32 __kmpc_omp_task(ident_t *, kmp_int32 gtid, kmp_task_t
  // *new_task);
  llvm::Value *TaskArgs[] = {emitUpdateLocation(CGF, Loc),
                             getThreadID(CGF, Loc), NewTask};
  // TODO: add check for untied tasks.
  CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_omp_task), TaskArgs);
}

InlinedOpenMPRegionRAII::InlinedOpenMPRegionRAII(
    CodeGenFunction &CGF, const OMPExecutableDirective &D)
    : CGF(CGF) {
  CGF.CapturedStmtInfo = new CGOpenMPInlinedRegionInfo(D, CGF.CapturedStmtInfo);
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CGF.EHStack.pushTerminate();
}

InlinedOpenMPRegionRAII::~InlinedOpenMPRegionRAII() {
  CGF.EHStack.popTerminate();
  auto *OldCSI =
      cast<CGOpenMPInlinedRegionInfo>(CGF.CapturedStmtInfo)->getOldCSI();
  delete CGF.CapturedStmtInfo;
  CGF.CapturedStmtInfo = OldCSI;
}

