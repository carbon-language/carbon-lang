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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"
#include <assert.h>

using namespace clang;
using namespace CodeGen;

CGOpenMPRuntime::CGOpenMPRuntime(CodeGenModule &CGM)
    : CGM(CGM), DefaultOpenMPPSource(nullptr) {
  IdentTy = llvm::StructType::create(
      "ident_t", CGM.Int32Ty /* reserved_1 */, CGM.Int32Ty /* flags */,
      CGM.Int32Ty /* reserved_2 */, CGM.Int32Ty /* reserved_3 */,
      CGM.Int8PtrTy /* psource */, NULL);
  // Build void (*kmpc_micro)(kmp_int32 *global_tid, kmp_int32 *bound_tid,...)
  llvm::Type *MicroParams[] = { llvm::PointerType::getUnqual(CGM.Int32Ty),
                                llvm::PointerType::getUnqual(CGM.Int32Ty) };
  Kmpc_MicroTy = llvm::FunctionType::get(CGM.VoidTy, MicroParams, true);
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
    llvm::GlobalVariable *DefaultOpenMPLocation = cast<llvm::GlobalVariable>(
        CGM.CreateRuntimeVariable(IdentTy, ".kmpc_default_loc.addr"));
    DefaultOpenMPLocation->setUnnamedAddr(true);
    DefaultOpenMPLocation->setConstant(true);
    DefaultOpenMPLocation->setLinkage(llvm::GlobalValue::PrivateLinkage);

    llvm::Constant *Zero = llvm::ConstantInt::get(CGM.Int32Ty, 0, true);
    llvm::Constant *Values[] = { Zero,
                                 llvm::ConstantInt::get(CGM.Int32Ty, Flags),
                                 Zero, Zero, DefaultOpenMPPSource };
    llvm::Constant *Init = llvm::ConstantStruct::get(IdentTy, Values);
    DefaultOpenMPLocation->setInitializer(Init);
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
  OpenMPLocMapTy::iterator I = OpenMPLocMap.find(CGF.CurFn);
  if (I != OpenMPLocMap.end()) {
    LocValue = I->second;
  } else {
    // Generate "ident_t .kmpc_loc.addr;"
    llvm::AllocaInst *AI = CGF.CreateTempAlloca(IdentTy, ".kmpc_loc.addr");
    AI->setAlignment(CGM.getDataLayout().getPrefTypeAlignment(IdentTy));
    OpenMPLocMap[CGF.CurFn] = AI;
    LocValue = AI;

    CGBuilderTy::InsertPointGuard IPG(CGF.Builder);
    CGF.Builder.SetInsertPoint(CGF.AllocaInsertPt);
    CGF.Builder.CreateMemCpy(LocValue, GetOrCreateDefaultOpenMPLocation(Flags),
                             llvm::ConstantExpr::getSizeOf(IdentTy),
                             CGM.PointerAlignInBytes);
  }

  // char **psource = &.kmpc_loc_<flags>.addr.psource;
  llvm::Value *PSource =
      CGF.Builder.CreateConstInBoundsGEP2_32(LocValue, 0, IdentField_PSource);

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
  // *psource = ";<File>;<Function>;<Line>;<Column>;;";
  CGF.Builder.CreateStore(CGF.Builder.CreateGlobalStringPtr(OS2.str()),
                          PSource);
  return LocValue;
}

llvm::Value *CGOpenMPRuntime::GetOpenMPGlobalThreadNum(CodeGenFunction &CGF,
                                                       SourceLocation Loc) {
  assert(CGF.CurFn && "No function in current CodeGenFunction.");

  llvm::Value *GTid = nullptr;
  OpenMPGtidMapTy::iterator I = OpenMPGtidMap.find(CGF.CurFn);
  if (I != OpenMPGtidMap.end()) {
    GTid = I->second;
  } else {
    // Generate "int32 .kmpc_global_thread_num.addr;"
    CGBuilderTy::InsertPointGuard IPG(CGF.Builder);
    CGF.Builder.SetInsertPoint(CGF.AllocaInsertPt);
    llvm::Value *Args[] = { EmitOpenMPUpdateLocation(CGF, Loc) };
    GTid = CGF.EmitRuntimeCall(
        CreateRuntimeFunction(OMPRTL__kmpc_global_thread_num), Args);
    OpenMPGtidMap[CGF.CurFn] = GTid;
  }
  return GTid;
}

void CGOpenMPRuntime::FunctionFinished(CodeGenFunction &CGF) {
  assert(CGF.CurFn && "No function in current CodeGenFunction.");
  if (OpenMPGtidMap.count(CGF.CurFn))
    OpenMPGtidMap.erase(CGF.CurFn);
  if (OpenMPLocMap.count(CGF.CurFn))
    OpenMPLocMap.erase(CGF.CurFn);
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
    llvm::Type *TypeParams[] = { getIdentTyPointerTy(), CGM.Int32Ty,
                                 getKmpc_MicroPointerTy() };
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, true);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_fork_call");
    break;
  }
  case OMPRTL__kmpc_global_thread_num: {
    // Build kmp_int32 __kmpc_global_thread_num(ident_t *loc);
    llvm::Type *TypeParams[] = { getIdentTyPointerTy() };
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_global_thread_num");
    break;
  }
  }
  return RTLFn;
}
