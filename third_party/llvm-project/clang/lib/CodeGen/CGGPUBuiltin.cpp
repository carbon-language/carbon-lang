//===------ CGGPUBuiltin.cpp - Codegen for GPU builtins -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generates code for built-in GPU calls which are not runtime-specific.
// (Runtime-specific codegen lives in programming model specific files.)
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "clang/Basic/Builtins.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/Utils/AMDGPUEmitPrintf.h"

using namespace clang;
using namespace CodeGen;

static llvm::Function *GetVprintfDeclaration(llvm::Module &M) {
  llvm::Type *ArgTypes[] = {llvm::Type::getInt8PtrTy(M.getContext()),
                            llvm::Type::getInt8PtrTy(M.getContext())};
  llvm::FunctionType *VprintfFuncType = llvm::FunctionType::get(
      llvm::Type::getInt32Ty(M.getContext()), ArgTypes, false);

  if (auto* F = M.getFunction("vprintf")) {
    // Our CUDA system header declares vprintf with the right signature, so
    // nobody else should have been able to declare vprintf with a bogus
    // signature.
    assert(F->getFunctionType() == VprintfFuncType);
    return F;
  }

  // vprintf doesn't already exist; create a declaration and insert it into the
  // module.
  return llvm::Function::Create(
      VprintfFuncType, llvm::GlobalVariable::ExternalLinkage, "vprintf", &M);
}

// Transforms a call to printf into a call to the NVPTX vprintf syscall (which
// isn't particularly special; it's invoked just like a regular function).
// vprintf takes two args: A format string, and a pointer to a buffer containing
// the varargs.
//
// For example, the call
//
//   printf("format string", arg1, arg2, arg3);
//
// is converted into something resembling
//
//   struct Tmp {
//     Arg1 a1;
//     Arg2 a2;
//     Arg3 a3;
//   };
//   char* buf = alloca(sizeof(Tmp));
//   *(Tmp*)buf = {a1, a2, a3};
//   vprintf("format string", buf);
//
// buf is aligned to the max of {alignof(Arg1), ...}.  Furthermore, each of the
// args is itself aligned to its preferred alignment.
//
// Note that by the time this function runs, E's args have already undergone the
// standard C vararg promotion (short -> int, float -> double, etc.).
RValue
CodeGenFunction::EmitNVPTXDevicePrintfCallExpr(const CallExpr *E,
                                               ReturnValueSlot ReturnValue) {
  assert(getTarget().getTriple().isNVPTX());
  assert(E->getBuiltinCallee() == Builtin::BIprintf);
  assert(E->getNumArgs() >= 1); // printf always has at least one arg.

  const llvm::DataLayout &DL = CGM.getDataLayout();
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();

  CallArgList Args;
  EmitCallArgs(Args,
               E->getDirectCallee()->getType()->getAs<FunctionProtoType>(),
               E->arguments(), E->getDirectCallee(),
               /* ParamsToSkip = */ 0);

  // We don't know how to emit non-scalar varargs.
  if (std::any_of(Args.begin() + 1, Args.end(), [&](const CallArg &A) {
        return !A.getRValue(*this).isScalar();
      })) {
    CGM.ErrorUnsupported(E, "non-scalar arg to printf");
    return RValue::get(llvm::ConstantInt::get(IntTy, 0));
  }

  // Construct and fill the args buffer that we'll pass to vprintf.
  llvm::Value *BufferPtr;
  if (Args.size() <= 1) {
    // If there are no args, pass a null pointer to vprintf.
    BufferPtr = llvm::ConstantPointerNull::get(llvm::Type::getInt8PtrTy(Ctx));
  } else {
    llvm::SmallVector<llvm::Type *, 8> ArgTypes;
    for (unsigned I = 1, NumArgs = Args.size(); I < NumArgs; ++I)
      ArgTypes.push_back(Args[I].getRValue(*this).getScalarVal()->getType());

    // Using llvm::StructType is correct only because printf doesn't accept
    // aggregates.  If we had to handle aggregates here, we'd have to manually
    // compute the offsets within the alloca -- we wouldn't be able to assume
    // that the alignment of the llvm type was the same as the alignment of the
    // clang type.
    llvm::Type *AllocaTy = llvm::StructType::create(ArgTypes, "printf_args");
    llvm::Value *Alloca = CreateTempAlloca(AllocaTy);

    for (unsigned I = 1, NumArgs = Args.size(); I < NumArgs; ++I) {
      llvm::Value *P = Builder.CreateStructGEP(AllocaTy, Alloca, I - 1);
      llvm::Value *Arg = Args[I].getRValue(*this).getScalarVal();
      Builder.CreateAlignedStore(Arg, P, DL.getPrefTypeAlign(Arg->getType()));
    }
    BufferPtr = Builder.CreatePointerCast(Alloca, llvm::Type::getInt8PtrTy(Ctx));
  }

  // Invoke vprintf and return.
  llvm::Function* VprintfFunc = GetVprintfDeclaration(CGM.getModule());
  return RValue::get(Builder.CreateCall(
      VprintfFunc, {Args[0].getRValue(*this).getScalarVal(), BufferPtr}));
}

RValue
CodeGenFunction::EmitAMDGPUDevicePrintfCallExpr(const CallExpr *E,
                                                ReturnValueSlot ReturnValue) {
  assert(getTarget().getTriple().getArch() == llvm::Triple::amdgcn);
  assert(E->getBuiltinCallee() == Builtin::BIprintf ||
         E->getBuiltinCallee() == Builtin::BI__builtin_printf);
  assert(E->getNumArgs() >= 1); // printf always has at least one arg.

  CallArgList CallArgs;
  EmitCallArgs(CallArgs,
               E->getDirectCallee()->getType()->getAs<FunctionProtoType>(),
               E->arguments(), E->getDirectCallee(),
               /* ParamsToSkip = */ 0);

  SmallVector<llvm::Value *, 8> Args;
  for (auto A : CallArgs) {
    // We don't know how to emit non-scalar varargs.
    if (!A.getRValue(*this).isScalar()) {
      CGM.ErrorUnsupported(E, "non-scalar arg to printf");
      return RValue::get(llvm::ConstantInt::get(IntTy, -1));
    }

    llvm::Value *Arg = A.getRValue(*this).getScalarVal();
    Args.push_back(Arg);
  }

  llvm::IRBuilder<> IRB(Builder.GetInsertBlock(), Builder.GetInsertPoint());
  IRB.SetCurrentDebugLocation(Builder.getCurrentDebugLocation());
  auto Printf = llvm::emitAMDGPUPrintfCall(IRB, Args);
  Builder.SetInsertPoint(IRB.GetInsertBlock(), IRB.GetInsertPoint());
  return RValue::get(Printf);
}
