//===--- RuntimeDebugBuilder.cpp - Helper to insert prints into LLVM-IR ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/RuntimeDebugBuilder.h"

#include "llvm/IR/Module.h"

using namespace llvm;
using namespace polly;

Function *RuntimeDebugBuilder::getPrintF(PollyIRBuilder &Builder) {
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  const char *Name = "printf";
  Function *F = M->getFunction(Name);

  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    FunctionType *Ty =
        FunctionType::get(Builder.getInt32Ty(), Builder.getInt8PtrTy(), true);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  return F;
}

void RuntimeDebugBuilder::createFlush(PollyIRBuilder &Builder) {
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  const char *Name = "fflush";
  Function *F = M->getFunction(Name);

  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    FunctionType *Ty =
        FunctionType::get(Builder.getInt32Ty(), Builder.getInt8PtrTy(), false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  // fflush(NULL) flushes _all_ open output streams.
  //
  // fflush is declared as 'int fflush(FILE *stream)'. As we only pass on a NULL
  // pointer, the type we point to does conceptually not matter. However, if
  // fflush is already declared in this translation unit, we use the very same
  // type to ensure that LLVM does not complain about mismatching types.
  Builder.CreateCall(F, Constant::getNullValue(F->arg_begin()->getType()));
}

void RuntimeDebugBuilder::createStrPrinter(PollyIRBuilder &Builder,
                                           const std::string &String) {
  Value *StringValue = Builder.CreateGlobalStringPtr(String);
  Builder.CreateCall(getPrintF(Builder), StringValue);

  createFlush(Builder);
}

void RuntimeDebugBuilder::createValuePrinter(PollyIRBuilder &Builder,
                                             Value *V) {
  const char *Format = nullptr;

  Type *Ty = V->getType();
  if (Ty->isIntegerTy())
    Format = "%ld";
  else if (Ty->isFloatingPointTy())
    Format = "%lf";
  else if (Ty->isPointerTy())
    Format = "%p";

  assert(Format && Ty->getPrimitiveSizeInBits() <= 64 && "Bad type to print.");

  Value *FormatString = Builder.CreateGlobalStringPtr(Format);
  Builder.CreateCall2(getPrintF(Builder), FormatString, V);
  createFlush(Builder);
}
