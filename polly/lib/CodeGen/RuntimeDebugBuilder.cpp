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

  Builder.CreateCall(F, Constant::getNullValue(Builder.getInt8PtrTy()));
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
