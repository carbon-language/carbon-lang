//===- IRBindings.cpp - Additional bindings for ir ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines additional C bindings for the ir component.
//
//===----------------------------------------------------------------------===//

#include "IRBindings.h"

#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"

using namespace llvm;

void LLVMAddFunctionAttr2(LLVMValueRef Fn, uint64_t PA) {
  Function *Func = unwrap<Function>(Fn);
  const AttributeSet PAL = Func->getAttributes();
  AttrBuilder B(PA);
  const AttributeSet PALnew =
    PAL.addAttributes(Func->getContext(), AttributeSet::FunctionIndex,
                      AttributeSet::get(Func->getContext(),
                                        AttributeSet::FunctionIndex, B));
  Func->setAttributes(PALnew);
}

uint64_t LLVMGetFunctionAttr2(LLVMValueRef Fn) {
  Function *Func = unwrap<Function>(Fn);
  const AttributeSet PAL = Func->getAttributes();
  return PAL.Raw(AttributeSet::FunctionIndex);
}

void LLVMRemoveFunctionAttr2(LLVMValueRef Fn, uint64_t PA) {
  Function *Func = unwrap<Function>(Fn);
  const AttributeSet PAL = Func->getAttributes();
  AttrBuilder B(PA);
  const AttributeSet PALnew =
    PAL.removeAttributes(Func->getContext(), AttributeSet::FunctionIndex,
                         AttributeSet::get(Func->getContext(),
                                           AttributeSet::FunctionIndex, B));
  Func->setAttributes(PALnew);
}
