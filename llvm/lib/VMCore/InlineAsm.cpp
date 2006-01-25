//===-- InlineAsm.cpp - Implement the InlineAsm class ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the InlineAsm class.
//
//===----------------------------------------------------------------------===//

#include "llvm/InlineAsm.h"
#include "llvm/DerivedTypes.h"
using namespace llvm;

// NOTE: when memoizing the function type, we have to be careful to handle the
// case when the type gets refined.

InlineAsm *InlineAsm::get(const FunctionType *Ty, const std::string &AsmString,
                          const std::string &Constraints, bool hasSideEffects) {
  // FIXME: memoize!
  return new InlineAsm(Ty, AsmString, Constraints, hasSideEffects);  
}

InlineAsm::InlineAsm(const FunctionType *Ty, const std::string &asmString,
                     const std::string &constraints, bool hasSideEffects)
  : Value(PointerType::get(Ty), Value::InlineAsmVal), AsmString(asmString), 
    Constraints(constraints), HasSideEffects(hasSideEffects) {

  // Do various checks on the constraint string and type.
  assert(Verify(Ty, constraints) && "Function type not legal for constraints!");
}

const FunctionType *InlineAsm::getFunctionType() const {
  return cast<FunctionType>(getType()->getElementType());
}

bool InlineAsm::Verify(const FunctionType *Ty, const std::string &Constraints) {
  return true;
}
