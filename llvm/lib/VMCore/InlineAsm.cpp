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
#include "llvm/Module.h"
#include "llvm/Support/LeakDetector.h"
using namespace llvm;

InlineAsm::InlineAsm(const FunctionType *Ty, const std::string &asmString,
                     const std::string &constraints, bool hasSideEffects,
                     const std::string &name, Module *ParentModule)
  : Value(PointerType::get(Ty), Value::InlineAsmVal, name), 
    Parent(0), AsmString(asmString), Constraints(constraints), 
    AsmHasSideEffects(hasSideEffects) {
  LeakDetector::addGarbageObject(this);

  if (ParentModule)
    ParentModule->getInlineAsmList().push_back(this);
}

const FunctionType *InlineAsm::getFunctionType() const {
  return cast<FunctionType>(getType()->getElementType());
}

void InlineAsm::setParent(Module *parent) {
  if (getParent())
    LeakDetector::addGarbageObject(this);
  Parent = parent;
  if (getParent())
    LeakDetector::removeGarbageObject(this);
}

void InlineAsm::removeFromParent() {
  getParent()->getInlineAsmList().remove(this);
}

void InlineAsm::eraseFromParent() {
  getParent()->getInlineAsmList().erase(this);
}
