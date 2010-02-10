//===---- IRBuilder.cpp - Builder for LLVM Instrs -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the IRBuilder class, which is used as a convenient way
// to create LLVM instructions with a consistent and simplified interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/IRBuilder.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Function.h"
#include "llvm/LLVMContext.h"
using namespace llvm;

/// CreateGlobalString - Make a new global variable with an initializer that
/// has array of i8 type filled in with the nul terminated string value
/// specified.  If Name is specified, it is the name of the global variable
/// created.
Value *IRBuilderBase::CreateGlobalString(const char *Str, const Twine &Name) {
  Constant *StrConstant = ConstantArray::get(Context, Str, true);
  Module &M = *BB->getParent()->getParent();
  GlobalVariable *GV = new GlobalVariable(M, StrConstant->getType(),
                                          true, GlobalValue::InternalLinkage,
                                          StrConstant, "", 0, false);
  GV->setName(Name);
  return GV;
}

/// SetCurrentDebugLocation - Set location information used by debugging
/// information.
void IRBuilderBase::SetCurrentDebugLocation(MDNode *L) {
  if (DbgMDKind == 0) 
    DbgMDKind = Context.getMDKindID("dbg");
  CurDbgLocation = L;
}

void IRBuilderBase::SetInstDebugLocation(Instruction *I) const {
  if (CurDbgLocation)
    I->setMetadata(DbgMDKind, CurDbgLocation);
}

const Type *IRBuilderBase::getCurrentFunctionReturnType() const {
  assert(BB && BB->getParent() && "No current function!");
  return BB->getParent()->getReturnType();
}
