//===-- Globals.cpp - Implement the GlobalValue & GlobalVariable class ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the GlobalValue & GlobalVariable classes for the VMCore
// library.
//
//===----------------------------------------------------------------------===//

#include "llvm/GlobalVariable.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/SymbolTable.h"
#include "llvm/Support/LeakDetector.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//                            GlobalValue Class
//===----------------------------------------------------------------------===//

/// This could be named "SafeToDestroyGlobalValue". It just makes sure that
/// there are no non-constant uses of this GlobalValue. If there aren't then
/// this and the transitive closure of the constants can be deleted. See the
/// destructor for details.
static bool removeDeadConstantUsers(Constant* C) {
  if (isa<GlobalValue>(C)) return false; // Cannot remove this

  while (!C->use_empty())
    if (Constant *User = dyn_cast<Constant>(C->use_back())) {
      if (!removeDeadConstantUsers(User))
        return false; // Constant wasn't dead
    } else {
      return false; // Non-constant usage;
    }

  C->destroyConstant();
  return true;
}

/// removeDeadConstantUsers - If there are any dead constant users dangling
/// off of this global value, remove them.  This method is useful for clients
/// that want to check to see if a global is unused, but don't want to deal
/// with potentially dead constants hanging off of the globals.
///
/// This function returns true if the global value is now dead.  If all
/// users of this global are not dead, this method may return false and
/// leave some of them around.
void GlobalValue::removeDeadConstantUsers() {
  while(!use_empty()) {
    if (Constant* User = dyn_cast<Constant>(use_back())) {
      if (!::removeDeadConstantUsers(User))
        return; // Constant wasn't dead
    } else {
      return; // Non-constant usage;
    }
  }
}

/// Override destroyConstant to make sure it doesn't get called on
/// GlobalValue's because they shouldn't be treated like other constants.
void GlobalValue::destroyConstant() {
  assert(0 && "You can't GV->destroyConstant()!");
  abort();
}
//===----------------------------------------------------------------------===//
// GlobalVariable Implementation
//===----------------------------------------------------------------------===//

GlobalVariable::GlobalVariable(const Type *Ty, bool constant, LinkageTypes Link,
                               Constant *InitVal,
                               const std::string &Name, Module *ParentModule)
  : GlobalValue(PointerType::get(Ty), Value::GlobalVariableVal,
                &Initializer, InitVal != 0, Link, Name),
    isConstantGlobal(constant) {
  if (InitVal) {
    assert(InitVal->getType() == Ty &&
           "Initializer should be the same type as the GlobalVariable!");
    Initializer.init(InitVal, this);
  } else {
    Initializer.init(0, this);
  }

  LeakDetector::addGarbageObject(this);

  if (ParentModule)
    ParentModule->getGlobalList().push_back(this);
}

void GlobalVariable::setParent(Module *parent) {
  if (getParent())
    LeakDetector::addGarbageObject(this);
  Parent = parent;
  if (getParent())
    LeakDetector::removeGarbageObject(this);
}

void GlobalVariable::removeFromParent() {
  getParent()->getGlobalList().remove(this);
}

void GlobalVariable::eraseFromParent() {
  getParent()->getGlobalList().erase(this);
}

void GlobalVariable::replaceUsesOfWithOnConstant(Value *From, Value *To,
                                                 Use *U) {
  // If you call this, then you better know this GVar has a constant
  // initializer worth replacing. Enforce that here.
  assert(getNumOperands() == 1 &&
         "Attempt to replace uses of Constants on a GVar with no initializer");

  // And, since you know it has an initializer, the From value better be
  // the initializer :)
  assert(getOperand(0) == From &&
         "Attempt to replace wrong constant initializer in GVar");

  // And, you better have a constant for the replacement value
  assert(isa<Constant>(To) &&
         "Attempt to replace GVar initializer with non-constant");

  // Okay, preconditions out of the way, replace the constant initializer.
  this->setOperand(0, cast<Constant>(To));
}
