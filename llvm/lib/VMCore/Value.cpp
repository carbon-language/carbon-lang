//===-- Value.cpp - Implement the Value class -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Value and User classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/Constant.h"
#include "llvm/DerivedTypes.h"
#include "llvm/InstrTypes.h"
#include "llvm/Module.h"
#include "llvm/SymbolTable.h"
#include "llvm/Support/LeakDetector.h"
#include <algorithm>
#include <iostream>
using namespace llvm;

//===----------------------------------------------------------------------===//
//                                Value Class
//===----------------------------------------------------------------------===//

static inline const Type *checkType(const Type *Ty) {
  assert(Ty && "Value defined with a null type: Error!");
  return Ty;
}

Value::Value(const Type *ty, unsigned scid, const std::string &name)
  : SubclassID(scid), SubclassData(0), Ty(checkType(ty)),
    UseList(0), Name(name) {
  if (!isa<Constant>(this) && !isa<BasicBlock>(this))
    assert((Ty->isFirstClassType() || Ty == Type::VoidTy ||
           isa<OpaqueType>(ty)) &&
           "Cannot create non-first-class values except for constants!");
  if (ty == Type::VoidTy)
    assert(name.empty() && "Cannot have named void values!");
}

Value::~Value() {
#ifndef NDEBUG      // Only in -g mode...
  // Check to make sure that there are no uses of this value that are still
  // around when the value is destroyed.  If there are, then we have a dangling
  // reference and something is wrong.  This code is here to print out what is
  // still being referenced.  The value in question should be printed as
  // a <badref>
  //
  if (use_begin() != use_end()) {
    std::cerr << "While deleting: " << *Ty << " %" << Name << "\n";
    for (use_iterator I = use_begin(), E = use_end(); I != E; ++I)
      std::cerr << "Use still stuck around after Def is destroyed:"
                << **I << "\n";
  }
#endif
  assert(use_begin() == use_end() && "Uses remain when a value is destroyed!");

  // There should be no uses of this object anymore, remove it.
  LeakDetector::removeGarbageObject(this);
}

/// hasNUses - Return true if this Value has exactly N users.
///
bool Value::hasNUses(unsigned N) const {
  use_const_iterator UI = use_begin(), E = use_end();

  for (; N; --N, ++UI)
    if (UI == E) return false;  // Too few.
  return UI == E;
}

/// hasNUsesOrMore - Return true if this value has N users or more.  This is
/// logically equivalent to getNumUses() >= N.
///
bool Value::hasNUsesOrMore(unsigned N) const {
  use_const_iterator UI = use_begin(), E = use_end();

  for (; N; --N, ++UI)
    if (UI == E) return false;  // Too few.

  return true;
}


/// getNumUses - This method computes the number of uses of this Value.  This
/// is a linear time operation.  Use hasOneUse or hasNUses to check for specific
/// values.
unsigned Value::getNumUses() const {
  return (unsigned)std::distance(use_begin(), use_end());
}


void Value::setName(const std::string &name) {
  if (Name == name) return;   // Name is already set.

  // Get the symbol table to update for this object.
  SymbolTable *ST = 0;
  if (Instruction *I = dyn_cast<Instruction>(this)) {
    if (BasicBlock *P = I->getParent())
      if (Function *PP = P->getParent())
        ST = &PP->getSymbolTable();
  } else if (BasicBlock *BB = dyn_cast<BasicBlock>(this)) {
    if (Function *P = BB->getParent()) ST = &P->getSymbolTable();
  } else if (GlobalValue *GV = dyn_cast<GlobalValue>(this)) {
    if (Module *P = GV->getParent()) ST = &P->getSymbolTable();
  } else if (Argument *A = dyn_cast<Argument>(this)) {
    if (Function *P = A->getParent()) ST = &P->getSymbolTable();
  } else {
    assert(isa<Constant>(this) && "Unknown value type!");
    return;  // no name is setable for this.
  }

  if (!ST)  // No symbol table to update?  Just do the change.
    Name = name;
  else if (hasName()) {
    if (!name.empty()) {    // Replacing name.
      ST->changeName(this, name);
    } else {                // Transitioning from hasName -> noname.
      ST->remove(this);
      Name.clear();
    }
  } else {                  // Transitioning from noname -> hasName.
    Name = name;
    ST->insert(this);
  }
}

// uncheckedReplaceAllUsesWith - This is exactly the same as replaceAllUsesWith,
// except that it doesn't have all of the asserts.  The asserts fail because we
// are half-way done resolving types, which causes some types to exist as two
// different Type*'s at the same time.  This is a sledgehammer to work around
// this problem.
//
void Value::uncheckedReplaceAllUsesWith(Value *New) {
  while (!use_empty()) {
    Use &U = *UseList;
    // Must handle Constants specially, we cannot call replaceUsesOfWith on a
    // constant!
    if (Constant *C = dyn_cast<Constant>(U.getUser())) {
      if (!isa<GlobalValue>(C))
        C->replaceUsesOfWithOnConstant(this, New, &U);
      else
        U.set(New);
    } else {
      U.set(New);
    }
  }
}

void Value::replaceAllUsesWith(Value *New) {
  assert(New && "Value::replaceAllUsesWith(<null>) is invalid!");
  assert(New != this && "this->replaceAllUsesWith(this) is NOT valid!");
  assert(New->getType() == getType() &&
         "replaceAllUses of value with new value of different type!");

  uncheckedReplaceAllUsesWith(New);
}

//===----------------------------------------------------------------------===//
//                                 User Class
//===----------------------------------------------------------------------===//

// replaceUsesOfWith - Replaces all references to the "From" definition with
// references to the "To" definition.
//
void User::replaceUsesOfWith(Value *From, Value *To) {
  if (From == To) return;   // Duh what?

  assert(!isa<Constant>(this) || isa<GlobalValue>(this) &&
         "Cannot call User::replaceUsesofWith on a constant!");

  for (unsigned i = 0, E = getNumOperands(); i != E; ++i)
    if (getOperand(i) == From) {  // Is This operand is pointing to oldval?
      // The side effects of this setOperand call include linking to
      // "To", adding "this" to the uses list of To, and
      // most importantly, removing "this" from the use list of "From".
      setOperand(i, To); // Fix it now...
    }
}

