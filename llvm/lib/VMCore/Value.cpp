//===-- Value.cpp - Implement the Value class -----------------------------===//
//
// This file implements the Value and User classes. 
//
//===----------------------------------------------------------------------===//

#include "llvm/InstrTypes.h"
#include "llvm/SymbolTable.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constant.h"
#include "Support/LeakDetector.h"
#include <algorithm>

//===----------------------------------------------------------------------===//
//                                Value Class
//===----------------------------------------------------------------------===//

static inline const Type *checkType(const Type *Ty) {
  assert(Ty && "Value defined with a null type: Error!");
  return Ty;
}

Value::Value(const Type *ty, ValueTy vty, const std::string &name)
  : Name(name), Ty(checkType(ty)) {
  VTy = vty;
}

Value::~Value() {
#ifndef NDEBUG      // Only in -g mode...
  // Check to make sure that there are no uses of this value that are still
  // around when the value is destroyed.  If there are, then we have a dangling
  // reference and something is wrong.  This code is here to print out what is
  // still being referenced.  The value in question should be printed as 
  // a <badref>
  //
  if (Uses.begin() != Uses.end()) {
    std::cerr << "While deleting: " << *Ty << "%" << Name << "\n";
    for (use_const_iterator I = Uses.begin(); I != Uses.end(); ++I)
      std::cerr << "Use still stuck around after Def is destroyed:"
                << **I << "\n";
  }
#endif
  assert(Uses.begin() == Uses.end() &&"Uses remain when a value is destroyed!");

  // There should be no uses of this object anymore, remove it.
  LeakDetector::removeGarbageObject(this);
}




void Value::replaceAllUsesWith(Value *New) {
  assert(New && "Value::replaceAllUsesWith(<null>) is invalid!");
  assert(New != this && "this->replaceAllUsesWith(this) is NOT valid!");
  assert(New->getType() == getType() &&
         "replaceAllUses of value with new value of different type!");
  while (!Uses.empty()) {
    User *Use = Uses.back();
    // Must handle Constants specially, we cannot call replaceUsesOfWith on a
    // constant!
    if (Constant *C = dyn_cast<Constant>(Use)) {
      C->replaceUsesOfWithOnConstant(this, New);
    } else {
      Use->replaceUsesOfWith(this, New);
    }
  }
}

// uncheckedReplaceAllUsesWith - This is exactly the same as replaceAllUsesWith,
// except that it doesn't have all of the asserts.  The asserts fail because we
// are half-way done resolving types, which causes some types to exist as two
// different Type*'s at the same time.  This is a sledgehammer to work around
// this problem.
//
void Value::uncheckedReplaceAllUsesWith(Value *New) {
  while (!Uses.empty()) {
    User *Use = Uses.back();
    // Must handle Constants specially, we cannot call replaceUsesOfWith on a
    // constant!
    if (Constant *C = dyn_cast<Constant>(Use)) {
      C->replaceUsesOfWithOnConstant(this, New, true);
    } else {
      Use->replaceUsesOfWith(this, New);
    }
  }
}


void Value::killUse(User *U) {
  if (U == 0) return;
  unsigned i;

  // Scan backwards through the uses list looking for the user.  We do this
  // because vectors like to be accessed on the end.  This is incredibly
  // important from a performance perspective.
  for (i = Uses.size()-1; Uses[i] != U; --i)
    /* empty */;

  assert(i < Uses.size() && "Use not in uses list!!");
  Uses[i] = Uses.back();
  Uses.pop_back();
}

//===----------------------------------------------------------------------===//
//                                 User Class
//===----------------------------------------------------------------------===//

User::User(const Type *Ty, ValueTy vty, const std::string &name) 
  : Value(Ty, vty, name) {
}

// replaceUsesOfWith - Replaces all references to the "From" definition with
// references to the "To" definition.
//
void User::replaceUsesOfWith(Value *From, Value *To) {
  if (From == To) return;   // Duh what?

  assert(!isa<Constant>(this) &&
         "Cannot call User::replaceUsesofWith on a constant!");

  for (unsigned i = 0, E = getNumOperands(); i != E; ++i)
    if (getOperand(i) == From) {  // Is This operand is pointing to oldval?
      // The side effects of this setOperand call include linking to
      // "To", adding "this" to the uses list of To, and
      // most importantly, removing "this" from the use list of "From".
      setOperand(i, To); // Fix it now...
    }
}
