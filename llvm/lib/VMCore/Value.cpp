//===-- Value.cpp - Implement the Value class -----------------------------===//
//
// This file implements the Value, User, and SymTabValue classes. 
//
//===----------------------------------------------------------------------===//

#include "llvm/ValueHolderImpl.h"
#include "llvm/InstrTypes.h"
#include "llvm/SymbolTable.h"
#include "llvm/SymTabValue.h"
#include "llvm/Type.h"
#ifndef NDEBUG      // Only in -g mode...
#include "llvm/Assembly/Writer.h"
#endif
#include <algorithm>

//===----------------------------------------------------------------------===//
//                                Value Class
//===----------------------------------------------------------------------===//

Value::Value(const Type *ty, ValueTy vty, const string &name = "")
  : Name(name), Ty(ty, this) {
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
    cerr << "While deleting: " << this;
    for (use_const_iterator I = Uses.begin(); I != Uses.end(); ++I)
      cerr << "Use still stuck around after Def is destroyed:" << *I << endl;
  }
#endif
  assert(Uses.begin() == Uses.end());
}

void Value::replaceAllUsesWith(Value *D) {
  assert(D && "Value::replaceAllUsesWith(<null>) is invalid!");
  assert(D != this && "V->replaceAllUsesWith(V) is NOT valid!");
  while (!Uses.empty()) {
    User *Use = Uses.back();
#ifndef NDEBUG
    unsigned NumUses = Uses.size();
#endif
    Use->replaceUsesOfWith(this, D);

#ifndef NDEBUG      // only in -g mode...
    if (Uses.size() == NumUses)
      cerr << "Use: " << Use << "replace with: " << D; 
#endif
    assert(Uses.size() != NumUses && "Didn't remove definition!");
  }
}

// refineAbstractType - This function is implemented because we use
// potentially abstract types, and these types may be resolved to more
// concrete types after we are constructed.  For the value class, we simply
// change Ty to point to the right type.  :)
//
void Value::refineAbstractType(const DerivedType *OldTy, const Type *NewTy) {
  assert(Ty.get() == (const Type*)OldTy &&"Can't refine anything but my type!");
  Ty = NewTy;
}

void Value::killUse(User *i) {
  if (i == 0) return;
  use_iterator I = find(Uses.begin(), Uses.end(), i);

  assert(I != Uses.end() && "Use not in uses list!!");
  Uses.erase(I);
}

User *Value::use_remove(use_iterator &I) {
  assert(I != Uses.end() && "Trying to remove the end of the use list!!!");
  User *i = *I;
  I = Uses.erase(I);
  return i;
}

#ifndef NDEBUG      // Only in -g mode...
void Value::dump() const {
  cerr << this;
}
#endif

//===----------------------------------------------------------------------===//
//                                 User Class
//===----------------------------------------------------------------------===//

User::User(const Type *Ty, ValueTy vty, const string &name) 
  : Value(Ty, vty, name) {
}

// replaceUsesOfWith - Replaces all references to the "From" definition with
// references to the "To" definition.
//
void User::replaceUsesOfWith(Value *From, Value *To) {
  if (From == To) return;   // Duh what?

  for (unsigned i = 0, E = getNumOperands(); i != E; ++i)
    if (getOperand(i) == From) {  // Is This operand is pointing to oldval?
      // The side effects of this setOperand call include linking to
      // "To", adding "this" to the uses list of To, and
      // most importantly, removing "this" from the use list of "From".
      setOperand(i, To); // Fix it now...
    }
}


//===----------------------------------------------------------------------===//
//                             SymTabValue Class
//===----------------------------------------------------------------------===//

SymTabValue::SymTabValue(Value *p) : ValueParent(p) { 
  assert(ValueParent && "SymTavValue without parent!?!");
  ParentSymTab = SymTab = 0;
}


SymTabValue::~SymTabValue() {
  delete SymTab;
}

void SymTabValue::setParentSymTab(SymbolTable *ST) {
  ParentSymTab = ST;
  if (SymTab) 
    SymTab->setParentSymTab(ST);
}

SymbolTable *SymTabValue::getSymbolTableSure() {
  if (!SymTab) SymTab = new SymbolTable(ParentSymTab);
  return SymTab;
}

// hasSymbolTable() - Returns true if there is a symbol table allocated to
// this object AND if there is at least one name in it!
//
bool SymTabValue::hasSymbolTable() const {
  if (!SymTab) return false;

  for (SymbolTable::const_iterator I = SymTab->begin(); 
       I != SymTab->end(); ++I) {
    if (I->second.begin() != I->second.end())
      return true;                                // Found nonempty type plane!
  }
  
  return false;
}
