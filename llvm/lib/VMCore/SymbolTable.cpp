//===-- SymbolTable.cpp - Implement the SymbolTable class -------------------=//
//
// This file implements the SymbolTable class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/SymbolTable.h"
#include "llvm/InstrTypes.h"
#include "llvm/Support/StringExtras.h"
#include "llvm/DerivedTypes.h"

SymbolTable::~SymbolTable() {
  // Drop all abstract type references in the type plane...
  iterator TyPlane = find(Type::TypeTy);
  if (TyPlane != end()) {
    VarMap &TyP = TyPlane->second;
    for (VarMap::iterator I = TyP.begin(), E = TyP.end(); I != E; ++I) {
      const Type *Ty = cast<const Type>(I->second);
      if (Ty->isAbstract())   // If abstract, drop the reference...
	cast<DerivedType>(Ty)->removeAbstractTypeUser(this);
    }
  }
#ifndef NDEBUG   // Only do this in -g mode...
  bool LeftoverValues = true;
  for (iterator i = begin(); i != end(); ++i) {
    for (type_iterator I = i->second.begin(); I != i->second.end(); ++I)
      if (!isa<ConstPoolVal>(I->second) && !isa<Type>(I->second)) {
	cerr << "Value still in symbol table! Type = '"
	     << i->first->getDescription() << "' Name = '" << I->first << "'\n";
	LeftoverValues = false;
      }
  }
  
  assert(LeftoverValues && "Values remain in symbol table!");
#endif
}

SymbolTable::type_iterator SymbolTable::type_find(const Value *D) {
  assert(D->hasName() && "type_find(Value*) only works on named nodes!");
  return type_find(D->getType(), D->getName());
}


// find - returns end(Ty->getIDNumber()) on failure...
SymbolTable::type_iterator SymbolTable::type_find(const Type *Ty, 
                                                  const string &Name) {
  iterator I = find(Ty);
  if (I == end()) {      // Not in collection yet... insert dummy entry
    (*this)[Ty] = VarMap();
    I = find(Ty);
    assert(I != end() && "How did insert fail?");
  }

  return I->second.find(Name);
}

// getUniqueName - Given a base name, return a string that is either equal to
// it (or derived from it) that does not already occur in the symbol table for
// the specified type.
//
string SymbolTable::getUniqueName(const Type *Ty, const string &BaseName) {
  iterator I = find(Ty);
  if (I == end()) return BaseName;

  string TryName = BaseName;
  unsigned Counter = 0;
  type_iterator End = I->second.end();

  while (I->second.find(TryName) != End)     // Loop until we find unoccupied
    TryName = BaseName + utostr(++Counter);  // Name in the symbol table
  return TryName;
}



// lookup - Returns null on failure...
Value *SymbolTable::lookup(const Type *Ty, const string &Name) {
  iterator I = find(Ty);
  if (I != end()) {                      // We have symbols in that plane...
    type_iterator J = I->second.find(Name);
    if (J != I->second.end())            // and the name is in our hash table...
      return J->second;
  }

  return ParentSymTab ? ParentSymTab->lookup(Ty, Name) : 0;
}

void SymbolTable::remove(Value *N) {
  assert(N->hasName() && "Value doesn't have name!");
  assert(type_find(N) != type_end(N->getType()) && 
         "Value not in symbol table!");
  type_remove(type_find(N));
}


#define DEBUG_SYMBOL_TABLE 0

Value *SymbolTable::type_remove(const type_iterator &It) {
  Value *Result = It->second;
  const Type *Ty = Result->getType();
#if DEBUG_SYMBOL_TABLE
  cerr << this << " Removing Value: " << Result->getName() << endl;
#endif

  // Remove the value from the plane...
  find(Ty)->second.erase(It);

  // If we are removing an abstract type, remove the symbol table from it's use
  // list...
  if (Ty == Type::TypeTy) {
    const Type *T = cast<const Type>(Result);
    if (T->isAbstract())
      cast<DerivedType>(T)->removeAbstractTypeUser(this);
  }

  return Result;
}

// insertEntry - Insert a value into the symbol table with the specified
// name...
//
void SymbolTable::insertEntry(const string &Name, const Type *VTy, Value *V) {
  // TODO: The typeverifier should catch this when its implemented
  assert(lookup(VTy, Name) == 0 && 
	 "SymbolTable::insertEntry - Name already in symbol table!");

#if DEBUG_SYMBOL_TABLE
  cerr << this << " Inserting definition: " << Name << ": " 
       << VTy->getDescription() << endl;
#endif

  iterator I = find(VTy);
  if (I == end()) {      // Not in collection yet... insert dummy entry
    // Insert a new empty element.  I points to the new elements.
    I = super::insert(make_pair(VTy, VarMap())).first;
    assert(I != end() && "How did insert fail?");

    // Check to see if the type is abstract.  If so, it might be refined in the
    // future, which would cause the plane of the old type to get merged into
    // a new type plane.
    //
    if (VTy->isAbstract())
      cast<DerivedType>(VTy)->addAbstractTypeUser(this);
  }

  I->second.insert(make_pair(Name, V));

  // If we are adding an abstract type, add the symbol table to it's use list.
  if (VTy == Type::TypeTy) {
    const Type *T = cast<const Type>(V);
    if (T->isAbstract())
      cast<DerivedType>(T)->addAbstractTypeUser(this);
  }
}

// This function is called when one of the types in the type plane are refined
void SymbolTable::refineAbstractType(const DerivedType *OldType,
				     const Type *NewType) {
  if (OldType == NewType) return;  // Noop, don't waste time dinking around

  // Search to see if we have any values of the type oldtype.  If so, we need to
  // move them into the newtype plane...
  iterator TPI = find(OldType);
  if (TPI != end()) {
    VarMap &OldPlane = TPI->second;
    while (!OldPlane.empty()) {
      pair<const string, Value*> V = *OldPlane.begin();
      OldPlane.erase(OldPlane.begin());
      insertEntry(V.first, NewType, V.second);
    }

    // Ok, now we are not referencing the type anymore... take me off your user
    // list please!
    OldType->removeAbstractTypeUser(this);
  }

  TPI = find(Type::TypeTy);
  assert(TPI != end() &&"Type plane not in symbol table but we contain types!");

  // Loop over all of the types in the symbol table, replacing any references to
  // OldType with references to NewType.  Note that there may be multiple
  // occurances, and although we only need to remove one at a time, it's faster
  // to remove them all in one pass.
  //
  VarMap &TyPlane = TPI->second;
  for (VarMap::iterator I = TyPlane.begin(), E = TyPlane.end(); I != E; ++I)
    if (I->second == (Value*)OldType) {  // FIXME when Types aren't const.
      OldType->removeAbstractTypeUser(this);
      I->second = (Value*)NewType;  // TODO FIXME when types aren't const
      if (NewType->isAbstract())
	cast<const DerivedType>(NewType)->addAbstractTypeUser(this);
    }
}


#ifndef NDEBUG
#include "llvm/Assembly/Writer.h"
#include <algorithm>

static void DumpVal(const pair<const string, Value *> &V) {
  cout << "  '%" << V.first << "' = " << V.second << endl;
}

static void DumpPlane(const pair<const Type *, map<const string, Value *> >&P) {
  cout << "  Plane: " << P.first << endl;
  for_each(P.second.begin(), P.second.end(), DumpVal);
}

void SymbolTable::dump() const {
  cout << "Symbol table dump:\n";
  for_each(begin(), end(), DumpPlane);

  if (ParentSymTab) {
    cout << "Parent ";
    ParentSymTab->dump();
  }
}

#endif
