//===-- SymbolTable.cpp - Implement the SymbolTable class -------------------=//
//
// This file implements the SymbolTable class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/SymbolTable.h"
#include "llvm/InstrTypes.h"
#include "llvm/Support/StringExtras.h"
#ifndef NDEBUG
#include "llvm/BasicBlock.h"   // Required for assertions to work.
#include "llvm/Type.h"
#endif

SymbolTable::~SymbolTable() {
#ifndef NDEBUG   // Only do this in -g mode...
  bool Good = true;
  for (iterator i = begin(); i != end(); ++i) {
    if (i->second.begin() != i->second.end()) {
      for (type_iterator I = i->second.begin(); I != i->second.end(); ++I)
        cerr << "Value still in symbol table! Type = " << i->first->getName() 
             << "  Name = " << I->first << endl;
      Good = false;
    }
  }
  assert(Good && "Values remain in symbol table!");
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
#if DEBUG_SYMBOL_TABLE
  cerr << this << " Removing Value: " << Result->getName() << endl;
#endif

  find(Result->getType())->second.erase(It);

  return Result;
}

void SymbolTable::insert(Value *N) {
  assert(N->hasName() && "Value must be named to go into symbol table!");

  // TODO: The typeverifier should catch this when its implemented
  if (lookup(N->getType(), N->getName())) {
    cerr << "SymbolTable WARNING: Name already in symbol table: '" 
         << N->getName() << "'\n";
    abort();  // TODO: REMOVE THIS
  }

#if DEBUG_SYMBOL_TABLE
  cerr << this << " Inserting definition: " << N->getName() << ": " 
       << N->getType()->getName() << endl;
#endif

  iterator I = find(N->getType());
  if (I == end()) {      // Not in collection yet... insert dummy entry
    (*this)[N->getType()] = VarMap();
    I = find(N->getType());
    assert(I != end() && "How did insert fail?");
  }

  I->second.insert(make_pair(N->getName(), N));
}

