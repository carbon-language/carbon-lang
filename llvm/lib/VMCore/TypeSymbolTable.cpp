//===-- TypeSymbolTable.cpp - Implement the TypeSymbolTable class ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and revised by Reid
// Spencer. It is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TypeSymbolTable class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/TypeSymbolTable.h"
#include "llvm/DerivedTypes.h"
#include "llvm/ADT/StringExtras.h"
#include <iostream>

using namespace llvm;

#define DEBUG_SYMBOL_TABLE 0
#define DEBUG_ABSTYPE 0

TypeSymbolTable::~TypeSymbolTable() {
  // Drop all abstract type references in the type plane...
  for (iterator TI = tmap.begin(), TE = tmap.end(); TI != TE; ++TI) {
    if (TI->second->isAbstract())   // If abstract, drop the reference...
      cast<DerivedType>(TI->second)->removeAbstractTypeUser(this);
  }
}

std::string TypeSymbolTable::getUniqueName(const std::string &BaseName) const {
  std::string TryName = BaseName;
  const_iterator End = tmap.end();

  // See if the name exists
  while (tmap.find(TryName) != End)            // Loop until we find a free
    TryName = BaseName + utostr(++LastUnique); // name in the symbol table
  return TryName;
}

// lookup a type by name - returns null on failure
Type* TypeSymbolTable::lookup(const std::string& Name) const {
  const_iterator TI = tmap.find(Name);
  if (TI != tmap.end())
    return const_cast<Type*>(TI->second);
  return 0;
}

// Erase a specific type from the symbol table
bool TypeSymbolTable::erase(Type *N) {
  for (iterator TI = tmap.begin(), TE = tmap.end(); TI != TE; ++TI) {
    if (TI->second == N) {
      this->erase(TI);
      return true;
    }
  }
  return false;
}

// remove - Remove a type from the symbol table...
Type* TypeSymbolTable::erase(iterator Entry) {
  assert(Entry != tmap.end() && "Invalid entry to remove!");

  const Type* Result = Entry->second;

#if DEBUG_SYMBOL_TABLE
  dump();
  std::cerr << " Removing Value: " << Result->getName() << "\n";
#endif

  tmap.erase(Entry);

  // If we are removing an abstract type, remove the symbol table from it's use
  // list...
  if (Result->isAbstract()) {
#if DEBUG_ABSTYPE
    std::cerr << "Removing abstract type from symtab" << Result->getDescription()<<"\n";
#endif
    cast<DerivedType>(Result)->removeAbstractTypeUser(this);
  }

  return const_cast<Type*>(Result);
}


// insert - Insert a type into the symbol table with the specified name...
void TypeSymbolTable::insert(const std::string& Name, const Type* T) {
  assert(T && "Can't insert null type into symbol table!");

  // Check to see if there is a naming conflict.  If so, rename this type!
  std::string UniqueName = Name;
  if (lookup(Name))
    UniqueName = getUniqueName(Name);

#if DEBUG_SYMBOL_TABLE
  dump();
  std::cerr << " Inserting type: " << UniqueName << ": "
            << T->getDescription() << "\n";
#endif

  // Insert the tmap entry
  tmap.insert(make_pair(UniqueName, T));

  // If we are adding an abstract type, add the symbol table to it's use list.
  if (T->isAbstract()) {
    cast<DerivedType>(T)->addAbstractTypeUser(this);
#if DEBUG_ABSTYPE
    std::cerr << "Added abstract type to ST: " << T->getDescription() << "\n";
#endif
  }
}

// Strip the symbol table of its names.
bool TypeSymbolTable::strip() {
  bool RemovedSymbol = false;
  for (iterator TI = tmap.begin(); TI != tmap.end(); ) {
    erase(TI++);
    RemovedSymbol = true;
  }

  return RemovedSymbol;
}

/// rename - Given a value with a non-empty name, remove its existing entry
/// from the symbol table and insert a new one for Name.  This is equivalent to
/// doing "remove(V), V->Name = Name, insert(V)", but is faster, and will not
/// temporarily remove the symbol table plane if V is the last value in the
/// symtab with that name (which could invalidate iterators to that plane).
bool TypeSymbolTable::rename(Type *T, const std::string &name) {
  for (iterator TI = tmap.begin(), TE = tmap.end(); TI != TE; ++TI) {
    if (TI->second == T) {
      // Remove the old entry.
      tmap.erase(TI);
      // Add the new entry.
      this->insert(name,T);
      return true;
    }
  }
  return false;
}

// This function is called when one of the types in the type plane are refined
void TypeSymbolTable::refineAbstractType(const DerivedType *OldType,
                                         const Type *NewType) {

  // Loop over all of the types in the symbol table, replacing any references
  // to OldType with references to NewType.  Note that there may be multiple
  // occurrences, and although we only need to remove one at a time, it's
  // faster to remove them all in one pass.
  //
  for (iterator I = begin(), E = end(); I != E; ++I) {
    if (I->second == (Type*)OldType) {  // FIXME when Types aren't const.
#if DEBUG_ABSTYPE
      std::cerr << "Removing type " << OldType->getDescription() << "\n";
#endif
      OldType->removeAbstractTypeUser(this);

      I->second = (Type*)NewType;  // TODO FIXME when types aren't const
      if (NewType->isAbstract()) {
#if DEBUG_ABSTYPE
        std::cerr << "Added type " << NewType->getDescription() << "\n";
#endif
        cast<DerivedType>(NewType)->addAbstractTypeUser(this);
      }
    }
  }
}


// Handle situation where type becomes Concreate from Abstract
void TypeSymbolTable::typeBecameConcrete(const DerivedType *AbsTy) {
  // Loop over all of the types in the symbol table, dropping any abstract
  // type user entries for AbsTy which occur because there are names for the
  // type.
  for (iterator TI = begin(), TE = end(); TI != TE; ++TI)
    if (TI->second == const_cast<Type*>(static_cast<const Type*>(AbsTy)))
      AbsTy->removeAbstractTypeUser(this);
}

static void DumpTypes(const std::pair<const std::string, const Type*>& T ) {
  std::cerr << "  '" << T.first << "' = ";
  T.second->dump();
  std::cerr << "\n";
}

void TypeSymbolTable::dump() const {
  std::cerr << "TypeSymbolPlane: ";
  for_each(tmap.begin(), tmap.end(), DumpTypes);
}

// vim: sw=2 ai
