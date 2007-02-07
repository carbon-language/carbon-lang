//===-- ValueSymbolTable.cpp - Implement the ValueSymbolTable class -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group.  It is distributed under 
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ValueSymbolTable class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "valuesymtab"
#include "llvm/GlobalValue.h"
#include "llvm/Type.h"
#include "llvm/ValueSymbolTable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
using namespace llvm;

// Class destructor
ValueSymbolTable::~ValueSymbolTable() {
#ifndef NDEBUG   // Only do this in -g mode...
  bool LeftoverValues = true;
  for (iterator VI = vmap.begin(), VE = vmap.end(); VI != VE; ++VI)
    if (!isa<Constant>(VI->second) ) {
      DEBUG(DOUT << "Value still in symbol table! Type = '"
           << VI->second->getType()->getDescription() << "' Name = '"
           << VI->first << "'\n");
      LeftoverValues = false;
    }
  assert(LeftoverValues && "Values remain in symbol table!");
#endif
}

// getUniqueName - Given a base name, return a string that is either equal to
// it (or derived from it) that does not already occur in the symbol table for
// the specified type.
//
std::string ValueSymbolTable::getUniqueName(const std::string &BaseName) const {
  std::string TryName = BaseName;
  const_iterator End = vmap.end();

  // See if the name exists
  while (vmap.find(TryName) != End)            // Loop until we find a free
    TryName = BaseName + utostr(++LastUnique); // name in the symbol table
  return TryName;
}


// lookup a value - Returns null on failure...
//
Value *ValueSymbolTable::lookup(const std::string &Name) const {
  const_iterator VI = vmap.find(Name);
  if (VI != vmap.end())                   // We found the symbol
    return const_cast<Value*>(VI->second);
  return 0;
}

// Insert a value into the symbol table with the specified name...
//
void ValueSymbolTable::insert(Value* V) {
  assert(V && "Can't insert null Value into symbol table!");
  assert(V->hasName() && "Can't insert nameless Value into symbol table");

  // Try inserting the name, assuming it won't conflict.
  if (vmap.insert(make_pair(V->Name, V)).second) {
    DOUT << " Inserted value: " << V->Name << ": " << *V << "\n";
    return;
  }
  
  // Otherwise, there is a naming conflict.  Rename this value.
  std::string UniqueName = V->getName();
  unsigned BaseSize = UniqueName.size();
  do {
    // Trim any suffix off.
    UniqueName.resize(BaseSize);
    UniqueName += utostr(++LastUnique);
  } while (!vmap.insert(make_pair(UniqueName, V)).second);

  DEBUG(DOUT << " Inserting value: " << UniqueName << ": " << *V << "\n");

  // Insert the vmap entry
  V->Name = UniqueName;
}

// Remove a value
void ValueSymbolTable::remove(Value *V) {
  assert(V->hasName() && "Value doesn't have name!");
  iterator Entry = vmap.find(V->getName());
  assert(Entry != vmap.end() && "Entry was not in the symtab!");

  DEBUG(DOUT << " Removing Value: " << Entry->second->getName() << "\n");

  // Remove the value from the plane...
  vmap.erase(Entry);
}

// DumpVal - a std::for_each function for dumping a value
//
static void DumpVal(const std::pair<const std::string, Value *> &V) {
  DOUT << "  '" << V.first << "' = ";
  V.second->dump();
  DOUT << "\n";
}

// dump - print out the symbol table
//
void ValueSymbolTable::dump() const {
  DOUT << "ValueSymbolTable:\n";
  for_each(vmap.begin(), vmap.end(), DumpVal);
}
