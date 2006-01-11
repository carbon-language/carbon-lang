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

#include "llvm/GlobalValue.h"
#include "llvm/Type.h"
#include "llvm/ValueSymbolTable.h"
#include "llvm/ADT/StringExtras.h"
#include <algorithm>
#include <iostream>

using namespace llvm;

#define DEBUG_SYMBOL_TABLE 0
#define DEBUG_ABSTYPE 0

// Class destructor
ValueSymbolTable::~ValueSymbolTable() {
#ifndef NDEBUG   // Only do this in -g mode...
  bool LeftoverValues = true;
  for (iterator VI = vmap.begin(), VE = vmap.end(); VI != VE; ++VI)
    if (!isa<Constant>(VI->second) ) {
      std::cerr << "Value still in symbol table! Type = '"
                << VI->second->getType()->getDescription() << "' Name = '"
                << VI->first << "'\n";
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

// Strip the symbol table of its names.
//
bool ValueSymbolTable::strip() {
  bool RemovedSymbol = false;
  for (iterator VI = vmap.begin(), VE = vmap.end(); VI != VE; ) {
    Value *V = VI->second;
    ++VI;
    if (!isa<GlobalValue>(V) || cast<GlobalValue>(V)->hasInternalLinkage()) {
      // Set name to "", removing from symbol table!
      V->setName("");
      RemovedSymbol = true;
    }
  }
  return RemovedSymbol;
}

// Insert a value into the symbol table with the specified name...
//
void ValueSymbolTable::insert(Value* V) {
  assert(V && "Can't insert null Value into symbol table!");
  assert(V->hasName() && "Can't insert nameless Value into symbol table");

  // Check to see if there is a naming conflict.  If so, rename this type!
  std::string UniqueName = getUniqueName(V->getName());

#if DEBUG_SYMBOL_TABLE
  dump();
  std::cerr << " Inserting value: " << UniqueName << ": " << V->dump() << "\n";
#endif

  // Insert the vmap entry
  vmap.insert(make_pair(UniqueName, V));
}

// Remove a value
bool ValueSymbolTable::erase(Value *V) {
  assert(V->hasName() && "Value doesn't have name!");
  iterator Entry = vmap.find(V->getName());
  if (Entry == vmap.end())
    return false;

#if DEBUG_SYMBOL_TABLE
  dump();
  std::cerr << " Removing Value: " << Entry->second->getName() << "\n";
#endif

  // Remove the value from the plane...
  vmap.erase(Entry);
  return true;
}


// rename - Given a value with a non-empty name, remove its existing entry
// from the symbol table and insert a new one for Name.  This is equivalent to
// doing "remove(V), V->Name = Name, insert(V)", 
//
bool ValueSymbolTable::rename(Value *V, const std::string &name) {
  assert(V && "Can't rename a null Value");
  assert(V->hasName() && "Can't rename a nameless Value");
  assert(!V->getName().empty() && "Can't rename an Value with null name");
  assert(V->getName() != name && "Can't rename a Value with same name");
  assert(!name.empty() && "Can't rename a named Value with a null name");

  // Find the name
  iterator VI = vmap.find(V->getName());

  // If we didn't find it, we're done
  if (VI == vmap.end())
    return false;

  // Remove the old entry.
  vmap.erase(VI);

  // See if we can insert the new name.
  VI = vmap.lower_bound(name);

  // Is there a naming conflict?
  if (VI != vmap.end() && VI->first == name) {
    V->Name = getUniqueName( name);
    vmap.insert(make_pair(V->Name, V));
  } else {
    V->Name = name;
    vmap.insert(VI, make_pair(name, V));
  }

  return true;
}

// DumpVal - a std::for_each function for dumping a value
//
static void DumpVal(const std::pair<const std::string, Value *> &V) {
  std::cerr << "  '" << V.first << "' = ";
  V.second->dump();
  std::cerr << "\n";
}

// dump - print out the symbol table
//
void ValueSymbolTable::dump() const {
  std::cerr << "ValueSymbolTable:\n";
  for_each(vmap.begin(), vmap.end(), DumpVal);
}
