//===-- ValueSymbolTable.cpp - Implement the ValueSymbolTable class -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
using namespace llvm;

// Class destructor
ValueSymbolTable::~ValueSymbolTable() {
#ifndef NDEBUG   // Only do this in -g mode...
  for (iterator VI = vmap.begin(), VE = vmap.end(); VI != VE; ++VI)
    cerr << "Value still in symbol table! Type = '"
         << VI->getValue()->getType()->getDescription() << "' Name = '"
         << VI->getKeyData() << "'\n";
  assert(vmap.empty() && "Values remain in symbol table!");
#endif
}

// getUniqueName - Given a base name, return a string that is either equal to
// it (or derived from it) that does not already occur in the symbol table for
// the specified type.
//
std::string ValueSymbolTable::getUniqueName(const std::string &BaseName) const {
  std::string TryName = BaseName;

  // See if the name exists
  while (vmap.find(&TryName[0], &TryName[TryName.size()]) != vmap.end())
    // Loop until we find a free name in the symbol table.
    TryName = BaseName + utostr(++LastUnique);
  return TryName;
}


// lookup a value - Returns null on failure...
//
Value *ValueSymbolTable::lookup(const std::string &Name) const {
  const_iterator VI = vmap.find(&Name[0], &Name[Name.size()]);
  if (VI != vmap.end())                   // We found the symbol
    return VI->getValue();
  return 0;
}

Value *ValueSymbolTable::lookup(const char *NameBegin,
                                const char *NameEnd) const {
  const_iterator VI = vmap.find(NameBegin, NameEnd);
  if (VI != vmap.end())                   // We found the symbol
    return VI->getValue();
  return 0;
}

// Insert a value into the symbol table with the specified name...
//
void ValueSymbolTable::reinsertValue(Value* V) {
  assert(V->hasName() && "Can't insert nameless Value into symbol table");

  // Try inserting the name, assuming it won't conflict.
  if (vmap.insert(V->Name)) {
    //DOUT << " Inserted value: " << V->Name << ": " << *V << "\n";
    return;
  }
  
  // FIXME: this could be much more efficient.
  
  // Otherwise, there is a naming conflict.  Rename this value.
  std::string UniqueName = V->getName();
  
  V->Name->Destroy();
  
  unsigned BaseSize = UniqueName.size();
  while (1) {
    // Trim any suffix off.
    UniqueName.resize(BaseSize);
    UniqueName += utostr(++LastUnique);
    // Try insert the vmap entry with this suffix.
    ValueName &NewName = vmap.GetOrCreateValue(&UniqueName[0],
                                               &UniqueName[UniqueName.size()]);
    if (NewName.getValue() == 0) {
      // Newly inserted name.  Success!
      NewName.setValue(V);
      V->Name = &NewName;
      //DEBUG(DOUT << " Inserted value: " << UniqueName << ": " << *V << "\n");
      return;
    }
  }
}

void ValueSymbolTable::removeValueName(ValueName *V) {
  //DEBUG(DOUT << " Removing Value: " << V->getKeyData() << "\n");
  // Remove the value from the plane.
  vmap.remove(V);
}

/// createValueName - This method attempts to create a value name and insert
/// it into the symbol table with the specified name.  If it conflicts, it
/// auto-renames the name and returns that instead.
ValueName *ValueSymbolTable::createValueName(const char *NameStart,
                                             unsigned NameLen, Value *V) {
  ValueName &Entry = vmap.GetOrCreateValue(NameStart, NameStart+NameLen);
  if (Entry.getValue() == 0) {
    Entry.setValue(V);
    //DEBUG(DOUT << " Inserted value: " << Entry.getKeyData() << ": "
    //           << *V << "\n");
    return &Entry;
  }
  
  // FIXME: this could be much more efficient.
  
  // Otherwise, there is a naming conflict.  Rename this value.
  std::string UniqueName(NameStart, NameStart+NameLen);
  while (1) {
    // Trim any suffix off.
    UniqueName.resize(NameLen);
    UniqueName += utostr(++LastUnique);
    // Try insert the vmap entry with this suffix.
    ValueName &NewName = vmap.GetOrCreateValue(&UniqueName[0],
                                               &UniqueName[UniqueName.size()]);
    if (NewName.getValue() == 0) {
      // Newly inserted name.  Success!
      NewName.setValue(V);
      //DEBUG(DOUT << " Inserted value: " << UniqueName << ": " << *V << "\n");
      return &NewName;
    }
  }
}


// dump - print out the symbol table
//
void ValueSymbolTable::dump() const {
  //DOUT << "ValueSymbolTable:\n";
  for (const_iterator I = begin(), E = end(); I != E; ++I) {
    //DOUT << "  '" << I->getKeyData() << "' = ";
    I->getValue()->dump();
    //DOUT << "\n";
  }
}
