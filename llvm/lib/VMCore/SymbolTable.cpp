//===-- SymbolTable.cpp - Implement the SymbolTable class -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and revised by Reid
// Spencer. It is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SymbolTable class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/SymbolTable.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
using namespace llvm;

#define DEBUG_SYMBOL_TABLE 0
#define DEBUG_ABSTYPE 0

SymbolTable::~SymbolTable() {
 // TODO: FIXME: BIG ONE: This doesn't unreference abstract types for the
 // planes that could still have entries!

#ifndef NDEBUG   // Only do this in -g mode...
  bool LeftoverValues = true;
  for (plane_iterator PI = pmap.begin(); PI != pmap.end(); ++PI) {
    for (value_iterator VI = PI->second.begin(); VI != PI->second.end(); ++VI)
      if (!isa<Constant>(VI->second) ) {
        DOUT << "Value still in symbol table! Type = '"
             << PI->first->getDescription() << "' Name = '"
             << VI->first << "'\n";
        LeftoverValues = false;
      }
  }

  assert(LeftoverValues && "Values remain in symbol table!");
#endif
}

// getUniqueName - Given a base name, return a string that is either equal to
// it (or derived from it) that does not already occur in the symbol table for
// the specified type.
//
std::string SymbolTable::getUniqueName(const Type *Ty,
                                       const std::string &BaseName) const {
  // Find the plane
  plane_const_iterator PI = pmap.find(Ty);
  if (PI == pmap.end()) return BaseName;

  std::string TryName = BaseName;
  const ValueMap& vmap = PI->second;
  value_const_iterator End = vmap.end();

  // See if the name exists
  while (vmap.find(TryName) != End)            // Loop until we find a free
    TryName = BaseName + utostr(++LastUnique); // name in the symbol table
  return TryName;
}


// lookup a value - Returns null on failure...
Value *SymbolTable::lookup(const Type *Ty, const std::string &Name) const {
  plane_const_iterator PI = pmap.find(Ty);
  if (PI != pmap.end()) {                // We have symbols in that plane.
    value_const_iterator VI = PI->second.find(Name);
    if (VI != PI->second.end())          // and the name is in our hash table.
      return VI->second;
  }
  return 0;
}


/// changeName - Given a value with a non-empty name, remove its existing entry
/// from the symbol table and insert a new one for Name.  This is equivalent to
/// doing "remove(V), V->Name = Name, insert(V)", but is faster, and will not
/// temporarily remove the symbol table plane if V is the last value in the
/// symtab with that name (which could invalidate iterators to that plane).
void SymbolTable::changeName(Value *V, const std::string &name) {
  assert(!V->getName().empty() && !name.empty() && V->getName() != name &&
         "Illegal use of this method!");

  plane_iterator PI = pmap.find(V->getType());
  assert(PI != pmap.end() && "Value doesn't have an entry in this table?");
  ValueMap &VM = PI->second;

  value_iterator VI = VM.find(V->getName());
  assert(VI != VM.end() && "Value does have an entry in this table?");

  // Remove the old entry.
  VM.erase(VI);

  // See if we can insert the new name.
  VI = VM.lower_bound(name);

  // Is there a naming conflict?
  if (VI != VM.end() && VI->first == name) {
    V->Name = getUniqueName(V->getType(), name);
    VM.insert(make_pair(V->Name, V));
  } else {
    V->Name = name;
    VM.insert(VI, make_pair(name, V));
  }
}

// Remove a value
void SymbolTable::remove(Value *N) {
  assert(N->hasName() && "Value doesn't have name!");

  plane_iterator PI = pmap.find(N->getType());
  assert(PI != pmap.end() &&
         "Trying to remove a value that doesn't have a type plane yet!");
  ValueMap &VM = PI->second;
  value_iterator Entry = VM.find(N->getName());
  assert(Entry != VM.end() && "Invalid entry to remove!");

#if DEBUG_SYMBOL_TABLE
  dump();
  DOUT << " Removing Value: " << Entry->second->getName() << "\n";
#endif

  // Remove the value from the plane...
  VM.erase(Entry);

  // If the plane is empty, remove it now!
  if (VM.empty()) {
    // If the plane represented an abstract type that we were interested in,
    // unlink ourselves from this plane.
    //
    if (N->getType()->isAbstract()) {
#if DEBUG_ABSTYPE
      DOUT << "Plane Empty: Removing type: "
           << N->getType()->getDescription() << "\n";
#endif
      cast<DerivedType>(N->getType())->removeAbstractTypeUser(this);
    }

    pmap.erase(PI);
  }
}


// insertEntry - Insert a value into the symbol table with the specified name.
void SymbolTable::insertEntry(const std::string &Name, const Type *VTy,
                              Value *V) {
  plane_iterator PI = pmap.find(VTy);   // Plane iterator
  value_iterator VI;                    // Actual value iterator
  ValueMap *VM;                         // The plane we care about.

#if DEBUG_SYMBOL_TABLE
  dump();
  DOUT << " Inserting definition: " << Name << ": "
       << VTy->getDescription() << "\n";
#endif

  if (PI == pmap.end()) {      // Not in collection yet... insert dummy entry
    // Insert a new empty element.  I points to the new elements.
    VM = &pmap.insert(make_pair(VTy, ValueMap())).first->second;
    VI = VM->end();

    // Check to see if the type is abstract.  If so, it might be refined in the
    // future, which would cause the plane of the old type to get merged into
    // a new type plane.
    //
    if (VTy->isAbstract()) {
      cast<DerivedType>(VTy)->addAbstractTypeUser(this);
#if DEBUG_ABSTYPE
      DOUT << "Added abstract type value: " << VTy->getDescription()
           << "\n";
#endif
    }

  } else {
    // Check to see if there is a naming conflict.  If so, rename this value!
    VM = &PI->second;
    VI = VM->lower_bound(Name);
    if (VI != VM->end() && VI->first == Name) {
      V->Name = getUniqueName(VTy, Name);
      VM->insert(make_pair(V->Name, V));
      return;
    }
  }

  VM->insert(VI, make_pair(Name, V));
}



// Strip the symbol table of its names.
bool SymbolTable::strip() {
  bool RemovedSymbol = false;
  for (plane_iterator I = pmap.begin(); I != pmap.end();) {
    // Removing items from the plane can cause the plane itself to get deleted.
    // If this happens, make sure we incremented our plane iterator already!
    ValueMap &Plane = (I++)->second;
    value_iterator B = Plane.begin(), Bend = Plane.end();
    while (B != Bend) {   // Found nonempty type plane!
      Value *V = B->second;
      ++B;
      if (!isa<GlobalValue>(V) || cast<GlobalValue>(V)->hasInternalLinkage()) {
        // Set name to "", removing from symbol table!
        V->setName("");
        RemovedSymbol = true;
      }
    }
  }

  return RemovedSymbol;
}


// This function is called when one of the types in the type plane are refined
void SymbolTable::refineAbstractType(const DerivedType *OldType,
                                     const Type *NewType) {

  // Search to see if we have any values of the type Oldtype.  If so, we need to
  // move them into the newtype plane...
  plane_iterator PI = pmap.find(OldType);
  if (PI != pmap.end()) {
    // Get a handle to the new type plane...
    plane_iterator NewTypeIt = pmap.find(NewType);
    if (NewTypeIt == pmap.end()) {      // If no plane exists, add one
      NewTypeIt = pmap.insert(make_pair(NewType, ValueMap())).first;

      if (NewType->isAbstract()) {
        cast<DerivedType>(NewType)->addAbstractTypeUser(this);
#if DEBUG_ABSTYPE
        DOUT << "[Added] refined to abstype: " << NewType->getDescription()
             << "\n";
#endif
      }
    }

    ValueMap &NewPlane = NewTypeIt->second;
    ValueMap &OldPlane = PI->second;
    while (!OldPlane.empty()) {
      std::pair<const std::string, Value*> V = *OldPlane.begin();

      // Check to see if there is already a value in the symbol table that this
      // would collide with.
      value_iterator VI = NewPlane.find(V.first);
      if (VI != NewPlane.end() && VI->second == V.second) {
        // No action

      } else if (VI != NewPlane.end()) {
        // The only thing we are allowing for now is two external global values
        // folded into one.
        //
        GlobalValue *ExistGV = dyn_cast<GlobalValue>(VI->second);
        GlobalValue *NewGV = dyn_cast<GlobalValue>(V.second);

        if (ExistGV && NewGV) {
          assert((ExistGV->isDeclaration() || NewGV->isDeclaration()) &&
                 "Two planes folded together with overlapping value names!");

          // Make sure that ExistGV is the one we want to keep!
          if (!NewGV->isDeclaration())
            std::swap(NewGV, ExistGV);

          // Ok we have two external global values.  Make all uses of the new
          // one use the old one...
          NewGV->uncheckedReplaceAllUsesWith(ExistGV);

          // Update NewGV's name, we're about the remove it from the symbol
          // table.
          NewGV->Name = "";

          // Now we can remove this global from the module entirely...
          Module *M = NewGV->getParent();
          if (Function *F = dyn_cast<Function>(NewGV))
            M->getFunctionList().remove(F);
          else
            M->getGlobalList().remove(cast<GlobalVariable>(NewGV));
          delete NewGV;
        } else {
          // If they are not global values, they must be just random values who
          // happen to conflict now that types have been resolved.  If this is
          // the case, reinsert the value into the new plane, allowing it to get
          // renamed.
          assert(V.second->getType() == NewType &&"Type resolution is broken!");
          insert(V.second);
        }
      } else {
        insertEntry(V.first, NewType, V.second);
      }
      // Remove the item from the old type plane
      OldPlane.erase(OldPlane.begin());
    }

    // Ok, now we are not referencing the type anymore... take me off your user
    // list please!
#if DEBUG_ABSTYPE
    DOUT << "Removing type " << OldType->getDescription() << "\n";
#endif
    OldType->removeAbstractTypeUser(this);

    // Remove the plane that is no longer used
    pmap.erase(PI);
  }
}


// Handle situation where type becomes Concreate from Abstract
void SymbolTable::typeBecameConcrete(const DerivedType *AbsTy) {
  plane_iterator PI = pmap.find(AbsTy);

  // If there are any values in the symbol table of this type, then the type
  // plane is a use of the abstract type which must be dropped.
  if (PI != pmap.end())
    AbsTy->removeAbstractTypeUser(this);
}

static void DumpVal(const std::pair<const std::string, Value *> &V) {
  DOUT << "  '" << V.first << "' = ";
  V.second->dump();
  DOUT << "\n";
}

static void DumpPlane(const std::pair<const Type *,
                                      std::map<const std::string, Value *> >&P){
  P.first->dump();
  DOUT << "\n";
  for_each(P.second.begin(), P.second.end(), DumpVal);
}

void SymbolTable::dump() const {
  DOUT << "Symbol table dump:\n  Plane:";
  for_each(pmap.begin(), pmap.end(), DumpPlane);
}

// vim: sw=2 ai
