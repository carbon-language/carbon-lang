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
#include "Support/StringExtras.h"
#include <algorithm>

using namespace llvm;

#define DEBUG_SYMBOL_TABLE 0
#define DEBUG_ABSTYPE 0

SymbolTable::~SymbolTable() {
  // Drop all abstract type references in the type plane...
  for (type_iterator TI = tmap.begin(), TE = tmap.end(); TI != TE; ++TI) {
    if (TI->second->isAbstract())   // If abstract, drop the reference...
      cast<DerivedType>(TI->second)->removeAbstractTypeUser(this);
  }

 // TODO: FIXME: BIG ONE: This doesn't unreference abstract types for the 
 // planes that could still have entries!

#ifndef NDEBUG   // Only do this in -g mode...
  bool LeftoverValues = true;
  for (plane_iterator PI = pmap.begin(); PI != pmap.end(); ++PI) {
    for (value_iterator VI = PI->second.begin(); VI != PI->second.end(); ++VI)
      if (!isa<Constant>(VI->second) ) {
	std::cerr << "Value still in symbol table! Type = '"
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
  if (PI != pmap.end()) {                  // We have symbols in that plane...
    value_const_iterator VI = PI->second.find(Name);
    if (VI != PI->second.end())            // and the name is in our hash table...
      return VI->second;
  }
  return 0;
}


// lookup a type by name - returns null on failure
Type* SymbolTable::lookupType( const std::string& Name ) const {
  type_const_iterator TI = tmap.find( Name );
  if ( TI != tmap.end() )
    return TI->second;
  return 0;
}

// Remove a value
void SymbolTable::remove(Value *N) {
  assert(N->hasName() && "Value doesn't have name!");
  if (InternallyInconsistent) return;

  plane_iterator PI = pmap.find(N->getType());
  assert(PI != pmap.end() &&
         "Trying to remove a value that doesn't have a type plane yet!");
  removeEntry(PI, PI->second.find(N->getName()));
}


// removeEntry - Remove a value from the symbol table...
Value *SymbolTable::removeEntry(plane_iterator Plane, value_iterator Entry) {
  if (InternallyInconsistent) return 0;
  assert(Plane != pmap.end() &&
         Entry != Plane->second.end() && "Invalid entry to remove!");

  Value *Result = Entry->second;
  const Type *Ty = Result->getType();
#if DEBUG_SYMBOL_TABLE
  dump();
  std::cerr << " Removing Value: " << Result->getName() << "\n";
#endif

  // Remove the value from the plane...
  Plane->second.erase(Entry);

  // If the plane is empty, remove it now!
  if (Plane->second.empty()) {
    // If the plane represented an abstract type that we were interested in,
    // unlink ourselves from this plane.
    //
    if (Plane->first->isAbstract()) {
#if DEBUG_ABSTYPE
      std::cerr << "Plane Empty: Removing type: "
                << Plane->first->getDescription() << "\n";
#endif
      cast<DerivedType>(Plane->first)->removeAbstractTypeUser(this);
    }

    pmap.erase(Plane);
  }
  return Result;
}


// remove - Remove a type
void SymbolTable::remove(Type* Ty ) {
  type_iterator TI = this->type_begin();
  type_iterator TE = this->type_end();

  // Search for the entry
  while ( TI != TE && TI->second != Ty )
    ++TI;

  if ( TI != TE )
    this->removeEntry( TI );
}


// removeEntry - Remove a type from the symbol table...
Type* SymbolTable::removeEntry(type_iterator Entry) {
  if (InternallyInconsistent) return 0;
  assert( Entry != tmap.end() && "Invalid entry to remove!");

  Type* Result = Entry->second;

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

  return Result;
}


// insertEntry - Insert a value into the symbol table with the specified name.
void SymbolTable::insertEntry(const std::string &Name, const Type *VTy,
                              Value *V) {
  // Check to see if there is a naming conflict.  If so, rename this value!
  if (lookup(VTy, Name)) {
    std::string UniqueName = getUniqueName(VTy, Name);
    assert(InternallyInconsistent == false && "Infinite loop inserting value!");
    InternallyInconsistent = true;
    V->setName(UniqueName, this);
    InternallyInconsistent = false;
    return;
  }

#if DEBUG_SYMBOL_TABLE
  dump();
  std::cerr << " Inserting definition: " << Name << ": " 
            << VTy->getDescription() << "\n";
#endif

  plane_iterator PI = pmap.find(VTy);
  if (PI == pmap.end()) {      // Not in collection yet... insert dummy entry
    // Insert a new empty element.  I points to the new elements.
    PI = pmap.insert(make_pair(VTy, ValueMap())).first;
    assert(PI != pmap.end() && "How did insert fail?");

    // Check to see if the type is abstract.  If so, it might be refined in the
    // future, which would cause the plane of the old type to get merged into
    // a new type plane.
    //
    if (VTy->isAbstract()) {
      cast<DerivedType>(VTy)->addAbstractTypeUser(this);
#if DEBUG_ABSTYPE
      std::cerr << "Added abstract type value: " << VTy->getDescription()
                << "\n";
#endif
    }
  }

  PI->second.insert(make_pair(Name, V));
}


// insertEntry - Insert a value into the symbol table with the specified
// name...
//
void SymbolTable::insertEntry(const std::string& Name, Type* T) {

  // Check to see if there is a naming conflict.  If so, rename this type!
  std::string UniqueName = Name;
  if (lookupType(Name))
    UniqueName = getUniqueName(T, Name);

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


// Determine how many entries for a given type.
unsigned SymbolTable::type_size(const Type *Ty) const {
  plane_const_iterator PI = pmap.find(Ty);
  if ( PI == pmap.end() ) return 0;
  return PI->second.size();
}


// Get the name of a value
std::string SymbolTable::get_name( const Value* V ) const {
  value_const_iterator VI = this->value_begin( V->getType() );
  value_const_iterator VE = this->value_end( V->getType() );

  // Search for the entry
  while ( VI != VE && VI->second != V )
    ++VI;

  if ( VI != VE )
    return VI->first;

  return "";
}


// Get the name of a type
std::string SymbolTable::get_name( const Type* T ) const {
  if (tmap.empty()) return ""; // No types at all.

  type_const_iterator TI = tmap.begin();
  type_const_iterator TE = tmap.end();

  // Search for the entry
  while (TI != TE && TI->second != T )
    ++TI;

  if (TI != TE)  // Must have found an entry!
    return TI->first;
  return "";     // Must not have found anything...
}


// Strip the symbol table of its names.
bool SymbolTable::strip( void ) {
  bool RemovedSymbol = false;
  for (plane_iterator I = pmap.begin(); I != pmap.end();) {
    // Removing items from the plane can cause the plane itself to get deleted.
    // If this happens, make sure we incremented our plane iterator already!
    ValueMap &Plane = (I++)->second;
    value_iterator B = Plane.begin(), Bend = Plane.end();
    while (B != Bend) {   // Found nonempty type plane!
      Value *V = B->second;
      if (isa<Constant>(V)) {
	remove(V);
        RemovedSymbol = true;
      } else {
        if (!isa<GlobalValue>(V) || cast<GlobalValue>(V)->hasInternalLinkage()){
          // Set name to "", removing from symbol table!
          V->setName("", this);
          RemovedSymbol = true;
        }
      }
      ++B;
    }
  }

  for (type_iterator TI = tmap.begin(); TI != tmap.end(); ) {
    Type* T = (TI++)->second;
    remove(T);
    RemovedSymbol = true;
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
        std::cerr << "[Added] refined to abstype: " << NewType->getDescription()
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
          assert((ExistGV->isExternal() || NewGV->isExternal()) &&
                 "Two planes folded together with overlapping value names!");

          // Make sure that ExistGV is the one we want to keep!
          if (!NewGV->isExternal())
            std::swap(NewGV, ExistGV);

          // Ok we have two external global values.  Make all uses of the new
          // one use the old one...
          NewGV->uncheckedReplaceAllUsesWith(ExistGV);
          
          // Now we just convert it to an unnamed method... which won't get
          // added to our symbol table.  The problem is that if we call
          // setName on the method that it will try to remove itself from
          // the symbol table and die... because it's not in the symtab
          // right now.  To fix this, we have an internally consistent flag
          // that turns remove into a noop.  Thus the name will get null'd
          // out, but the symbol table won't get upset.
          //
          assert(InternallyInconsistent == false &&
                 "Symbol table already inconsistent!");
          InternallyInconsistent = true;

          // Remove newM from the symtab
          NewGV->setName("");
          InternallyInconsistent = false;

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
    std::cerr << "Removing type " << OldType->getDescription() << "\n";
#endif
    OldType->removeAbstractTypeUser(this);

    // Remove the plane that is no longer used
    pmap.erase(PI);
  }

  // Loop over all of the types in the symbol table, replacing any references
  // to OldType with references to NewType.  Note that there may be multiple
  // occurrences, and although we only need to remove one at a time, it's
  // faster to remove them all in one pass.
  //
  for (type_iterator I = type_begin(), E = type_end(); I != E; ++I) {
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
void SymbolTable::typeBecameConcrete(const DerivedType *AbsTy) {
  plane_iterator PI = pmap.find(AbsTy);

  // If there are any values in the symbol table of this type, then the type
  // plane is a use of the abstract type which must be dropped.
  if (PI != pmap.end())
    AbsTy->removeAbstractTypeUser(this);

  // Loop over all of the types in the symbol table, dropping any abstract
  // type user entries for AbsTy which occur because there are names for the
  // type.
  for (type_iterator TI = type_begin(), TE = type_end(); TI != TE; ++TI)
    if (TI->second == (Type*)AbsTy)   // FIXME when Types aren't const.
      AbsTy->removeAbstractTypeUser(this);
}

static void DumpVal(const std::pair<const std::string, Value *> &V) {
  std::cerr << "  '" << V.first << "' = ";
  V.second->dump();
  std::cerr << "\n";
}

static void DumpPlane(const std::pair<const Type *,
                                      std::map<const std::string, Value *> >&P){
  P.first->dump();
  std::cerr << "\n";
  for_each(P.second.begin(), P.second.end(), DumpVal);
}

static void DumpTypes(const std::pair<const std::string, Type*>& T ) {
  std::cerr << "  '" << T.first << "' = ";
  T.second->dump();
  std::cerr << "\n";
}

void SymbolTable::dump() const {
  std::cerr << "Symbol table dump:\n  Plane:";
  for_each(pmap.begin(), pmap.end(), DumpPlane);
  std::cerr << "  Types: ";
  for_each(tmap.begin(), tmap.end(), DumpTypes);
}

// vim: sw=2 ai
