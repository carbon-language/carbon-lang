//===-- SymbolTable.cpp - Implement the SymbolTable class -------------------=//
//
// This file implements the SymbolTable class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/SymbolTable.h"
#include "llvm/InstrTypes.h"
#include "llvm/Support/StringExtras.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Method.h"

#define DEBUG_SYMBOL_TABLE 0
#define DEBUG_ABSTYPE 0

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

  iterator I = find(N->getType());
  removeEntry(I, I->second.find(N->getName()));
}

// removeEntry - Remove a value from the symbol table...
//
Value *SymbolTable::removeEntry(iterator Plane, type_iterator Entry) {
  assert(Plane != super::end() &&
         Entry != Plane->second.end() && "Invalid entry to remove!");

  Value *Result = Entry->second;
  const Type *Ty = Result->getType();
#if DEBUG_SYMBOL_TABLE
  cerr << this << " Removing Value: " << Result->getName() << endl;
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
      cerr << "Plane Empty: Removing type: " << Plane->first->getDescription()
           << endl;
#endif
      cast<DerivedType>(Plane->first)->removeAbstractTypeUser(this);
    }

    erase(Plane);
  }

  // If we are removing an abstract type, remove the symbol table from it's use
  // list...
  if (Ty == Type::TypeTy) {
    const Type *T = cast<const Type>(Result);
    if (T->isAbstract()) {
#if DEBUG_ABSTYPE
      cerr << "Removing abs type from symtab" << T->getDescription() << endl;
#endif
      cast<DerivedType>(T)->removeAbstractTypeUser(this);
    }
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
    if (VTy->isAbstract()) {
      cast<DerivedType>(VTy)->addAbstractTypeUser(this);
#if DEBUG_ABSTYPE
      cerr << "Added abstract type value: " << VTy->getDescription() << endl;
#endif
    }
  }

  I->second.insert(make_pair(Name, V));

  // If we are adding an abstract type, add the symbol table to it's use list.
  if (VTy == Type::TypeTy) {
    const Type *T = cast<const Type>(V);
    if (T->isAbstract()) {
      cast<DerivedType>(T)->addAbstractTypeUser(this);
#if DEBUG_ABSTYPE
      cerr << "Added abstract type to ST: " << T->getDescription() << endl;
#endif
    }
  }
}

// This function is called when one of the types in the type plane are refined
void SymbolTable::refineAbstractType(const DerivedType *OldType,
				     const Type *NewType) {
  if (OldType == NewType) return;  // Noop, don't waste time dinking around

  // Get a handle to the new type plane...
  iterator NewTypeIt = find(NewType);
  if (NewTypeIt == super::end()) {      // If no plane exists, add one
    NewTypeIt = super::insert(make_pair(NewType, VarMap())).first;

    if (NewType->isAbstract()) {
      cast<DerivedType>(NewType)->addAbstractTypeUser(this);
#if DEBUG_ABSTYPE
      cerr << "refined to abstype: " << NewType->getDescription() <<endl;
#endif
    }
  }

  VarMap &NewPlane = NewTypeIt->second;

  // Search to see if we have any values of the type oldtype.  If so, we need to
  // move them into the newtype plane...
  iterator TPI = find(OldType);
  if (TPI != end()) {
    VarMap &OldPlane = TPI->second;
    while (!OldPlane.empty()) {
      pair<const string, Value*> V = *OldPlane.begin();

      // Check to see if there is already a value in the symbol table that this
      // would collide with.
      type_iterator TI = NewPlane.find(V.first);
      if (TI != NewPlane.end() && TI->second == V.second) {
        // No action

      } else if (TI != NewPlane.end()) {
        // The only thing we are allowing for now is two method prototypes being
        // folded into one.
        //
        if (Method *ExistM = dyn_cast<Method>(TI->second))
          if (Method *NewM = dyn_cast<Method>(V.second))
            if (ExistM->isExternal() && NewM->isExternal()) {
              // Ok we have two external methods.  Make all uses of the new one
              // use the old one...
              //
              NewM->replaceAllUsesWith(ExistM);

              // Now we just convert it to an unnamed method... which won't get
              // added to our symbol table.  The problem is that if we call
              // setName on the method that it will try to remove itself from
              // the symbol table and die... because it's not in the symtab
              // right now.  To fix this, we temporarily insert it (by setting
              // TI's entry to the old value.  Then after it is removed, we
              // restore ExistM into the symbol table.
              //
              if (NewM->getType() == NewType) {
                TI->second = NewM;     // Add newM to the symtab

                // Remove newM from the symtab
                NewM->setName("");

                // Readd ExistM to the symbol table....
                NewPlane.insert(make_pair(V.first, ExistM));
              } else {
                NewM->setName("");
              }
              continue;
            }
        assert(0 && "Two ploanes folded together with overlapping "
               "value names!");
      } else {
        insertEntry(V.first, NewType, V.second);

      }
      // Remove the item from the old type plane
      OldPlane.erase(OldPlane.begin());
    }

    // Ok, now we are not referencing the type anymore... take me off your user
    // list please!
#if DEBUG_ABSTYPE
    cerr << "Removing type " << OldType->getDescription() << endl;
#endif
    OldType->removeAbstractTypeUser(this);

    // Remove the plane that is no longer used
    erase(TPI);
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
#if DEBUG_ABSTYPE
      cerr << "Removing type " << OldType->getDescription() << endl;
#endif
      OldType->removeAbstractTypeUser(this);

      I->second = (Value*)NewType;  // TODO FIXME when types aren't const
      if (NewType->isAbstract()) {
#if DEBUG_ABSTYPE
        cerr << "Added type " << NewType->getDescription() << endl;
#endif
	cast<const DerivedType>(NewType)->addAbstractTypeUser(this);
      }
    }
}


#ifndef NDEBUG
#include "llvm/Assembly/Writer.h"
#include <algorithm>

static void DumpVal(const pair<const string, Value *> &V) {
  cout << "  '" << V.first << "' = " << V.second << endl;
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
