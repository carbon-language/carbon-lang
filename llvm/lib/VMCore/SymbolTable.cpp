//===-- SymbolTable.cpp - Implement the SymbolTable class -------------------=//
//
// This file implements the SymbolTable class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/SymbolTable.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "Support/StringExtras.h"
#include <algorithm>

#define DEBUG_SYMBOL_TABLE 0
#define DEBUG_ABSTYPE 0

SymbolTable::~SymbolTable() {
  // Drop all abstract type references in the type plane...
  iterator TyPlane = find(Type::TypeTy);
  if (TyPlane != end()) {
    VarMap &TyP = TyPlane->second;
    for (VarMap::iterator I = TyP.begin(), E = TyP.end(); I != E; ++I) {
      const Type *Ty = cast<Type>(I->second);
      if (Ty->isAbstract())   // If abstract, drop the reference...
	cast<DerivedType>(Ty)->removeAbstractTypeUser(this);
    }
  }

 // TODO: FIXME: BIG ONE: This doesn't unreference abstract types for the planes
 // that could still have entries!

#ifndef NDEBUG   // Only do this in -g mode...
  bool LeftoverValues = true;
  for (iterator i = begin(); i != end(); ++i) {
    for (type_iterator I = i->second.begin(); I != i->second.end(); ++I)
      if (!isa<Constant>(I->second) && !isa<Type>(I->second)) {
	std::cerr << "Value still in symbol table! Type = '"
                  << i->first->getDescription() << "' Name = '"
                  << I->first << "'\n";
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
                                       const std::string &BaseName) {
  iterator I = find(Ty);
  if (I == end()) return BaseName;

  std::string TryName = BaseName;
  unsigned Counter = 0;
  type_iterator End = I->second.end();

  while (I->second.find(TryName) != End)     // Loop until we find unoccupied
    TryName = BaseName + utostr(++Counter);  // Name in the symbol table
  return TryName;
}



// lookup - Returns null on failure...
Value *SymbolTable::lookup(const Type *Ty, const std::string &Name) {
  iterator I = find(Ty);
  if (I != end()) {                      // We have symbols in that plane...
    type_iterator J = I->second.find(Name);
    if (J != I->second.end())            // and the name is in our hash table...
      return J->second;
  }

  return 0;
}

void SymbolTable::remove(Value *N) {
  assert(N->hasName() && "Value doesn't have name!");
  if (InternallyInconsistent) return;

  iterator I = find(N->getType());
  assert(I != end() &&
         "Trying to remove a type that doesn't have a plane yet!");
  removeEntry(I, I->second.find(N->getName()));
}

// removeEntry - Remove a value from the symbol table...
//
Value *SymbolTable::removeEntry(iterator Plane, type_iterator Entry) {
  if (InternallyInconsistent) return 0;
  assert(Plane != super::end() &&
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

    erase(Plane);
  }

  // If we are removing an abstract type, remove the symbol table from it's use
  // list...
  if (Ty == Type::TypeTy) {
    const Type *T = cast<Type>(Result);
    if (T->isAbstract()) {
#if DEBUG_ABSTYPE
      std::cerr << "Removing abs type from symtab" << T->getDescription()<<"\n";
#endif
      cast<DerivedType>(T)->removeAbstractTypeUser(this);
    }
  }

  return Result;
}

// insertEntry - Insert a value into the symbol table with the specified
// name...
//
void SymbolTable::insertEntry(const std::string &Name, const Type *VTy,
                              Value *V) {

  // Check to see if there is a naming conflict.  If so, rename this value!
  if (lookup(VTy, Name)) {
    std::string UniqueName = getUniqueName(VTy, Name);
    assert(InternallyInconsistent == false && "Infinite loop inserting entry!");
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
      std::cerr << "Added abstract type value: " << VTy->getDescription()
                << "\n";
#endif
    }
  }

  I->second.insert(make_pair(Name, V));

  // If we are adding an abstract type, add the symbol table to it's use list.
  if (VTy == Type::TypeTy) {
    const Type *T = cast<Type>(V);
    if (T->isAbstract()) {
      cast<DerivedType>(T)->addAbstractTypeUser(this);
#if DEBUG_ABSTYPE
      std::cerr << "Added abstract type to ST: " << T->getDescription() << "\n";
#endif
    }
  }
}

// This function is called when one of the types in the type plane are refined
void SymbolTable::refineAbstractType(const DerivedType *OldType,
				     const Type *NewType) {
  // Search to see if we have any values of the type oldtype.  If so, we need to
  // move them into the newtype plane...
  iterator TPI = find(OldType);
  if (OldType != NewType && TPI != end()) {
    // Get a handle to the new type plane...
    iterator NewTypeIt = find(NewType);
    if (NewTypeIt == super::end()) {      // If no plane exists, add one
      NewTypeIt = super::insert(make_pair(NewType, VarMap())).first;
      
      if (NewType->isAbstract()) {
        cast<DerivedType>(NewType)->addAbstractTypeUser(this);
#if DEBUG_ABSTYPE
        std::cerr << "[Added] refined to abstype: " << NewType->getDescription()
                  << "\n";
#endif
      }
    }

    VarMap &NewPlane = NewTypeIt->second;
    VarMap &OldPlane = TPI->second;
    while (!OldPlane.empty()) {
      std::pair<const std::string, Value*> V = *OldPlane.begin();

      // Check to see if there is already a value in the symbol table that this
      // would collide with.
      type_iterator TI = NewPlane.find(V.first);
      if (TI != NewPlane.end() && TI->second == V.second) {
        // No action

      } else if (TI != NewPlane.end()) {
        // The only thing we are allowing for now is two external global values
        // folded into one.
        //
        GlobalValue *ExistGV = dyn_cast<GlobalValue>(TI->second);
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
    erase(TPI);
  } else if (TPI != end()) {
    assert(OldType == NewType);
#if DEBUG_ABSTYPE
    std::cerr << "Removing SELF type " << OldType->getDescription() << "\n";
#endif
    OldType->removeAbstractTypeUser(this);
  }

  TPI = find(Type::TypeTy);
  if (TPI != end()) {  
    // Loop over all of the types in the symbol table, replacing any references
    // to OldType with references to NewType.  Note that there may be multiple
    // occurrences, and although we only need to remove one at a time, it's
    // faster to remove them all in one pass.
    //
    VarMap &TyPlane = TPI->second;
    for (VarMap::iterator I = TyPlane.begin(), E = TyPlane.end(); I != E; ++I)
      if (I->second == (Value*)OldType) {  // FIXME when Types aren't const.
#if DEBUG_ABSTYPE
        std::cerr << "Removing type " << OldType->getDescription() << "\n";
#endif
        OldType->removeAbstractTypeUser(this);
        
        I->second = (Value*)NewType;  // TODO FIXME when types aren't const
        if (NewType->isAbstract()) {
#if DEBUG_ABSTYPE
          std::cerr << "Added type " << NewType->getDescription() << "\n";
#endif
          cast<DerivedType>(NewType)->addAbstractTypeUser(this);
        }
      }
  }
}

static void DumpVal(const std::pair<const std::string, Value *> &V) {
  std::cout << "  '" << V.first << "' = ";
  V.second->dump();
  std::cout << "\n";
}

static void DumpPlane(const std::pair<const Type *,
                                      std::map<const std::string, Value *> >&P){
  std::cout << "  Plane: ";
  P.first->dump();
  std::cout << "\n";
  for_each(P.second.begin(), P.second.end(), DumpVal);
}

void SymbolTable::dump() const {
  std::cout << "Symbol table dump:\n";
  for_each(begin(), end(), DumpPlane);
}
