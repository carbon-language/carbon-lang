//===-- ValueEnumerator.cpp - Number values and types for bitcode writer --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ValueEnumerator class.
//
//===----------------------------------------------------------------------===//

#include "ValueEnumerator.h"
#include "llvm/Module.h"
#include "llvm/TypeSymbolTable.h"
#include "llvm/ValueSymbolTable.h"
using namespace llvm;

/// ValueEnumerator - Enumerate module-level information.
ValueEnumerator::ValueEnumerator(const Module *M) {
  // Enumerate the global variables.
  for (Module::const_global_iterator I = M->global_begin(),
         E = M->global_end(); I != E; ++I)
    EnumerateValue(I);

  // Enumerate the functions.
  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I)
    EnumerateValue(I);

  // Enumerate the global variable initializers.
  for (Module::const_global_iterator I = M->global_begin(),
         E = M->global_end(); I != E; ++I)
    if (I->hasInitializer())
      EnumerateValue(I->getInitializer());

  // FIXME: Implement the 'string constant' optimization.

  // Enumerate types used by the type symbol table.
  EnumerateTypeSymbolTable(M->getTypeSymbolTable());

  // Insert constants that are named at module level into the slot pool so that
  // the module symbol table can refer to them...
  EnumerateValueSymbolTable(M->getValueSymbolTable());
  
  // Enumerate types used by function bodies.
  for (Module::const_iterator F = M->begin(), E = M->end(); F != E; ++F) {
    for (Function::const_iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
      for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I!=E;++I){
        for (User::const_op_iterator OI = I->op_begin(), E = I->op_end(); 
             OI != E; ++OI)
          EnumerateType((*OI)->getType());
        EnumerateType(I->getType());
      }
  }
    
  
  // FIXME: std::partition the type and value tables so that first-class types
  // come earlier than aggregates.
  
  // FIXME: Sort type/value tables by frequency.
}

/// EnumerateTypeSymbolTable - Insert all of the types in the specified symbol
/// table.
void ValueEnumerator::EnumerateTypeSymbolTable(const TypeSymbolTable &TST) {
  for (TypeSymbolTable::const_iterator TI = TST.begin(), TE = TST.end(); 
       TI != TE; ++TI)
    EnumerateType(TI->second);
}

/// EnumerateValueSymbolTable - Insert all of the values in the specified symbol
/// table into the values table.
void ValueEnumerator::EnumerateValueSymbolTable(const ValueSymbolTable &VST) {
  for (ValueSymbolTable::const_iterator VI = VST.begin(), VE = VST.end(); 
       VI != VE; ++VI)
    EnumerateValue(VI->getValue());
}

void ValueEnumerator::EnumerateValue(const Value *V) {
  assert(V->getType() != Type::VoidTy && "Can't insert void values!");
  
  // Check to see if it's already in!
  unsigned &ValueID = ValueMap[V];
  if (ValueID) {
    // Increment use count.
    Values[ValueID-1].second++;
    return;
  }
  
  // Add the value.
  Values.push_back(std::make_pair(V, 1U));
  ValueID = Values.size();

  if (const Constant *C = dyn_cast<Constant>(V)) {
    if (isa<GlobalValue>(C)) {
      // Initializers for globals are handled explicitly elsewhere.
    } else {
      // This makes sure that if a constant has uses (for example an array of
      // const ints), that they are inserted also.
      for (User::const_op_iterator I = C->op_begin(), E = C->op_end();
           I != E; ++I)
        EnumerateValue(*I);
    }
  }

  EnumerateType(V->getType());
}


void ValueEnumerator::EnumerateType(const Type *Ty) {
  unsigned &TypeID = TypeMap[Ty];
  
  if (TypeID) {
    // If we've already seen this type, just increase its occurrence count.
    Types[TypeID-1].second++;
    return;
  }
  
  // First time we saw this type, add it.
  Types.push_back(std::make_pair(Ty, 1U));
  TypeID = Types.size();
  
  // Enumerate subtypes.
  for (Type::subtype_iterator I = Ty->subtype_begin(), E = Ty->subtype_end();
       I != E; ++I)
    EnumerateType(*I);
}



#if 0

void SlotCalculator::incorporateFunction(const Function *F) {
  SC_DEBUG("begin processFunction!\n");
  
  // Iterate over function arguments, adding them to the value table...
  for(Function::const_arg_iterator I = F->arg_begin(), E = F->arg_end();
      I != E; ++I)
    CreateFunctionValueSlot(I);
  
  SC_DEBUG("Inserting Instructions:\n");
  
  // Add all of the instructions to the type planes...
  for (Function::const_iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
    CreateFunctionValueSlot(BB);
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I!=E; ++I) {
      if (I->getType() != Type::VoidTy)
        CreateFunctionValueSlot(I);
    }
  }
  
  SC_DEBUG("end processFunction!\n");
}

void SlotCalculator::purgeFunction() {
  SC_DEBUG("begin purgeFunction!\n");
  
  // Next, remove values from existing type planes
  for (DenseMap<unsigned,unsigned,
          ModuleLevelDenseMapKeyInfo>::iterator I = ModuleLevel.begin(),
       E = ModuleLevel.end(); I != E; ++I) {
    unsigned PlaneNo = I->first;
    unsigned ModuleLev = I->second;
    
    // Pop all function-local values in this type-plane off of Table.
    TypePlane &Plane = getPlane(PlaneNo);
    assert(ModuleLev < Plane.size() && "module levels higher than elements?");
    for (unsigned i = ModuleLev, e = Plane.size(); i != e; ++i) {
      NodeMap.erase(Plane.back());       // Erase from nodemap
      Plane.pop_back();                  // Shrink plane
    }
  }

  ModuleLevel.clear();

  // Finally, remove any type planes defined by the function...
  while (Table.size() > NumModuleTypes) {
    TypePlane &Plane = Table.back();
    SC_DEBUG("Removing Plane " << (Table.size()-1) << " of size "
             << Plane.size() << "\n");
    for (unsigned i = 0, e = Plane.size(); i != e; ++i)
      NodeMap.erase(Plane[i]);   // Erase from nodemap
    
    Table.pop_back();                // Nuke the plane, we don't like it.
  }
  
  SC_DEBUG("end purgeFunction!\n");
}

inline static bool hasImplicitNull(const Type* Ty) {
  return Ty != Type::LabelTy && Ty != Type::VoidTy && !isa<OpaqueType>(Ty);
}

void SlotCalculator::CreateFunctionValueSlot(const Value *V) {
  assert(!NodeMap.count(V) && "Function-local value can't be inserted!");
  
  const Type *Ty = V->getType();
  assert(Ty != Type::VoidTy && "Can't insert void values!");
  assert(!isa<Constant>(V) && "Not a function-local value!");
  
  unsigned TyPlane = getOrCreateTypeSlot(Ty);
  if (Table.size() <= TyPlane)    // Make sure we have the type plane allocated.
    Table.resize(TyPlane+1, TypePlane());
  
  // If this is the first value noticed of this type within this function,
  // remember the module level for this type plane in ModuleLevel.  This reminds
  // us to remove the values in purgeFunction and tells us how many to remove.
  if (TyPlane < NumModuleTypes)
    ModuleLevel.insert(std::make_pair(TyPlane, Table[TyPlane].size()));
  
  // If this is the first value to get inserted into the type plane, make sure
  // to insert the implicit null value.
  if (Table[TyPlane].empty()) {
    // Label's and opaque types can't have a null value.
    if (hasImplicitNull(Ty)) {
      Value *ZeroInitializer = Constant::getNullValue(Ty);
      
      // If we are pushing zeroinit, it will be handled below.
      if (V != ZeroInitializer) {
        Table[TyPlane].push_back(ZeroInitializer);
        NodeMap[ZeroInitializer] = 0;
      }
    }
  }
  
  // Insert node into table and NodeMap...
  NodeMap[V] = Table[TyPlane].size();
  Table[TyPlane].push_back(V);
  
  SC_DEBUG("  Inserting value [" << TyPlane << "] = " << *V << " slot=" <<
           NodeMap[V] << "\n");
}

#endif
