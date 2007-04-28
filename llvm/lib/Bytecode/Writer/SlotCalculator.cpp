//===-- SlotCalculator.cpp - Calculate what slots values land in ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a useful analysis step to figure out what numbered slots
// values in a program will land in (keeping track of per plane information).
//
// This is used when writing a file to disk, either in bytecode or assembly.
//
//===----------------------------------------------------------------------===//

#include "SlotCalculator.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/TypeSymbolTable.h"
#include "llvm/Type.h"
#include "llvm/ValueSymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <functional>
using namespace llvm;

#ifndef NDEBUG
#include "llvm/Support/Streams.h"
#include "llvm/Support/CommandLine.h"
static cl::opt<bool> SlotCalculatorDebugOption("scdebug",cl::init(false), 
    cl::desc("Enable SlotCalculator debug output"), cl::Hidden);
#define SC_DEBUG(X) if (SlotCalculatorDebugOption) cerr << X
#else
#define SC_DEBUG(X)
#endif

void SlotCalculator::insertPrimitives() {
  // Preload the table with the built-in types. These built-in types are
  // inserted first to ensure that they have low integer indices which helps to
  // keep bytecode sizes small. Note that the first group of indices must match
  // the Type::TypeIDs for the primitive types. After that the integer types are
  // added, but the order and value is not critical. What is critical is that 
  // the indices of these "well known" slot numbers be properly maintained in
  // Reader.h which uses them directly to extract values of these types.
  SC_DEBUG("Inserting primitive types:\n");
                                    // See WellKnownTypeSlots in Reader.h
  getOrCreateTypeSlot(Type::VoidTy  ); // 0: VoidTySlot
  getOrCreateTypeSlot(Type::FloatTy ); // 1: FloatTySlot
  getOrCreateTypeSlot(Type::DoubleTy); // 2: DoubleTySlot
  getOrCreateTypeSlot(Type::LabelTy ); // 3: LabelTySlot
  assert(TypeMap.size() == Type::FirstDerivedTyID &&"Invalid primitive insert");
  // Above here *must* correspond 1:1 with the primitive types.
  getOrCreateTypeSlot(Type::Int1Ty  ); // 4: Int1TySlot
  getOrCreateTypeSlot(Type::Int8Ty  ); // 5: Int8TySlot
  getOrCreateTypeSlot(Type::Int16Ty ); // 6: Int16TySlot
  getOrCreateTypeSlot(Type::Int32Ty ); // 7: Int32TySlot
  getOrCreateTypeSlot(Type::Int64Ty ); // 8: Int64TySlot
}

SlotCalculator::SlotCalculator(const Module *M) {
  assert(M);
  TheModule = M;

  insertPrimitives();
  processModule();
}

// processModule - Process all of the module level function declarations and
// types that are available.
//
void SlotCalculator::processModule() {
  SC_DEBUG("begin processModule!\n");

  // Add all of the global variables to the value table...
  //
  for (Module::const_global_iterator I = TheModule->global_begin(),
         E = TheModule->global_end(); I != E; ++I)
    CreateSlotIfNeeded(I);

  // Scavenge the types out of the functions, then add the functions themselves
  // to the value table...
  //
  for (Module::const_iterator I = TheModule->begin(), E = TheModule->end();
       I != E; ++I)
    CreateSlotIfNeeded(I);

  // Add all of the global aliases to the value table...
  //
  for (Module::const_alias_iterator I = TheModule->alias_begin(),
         E = TheModule->alias_end(); I != E; ++I)
    CreateSlotIfNeeded(I);

  // Add all of the module level constants used as initializers
  //
  for (Module::const_global_iterator I = TheModule->global_begin(),
         E = TheModule->global_end(); I != E; ++I)
    if (I->hasInitializer())
      CreateSlotIfNeeded(I->getInitializer());

  // Add all of the module level constants used as aliasees
  //
  for (Module::const_alias_iterator I = TheModule->alias_begin(),
         E = TheModule->alias_end(); I != E; ++I)
    if (I->getAliasee())
      CreateSlotIfNeeded(I->getAliasee());

  // Now that all global constants have been added, rearrange constant planes
  // that contain constant strings so that the strings occur at the start of the
  // plane, not somewhere in the middle.
  //
  for (unsigned plane = 0, e = Table.size(); plane != e; ++plane) {
    if (const ArrayType *AT = dyn_cast<ArrayType>(Types[plane]))
      if (AT->getElementType() == Type::Int8Ty) {
        TypePlane &Plane = Table[plane];
        unsigned FirstNonStringID = 0;
        for (unsigned i = 0, e = Plane.size(); i != e; ++i)
          if (isa<ConstantAggregateZero>(Plane[i]) ||
              (isa<ConstantArray>(Plane[i]) &&
               cast<ConstantArray>(Plane[i])->isString())) {
            // Check to see if we have to shuffle this string around.  If not,
            // don't do anything.
            if (i != FirstNonStringID) {
              // Swap the plane entries....
              std::swap(Plane[i], Plane[FirstNonStringID]);

              // Keep the NodeMap up to date.
              NodeMap[Plane[i]] = i;
              NodeMap[Plane[FirstNonStringID]] = FirstNonStringID;
            }
            ++FirstNonStringID;
          }
      }
  }

  // Scan all of the functions for their constants, which allows us to emit
  // more compact modules.
  SC_DEBUG("Inserting function constants:\n");
  for (Module::const_iterator F = TheModule->begin(), E = TheModule->end();
       F != E; ++F) {
    for (Function::const_iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
      for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I!=E;++I){
        for (User::const_op_iterator OI = I->op_begin(), E = I->op_end(); 
             OI != E; ++OI) {
          if ((isa<Constant>(*OI) && !isa<GlobalValue>(*OI)) ||
              isa<InlineAsm>(*OI))
            CreateSlotIfNeeded(*OI);
        }
        getOrCreateTypeSlot(I->getType());
      }
  }

  // Insert constants that are named at module level into the slot pool so that
  // the module symbol table can refer to them...
  SC_DEBUG("Inserting SymbolTable values:\n");
  processTypeSymbolTable(&TheModule->getTypeSymbolTable());
  processValueSymbolTable(&TheModule->getValueSymbolTable());

  // Now that we have collected together all of the information relevant to the
  // module, compactify the type table if it is particularly big and outputting
  // a bytecode file.  The basic problem we run into is that some programs have
  // a large number of types, which causes the type field to overflow its size,
  // which causes instructions to explode in size (particularly call
  // instructions).  To avoid this behavior, we "sort" the type table so that
  // all non-value types are pushed to the end of the type table, giving nice
  // low numbers to the types that can be used by instructions, thus reducing
  // the amount of explodage we suffer.
  if (Types.size() >= 64) {
    unsigned FirstNonValueTypeID = 0;
    for (unsigned i = 0, e = Types.size(); i != e; ++i)
      if (Types[i]->isFirstClassType() || Types[i]->isPrimitiveType()) {
        // Check to see if we have to shuffle this type around.  If not, don't
        // do anything.
        if (i != FirstNonValueTypeID) {
          // Swap the type ID's.
          std::swap(Types[i], Types[FirstNonValueTypeID]);

          // Keep the TypeMap up to date.
          TypeMap[Types[i]] = i;
          TypeMap[Types[FirstNonValueTypeID]] = FirstNonValueTypeID;

          // When we move a type, make sure to move its value plane as needed.
          if (Table.size() > FirstNonValueTypeID) {
            if (Table.size() <= i) Table.resize(i+1);
            std::swap(Table[i], Table[FirstNonValueTypeID]);
          }
        }
        ++FirstNonValueTypeID;
      }
  }
    
  NumModuleTypes = getNumPlanes();

  SC_DEBUG("end processModule!\n");
}

// processTypeSymbolTable - Insert all of the type sin the specified symbol
// table.
void SlotCalculator::processTypeSymbolTable(const TypeSymbolTable *TST) {
  for (TypeSymbolTable::const_iterator TI = TST->begin(), TE = TST->end(); 
       TI != TE; ++TI )
    getOrCreateTypeSlot(TI->second);
}

// processSymbolTable - Insert all of the values in the specified symbol table
// into the values table...
//
void SlotCalculator::processValueSymbolTable(const ValueSymbolTable *VST) {
  for (ValueSymbolTable::const_iterator VI = VST->begin(), VE = VST->end(); 
       VI != VE; ++VI)
    CreateSlotIfNeeded(VI->getValue());
}

void SlotCalculator::CreateSlotIfNeeded(const Value *V) {
  // Check to see if it's already in!
  if (NodeMap.count(V)) return;

  const Type *Ty = V->getType();
  assert(Ty != Type::VoidTy && "Can't insert void values!");
  
  if (const Constant *C = dyn_cast<Constant>(V)) {
    if (isa<GlobalValue>(C)) {
      // Initializers for globals are handled explicitly elsewhere.
    } else if (isa<ConstantArray>(C) && cast<ConstantArray>(C)->isString()) {
      // Do not index the characters that make up constant strings.  We emit
      // constant strings as special entities that don't require their
      // individual characters to be emitted.
      if (!C->isNullValue())
        ConstantStrings.push_back(cast<ConstantArray>(C));
    } else {
      // This makes sure that if a constant has uses (for example an array of
      // const ints), that they are inserted also.
      for (User::const_op_iterator I = C->op_begin(), E = C->op_end();
           I != E; ++I)
        CreateSlotIfNeeded(*I);
    }
  }

  unsigned TyPlane = getOrCreateTypeSlot(Ty);
  if (Table.size() <= TyPlane)    // Make sure we have the type plane allocated.
    Table.resize(TyPlane+1, TypePlane());
  
  // If this is the first value to get inserted into the type plane, make sure
  // to insert the implicit null value.
  if (Table[TyPlane].empty()) {
    // Label's and opaque types can't have a null value.
    if (Ty != Type::LabelTy && !isa<OpaqueType>(Ty)) {
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


unsigned SlotCalculator::getOrCreateTypeSlot(const Type *Ty) {
  TypeMapType::iterator TyIt = TypeMap.find(Ty);
  if (TyIt != TypeMap.end()) return TyIt->second;

  // Insert into TypeMap.
  unsigned ResultSlot = TypeMap[Ty] = Types.size();
  Types.push_back(Ty);
  SC_DEBUG("  Inserting type [" << ResultSlot << "] = " << *Ty << "\n" );
  
  // Loop over any contained types in the definition, ensuring they are also
  // inserted.
  for (Type::subtype_iterator I = Ty->subtype_begin(), E = Ty->subtype_end();
       I != E; ++I)
    getOrCreateTypeSlot(*I);

  return ResultSlot;
}



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
