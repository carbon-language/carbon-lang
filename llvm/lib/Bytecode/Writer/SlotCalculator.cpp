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
#include "llvm/Analysis/ConstantsScanner.h"
#include "llvm/ADT/PostOrderIterator.h"
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
  insertType(Type::VoidTy,   true); // 0: VoidTySlot
  insertType(Type::FloatTy,  true); // 1: FloatTySlot
  insertType(Type::DoubleTy, true); // 2: DoubleTySlot
  insertType(Type::LabelTy,  true); // 3: LabelTySlot
  assert(TypeMap.size() == Type::FirstDerivedTyID && "Invalid primitive insert");
  // Above here *must* correspond 1:1 with the primitive types.
  insertType(Type::Int1Ty,   true); // 4: BoolTySlot
  insertType(Type::Int8Ty,   true); // 5: Int8TySlot
  insertType(Type::Int16Ty,  true); // 6: Int16TySlot
  insertType(Type::Int32Ty,  true); // 7: Int32TySlot
  insertType(Type::Int64Ty,  true); // 8: Int64TySlot
}

SlotCalculator::SlotCalculator(const Module *M ) {
  ModuleContainsAllFunctionConstants = false;
  ModuleTypeLevel = 0;
  TheModule = M;

  insertPrimitives();

  if (M == 0) return;   // Empty table...
  processModule();
}

SlotCalculator::SlotCalculator(const Function *M ) {
  ModuleContainsAllFunctionConstants = false;
  TheModule = M ? M->getParent() : 0;

  insertPrimitives();

  if (TheModule == 0) return;   // Empty table...

  processModule();              // Process module level stuff
  incorporateFunction(M);       // Start out in incorporated state
}

SlotCalculator::TypePlane &SlotCalculator::getPlane(unsigned Plane) {
  // Okay we are just returning an entry out of the main Table.  Make sure the
  // plane exists and return it.
  if (Plane >= Table.size())
    Table.resize(Plane+1);
  return Table[Plane];
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
    getOrCreateSlot(I);

  // Scavenge the types out of the functions, then add the functions themselves
  // to the value table...
  //
  for (Module::const_iterator I = TheModule->begin(), E = TheModule->end();
       I != E; ++I)
    getOrCreateSlot(I);

  // Add all of the module level constants used as initializers
  //
  for (Module::const_global_iterator I = TheModule->global_begin(),
         E = TheModule->global_end(); I != E; ++I)
    if (I->hasInitializer())
      getOrCreateSlot(I->getInitializer());

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
  // more compact modules.  This is optional, and is just used to compactify
  // the constants used by different functions together.
  //
  // This functionality tends to produce smaller bytecode files.  This should
  // not be used in the future by clients that want to, for example, build and
  // emit functions on the fly.  For now, however, it is unconditionally
  // enabled.
  ModuleContainsAllFunctionConstants = true;

  SC_DEBUG("Inserting function constants:\n");
  for (Module::const_iterator F = TheModule->begin(), E = TheModule->end();
       F != E; ++F) {
    for (const_inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      for (User::const_op_iterator OI = I->op_begin(), E = I->op_end(); 
           OI != E; ++OI) {
        if ((isa<Constant>(*OI) && !isa<GlobalValue>(*OI)) ||
            isa<InlineAsm>(*OI))
          getOrCreateSlot(*OI);
      }
      getOrCreateSlot(I->getType());
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

  SC_DEBUG("end processModule!\n");
}

// processTypeSymbolTable - Insert all of the type sin the specified symbol
// table.
void SlotCalculator::processTypeSymbolTable(const TypeSymbolTable *TST) {
  for (TypeSymbolTable::const_iterator TI = TST->begin(), TE = TST->end(); 
       TI != TE; ++TI )
    getOrCreateSlot(TI->second);
}

// processSymbolTable - Insert all of the values in the specified symbol table
// into the values table...
//
void SlotCalculator::processValueSymbolTable(const ValueSymbolTable *VST) {
  for (ValueSymbolTable::const_iterator VI = VST->begin(), VE = VST->end(); 
       VI != VE; ++VI)
    getOrCreateSlot(VI->second);
}

void SlotCalculator::incorporateFunction(const Function *F) {
  assert((ModuleLevel.empty() ||
          ModuleTypeLevel == 0) && "Module already incorporated!");

  SC_DEBUG("begin processFunction!\n");

  // Update the ModuleLevel entries to be accurate.
  ModuleLevel.resize(getNumPlanes());
  for (unsigned i = 0, e = getNumPlanes(); i != e; ++i)
    ModuleLevel[i] = getPlane(i).size();
  ModuleTypeLevel = Types.size();

  // Iterate over function arguments, adding them to the value table...
  for(Function::const_arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E; ++I)
    getOrCreateSlot(I);

  if (!ModuleContainsAllFunctionConstants) {
    // Iterate over all of the instructions in the function, looking for
    // constant values that are referenced.  Add these to the value pools
    // before any nonconstant values.  This will be turned into the constant
    // pool for the bytecode writer.
    //

    // Emit all of the constants that are being used by the instructions in
    // the function...
    for (constant_iterator CI = constant_begin(F), CE = constant_end(F);
         CI != CE; ++CI)
      getOrCreateSlot(*CI);
  }

  SC_DEBUG("Inserting Instructions:\n");

  // Add all of the instructions to the type planes...
  for (Function::const_iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
    getOrCreateSlot(BB);
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I!=E; ++I) {
      getOrCreateSlot(I);
    }
  }

  SC_DEBUG("end processFunction!\n");
}

void SlotCalculator::purgeFunction() {
  assert((ModuleLevel.size() != 0 ||
          ModuleTypeLevel != 0) && "Module not incorporated!");
  unsigned NumModuleTypes = ModuleLevel.size();

  SC_DEBUG("begin purgeFunction!\n");

  // Next, remove values from existing type planes
  for (unsigned i = 0; i != NumModuleTypes; ++i) {
    // Size of plane before function came
    unsigned ModuleLev = getModuleLevel(i);
    assert(int(ModuleLev) >= 0 && "BAD!");

    TypePlane &Plane = getPlane(i);

    assert(ModuleLev <= Plane.size() && "module levels higher than elements?");
    while (Plane.size() != ModuleLev) {
      assert(!isa<GlobalValue>(Plane.back()) &&
             "Functions cannot define globals!");
      NodeMap.erase(Plane.back());       // Erase from nodemap
      Plane.pop_back();                  // Shrink plane
    }
  }

  // We don't need this state anymore, free it up.
  ModuleLevel.clear();
  ModuleTypeLevel = 0;

  // Finally, remove any type planes defined by the function...
  while (Table.size() > NumModuleTypes) {
    TypePlane &Plane = Table.back();
    SC_DEBUG("Removing Plane " << (Table.size()-1) << " of size "
             << Plane.size() << "\n");
    while (Plane.size()) {
      assert(!isa<GlobalValue>(Plane.back()) &&
             "Functions cannot define globals!");
      NodeMap.erase(Plane.back());   // Erase from nodemap
      Plane.pop_back();              // Shrink plane
    }

    Table.pop_back();                // Nuke the plane, we don't like it.
  }

  SC_DEBUG("end purgeFunction!\n");
}

static inline bool hasNullValue(const Type *Ty) {
  return Ty != Type::LabelTy && Ty != Type::VoidTy && !isa<OpaqueType>(Ty);
}


int SlotCalculator::getSlot(const Value *V) const {
  std::map<const Value*, unsigned>::const_iterator I = NodeMap.find(V);
  if (I != NodeMap.end())
    return (int)I->second;

  return -1;
}

int SlotCalculator::getSlot(const Type*T) const {
  std::map<const Type*, unsigned>::const_iterator I = TypeMap.find(T);
  if (I != TypeMap.end())
    return (int)I->second;

  return -1;
}

int SlotCalculator::getOrCreateSlot(const Value *V) {
  if (V->getType() == Type::VoidTy) return -1;

  int SlotNo = getSlot(V);        // Check to see if it's already in!
  if (SlotNo != -1) return SlotNo;

  if (const GlobalValue *GV = dyn_cast<GlobalValue>(V))
    assert(GV->getParent() != 0 && "Global not embedded into a module!");

  if (!isa<GlobalValue>(V))  // Initializers for globals are handled explicitly
    if (const Constant *C = dyn_cast<Constant>(V)) {

      // Do not index the characters that make up constant strings.  We emit
      // constant strings as special entities that don't require their
      // individual characters to be emitted.
      if (!isa<ConstantArray>(C) || !cast<ConstantArray>(C)->isString()) {
        // This makes sure that if a constant has uses (for example an array of
        // const ints), that they are inserted also.
        //
        for (User::const_op_iterator I = C->op_begin(), E = C->op_end();
             I != E; ++I)
          getOrCreateSlot(*I);
      } else {
        assert(ModuleLevel.empty() &&
               "How can a constant string be directly accessed in a function?");
        // Otherwise, if we are emitting a bytecode file and this IS a string,
        // remember it.
        if (!C->isNullValue())
          ConstantStrings.push_back(cast<ConstantArray>(C));
      }
    }

  return insertValue(V);
}

int SlotCalculator::getOrCreateSlot(const Type* T) {
  int SlotNo = getSlot(T);        // Check to see if it's already in!
  if (SlotNo != -1) return SlotNo;
  return insertType(T);
}

int SlotCalculator::insertValue(const Value *D, bool dontIgnore) {
  assert(D && "Can't insert a null value!");
  assert(getSlot(D) == -1 && "Value is already in the table!");

  // If this node does not contribute to a plane, or if the node has a
  // name and we don't want names, then ignore the silly node... Note that types
  // do need slot numbers so that we can keep track of where other values land.
  //
  if (!dontIgnore)                               // Don't ignore nonignorables!
    if (D->getType() == Type::VoidTy ) {         // Ignore void type nodes
      SC_DEBUG("ignored value " << *D << "\n");
      return -1;                  // We do need types unconditionally though
    }

  // Okay, everything is happy, actually insert the silly value now...
  return doInsertValue(D);
}

int SlotCalculator::insertType(const Type *Ty, bool dontIgnore) {
  assert(Ty && "Can't insert a null type!");
  assert(getSlot(Ty) == -1 && "Type is already in the table!");

  // Insert the current type before any subtypes.  This is important because
  // recursive types elements are inserted in a bottom up order.  Changing
  // this here can break things.  For example:
  //
  //    global { \2 * } { { \2 }* null }
  //
  int ResultSlot = doInsertType(Ty);
  SC_DEBUG("  Inserted type: " << Ty->getDescription() << " slot=" <<
           ResultSlot << "\n");

  // Loop over any contained types in the definition... in post
  // order.
  for (po_iterator<const Type*> I = po_begin(Ty), E = po_end(Ty);
       I != E; ++I) {
    if (*I != Ty) {
      const Type *SubTy = *I;
      // If we haven't seen this sub type before, add it to our type table!
      if (getSlot(SubTy) == -1) {
        SC_DEBUG("  Inserting subtype: " << SubTy->getDescription() << "\n");
        doInsertType(SubTy);
        SC_DEBUG("  Inserted subtype: " << SubTy->getDescription() << "\n");
      }
    }
  }
  return ResultSlot;
}

// doInsertValue - This is a small helper function to be called only
// be insertValue.
//
int SlotCalculator::doInsertValue(const Value *D) {
  const Type *Typ = D->getType();
  unsigned Ty;

  // Used for debugging DefSlot=-1 assertion...
  //if (Typ == Type::TypeTy)
  //  llvm_cerr << "Inserting type '"<<cast<Type>(D)->getDescription() <<"'!\n";

  if (Typ->isDerivedType()) {
    int ValSlot = getSlot(Typ);
    if (ValSlot == -1) {                // Have we already entered this type?
      // Nope, this is the first we have seen the type, process it.
      ValSlot = insertType(Typ, true);
      assert(ValSlot != -1 && "ProcessType returned -1 for a type?");
    }
    Ty = (unsigned)ValSlot;
  } else {
    Ty = Typ->getTypeID();
  }

  if (Table.size() <= Ty)    // Make sure we have the type plane allocated...
    Table.resize(Ty+1, TypePlane());

  // If this is the first value to get inserted into the type plane, make sure
  // to insert the implicit null value...
  if (Table[Ty].empty() && hasNullValue(Typ)) {
    Value *ZeroInitializer = Constant::getNullValue(Typ);

    // If we are pushing zeroinit, it will be handled below.
    if (D != ZeroInitializer) {
      Table[Ty].push_back(ZeroInitializer);
      NodeMap[ZeroInitializer] = 0;
    }
  }

  // Insert node into table and NodeMap...
  unsigned DestSlot = NodeMap[D] = Table[Ty].size();
  Table[Ty].push_back(D);

  SC_DEBUG("  Inserting value [" << Ty << "] = " << *D << " slot=" <<
           DestSlot << " [");
  // G = Global, C = Constant, T = Type, F = Function, o = other
  SC_DEBUG((isa<GlobalVariable>(D) ? "G" : (isa<Constant>(D) ? "C" :
           (isa<Function>(D) ? "F" : "o"))));
  SC_DEBUG("]\n");
  return (int)DestSlot;
}

// doInsertType - This is a small helper function to be called only
// be insertType.
//
int SlotCalculator::doInsertType(const Type *Ty) {

  // Insert node into table and NodeMap...
  unsigned DestSlot = TypeMap[Ty] = Types.size();
  Types.push_back(Ty);

  SC_DEBUG("  Inserting type [" << DestSlot << "] = " << *Ty << "\n" );
  return (int)DestSlot;
}
