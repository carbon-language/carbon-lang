//===-- SlotCalculator.cpp - Calculate what slots values land in ----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements a useful analysis step to figure out what numbered 
// slots values in a program will land in (keeping track of per plane
// information as required.
//
// This is used primarily for when writing a file to disk, either in bytecode
// or source format.
//
//===----------------------------------------------------------------------===//

#include "llvm/SlotCalculator.h"
#include "llvm/Analysis/ConstantsScanner.h"
#include "llvm/Module.h"
#include "llvm/iOther.h"
#include "llvm/Constant.h"
#include "llvm/DerivedTypes.h"
#include "llvm/SymbolTable.h"
#include "Support/PostOrderIterator.h"
#include "Support/STLExtras.h"
#include <algorithm>
using namespace llvm;

#if 0
#define SC_DEBUG(X) std::cerr << X
#else
#define SC_DEBUG(X)
#endif

SlotCalculator::SlotCalculator(const Module *M, bool IgnoreNamed) {
  IgnoreNamedNodes = IgnoreNamed;
  TheModule = M;

  // Preload table... Make sure that all of the primitive types are in the table
  // and that their Primitive ID is equal to their slot #
  //
  SC_DEBUG("Inserting primitive types:\n");
  for (unsigned i = 0; i < Type::FirstDerivedTyID; ++i) {
    assert(Type::getPrimitiveType((Type::PrimitiveID)i));
    insertValue(Type::getPrimitiveType((Type::PrimitiveID)i), true);
  }

  if (M == 0) return;   // Empty table...
  processModule();
}

SlotCalculator::SlotCalculator(const Function *M, bool IgnoreNamed) {
  IgnoreNamedNodes = IgnoreNamed;
  TheModule = M ? M->getParent() : 0;

  // Preload table... Make sure that all of the primitive types are in the table
  // and that their Primitive ID is equal to their slot #
  //
  SC_DEBUG("Inserting primitive types:\n");
  for (unsigned i = 0; i < Type::FirstDerivedTyID; ++i) {
    assert(Type::getPrimitiveType((Type::PrimitiveID)i));
    insertValue(Type::getPrimitiveType((Type::PrimitiveID)i), true);
  }

  if (TheModule == 0) return;   // Empty table...

  processModule();              // Process module level stuff
  incorporateFunction(M);         // Start out in incorporated state
}


// processModule - Process all of the module level function declarations and
// types that are available.
//
void SlotCalculator::processModule() {
  SC_DEBUG("begin processModule!\n");

  // Add all of the global variables to the value table...
  //
  for (Module::const_giterator I = TheModule->gbegin(), E = TheModule->gend();
       I != E; ++I)
    getOrCreateSlot(I);

  // Scavenge the types out of the functions, then add the functions themselves
  // to the value table...
  //
  for (Module::const_iterator I = TheModule->begin(), E = TheModule->end();
       I != E; ++I)
    getOrCreateSlot(I);

  // Add all of the module level constants used as initializers
  //
  for (Module::const_giterator I = TheModule->gbegin(), E = TheModule->gend();
       I != E; ++I)
    if (I->hasInitializer())
      getOrCreateSlot(I->getInitializer());

#if 0
  // FIXME: Empirically, this causes the bytecode files to get BIGGER, because
  // it explodes the operand size numbers to be bigger than can be handled
  // compactly, which offsets the ~40% savings in constant sizes.  Whoops.

  // If we are emitting a bytecode file, scan all of the functions for their
  // constants, which allows us to emit more compact modules.  This is optional,
  // and is just used to compactify the constants used by different functions
  // together.
  if (!IgnoreNamedNodes) {
    SC_DEBUG("Inserting function constants:\n");
    for (Module::const_iterator F = TheModule->begin(), E = TheModule->end();
         F != E; ++F)
      for_each(constant_begin(F), constant_end(F),
               bind_obj(this, &SlotCalculator::getOrCreateSlot));
  }
#endif

  // Insert constants that are named at module level into the slot pool so that
  // the module symbol table can refer to them...
  //
  if (!IgnoreNamedNodes) {
    SC_DEBUG("Inserting SymbolTable values:\n");
    processSymbolTable(&TheModule->getSymbolTable());
  }

  // Now that we have collected together all of the information relevant to the
  // module, compactify the type table if it is particularly big and outputting
  // a bytecode file.  The basic problem we run into is that some programs have
  // a large number of types, which causes the type field to overflow its size,
  // which causes instructions to explode in size (particularly call
  // instructions).  To avoid this behavior, we "sort" the type table so that
  // all non-value types are pushed to the end of the type table, giving nice
  // low numbers to the types that can be used by instructions, thus reducing
  // the amount of explodage we suffer.
  if (!IgnoreNamedNodes && Table[Type::TypeTyID].size() >= 64) {
    // Scan through the type table moving value types to the start of the table.
    TypePlane *Types = &Table[Type::TypeTyID];
    unsigned FirstNonValueTypeID = 0;
    for (unsigned i = 0, e = Types->size(); i != e; ++i)
      if (cast<Type>((*Types)[i])->isFirstClassType() ||
          cast<Type>((*Types)[i])->isPrimitiveType()) {
        // Check to see if we have to shuffle this type around.  If not, don't
        // do anything.
        if (i != FirstNonValueTypeID) {
          assert(i != Type::TypeTyID && FirstNonValueTypeID != Type::TypeTyID &&
                 "Cannot move around the type plane!");

          // Swap the type ID's.
          std::swap((*Types)[i], (*Types)[FirstNonValueTypeID]);

          // Keep the NodeMap up to date.
          NodeMap[(*Types)[i]] = i;
          NodeMap[(*Types)[FirstNonValueTypeID]] = FirstNonValueTypeID;

          // When we move a type, make sure to move its value plane as needed.
          if (Table.size() > FirstNonValueTypeID) {
            if (Table.size() <= i) Table.resize(i+1);
            std::swap(Table[i], Table[FirstNonValueTypeID]);
            Types = &Table[Type::TypeTyID];
          }
        }
        ++FirstNonValueTypeID;
      }
  }

  SC_DEBUG("end processModule!\n");
}

// processSymbolTable - Insert all of the values in the specified symbol table
// into the values table...
//
void SlotCalculator::processSymbolTable(const SymbolTable *ST) {
  for (SymbolTable::const_iterator I = ST->begin(), E = ST->end(); I != E; ++I)
    for (SymbolTable::type_const_iterator TI = I->second.begin(), 
	   TE = I->second.end(); TI != TE; ++TI)
      getOrCreateSlot(TI->second);
}

void SlotCalculator::processSymbolTableConstants(const SymbolTable *ST) {
  for (SymbolTable::const_iterator I = ST->begin(), E = ST->end(); I != E; ++I)
    for (SymbolTable::type_const_iterator TI = I->second.begin(), 
	   TE = I->second.end(); TI != TE; ++TI)
      if (isa<Constant>(TI->second))
	getOrCreateSlot(TI->second);
}


void SlotCalculator::incorporateFunction(const Function *F) {
  assert(ModuleLevel.size() == 0 && "Module already incorporated!");

  SC_DEBUG("begin processFunction!\n");

  // Save the Table state before we process the function...
  for (unsigned i = 0; i < Table.size(); ++i)
    ModuleLevel.push_back(Table[i].size());

  SC_DEBUG("Inserting function arguments\n");

  // Iterate over function arguments, adding them to the value table...
  for(Function::const_aiterator I = F->abegin(), E = F->aend(); I != E; ++I)
    getOrCreateSlot(I);

  // Iterate over all of the instructions in the function, looking for constant
  // values that are referenced.  Add these to the value pools before any
  // nonconstant values.  This will be turned into the constant pool for the
  // bytecode writer.
  //
  if (!IgnoreNamedNodes) {                // Assembly writer does not need this!
    SC_DEBUG("Inserting function constants:\n";
	     for (constant_iterator I = constant_begin(F), E = constant_end(F);
		  I != E; ++I) {
	       std::cerr << "  " << *I->getType() << " " << *I << "\n";
	     });

    // Emit all of the constants that are being used by the instructions in the
    // function...
    for_each(constant_begin(F), constant_end(F),
	     bind_obj(this, &SlotCalculator::getOrCreateSlot));

    // If there is a symbol table, it is possible that the user has names for
    // constants that are not being used.  In this case, we will have problems
    // if we don't emit the constants now, because otherwise we will get 
    // symbol table references to constants not in the output.  Scan for these
    // constants now.
    //
    processSymbolTableConstants(&F->getSymbolTable());
  }

  SC_DEBUG("Inserting Labels:\n");

  // Iterate over basic blocks, adding them to the value table...
  for (Function::const_iterator I = F->begin(), E = F->end(); I != E; ++I)
    getOrCreateSlot(I);

  SC_DEBUG("Inserting Instructions:\n");

  // Add all of the instructions to the type planes...
  for (Function::const_iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I!=E; ++I) {
      getOrCreateSlot(I);
      if (const VANextInst *VAN = dyn_cast<VANextInst>(I))
        getOrCreateSlot(VAN->getArgType());
    }

  if (!IgnoreNamedNodes) {
    SC_DEBUG("Inserting SymbolTable values:\n");
    processSymbolTable(&F->getSymbolTable());
  }

  SC_DEBUG("end processFunction!\n");
}

void SlotCalculator::purgeFunction() {
  assert(ModuleLevel.size() != 0 && "Module not incorporated!");
  unsigned NumModuleTypes = ModuleLevel.size();

  SC_DEBUG("begin purgeFunction!\n");

  // First, remove values from existing type planes
  for (unsigned i = 0; i < NumModuleTypes; ++i) {
    unsigned ModuleSize = ModuleLevel[i];  // Size of plane before function came
    TypePlane &CurPlane = Table[i];
    //SC_DEBUG("Processing Plane " <<i<< " of size " << CurPlane.size() <<"\n");
	     
    while (CurPlane.size() != ModuleSize) {
      //SC_DEBUG("  Removing [" << i << "] Value=" << CurPlane.back() << "\n");
      std::map<const Value *, unsigned>::iterator NI =
        NodeMap.find(CurPlane.back());
      assert(NI != NodeMap.end() && "Node not in nodemap?");
      NodeMap.erase(NI);   // Erase from nodemap
      CurPlane.pop_back();                            // Shrink plane
    }
  }

  // We don't need this state anymore, free it up.
  ModuleLevel.clear();

  // Next, remove any type planes defined by the function...
  while (NumModuleTypes != Table.size()) {
    TypePlane &Plane = Table.back();
    SC_DEBUG("Removing Plane " << (Table.size()-1) << " of size "
	     << Plane.size() << "\n");
    while (Plane.size()) {
      NodeMap.erase(NodeMap.find(Plane.back()));   // Erase from nodemap
      Plane.pop_back();                            // Shrink plane
    }

    Table.pop_back();                      // Nuke the plane, we don't like it.
  }

  SC_DEBUG("end purgeFunction!\n");
}

int SlotCalculator::getSlot(const Value *D) const {
  std::map<const Value*, unsigned>::const_iterator I = NodeMap.find(D);
  if (I == NodeMap.end()) return -1;
 
  return (int)I->second;
}


int SlotCalculator::getOrCreateSlot(const Value *V) {
  int SlotNo = getSlot(V);        // Check to see if it's already in!
  if (SlotNo != -1) return SlotNo;

  if (!isa<GlobalValue>(V))
    if (const Constant *C = dyn_cast<Constant>(V)) {
      // This makes sure that if a constant has uses (for example an array of
      // const ints), that they are inserted also.
      //
      for (User::const_op_iterator I = C->op_begin(), E = C->op_end();
           I != E; ++I)
        getOrCreateSlot(*I);
    }

  return insertValue(V);
}


int SlotCalculator::insertValue(const Value *D, bool dontIgnore) {
  assert(D && "Can't insert a null value!");
  assert(getSlot(D) == -1 && "Value is already in the table!");

  // If this node does not contribute to a plane, or if the node has a 
  // name and we don't want names, then ignore the silly node... Note that types
  // do need slot numbers so that we can keep track of where other values land.
  //
  if (!dontIgnore)                               // Don't ignore nonignorables!
    if (D->getType() == Type::VoidTy ||          // Ignore void type nodes
	(IgnoreNamedNodes &&                     // Ignore named and constants
	 (D->hasName() || isa<Constant>(D)) && !isa<Type>(D))) {
      SC_DEBUG("ignored value " << *D << "\n");
      return -1;                  // We do need types unconditionally though
    }

  // If it's a type, make sure that all subtypes of the type are included...
  if (const Type *TheTy = dyn_cast<Type>(D)) {

    // Insert the current type before any subtypes.  This is important because
    // recursive types elements are inserted in a bottom up order.  Changing
    // this here can break things.  For example:
    //
    //    global { \2 * } { { \2 }* null }
    //
    int ResultSlot = doInsertValue(TheTy);
    SC_DEBUG("  Inserted type: " << TheTy->getDescription() << " slot=" <<
             ResultSlot << "\n");

    // Loop over any contained types in the definition... in post
    // order.
    //
    for (po_iterator<const Type*> I = po_begin(TheTy), E = po_end(TheTy);
         I != E; ++I) {
      if (*I != TheTy) {
        const Type *SubTy = *I;
	// If we haven't seen this sub type before, add it to our type table!
        if (getSlot(SubTy) == -1) {
          SC_DEBUG("  Inserting subtype: " << SubTy->getDescription() << "\n");
          int Slot = doInsertValue(SubTy);
          SC_DEBUG("  Inserted subtype: " << SubTy->getDescription() << 
                   " slot=" << Slot << "\n");
        }
      }
    }
    return ResultSlot;
  }

  // Okay, everything is happy, actually insert the silly value now...
  return doInsertValue(D);
}


// doInsertValue - This is a small helper function to be called only
// be insertValue.
//
int SlotCalculator::doInsertValue(const Value *D) {
  const Type *Typ = D->getType();
  unsigned Ty;

  // Used for debugging DefSlot=-1 assertion...
  //if (Typ == Type::TypeTy)
  //  cerr << "Inserting type '" << cast<Type>(D)->getDescription() << "'!\n";

  if (Typ->isDerivedType()) {
    int ValSlot = getSlot(Typ);
    if (ValSlot == -1) {                // Have we already entered this type?
      // Nope, this is the first we have seen the type, process it.
      ValSlot = insertValue(Typ, true);
      assert(ValSlot != -1 && "ProcessType returned -1 for a type?");
    }
    Ty = (unsigned)ValSlot;
  } else {
    Ty = Typ->getPrimitiveID();
  }
  
  if (Table.size() <= Ty)    // Make sure we have the type plane allocated...
    Table.resize(Ty+1, TypePlane());

  // If this is the first value to get inserted into the type plane, make sure
  // to insert the implicit null value...
  if (Table[Ty].empty() && Ty >= Type::FirstDerivedTyID && !IgnoreNamedNodes) {
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

  SC_DEBUG("  Inserting value [" << Ty << "] = " << D << " slot=" << 
	   DestSlot << " [");
  // G = Global, C = Constant, T = Type, F = Function, o = other
  SC_DEBUG((isa<GlobalVariable>(D) ? "G" : (isa<Constant>(D) ? "C" : 
           (isa<Type>(D) ? "T" : (isa<Function>(D) ? "F" : "o")))));
  SC_DEBUG("]\n");
  return (int)DestSlot;
}
