//===-- SlotCalculator.cpp - Calculate what slots values land in ----------===//
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
#include "Support/DepthFirstIterator.h"
#include "Support/STLExtras.h"
#include <algorithm>

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
  for (unsigned i = 0; i < Type::FirstDerivedTyID; ++i) {
    assert(Type::getPrimitiveType((Type::PrimitiveID)i));
    insertVal(Type::getPrimitiveType((Type::PrimitiveID)i), true);
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
  for (unsigned i = 0; i < Type::FirstDerivedTyID; ++i) {
    assert(Type::getPrimitiveType((Type::PrimitiveID)i));
    insertVal(Type::getPrimitiveType((Type::PrimitiveID)i), true);
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
    insertValue(I);

  // Scavenge the types out of the functions, then add the functions themselves
  // to the value table...
  //
  for (Module::const_iterator I = TheModule->begin(), E = TheModule->end();
       I != E; ++I)
    insertValue(I);

  // Add all of the module level constants used as initializers
  //
  for (Module::const_giterator I = TheModule->gbegin(), E = TheModule->gend();
       I != E; ++I)
    if (I->hasInitializer())
      insertValue(I->getInitializer());

  // Insert constants that are named at module level into the slot pool so that
  // the module symbol table can refer to them...
  //
  if (!IgnoreNamedNodes) {
    SC_DEBUG("Inserting SymbolTable values:\n");
    processSymbolTable(&TheModule->getSymbolTable());
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
      insertValue(TI->second);
}

void SlotCalculator::processSymbolTableConstants(const SymbolTable *ST) {
  for (SymbolTable::const_iterator I = ST->begin(), E = ST->end(); I != E; ++I)
    for (SymbolTable::type_const_iterator TI = I->second.begin(), 
	   TE = I->second.end(); TI != TE; ++TI)
      if (isa<Constant>(TI->second))
	insertValue(TI->second);
}


void SlotCalculator::incorporateFunction(const Function *M) {
  assert(ModuleLevel.size() == 0 && "Module already incorporated!");

  SC_DEBUG("begin processFunction!\n");

  // Save the Table state before we process the function...
  for (unsigned i = 0; i < Table.size(); ++i)
    ModuleLevel.push_back(Table[i].size());

  SC_DEBUG("Inserting function arguments\n");

  // Iterate over function arguments, adding them to the value table...
  for(Function::const_aiterator I = M->abegin(), E = M->aend(); I != E; ++I)
    insertValue(I);

  // Iterate over all of the instructions in the function, looking for constant
  // values that are referenced.  Add these to the value pools before any
  // nonconstant values.  This will be turned into the constant pool for the
  // bytecode writer.
  //
  if (!IgnoreNamedNodes) {                // Assembly writer does not need this!
    SC_DEBUG("Inserting function constants:\n";
	     for (constant_iterator I = constant_begin(M), E = constant_end(M);
		  I != E; ++I) {
	       std::cerr << "  " << *I->getType() << " " << *I << "\n";
	     });

    // Emit all of the constants that are being used by the instructions in the
    // function...
    for_each(constant_begin(M), constant_end(M),
	     bind_obj(this, &SlotCalculator::insertValue));

    // If there is a symbol table, it is possible that the user has names for
    // constants that are not being used.  In this case, we will have problems
    // if we don't emit the constants now, because otherwise we will get 
    // symboltable references to constants not in the output.  Scan for these
    // constants now.
    //
    processSymbolTableConstants(&M->getSymbolTable());
  }

  SC_DEBUG("Inserting Labels:\n");

  // Iterate over basic blocks, adding them to the value table...
  for (Function::const_iterator I = M->begin(), E = M->end(); I != E; ++I)
    insertValue(I);

  SC_DEBUG("Inserting Instructions:\n");

  // Add all of the instructions to the type planes...
  for_each(inst_begin(M), inst_end(M),
	   bind_obj(this, &SlotCalculator::insertValue));

  if (!IgnoreNamedNodes) {
    SC_DEBUG("Inserting SymbolTable values:\n");
    processSymbolTable(&M->getSymbolTable());
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

int SlotCalculator::getValSlot(const Value *D) const {
  std::map<const Value*, unsigned>::const_iterator I = NodeMap.find(D);
  if (I == NodeMap.end()) return -1;
 
  return (int)I->second;
}


int SlotCalculator::insertValue(const Value *V) {
  int SlotNo = getValSlot(V);        // Check to see if it's already in!
  if (SlotNo != -1) return SlotNo;

  if (!isa<GlobalValue>(V))
    if (const Constant *C = dyn_cast<Constant>(V)) {
      // This makes sure that if a constant has uses (for example an array of
      // const ints), that they are inserted also.
      //
      for (User::const_op_iterator I = C->op_begin(), E = C->op_end();
           I != E; ++I)
        insertValue(*I);
    }

  return insertVal(V);
}


int SlotCalculator::insertVal(const Value *D, bool dontIgnore) {
  assert(D && "Can't insert a null value!");
  assert(getValSlot(D) == -1 && "Value is already in the table!");

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
    int ResultSlot;
    if ((ResultSlot = getValSlot(TheTy)) == -1) {
      ResultSlot = doInsertVal(TheTy);
      SC_DEBUG("  Inserted type: " << TheTy->getDescription() << " slot=" <<
	       ResultSlot << "\n");
    }

    // Loop over any contained types in the definition... in reverse depth first
    // order.  This assures that all of the leafs of a type are output before
    // the type itself is. This also assures us that we will not hit infinite
    // recursion on recursive types...
    //
    for (df_iterator<const Type*> I = df_begin(TheTy, true), 
                                  E = df_end(TheTy); I != E; ++I)
      if (*I != TheTy) {
	// If we haven't seen this sub type before, add it to our type table!
	const Type *SubTy = *I;
	if (getValSlot(SubTy) == -1) {
	  SC_DEBUG("  Inserting subtype: " << SubTy->getDescription() << "\n");
	  int Slot = doInsertVal(SubTy);
	  SC_DEBUG("  Inserted subtype: " << SubTy->getDescription() << 
		   " slot=" << Slot << "\n");
	}
      }
    return ResultSlot;
  }

  // Okay, everything is happy, actually insert the silly value now...
  return doInsertVal(D);
}


// doInsertVal - This is a small helper function to be called only be insertVal.
//
int SlotCalculator::doInsertVal(const Value *D) {
  const Type *Typ = D->getType();
  unsigned Ty;

  // Used for debugging DefSlot=-1 assertion...
  //if (Typ == Type::TypeTy)
  //  cerr << "Inserting type '" << cast<Type>(D)->getDescription() << "'!\n";

  if (Typ->isDerivedType()) {
    int ValSlot = getValSlot(Typ);
    if (ValSlot == -1) {                // Have we already entered this type?
      // Nope, this is the first we have seen the type, process it.
      ValSlot = insertVal(Typ, true);
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
