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

#include "llvm/Analysis/SlotCalculator.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iOther.h"
#include "llvm/Module.h"
#include "llvm/SymbolTable.h"
#include "llvm/Analysis/ConstantsScanner.h"
#include "Support/PostOrderIterator.h"
#include "Support/STLExtras.h"
#include <algorithm>
using namespace llvm;

#if 0
#define SC_DEBUG(X) std::cerr << X
#else
#define SC_DEBUG(X)
#endif

SlotCalculator::SlotCalculator(const Module *M, bool buildBytecodeInfo) {
  BuildBytecodeInfo = buildBytecodeInfo;
  ModuleContainsAllFunctionConstants = false;
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

SlotCalculator::SlotCalculator(const Function *M, bool buildBytecodeInfo) {
  BuildBytecodeInfo = buildBytecodeInfo;
  ModuleContainsAllFunctionConstants = false;
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
  incorporateFunction(M);       // Start out in incorporated state
}

unsigned SlotCalculator::getGlobalSlot(const Value *V) const {
  assert(!CompactionTable.empty() &&
         "This method can only be used when compaction is enabled!");
  if (const ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(V))
    V = CPR->getValue();
  std::map<const Value*, unsigned>::const_iterator I = NodeMap.find(V);
  assert(I != NodeMap.end() && "Didn't find global slot entry!");
  return I->second;
}

SlotCalculator::TypePlane &SlotCalculator::getPlane(unsigned Plane) {
  unsigned PIdx = Plane;
  if (CompactionTable.empty()) {                // No compaction table active?
    // fall out
  } else if (!CompactionTable[Plane].empty()) { // Compaction table active.
    assert(Plane < CompactionTable.size());
    return CompactionTable[Plane];
  } else {
    // Final case: compaction table active, but this plane is not
    // compactified.  If the type plane is compactified, unmap back to the
    // global type plane corresponding to "Plane".
    if (!CompactionTable[Type::TypeTyID].empty()) {
      const Type *Ty = cast<Type>(CompactionTable[Type::TypeTyID][Plane]);
      std::map<const Value*, unsigned>::iterator It = NodeMap.find(Ty);
      assert(It != NodeMap.end() && "Type not in global constant map?");
      PIdx = It->second;
    }
  }

  // Okay we are just returning an entry out of the main Table.  Make sure the
  // plane exists and return it.
  if (PIdx >= Table.size())
    Table.resize(PIdx+1);
  return Table[PIdx];
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

  // Now that all global constants have been added, rearrange constant planes
  // that contain constant strings so that the strings occur at the start of the
  // plane, not somewhere in the middle.
  //
  if (BuildBytecodeInfo) {
    TypePlane &Types = Table[Type::TypeTyID];
    for (unsigned plane = 0, e = Table.size(); plane != e; ++plane) {
      if (const ArrayType *AT = dyn_cast<ArrayType>(Types[plane]))
        if (AT->getElementType() == Type::SByteTy ||
            AT->getElementType() == Type::UByteTy) {
          TypePlane &Plane = Table[plane];
          unsigned FirstNonStringID = 0;
          for (unsigned i = 0, e = Plane.size(); i != e; ++i)
            if (isa<ConstantAggregateZero>(Plane[i]) ||
                cast<ConstantArray>(Plane[i])->isString()) {
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
  }
  
  // If we are emitting a bytecode file, scan all of the functions for their
  // constants, which allows us to emit more compact modules.  This is optional,
  // and is just used to compactify the constants used by different functions
  // together.
  //
  // This functionality is completely optional for the bytecode writer, but
  // tends to produce smaller bytecode files.  This should not be used in the
  // future by clients that want to, for example, build and emit functions on
  // the fly.  For now, however, it is unconditionally enabled when building
  // bytecode information.
  //
  if (BuildBytecodeInfo) {
    ModuleContainsAllFunctionConstants = true;

    SC_DEBUG("Inserting function constants:\n");
    for (Module::const_iterator F = TheModule->begin(), E = TheModule->end();
         F != E; ++F) {
      for (const_inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I){
        for (unsigned op = 0, e = I->getNumOperands(); op != e; ++op)
          if (isa<Constant>(I->getOperand(op)))
            getOrCreateSlot(I->getOperand(op));
        getOrCreateSlot(I->getType());
        if (const VANextInst *VAN = dyn_cast<VANextInst>(*I))
          getOrCreateSlot(VAN->getArgType());
      }
      processSymbolTableConstants(&F->getSymbolTable());
    }
  }

  // Insert constants that are named at module level into the slot pool so that
  // the module symbol table can refer to them...
  //
  if (BuildBytecodeInfo) {
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
  if (BuildBytecodeInfo && Table[Type::TypeTyID].size() >= 64) {
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
      if (isa<Constant>(TI->second) || isa<Type>(TI->second))
	getOrCreateSlot(TI->second);
}


void SlotCalculator::incorporateFunction(const Function *F) {
  assert(ModuleLevel.size() == 0 && "Module already incorporated!");

  SC_DEBUG("begin processFunction!\n");

  // If we emitted all of the function constants, build a compaction table.
  if (BuildBytecodeInfo && ModuleContainsAllFunctionConstants)
    buildCompactionTable(F);

  // Update the ModuleLevel entries to be accurate.
  ModuleLevel.resize(getNumPlanes());
  for (unsigned i = 0, e = getNumPlanes(); i != e; ++i)
    ModuleLevel[i] = getPlane(i).size();

  // Iterate over function arguments, adding them to the value table...
  for(Function::const_aiterator I = F->abegin(), E = F->aend(); I != E; ++I)
    getOrCreateSlot(I);

  if (BuildBytecodeInfo &&              // Assembly writer does not need this!
      !ModuleContainsAllFunctionConstants) {
    // Iterate over all of the instructions in the function, looking for
    // constant values that are referenced.  Add these to the value pools
    // before any nonconstant values.  This will be turned into the constant
    // pool for the bytecode writer.
    //
    
    // Emit all of the constants that are being used by the instructions in
    // the function...
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

  SC_DEBUG("Inserting Instructions:\n");

  // Add all of the instructions to the type planes...
  for (Function::const_iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
    getOrCreateSlot(BB);
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I!=E; ++I) {
      getOrCreateSlot(I);
      if (const VANextInst *VAN = dyn_cast<VANextInst>(I))
        getOrCreateSlot(VAN->getArgType());
    }
  }

  // If we are building a compaction table, prune out planes that do not benefit
  // from being compactified.
  if (!CompactionTable.empty())
    pruneCompactionTable();

  SC_DEBUG("end processFunction!\n");
}

void SlotCalculator::purgeFunction() {
  assert(ModuleLevel.size() != 0 && "Module not incorporated!");
  unsigned NumModuleTypes = ModuleLevel.size();

  SC_DEBUG("begin purgeFunction!\n");

  // First, free the compaction map if used.
  CompactionNodeMap.clear();

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

  // Finally, remove any type planes defined by the function...
  if (!CompactionTable.empty()) {
    CompactionTable.clear();
  } else {
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
  }

  SC_DEBUG("end purgeFunction!\n");
}

static inline bool hasNullValue(unsigned TyID) {
  return TyID != Type::LabelTyID && TyID != Type::TypeTyID &&
         TyID != Type::VoidTyID;
}

/// getOrCreateCompactionTableSlot - This method is used to build up the initial
/// approximation of the compaction table.
unsigned SlotCalculator::getOrCreateCompactionTableSlot(const Value *V) {
  if (const ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(V))
    V = CPR->getValue();
  std::map<const Value*, unsigned>::iterator I =
    CompactionNodeMap.lower_bound(V);
  if (I != CompactionNodeMap.end() && I->first == V)
    return I->second;  // Already exists?

  // Make sure the type is in the table.
  unsigned Ty;
  if (!CompactionTable[Type::TypeTyID].empty())
    Ty = getOrCreateCompactionTableSlot(V->getType());
  else    // If the type plane was decompactified, use the global plane ID
    Ty = getSlot(V->getType());
  if (CompactionTable.size() <= Ty)
    CompactionTable.resize(Ty+1);

  assert(!isa<Type>(V) || ModuleLevel.empty());

  TypePlane &TyPlane = CompactionTable[Ty];

  // Make sure to insert the null entry if the thing we are inserting is not a
  // null constant.
  if (TyPlane.empty() && hasNullValue(V->getType()->getPrimitiveID())) {
    Value *ZeroInitializer = Constant::getNullValue(V->getType());
    if (V != ZeroInitializer) {
      TyPlane.push_back(ZeroInitializer);
      CompactionNodeMap[ZeroInitializer] = 0;
    }
  }

  unsigned SlotNo = TyPlane.size();
  TyPlane.push_back(V);
  CompactionNodeMap.insert(std::make_pair(V, SlotNo));
  return SlotNo;
}


/// buildCompactionTable - Since all of the function constants and types are
/// stored in the module-level constant table, we don't need to emit a function
/// constant table.  Also due to this, the indices for various constants and
/// types might be very large in large programs.  In order to avoid blowing up
/// the size of instructions in the bytecode encoding, we build a compaction
/// table, which defines a mapping from function-local identifiers to global
/// identifiers.
void SlotCalculator::buildCompactionTable(const Function *F) {
  assert(CompactionNodeMap.empty() && "Compaction table already built!");
  // First step, insert the primitive types.
  CompactionTable.resize(Type::TypeTyID+1);
  for (unsigned i = 0; i != Type::FirstDerivedTyID; ++i) {
    const Type *PrimTy = Type::getPrimitiveType((Type::PrimitiveID)i);
    CompactionTable[Type::TypeTyID].push_back(PrimTy);
    CompactionNodeMap[PrimTy] = i;
  }

  // Next, include any types used by function arguments.
  for (Function::const_aiterator I = F->abegin(), E = F->aend(); I != E; ++I)
    getOrCreateCompactionTableSlot(I->getType());

  // Next, find all of the types and values that are referred to by the
  // instructions in the program.
  for (const_inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
    getOrCreateCompactionTableSlot(I->getType());
    for (unsigned op = 0, e = I->getNumOperands(); op != e; ++op)
      if (isa<Constant>(I->getOperand(op)) ||
          isa<GlobalValue>(I->getOperand(op)))
        getOrCreateCompactionTableSlot(I->getOperand(op));
    if (const VANextInst *VAN = dyn_cast<VANextInst>(*I))
      getOrCreateCompactionTableSlot(VAN->getArgType());
  }

  const SymbolTable &ST = F->getSymbolTable();
  for (SymbolTable::const_iterator I = ST.begin(), E = ST.end(); I != E; ++I)
    for (SymbolTable::type_const_iterator TI = I->second.begin(), 
	   TE = I->second.end(); TI != TE; ++TI)
      if (isa<Constant>(TI->second) || isa<Type>(TI->second) ||
          isa<GlobalValue>(TI->second))
	getOrCreateCompactionTableSlot(TI->second);

  // Now that we have all of the values in the table, and know what types are
  // referenced, make sure that there is at least the zero initializer in any
  // used type plane.  Since the type was used, we will be emitting instructions
  // to the plane even if there are no constants in it.
  CompactionTable.resize(CompactionTable[Type::TypeTyID].size());
  for (unsigned i = 0, e = CompactionTable.size(); i != e; ++i)
    if (CompactionTable[i].empty() && i != Type::VoidTyID &&
        i != Type::LabelTyID) {
      const Type *Ty = cast<Type>(CompactionTable[Type::TypeTyID][i]);
      getOrCreateCompactionTableSlot(Constant::getNullValue(Ty));
    }
  
  // Okay, now at this point, we have a legal compaction table.  Since we want
  // to emit the smallest possible binaries, do not compactify the type plane if
  // it will not save us anything.  Because we have not yet incorporated the
  // function body itself yet, we don't know whether or not it's a good idea to
  // compactify other planes.  We will defer this decision until later.
  TypePlane &GlobalTypes = Table[Type::TypeTyID];
  
  // All of the values types will be scrunched to the start of the types plane
  // of the global table.  Figure out just how many there are.
  assert(!GlobalTypes.empty() && "No global types???");
  unsigned NumFCTypes = GlobalTypes.size()-1;
  while (!cast<Type>(GlobalTypes[NumFCTypes])->isFirstClassType())
    --NumFCTypes;

  // If there are fewer that 64 types, no instructions will be exploded due to
  // the size of the type operands.  Thus there is no need to compactify types.
  // Also, if the compaction table contains most of the entries in the global
  // table, there really is no reason to compactify either.
  if (NumFCTypes < 64) {
    // Decompactifying types is tricky, because we have to move type planes all
    // over the place.  At least we don't need to worry about updating the
    // CompactionNodeMap for non-types though.
    std::vector<TypePlane> TmpCompactionTable;
    std::swap(CompactionTable, TmpCompactionTable);
    TypePlane Types;
    std::swap(Types, TmpCompactionTable[Type::TypeTyID]);
    
    // Move each plane back over to the uncompactified plane
    while (!Types.empty()) {
      const Type *Ty = cast<Type>(Types.back());
      Types.pop_back();
      CompactionNodeMap.erase(Ty);  // Decompactify type!

      if (Ty != Type::TypeTy) {
        // Find the global slot number for this type.
        int TySlot = getSlot(Ty);
        assert(TySlot != -1 && "Type doesn't exist in global table?");
        
        // Now we know where to put the compaction table plane.
        if (CompactionTable.size() <= unsigned(TySlot))
          CompactionTable.resize(TySlot+1);
        // Move the plane back into the compaction table.
        std::swap(CompactionTable[TySlot], TmpCompactionTable[Types.size()]);

        // And remove the empty plane we just moved in.
        TmpCompactionTable.pop_back();
      }
    }
  }
}


/// pruneCompactionTable - Once the entire function being processed has been
/// incorporated into the current compaction table, look over the compaction
/// table and check to see if there are any values whose compaction will not
/// save us any space in the bytecode file.  If compactifying these values
/// serves no purpose, then we might as well not even emit the compactification
/// information to the bytecode file, saving a bit more space.
///
/// Note that the type plane has already been compactified if possible.
///
void SlotCalculator::pruneCompactionTable() {
  TypePlane &TyPlane = CompactionTable[Type::TypeTyID];
  for (unsigned ctp = 0, e = CompactionTable.size(); ctp != e; ++ctp)
    if (ctp != Type::TypeTyID && !CompactionTable[ctp].empty()) {
      TypePlane &CPlane = CompactionTable[ctp];
      unsigned GlobalSlot = ctp;
      if (!TyPlane.empty())
        GlobalSlot = getGlobalSlot(TyPlane[ctp]);

      if (GlobalSlot >= Table.size())
        Table.resize(GlobalSlot+1);
      TypePlane &GPlane = Table[GlobalSlot];
      
      unsigned ModLevel = getModuleLevel(ctp);
      unsigned NumFunctionObjs = CPlane.size()-ModLevel;

      // If the maximum index required if all entries in this plane were merged
      // into the global plane is less than 64, go ahead and eliminate the
      // plane.
      bool PrunePlane = GPlane.size() + NumFunctionObjs < 64;

      // If there are no function-local values defined, and the maximum
      // referenced global entry is less than 64, we don't need to compactify.
      if (!PrunePlane && NumFunctionObjs == 0) {
        unsigned MaxIdx = 0;
        for (unsigned i = 0; i != ModLevel; ++i) {
          unsigned Idx = NodeMap[CPlane[i]];
          if (Idx > MaxIdx) MaxIdx = Idx;
        }
        PrunePlane = MaxIdx < 64;
      }

      // Ok, finally, if we decided to prune this plane out of the compaction
      // table, do so now.
      if (PrunePlane) {
        TypePlane OldPlane;
        std::swap(OldPlane, CPlane);

        // Loop over the function local objects, relocating them to the global
        // table plane.
        for (unsigned i = ModLevel, e = OldPlane.size(); i != e; ++i) {
          const Value *V = OldPlane[i];
          CompactionNodeMap.erase(V);
          assert(NodeMap.count(V) == 0 && "Value already in table??");
          getOrCreateSlot(V);
        }

        // For compactified global values, just remove them from the compaction
        // node map.
        for (unsigned i = 0; i != ModLevel; ++i)
          CompactionNodeMap.erase(OldPlane[i]);

        // Update the new modulelevel for this plane.
        assert(ctp < ModuleLevel.size() && "Cannot set modulelevel!");
        ModuleLevel[ctp] = GPlane.size()-NumFunctionObjs;
        assert((int)ModuleLevel[ctp] >= 0 && "Bad computation!");
      }
    }
}


int SlotCalculator::getSlot(const Value *V) const {
  // If there is a CompactionTable active...
  if (!CompactionNodeMap.empty()) {
    std::map<const Value*, unsigned>::const_iterator I =
      CompactionNodeMap.find(V);
    if (I != CompactionNodeMap.end())
      return (int)I->second;
    // Otherwise, if it's not in the compaction table, it must be in a
    // non-compactified plane.
  }

  std::map<const Value*, unsigned>::const_iterator I = NodeMap.find(V);
  if (I != NodeMap.end())
    return (int)I->second;

  // Do not number ConstantPointerRef's at all.  They are an abomination.
  if (const ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(V))
    return getSlot(CPR->getValue());

  return -1;
}


int SlotCalculator::getOrCreateSlot(const Value *V) {
  if (V->getType() == Type::VoidTy) return -1;

  int SlotNo = getSlot(V);        // Check to see if it's already in!
  if (SlotNo != -1) return SlotNo;

  // Do not number ConstantPointerRef's at all.  They are an abomination.
  if (const ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(V))
    return getOrCreateSlot(CPR->getValue());

  if (!isa<GlobalValue>(V))  // Initializers for globals are handled explicitly
    if (const Constant *C = dyn_cast<Constant>(V)) {
      assert(CompactionNodeMap.empty() &&
             "All needed constants should be in the compaction map already!");

      // If we are emitting a bytecode file, do not index the characters that
      // make up constant strings.  We emit constant strings as special
      // entities that don't require their individual characters to be emitted.
      if (!BuildBytecodeInfo || !isa<ConstantArray>(C) ||
          !cast<ConstantArray>(C)->isString()) {
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


int SlotCalculator::insertValue(const Value *D, bool dontIgnore) {
  assert(D && "Can't insert a null value!");
  assert(getSlot(D) == -1 && "Value is already in the table!");

  // If we are building a compaction map, and if this plane is being compacted,
  // insert the value into the compaction map, not into the global map.
  if (!CompactionNodeMap.empty()) {
    if (D->getType() == Type::VoidTy) return -1;  // Do not insert void values
    assert(!isa<Type>(D) && !isa<Constant>(D) && !isa<GlobalValue>(D) &&
           "Types, constants, and globals should be in global SymTab!");

    int Plane = getSlot(D->getType());
    assert(Plane != -1 && CompactionTable.size() > (unsigned)Plane &&
           "Didn't find value type!");
    if (!CompactionTable[Plane].empty())
      return getOrCreateCompactionTableSlot(D);
  }

  // If this node does not contribute to a plane, or if the node has a 
  // name and we don't want names, then ignore the silly node... Note that types
  // do need slot numbers so that we can keep track of where other values land.
  //
  if (!dontIgnore)                               // Don't ignore nonignorables!
    if (D->getType() == Type::VoidTy ||          // Ignore void type nodes
	(!BuildBytecodeInfo &&                   // Ignore named and constants
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
    int ValSlot;
    if (CompactionTable.empty())
      ValSlot = getSlot(Typ);
    else
      ValSlot = getGlobalSlot(Typ);
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
  if (Table[Ty].empty() && BuildBytecodeInfo && hasNullValue(Ty)) {
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
