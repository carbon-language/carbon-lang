//===-- SlotCalculator.cpp - Calculate what slots values land in ------------=//
//
// This file implements a useful analysis step to figure out what numbered 
// slots values in a program will land in (keeping track of per plane
// information as required.
//
// This is used primarily for when writing a file to disk, either in bytecode
// or source format.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/SlotCalculator.h"
#include "llvm/ConstantPool.h"
#include "llvm/Method.h"
#include "llvm/Module.h"
#include "llvm/BasicBlock.h"
#include "llvm/ConstPoolVals.h"
#include "llvm/iOther.h"
#include "llvm/DerivedTypes.h"

SlotCalculator::SlotCalculator(const Module *M, bool IgnoreNamed) {
  IgnoreNamedNodes = IgnoreNamed;
  TheModule = M;

  // Preload table... Make sure that all of the primitive types are in the table
  // and that their Primitive ID is equal to their slot #
  //
  for (unsigned i = 0; i < Type::FirstDerivedTyID; ++i) {
    assert(Type::getPrimitiveType((Type::PrimitiveID)i));
    insertVal(Type::getPrimitiveType((Type::PrimitiveID)i));
  }

  if (M == 0) return;   // Empty table...

  bool Result = processModule(M);
  assert(Result == false && "Error in processModule!");
}

SlotCalculator::SlotCalculator(const Method *M, bool IgnoreNamed) {
  IgnoreNamedNodes = IgnoreNamed;
  TheModule = M ? M->getParent() : 0;

  // Preload table... Make sure that all of the primitive types are in the table
  // and that their Primitive ID is equal to their slot #
  //
  for (unsigned i = 0; i < Type::FirstDerivedTyID; ++i) {
    assert(Type::getPrimitiveType((Type::PrimitiveID)i));
    insertVal(Type::getPrimitiveType((Type::PrimitiveID)i));
  }

  if (TheModule == 0) return;   // Empty table...

  bool Result = processModule(TheModule);
  assert(Result == false && "Error in processModule!");

  incorporateMethod(M);
}

void SlotCalculator::incorporateMethod(const Method *M) {
  assert(ModuleLevel.size() == 0 && "Module already incorporated!");

  // Save the Table state before we process the method...
  for (unsigned i = 0; i < Table.size(); ++i) {
    ModuleLevel.push_back(Table[i].size());
  }

  // Process the method to incorporate its values into our table
  processMethod(M);
}

void SlotCalculator::purgeMethod() {
  assert(ModuleLevel.size() != 0 && "Module not incorporated!");
  unsigned NumModuleTypes = ModuleLevel.size();

  // First, remove values from existing type planes
  for (unsigned i = 0; i < NumModuleTypes; ++i) {
    unsigned ModuleSize = ModuleLevel[i];  // Size of plane before method came
    while (Table[i].size() != ModuleSize) {
      NodeMap.erase(NodeMap.find(Table[i].back()));   // Erase from nodemap
      Table[i].pop_back();                            // Shrink plane
    }
  }

  // We don't need this state anymore, free it up.
  ModuleLevel.clear();

  // Next, remove any type planes defined by the method...
  while (NumModuleTypes != Table.size()) {
    TypePlane &Plane = Table.back();
    while (Plane.size()) {
      NodeMap.erase(NodeMap.find(Plane.back()));   // Erase from nodemap
      Plane.pop_back();                            // Shrink plane
    }

    Table.pop_back();                      // Nuke the plane, we don't like it.
  }
}

bool SlotCalculator::processConstant(const ConstPoolVal *CPV) { 
  //cerr << "Inserting constant: '" << CPV->getStrValue() << endl;
  insertVal(CPV);
  return false;
}

// processType - This callback occurs when an derived type is discovered
// at the class level. This activity occurs when processing a constant pool.
//
bool SlotCalculator::processType(const Type *Ty) { 
  //cerr << "processType: " << Ty->getName() << endl;
  // TODO: Don't leak memory!!!  Free this in the dtor!
  insertVal(new ConstPoolType(Ty));
  return false; 
}

bool SlotCalculator::visitMethod(const Method *M) {
  //cerr << "visitMethod: '" << M->getType()->getName() << "'\n";
  insertVal(M);
  return false; 
}

bool SlotCalculator::processMethodArgument(const MethodArgument *MA) {
  insertVal(MA);
  return false;
}

bool SlotCalculator::processBasicBlock(const BasicBlock *BB) {
  insertVal(BB);
  ModuleAnalyzer::processBasicBlock(BB);  // Lets visit the instructions too!
  return false;
}

bool SlotCalculator::processInstruction(const Instruction *I) {
  insertVal(I);
  return false;
}

int SlotCalculator::getValSlot(const Value *D) const {
  map<const Value*, unsigned>::const_iterator I = NodeMap.find(D);
  if (I == NodeMap.end()) return -1;
 
  return (int)I->second;
}

void SlotCalculator::insertVal(const Value *D) {
  if (D == 0) return;

  // If this node does not contribute to a plane, or if the node has a 
  // name and we don't want names, then ignore the silly node...
  //
  if (D->getType() == Type::VoidTy || (IgnoreNamedNodes && D->hasName())) 
    return;

  const Type *Typ = D->getType();
  unsigned Ty = Typ->getPrimitiveID();
  if (Typ->isDerivedType()) {
    int DefSlot = getValSlot(Typ);
    if (DefSlot == -1) {                // Have we already entered this type?
      // This can happen if a type is first seen in an instruction.  For 
      // example, if you say 'malloc uint', this defines a type 'uint*' that
      // may be undefined at this point.
      //
      cerr << "SHOULDN'T HAPPEN Adding Type ba: " << Typ->getName() << endl;
      assert(0 && "Shouldn't this be taken care of by processType!?!?!");
      // Nope... add this to the Type plane now!
      insertVal(Typ);

      DefSlot = getValSlot(Typ);
      assert(DefSlot >= 0 && "Type didn't get inserted correctly!");
    }
    Ty = (unsigned)DefSlot;
  }
  
  if (Table.size() <= Ty)    // Make sure we have the type plane allocated...
    Table.resize(Ty+1, TypePlane());
  
  // Insert node into table and NodeMap...
  NodeMap[D] = Table[Ty].size();

  if (Typ == Type::TypeTy && !D->isType()) {
    // If it's a type constant, add the Type also
      
    // All Type instances should be constant types!
    const ConstPoolType *CPT = (const ConstPoolType*)D->castConstantAsserting();
    int Slot = getValSlot(CPT->getValue());
    if (Slot == -1) {
      // Only add if it's not already here!
      NodeMap[CPT->getValue()] = Table[Ty].size();
    } else if (!CPT->hasName()) {    // If the type has no name...
      NodeMap[D] = (unsigned)Slot;   // Don't readd type, merge.
      return;
    }
  }
  Table[Ty].push_back(D);
}
