//===-- llvm/Analysis/ModuleAnalyzer.cpp - Module analysis driver ----------==//
//
// This class provides a nice interface to traverse a module in a predictable
// way.  This is used by the AssemblyWriter, BytecodeWriter, and SlotCalculator
// to do analysis of a module.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ModuleAnalyzer.h"
#include "llvm/ConstantPool.h"
#include "llvm/Method.h"
#include "llvm/Module.h"
#include "llvm/BasicBlock.h"
#include "llvm/DerivedTypes.h"
#include "llvm/ConstPoolVals.h"
#include <map>

// processModule - Driver function to call all of my subclasses virtual methods.
//
bool ModuleAnalyzer::processModule(const Module *M) {
  // Loop over the constant pool, process all of the constants...
  if (processConstPool(M->getConstantPool(), false))
    return true;

  return processMethods(M);
}

inline bool ModuleAnalyzer::handleType(set<const Type *> &TypeSet, 
				       const Type *T) {
  if (!T->isDerivedType()) return false;    // Boring boring types...
  if (TypeSet.count(T) != 0) return false;  // Already found this type...
  TypeSet.insert(T);                        // Add it to the set
  
  // Recursively process interesting types...
  switch (T->getPrimitiveID()) {
  case Type::MethodTyID: {
    const MethodType *MT = (const MethodType *)T;
    if (handleType(TypeSet, MT->getReturnType())) return true;
    const MethodType::ParamTypes &Params = MT->getParamTypes();

    for (MethodType::ParamTypes::const_iterator I = Params.begin();
	 I != Params.end(); ++I)
      if (handleType(TypeSet, *I)) return true;
    break;
  }

  case Type::ArrayTyID:
    if (handleType(TypeSet, ((const ArrayType *)T)->getElementType()))
      return true;
    break;

  case Type::StructTyID: {
    const StructType *ST = (const StructType*)T;
    const StructType::ElementTypes &Elements = ST->getElementTypes();
    for (StructType::ElementTypes::const_iterator I = Elements.begin();
	 I != Elements.end(); ++I)
      if (handleType(TypeSet, *I)) return true;
    break;
  }

  case Type::PointerTyID:
    if (handleType(TypeSet, ((const PointerType *)T)->getValueType()))
      return true;
    break;

  default:
    cerr << "ModuleAnalyzer::handleType, type unknown: '" 
	 << T->getName() << "'\n";
    break;
  }

  return processType(T);
}


bool ModuleAnalyzer::processConstPool(const ConstantPool &CP, bool isMethod) {
  // TypeSet - Keep track of which types have already been processType'ed.  We 
  // don't want to reprocess the same type more than once.
  //
  set<const Type *> TypeSet;

  for (ConstantPool::plane_const_iterator PI = CP.begin(); 
       PI != CP.end(); ++PI) {
    const ConstantPool::PlaneType &Plane = **PI;
    if (Plane.empty()) continue;        // Skip empty type planes...

    if (processConstPoolPlane(CP, Plane, isMethod)) return true;

    for (ConstantPool::PlaneType::const_iterator CI = Plane.begin(); 
	 CI != Plane.end(); CI++) {
      if ((*CI)->getType() == Type::TypeTy)
	if (handleType(TypeSet, ((const ConstPoolType*)(*CI))->getValue())) 
	  return true;
      if (handleType(TypeSet, (*CI)->getType())) return true;

      if (processConstant(*CI)) return true;
    }
  }
  
  if (!isMethod) {
    assert(CP.getParent()->getValueType() == Value::ModuleVal);
    const Module *M = (const Module*)CP.getParent();
    // Process the method types after the constant pool...
    for (Module::MethodListType::const_iterator I = M->getMethodList().begin();
	 I != M->getMethodList().end(); I++) {
      if (handleType(TypeSet, (*I)->getType())) return true;
      if (visitMethod(*I)) return true;
    }
  }
  return false;
}

bool ModuleAnalyzer::processMethods(const Module *M) {
  for (Module::MethodListType::const_iterator I = M->getMethodList().begin();
       I != M->getMethodList().end(); I++)
    if (processMethod(*I)) return true;

  return false;
}

bool ModuleAnalyzer::processMethod(const Method *M) {
  // Loop over the arguments, processing them...
  const Method::ArgumentListType &ArgList = M->getArgumentList();
  for (Method::ArgumentListType::const_iterator AI = ArgList.begin(); 
       AI != ArgList.end(); AI++)
    if (processMethodArgument(*AI)) return true;

  // Loop over the constant pool, adding the constants to the table...
  processConstPool(M->getConstantPool(), true);
  
  // Loop over all the basic blocks, in order...
  Method::BasicBlocksType::const_iterator BBI = M->getBasicBlocks().begin();
  for (; BBI != M->getBasicBlocks().end(); BBI++) 
    if (processBasicBlock(*BBI)) return true;
  return false;
}

bool ModuleAnalyzer::processBasicBlock(const BasicBlock *BB) {
  // Process all of the instructions in the basic block
  BasicBlock::InstListType::const_iterator Inst = BB->getInstList().begin();
  for (; Inst != BB->getInstList().end(); Inst++) {
    if (preProcessInstruction(*Inst) || processInstruction(*Inst)) return true;
  }
  return false;
}

bool ModuleAnalyzer::preProcessInstruction(const Instruction *I) {
  
  return false;
}
