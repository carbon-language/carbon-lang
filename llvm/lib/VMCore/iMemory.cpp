//===-- iMemory.cpp - Implement Memory instructions --------------*- C++ -*--=//
//
// This file implements the various memory related classes defined in iMemory.h
//
//===----------------------------------------------------------------------===//

#include "llvm/iMemory.h"
#include "llvm/ConstPoolVals.h"

const Type *LoadInst::getIndexedType(const Type *Ptr, 
				     const vector<ConstPoolVal*> &Idx) {
  if (!Ptr->isPointerType()) return 0;   // Type isn't a pointer type!
 
  // Get the type pointed to...
  Ptr = ((const PointerType*)Ptr)->getValueType();

  if (Ptr->isStructType()) {
    unsigned CurIDX = 0;
    while (Ptr->isStructType()) {
      if (Idx.size() == CurIDX) return 0;       // Can't load a whole structure!
      if (Idx[CurIDX]->getType() != Type::UByteTy) return 0; // Illegal idx
      unsigned NextIdx = ((ConstPoolUInt*)Idx[CurIDX++])->getValue();
      
      const StructType *ST = (const StructType *)Ptr;
      Ptr = ST->getElementTypes()[NextIdx];
    }
    return Ptr;
  } else if (Ptr->isArrayType()) {
    assert(0 && "Loading from arrays not implemented yet!");
  } else {
    return (Idx.size() == 0) ? Ptr : 0;  // Load directly through ptr
  }
}


LoadInst::LoadInst(Value *Ptr, const vector<ConstPoolVal*> &Idx,
		   const string &Name = "")
  : Instruction(getIndexedType(Ptr->getType(), Idx), Load, Name) {
  assert(getIndexedType(Ptr->getType(), Idx) && "Load operands invalid!");
  assert(Ptr->getType()->isPointerType() && "Can't free nonpointer!");
  Operands.reserve(1+Idx.size());
  Operands.push_back(Use(Ptr, this));

  for (unsigned i = 0, E = Idx.size(); i != E; ++i)
    Operands.push_back(Use(Idx[i], this));
}

