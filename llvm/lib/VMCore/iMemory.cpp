//===-- iMemory.cpp - Implement Memory instructions --------------*- C++ -*--=//
//
// This file implements the various memory related classes defined in iMemory.h
//
//===----------------------------------------------------------------------===//

#include "llvm/iMemory.h"
#include "llvm/ConstPoolVals.h"

//===----------------------------------------------------------------------===//
//                        MemAccessInst Implementation
//===----------------------------------------------------------------------===//

// getIndexedType - Returns the type of the element that would be loaded with
// a load instruction with the specified parameters.
//
// A null type is returned if the indices are invalid for the specified 
// pointer type.
//
const Type* MemAccessInst::getIndexedType(const Type *Ptr, 
					  const vector<ConstPoolVal*> &Idx,
					  bool AllowStructLeaf = false) {
  if (!Ptr->isPointerType()) return 0;   // Type isn't a pointer type!
 
  // Get the type pointed to...
  Ptr = ((const PointerType*)Ptr)->getValueType();
  
  if (Ptr->isStructType()) {
    unsigned CurIDX = 0;
    while (Ptr->isStructType()) {
      if (Idx.size() == CurIDX) 
	return AllowStructLeaf ? Ptr : 0;   // Can't load a whole structure!?!?
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

//===----------------------------------------------------------------------===//
//                           LoadInst Implementation
//===----------------------------------------------------------------------===//

LoadInst::LoadInst(Value *Ptr, const vector<ConstPoolVal*> &Idx,
		   const string &Name = "")
  : MemAccessInst(getIndexedType(Ptr->getType(), Idx), Load, Idx, Name) {
  assert(getIndexedType(Ptr->getType(), Idx) && "Load operands invalid!");
  Operands.reserve(1+Idx.size());
  Operands.push_back(Use(Ptr, this));
  
  for (unsigned i = 0, E = Idx.size(); i != E; ++i)
    Operands.push_back(Use(Idx[i], this));
  
}


//===----------------------------------------------------------------------===//
//                           StoreInst Implementation
//===----------------------------------------------------------------------===//

StoreInst::StoreInst(Value *Val, Value *Ptr, const vector<ConstPoolVal*> &Idx,
		     const string &Name = "")
  : MemAccessInst(Type::VoidTy, Store, Idx, Name) {
  assert(getIndexedType(Ptr->getType(), Idx) && "Store operands invalid!");
  
  Operands.reserve(2+Idx.size());
  Operands.push_back(Use(Val, this));
  Operands.push_back(Use(Ptr, this));

  for (unsigned i = 0, E = Idx.size(); i != E; ++i)
    Operands.push_back(Use(Idx[i], this));
}


//===----------------------------------------------------------------------===//
//                       GetElementPtrInst Implementation
//===----------------------------------------------------------------------===//

GetElementPtrInst::GetElementPtrInst(Value *Ptr, 
				     const vector<ConstPoolVal*> &Idx,
				     const string &Name = "")
  : MemAccessInst(PointerType::get(getIndexedType(Ptr->getType(), Idx, true)),
		  GetElementPtr, Idx, Name) {
  assert(getIndexedType(Ptr->getType(), Idx, true) && "gep operands invalid!");
  Operands.reserve(1+Idx.size());
  Operands.push_back(Use(Ptr, this));

  for (unsigned i = 0, E = Idx.size(); i != E; ++i)
    Operands.push_back(Use(Idx[i], this));
}

bool GetElementPtrInst::isStructSelector() const {
  return ((PointerType*)Operands[0]->getType())->getValueType()->isStructType();
}
