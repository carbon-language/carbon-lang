//===-- iMemory.cpp - Implement Memory instructions --------------*- C++ -*--=//
//
// This file implements the various memory related classes defined in iMemory.h
//
//===----------------------------------------------------------------------===//

#include "llvm/iMemory.h"

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
					  const vector<Value*> &Idx,
					  bool AllowCompositeLeaf = false) {
  if (!Ptr->isPointerType()) return 0;   // Type isn't a pointer type!
 
  // Get the type pointed to...
  Ptr = cast<PointerType>(Ptr)->getElementType();
  
  unsigned CurIDX = 0;
  while (const CompositeType *ST = dyn_cast<CompositeType>(Ptr)) {
    if (Idx.size() == CurIDX)
      return AllowCompositeLeaf ? Ptr : 0;   // Can't load a whole structure!?!?

    Value *Index = Idx[CurIDX++];
    if (!ST->indexValid(Index)) return 0;
    Ptr = ST->getTypeAtIndex(Index);
  }
  return CurIDX == Idx.size() ? Ptr : 0;
}


#if 1
#include "llvm/ConstantVals.h"
const vector<Constant*> MemAccessInst::getIndicesBROKEN() const {
  cerr << "MemAccessInst::getIndices() does not do what you want it to.  Talk"
       << " to Chris about this.  We can phase it out after the paper.\n";

  vector<Constant*> RetVal;

  // THIS CODE WILL FAIL IF A NON CONSTANT INDEX IS USED AS AN ARRAY INDEX
  // THIS IS WHY YOU SHOULD NOT USE THIS FUNCTION ANY MORE!!!
  for (unsigned i = getFirstIndexOperandNumber(); i < getNumOperands(); ++i)
    RetVal.push_back(cast<Constant>(getOperand(i)));

  return RetVal;
}
#endif

//===----------------------------------------------------------------------===//
//                           LoadInst Implementation
//===----------------------------------------------------------------------===//

LoadInst::LoadInst(Value *Ptr, const vector<Value*> &Idx,
		   const string &Name = "")
  : MemAccessInst(getIndexedType(Ptr->getType(), Idx), Load, Name) {
  assert(getIndexedType(Ptr->getType(), Idx) && "Load operands invalid!");
  Operands.reserve(1+Idx.size());
  Operands.push_back(Use(Ptr, this));
  
  for (unsigned i = 0, E = Idx.size(); i != E; ++i)
    Operands.push_back(Use(Idx[i], this));
  
}

LoadInst::LoadInst(Value *Ptr, const string &Name = "")
  : MemAccessInst(cast<PointerType>(Ptr->getType())->getElementType(),
                  Load, Name) {
  Operands.reserve(1);
  Operands.push_back(Use(Ptr, this));
}


//===----------------------------------------------------------------------===//
//                           StoreInst Implementation
//===----------------------------------------------------------------------===//

StoreInst::StoreInst(Value *Val, Value *Ptr, const vector<Value*> &Idx,
		     const string &Name = "")
  : MemAccessInst(Type::VoidTy, Store, Name) {
  assert(getIndexedType(Ptr->getType(), Idx) && "Store operands invalid!");
  
  Operands.reserve(2+Idx.size());
  Operands.push_back(Use(Val, this));
  Operands.push_back(Use(Ptr, this));

  for (unsigned i = 0, E = Idx.size(); i != E; ++i)
    Operands.push_back(Use(Idx[i], this));
}

StoreInst::StoreInst(Value *Val, Value *Ptr, const string &Name = "")
  : MemAccessInst(Type::VoidTy, Store, Name) {
  
  Operands.reserve(2);
  Operands.push_back(Use(Val, this));
  Operands.push_back(Use(Ptr, this));
}


//===----------------------------------------------------------------------===//
//                       GetElementPtrInst Implementation
//===----------------------------------------------------------------------===//

GetElementPtrInst::GetElementPtrInst(Value *Ptr, const vector<Value*> &Idx,
				     const string &Name = "")
  : MemAccessInst(PointerType::get(getIndexedType(Ptr->getType(), Idx, true)),
		  GetElementPtr, Name) {
  assert(getIndexedType(Ptr->getType(), Idx, true) && "gep operands invalid!");
  Operands.reserve(1+Idx.size());
  Operands.push_back(Use(Ptr, this));

  for (unsigned i = 0, E = Idx.size(); i != E; ++i)
    Operands.push_back(Use(Idx[i], this));
}

bool GetElementPtrInst::isStructSelector() const {
  return ((PointerType*)Operands[0]->getType())->getElementType()->isStructType();
}
