//===-- iMemory.cpp - Implement Memory instructions --------------*- C++ -*--=//
//
// This file implements the various memory related classes defined in iMemory.h
//
//===----------------------------------------------------------------------===//

#include "llvm/iMemory.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"

AllocationInst::AllocationInst(const Type *Ty, Value *ArraySize, unsigned iTy, 
                               const std::string &Name, Instruction *InsertBef)
  : Instruction(PointerType::get(Ty), iTy, Name, InsertBef) {

  // ArraySize defaults to 1.
  if (!ArraySize) ArraySize = ConstantUInt::get(Type::UIntTy, 1);

  Operands.reserve(1);
  assert(ArraySize->getType() == Type::UIntTy &&
         "Malloc/Allocation array size != UIntTy!");

  Operands.push_back(Use(ArraySize, this));
}

bool AllocationInst::isArrayAllocation() const {
  return getOperand(0) != ConstantUInt::get(Type::UIntTy, 1);
}

const Type *AllocationInst::getAllocatedType() const {
  return getType()->getElementType();
}

AllocaInst::AllocaInst(const AllocaInst &AI)
  : AllocationInst(AI.getType()->getElementType(), (Value*)AI.getOperand(0),
                   Instruction::Alloca) {
}

MallocInst::MallocInst(const MallocInst &MI)
  : AllocationInst(MI.getType()->getElementType(), (Value*)MI.getOperand(0),
                   Instruction::Malloc) {
}

//===----------------------------------------------------------------------===//
//                             FreeInst Implementation
//===----------------------------------------------------------------------===//

FreeInst::FreeInst(Value *Ptr, Instruction *InsertBefore)
  : Instruction(Type::VoidTy, Free, "", InsertBefore) {
  assert(isa<PointerType>(Ptr->getType()) && "Can't free nonpointer!");
  Operands.reserve(1);
  Operands.push_back(Use(Ptr, this));
}


//===----------------------------------------------------------------------===//
//                           LoadInst Implementation
//===----------------------------------------------------------------------===//

LoadInst::LoadInst(Value *Ptr, const std::string &Name, Instruction *InsertBef)
  : Instruction(cast<PointerType>(Ptr->getType())->getElementType(),
                Load, Name, InsertBef) {
  Operands.reserve(1);
  Operands.push_back(Use(Ptr, this));
}


//===----------------------------------------------------------------------===//
//                           StoreInst Implementation
//===----------------------------------------------------------------------===//

StoreInst::StoreInst(Value *Val, Value *Ptr, Instruction *InsertBefore)
  : Instruction(Type::VoidTy, Store, "", InsertBefore) {
  
  Operands.reserve(2);
  Operands.push_back(Use(Val, this));
  Operands.push_back(Use(Ptr, this));
}


//===----------------------------------------------------------------------===//
//                       GetElementPtrInst Implementation
//===----------------------------------------------------------------------===//

// checkType - Simple wrapper function to give a better assertion failure
// message on bad indexes for a gep instruction.
//
static inline const Type *checkType(const Type *Ty) {
  assert(Ty && "Invalid indices for type!");
  return Ty;
}

GetElementPtrInst::GetElementPtrInst(Value *Ptr, const std::vector<Value*> &Idx,
				     const std::string &Name, Instruction *InBe)
  : Instruction(PointerType::get(checkType(getIndexedType(Ptr->getType(),
                                                            Idx, true))),
		  GetElementPtr, Name, InBe) {
  assert(getIndexedType(Ptr->getType(), Idx, true) && "gep operands invalid!");
  Operands.reserve(1+Idx.size());
  Operands.push_back(Use(Ptr, this));

  for (unsigned i = 0, E = Idx.size(); i != E; ++i)
    Operands.push_back(Use(Idx[i], this));
}

// getIndexedType - Returns the type of the element that would be loaded with
// a load instruction with the specified parameters.
//
// A null type is returned if the indices are invalid for the specified 
// pointer type.
//
const Type* GetElementPtrInst::getIndexedType(const Type *Ptr, 
                                              const std::vector<Value*> &Idx,
                                              bool AllowCompositeLeaf) {
  if (!isa<PointerType>(Ptr)) return 0;   // Type isn't a pointer type!

  // Handle the special case of the empty set index set...
  if (Idx.empty()) return cast<PointerType>(Ptr)->getElementType();
 
  unsigned CurIdx = 0;
  while (const CompositeType *CT = dyn_cast<CompositeType>(Ptr)) {
    if (Idx.size() == CurIdx) {
      if (AllowCompositeLeaf || CT->isFirstClassType()) return Ptr;
      return 0;   // Can't load a whole structure or array!?!?
    }

    Value *Index = Idx[CurIdx++];
    if (isa<PointerType>(CT) && CurIdx != 1)
      return 0;  // Can only index into pointer types at the first index!
    if (!CT->indexValid(Index)) return 0;
    Ptr = CT->getTypeAtIndex(Index);
  }
  return CurIdx == Idx.size() ? Ptr : 0;
}
