//===-- Instructions.cpp - Implement the LLVM instructions ----------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the LLVM instructions...
//
//===----------------------------------------------------------------------===//

#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Support/CallSite.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//                        CallInst Implementation
//===----------------------------------------------------------------------===//

void CallInst::init(Value *Func, const std::vector<Value*> &Params)
{
  Operands.reserve(1+Params.size());
  Operands.push_back(Use(Func, this));

  const FunctionType *FTy = 
    cast<FunctionType>(cast<PointerType>(Func->getType())->getElementType());

  assert((Params.size() == FTy->getNumParams() || 
          (FTy->isVarArg() && Params.size() > FTy->getNumParams())) &&
         "Calling a function with bad signature");
  for (unsigned i = 0; i != Params.size(); i++)
    Operands.push_back(Use(Params[i], this));
}

void CallInst::init(Value *Func, Value *Actual1, Value *Actual2)
{
  Operands.reserve(3);
  Operands.push_back(Use(Func, this));
  
  const FunctionType *MTy = 
    cast<FunctionType>(cast<PointerType>(Func->getType())->getElementType());

  assert((MTy->getNumParams() == 2 ||
          (MTy->isVarArg() && MTy->getNumParams() == 0)) &&
         "Calling a function with bad signature");
  Operands.push_back(Use(Actual1, this));
  Operands.push_back(Use(Actual2, this));
}

void CallInst::init(Value *Func, Value *Actual)
{
  Operands.reserve(2);
  Operands.push_back(Use(Func, this));
  
  const FunctionType *MTy = 
    cast<FunctionType>(cast<PointerType>(Func->getType())->getElementType());

  assert((MTy->getNumParams() == 1 ||
          (MTy->isVarArg() && MTy->getNumParams() == 0)) &&
         "Calling a function with bad signature");
  Operands.push_back(Use(Actual, this));
}

void CallInst::init(Value *Func)
{
  Operands.reserve(1);
  Operands.push_back(Use(Func, this));
  
  const FunctionType *MTy = 
    cast<FunctionType>(cast<PointerType>(Func->getType())->getElementType());

  assert(MTy->getNumParams() == 0 && "Calling a function with bad signature");
}

CallInst::CallInst(Value *Func, const std::vector<Value*> &Params, 
                   const std::string &Name, Instruction *InsertBefore) 
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                 ->getElementType())->getReturnType(),
                Instruction::Call, Name, InsertBefore) {
  init(Func, Params);
}

CallInst::CallInst(Value *Func, const std::vector<Value*> &Params, 
                   const std::string &Name, BasicBlock *InsertAtEnd) 
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                 ->getElementType())->getReturnType(),
                Instruction::Call, Name, InsertAtEnd) {
  init(Func, Params);
}

CallInst::CallInst(Value *Func, Value *Actual1, Value *Actual2,
                   const std::string &Name, Instruction  *InsertBefore)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call, Name, InsertBefore) {
  init(Func, Actual1, Actual2);
}

CallInst::CallInst(Value *Func, Value *Actual1, Value *Actual2,
                   const std::string &Name, BasicBlock  *InsertAtEnd)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call, Name, InsertAtEnd) {
  init(Func, Actual1, Actual2);
}

CallInst::CallInst(Value *Func, Value* Actual, const std::string &Name,
                   Instruction  *InsertBefore)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call, Name, InsertBefore) {
  init(Func, Actual);
}

CallInst::CallInst(Value *Func, Value* Actual, const std::string &Name,
                   BasicBlock  *InsertAtEnd)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call, Name, InsertAtEnd) {
  init(Func, Actual);
}

CallInst::CallInst(Value *Func, const std::string &Name,
                   Instruction *InsertBefore)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call, Name, InsertBefore) {
  init(Func);
}

CallInst::CallInst(Value *Func, const std::string &Name,
                   BasicBlock *InsertAtEnd)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call, Name, InsertAtEnd) {
  init(Func);
}

CallInst::CallInst(const CallInst &CI) 
  : Instruction(CI.getType(), Instruction::Call) {
  Operands.reserve(CI.Operands.size());
  for (unsigned i = 0; i < CI.Operands.size(); ++i)
    Operands.push_back(Use(CI.Operands[i], this));
}


//===----------------------------------------------------------------------===//
//                        InvokeInst Implementation
//===----------------------------------------------------------------------===//

void InvokeInst::init(Value *Fn, BasicBlock *IfNormal, BasicBlock *IfException,
                      const std::vector<Value*> &Params)
{
  Operands.reserve(3+Params.size());
  Operands.push_back(Use(Fn, this));
  Operands.push_back(Use((Value*)IfNormal, this));
  Operands.push_back(Use((Value*)IfException, this));
  const FunctionType *MTy = 
    cast<FunctionType>(cast<PointerType>(Fn->getType())->getElementType());
  
  assert((Params.size() == MTy->getNumParams()) || 
         (MTy->isVarArg() && Params.size() > MTy->getNumParams()) &&
         "Calling a function with bad signature");
  
  for (unsigned i = 0; i < Params.size(); i++)
    Operands.push_back(Use(Params[i], this));
}

InvokeInst::InvokeInst(Value *Fn, BasicBlock *IfNormal,
                       BasicBlock *IfException,
                       const std::vector<Value*> &Params,
                       const std::string &Name, Instruction *InsertBefore)
  : TerminatorInst(cast<FunctionType>(cast<PointerType>(Fn->getType())
                                    ->getElementType())->getReturnType(),
                   Instruction::Invoke, Name, InsertBefore) {
  init(Fn, IfNormal, IfException, Params);
}

InvokeInst::InvokeInst(Value *Fn, BasicBlock *IfNormal,
                       BasicBlock *IfException,
                       const std::vector<Value*> &Params,
                       const std::string &Name, BasicBlock *InsertAtEnd)
  : TerminatorInst(cast<FunctionType>(cast<PointerType>(Fn->getType())
                                    ->getElementType())->getReturnType(),
                   Instruction::Invoke, Name, InsertAtEnd) {
  init(Fn, IfNormal, IfException, Params);
}

InvokeInst::InvokeInst(const InvokeInst &CI) 
  : TerminatorInst(CI.getType(), Instruction::Invoke) {
  Operands.reserve(CI.Operands.size());
  for (unsigned i = 0; i < CI.Operands.size(); ++i)
    Operands.push_back(Use(CI.Operands[i], this));
}

//===----------------------------------------------------------------------===//
//                        ReturnInst Implementation
//===----------------------------------------------------------------------===//

void ReturnInst::init(Value* RetVal) {
  if (RetVal && RetVal->getType() != Type::VoidTy) {
    assert(!isa<BasicBlock>(RetVal) && 
           "Cannot return basic block.  Probably using the incorrect ctor");
    Operands.reserve(1);
    Operands.push_back(Use(RetVal, this));
  }
}

// Out-of-line ReturnInst method, put here so the C++ compiler can choose to
// emit the vtable for the class in this translation unit.
void ReturnInst::setSuccessor(unsigned idx, BasicBlock *NewSucc) {
  assert(0 && "ReturnInst has no successors!");
}

//===----------------------------------------------------------------------===//
//                        UnwindInst Implementation
//===----------------------------------------------------------------------===//

// Likewise for UnwindInst
void UnwindInst::setSuccessor(unsigned idx, BasicBlock *NewSucc) {
  assert(0 && "UnwindInst has no successors!");
}

//===----------------------------------------------------------------------===//
//                      UnreachableInst Implementation
//===----------------------------------------------------------------------===//

void UnreachableInst::setSuccessor(unsigned idx, BasicBlock *NewSucc) {
  assert(0 && "UnreachableInst has no successors!");
}

//===----------------------------------------------------------------------===//
//                        BranchInst Implementation
//===----------------------------------------------------------------------===//

void BranchInst::init(BasicBlock *IfTrue)
{
  assert(IfTrue != 0 && "Branch destination may not be null!");
  Operands.reserve(1);
  Operands.push_back(Use(IfTrue, this));
}

void BranchInst::init(BasicBlock *IfTrue, BasicBlock *IfFalse, Value *Cond)
{
  assert(IfTrue && IfFalse && Cond &&
         "Branch destinations and condition may not be null!");
  assert(Cond && Cond->getType() == Type::BoolTy && 
         "May only branch on boolean predicates!");
  Operands.reserve(3);
  Operands.push_back(Use(IfTrue, this));
  Operands.push_back(Use(IfFalse, this));
  Operands.push_back(Use(Cond, this));
}

BranchInst::BranchInst(const BranchInst &BI) : TerminatorInst(Instruction::Br) {
  Operands.reserve(BI.Operands.size());
  Operands.push_back(Use(BI.Operands[0], this));
  if (BI.Operands.size() != 1) {
    assert(BI.Operands.size() == 3 && "BR can have 1 or 3 operands!");
    Operands.push_back(Use(BI.Operands[1], this));
    Operands.push_back(Use(BI.Operands[2], this));
  }
}

//===----------------------------------------------------------------------===//
//                        AllocationInst Implementation
//===----------------------------------------------------------------------===//

void AllocationInst::init(const Type *Ty, Value *ArraySize, unsigned iTy) {
  assert(Ty != Type::VoidTy && "Cannot allocate void elements!");
  // ArraySize defaults to 1.
  if (!ArraySize) ArraySize = ConstantUInt::get(Type::UIntTy, 1);

  Operands.reserve(1);
  assert(ArraySize->getType() == Type::UIntTy &&
         "Malloc/Allocation array size != UIntTy!");

  Operands.push_back(Use(ArraySize, this));
}

AllocationInst::AllocationInst(const Type *Ty, Value *ArraySize, unsigned iTy, 
                               const std::string &Name,
                               Instruction *InsertBefore)
  : Instruction(PointerType::get(Ty), iTy, Name, InsertBefore) {
  init(Ty, ArraySize, iTy);
}

AllocationInst::AllocationInst(const Type *Ty, Value *ArraySize, unsigned iTy, 
                               const std::string &Name,
                               BasicBlock *InsertAtEnd)
  : Instruction(PointerType::get(Ty), iTy, Name, InsertAtEnd) {
  init(Ty, ArraySize, iTy);
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

void FreeInst::init(Value *Ptr)
{
  assert(Ptr && isa<PointerType>(Ptr->getType()) && "Can't free nonpointer!");
  Operands.reserve(1);
  Operands.push_back(Use(Ptr, this));
}

FreeInst::FreeInst(Value *Ptr, Instruction *InsertBefore)
  : Instruction(Type::VoidTy, Free, "", InsertBefore) {
  init(Ptr);
}

FreeInst::FreeInst(Value *Ptr, BasicBlock *InsertAtEnd)
  : Instruction(Type::VoidTy, Free, "", InsertAtEnd) {
  init(Ptr);
}


//===----------------------------------------------------------------------===//
//                           LoadInst Implementation
//===----------------------------------------------------------------------===//

void LoadInst::init(Value *Ptr) {
  assert(Ptr && isa<PointerType>(Ptr->getType()) && 
         "Ptr must have pointer type.");
  Operands.reserve(1);
  Operands.push_back(Use(Ptr, this));
}

LoadInst::LoadInst(Value *Ptr, const std::string &Name, Instruction *InsertBef)
  : Instruction(cast<PointerType>(Ptr->getType())->getElementType(),
                Load, Name, InsertBef), Volatile(false) {
  init(Ptr);
}

LoadInst::LoadInst(Value *Ptr, const std::string &Name, BasicBlock *InsertAE)
  : Instruction(cast<PointerType>(Ptr->getType())->getElementType(),
                Load, Name, InsertAE), Volatile(false) {
  init(Ptr);
}

LoadInst::LoadInst(Value *Ptr, const std::string &Name, bool isVolatile,
                   Instruction *InsertBef)
  : Instruction(cast<PointerType>(Ptr->getType())->getElementType(),
                Load, Name, InsertBef), Volatile(isVolatile) {
  init(Ptr);
}

LoadInst::LoadInst(Value *Ptr, const std::string &Name, bool isVolatile,
                   BasicBlock *InsertAE)
  : Instruction(cast<PointerType>(Ptr->getType())->getElementType(),
                Load, Name, InsertAE), Volatile(isVolatile) {
  init(Ptr);
}


//===----------------------------------------------------------------------===//
//                           StoreInst Implementation
//===----------------------------------------------------------------------===//

StoreInst::StoreInst(Value *Val, Value *Ptr, Instruction *InsertBefore)
  : Instruction(Type::VoidTy, Store, "", InsertBefore), Volatile(false) {
  init(Val, Ptr);
}

StoreInst::StoreInst(Value *Val, Value *Ptr, BasicBlock *InsertAtEnd)
  : Instruction(Type::VoidTy, Store, "", InsertAtEnd), Volatile(false) {
  init(Val, Ptr);
}

StoreInst::StoreInst(Value *Val, Value *Ptr, bool isVolatile, 
                     Instruction *InsertBefore)
  : Instruction(Type::VoidTy, Store, "", InsertBefore), Volatile(isVolatile) {
  init(Val, Ptr);
}

StoreInst::StoreInst(Value *Val, Value *Ptr, bool isVolatile, 
                     BasicBlock *InsertAtEnd)
  : Instruction(Type::VoidTy, Store, "", InsertAtEnd), Volatile(isVolatile) {
  init(Val, Ptr);
}

void StoreInst::init(Value *Val, Value *Ptr) {
  assert(isa<PointerType>(Ptr->getType()) && "Ptr must have pointer type!");
  assert(Val->getType() == cast<PointerType>(Ptr->getType())->getElementType()
         && "Ptr must be a pointer to Val type!");

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

void GetElementPtrInst::init(Value *Ptr, const std::vector<Value*> &Idx)
{
  Operands.reserve(1+Idx.size());
  Operands.push_back(Use(Ptr, this));

  for (unsigned i = 0, E = Idx.size(); i != E; ++i)
    Operands.push_back(Use(Idx[i], this));
}

void GetElementPtrInst::init(Value *Ptr, Value *Idx0, Value *Idx1) {
  Operands.reserve(3);
  Operands.push_back(Use(Ptr, this));
  Operands.push_back(Use(Idx0, this));
  Operands.push_back(Use(Idx1, this));
}

GetElementPtrInst::GetElementPtrInst(Value *Ptr, const std::vector<Value*> &Idx,
				     const std::string &Name, Instruction *InBe)
  : Instruction(PointerType::get(checkType(getIndexedType(Ptr->getType(),
                                                          Idx, true))),
                GetElementPtr, Name, InBe) {
  init(Ptr, Idx);
}

GetElementPtrInst::GetElementPtrInst(Value *Ptr, const std::vector<Value*> &Idx,
				     const std::string &Name, BasicBlock *IAE)
  : Instruction(PointerType::get(checkType(getIndexedType(Ptr->getType(),
                                                          Idx, true))),
                GetElementPtr, Name, IAE) {
  init(Ptr, Idx);
}

GetElementPtrInst::GetElementPtrInst(Value *Ptr, Value *Idx0, Value *Idx1,
                                     const std::string &Name, Instruction *InBe)
  : Instruction(PointerType::get(checkType(getIndexedType(Ptr->getType(),
                                                          Idx0, Idx1, true))),
                GetElementPtr, Name, InBe) {
  init(Ptr, Idx0, Idx1);
}

GetElementPtrInst::GetElementPtrInst(Value *Ptr, Value *Idx0, Value *Idx1,
		                     const std::string &Name, BasicBlock *IAE)
  : Instruction(PointerType::get(checkType(getIndexedType(Ptr->getType(),
                                                          Idx0, Idx1, true))),
                GetElementPtr, Name, IAE) {
  init(Ptr, Idx0, Idx1);
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
  if (Idx.empty())
    if (AllowCompositeLeaf ||
        cast<PointerType>(Ptr)->getElementType()->isFirstClassType())
      return cast<PointerType>(Ptr)->getElementType();
    else
      return 0;
 
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

    // If the new type forwards to another type, then it is in the middle
    // of being refined to another type (and hence, may have dropped all
    // references to what it was using before).  So, use the new forwarded
    // type.
    if (const Type * Ty = Ptr->getForwardedType()) {
      Ptr = Ty;
    }
  }
  return CurIdx == Idx.size() ? Ptr : 0;
}

const Type* GetElementPtrInst::getIndexedType(const Type *Ptr, 
                                              Value *Idx0, Value *Idx1,
                                              bool AllowCompositeLeaf) {
  const PointerType *PTy = dyn_cast<PointerType>(Ptr);
  if (!PTy) return 0;   // Type isn't a pointer type!

  // Check the pointer index.
  if (!PTy->indexValid(Idx0)) return 0;

  const CompositeType *CT = dyn_cast<CompositeType>(PTy->getElementType());
  if (!CT || !CT->indexValid(Idx1)) return 0;

  const Type *ElTy = CT->getTypeAtIndex(Idx1);
  if (AllowCompositeLeaf || ElTy->isFirstClassType())
    return ElTy;
  return 0;
}

//===----------------------------------------------------------------------===//
//                             BinaryOperator Class
//===----------------------------------------------------------------------===//

void BinaryOperator::init(BinaryOps iType, Value *S1, Value *S2)
{
  Operands.reserve(2);
  Operands.push_back(Use(S1, this));
  Operands.push_back(Use(S2, this));
  assert(S1 && S2 && S1->getType() == S2->getType());

#ifndef NDEBUG
  switch (iType) {
  case Add: case Sub:
  case Mul: case Div:
  case Rem:
    assert(getType() == S1->getType() &&
           "Arithmetic operation should return same type as operands!");
    assert((getType()->isInteger() || 
            getType()->isFloatingPoint() || 
            isa<PackedType>(getType()) ) && 
          "Tried to create an arithmetic operation on a non-arithmetic type!");
    break;
  case And: case Or:
  case Xor:
    assert(getType() == S1->getType() &&
           "Logical operation should return same type as operands!");
    assert(getType()->isIntegral() &&
           "Tried to create a logical operation on a non-integral type!");
    break;
  case SetLT: case SetGT: case SetLE:
  case SetGE: case SetEQ: case SetNE:
    assert(getType() == Type::BoolTy && "Setcc must return bool!");
  default:
    break;
  }
#endif
}

BinaryOperator *BinaryOperator::create(BinaryOps Op, Value *S1, Value *S2,
				       const std::string &Name,
                                       Instruction *InsertBefore) {
  assert(S1->getType() == S2->getType() &&
         "Cannot create binary operator with two operands of differing type!");
  switch (Op) {
  // Binary comparison operators...
  case SetLT: case SetGT: case SetLE:
  case SetGE: case SetEQ: case SetNE:
    return new SetCondInst(Op, S1, S2, Name, InsertBefore);

  default:
    return new BinaryOperator(Op, S1, S2, S1->getType(), Name, InsertBefore);
  }
}

BinaryOperator *BinaryOperator::create(BinaryOps Op, Value *S1, Value *S2,
				       const std::string &Name,
                                       BasicBlock *InsertAtEnd) {
  BinaryOperator *Res = create(Op, S1, S2, Name);
  InsertAtEnd->getInstList().push_back(Res);
  return Res;
}

BinaryOperator *BinaryOperator::createNeg(Value *Op, const std::string &Name,
                                          Instruction *InsertBefore) {
  if (!Op->getType()->isFloatingPoint())
    return new BinaryOperator(Instruction::Sub,
                              Constant::getNullValue(Op->getType()), Op,
                              Op->getType(), Name, InsertBefore);
  else
    return new BinaryOperator(Instruction::Sub,
                              ConstantFP::get(Op->getType(), -0.0), Op,
                              Op->getType(), Name, InsertBefore);
}

BinaryOperator *BinaryOperator::createNeg(Value *Op, const std::string &Name,
                                          BasicBlock *InsertAtEnd) {
  if (!Op->getType()->isFloatingPoint())
    return new BinaryOperator(Instruction::Sub,
                              Constant::getNullValue(Op->getType()), Op,
                              Op->getType(), Name, InsertAtEnd);
  else
    return new BinaryOperator(Instruction::Sub,
                              ConstantFP::get(Op->getType(), -0.0), Op,
                              Op->getType(), Name, InsertAtEnd);
}

BinaryOperator *BinaryOperator::createNot(Value *Op, const std::string &Name,
                                          Instruction *InsertBefore) {
  return new BinaryOperator(Instruction::Xor, Op,
                            ConstantIntegral::getAllOnesValue(Op->getType()),
                            Op->getType(), Name, InsertBefore);
}

BinaryOperator *BinaryOperator::createNot(Value *Op, const std::string &Name,
                                          BasicBlock *InsertAtEnd) {
  return new BinaryOperator(Instruction::Xor, Op,
                            ConstantIntegral::getAllOnesValue(Op->getType()),
                            Op->getType(), Name, InsertAtEnd);
}


// isConstantAllOnes - Helper function for several functions below
static inline bool isConstantAllOnes(const Value *V) {
  return isa<ConstantIntegral>(V) &&cast<ConstantIntegral>(V)->isAllOnesValue();
}

bool BinaryOperator::isNeg(const Value *V) {
  if (const BinaryOperator *Bop = dyn_cast<BinaryOperator>(V))
    if (Bop->getOpcode() == Instruction::Sub)
      if (!V->getType()->isFloatingPoint())
        return Bop->getOperand(0) == Constant::getNullValue(Bop->getType());
      else
        return Bop->getOperand(0) == ConstantFP::get(Bop->getType(), -0.0);
  return false;
}

bool BinaryOperator::isNot(const Value *V) {
  if (const BinaryOperator *Bop = dyn_cast<BinaryOperator>(V))
    return (Bop->getOpcode() == Instruction::Xor &&
            (isConstantAllOnes(Bop->getOperand(1)) ||
             isConstantAllOnes(Bop->getOperand(0))));
  return false;
}

Value *BinaryOperator::getNegArgument(BinaryOperator *Bop) {
  assert(isNeg(Bop) && "getNegArgument from non-'neg' instruction!");
  return Bop->getOperand(1);
}

const Value *BinaryOperator::getNegArgument(const BinaryOperator *Bop) {
  return getNegArgument((BinaryOperator*)Bop);
}

Value *BinaryOperator::getNotArgument(BinaryOperator *Bop) {
  assert(isNot(Bop) && "getNotArgument on non-'not' instruction!");
  Value *Op0 = Bop->getOperand(0);
  Value *Op1 = Bop->getOperand(1);
  if (isConstantAllOnes(Op0)) return Op1;

  assert(isConstantAllOnes(Op1));
  return Op0;
}

const Value *BinaryOperator::getNotArgument(const BinaryOperator *Bop) {
  return getNotArgument((BinaryOperator*)Bop);
}


// swapOperands - Exchange the two operands to this instruction.  This
// instruction is safe to use on any binary instruction and does not
// modify the semantics of the instruction.  If the instruction is
// order dependent (SetLT f.e.) the opcode is changed.
//
bool BinaryOperator::swapOperands() {
  if (isCommutative())
    ;  // If the instruction is commutative, it is safe to swap the operands
  else if (SetCondInst *SCI = dyn_cast<SetCondInst>(this))
    /// FIXME: SetCC instructions shouldn't all have different opcodes.
    setOpcode(SCI->getSwappedCondition());
  else
    return true;   // Can't commute operands

  std::swap(Operands[0], Operands[1]);
  return false;
}


//===----------------------------------------------------------------------===//
//                             SetCondInst Class
//===----------------------------------------------------------------------===//

SetCondInst::SetCondInst(BinaryOps Opcode, Value *S1, Value *S2, 
                         const std::string &Name, Instruction *InsertBefore)
  : BinaryOperator(Opcode, S1, S2, Type::BoolTy, Name, InsertBefore) {

  // Make sure it's a valid type... getInverseCondition will assert out if not.
  assert(getInverseCondition(Opcode));
}

SetCondInst::SetCondInst(BinaryOps Opcode, Value *S1, Value *S2, 
                         const std::string &Name, BasicBlock *InsertAtEnd)
  : BinaryOperator(Opcode, S1, S2, Type::BoolTy, Name, InsertAtEnd) {

  // Make sure it's a valid type... getInverseCondition will assert out if not.
  assert(getInverseCondition(Opcode));
}

// getInverseCondition - Return the inverse of the current condition opcode.
// For example seteq -> setne, setgt -> setle, setlt -> setge, etc...
//
Instruction::BinaryOps SetCondInst::getInverseCondition(BinaryOps Opcode) {
  switch (Opcode) {
  default:
    assert(0 && "Unknown setcc opcode!");
  case SetEQ: return SetNE;
  case SetNE: return SetEQ;
  case SetGT: return SetLE;
  case SetLT: return SetGE;
  case SetGE: return SetLT;
  case SetLE: return SetGT;
  }
}

// getSwappedCondition - Return the condition opcode that would be the result
// of exchanging the two operands of the setcc instruction without changing
// the result produced.  Thus, seteq->seteq, setle->setge, setlt->setgt, etc.
//
Instruction::BinaryOps SetCondInst::getSwappedCondition(BinaryOps Opcode) {
  switch (Opcode) {
  default: assert(0 && "Unknown setcc instruction!");
  case SetEQ: case SetNE: return Opcode;
  case SetGT: return SetLT;
  case SetLT: return SetGT;
  case SetGE: return SetLE;
  case SetLE: return SetGE;
  }
}

//===----------------------------------------------------------------------===//
//                        SwitchInst Implementation
//===----------------------------------------------------------------------===//

void SwitchInst::init(Value *Value, BasicBlock *Default)
{
  assert(Value && Default);
  Operands.push_back(Use(Value, this));
  Operands.push_back(Use(Default, this));
}

SwitchInst::SwitchInst(const SwitchInst &SI) 
  : TerminatorInst(Instruction::Switch) {
  Operands.reserve(SI.Operands.size());

  for (unsigned i = 0, E = SI.Operands.size(); i != E; i+=2) {
    Operands.push_back(Use(SI.Operands[i], this));
    Operands.push_back(Use(SI.Operands[i+1], this));
  }
}

/// addCase - Add an entry to the switch instruction...
///
void SwitchInst::addCase(Constant *OnVal, BasicBlock *Dest) {
  Operands.push_back(Use((Value*)OnVal, this));
  Operands.push_back(Use((Value*)Dest, this));
}

/// removeCase - This method removes the specified successor from the switch
/// instruction.  Note that this cannot be used to remove the default
/// destination (successor #0).
///
void SwitchInst::removeCase(unsigned idx) {
  assert(idx != 0 && "Cannot remove the default case!");
  assert(idx*2 < Operands.size() && "Successor index out of range!!!");
  Operands.erase(Operands.begin()+idx*2, Operands.begin()+(idx+1)*2);  
}


// Define these methods here so vtables don't get emitted into every translation
// unit that uses these classes.

GetElementPtrInst *GetElementPtrInst::clone() const {
  return new GetElementPtrInst(*this);
}

BinaryOperator *BinaryOperator::clone() const {
  return create(getOpcode(), Operands[0], Operands[1]);
}

MallocInst *MallocInst::clone() const { return new MallocInst(*this); }
AllocaInst *AllocaInst::clone() const { return new AllocaInst(*this); }
FreeInst   *FreeInst::clone()   const { return new FreeInst(Operands[0]); }
LoadInst   *LoadInst::clone()   const { return new LoadInst(*this); }
StoreInst  *StoreInst::clone()  const { return new StoreInst(*this); }
CastInst   *CastInst::clone()   const { return new CastInst(*this); }
CallInst   *CallInst::clone()   const { return new CallInst(*this); }
ShiftInst  *ShiftInst::clone()  const { return new ShiftInst(*this); }
SelectInst *SelectInst::clone() const { return new SelectInst(*this); }
VANextInst *VANextInst::clone() const { return new VANextInst(*this); }
VAArgInst  *VAArgInst::clone()  const { return new VAArgInst(*this); }
PHINode    *PHINode::clone()    const { return new PHINode(*this); }
ReturnInst *ReturnInst::clone() const { return new ReturnInst(*this); }
BranchInst *BranchInst::clone() const { return new BranchInst(*this); }
SwitchInst *SwitchInst::clone() const { return new SwitchInst(*this); }
InvokeInst *InvokeInst::clone() const { return new InvokeInst(*this); }
UnwindInst *UnwindInst::clone() const { return new UnwindInst(); }
UnreachableInst *UnreachableInst::clone() const { return new UnreachableInst();}
