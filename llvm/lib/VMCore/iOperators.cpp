//===-- iOperators.cpp - Implement binary Operators ------------*- C++ -*--===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the nontrivial binary operator instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/iOperators.h"
#include "llvm/Type.h"
#include "llvm/Constants.h"
#include "llvm/BasicBlock.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//                             BinaryOperator Class
//===----------------------------------------------------------------------===//

BinaryOperator::BinaryOperator(BinaryOps iType, Value *S1, Value *S2, 
                               const Type *Ty, const std::string &Name,
                               Instruction *InsertBefore)
  : Instruction(Ty, iType, Name, InsertBefore) {

  Operands.reserve(2);
  Operands.push_back(Use(S1, this));
  Operands.push_back(Use(S2, this));
  assert(S1 && S2 && S1->getType() == S2->getType());

#ifndef NDEBUG
  switch (iType) {
  case Add: case Sub:
  case Mul: case Div:
  case Rem:
    assert(Ty == S1->getType() &&
           "Arithmetic operation should return same type as operands!");
    assert((Ty->isInteger() || Ty->isFloatingPoint()) && 
           "Tried to create an arithmetic operation on a non-arithmetic type!");
    break;
  case And: case Or:
  case Xor:
    assert(Ty == S1->getType() &&
           "Logical operation should return same type as operands!");
    assert(Ty->isIntegral() &&
           "Tried to create an logical operation on a non-integral type!");
    break;
  case SetLT: case SetGT: case SetLE:
  case SetGE: case SetEQ: case SetNE:
    assert(Ty == Type::BoolTy && "Setcc must return bool!");
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

BinaryOperator *BinaryOperator::createNeg(Value *Op, const std::string &Name,
                                          Instruction *InsertBefore) {
  return new BinaryOperator(Instruction::Sub,
                            Constant::getNullValue(Op->getType()), Op,
                            Op->getType(), Name, InsertBefore);
}

BinaryOperator *BinaryOperator::createNot(Value *Op, const std::string &Name,
                                          Instruction *InsertBefore) {
  return new BinaryOperator(Instruction::Xor, Op,
                            ConstantIntegral::getAllOnesValue(Op->getType()),
                            Op->getType(), Name, InsertBefore);
}


// isConstantAllOnes - Helper function for several functions below
static inline bool isConstantAllOnes(const Value *V) {
  return isa<ConstantIntegral>(V) &&cast<ConstantIntegral>(V)->isAllOnesValue();
}

bool BinaryOperator::isNeg(const Value *V) {
  if (const BinaryOperator *Bop = dyn_cast<BinaryOperator>(V))
    return Bop->getOpcode() == Instruction::Sub &&
      Bop->getOperand(0) == Constant::getNullValue(Bop->getType());
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
    iType = SCI->getSwappedCondition();
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
