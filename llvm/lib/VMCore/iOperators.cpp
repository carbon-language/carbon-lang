//===-- iOperators.cpp - Implement binary Operators ------------*- C++ -*--===//
//
// This file implements the nontrivial binary operator instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/iOperators.h"
#include "llvm/Type.h"
#include "llvm/Constants.h"

//===----------------------------------------------------------------------===//
//                             BinaryOperator Class
//===----------------------------------------------------------------------===//

BinaryOperator *BinaryOperator::create(BinaryOps Op, Value *S1, Value *S2,
				       const std::string &Name) {
  switch (Op) {
  // Binary comparison operators...
  case SetLT: case SetGT: case SetLE:
  case SetGE: case SetEQ: case SetNE:
    return new SetCondInst(Op, S1, S2, Name);

  default:
    return new GenericBinaryInst(Op, S1, S2, Name);
  }
}

BinaryOperator *BinaryOperator::createNeg(Value *Op, const std::string &Name) {
  return new GenericBinaryInst(Instruction::Sub,
                               Constant::getNullValue(Op->getType()), Op, Name);
}

BinaryOperator *BinaryOperator::createNot(Value *Op, const std::string &Name) {
  return new GenericBinaryInst(Instruction::Xor, Op,
                               ConstantIntegral::getAllOnesValue(Op->getType()),
                               Name);
}


// swapOperands - Exchange the two operands to this instruction.  This
// instruction is safe to use on any binary instruction and does not
// modify the semantics of the instruction.  If the instruction is
// order dependant (SetLT f.e.) the opcode is changed.
//
bool BinaryOperator::swapOperands() {
  switch (getOpcode()) {
    // Instructions that don't need opcode modification
  case Add: case Mul:
  case And: case Xor:
  case Or:
  case SetEQ: case SetNE:
    break;
    // Instructions that need opcode modification
  case SetGT: iType = SetLT; break;
  case SetLT: iType = SetGT; break;
  case SetGE: iType = SetLE; break;
  case SetLE: iType = SetGE; break;
    // Error on the side of caution
  default:
    return true;
  }
  std::swap(Operands[0], Operands[1]);
  return false;
}


//===----------------------------------------------------------------------===//
//                             SetCondInst Class
//===----------------------------------------------------------------------===//

SetCondInst::SetCondInst(BinaryOps opType, Value *S1, Value *S2, 
                         const std::string &Name) 
  : BinaryOperator(opType, S1, S2, Name) {

  OpType = opType;
  setType(Type::BoolTy);   // setcc instructions always return bool type.

  // Make sure it's a valid type...
  assert(getOpcodeName() != 0);
}
