//===-- iBinaryOperators.cpp - Implement the BinaryOperators -----*- C++ -*--=//
//
// This file implements the nontrivial binary operator instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/iBinary.h"
#include "llvm/Type.h"

//===----------------------------------------------------------------------===//
//                             SetCondInst Class
//===----------------------------------------------------------------------===//

SetCondInst::SetCondInst(BinaryOps opType, Value *S1, Value *S2, 
                         const string &Name) 
  : BinaryOperator(opType, S1, S2, Name) {

  OpType = opType;
  setType(Type::BoolTy);   // setcc instructions always return bool type.

  // Make sure it's a valid type...
  assert(getOpcode() != "Invalid opcode type to SetCondInst class!");
}

string SetCondInst::getOpcode() const {
  switch (OpType) {
  case SetLE:  return "setle";
  case SetGE:  return "setge";
  case SetLT:  return "setlt";
  case SetGT:  return "setgt";
  case SetEQ:  return "seteq";
  case SetNE:  return "setne";
  default:
    assert(0 && "Invalid opcode type to SetCondInst class!");
    return "invalid opcode type to SetCondInst";
  }
}
