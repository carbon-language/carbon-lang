//===-- iBinaryOperators.cpp - Implement the BinaryOperators -----*- C++ -*--=//
//
// This file implements the nontrivial binary operator instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/iBinary.h"
#include "llvm/Type.h"

BinaryOperator *BinaryOperator::create(unsigned Op, Value *S1, Value *S2,
				       const string &Name) {
  switch (Op) {
  case Add: return new AddInst(S1, S2, Name);
  case Sub: return new SubInst(S1, S2, Name);
  case SetLT:
  case SetGT:
  case SetLE:
  case SetGE:
  case SetEQ:
  case SetNE:
    return new SetCondInst((BinaryOps)Op, S1, S2, Name);

  default:
    cerr << "Don't know how to GetBinaryOperator " << Op << endl;
    return 0;
  }
}

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
