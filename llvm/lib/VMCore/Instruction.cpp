//===-- Instruction.cpp - Implement the Instruction class --------*- C++ -*--=//
//
// This file implements the Instruction class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Instruction.h"
#include "llvm/BasicBlock.h"
#include "llvm/Method.h"
#include "llvm/SymbolTable.h"
#include "llvm/iBinary.h"
#include "llvm/iUnary.h"

Instruction::Instruction(const Type *ty, unsigned it, const string &Name) 
  : User(ty, Value::InstructionVal, Name) {
  Parent = 0;
  iType = it;
}

Instruction::~Instruction() {
  assert(getParent() == 0 && "Instruction still embeded in basic block!");
}

// Specialize setName to take care of symbol table majik
void Instruction::setName(const string &name) {
  BasicBlock *P = 0; Method *PP = 0;
  if ((P = getParent()) && (PP = P->getParent()) && hasName())
    PP->getSymbolTable()->remove(this);
  Value::setName(name);
  if (PP && hasName()) PP->getSymbolTableSure()->insert(this);
}

BinaryOperator *BinaryOperator::getBinaryOperator(unsigned Op, 
						  Value *S1, Value *S2) {
  switch (Op) {
  case Add:
    return new AddInst(S1, S2);
  case Sub:
    return new SubInst(S1, S2);

  case SetLT:
  case SetGT:
  case SetLE:
  case SetGE:
  case SetEQ:
  case SetNE:
    return new SetCondInst((BinaryOps)Op, S1, S2);

  default:
    cerr << "Don't know how to GetBinaryOperator " << Op << endl;
    return 0;
  }
}


UnaryOperator *UnaryOperator::getUnaryOperator(unsigned Op, Value *Source) {
  switch (Op) {
  default:
    cerr << "Don't know how to GetUnaryOperator " << Op << endl;
    return 0;
  }
}
