//===-- iBranch.cpp - Implement the Branch instruction -----------*- C++ -*--=//
//
// This file implements the 'br' instruction, which can represent either a 
// conditional or unconditional branch.
//
//===----------------------------------------------------------------------===//

#include "llvm/iTerminators.h"
#include "llvm/BasicBlock.h"
#ifndef NDEBUG
#include "llvm/Type.h"       // Only used for assertions...
#include "llvm/Assembly/Writer.h"
#endif

BranchInst::BranchInst(BasicBlock *True, BasicBlock *False, Value *Cond) 
  : TerminatorInst(Instruction::Br), TrueDest(True, this), 
    FalseDest(False, this), Condition(Cond, this) {
  assert(True != 0 && "True branch destination may not be null!!!");

#ifndef NDEBUG
  if (Cond != 0 && Cond->getType() != Type::BoolTy)
    cerr << "Bad Condition: " << Cond << endl;
#endif
  assert((Cond == 0 || Cond->getType() == Type::BoolTy) && 
         "May only branch on boolean predicates!!!!");
}

BranchInst::BranchInst(const BranchInst &BI) 
  : TerminatorInst(Instruction::Br), TrueDest(BI.TrueDest, this), 
    FalseDest(BI.FalseDest, this), Condition(BI.Condition, this) {
}


void BranchInst::dropAllReferences() {
  Condition = 0;
  TrueDest = FalseDest = 0;
}

const Value *BranchInst::getOperand(unsigned i) const {
    return (i == 0) ? (Value*)TrueDest : 
          ((i == 1) ? (Value*)FalseDest : 
          ((i == 2) ? (Value*)Condition : 0));
}

const BasicBlock *BranchInst::getSuccessor(unsigned i) const {
  return (i == 0) ? (const BasicBlock*)TrueDest : 
        ((i == 1) ? (const BasicBlock*)FalseDest : 0);
}

bool BranchInst::setOperand(unsigned i, Value *Val) { 
  switch (i) {
  case 0:
    assert(Val && "Can't change primary direction to 0!");
    assert(Val->getType() == Type::LabelTy);
    TrueDest = (BasicBlock*)Val;
    return true;
  case 1:
    assert(Val == 0 || Val->getType() == Type::LabelTy);
    FalseDest = (BasicBlock*)Val;
    return true;
  case 2:
    Condition = Val;
    assert(!Condition || Condition->getType() == Type::BoolTy && 
           "Condition expr must be a boolean expression!");
    return true;
  } 

  return false;
}
