//===-- iBranch.cpp - Implement the Branch instruction --------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the 'br' instruction, which can represent either a 
// conditional or unconditional branch.
//
//===----------------------------------------------------------------------===//

#include "llvm/iTerminators.h"
#include "llvm/BasicBlock.h"
#include "llvm/Type.h"
using namespace llvm;

// Out-of-line ReturnInst method, put here so the C++ compiler can choose to
// emit the vtable for the class in this translation unit.
void ReturnInst::setSuccessor(unsigned idx, BasicBlock *NewSucc) {
  assert(0 && "ReturnInst has no successors!");
}

// Likewise for UnwindInst
void UnwindInst::setSuccessor(unsigned idx, BasicBlock *NewSucc) {
  assert(0 && "UnwindInst has no successors!");
}

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
