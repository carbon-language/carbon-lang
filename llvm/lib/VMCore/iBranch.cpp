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

namespace llvm {

BranchInst::BranchInst(BasicBlock *True, BasicBlock *False, Value *Cond,
                       Instruction *InsertBefore) 
  : TerminatorInst(Instruction::Br, InsertBefore) {
  assert(True != 0 && "True branch destination may not be null!!!");
  Operands.reserve(False ? 3 : 1);
  Operands.push_back(Use(True, this));
  if (False) {
    Operands.push_back(Use(False, this));
    Operands.push_back(Use(Cond, this));
  }

  assert(!!False == !!Cond &&
	 "Either both cond and false or neither can be specified!");
  assert((Cond == 0 || Cond->getType() == Type::BoolTy) && 
         "May only branch on boolean predicates!!!!");
}

BranchInst::BranchInst(BasicBlock *True, Instruction *InsertBefore) 
  : TerminatorInst(Instruction::Br, InsertBefore) {
  assert(True != 0 && "True branch destination may not be null!!!");
  Operands.reserve(1);
  Operands.push_back(Use(True, this));
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

} // End llvm namespace
