//===-- lib/MC/MCFunction.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCFunction.h"
#include "llvm/MC/MCAtom.h"
#include "llvm/MC/MCModule.h"
#include <algorithm>

using namespace llvm;

// MCFunction

MCFunction::MCFunction(StringRef Name, MCModule *Parent)
  : Name(Name), ParentModule(Parent)
{}

MCFunction::~MCFunction() {
  for (iterator I = begin(), E = end(); I != E; ++I)
    delete *I;
}

MCBasicBlock &MCFunction::createBlock(const MCTextAtom &TA) {
  MCBasicBlock *MCBB = new MCBasicBlock(TA, this);
  Blocks.push_back(MCBB);
  return *MCBB;
}

MCBasicBlock *MCFunction::find(uint64_t StartAddr) {
  for (const_iterator I = begin(), E = end(); I != E; ++I)
    if ((*I)->getInsts()->getBeginAddr() == StartAddr)
      return *I;
  return 0;
}

const MCBasicBlock *MCFunction::find(uint64_t StartAddr) const {
  return const_cast<MCFunction *>(this)->find(StartAddr);
}

// MCBasicBlock

MCBasicBlock::MCBasicBlock(const MCTextAtom &Insts, MCFunction *Parent)
  : Insts(&Insts), Parent(Parent) {
  getParent()->getParent()->trackBBForAtom(&Insts, this);
}

void MCBasicBlock::addSuccessor(const MCBasicBlock *MCBB) {
  if (!isSuccessor(MCBB))
    Successors.push_back(MCBB);
}

bool MCBasicBlock::isSuccessor(const MCBasicBlock *MCBB) const {
  return std::find(Successors.begin(), Successors.end(),
                   MCBB) != Successors.end();
}

void MCBasicBlock::addPredecessor(const MCBasicBlock *MCBB) {
  if (!isPredecessor(MCBB))
    Predecessors.push_back(MCBB);
}

bool MCBasicBlock::isPredecessor(const MCBasicBlock *MCBB) const {
  return std::find(Predecessors.begin(), Predecessors.end(),
                   MCBB) != Predecessors.end();
}

void MCBasicBlock::splitBasicBlock(MCBasicBlock *SplitBB) {
  assert(Insts->getEndAddr() + 1 == SplitBB->Insts->getBeginAddr() &&
         "Splitting unrelated basic blocks!");
  SplitBB->addPredecessor(this);
  assert(SplitBB->Successors.empty() &&
         "Split basic block shouldn't already have successors!");
  SplitBB->Successors = Successors;
  Successors.clear();
  addSuccessor(SplitBB);
}
