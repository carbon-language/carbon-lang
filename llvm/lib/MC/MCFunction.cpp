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
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace llvm;

// MCFunction

MCFunction::MCFunction(StringRef Name)
  : Name(Name)
{}

MCFunction::~MCFunction() {
  for (iterator I = begin(), E = end(); I != E; ++I)
    delete *I;
}

MCBasicBlock &MCFunction::createBlock(const MCTextAtom &TA) {
  Blocks.push_back(new MCBasicBlock(TA, this));
  return *Blocks.back();
}

// MCBasicBlock

MCBasicBlock::MCBasicBlock(const MCTextAtom &Insts, MCFunction *Parent)
  : Insts(&Insts), Parent(Parent)
{}

void MCBasicBlock::addSuccessor(const MCBasicBlock *MCBB) {
  Successors.push_back(MCBB);
}

bool MCBasicBlock::isSuccessor(const MCBasicBlock *MCBB) const {
  return std::find(Successors.begin(), Successors.end(),
                   MCBB) != Successors.end();
}

void MCBasicBlock::addPredecessor(const MCBasicBlock *MCBB) {
  Predecessors.push_back(MCBB);
}

bool MCBasicBlock::isPredecessor(const MCBasicBlock *MCBB) const {
  return std::find(Predecessors.begin(), Predecessors.end(),
                   MCBB) != Predecessors.end();
}
