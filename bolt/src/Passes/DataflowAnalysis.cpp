//===--- Passes/DataflowAnalysis.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "DataflowAnalysis.h"

#define DEBUG_TYPE "dataflow"

namespace llvm {

raw_ostream &operator<<(raw_ostream &OS, const BitVector &State) {
  LLVM_DEBUG({
    OS << "BitVector(";
    const char *Sep = "";
    if (State.count() > (State.size() >> 1)) {
      OS << "all, except: ";
      BitVector BV = State;
      BV.flip();
      for (int I = BV.find_first(); I != -1; I = BV.find_next(I)) {
        OS << Sep << I;
        Sep = " ";
      }
      OS << ")";
      return OS;
    }
    for (int I = State.find_first(); I != -1; I = State.find_next(I)) {
      OS << Sep << I;
      Sep = " ";
    }
    OS << ")";
    return OS;
  });
  OS << "BitVector";
  return OS;
}

namespace bolt {

void doForAllPreds(const BinaryContext &BC, const BinaryBasicBlock &BB,
                   std::function<void(ProgramPoint)> Task) {
  for (BinaryBasicBlock *Pred : BB.predecessors()) {
    if (Pred->isValid())
      Task(ProgramPoint::getLastPointAt(*Pred));
  }
  if (!BB.isLandingPad())
    return;
  for (BinaryBasicBlock *Thrower : BB.throwers()) {
    for (MCInst &Inst : *Thrower) {
      if (!BC.MIB->isInvoke(Inst))
        continue;
      const Optional<MCPlus::MCLandingPad> EHInfo = BC.MIB->getEHInfo(Inst);
      if (!EHInfo || EHInfo->first != BB.getLabel())
        continue;
      Task(ProgramPoint(&Inst));
    }
  }
}

/// Operates on all successors of a basic block.
void doForAllSuccs(const BinaryBasicBlock &BB,
                   std::function<void(ProgramPoint)> Task) {
  for (BinaryBasicBlock *Succ : BB.successors()) {
    if (Succ->isValid())
      Task(ProgramPoint::getFirstPointAt(*Succ));
  }
}

void RegStatePrinter::print(raw_ostream &OS, const BitVector &State) const {
  if (State.all()) {
    OS << "(all)";
    return;
  }
  if (State.count() > (State.size() >> 1)) {
    OS << "all, except: ";
    BitVector BV = State;
    BV.flip();
    for (int I = BV.find_first(); I != -1; I = BV.find_next(I)) {
      OS << BC.MRI->getName(I) << " ";
    }
    return;
  }
  for (int I = State.find_first(); I != -1; I = State.find_next(I)) {
    OS << BC.MRI->getName(I) << " ";
  }
}

} // namespace bolt
} // namespace llvm
