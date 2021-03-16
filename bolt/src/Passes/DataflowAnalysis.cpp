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
    auto Sep = "";
    if (State.count() > (State.size() >> 1)) {
      OS << "all, except: ";
      auto BV = State;
      BV.flip();
      for (auto I = BV.find_first(); I != -1; I = BV.find_next(I)) {
        OS << Sep << I;
        Sep = " ";
      }
      OS << ")";
      return OS;
    }
    for (auto I = State.find_first(); I != -1; I = State.find_next(I)) {
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
  for (auto Pred : BB.predecessors()) {
    if (Pred->isValid())
      Task(ProgramPoint::getLastPointAt(*Pred));
  }
  if (!BB.isLandingPad())
    return;
  for (auto Thrower : BB.throwers()) {
    for (auto &Inst : *Thrower) {
      if (!BC.MIB->isInvoke(Inst))
        continue;
      const auto EHInfo = BC.MIB->getEHInfo(Inst);
      if (!EHInfo || EHInfo->first != BB.getLabel())
        continue;
      Task(ProgramPoint(&Inst));
    }
  }
}

/// Operates on all successors of a basic block.
void doForAllSuccs(const BinaryBasicBlock &BB,
                   std::function<void(ProgramPoint)> Task) {
  for (auto Succ : BB.successors()) {
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
    auto BV = State;
    BV.flip();
    for (auto I = BV.find_first(); I != -1; I = BV.find_next(I)) {
      OS << BC.MRI->getName(I) << " ";
    }
    return;
  }
  for (auto I = State.find_first(); I != -1; I = State.find_next(I)) {
    OS << BC.MRI->getName(I) << " ";
  }
}

} // namespace bolt
} // namespace llvm
