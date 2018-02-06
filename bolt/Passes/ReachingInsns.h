//===--- Passes/ReachingInsns.h -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_REACHINGINSNS_H
#define LLVM_TOOLS_LLVM_BOLT_PASSES_REACHINGINSNS_H

#include "DataflowAnalysis.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Timer.h"

namespace opts {
extern llvm::cl::opt<bool> TimeOpts;
}

namespace llvm {
namespace bolt {

template <bool Backward = false>
class ReachingInsns
    : public InstrsDataflowAnalysis<ReachingInsns<Backward>, Backward> {
  friend class DataflowAnalysis<ReachingInsns<Backward>, BitVector, Backward>;

public:
  ReachingInsns(const BinaryContext &BC, BinaryFunction &BF)
      : InstrsDataflowAnalysis<ReachingInsns, Backward>(BC, BF) {}
  virtual ~ReachingInsns() {}

  bool isInLoop(const BinaryBasicBlock &BB) {
    const MCInst *First = BB.begin() != BB.end() ? &*BB.begin() : nullptr;
    assert(First && "This analysis does not work for empty BB");
    return ((*this->getStateAt(BB))[this->ExprToIdx[First]]);
  }

  bool isInLoop(const MCInst &Inst) {
    const BinaryBasicBlock *BB = InsnToBB[&Inst];
    assert(BB && "Unknown instruction");
    return isInLoop(*BB);
  }

  void run() {
    NamedRegionTimer T1("RI", "Reaching Insns", "Dataflow", "Dataflow",
                        opts::TimeOpts);
    InstrsDataflowAnalysis<ReachingInsns<Backward>, Backward>::run();
  }

protected:
  std::unordered_map<const MCInst *, BinaryBasicBlock *> InsnToBB;

  void preflight() {
    for (auto &BB : this->Func) {
      for (auto &Inst : BB) {
        this->Expressions.push_back(&Inst);
        this->ExprToIdx[&Inst] = this->NumInstrs++;
        InsnToBB[&Inst] = &BB;
      }
    }
  }

  BitVector getStartingStateAtBB(const BinaryBasicBlock &BB) {
    return BitVector(this->NumInstrs, false);
  }

  BitVector getStartingStateAtPoint(const MCInst &Point) {
    return BitVector(this->NumInstrs, false);
  }

  void doConfluence(BitVector &StateOut, const BitVector &StateIn) {
    StateOut |= StateIn;
  }

  BitVector computeNext(const MCInst &Point, const BitVector &Cur) {
    BitVector Next = Cur;
    // Gen
    if (!this->BC.MIA->isCFI(Point)) {
      Next.set(this->ExprToIdx[&Point]);
    }
    return Next;
  }

  StringRef getAnnotationName() const {
    if (Backward)
      return StringRef("ReachingInsnsBackward");
    return StringRef("ReachingInsns");
  }
};

} // end namespace bolt
} // end namespace llvm

#endif
