//===--- Passes/StackReachingUses.h ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_STACKREACHINGUSES_H
#define LLVM_TOOLS_LLVM_BOLT_PASSES_STACKREACHINGUSES_H

#include "DataflowAnalysis.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Timer.h"

namespace opts {
extern llvm::cl::opt<bool> TimeOpts;
}

namespace llvm {
namespace bolt {

class FrameAnalysis;
struct FrameIndexEntry;

class StackReachingUses
    : public InstrsDataflowAnalysis<StackReachingUses, /*Backward=*/true> {
  friend class DataflowAnalysis<StackReachingUses, BitVector, true>;

public:
  StackReachingUses(const FrameAnalysis &FA, const BinaryContext &BC,
                    BinaryFunction &BF)
      : InstrsDataflowAnalysis(BC, BF), FA(FA) {}
  virtual ~StackReachingUses() {}

  bool isStoreUsed(const FrameIndexEntry &StoreFIE, ExprIterator Candidates,
                   bool IncludeLocalAccesses = true) const;

  void run() {
    NamedRegionTimer T1("SRU", "Dataflow", opts::TimeOpts);
    InstrsDataflowAnalysis<StackReachingUses, true>::run();
  }

protected:
  // Reference to the result of stack frame analysis
  const FrameAnalysis &FA;

  void preflight();

  BitVector getStartingStateAtBB(const BinaryBasicBlock &BB) {
    return BitVector(NumInstrs, false);
  }

  BitVector getStartingStateAtPoint(const MCInst &Point) {
    return BitVector(NumInstrs, false);
  }

  void doConfluence(BitVector &StateOut, const BitVector &StateIn) {
    StateOut |= StateIn;
  }

  // Define the function computing the kill set -- whether expression Y, a
  // tracked expression, will be considered to be dead after executing X.
  bool doesXKillsY(const MCInst *X, const MCInst *Y);
  BitVector computeNext(const MCInst &Point, const BitVector &Cur);

  StringRef getAnnotationName() const { return StringRef("StackReachingUses"); }
};

} // end namespace bolt
} // end namespace llvm

#endif
