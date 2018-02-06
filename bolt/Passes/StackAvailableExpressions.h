//===--- Passes/StackAvailableExpressions.h -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_STACKAVAILABLEEXPRESSIONS_H
#define LLVM_TOOLS_LLVM_BOLT_PASSES_STACKAVAILABLEEXPRESSIONS_H

#include "DataflowAnalysis.h"
#include "RegAnalysis.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Timer.h"

namespace opts {
extern llvm::cl::opt<bool> TimeOpts;
}

namespace llvm {
namespace bolt {

class FrameAnalysis;

class StackAvailableExpressions
    : public InstrsDataflowAnalysis<StackAvailableExpressions> {
  friend class DataflowAnalysis<StackAvailableExpressions, BitVector>;

public:
  StackAvailableExpressions(const RegAnalysis &RA, const FrameAnalysis &FA,
                            const BinaryContext &BC, BinaryFunction &BF);
  virtual ~StackAvailableExpressions() {}

  void run() {
    NamedRegionTimer T1("SAE", "Stack Available Expressions", "Dataflow",
                        "Dataflow", opts::TimeOpts);
    InstrsDataflowAnalysis<StackAvailableExpressions>::run();
  }

protected:
  const RegAnalysis &RA;
  const FrameAnalysis &FA;

  void preflight();
  BitVector getStartingStateAtBB(const BinaryBasicBlock &BB);
  BitVector getStartingStateAtPoint(const MCInst &Point);
  void doConfluence(BitVector &StateOut, const BitVector &StateIn);
  /// Define the function computing the kill set -- whether expression Y, a
  /// tracked expression, will be considered to be dead after executing X.
  bool doesXKillsY(const MCInst *X, const MCInst *Y);
  BitVector computeNext(const MCInst &Point, const BitVector &Cur);

  StringRef getAnnotationName() const {
    return StringRef("StackAvailableExpressions");
  }
};

} // namespace bolt
} // namespace llvm

#endif
