//===--- Passes/StackAllocationAnalysis.h ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_STACKALLOCATIONANALYSIS_H
#define LLVM_TOOLS_LLVM_BOLT_PASSES_STACKALLOCATIONANALYSIS_H

#include "DataflowAnalysis.h"
#include "StackPointerTracking.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Timer.h"

namespace opts {
extern llvm::cl::opt<bool> TimeOpts;
}

namespace llvm {
namespace bolt {

/// Perform a dataflow analysis to track the value of SP as an offset relative
/// to the CFA.
class StackAllocationAnalysis
    : public InstrsDataflowAnalysis<StackAllocationAnalysis,
                                    /*Backward=*/false> {
  friend class DataflowAnalysis<StackAllocationAnalysis, BitVector>;

  StackPointerTracking &SPT;

public:
  StackAllocationAnalysis(const BinaryContext &BC, BinaryFunction &BF,
                          StackPointerTracking &SPT)
      : InstrsDataflowAnalysis<StackAllocationAnalysis, false>(BC, BF),
        SPT(SPT) {}
  virtual ~StackAllocationAnalysis() {}

  void run() {
    NamedRegionTimer T1("SAA", "Stack Allocation Analysis", "Dataflow",
                        "Dataflow", opts::TimeOpts);
    InstrsDataflowAnalysis<StackAllocationAnalysis, false>::run();
  }

protected:
  void preflight();

  BitVector getStartingStateAtBB(const BinaryBasicBlock &BB);

  BitVector getStartingStateAtPoint(const MCInst &Point);

  void doConfluence(BitVector &StateOut, const BitVector &StateIn);

  BitVector doKill(const MCInst &Point, const BitVector &StateIn,
                   int DeallocSize);

  void doConfluenceWithLP(BitVector &StateOut, const BitVector &StateIn,
                          const MCInst &Invoke);

  BitVector computeNext(const MCInst &Point, const BitVector &Cur);

  StringRef getAnnotationName() const {
    return StringRef("StackAllocationAnalysis");
  }
};

} // end namespace bolt
} // end namespace llvm

#endif
