//===--- Passes/LivenessAnalysis.h ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_LIVENESSANALYSIS_H
#define LLVM_TOOLS_LLVM_BOLT_PASSES_LIVENESSANALYSIS_H

#include "DataflowAnalysis.h"
#include "FrameAnalysis.h"

namespace llvm {
namespace bolt {

class LivenessAnalysis
    : public DataflowAnalysis<LivenessAnalysis, BitVector, true> {
  friend class DataflowAnalysis<LivenessAnalysis, BitVector, true>;

public:
  LivenessAnalysis(const FrameAnalysis &FA, const BinaryContext &BC,
                   BinaryFunction &BF)
      : DataflowAnalysis<LivenessAnalysis, BitVector, true>(BC, BF), FA(FA),
        NumRegs(BC.MRI->getNumRegs()) {}
  virtual ~LivenessAnalysis();

protected:
  /// Reference to the result of stack frame analysis
  const FrameAnalysis &FA;
  const uint16_t NumRegs;

  void preflight() {}

  BitVector getStartingStateAtBB(const BinaryBasicBlock &BB) {
    return BitVector(NumRegs, false);
  }

  BitVector getStartingStateAtPoint(const MCInst &Point) {
    return BitVector(NumRegs, false);
  }

  void doConfluence(BitVector &StateOut, const BitVector &StateIn) {
    StateOut |= StateIn;
  }

  BitVector computeNext(const MCInst &Point, const BitVector &Cur) {
    BitVector Next = Cur;
    // Kill
    auto Written = BitVector(NumRegs, false);
    if (this->BC.MIA->isCall(Point))
      FA.getInstClobberList(this->BC, Point, Written);
    else
      this->BC.MIA->getWrittenRegs(Point, Written, *this->BC.MRI);
    Written.flip();
    Next &= Written;
    // Gen
    if (!this->BC.MIA->isCFI(Point)) {
      auto Used = BitVector(NumRegs, false);
      this->BC.MIA->getUsedRegs(Point, Used, *this->BC.MRI);
      Next |= Used;
    }
    return Next;
  }

  StringRef getAnnotationName() const {
    return StringRef("LivenessAnalysis");
  }
};

} // end namespace bolt
} // end namespace llvm


#endif
