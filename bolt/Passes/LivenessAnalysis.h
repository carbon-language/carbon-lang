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
#include "RegAnalysis.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Timer.h"

namespace opts {
extern llvm::cl::opt<bool> AssumeABI;
extern llvm::cl::opt<bool> TimeOpts;
}

namespace llvm {
namespace bolt {

class LivenessAnalysis
  : public DataflowAnalysis<LivenessAnalysis, BitVector, true, RegStatePrinter> {
  using Parent = DataflowAnalysis<LivenessAnalysis,
                                  BitVector,
                                  true,
                                  RegStatePrinter>;
  friend class DataflowAnalysis<LivenessAnalysis,
                                BitVector,
                                true,
                                RegStatePrinter>;

public:
  LivenessAnalysis(const RegAnalysis &RA, const BinaryContext &BC,
                   BinaryFunction &BF)
      : Parent(BC, BF), RA(RA), NumRegs(BC.MRI->getNumRegs()) {}
  virtual ~LivenessAnalysis();

  bool isAlive(ProgramPoint PP, MCPhysReg Reg) const {
    BitVector BV = (*this->getStateAt(PP));
    const BitVector &RegAliases = BC.MIA->getAliases(Reg, *BC.MRI);
    BV &= RegAliases;
    return BV.any();
  }

  void run() {
    NamedRegionTimer T1("LA", "Dataflow", opts::TimeOpts);
    Parent::run();
  }

  // Return a usable general-purpose reg after point P. Return 0 if no reg is
  // available.
  MCPhysReg scavengeRegAfter(ProgramPoint P) {
    BitVector BV = *this->getStateAt(P);
    BV.flip();
    BitVector GPRegs(NumRegs, false);
    this->BC.MIA->getGPRegs(GPRegs, *this->BC.MRI);
    BV &= GPRegs;
    int Reg = BV.find_first();
    return Reg != -1 ? Reg : 0;
  }

protected:
  /// Reference to the result of reg analysis
  const RegAnalysis &RA;
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
    bool IsCall = this->BC.MIA->isCall(Point);
    // Kill
    auto Written = BitVector(NumRegs, false);
    if (!IsCall) {
      this->BC.MIA->getWrittenRegs(Point, Written, *this->BC.MRI);
    } else {
      RA.getInstClobberList(Point, Written);
      // When clobber list is conservative, it is clobbering all/most registers,
      // a conservative estimate because it knows nothing about this call.
      // For our purposes, assume it kills no registers/callee-saved regs
      // because we don't really know what's going on.
      if (RA.isConservative(Written)) {
        Written.reset();
        BC.MIA->getCalleeSavedRegs(Written, *this->BC.MRI);
      }
    }
    Written.flip();
    Next &= Written;
    // Gen
    if (!this->BC.MIA->isCFI(Point)) {
      auto Used = BitVector(NumRegs, false);
      RA.getInstUsedRegsList(Point, Used, /*GetClobbers*/false);
      if (IsCall &&
          (!BC.MIA->isTailCall(Point) || !BC.MIA->isConditionalBranch(Point))) {
        // Never gen FLAGS from a non-conditional call... this is overly
        // conservative
        Used.reset(BC.MIA->getFlagsReg());
      }
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
