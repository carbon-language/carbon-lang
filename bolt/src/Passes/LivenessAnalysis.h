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
    const BitVector &RegAliases = BC.MIB->getAliases(Reg);
    BV &= RegAliases;
    return BV.any();
  }

  void run() {
    NamedRegionTimer T1("LA", "Liveness Analysis", "Dataflow", "Dataflow",
                        opts::TimeOpts);
    Parent::run();
  }

  // Return a usable general-purpose reg after point P. Return 0 if no reg is
  // available.
  MCPhysReg scavengeRegAfter(ProgramPoint P) {
    BitVector BV = *this->getStateAt(P);
    BV.flip();
    BitVector GPRegs(NumRegs, false);
    this->BC.MIB->getGPRegs(GPRegs, /*IncludeAlias=*/false);
    // Ignore the register used for frame pointer even if it is not alive (it
    // may be used by CFI which is not represented in our dataflow).
    auto FP = BC.MIB->getAliases(BC.MIB->getFramePointer());
    FP.flip();
    BV &= GPRegs;
    BV &= FP;
    int Reg = BV.find_first();
    return Reg != -1 ? Reg : 0;
  }

protected:
  /// Reference to the result of reg analysis
  const RegAnalysis &RA;
  const uint16_t NumRegs;

  void preflight() {}

  BitVector getStartingStateAtBB(const BinaryBasicBlock &BB) {
    // Entry points start with default live out (registers used as return
    // values).
    if (BB.succ_size() == 0) {
      BitVector State(NumRegs, false);
      if (opts::AssumeABI) {
        BC.MIB->getDefaultLiveOut(State);
        BC.MIB->getCalleeSavedRegs(State);
      } else {
        State.set();
        State.reset(BC.MIB->getFlagsReg());
      }
      return State;
    }
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
    bool IsCall = this->BC.MIB->isCall(Point);
    // Kill
    auto Written = BitVector(NumRegs, false);
    if (!IsCall) {
      this->BC.MIB->getWrittenRegs(Point, Written);
    } else {
      RA.getInstClobberList(Point, Written);
      // When clobber list is conservative, it is clobbering all/most registers,
      // a conservative estimate because it knows nothing about this call.
      // For our purposes, assume it kills no registers/callee-saved regs
      // because we don't really know what's going on.
      if (RA.isConservative(Written)) {
        Written.reset();
        BC.MIB->getDefaultLiveOut(Written);
        // If ABI is respected, everything except CSRs should be dead after a
        // call
        if (opts::AssumeABI) {
          auto CSR = BitVector(NumRegs, false);
          BC.MIB->getCalleeSavedRegs(CSR);
          CSR.flip();
          Written |= CSR;
        }
      }
    }
    Written.flip();
    Next &= Written;
    // Gen
    if (!this->BC.MIB->isCFI(Point)) {
      if (BC.MIB->isCleanRegXOR(Point))
        return Next;

      auto Used = BitVector(NumRegs, false);
      if (IsCall) {
        RA.getInstUsedRegsList(Point, Used, /*GetClobbers*/true);
        if (RA.isConservative(Used)) {
          Used = BC.MIB->getRegsUsedAsParams();
          BC.MIB->getDefaultLiveOut(Used);
        }
      }
      const auto InstInfo = BC.MII->get(Point.getOpcode());
      for (unsigned I = 0, E = Point.getNumOperands(); I != E; ++I) {
        if (!Point.getOperand(I).isReg() || I < InstInfo.getNumDefs())
          continue;
        Used |= BC.MIB->getAliases(Point.getOperand(I).getReg(),
                                   /*OnlySmaller=*/false);
      }
      for (auto
             I = InstInfo.getImplicitUses(),
             E = InstInfo.getImplicitUses() + InstInfo.getNumImplicitUses();
           I != E; ++I) {
        Used |= BC.MIB->getAliases(*I, false);
      }
      if (IsCall &&
          (!BC.MIB->isTailCall(Point) || !BC.MIB->isConditionalBranch(Point))) {
        // Never gen FLAGS from a non-conditional call... this is overly
        // conservative
        Used.reset(BC.MIB->getFlagsReg());
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
