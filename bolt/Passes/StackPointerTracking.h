//===--- Passes/StackPointerTracking.h ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_STACKPOINTERTRACKING_H
#define LLVM_TOOLS_LLVM_BOLT_PASSES_STACKPOINTERTRACKING_H

#include "DataflowAnalysis.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Timer.h"

namespace opts {
extern llvm::cl::opt<bool> TimeOpts;
}

namespace llvm {
namespace bolt {

/// Perform a dataflow analysis to track the value of SP as an offset relative
/// to the CFA.
template <typename Derived>
class StackPointerTrackingBase
    : public DataflowAnalysis<Derived, std::pair<int, int>> {
  friend class DataflowAnalysis<Derived, std::pair<int, int>>;

protected:
  void preflight() {}

  int getEmpty() { return EMPTY; }

  std::pair<int, int> getStartingStateAtBB(const BinaryBasicBlock &BB) {
    // Entry BB start with offset 8 from CFA.
    // All others start with EMPTY (meaning we don't know anything).
    if (BB.isEntryPoint())
      return std::make_pair(-8, getEmpty());
    return std::make_pair(getEmpty(), getEmpty());
  }

  std::pair<int, int> getStartingStateAtPoint(const MCInst &Point) {
    return std::make_pair(getEmpty(), getEmpty());
  }

  void doConfluenceSingleReg(int &StateOut, const int &StateIn) {
    if (StateOut == EMPTY) {
      StateOut = StateIn;
      return;
    }
    if (StateIn == EMPTY || StateIn == StateOut)
      return;

    // We can't agree on a specific value from this point on
    StateOut = SUPERPOSITION;
  }

  void doConfluence(std::pair<int, int> &StateOut,
                    const std::pair<int, int> &StateIn) {
    doConfluenceSingleReg(StateOut.first, StateIn.first);
    doConfluenceSingleReg(StateOut.second, StateIn.second);
  }

  void doConfluenceWithLP(std::pair<int, int> &StateOut,
                          const std::pair<int, int> &StateIn,
                          const MCInst &Invoke) {
    int SPVal = StateIn.first;
    for (const auto &Operand : Invoke) {
      if (Operand.isGnuArgsSize()) {
        auto ArgsSize = Operand.getGnuArgsSize();
        if (SPVal != EMPTY && SPVal != SUPERPOSITION) {
          SPVal += ArgsSize;
        }
      }
    }
    doConfluenceSingleReg(StateOut.first, SPVal);
    doConfluenceSingleReg(StateOut.second, StateIn.second);
  }

  int computeNextSP(const MCInst &Point, int SPVal, int FPVal) {
    const auto &MIA = this->BC.MIA;

    if (int Sz = MIA->getPushSize(Point)) {
      if (SPVal == EMPTY || SPVal == SUPERPOSITION)
        return SPVal;

      return SPVal - Sz;
    }

    if (int Sz = MIA->getPopSize(Point)) {
      if (SPVal == EMPTY || SPVal == SUPERPOSITION)
        return SPVal;

      return SPVal + Sz;
    }

    MCPhysReg From, To;
    if (MIA->isRegToRegMove(Point, From, To) && To == MIA->getStackPointer() &&
        From == MIA->getFramePointer()) {
      if (FPVal == EMPTY || FPVal == SUPERPOSITION)
        return FPVal;

      if (MIA->isLeave(Point))
        return FPVal + 8;
      else
        return FPVal;
    }

    if (this->BC.MII->get(Point.getOpcode())
            .hasDefOfPhysReg(Point, MIA->getStackPointer(), *this->BC.MRI)) {
      std::pair<MCPhysReg, int64_t> SP;
      if (SPVal != EMPTY && SPVal != SUPERPOSITION)
        SP = std::make_pair(MIA->getStackPointer(), SPVal);
      else
        SP = std::make_pair(0, 0);
      std::pair<MCPhysReg, int64_t> FP;
      if (FPVal != EMPTY && FPVal != SUPERPOSITION)
        FP = std::make_pair(MIA->getFramePointer(), FPVal);
      else
        FP = std::make_pair(0, 0);
      int64_t Output;
      if (!MIA->evaluateSimple(Point, Output, SP, FP)) {
        if (SPVal == EMPTY && FPVal == EMPTY)
          return SPVal;
        return SUPERPOSITION;
      }

      return static_cast<int>(Output);
    }

    return SPVal;
  }

  int computeNextFP(const MCInst &Point, int SPVal, int FPVal) {
    const auto &MIA = this->BC.MIA;

    MCPhysReg From, To;
    if (MIA->isRegToRegMove(Point, From, To) && To == MIA->getFramePointer() &&
        From == MIA->getStackPointer()) {
      HasFramePointer = true;
      return SPVal;
    }

    if (this->BC.MII->get(Point.getOpcode())
            .hasDefOfPhysReg(Point, MIA->getFramePointer(), *this->BC.MRI)) {
      std::pair<MCPhysReg, int64_t> FP;
      if (FPVal != EMPTY && FPVal != SUPERPOSITION)
        FP = std::make_pair(MIA->getFramePointer(), FPVal);
      else
        FP = std::make_pair(0, 0);
      std::pair<MCPhysReg, int64_t> SP;
      if (SPVal != EMPTY && SPVal != SUPERPOSITION)
        SP = std::make_pair(MIA->getStackPointer(), SPVal);
      else
        SP = std::make_pair(0, 0);
      int64_t Output;
      if (!MIA->evaluateSimple(Point, Output, SP, FP)) {
        if (SPVal == EMPTY && FPVal == EMPTY)
          return FPVal;
        return SUPERPOSITION;
      }

      if (!HasFramePointer) {
        if (MIA->escapesVariable(Point, false)) {
          HasFramePointer = true;
        }
      }
      return static_cast<int>(Output);
    }

    return FPVal;
  }

  std::pair<int, int> computeNext(const MCInst &Point,
                                  const std::pair<int, int> &Cur) {
    return std::make_pair(computeNextSP(Point, Cur.first, Cur.second),
                          computeNextFP(Point, Cur.first, Cur.second));
  }

  StringRef getAnnotationName() const {
    return StringRef("StackPointerTracking");
  }

public:
  StackPointerTrackingBase(const BinaryContext &BC, BinaryFunction &BF)
      : DataflowAnalysis<Derived, std::pair<int, int>>(BC, BF) {}
  virtual ~StackPointerTrackingBase() {}
  bool HasFramePointer{false};

  static constexpr int SUPERPOSITION = std::numeric_limits<int>::max();
  static constexpr int EMPTY = std::numeric_limits<int>::min();
};

class StackPointerTracking
    : public StackPointerTrackingBase<StackPointerTracking> {
  friend class DataflowAnalysis<StackPointerTracking, std::pair<int, int>>;

public:
  StackPointerTracking(const BinaryContext &BC, BinaryFunction &BF);
  virtual ~StackPointerTracking() {}

  void run() {
    NamedRegionTimer T1("SPT", "Dataflow", opts::TimeOpts);
    StackPointerTrackingBase<StackPointerTracking>::run();
  }
};

} // end namespace bolt

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const std::pair<int, int> &Val);

} // end namespace llvm


#endif
