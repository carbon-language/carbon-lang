//===---------------- lib/CodeGen/CalcSpillWeights.h ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_CODEGEN_CALCSPILLWEIGHTS_H
#define LLVM_CODEGEN_CALCSPILLWEIGHTS_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {

  class LiveInterval;
  class LiveIntervals;
  class MachineLoopInfo;

  /// VirtRegAuxInfo - Calculate auxiliary information for a virtual
  /// register such as its spill weight and allocation hint.
  class VirtRegAuxInfo {
    MachineFunction &mf_;
    LiveIntervals &lis_;
    const MachineLoopInfo &loops_;
    DenseMap<unsigned, float> hint_;
  public:
    VirtRegAuxInfo(MachineFunction &mf, LiveIntervals &lis,
                   const MachineLoopInfo &loops) :
      mf_(mf), lis_(lis), loops_(loops) {}

    /// CalculateRegClass - recompute the register class for li from its uses.
    /// Since the register class can affect the allocation hint, this function
    /// should be called before CalculateWeightAndHint if both are called.
    void CalculateRegClass(LiveInterval &li);

    /// CalculateWeightAndHint - (re)compute li's spill weight and allocation
    /// hint.
    void CalculateWeightAndHint(LiveInterval &li);
  };

  /// CalculateSpillWeights - Compute spill weights for all virtual register
  /// live intervals.
  class CalculateSpillWeights : public MachineFunctionPass {
  public:
    static char ID;

    CalculateSpillWeights() : MachineFunctionPass(ID) {}

    virtual void getAnalysisUsage(AnalysisUsage &au) const;

    virtual bool runOnMachineFunction(MachineFunction &fn);

  private:
    /// Returns true if the given live interval is zero length.
    bool isZeroLengthInterval(LiveInterval *li) const;
  };

}

#endif // LLVM_CODEGEN_CALCSPILLWEIGHTS_H
