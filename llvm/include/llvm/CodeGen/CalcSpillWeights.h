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

namespace llvm {

  class LiveInterval;

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
