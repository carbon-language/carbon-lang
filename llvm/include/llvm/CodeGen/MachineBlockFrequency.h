//====----- MachineBlockFrequency.h - MachineBlock Frequency Analysis ----====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Loops should be simplified before this analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEBLOCKFREQUENCY_H
#define LLVM_CODEGEN_MACHINEBLOCKFREQUENCY_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include <climits>

namespace llvm {

class MachineBranchProbabilityInfo;
template<class BlockT, class FunctionT, class BranchProbInfoT>
class BlockFrequencyImpl;

/// MachineBlockFrequency pass uses BlockFrequencyImpl implementation to estimate
/// machine basic block frequencies.
class MachineBlockFrequency : public MachineFunctionPass {

  BlockFrequencyImpl<MachineBasicBlock, MachineFunction, MachineBranchProbabilityInfo> *MBFI;

public:
  static char ID;

  MachineBlockFrequency();

  ~MachineBlockFrequency();

  void getAnalysisUsage(AnalysisUsage &AU) const;

  bool runOnMachineFunction(MachineFunction &F);

  /// getblockFreq - Return block frequency. Never return 0, value must be
  /// positive. Please note that initial frequency is equal to 1024. It means
  /// that we should not rely on the value itself, but only on the comparison to
  /// the other block frequencies. We do this to avoid using of the floating
  /// points.
  uint32_t getBlockFreq(MachineBasicBlock *MBB);
};

}

#endif
