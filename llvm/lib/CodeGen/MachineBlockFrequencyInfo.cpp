//====----- MachineBlockFrequencyInfo.cpp - Machine Block Frequency Analysis ----====//
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

#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/Analysis/BlockFrequencyImpl.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

INITIALIZE_PASS_BEGIN(MachineBlockFrequencyInfo, "machine-block-freq",
                      "Machine Block Frequency Analysis", true, true)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfo)
INITIALIZE_PASS_END(MachineBlockFrequencyInfo, "machine-block-freq",
                    "Machine Block Frequency Analysis", true, true)

char MachineBlockFrequencyInfo::ID = 0;


MachineBlockFrequencyInfo::MachineBlockFrequencyInfo() : MachineFunctionPass(ID) {
  initializeMachineBlockFrequencyInfoPass(*PassRegistry::getPassRegistry());
  MBFI = new BlockFrequencyImpl<MachineBasicBlock, MachineFunction,
                                MachineBranchProbabilityInfo>();
}

MachineBlockFrequencyInfo::~MachineBlockFrequencyInfo() {
  delete MBFI;
}

void MachineBlockFrequencyInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineBranchProbabilityInfo>();
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool MachineBlockFrequencyInfo::runOnMachineFunction(MachineFunction &F) {
  MachineBranchProbabilityInfo &MBPI = getAnalysis<MachineBranchProbabilityInfo>();
  MBFI->doFunction(&F, &MBPI);
  return false;
}

BlockFrequency MachineBlockFrequencyInfo::
getBlockFreq(const MachineBasicBlock *MBB) const {
  return MBFI->getBlockFreq(MBB);
}
