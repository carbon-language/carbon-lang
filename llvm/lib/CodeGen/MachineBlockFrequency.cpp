//====----- MachineBlockFrequency.cpp - Machine Block Frequency Analysis ----====//
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

#include "llvm/InitializePasses.h"
#include "llvm/Analysis/BlockFrequencyImpl.h"
#include "llvm/CodeGen/MachineBlockFrequency.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"

using namespace llvm;

INITIALIZE_PASS_BEGIN(MachineBlockFrequency, "machine-block-freq",
                      "Machine Block Frequency Analysis", true, true)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfo)
INITIALIZE_PASS_END(MachineBlockFrequency, "machine-block-freq",
                    "Machine Block Frequency Analysis", true, true)

char MachineBlockFrequency::ID = 0;


MachineBlockFrequency::MachineBlockFrequency() : MachineFunctionPass(ID) {
  initializeMachineBlockFrequencyPass(*PassRegistry::getPassRegistry());
  MBFI = new BlockFrequencyImpl<MachineBasicBlock, MachineFunction,
                                MachineBranchProbabilityInfo>();
}

MachineBlockFrequency::~MachineBlockFrequency() {
  delete MBFI;
}

void MachineBlockFrequency::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineBranchProbabilityInfo>();
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool MachineBlockFrequency::runOnMachineFunction(MachineFunction &F) {
  MachineBranchProbabilityInfo &MBPI = getAnalysis<MachineBranchProbabilityInfo>();
  MBFI->doFunction(&F, &MBPI);
  return false;
}

/// getblockFreq - Return block frequency. Return 0 if we don't have the
/// information. Please note that initial frequency is equal to 1024. It means
/// that we should not rely on the value itself, but only on the comparison to
/// the other block frequencies. We do this to avoid using of floating points.
///
uint32_t MachineBlockFrequency::getBlockFreq(MachineBasicBlock *MBB) {
  return MBFI->getBlockFreq(MBB);
}
