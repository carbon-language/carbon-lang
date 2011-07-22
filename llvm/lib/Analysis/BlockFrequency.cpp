//=======-------- BlockFrequency.cpp - Block Frequency Analysis -------=======//
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
#include "llvm/Analysis/BlockFrequency.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"

using namespace llvm;

INITIALIZE_PASS_BEGIN(BlockFrequency, "block-freq", "Block Frequency Analysis",
                      true, true)
INITIALIZE_PASS_DEPENDENCY(BranchProbabilityInfo)
INITIALIZE_PASS_END(BlockFrequency, "block-freq", "Block Frequency Analysis",
                    true, true)

char BlockFrequency::ID = 0;


BlockFrequency::BlockFrequency() : FunctionPass(ID) {
  initializeBlockFrequencyPass(*PassRegistry::getPassRegistry());
  BFI = new BlockFrequencyImpl<BasicBlock, Function, BranchProbabilityInfo>();
}

BlockFrequency::~BlockFrequency() {
  delete BFI;
}

void BlockFrequency::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<BranchProbabilityInfo>();
  AU.setPreservesAll();
}

bool BlockFrequency::runOnFunction(Function &F) {
  BranchProbabilityInfo &BPI = getAnalysis<BranchProbabilityInfo>();
  BFI->doFunction(&F, &BPI);
  return false;
}

/// getblockFreq - Return block frequency. Return 0 if we don't have the
/// information. Please note that initial frequency is equal to 1024. It means
/// that we should not rely on the value itself, but only on the comparison to
/// the other block frequencies. We do this to avoid using of floating points.
///
uint32_t BlockFrequency::getBlockFreq(BasicBlock *BB) {
  return BFI->getBlockFreq(BB);
}
