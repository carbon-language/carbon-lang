//===- MachineBranchProbabilityInfo.cpp - Machine Branch Probability Info -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This analysis uses probability info stored in Machine Basic Blocks.
//
//===----------------------------------------------------------------------===//

#include "llvm/Instructions.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

INITIALIZE_PASS_BEGIN(MachineBranchProbabilityInfo, "machine-branch-prob",
                      "Machine Branch Probability Analysis", false, true)
INITIALIZE_PASS_END(MachineBranchProbabilityInfo, "machine-branch-prob",
                    "Machine Branch Probability Analysis", false, true)

char MachineBranchProbabilityInfo::ID = 0;

uint32_t MachineBranchProbabilityInfo::
getSumForBlock(MachineBasicBlock *MBB) const {
  uint32_t Sum = 0;

  for (MachineBasicBlock::const_succ_iterator I = MBB->succ_begin(),
       E = MBB->succ_end(); I != E; ++I) {
    MachineBasicBlock *Succ = *I;
    uint32_t Weight = getEdgeWeight(MBB, Succ);
    uint32_t PrevSum = Sum;

    Sum += Weight;
    assert(Sum > PrevSum); (void) PrevSum;
  }

  return Sum;
}

uint32_t
MachineBranchProbabilityInfo::getEdgeWeight(MachineBasicBlock *Src,
                                            MachineBasicBlock *Dst) const {
  uint32_t Weight = Src->getSuccWeight(Dst);
  if (!Weight)
    return DEFAULT_WEIGHT;
  return Weight;
}

bool MachineBranchProbabilityInfo::isEdgeHot(MachineBasicBlock *Src,
                                             MachineBasicBlock *Dst) const {
  // Hot probability is at least 4/5 = 80%
  // FIXME: Compare against a static "hot" BranchProbability.
  return getEdgeProbability(Src, Dst) > BranchProbability(4, 5);
}

MachineBasicBlock *
MachineBranchProbabilityInfo::getHotSucc(MachineBasicBlock *MBB) const {
  uint32_t Sum = 0;
  uint32_t MaxWeight = 0;
  MachineBasicBlock *MaxSucc = 0;

  for (MachineBasicBlock::const_succ_iterator I = MBB->succ_begin(),
       E = MBB->succ_end(); I != E; ++I) {
    MachineBasicBlock *Succ = *I;
    uint32_t Weight = getEdgeWeight(MBB, Succ);
    uint32_t PrevSum = Sum;

    Sum += Weight;
    assert(Sum > PrevSum); (void) PrevSum;

    if (Weight > MaxWeight) {
      MaxWeight = Weight;
      MaxSucc = Succ;
    }
  }

  if (BranchProbability(MaxWeight, Sum) >= BranchProbability(4, 5))
    return MaxSucc;

  return 0;
}

BranchProbability
MachineBranchProbabilityInfo::getEdgeProbability(MachineBasicBlock *Src,
                                                 MachineBasicBlock *Dst) const {
  uint32_t N = getEdgeWeight(Src, Dst);
  uint32_t D = getSumForBlock(Src);

  return BranchProbability(N, D);
}

raw_ostream &MachineBranchProbabilityInfo::
printEdgeProbability(raw_ostream &OS, MachineBasicBlock *Src,
                     MachineBasicBlock *Dst) const {

  const BranchProbability Prob = getEdgeProbability(Src, Dst);
  OS << "edge MBB#" << Src->getNumber() << " -> MBB#" << Dst->getNumber()
     << " probability is "  << Prob 
     << (isEdgeHot(Src, Dst) ? " [HOT edge]\n" : "\n");

  return OS;
}
