//=- MachineBranchProbabilityInfo.h - Branch Probability Analysis -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is used to evaluate branch probabilties on machine basic blocks.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEBRANCHPROBABILITYINFO_H
#define LLVM_CODEGEN_MACHINEBRANCHPROBABILITYINFO_H

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/Pass.h"
#include "llvm/Support/BranchProbability.h"
#include <climits>
#include <numeric>

namespace llvm {

class MachineBranchProbabilityInfo : public ImmutablePass {
  virtual void anchor();

  // Default weight value. Used when we don't have information about the edge.
  // TODO: DEFAULT_WEIGHT makes sense during static predication, when none of
  // the successors have a weight yet. But it doesn't make sense when providing
  // weight to an edge that may have siblings with non-zero weights. This can
  // be handled various ways, but it's probably fine for an edge with unknown
  // weight to just "inherit" the non-zero weight of an adjacent successor.
  static const uint32_t DEFAULT_WEIGHT = 16;

public:
  static char ID;

  MachineBranchProbabilityInfo() : ImmutablePass(ID) {
    PassRegistry &Registry = *PassRegistry::getPassRegistry();
    initializeMachineBranchProbabilityInfoPass(Registry);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  // Return edge weight. If we don't have any informations about it - return
  // DEFAULT_WEIGHT.
  uint32_t getEdgeWeight(const MachineBasicBlock *Src,
                         const MachineBasicBlock *Dst) const;

  // Same thing, but using a const_succ_iterator from Src. This is faster when
  // the iterator is already available.
  uint32_t getEdgeWeight(const MachineBasicBlock *Src,
                         MachineBasicBlock::const_succ_iterator Dst) const;

  // Get sum of the block successors' weights, potentially scaling them to fit
  // within 32-bits. If scaling is required, sets Scale based on the necessary
  // adjustment. Any edge weights used with the sum should be divided by Scale.
  uint32_t getSumForBlock(const MachineBasicBlock *MBB, uint32_t &Scale) const;

  // A 'Hot' edge is an edge which probability is >= 80%.
  bool isEdgeHot(const MachineBasicBlock *Src,
                 const MachineBasicBlock *Dst) const;

  // Return a hot successor for the block BB or null if there isn't one.
  // NB: This routine's complexity is linear on the number of successors.
  MachineBasicBlock *getHotSucc(MachineBasicBlock *MBB) const;

  // Return a probability as a fraction between 0 (0% probability) and
  // 1 (100% probability), however the value is never equal to 0, and can be 1
  // only iff SRC block has only one successor.
  // NB: This routine's complexity is linear on the number of successors of
  // Src. Querying sequentially for each successor's probability is a quadratic
  // query pattern.
  BranchProbability getEdgeProbability(const MachineBasicBlock *Src,
                                       const MachineBasicBlock *Dst) const;

  // Print value between 0 (0% probability) and 1 (100% probability),
  // however the value is never equal to 0, and can be 1 only iff SRC block
  // has only one successor.
  raw_ostream &printEdgeProbability(raw_ostream &OS,
                                    const MachineBasicBlock *Src,
                                    const MachineBasicBlock *Dst) const;

  // Normalize a list of weights by scaling them down so that the sum of them
  // doesn't exceed UINT32_MAX. Return the scale.
  template <class WeightListIter>
  static uint32_t normalizeEdgeWeights(WeightListIter Begin,
                                       WeightListIter End);
};

template <class WeightListIter>
uint32_t
MachineBranchProbabilityInfo::normalizeEdgeWeights(WeightListIter Begin,
                                                   WeightListIter End) {
  // First we compute the sum with 64-bits of precision.
  uint64_t Sum = std::accumulate(Begin, End, uint64_t(0));

  // If the computed sum fits in 32-bits, we're done.
  if (Sum <= UINT32_MAX)
    return 1;

  // Otherwise, compute the scale necessary to cause the weights to fit, and
  // re-sum with that scale applied.
  assert((Sum / UINT32_MAX) < UINT32_MAX &&
         "The sum of weights exceeds UINT32_MAX^2!");
  uint32_t Scale = (Sum / UINT32_MAX) + 1;
  for (auto I = Begin; I != End; ++I)
    *I /= Scale;
  return Scale;
}

}


#endif
