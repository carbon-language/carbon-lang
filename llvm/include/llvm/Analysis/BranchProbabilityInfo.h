//===--- BranchProbabilityInfo.h - Branch Probability Analysis --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is used to evaluate branch probabilties.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_BRANCHPROBABILITYINFO_H
#define LLVM_ANALYSIS_BRANCHPROBABILITYINFO_H

#include "llvm/InitializePasses.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Analysis/LoopInfo.h"

namespace llvm {

class raw_ostream;

class BranchProbabilityInfo : public FunctionPass {

  // Default weight value. Used when we don't have information about the edge.
  // TODO: DEFAULT_WEIGHT makes sense during static predication, when none of
  // the successors have a weight yet. But it doesn't make sense when providing
  // weight to an edge that may have siblings with non-zero weights. This can
  // be handled various ways, but it's probably fine for an edge with unknown
  // weight to just "inherit" the non-zero weight of an adjacent successor.
  static const uint32_t DEFAULT_WEIGHT = 16;

  typedef std::pair<BasicBlock *, BasicBlock *> Edge;

  DenseMap<Edge, uint32_t> Weights;

  // Get sum of the block successors' weights.
  uint32_t getSumForBlock(BasicBlock *BB) const;

  // Get sum of the edge weights going to the BB block.
  uint32_t getBackSumForBlock(BasicBlock *BB) const;

public:
  static char ID;

  BranchProbabilityInfo() : FunctionPass(ID) {
    initializeBranchProbabilityInfoPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<LoopInfo>();
    AU.setPreservesAll();
  }

  bool runOnFunction(Function &F);

  // Returned value is between 1 and UINT32_MAX. Look at
  // BranchProbabilityInfo.cpp for details.
  uint32_t getEdgeWeight(BasicBlock *Src, BasicBlock *Dst) const;

  // Look at BranchProbabilityInfo.cpp for details. Use it with caution!
  void setEdgeWeight(BasicBlock *Src, BasicBlock *Dst, uint32_t Weight);

  // A 'Hot' edge is an edge which probability is >= 80%.
  bool isEdgeHot(BasicBlock *Src, BasicBlock *Dst) const;

  // Return a hot successor for the block BB or null if there isn't one.
  BasicBlock *getHotSucc(BasicBlock *BB) const;

  // Return a probability as a fraction between 0 (0% probability) and
  // 1 (100% probability), however the value is never equal to 0, and can be 1
  // only iff SRC block has only one successor.
  BranchProbability getEdgeProbability(BasicBlock *Src, BasicBlock *Dst) const;

  // Return a probability of getting to the DST block through SRC->DST edge.
  // Returned value is a fraction between 0 (0% probability) and
  // 1 (100% probability), however the value is never equal to 0, and can be 1
  // only iff DST block has only one predecesor.
  BranchProbability getBackEdgeProbability(BasicBlock *Src,
                                           BasicBlock *Dst) const;

  // Print value between 0 (0% probability) and 1 (100% probability),
  // however the value is never equal to 0, and can be 1 only iff SRC block
  // has only one successor.
  raw_ostream &printEdgeProbability(raw_ostream &OS, BasicBlock *Src,
                                    BasicBlock *Dst) const;
};

}

#endif
