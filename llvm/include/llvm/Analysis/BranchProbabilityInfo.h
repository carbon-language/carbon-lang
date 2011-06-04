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
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/Debug.h"

namespace llvm {

class BranchProbabilityInfo : public FunctionPass {

  // Default weight value. Used when we don't have information about the edge.
  static const unsigned int DEFAULT_WEIGHT = 16;

  typedef std::pair<BasicBlock *, BasicBlock *> Edge;

  DenseMap<Edge, unsigned> Weights;

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

  // Returned value is between 1 and UINT_MAX. Look at BranchProbabilityInfo.cpp
  // for details.
  unsigned getEdgeWeight(BasicBlock *Src, BasicBlock *Dst) const;

  // Look at BranchProbabilityInfo.cpp for details. Use it with caution!
  void setEdgeWeight(BasicBlock *Src, BasicBlock *Dst, unsigned Weight);

  // A 'Hot' edge is an edge which probability is >= 80%.
  bool isEdgeHot(BasicBlock *Src, BasicBlock *Dst) const;

  // Return a hot successor for the block BB or null if there isn't one.
  BasicBlock *getHotSucc(BasicBlock *BB) const;

  // Print value between 0 (0% probability) and 1 (100% probability),
  // however the value is never equal to 0, and can be 1 only iff SRC block
  // has only one successor.
  raw_ostream &printEdgeProbability(raw_ostream &OS, BasicBlock *Src,
                                   BasicBlock *Dst) const;
};

}

#endif
