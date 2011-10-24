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
#include "llvm/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/BranchProbability.h"

namespace llvm {
class LoopInfo;
class raw_ostream;

class BranchProbabilityInfo : public FunctionPass {
public:
  static char ID;

  BranchProbabilityInfo() : FunctionPass(ID) {
    initializeBranchProbabilityInfoPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const;
  bool runOnFunction(Function &F);
  void print(raw_ostream &OS, const Module *M = 0) const;

  // Returned value is between 1 and UINT32_MAX. Look at
  // BranchProbabilityInfo.cpp for details.
  uint32_t getEdgeWeight(const BasicBlock *Src, const BasicBlock *Dst) const;

  // Look at BranchProbabilityInfo.cpp for details. Use it with caution!
  void setEdgeWeight(const BasicBlock *Src, const BasicBlock *Dst,
                     uint32_t Weight);

  // A 'Hot' edge is an edge which probability is >= 80%.
  bool isEdgeHot(const BasicBlock *Src, const BasicBlock *Dst) const;

  // Return a hot successor for the block BB or null if there isn't one.
  BasicBlock *getHotSucc(BasicBlock *BB) const;

  // Return a probability as a fraction between 0 (0% probability) and
  // 1 (100% probability), however the value is never equal to 0, and can be 1
  // only iff SRC block has only one successor.
  BranchProbability getEdgeProbability(const BasicBlock *Src,
                                       const BasicBlock *Dst) const;

  // Print value between 0 (0% probability) and 1 (100% probability),
  // however the value is never equal to 0, and can be 1 only iff SRC block
  // has only one successor.
  raw_ostream &printEdgeProbability(raw_ostream &OS, const BasicBlock *Src,
                                    const BasicBlock *Dst) const;

private:
  typedef std::pair<const BasicBlock *, const BasicBlock *> Edge;

  // Default weight value. Used when we don't have information about the edge.
  // TODO: DEFAULT_WEIGHT makes sense during static predication, when none of
  // the successors have a weight yet. But it doesn't make sense when providing
  // weight to an edge that may have siblings with non-zero weights. This can
  // be handled various ways, but it's probably fine for an edge with unknown
  // weight to just "inherit" the non-zero weight of an adjacent successor.
  static const uint32_t DEFAULT_WEIGHT = 16;

  DenseMap<Edge, uint32_t> Weights;

  /// \brief Handle to the LoopInfo analysis.
  LoopInfo *LI;

  /// \brief Track the last function we run over for printing.
  Function *LastF;

  /// \brief Get sum of the block successors' weights.
  uint32_t getSumForBlock(const BasicBlock *BB) const;

  bool calcMetadataWeights(BasicBlock *BB);
  bool calcReturnHeuristics(BasicBlock *BB);
  bool calcPointerHeuristics(BasicBlock *BB);
  bool calcLoopBranchHeuristics(BasicBlock *BB);
  bool calcZeroHeuristics(BasicBlock *BB);
  bool calcFloatingPointHeuristics(BasicBlock *BB);
};

}

#endif
