//==- Dominators.h - Construct the Dominance Tree Given CFG -----*- C++ --*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple, fast dominance algorithm for source-level
// CFGs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DOMINATORS_H
#define LLVM_CLANG_DOMINATORS_H

#include "clang/Analysis/CFG.h"
#include "clang/Analysis/AnalysisContext.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {

class CFG;
class CFGBlock;

class DominatorTree : public ManagedAnalysis {
  typedef llvm::DenseMap<const CFGBlock *, CFGBlock*> CFGBlockMapTy;

public:
  DominatorTree(AnalysisDeclContext &ac)
      : AC(ac) {}

  virtual ~DominatorTree();

  /// Return the immediate dominator node given a CFGBlock.
  /// For entry block, the dominator is itself.
  /// This is the same as using operator[] on this class.
  CFGBlock *getNode(const CFGBlock *B) const;

  /// This returns the Entry Block for the given CFG
  CFGBlock *getRootNode() { return RootNode; }
  const CFGBlock *getRootNode() const { return RootNode; }

  /// Returns true iff A dominates B and A != B.
  /// Note that this is not a constant time operation.
  bool properlyDominates(const CFGBlock *A, const CFGBlock *B) const;

  /// Returns true iff A dominates B.
  bool dominates(const CFGBlock *A, const CFGBlock *B) const;

  /// Find nearest common dominator for blocks A and B.
  /// Common dominator always exists, ex: entry block.
  const CFGBlock *findNearestCommonDominator(const CFGBlock *A,
                                       const CFGBlock *B) const;

  /// Constructs immediate dominator tree for a given CFG based on the algorithm
  /// described in this paper:
  ///
  ///  A Simple, Fast Dominance Algorithm
  ///  Keith D. Cooper, Timothy J. Harvey and Ken Kennedy
  ///  Software-Practice and Expreience, 2001;4:1-10.
  ///
  /// This implementation is simple and runs faster in practice than the classis
  /// Lengauer-Tarjan algorithm. For detailed discussions, refer to the paper.
  void BuildDominatorTree();

  /// Dump the immediate dominance tree
  void dump();

private:
  AnalysisDeclContext &AC;
  CFGBlock *RootNode;
  CFGBlockMapTy IDoms;
};

} // end namespace clang

#endif

