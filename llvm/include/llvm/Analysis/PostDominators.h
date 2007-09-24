//=- llvm/Analysis/PostDominators.h - Post Dominator Calculation-*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes interfaces to post dominance information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_POST_DOMINATORS_H
#define LLVM_ANALYSIS_POST_DOMINATORS_H

#include "llvm/Analysis/Dominators.h"

namespace llvm {

/// PostDominatorTree Class - Concrete subclass of DominatorTree that is used to
/// compute the a post-dominator tree.
///
struct PostDominatorTree : public DominatorTreeBase {
  static char ID; // Pass identification, replacement for typeid

  PostDominatorTree() : 
    DominatorTreeBase((intptr_t)&ID, true) {}

  virtual bool runOnFunction(Function &F) {
    reset();     // Reset from the last time we were run...
    PDTcalculate(*this, F);
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
private:
  unsigned DFSPass(BasicBlock *V, unsigned N);
  friend void PDTcalculate(PostDominatorTree& PDT, Function &F);
  friend void PDTLink(PostDominatorTree& PDT,BasicBlock *V,
                      BasicBlock *W, InfoRec &WInfo);
};


/// PostDominanceFrontier Class - Concrete subclass of DominanceFrontier that is
/// used to compute the a post-dominance frontier.
///
struct PostDominanceFrontier : public DominanceFrontierBase {
  static char ID;
  PostDominanceFrontier() 
    : DominanceFrontierBase((intptr_t) &ID, true) {}

  virtual bool runOnFunction(Function &) {
    Frontiers.clear();
    PostDominatorTree &DT = getAnalysis<PostDominatorTree>();
    Roots = DT.getRoots();
    if (const DomTreeNode *Root = DT.getRootNode())
      calculate(DT, Root);
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<PostDominatorTree>();
  }

private:
  const DomSetType &calculate(const PostDominatorTree &DT,
                              const DomTreeNode *Node);
};

} // End llvm namespace

// Make sure that any clients of this file link in PostDominators.cpp
FORCE_DEFINING_FILE_TO_BE_LINKED(PostDominanceFrontier)

#endif
