//=- llvm/Analysis/PostDominators.h - Post Dominator Calculation -*- C++ -*--=//
//
// This file exposes interfaces to post dominance information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_POST_DOMINATORS_H
#define LLVM_ANALYSIS_POST_DOMINATORS_H

#include "llvm/Analysis/Dominators.h"


//===-------------------------------------
// DominatorSet Class - Concrete subclass of DominatorSetBase that is used to
// compute the post-dominator set.
//
struct PostDominatorSet : public DominatorSetBase {
  PostDominatorSet() : DominatorSetBase(true) {}

  virtual bool runOnFunction(Function &F);

  // getAnalysisUsage - This obviously provides a dominator set, but it also
  // uses the UnifyFunctionExitNode pass if building post-dominators
  //
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
};



//===-------------------------------------
// ImmediatePostDominators Class - Concrete subclass of ImmediateDominatorsBase
// that is used to compute the immediate post-dominators.
//
struct ImmediatePostDominators : public ImmediateDominatorsBase {
  ImmediatePostDominators() : ImmediateDominatorsBase(true) {}

  virtual bool runOnFunction(Function &F) {
    IDoms.clear();     // Reset from the last time we were run...
    PostDominatorSet &DS = getAnalysis<PostDominatorSet>();
    Root = DS.getRoot();
    calcIDoms(DS);
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<PostDominatorSet>();
  }
};


//===-------------------------------------
// PostDominatorTree Class - Concrete subclass of DominatorTree that is used to
// compute the a post-dominator tree.
//
struct PostDominatorTree : public DominatorTreeBase {
  PostDominatorTree() : DominatorTreeBase(true) {}

  virtual bool runOnFunction(Function &F) {
    reset();     // Reset from the last time we were run...
    PostDominatorSet &DS = getAnalysis<PostDominatorSet>();
    Root = DS.getRoot();
    calculate(DS);
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<PostDominatorSet>();
  }
private:
  void calculate(const PostDominatorSet &DS);
};


//===-------------------------------------
// PostDominanceFrontier Class - Concrete subclass of DominanceFrontier that is
// used to compute the a post-dominance frontier.
//
struct PostDominanceFrontier : public DominanceFrontierBase {
  PostDominanceFrontier() : DominanceFrontierBase(true) {}

  virtual bool runOnFunction(Function &) {
    Frontiers.clear();
    PostDominatorTree &DT = getAnalysis<PostDominatorTree>();
    Root = DT.getRoot();
    calculate(DT, DT[Root]);
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<PostDominatorTree>();
  }

  // stub - dummy function, just ignore it
  static void stub();

private:
  const DomSetType &calculate(const PostDominatorTree &DT,
                              const DominatorTree::Node *Node);
};

// Make sure that any clients of this file link in PostDominators.cpp
static IncludeFile
POST_DOMINATOR_INCLUDE_FILE((void*)&PostDominanceFrontier::stub);

#endif
