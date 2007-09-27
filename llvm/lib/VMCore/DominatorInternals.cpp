//==- DominatorInternals.cpp - Dominator Calculation -------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Owen Anderson and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_LLVM_ANALYSIS_DOMINATOR_INTERNALS_H
#define LIB_LLVM_ANALYSIS_DOMINATOR_INTERNALS_H

#include "llvm/Analysis/Dominators.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
//===----------------------------------------------------------------------===//
//
// DominatorTree construction - This pass constructs immediate dominator
// information for a flow-graph based on the algorithm described in this
// document:
//
//   A Fast Algorithm for Finding Dominators in a Flowgraph
//   T. Lengauer & R. Tarjan, ACM TOPLAS July 1979, pgs 121-141.
//
// This implements both the O(n*ack(n)) and the O(n*log(n)) versions of EVAL and
// LINK, but it turns out that the theoretically slower O(n*log(n))
// implementation is actually faster than the "efficient" algorithm (even for
// large CFGs) because the constant overheads are substantially smaller.  The
// lower-complexity version can be enabled with the following #define:
//
#define BALANCE_IDOM_TREE 0
//
//===----------------------------------------------------------------------===//

namespace llvm {

void Compress(DominatorTreeBase& DT, BasicBlock *VIn) {

  std::vector<BasicBlock *> Work;
  SmallPtrSet<BasicBlock *, 32> Visited;
  BasicBlock *VInAncestor = DT.Info[VIn].Ancestor;
  DominatorTreeBase::InfoRec &VInVAInfo = DT.Info[VInAncestor];

  if (VInVAInfo.Ancestor != 0)
    Work.push_back(VIn);
  
  while (!Work.empty()) {
    BasicBlock *V = Work.back();
    DominatorTree::InfoRec &VInfo = DT.Info[V];
    BasicBlock *VAncestor = VInfo.Ancestor;
    DominatorTreeBase::InfoRec &VAInfo = DT.Info[VAncestor];

    // Process Ancestor first
    if (Visited.insert(VAncestor) &&
        VAInfo.Ancestor != 0) {
      Work.push_back(VAncestor);
      continue;
    } 
    Work.pop_back(); 

    // Update VInfo based on Ancestor info
    if (VAInfo.Ancestor == 0)
      continue;
    BasicBlock *VAncestorLabel = VAInfo.Label;
    BasicBlock *VLabel = VInfo.Label;
    if (DT.Info[VAncestorLabel].Semi < DT.Info[VLabel].Semi)
      VInfo.Label = VAncestorLabel;
    VInfo.Ancestor = VAInfo.Ancestor;
  }
}

BasicBlock *Eval(DominatorTreeBase& DT, BasicBlock *V) {
                 DominatorTreeBase::InfoRec &VInfo = DT.Info[V];
#if !BALANCE_IDOM_TREE
  // Higher-complexity but faster implementation
  if (VInfo.Ancestor == 0)
    return V;
  Compress(DT, V);
  return VInfo.Label;
#else
  // Lower-complexity but slower implementation
  if (VInfo.Ancestor == 0)
    return VInfo.Label;
  Compress(DT, V);
  BasicBlock *VLabel = VInfo.Label;

  BasicBlock *VAncestorLabel = DT.Info[VInfo.Ancestor].Label;
  if (DT.Info[VAncestorLabel].Semi >= DT.Info[VLabel].Semi)
    return VLabel;
  else
    return VAncestorLabel;
#endif
}

void Link(DominatorTreeBase& DT, BasicBlock *V, BasicBlock *W,
          DominatorTreeBase::InfoRec &WInfo) {
#if !BALANCE_IDOM_TREE
  // Higher-complexity but faster implementation
  WInfo.Ancestor = V;
#else
  // Lower-complexity but slower implementation
  BasicBlock *WLabel = WInfo.Label;
  unsigned WLabelSemi = DT.Info[WLabel].Semi;
  BasicBlock *S = W;
  InfoRec *SInfo = &DT.Info[S];

  BasicBlock *SChild = SInfo->Child;
  InfoRec *SChildInfo = &DT.Info[SChild];

  while (WLabelSemi < DT.Info[SChildInfo->Label].Semi) {
    BasicBlock *SChildChild = SChildInfo->Child;
    if (SInfo->Size+DT.Info[SChildChild].Size >= 2*SChildInfo->Size) {
      SChildInfo->Ancestor = S;
      SInfo->Child = SChild = SChildChild;
      SChildInfo = &DT.Info[SChild];
    } else {
      SChildInfo->Size = SInfo->Size;
      S = SInfo->Ancestor = SChild;
      SInfo = SChildInfo;
      SChild = SChildChild;
      SChildInfo = &DT.Info[SChild];
    }
  }

  DominatorTreeBase::InfoRec &VInfo = DT.Info[V];
  SInfo->Label = WLabel;

  assert(V != W && "The optimization here will not work in this case!");
  unsigned WSize = WInfo.Size;
  unsigned VSize = (VInfo.Size += WSize);

  if (VSize < 2*WSize)
    std::swap(S, VInfo.Child);

  while (S) {
    SInfo = &DT.Info[S];
    SInfo->Ancestor = V;
    S = SInfo->Child;
  }
#endif
}

}

#endif
