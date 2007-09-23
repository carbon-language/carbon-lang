//==- DominatorCalculation.h - Dominator Calculation -------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Owen Anderson and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_VMCORE_DOMINATOR_CALCULATION_H
#define LLVM_VMCORE_DOMINATOR_CALCULATION_H

#include "llvm/Analysis/Dominators.h"

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

void DTCompress(DominatorTree& DT, BasicBlock *VIn) {

  std::vector<BasicBlock *> Work;
  SmallPtrSet<BasicBlock *, 32> Visited;
  BasicBlock *VInAncestor = DT.Info[VIn].Ancestor;
  DominatorTree::InfoRec &VInVAInfo = DT.Info[VInAncestor];

  if (VInVAInfo.Ancestor != 0)
    Work.push_back(VIn);
  
  while (!Work.empty()) {
    BasicBlock *V = Work.back();
    DominatorTree::InfoRec &VInfo = DT.Info[V];
    BasicBlock *VAncestor = VInfo.Ancestor;
    DominatorTree::InfoRec &VAInfo = DT.Info[VAncestor];

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

BasicBlock *DTEval(DominatorTree& DT, BasicBlock *V) {
  DominatorTree::InfoRec &VInfo = DT.Info[V];
#if !BALANCE_IDOM_TREE
  // Higher-complexity but faster implementation
  if (VInfo.Ancestor == 0)
    return V;
  DTCompress(DT, V);
  return VInfo.Label;
#else
  // Lower-complexity but slower implementation
  if (VInfo.Ancestor == 0)
    return VInfo.Label;
  DTCompress(DT, V);
  BasicBlock *VLabel = VInfo.Label;

  BasicBlock *VAncestorLabel = DT.Info[VInfo.Ancestor].Label;
  if (DT.Info[VAncestorLabel].Semi >= DT.Info[VLabel].Semi)
    return VLabel;
  else
    return VAncestorLabel;
#endif
}

void DTLink(DominatorTree& DT, BasicBlock *V, BasicBlock *W,
            DominatorTree::InfoRec &WInfo) {
#if !BALANCE_IDOM_TREE
  // Higher-complexity but faster implementation
  WInfo.Ancestor = V;
#else
  // Lower-complexity but slower implementation
  BasicBlock *WLabel = WInfo.Label;
  unsigned WLabelSemi = Info[WLabel].Semi;
  BasicBlock *S = W;
  InfoRec *SInfo = &Info[S];

  BasicBlock *SChild = SInfo->Child;
  InfoRec *SChildInfo = &Info[SChild];

  while (WLabelSemi < Info[SChildInfo->Label].Semi) {
    BasicBlock *SChildChild = SChildInfo->Child;
    if (SInfo->Size+Info[SChildChild].Size >= 2*SChildInfo->Size) {
      SChildInfo->Ancestor = S;
      SInfo->Child = SChild = SChildChild;
      SChildInfo = &Info[SChild];
    } else {
      SChildInfo->Size = SInfo->Size;
      S = SInfo->Ancestor = SChild;
      SInfo = SChildInfo;
      SChild = SChildChild;
      SChildInfo = &Info[SChild];
    }
  }

  InfoRec &VInfo = Info[V];
  SInfo->Label = WLabel;

  assert(V != W && "The optimization here will not work in this case!");
  unsigned WSize = WInfo.Size;
  unsigned VSize = (VInfo.Size += WSize);

  if (VSize < 2*WSize)
    std::swap(S, VInfo.Child);

  while (S) {
    SInfo = &Info[S];
    SInfo->Ancestor = V;
    S = SInfo->Child;
  }
#endif
}

void DTcalculate(DominatorTree& DT, Function &F) {
  BasicBlock* Root = DT.Roots[0];

  // Add a node for the root...
  DT.DomTreeNodes[Root] = DT.RootNode = new DomTreeNode(Root, 0);

  DT.Vertex.push_back(0);

  // Step #1: Number blocks in depth-first order and initialize variables used
  // in later stages of the algorithm.
  unsigned N = DT.DFSPass(Root, 0);

  for (unsigned i = N; i >= 2; --i) {
    BasicBlock *W = DT.Vertex[i];
    DominatorTree::InfoRec &WInfo = DT.Info[W];

    // Step #2: Calculate the semidominators of all vertices
    for (pred_iterator PI = pred_begin(W), E = pred_end(W); PI != E; ++PI)
      if (DT.Info.count(*PI)) {  // Only if this predecessor is reachable!
        unsigned SemiU = DT.Info[DTEval(DT, *PI)].Semi;
        if (SemiU < WInfo.Semi)
          WInfo.Semi = SemiU;
      }

    DT.Info[DT.Vertex[WInfo.Semi]].Bucket.push_back(W);

    BasicBlock *WParent = WInfo.Parent;
    DTLink(DT, WParent, W, WInfo);

    // Step #3: Implicitly define the immediate dominator of vertices
    std::vector<BasicBlock*> &WParentBucket = DT.Info[WParent].Bucket;
    while (!WParentBucket.empty()) {
      BasicBlock *V = WParentBucket.back();
      WParentBucket.pop_back();
      BasicBlock *U = DTEval(DT, V);
      DT.IDoms[V] = DT.Info[U].Semi < DT.Info[V].Semi ? U : WParent;
    }
  }

  // Step #4: Explicitly define the immediate dominator of each vertex
  for (unsigned i = 2; i <= N; ++i) {
    BasicBlock *W = DT.Vertex[i];
    BasicBlock *&WIDom = DT.IDoms[W];
    if (WIDom != DT.Vertex[DT.Info[W].Semi])
      WIDom = DT.IDoms[WIDom];
  }

  // Loop over all of the reachable blocks in the function...
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
    if (BasicBlock *ImmDom = DT.getIDom(I)) {  // Reachable block.
      DomTreeNode *BBNode = DT.DomTreeNodes[I];
      if (BBNode) continue;  // Haven't calculated this node yet?

      // Get or calculate the node for the immediate dominator
      DomTreeNode *IDomNode = DT.getNodeForBlock(ImmDom);

      // Add a new tree node for this BasicBlock, and link it as a child of
      // IDomNode
      DomTreeNode *C = new DomTreeNode(I, IDomNode);
      DT.DomTreeNodes[I] = IDomNode->addChild(C);
    }

  // Free temporary memory used to construct idom's
  DT.Info.clear();
  DT.IDoms.clear();
  std::vector<BasicBlock*>().swap(DT.Vertex);

  DT.updateDFSNumbers();
}

}
#endif