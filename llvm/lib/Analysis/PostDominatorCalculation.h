//==- PostDominatorCalculation.h - Post-Dominator Calculation ----*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Owen Anderson and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// PostDominatorTree calculation implementation.
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_POST_DOMINATOR_CALCULATION_H
#define LLVM_ANALYSIS_POST_DOMINATOR_CALCULATION_H

#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/DominatorInternals.h"

namespace llvm {

void PDTcalculate(PostDominatorTree& PDT, Function &F) {
  // Step #1: Number blocks in depth-first order and initialize variables used
  // in later stages of the algorithm.
  unsigned N = 0;
  for (unsigned i = 0, e = PDT.Roots.size(); i != e; ++i)
    N = DFSPass<GraphTraits<Inverse<BasicBlock*> > >(PDT, PDT.Roots[i], N);
  
  for (unsigned i = N; i >= 2; --i) {
    BasicBlock *W = PDT.Vertex[i];
    PostDominatorTree::InfoRec &WInfo = PDT.Info[W];
    
    // Step #2: Calculate the semidominators of all vertices
    for (succ_iterator SI = succ_begin(W), SE = succ_end(W); SI != SE; ++SI)
      if (PDT.Info.count(*SI)) {  // Only if this predecessor is reachable!
        unsigned SemiU =
             PDT.Info[Eval<GraphTraits<Inverse<BasicBlock*> > >(PDT, *SI)].Semi;
        if (SemiU < WInfo.Semi)
          WInfo.Semi = SemiU;
      }
        
    PDT.Info[PDT.Vertex[WInfo.Semi]].Bucket.push_back(W);
    
    BasicBlock *WParent = WInfo.Parent;
    Link<GraphTraits<Inverse<BasicBlock*> > >(PDT, WParent, W, WInfo);
    
    // Step #3: Implicitly define the immediate dominator of vertices
    std::vector<BasicBlock*> &WParentBucket = PDT.Info[WParent].Bucket;
    while (!WParentBucket.empty()) {
      BasicBlock *V = WParentBucket.back();
      WParentBucket.pop_back();
      BasicBlock *U = Eval<GraphTraits<Inverse<BasicBlock*> > >(PDT, V);
      PDT.IDoms[V] = PDT.Info[U].Semi < PDT.Info[V].Semi ? U : WParent;
    }
  }
  
  // Step #4: Explicitly define the immediate dominator of each vertex
  for (unsigned i = 2; i <= N; ++i) {
    BasicBlock *W = PDT.Vertex[i];
    BasicBlock *&WIDom = PDT.IDoms[W];
    if (WIDom != PDT.Vertex[PDT.Info[W].Semi])
      WIDom = PDT.IDoms[WIDom];
  }
  
  if (PDT.Roots.empty()) return;

  // Add a node for the root.  This node might be the actual root, if there is
  // one exit block, or it may be the virtual exit (denoted by (BasicBlock *)0)
  // which postdominates all real exits if there are multiple exit blocks.
  BasicBlock *Root = PDT.Roots.size() == 1 ? PDT.Roots[0] : 0;
  PDT.DomTreeNodes[Root] = PDT.RootNode = new DomTreeNode(Root, 0);
  
  // Loop over all of the reachable blocks in the function...
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
    if (BasicBlock *ImmPostDom = PDT.getIDom(I)) {  // Reachable block.
      DomTreeNode *&BBNode = PDT.DomTreeNodes[I];
      if (!BBNode) {  // Haven't calculated this node yet?
                      // Get or calculate the node for the immediate dominator
        DomTreeNode *IPDomNode = PDT.getNodeForBlock(ImmPostDom);
        
        // Add a new tree node for this BasicBlock, and link it as a child of
        // IDomNode
        DomTreeNode *C = new DomTreeNode(I, IPDomNode);
        PDT.DomTreeNodes[I] = C;
        BBNode = IPDomNode->addChild(C);
      }
    }

  // Free temporary memory used to construct idom's
  PDT.IDoms.clear();
  PDT.Info.clear();
  std::vector<BasicBlock*>().swap(PDT.Vertex);

  // Start out with the DFS numbers being invalid.  Let them be computed if
  // demanded.
  PDT.DFSInfoValid = false;
}

}
#endif
