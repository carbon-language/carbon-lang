//===- PostDominators.cpp - Post-Dominator Calculation --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the post-dominator construction algorithms.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/PostDominators.h"
#include "llvm/Instructions.h"
#include "llvm/Support/CFG.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SetOperations.h"
#include "PostDominatorCalculation.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//  PostDominatorTree Implementation
//===----------------------------------------------------------------------===//

char PostDominatorTree::ID = 0;
char PostDominanceFrontier::ID = 0;
static RegisterPass<PostDominatorTree>
F("postdomtree", "Post-Dominator Tree Construction", true);

unsigned PostDominatorTree::DFSPass(BasicBlock *V, unsigned N) {
  std::vector<BasicBlock *> workStack;
  SmallPtrSet<BasicBlock *, 32> Visited;
  workStack.push_back(V);

  do {
    BasicBlock *currentBB = workStack.back();
    InfoRec &CurVInfo = Info[currentBB];

    // Visit each block only once.
    if (Visited.insert(currentBB)) {
      CurVInfo.Semi = ++N;
      CurVInfo.Label = currentBB;
      
      Vertex.push_back(currentBB);  // Vertex[n] = current;
      // Info[currentBB].Ancestor = 0;     
      // Ancestor[n] = 0
      // Child[currentBB] = 0;
      CurVInfo.Size = 1;       // Size[currentBB] = 1
    }

    // Visit children
    bool visitChild = false;
    for (pred_iterator PI = pred_begin(currentBB), PE = pred_end(currentBB); 
         PI != PE && !visitChild; ++PI) {
      InfoRec &SuccVInfo = Info[*PI];
      if (SuccVInfo.Semi == 0) {
        SuccVInfo.Parent = currentBB;
        if (!Visited.count(*PI)) {
          workStack.push_back(*PI);   
          visitChild = true;
        }
      }
    }

    // If all children are visited or if this block has no child then pop this
    // block out of workStack.
    if (!visitChild)
      workStack.pop_back();

  } while (!workStack.empty());

  return N;
}

//===----------------------------------------------------------------------===//
//  PostDominanceFrontier Implementation
//===----------------------------------------------------------------------===//

static RegisterPass<PostDominanceFrontier>
H("postdomfrontier", "Post-Dominance Frontier Construction", true);

const DominanceFrontier::DomSetType &
PostDominanceFrontier::calculate(const PostDominatorTree &DT,
                                 const DomTreeNode *Node) {
  // Loop over CFG successors to calculate DFlocal[Node]
  BasicBlock *BB = Node->getBlock();
  DomSetType &S = Frontiers[BB];       // The new set to fill in...
  if (getRoots().empty()) return S;

  if (BB)
    for (pred_iterator SI = pred_begin(BB), SE = pred_end(BB);
         SI != SE; ++SI) {
      // Does Node immediately dominate this predecessor?
      DomTreeNode *SINode = DT[*SI];
      if (SINode && SINode->getIDom() != Node)
        S.insert(*SI);
    }

  // At this point, S is DFlocal.  Now we union in DFup's of our children...
  // Loop through and visit the nodes that Node immediately dominates (Node's
  // children in the IDomTree)
  //
  for (DomTreeNode::const_iterator
         NI = Node->begin(), NE = Node->end(); NI != NE; ++NI) {
    DomTreeNode *IDominee = *NI;
    const DomSetType &ChildDF = calculate(DT, IDominee);

    DomSetType::const_iterator CDFI = ChildDF.begin(), CDFE = ChildDF.end();
    for (; CDFI != CDFE; ++CDFI) {
      if (!DT.properlyDominates(Node, DT[*CDFI]))
        S.insert(*CDFI);
    }
  }

  return S;
}

// Ensure that this .cpp file gets linked when PostDominators.h is used.
DEFINING_FILE_FOR(PostDominanceFrontier)
