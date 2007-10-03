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
#include "llvm/Analysis/DominatorInternals.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//  PostDominatorTree Implementation
//===----------------------------------------------------------------------===//

char PostDominatorTree::ID = 0;
char PostDominanceFrontier::ID = 0;
static RegisterPass<PostDominatorTree>
F("postdomtree", "Post-Dominator Tree Construction", true);

bool PostDominatorTree::runOnFunction(Function &F) {
  reset();     // Reset from the last time we were run...
    
  // Initialize the roots list
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    TerminatorInst *Insn = I->getTerminator();
    if (Insn->getNumSuccessors() == 0) {
      // Unreachable block is not a root node.
      if (!isa<UnreachableInst>(Insn))
        Roots.push_back(I);
    }
    
    // Prepopulate maps so that we don't get iterator invalidation issues later.
    IDoms[I] = 0;
    DomTreeNodes[I] = 0;
  }
  
  Vertex.push_back(0);
    
  Calculate<Inverse<BasicBlock*> >(*this, F);
  return false;
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
