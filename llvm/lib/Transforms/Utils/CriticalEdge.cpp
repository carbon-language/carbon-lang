//===-- CriticalEdge.cpp - Functions to detect and split critical edges ---===//
//
// These functions are here to detect and split critical edges in the CFG.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Local.h"
#include "llvm/iTerminators.h"
#include "llvm/iPHINode.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Support/CFG.h"

// isCriticalEdge - Return true if the specified edge is a critical edge.
// Critical edges are edges from a block with multiple successors to a block
// with multiple predecessors.
//
bool isCriticalEdge(const TerminatorInst *TI, unsigned SuccNum) {
  assert(SuccNum < TI->getNumSuccessors() && "Illegal edge specification!");
  if (TI->getNumSuccessors() <= 1) return false;

  const BasicBlock *Dest = TI->getSuccessor(SuccNum);
  pred_const_iterator I = pred_begin(Dest), E = pred_end(Dest);

  // If there is more than one predecessor, this is a critical edge...
  assert(I != E && "No preds, but we have an edge to the block?");
  ++I;        // Skip one edge due to the incoming arc from TI.
  return I != E;
}

// SplitCriticalEdge - Insert a new node node to split the critical edge.  This
// will update DominatorSet, ImmediateDominator and DominatorTree information if
// it is available, thus calling this pass will not invalidate either of them.
//
void SplitCriticalEdge(TerminatorInst *TI, unsigned SuccNum, Pass *P) {
  assert(isCriticalEdge(TI, SuccNum) &&
         "Cannot break a critical edge, if it isn't a critical edge");
  BasicBlock *TIBB = TI->getParent();

  // Create a new basic block, linking it into the CFG.
  BasicBlock *NewBB = new BasicBlock(TIBB->getName()+"_crit_edge");
  BasicBlock *DestBB = TI->getSuccessor(SuccNum);
  // Create our unconditional branch...
  BranchInst *BI = new BranchInst(DestBB);
  NewBB->getInstList().push_back(BI);
  
  // Branch to the new block, breaking the edge...
  TI->setSuccessor(SuccNum, NewBB);

  // Insert the block into the function... right after the block TI lives in.
  Function &F = *TIBB->getParent();
  F.getBasicBlockList().insert(TIBB->getNext(), NewBB);

  // If there are any PHI nodes in DestBB, we need to update them so that they
  // merge incoming values from NewBB instead of from TIBB.
  //
  for (BasicBlock::iterator I = DestBB->begin();
       PHINode *PN = dyn_cast<PHINode>(&*I); ++I) {
    // We no longer enter through TIBB, now we come in through NewBB.
    PN->replaceUsesOfWith(TIBB, NewBB);
  }

  // Now if we have a pass object, update analysis information.  Currently we
  // update DominatorSet and DominatorTree information if it's available.
  //
  if (P) {
    // Should we update DominatorSet information?
    if (DominatorSet *DS = P->getAnalysisToUpdate<DominatorSet>()) {
      // The blocks that dominate the new one are the blocks that dominate TIBB
      // plus the new block itself.
      DominatorSet::DomSetType DomSet = DS->getDominators(TIBB);
      DomSet.insert(NewBB);  // A block always dominates itself.
      DS->addBasicBlock(NewBB, DomSet);
    }

    // Should we update ImmdediateDominator information?
    if (ImmediateDominators *ID =
        P->getAnalysisToUpdate<ImmediateDominators>()) {
      // TIBB is the new immediate dominator for NewBB.  NewBB doesn't dominate
      // anything.
      ID->addNewBlock(NewBB, TIBB);
    }

    // Should we update DominatorTree information?
    if (DominatorTree *DT = P->getAnalysisToUpdate<DominatorTree>()) {
      DominatorTree::Node *TINode = DT->getNode(TIBB);

      // The new block is not the immediate dominator for any other nodes, but
      // TINode is the immediate dominator for the new node.
      //
      DT->createNewNode(NewBB, TINode);
    }
  }
}
