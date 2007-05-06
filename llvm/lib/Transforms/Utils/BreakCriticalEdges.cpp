//===- BreakCriticalEdges.cpp - Critical Edge Elimination Pass ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// BreakCriticalEdges pass - Break all of the critical edges in the CFG by
// inserting a dummy basic block.  This pass may be "required" by passes that
// cannot deal with critical edges.  For this usage, the structure type is
// forward declared.  This pass obviously invalidates the CFG, but can update
// forward dominator (set, immediate dominators, tree, and frontier)
// information.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "break-crit-edges"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumBroken, "Number of blocks inserted");

namespace {
  struct VISIBILITY_HIDDEN BreakCriticalEdges : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    BreakCriticalEdges() : FunctionPass((intptr_t)&ID) {}

    virtual bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addPreserved<ETForest>();
      AU.addPreserved<DominatorTree>();
      AU.addPreserved<DominanceFrontier>();
      AU.addPreserved<LoopInfo>();

      // No loop canonicalization guarantees are broken by this pass.
      AU.addPreservedID(LoopSimplifyID);
    }
  };

  char BreakCriticalEdges::ID = 0;
  RegisterPass<BreakCriticalEdges> X("break-crit-edges",
                                    "Break critical edges in CFG");
}

// Publically exposed interface to pass...
const PassInfo *llvm::BreakCriticalEdgesID = X.getPassInfo();
FunctionPass *llvm::createBreakCriticalEdgesPass() {
  return new BreakCriticalEdges();
}

// runOnFunction - Loop over all of the edges in the CFG, breaking critical
// edges as they are found.
//
bool BreakCriticalEdges::runOnFunction(Function &F) {
  bool Changed = false;
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    TerminatorInst *TI = I->getTerminator();
    if (TI->getNumSuccessors() > 1)
      for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
        if (SplitCriticalEdge(TI, i, this)) {
          ++NumBroken;
          Changed = true;
        }
  }

  return Changed;
}

//===----------------------------------------------------------------------===//
//    Implementation of the external critical edge manipulation functions
//===----------------------------------------------------------------------===//

// isCriticalEdge - Return true if the specified edge is a critical edge.
// Critical edges are edges from a block with multiple successors to a block
// with multiple predecessors.
//
bool llvm::isCriticalEdge(const TerminatorInst *TI, unsigned SuccNum,
                          bool AllowIdenticalEdges) {
  assert(SuccNum < TI->getNumSuccessors() && "Illegal edge specification!");
  if (TI->getNumSuccessors() == 1) return false;

  const BasicBlock *Dest = TI->getSuccessor(SuccNum);
  pred_const_iterator I = pred_begin(Dest), E = pred_end(Dest);

  // If there is more than one predecessor, this is a critical edge...
  assert(I != E && "No preds, but we have an edge to the block?");
  const BasicBlock *FirstPred = *I;
  ++I;        // Skip one edge due to the incoming arc from TI.
  if (!AllowIdenticalEdges)
    return I != E;
  
  // If AllowIdenticalEdges is true, then we allow this edge to be considered
  // non-critical iff all preds come from TI's block.
  for (; I != E; ++I)
    if (*I != FirstPred) return true;
  return false;
}

// SplitCriticalEdge - If this edge is a critical edge, insert a new node to
// split the critical edge.  This will update ETForest, ImmediateDominator,
// DominatorTree, and DominatorFrontier information if it is available, thus
// calling this pass will not invalidate any of them.  This returns true if
// the edge was split, false otherwise.  This ensures that all edges to that
// dest go to one block instead of each going to a different block.
//
bool llvm::SplitCriticalEdge(TerminatorInst *TI, unsigned SuccNum, Pass *P,
                             bool MergeIdenticalEdges) {
  if (!isCriticalEdge(TI, SuccNum, MergeIdenticalEdges)) return false;
  BasicBlock *TIBB = TI->getParent();
  BasicBlock *DestBB = TI->getSuccessor(SuccNum);

  // Create a new basic block, linking it into the CFG.
  BasicBlock *NewBB = new BasicBlock(TIBB->getName() + "." +
                                     DestBB->getName() + "_crit_edge");
  // Create our unconditional branch...
  new BranchInst(DestBB, NewBB);

  // Branch to the new block, breaking the edge.
  TI->setSuccessor(SuccNum, NewBB);

  // Insert the block into the function... right after the block TI lives in.
  Function &F = *TIBB->getParent();
  Function::iterator FBBI = TIBB;
  F.getBasicBlockList().insert(++FBBI, NewBB);
  
  // If there are any PHI nodes in DestBB, we need to update them so that they
  // merge incoming values from NewBB instead of from TIBB.
  //
  for (BasicBlock::iterator I = DestBB->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);
    // We no longer enter through TIBB, now we come in through NewBB.  Revector
    // exactly one entry in the PHI node that used to come from TIBB to come
    // from NewBB.
    int BBIdx = PN->getBasicBlockIndex(TIBB);
    PN->setIncomingBlock(BBIdx, NewBB);
  }
  
  // If there are any other edges from TIBB to DestBB, update those to go
  // through the split block, making those edges non-critical as well (and
  // reducing the number of phi entries in the DestBB if relevant).
  if (MergeIdenticalEdges) {
    for (unsigned i = SuccNum+1, e = TI->getNumSuccessors(); i != e; ++i) {
      if (TI->getSuccessor(i) != DestBB) continue;
      
      // Remove an entry for TIBB from DestBB phi nodes.
      DestBB->removePredecessor(TIBB);
      
      // We found another edge to DestBB, go to NewBB instead.
      TI->setSuccessor(i, NewBB);
    }
  }
  
  

  // If we don't have a pass object, we can't update anything...
  if (P == 0) return true;

  // Now update analysis information.  Since the only predecessor of NewBB is
  // the TIBB, TIBB clearly dominates NewBB.  TIBB usually doesn't dominate
  // anything, as there are other successors of DestBB.  However, if all other
  // predecessors of DestBB are already dominated by DestBB (e.g. DestBB is a
  // loop header) then NewBB dominates DestBB.
  SmallVector<BasicBlock*, 8> OtherPreds;

  for (pred_iterator I = pred_begin(DestBB), E = pred_end(DestBB); I != E; ++I)
    if (*I != NewBB)
      OtherPreds.push_back(*I);
  
  bool NewBBDominatesDestBB = true;
  
  // Update the forest?
  if (ETForest *EF = P->getAnalysisToUpdate<ETForest>()) {
    // NewBB is dominated by TIBB.
    EF->addNewBlock(NewBB, TIBB);
    
    // If NewBBDominatesDestBB hasn't been computed yet, do so with EF.
    if (!OtherPreds.empty()) {
      while (!OtherPreds.empty() && NewBBDominatesDestBB) {
        NewBBDominatesDestBB = EF->dominates(DestBB, OtherPreds.back());
        OtherPreds.pop_back();
      }
      OtherPreds.clear();
    }
    
    // If NewBBDominatesDestBB, then NewBB dominates DestBB, otherwise it
    // doesn't dominate anything.
    if (NewBBDominatesDestBB)
      EF->setImmediateDominator(DestBB, NewBB);
  }
  
  // Should we update DominatorTree information?
  if (DominatorTree *DT = P->getAnalysisToUpdate<DominatorTree>()) {
    DominatorTree::Node *TINode = DT->getNode(TIBB);

    // The new block is not the immediate dominator for any other nodes, but
    // TINode is the immediate dominator for the new node.
    //
    if (TINode) {       // Don't break unreachable code!
      DominatorTree::Node *NewBBNode = DT->createNewNode(NewBB, TINode);
      DominatorTree::Node *DestBBNode = 0;
     
      // If NewBBDominatesDestBB hasn't been computed yet, do so with DT.
      if (!OtherPreds.empty()) {
        DestBBNode = DT->getNode(DestBB);
        while (!OtherPreds.empty() && NewBBDominatesDestBB) {
          if (DominatorTree::Node *OPNode = DT->getNode(OtherPreds.back()))
            NewBBDominatesDestBB = DestBBNode->dominates(OPNode);
          OtherPreds.pop_back();
        }
        OtherPreds.clear();
      }
      
      // If NewBBDominatesDestBB, then NewBB dominates DestBB, otherwise it
      // doesn't dominate anything.
      if (NewBBDominatesDestBB) {
        if (!DestBBNode) DestBBNode = DT->getNode(DestBB);
        DT->changeImmediateDominator(DestBBNode, NewBBNode);
      }
    }
  }

  // Should we update DominanceFrontier information?
  if (DominanceFrontier *DF = P->getAnalysisToUpdate<DominanceFrontier>()) {
    // If NewBBDominatesDestBB hasn't been computed yet, do so with DF.
    if (!OtherPreds.empty()) {
      // FIXME: IMPLEMENT THIS!
      assert(0 && "Requiring domfrontiers but not idom/domtree/domset."
             " not implemented yet!");
    }
    
    // Since the new block is dominated by its only predecessor TIBB,
    // it cannot be in any block's dominance frontier.  If NewBB dominates
    // DestBB, its dominance frontier is the same as DestBB's, otherwise it is
    // just {DestBB}.
    DominanceFrontier::DomSetType NewDFSet;
    if (NewBBDominatesDestBB) {
      DominanceFrontier::iterator I = DF->find(DestBB);
      if (I != DF->end())
        DF->addBasicBlock(NewBB, I->second);
      else
        DF->addBasicBlock(NewBB, DominanceFrontier::DomSetType());
    } else {
      DominanceFrontier::DomSetType NewDFSet;
      NewDFSet.insert(DestBB);
      DF->addBasicBlock(NewBB, NewDFSet);
    }
  }
  
  // Update LoopInfo if it is around.
  if (LoopInfo *LI = P->getAnalysisToUpdate<LoopInfo>()) {
    // If one or the other blocks were not in a loop, the new block is not
    // either, and thus LI doesn't need to be updated.
    if (Loop *TIL = LI->getLoopFor(TIBB))
      if (Loop *DestLoop = LI->getLoopFor(DestBB)) {
        if (TIL == DestLoop) {
          // Both in the same loop, the NewBB joins loop.
          DestLoop->addBasicBlockToLoop(NewBB, *LI);
        } else if (TIL->contains(DestLoop->getHeader())) {
          // Edge from an outer loop to an inner loop.  Add to the outer loop.
          TIL->addBasicBlockToLoop(NewBB, *LI);
        } else if (DestLoop->contains(TIL->getHeader())) {
          // Edge from an inner loop to an outer loop.  Add to the outer loop.
          DestLoop->addBasicBlockToLoop(NewBB, *LI);
        } else {
          // Edge from two loops with no containment relation.  Because these
          // are natural loops, we know that the destination block must be the
          // header of its loop (adding a branch into a loop elsewhere would
          // create an irreducible loop).
          assert(DestLoop->getHeader() == DestBB &&
                 "Should not create irreducible loops!");
          if (Loop *P = DestLoop->getParentLoop())
            P->addBasicBlockToLoop(NewBB, *LI);
        }
      }
  }
  return true;
}
