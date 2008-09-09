//===-- BasicBlockUtils.cpp - BasicBlock Utilities -------------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of functions perform manipulations on basic blocks, and
// instructions contained within basic blocks.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Constant.h"
#include "llvm/Type.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/Dominators.h"
#include <algorithm>
using namespace llvm;

/// MergeBlockIntoPredecessor - Attempts to merge a block into its predecessor,
/// if possible.  The return value indicates success or failure.
bool llvm::MergeBlockIntoPredecessor(BasicBlock* BB, Pass* P) {
  pred_iterator PI(pred_begin(BB)), PE(pred_end(BB));
  // Can't merge the entry block.
  if (pred_begin(BB) == pred_end(BB)) return false;
  
  BasicBlock *PredBB = *PI++;
  for (; PI != PE; ++PI)  // Search all predecessors, see if they are all same
    if (*PI != PredBB) {
      PredBB = 0;       // There are multiple different predecessors...
      break;
    }
  
  // Can't merge if there are multiple predecessors.
  if (!PredBB) return false;
  // Don't break self-loops.
  if (PredBB == BB) return false;
  // Don't break invokes.
  if (isa<InvokeInst>(PredBB->getTerminator())) return false;
  
  succ_iterator SI(succ_begin(PredBB)), SE(succ_end(PredBB));
  BasicBlock* OnlySucc = BB;
  for (; SI != SE; ++SI)
    if (*SI != OnlySucc) {
      OnlySucc = 0;     // There are multiple distinct successors!
      break;
    }
  
  // Can't merge if there are multiple successors.
  if (!OnlySucc) return false;

  // Can't merge if there is PHI loop.
  for (BasicBlock::iterator BI = BB->begin(), BE = BB->end(); BI != BE; ++BI) {
    if (PHINode *PN = dyn_cast<PHINode>(BI)) {
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
        if (PN->getIncomingValue(i) == PN)
          return false;
    } else
      break;
  }

  // Begin by getting rid of unneeded PHIs.
  while (PHINode *PN = dyn_cast<PHINode>(&BB->front())) {
    PN->replaceAllUsesWith(PN->getIncomingValue(0));
    BB->getInstList().pop_front();  // Delete the phi node...
  }
  
  // Delete the unconditional branch from the predecessor...
  PredBB->getInstList().pop_back();
  
  // Move all definitions in the successor to the predecessor...
  PredBB->getInstList().splice(PredBB->end(), BB->getInstList());
  
  // Make all PHI nodes that referred to BB now refer to Pred as their
  // source...
  BB->replaceAllUsesWith(PredBB);
  
  // Inherit predecessors name if it exists.
  if (!PredBB->hasName())
    PredBB->takeName(BB);
  
  // Finally, erase the old block and update dominator info.
  if (P) {
    if (DominatorTree* DT = P->getAnalysisToUpdate<DominatorTree>()) {
      DomTreeNode* DTN = DT->getNode(BB);
      DomTreeNode* PredDTN = DT->getNode(PredBB);
  
      if (DTN) {
        SmallPtrSet<DomTreeNode*, 8> Children(DTN->begin(), DTN->end());
        for (SmallPtrSet<DomTreeNode*, 8>::iterator DI = Children.begin(),
             DE = Children.end(); DI != DE; ++DI)
          DT->changeImmediateDominator(*DI, PredDTN);

        DT->eraseNode(BB);
      }
    }
  }
  
  BB->eraseFromParent();
  
  
  return true;
}

/// ReplaceInstWithValue - Replace all uses of an instruction (specified by BI)
/// with a value, then remove and delete the original instruction.
///
void llvm::ReplaceInstWithValue(BasicBlock::InstListType &BIL,
                                BasicBlock::iterator &BI, Value *V) {
  Instruction &I = *BI;
  // Replaces all of the uses of the instruction with uses of the value
  I.replaceAllUsesWith(V);

  // Make sure to propagate a name if there is one already.
  if (I.hasName() && !V->hasName())
    V->takeName(&I);

  // Delete the unnecessary instruction now...
  BI = BIL.erase(BI);
}


/// ReplaceInstWithInst - Replace the instruction specified by BI with the
/// instruction specified by I.  The original instruction is deleted and BI is
/// updated to point to the new instruction.
///
void llvm::ReplaceInstWithInst(BasicBlock::InstListType &BIL,
                               BasicBlock::iterator &BI, Instruction *I) {
  assert(I->getParent() == 0 &&
         "ReplaceInstWithInst: Instruction already inserted into basic block!");

  // Insert the new instruction into the basic block...
  BasicBlock::iterator New = BIL.insert(BI, I);

  // Replace all uses of the old instruction, and delete it.
  ReplaceInstWithValue(BIL, BI, I);

  // Move BI back to point to the newly inserted instruction
  BI = New;
}

/// ReplaceInstWithInst - Replace the instruction specified by From with the
/// instruction specified by To.
///
void llvm::ReplaceInstWithInst(Instruction *From, Instruction *To) {
  BasicBlock::iterator BI(From);
  ReplaceInstWithInst(From->getParent()->getInstList(), BI, To);
}

/// RemoveSuccessor - Change the specified terminator instruction such that its
/// successor SuccNum no longer exists.  Because this reduces the outgoing
/// degree of the current basic block, the actual terminator instruction itself
/// may have to be changed.  In the case where the last successor of the block 
/// is deleted, a return instruction is inserted in its place which can cause a
/// surprising change in program behavior if it is not expected.
///
void llvm::RemoveSuccessor(TerminatorInst *TI, unsigned SuccNum) {
  assert(SuccNum < TI->getNumSuccessors() &&
         "Trying to remove a nonexistant successor!");

  // If our old successor block contains any PHI nodes, remove the entry in the
  // PHI nodes that comes from this branch...
  //
  BasicBlock *BB = TI->getParent();
  TI->getSuccessor(SuccNum)->removePredecessor(BB);

  TerminatorInst *NewTI = 0;
  switch (TI->getOpcode()) {
  case Instruction::Br:
    // If this is a conditional branch... convert to unconditional branch.
    if (TI->getNumSuccessors() == 2) {
      cast<BranchInst>(TI)->setUnconditionalDest(TI->getSuccessor(1-SuccNum));
    } else {                    // Otherwise convert to a return instruction...
      Value *RetVal = 0;

      // Create a value to return... if the function doesn't return null...
      if (BB->getParent()->getReturnType() != Type::VoidTy)
        RetVal = Constant::getNullValue(BB->getParent()->getReturnType());

      // Create the return...
      NewTI = ReturnInst::Create(RetVal);
    }
    break;

  case Instruction::Invoke:    // Should convert to call
  case Instruction::Switch:    // Should remove entry
  default:
  case Instruction::Ret:       // Cannot happen, has no successors!
    assert(0 && "Unhandled terminator instruction type in RemoveSuccessor!");
    abort();
  }

  if (NewTI)   // If it's a different instruction, replace.
    ReplaceInstWithInst(TI, NewTI);
}

/// SplitEdge -  Split the edge connecting specified block. Pass P must 
/// not be NULL. 
BasicBlock *llvm::SplitEdge(BasicBlock *BB, BasicBlock *Succ, Pass *P) {
  TerminatorInst *LatchTerm = BB->getTerminator();
  unsigned SuccNum = 0;
  for (unsigned i = 0, e = LatchTerm->getNumSuccessors(); ; ++i) {
    assert(i != e && "Didn't find edge?");
    if (LatchTerm->getSuccessor(i) == Succ) {
      SuccNum = i;
      break;
    }
  }
  
  // If this is a critical edge, let SplitCriticalEdge do it.
  if (SplitCriticalEdge(BB->getTerminator(), SuccNum, P))
    return LatchTerm->getSuccessor(SuccNum);

  // If the edge isn't critical, then BB has a single successor or Succ has a
  // single pred.  Split the block.
  BasicBlock::iterator SplitPoint;
  if (BasicBlock *SP = Succ->getSinglePredecessor()) {
    // If the successor only has a single pred, split the top of the successor
    // block.
    assert(SP == BB && "CFG broken");
    return SplitBlock(Succ, Succ->begin(), P);
  } else {
    // Otherwise, if BB has a single successor, split it at the bottom of the
    // block.
    assert(BB->getTerminator()->getNumSuccessors() == 1 &&
           "Should have a single succ!"); 
    return SplitBlock(BB, BB->getTerminator(), P);
  }
}

/// SplitBlock - Split the specified block at the specified instruction - every
/// thing before SplitPt stays in Old and everything starting with SplitPt moves
/// to a new block.  The two blocks are joined by an unconditional branch and
/// the loop info is updated.
///
BasicBlock *llvm::SplitBlock(BasicBlock *Old, Instruction *SplitPt, Pass *P) {

  LoopInfo &LI = P->getAnalysis<LoopInfo>();
  BasicBlock::iterator SplitIt = SplitPt;
  while (isa<PHINode>(SplitIt))
    ++SplitIt;
  BasicBlock *New = Old->splitBasicBlock(SplitIt, Old->getName()+".split");

  // The new block lives in whichever loop the old one did.
  if (Loop *L = LI.getLoopFor(Old))
    L->addBasicBlockToLoop(New, LI.getBase());

  if (DominatorTree *DT = P->getAnalysisToUpdate<DominatorTree>()) 
    {
      // Old dominates New. New node domiantes all other nodes dominated by Old.
      DomTreeNode *OldNode = DT->getNode(Old);
      std::vector<DomTreeNode *> Children;
      for (DomTreeNode::iterator I = OldNode->begin(), E = OldNode->end();
           I != E; ++I) 
        Children.push_back(*I);

      DomTreeNode *NewNode =   DT->addNewBlock(New,Old);

      for (std::vector<DomTreeNode *>::iterator I = Children.begin(),
             E = Children.end(); I != E; ++I) 
        DT->changeImmediateDominator(*I, NewNode);
    }

  if (DominanceFrontier *DF = P->getAnalysisToUpdate<DominanceFrontier>())
    DF->splitBlock(Old);
    
  return New;
}


/// SplitBlockPredecessors - This method transforms BB by introducing a new
/// basic block into the function, and moving some of the predecessors of BB to
/// be predecessors of the new block.  The new predecessors are indicated by the
/// Preds array, which has NumPreds elements in it.  The new block is given a
/// suffix of 'Suffix'.
///
/// This currently updates the LLVM IR, AliasAnalysis, DominatorTree and
/// DominanceFrontier, but no other analyses.
BasicBlock *llvm::SplitBlockPredecessors(BasicBlock *BB, 
                                         BasicBlock *const *Preds,
                                         unsigned NumPreds, const char *Suffix,
                                         Pass *P) {
  // Create new basic block, insert right before the original block.
  BasicBlock *NewBB =
    BasicBlock::Create(BB->getName()+Suffix, BB->getParent(), BB);
  
  // The new block unconditionally branches to the old block.
  BranchInst *BI = BranchInst::Create(BB, NewBB);
  
  // Move the edges from Preds to point to NewBB instead of BB.
  for (unsigned i = 0; i != NumPreds; ++i)
    Preds[i]->getTerminator()->replaceUsesOfWith(BB, NewBB);
  
  // Update dominator tree and dominator frontier if available.
  DominatorTree *DT = P ? P->getAnalysisToUpdate<DominatorTree>() : 0;
  if (DT)
    DT->splitBlock(NewBB);
  if (DominanceFrontier *DF = P ? P->getAnalysisToUpdate<DominanceFrontier>():0)
    DF->splitBlock(NewBB);
  AliasAnalysis *AA = P ? P->getAnalysisToUpdate<AliasAnalysis>() : 0;
  
  
  // Insert a new PHI node into NewBB for every PHI node in BB and that new PHI
  // node becomes an incoming value for BB's phi node.  However, if the Preds
  // list is empty, we need to insert dummy entries into the PHI nodes in BB to
  // account for the newly created predecessor.
  if (NumPreds == 0) {
    // Insert dummy values as the incoming value.
    for (BasicBlock::iterator I = BB->begin(); isa<PHINode>(I); ++I)
      cast<PHINode>(I)->addIncoming(UndefValue::get(I->getType()), NewBB);
    return NewBB;
  }
  
  // Otherwise, create a new PHI node in NewBB for each PHI node in BB.
  for (BasicBlock::iterator I = BB->begin(); isa<PHINode>(I); ) {
    PHINode *PN = cast<PHINode>(I++);
    
    // Check to see if all of the values coming in are the same.  If so, we
    // don't need to create a new PHI node.
    Value *InVal = PN->getIncomingValueForBlock(Preds[0]);
    for (unsigned i = 1; i != NumPreds; ++i)
      if (InVal != PN->getIncomingValueForBlock(Preds[i])) {
        InVal = 0;
        break;
      }
    
    if (InVal) {
      // If all incoming values for the new PHI would be the same, just don't
      // make a new PHI.  Instead, just remove the incoming values from the old
      // PHI.
      for (unsigned i = 0; i != NumPreds; ++i)
        PN->removeIncomingValue(Preds[i], false);
    } else {
      // If the values coming into the block are not the same, we need a PHI.
      // Create the new PHI node, insert it into NewBB at the end of the block
      PHINode *NewPHI =
        PHINode::Create(PN->getType(), PN->getName()+".ph", BI);
      if (AA) AA->copyValue(PN, NewPHI);
      
      // Move all of the PHI values for 'Preds' to the new PHI.
      for (unsigned i = 0; i != NumPreds; ++i) {
        Value *V = PN->removeIncomingValue(Preds[i], false);
        NewPHI->addIncoming(V, Preds[i]);
      }
      InVal = NewPHI;
    }
    
    // Add an incoming value to the PHI node in the loop for the preheader
    // edge.
    PN->addIncoming(InVal, NewBB);
    
    // Check to see if we can eliminate this phi node.
    if (Value *V = PN->hasConstantValue(DT != 0)) {
      Instruction *I = dyn_cast<Instruction>(V);
      if (!I || DT == 0 || DT->dominates(I, PN)) {
        PN->replaceAllUsesWith(V);
        if (AA) AA->deleteValue(PN);
        PN->eraseFromParent();
      }
    }
  }
  
  return NewBB;
}
