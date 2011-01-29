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
#include "llvm/IntrinsicInst.h"
#include "llvm/Constant.h"
#include "llvm/Type.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ValueHandle.h"
#include <algorithm>
using namespace llvm;

/// DeleteDeadBlock - Delete the specified block, which must have no
/// predecessors.
void llvm::DeleteDeadBlock(BasicBlock *BB) {
  assert((pred_begin(BB) == pred_end(BB) ||
         // Can delete self loop.
         BB->getSinglePredecessor() == BB) && "Block is not dead!");
  TerminatorInst *BBTerm = BB->getTerminator();
  
  // Loop through all of our successors and make sure they know that one
  // of their predecessors is going away.
  for (unsigned i = 0, e = BBTerm->getNumSuccessors(); i != e; ++i)
    BBTerm->getSuccessor(i)->removePredecessor(BB);
  
  // Zap all the instructions in the block.
  while (!BB->empty()) {
    Instruction &I = BB->back();
    // If this instruction is used, replace uses with an arbitrary value.
    // Because control flow can't get here, we don't care what we replace the
    // value with.  Note that since this block is unreachable, and all values
    // contained within it must dominate their uses, that all uses will
    // eventually be removed (they are themselves dead).
    if (!I.use_empty())
      I.replaceAllUsesWith(UndefValue::get(I.getType()));
    BB->getInstList().pop_back();
  }
  
  // Zap the block!
  BB->eraseFromParent();
}

/// FoldSingleEntryPHINodes - We know that BB has one predecessor.  If there are
/// any single-entry PHI nodes in it, fold them away.  This handles the case
/// when all entries to the PHI nodes in a block are guaranteed equal, such as
/// when the block has exactly one predecessor.
void llvm::FoldSingleEntryPHINodes(BasicBlock *BB, Pass *P) {
  if (!isa<PHINode>(BB->begin())) return;
  
  AliasAnalysis *AA = 0;
  MemoryDependenceAnalysis *MemDep = 0;
  if (P) {
    AA = P->getAnalysisIfAvailable<AliasAnalysis>();
    MemDep = P->getAnalysisIfAvailable<MemoryDependenceAnalysis>();
  }
  
  while (PHINode *PN = dyn_cast<PHINode>(BB->begin())) {
    if (PN->getIncomingValue(0) != PN)
      PN->replaceAllUsesWith(PN->getIncomingValue(0));
    else
      PN->replaceAllUsesWith(UndefValue::get(PN->getType()));
    
    if (MemDep)
      MemDep->removeInstruction(PN);  // Memdep updates AA itself.
    else if (AA && isa<PointerType>(PN->getType()))
      AA->deleteValue(PN);
    
    PN->eraseFromParent();
  }
}


/// DeleteDeadPHIs - Examine each PHI in the given block and delete it if it
/// is dead. Also recursively delete any operands that become dead as
/// a result. This includes tracing the def-use list from the PHI to see if
/// it is ultimately unused or if it reaches an unused cycle.
bool llvm::DeleteDeadPHIs(BasicBlock *BB) {
  // Recursively deleting a PHI may cause multiple PHIs to be deleted
  // or RAUW'd undef, so use an array of WeakVH for the PHIs to delete.
  SmallVector<WeakVH, 8> PHIs;
  for (BasicBlock::iterator I = BB->begin();
       PHINode *PN = dyn_cast<PHINode>(I); ++I)
    PHIs.push_back(PN);

  bool Changed = false;
  for (unsigned i = 0, e = PHIs.size(); i != e; ++i)
    if (PHINode *PN = dyn_cast_or_null<PHINode>(PHIs[i].operator Value*()))
      Changed |= RecursivelyDeleteDeadPHINode(PN);

  return Changed;
}

/// MergeBlockIntoPredecessor - Attempts to merge a block into its predecessor,
/// if possible.  The return value indicates success or failure.
bool llvm::MergeBlockIntoPredecessor(BasicBlock *BB, Pass *P) {
  // Don't merge away blocks who have their address taken.
  if (BB->hasAddressTaken()) return false;
  
  // Can't merge if there are multiple predecessors, or no predecessors.
  BasicBlock *PredBB = BB->getUniquePredecessor();
  if (!PredBB) return false;

  // Don't break self-loops.
  if (PredBB == BB) return false;
  // Don't break invokes.
  if (isa<InvokeInst>(PredBB->getTerminator())) return false;
  
  succ_iterator SI(succ_begin(PredBB)), SE(succ_end(PredBB));
  BasicBlock *OnlySucc = BB;
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
  if (isa<PHINode>(BB->front()))
    FoldSingleEntryPHINodes(BB, P);
  
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
    if (DominatorTree *DT = P->getAnalysisIfAvailable<DominatorTree>()) {
      if (DomTreeNode *DTN = DT->getNode(BB)) {
        DomTreeNode *PredDTN = DT->getNode(PredBB);
        SmallVector<DomTreeNode*, 8> Children(DTN->begin(), DTN->end());
        for (SmallVector<DomTreeNode*, 8>::iterator DI = Children.begin(),
             DE = Children.end(); DI != DE; ++DI)
          DT->changeImmediateDominator(*DI, PredDTN);

        DT->eraseNode(BB);
      }
      
      if (LoopInfo *LI = P->getAnalysisIfAvailable<LoopInfo>())
        LI->removeBlock(BB);
      
      if (MemoryDependenceAnalysis *MD =
            P->getAnalysisIfAvailable<MemoryDependenceAnalysis>())
        MD->invalidateCachedPredecessors();
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

/// GetSuccessorNumber - Search for the specified successor of basic block BB
/// and return its position in the terminator instruction's list of
/// successors.  It is an error to call this with a block that is not a
/// successor.
unsigned llvm::GetSuccessorNumber(BasicBlock *BB, BasicBlock *Succ) {
  TerminatorInst *Term = BB->getTerminator();
#ifndef NDEBUG
  unsigned e = Term->getNumSuccessors();
#endif
  for (unsigned i = 0; ; ++i) {
    assert(i != e && "Didn't find edge?");
    if (Term->getSuccessor(i) == Succ)
      return i;
  }
  return 0;
}

/// SplitEdge -  Split the edge connecting specified block. Pass P must 
/// not be NULL. 
BasicBlock *llvm::SplitEdge(BasicBlock *BB, BasicBlock *Succ, Pass *P) {
  unsigned SuccNum = GetSuccessorNumber(BB, Succ);
  
  // If this is a critical edge, let SplitCriticalEdge do it.
  TerminatorInst *LatchTerm = BB->getTerminator();
  if (SplitCriticalEdge(LatchTerm, SuccNum, P))
    return LatchTerm->getSuccessor(SuccNum);

  // If the edge isn't critical, then BB has a single successor or Succ has a
  // single pred.  Split the block.
  BasicBlock::iterator SplitPoint;
  if (BasicBlock *SP = Succ->getSinglePredecessor()) {
    // If the successor only has a single pred, split the top of the successor
    // block.
    assert(SP == BB && "CFG broken");
    SP = NULL;
    return SplitBlock(Succ, Succ->begin(), P);
  }
  
  // Otherwise, if BB has a single successor, split it at the bottom of the
  // block.
  assert(BB->getTerminator()->getNumSuccessors() == 1 &&
         "Should have a single succ!"); 
  return SplitBlock(BB, BB->getTerminator(), P);
}

/// SplitBlock - Split the specified block at the specified instruction - every
/// thing before SplitPt stays in Old and everything starting with SplitPt moves
/// to a new block.  The two blocks are joined by an unconditional branch and
/// the loop info is updated.
///
BasicBlock *llvm::SplitBlock(BasicBlock *Old, Instruction *SplitPt, Pass *P) {
  BasicBlock::iterator SplitIt = SplitPt;
  while (isa<PHINode>(SplitIt))
    ++SplitIt;
  BasicBlock *New = Old->splitBasicBlock(SplitIt, Old->getName()+".split");

  // The new block lives in whichever loop the old one did. This preserves
  // LCSSA as well, because we force the split point to be after any PHI nodes.
  if (LoopInfo *LI = P->getAnalysisIfAvailable<LoopInfo>())
    if (Loop *L = LI->getLoopFor(Old))
      L->addBasicBlockToLoop(New, LI->getBase());

  if (DominatorTree *DT = P->getAnalysisIfAvailable<DominatorTree>()) {
    // Old dominates New. New node dominates all other nodes dominated by Old.
    DomTreeNode *OldNode = DT->getNode(Old);
    std::vector<DomTreeNode *> Children;
    for (DomTreeNode::iterator I = OldNode->begin(), E = OldNode->end();
         I != E; ++I) 
      Children.push_back(*I);

      DomTreeNode *NewNode = DT->addNewBlock(New,Old);
      for (std::vector<DomTreeNode *>::iterator I = Children.begin(),
             E = Children.end(); I != E; ++I) 
        DT->changeImmediateDominator(*I, NewNode);
  }

  return New;
}


/// SplitBlockPredecessors - This method transforms BB by introducing a new
/// basic block into the function, and moving some of the predecessors of BB to
/// be predecessors of the new block.  The new predecessors are indicated by the
/// Preds array, which has NumPreds elements in it.  The new block is given a
/// suffix of 'Suffix'.
///
/// This currently updates the LLVM IR, AliasAnalysis, DominatorTree,
/// LoopInfo, and LCCSA but no other analyses. In particular, it does not
/// preserve LoopSimplify (because it's complicated to handle the case where one
/// of the edges being split is an exit of a loop with other exits).
///
BasicBlock *llvm::SplitBlockPredecessors(BasicBlock *BB, 
                                         BasicBlock *const *Preds,
                                         unsigned NumPreds, const char *Suffix,
                                         Pass *P) {
  // Create new basic block, insert right before the original block.
  BasicBlock *NewBB = BasicBlock::Create(BB->getContext(), BB->getName()+Suffix,
                                         BB->getParent(), BB);
  
  // The new block unconditionally branches to the old block.
  BranchInst *BI = BranchInst::Create(BB, NewBB);
  
  LoopInfo *LI = P ? P->getAnalysisIfAvailable<LoopInfo>() : 0;
  Loop *L = LI ? LI->getLoopFor(BB) : 0;
  bool PreserveLCSSA = P->mustPreserveAnalysisID(LCSSAID);

  // Move the edges from Preds to point to NewBB instead of BB.
  // While here, if we need to preserve loop analyses, collect
  // some information about how this split will affect loops.
  bool HasLoopExit = false;
  bool IsLoopEntry = !!L;
  bool SplitMakesNewLoopHeader = false;
  for (unsigned i = 0; i != NumPreds; ++i) {
    // This is slightly more strict than necessary; the minimum requirement
    // is that there be no more than one indirectbr branching to BB. And
    // all BlockAddress uses would need to be updated.
    assert(!isa<IndirectBrInst>(Preds[i]->getTerminator()) &&
           "Cannot split an edge from an IndirectBrInst");

    Preds[i]->getTerminator()->replaceUsesOfWith(BB, NewBB);

    if (LI) {
      // If we need to preserve LCSSA, determine if any of
      // the preds is a loop exit.
      if (PreserveLCSSA)
        if (Loop *PL = LI->getLoopFor(Preds[i]))
          if (!PL->contains(BB))
            HasLoopExit = true;
      // If we need to preserve LoopInfo, note whether any of the
      // preds crosses an interesting loop boundary.
      if (L) {
        if (L->contains(Preds[i]))
          IsLoopEntry = false;
        else
          SplitMakesNewLoopHeader = true;
      }
    }
  }

  // Update dominator tree if available.
  DominatorTree *DT = P ? P->getAnalysisIfAvailable<DominatorTree>() : 0;
  if (DT)
    DT->splitBlock(NewBB);

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

  AliasAnalysis *AA = P ? P->getAnalysisIfAvailable<AliasAnalysis>() : 0;

  if (L) {
    if (IsLoopEntry) {
      // Add the new block to the nearest enclosing loop (and not an
      // adjacent loop). To find this, examine each of the predecessors and
      // determine which loops enclose them, and select the most-nested loop
      // which contains the loop containing the block being split.
      Loop *InnermostPredLoop = 0;
      for (unsigned i = 0; i != NumPreds; ++i)
        if (Loop *PredLoop = LI->getLoopFor(Preds[i])) {
          // Seek a loop which actually contains the block being split (to
          // avoid adjacent loops).
          while (PredLoop && !PredLoop->contains(BB))
            PredLoop = PredLoop->getParentLoop();
          // Select the most-nested of these loops which contains the block.
          if (PredLoop &&
              PredLoop->contains(BB) &&
              (!InnermostPredLoop ||
               InnermostPredLoop->getLoopDepth() < PredLoop->getLoopDepth()))
            InnermostPredLoop = PredLoop;
        }
      if (InnermostPredLoop)
        InnermostPredLoop->addBasicBlockToLoop(NewBB, LI->getBase());
    } else {
      L->addBasicBlockToLoop(NewBB, LI->getBase());
      if (SplitMakesNewLoopHeader)
        L->moveToHeader(NewBB);
    }
  }
  
  // Otherwise, create a new PHI node in NewBB for each PHI node in BB.
  for (BasicBlock::iterator I = BB->begin(); isa<PHINode>(I); ) {
    PHINode *PN = cast<PHINode>(I++);
    
    // Check to see if all of the values coming in are the same.  If so, we
    // don't need to create a new PHI node, unless it's needed for LCSSA.
    Value *InVal = 0;
    if (!HasLoopExit) {
      InVal = PN->getIncomingValueForBlock(Preds[0]);
      for (unsigned i = 1; i != NumPreds; ++i)
        if (InVal != PN->getIncomingValueForBlock(Preds[i])) {
          InVal = 0;
          break;
        }
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
  }
  
  return NewBB;
}

/// FindFunctionBackedges - Analyze the specified function to find all of the
/// loop backedges in the function and return them.  This is a relatively cheap
/// (compared to computing dominators and loop info) analysis.
///
/// The output is added to Result, as pairs of <from,to> edge info.
void llvm::FindFunctionBackedges(const Function &F,
     SmallVectorImpl<std::pair<const BasicBlock*,const BasicBlock*> > &Result) {
  const BasicBlock *BB = &F.getEntryBlock();
  if (succ_begin(BB) == succ_end(BB))
    return;
  
  SmallPtrSet<const BasicBlock*, 8> Visited;
  SmallVector<std::pair<const BasicBlock*, succ_const_iterator>, 8> VisitStack;
  SmallPtrSet<const BasicBlock*, 8> InStack;
  
  Visited.insert(BB);
  VisitStack.push_back(std::make_pair(BB, succ_begin(BB)));
  InStack.insert(BB);
  do {
    std::pair<const BasicBlock*, succ_const_iterator> &Top = VisitStack.back();
    const BasicBlock *ParentBB = Top.first;
    succ_const_iterator &I = Top.second;
    
    bool FoundNew = false;
    while (I != succ_end(ParentBB)) {
      BB = *I++;
      if (Visited.insert(BB)) {
        FoundNew = true;
        break;
      }
      // Successor is in VisitStack, it's a back edge.
      if (InStack.count(BB))
        Result.push_back(std::make_pair(ParentBB, BB));
    }
    
    if (FoundNew) {
      // Go down one level if there is a unvisited successor.
      InStack.insert(BB);
      VisitStack.push_back(std::make_pair(BB, succ_begin(BB)));
    } else {
      // Go up one level.
      InStack.erase(VisitStack.pop_back_val().first);
    }
  } while (!VisitStack.empty()); 
}

/// FoldReturnIntoUncondBranch - This method duplicates the specified return
/// instruction into a predecessor which ends in an unconditional branch. If
/// the return instruction returns a value defined by a PHI, propagate the
/// right value into the return. It returns the new return instruction in the
/// predecessor.
ReturnInst *llvm::FoldReturnIntoUncondBranch(ReturnInst *RI, BasicBlock *BB,
                                             BasicBlock *Pred) {
  Instruction *UncondBranch = Pred->getTerminator();
  // Clone the return and add it to the end of the predecessor.
  Instruction *NewRet = RI->clone();
  Pred->getInstList().push_back(NewRet);
      
  // If the return instruction returns a value, and if the value was a
  // PHI node in "BB", propagate the right value into the return.
  for (User::op_iterator i = NewRet->op_begin(), e = NewRet->op_end();
       i != e; ++i)
    if (PHINode *PN = dyn_cast<PHINode>(*i))
      if (PN->getParent() == BB)
        *i = PN->getIncomingValueForBlock(Pred);
      
  // Update any PHI nodes in the returning block to realize that we no
  // longer branch to them.
  BB->removePredecessor(Pred);
  UncondBranch->eraseFromParent();
  return cast<ReturnInst>(NewRet);
}
