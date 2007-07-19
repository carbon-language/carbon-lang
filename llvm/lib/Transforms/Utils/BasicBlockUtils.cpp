//===-- BasicBlockUtils.cpp - BasicBlock Utilities -------------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/Dominators.h"
#include <algorithm>
using namespace llvm;

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
      NewTI = new ReturnInst(RetVal);
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
    L->addBasicBlockToLoop(New, LI);

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
