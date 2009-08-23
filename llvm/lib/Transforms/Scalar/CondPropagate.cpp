//===-- CondPropagate.cpp - Propagate Conditional Expressions -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass propagates information about conditional expressions through the
// program, allowing it to eliminate conditional branches in some cases.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "condprop"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/Type.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;

STATISTIC(NumBrThread, "Number of CFG edges threaded through branches");
STATISTIC(NumSwThread, "Number of CFG edges threaded through switches");

namespace {
  struct VISIBILITY_HIDDEN CondProp : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    CondProp() : FunctionPass(&ID) {}

    virtual bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(BreakCriticalEdgesID);
      //AU.addRequired<DominanceFrontier>();
    }

  private:
    bool MadeChange;
    SmallVector<BasicBlock *, 4> DeadBlocks;
    void SimplifyBlock(BasicBlock *BB);
    void SimplifyPredecessors(BranchInst *BI);
    void SimplifyPredecessors(SwitchInst *SI);
    void RevectorBlockTo(BasicBlock *FromBB, BasicBlock *ToBB);
    bool RevectorBlockTo(BasicBlock *FromBB, Value *Cond, BranchInst *BI);
  };
}
  
char CondProp::ID = 0;
static RegisterPass<CondProp> X("condprop", "Conditional Propagation");

FunctionPass *llvm::createCondPropagationPass() {
  return new CondProp();
}

bool CondProp::runOnFunction(Function &F) {
  bool EverMadeChange = false;
  DeadBlocks.clear();

  // While we are simplifying blocks, keep iterating.
  do {
    MadeChange = false;
    for (Function::iterator BB = F.begin(), E = F.end(); BB != E;)
      SimplifyBlock(BB++);
    EverMadeChange = EverMadeChange || MadeChange;
  } while (MadeChange);

  if (EverMadeChange) {
    while (!DeadBlocks.empty()) {
      BasicBlock *BB = DeadBlocks.back(); DeadBlocks.pop_back();
      DeleteDeadBlock(BB);
    }
  }
  return EverMadeChange;
}

void CondProp::SimplifyBlock(BasicBlock *BB) {
  if (BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator())) {
    // If this is a conditional branch based on a phi node that is defined in
    // this block, see if we can simplify predecessors of this block.
    if (BI->isConditional() && isa<PHINode>(BI->getCondition()) &&
        cast<PHINode>(BI->getCondition())->getParent() == BB)
      SimplifyPredecessors(BI);

  } else if (SwitchInst *SI = dyn_cast<SwitchInst>(BB->getTerminator())) {
    if (isa<PHINode>(SI->getCondition()) &&
        cast<PHINode>(SI->getCondition())->getParent() == BB)
      SimplifyPredecessors(SI);
  }

  // If possible, simplify the terminator of this block.
  if (ConstantFoldTerminator(BB))
    MadeChange = true;

  // If this block ends with an unconditional branch and the only successor has
  // only this block as a predecessor, merge the two blocks together.
  if (BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator()))
    if (BI->isUnconditional() && BI->getSuccessor(0)->getSinglePredecessor() &&
        BB != BI->getSuccessor(0)) {
      BasicBlock *Succ = BI->getSuccessor(0);
      
      // If Succ has any PHI nodes, they are all single-entry PHI's.  Eliminate
      // them.
      FoldSingleEntryPHINodes(Succ);
      
      // Remove BI.
      BI->eraseFromParent();

      // Move over all of the instructions.
      BB->getInstList().splice(BB->end(), Succ->getInstList());

      // Any phi nodes that had entries for Succ now have entries from BB.
      Succ->replaceAllUsesWith(BB);

      // Succ is now dead, but we cannot delete it without potentially
      // invalidating iterators elsewhere.  Just insert an unreachable
      // instruction in it and delete this block later on.
      new UnreachableInst(BB->getContext(), Succ);
      DeadBlocks.push_back(Succ);
      MadeChange = true;
    }
}

// SimplifyPredecessors(branches) - We know that BI is a conditional branch
// based on a PHI node defined in this block.  If the phi node contains constant
// operands, then the blocks corresponding to those operands can be modified to
// jump directly to the destination instead of going through this block.
void CondProp::SimplifyPredecessors(BranchInst *BI) {
  // TODO: We currently only handle the most trival case, where the PHI node has
  // one use (the branch), and is the only instruction besides the branch and dbg
  // intrinsics in the block.
  PHINode *PN = cast<PHINode>(BI->getCondition());

  if (PN->getNumIncomingValues() == 1) {
    // Eliminate single-entry PHI nodes.
    FoldSingleEntryPHINodes(PN->getParent());
    return;
  }
  
  
  if (!PN->hasOneUse()) return;

  BasicBlock *BB = BI->getParent();
  if (&*BB->begin() != PN)
    return;
  BasicBlock::iterator BBI = BB->begin();
  BasicBlock::iterator BBE = BB->end();
  while (BBI != BBE && isa<DbgInfoIntrinsic>(++BBI)) /* empty */;
  if (&*BBI != BI)
    return;

  // Ok, we have this really simple case, walk the PHI operands, looking for
  // constants.  Walk from the end to remove operands from the end when
  // possible, and to avoid invalidating "i".
  for (unsigned i = PN->getNumIncomingValues(); i != 0; --i) {
    Value *InVal = PN->getIncomingValue(i-1);
    if (!RevectorBlockTo(PN->getIncomingBlock(i-1), InVal, BI))
      continue;

    ++NumBrThread;

    // If there were two predecessors before this simplification, or if the
    // PHI node contained all the same value except for the one we just
    // substituted, the PHI node may be deleted.  Don't iterate through it the
    // last time.
    if (BI->getCondition() != PN) return;
  }
}

// SimplifyPredecessors(switch) - We know that SI is switch based on a PHI node
// defined in this block.  If the phi node contains constant operands, then the
// blocks corresponding to those operands can be modified to jump directly to
// the destination instead of going through this block.
void CondProp::SimplifyPredecessors(SwitchInst *SI) {
  // TODO: We currently only handle the most trival case, where the PHI node has
  // one use (the branch), and is the only instruction besides the branch and 
  // dbg intrinsics in the block.
  PHINode *PN = cast<PHINode>(SI->getCondition());
  if (!PN->hasOneUse()) return;

  BasicBlock *BB = SI->getParent();
  if (&*BB->begin() != PN)
    return;
  BasicBlock::iterator BBI = BB->begin();
  BasicBlock::iterator BBE = BB->end();
  while (BBI != BBE && isa<DbgInfoIntrinsic>(++BBI)) /* empty */;
  if (&*BBI != SI)
    return;

  bool RemovedPreds = false;

  // Ok, we have this really simple case, walk the PHI operands, looking for
  // constants.  Walk from the end to remove operands from the end when
  // possible, and to avoid invalidating "i".
  for (unsigned i = PN->getNumIncomingValues(); i != 0; --i)
    if (ConstantInt *CI = dyn_cast<ConstantInt>(PN->getIncomingValue(i-1))) {
      // If we have a constant, forward the edge from its current to its
      // ultimate destination.
      unsigned DestCase = SI->findCaseValue(CI);
      RevectorBlockTo(PN->getIncomingBlock(i-1),
                      SI->getSuccessor(DestCase));
      ++NumSwThread;
      RemovedPreds = true;

      // If there were two predecessors before this simplification, or if the
      // PHI node contained all the same value except for the one we just
      // substituted, the PHI node may be deleted.  Don't iterate through it the
      // last time.
      if (SI->getCondition() != PN) return;
    }
}


// RevectorBlockTo - Revector the unconditional branch at the end of FromBB to
// the ToBB block, which is one of the successors of its current successor.
void CondProp::RevectorBlockTo(BasicBlock *FromBB, BasicBlock *ToBB) {
  BranchInst *FromBr = cast<BranchInst>(FromBB->getTerminator());
  assert(FromBr->isUnconditional() && "FromBB should end with uncond br!");

  // Get the old block we are threading through.
  BasicBlock *OldSucc = FromBr->getSuccessor(0);

  // OldSucc had multiple successors. If ToBB has multiple predecessors, then 
  // the edge between them would be critical, which we already took care of.
  // If ToBB has single operand PHI node then take care of it here.
  FoldSingleEntryPHINodes(ToBB);

  // Update PHI nodes in OldSucc to know that FromBB no longer branches to it.
  OldSucc->removePredecessor(FromBB);

  // Change FromBr to branch to the new destination.
  FromBr->setSuccessor(0, ToBB);

  MadeChange = true;
}

bool CondProp::RevectorBlockTo(BasicBlock *FromBB, Value *Cond, BranchInst *BI){
  BranchInst *FromBr = cast<BranchInst>(FromBB->getTerminator());
  if (!FromBr->isUnconditional())
    return false;

  // Get the old block we are threading through.
  BasicBlock *OldSucc = FromBr->getSuccessor(0);

  // If the condition is a constant, simply revector the unconditional branch at
  // the end of FromBB to one of the successors of its current successor.
  if (ConstantInt *CB = dyn_cast<ConstantInt>(Cond)) {
    BasicBlock *ToBB = BI->getSuccessor(CB->isZero());

    // OldSucc had multiple successors. If ToBB has multiple predecessors, then 
    // the edge between them would be critical, which we already took care of.
    // If ToBB has single operand PHI node then take care of it here.
    FoldSingleEntryPHINodes(ToBB);

    // Update PHI nodes in OldSucc to know that FromBB no longer branches to it.
    OldSucc->removePredecessor(FromBB);

    // Change FromBr to branch to the new destination.
    FromBr->setSuccessor(0, ToBB);
  } else {
    BasicBlock *Succ0 = BI->getSuccessor(0);
    // Do not perform transform if the new destination has PHI nodes. The
    // transform will add new preds to the PHI's.
    if (isa<PHINode>(Succ0->begin()))
      return false;

    BasicBlock *Succ1 = BI->getSuccessor(1);
    if (isa<PHINode>(Succ1->begin()))
      return false;

    // Insert the new conditional branch.
    BranchInst::Create(Succ0, Succ1, Cond, FromBr);

    FoldSingleEntryPHINodes(Succ0);
    FoldSingleEntryPHINodes(Succ1);

    // Update PHI nodes in OldSucc to know that FromBB no longer branches to it.
    OldSucc->removePredecessor(FromBB);

    // Delete the old branch.
    FromBr->eraseFromParent();
  }

  MadeChange = true;
  return true;
}
