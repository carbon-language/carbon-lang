//===-- LCSSA.cpp - Convert loops into loop-closed SSA form ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Owen Anderson and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass transforms loops by placing phi nodes at the end of the loops for
// all values that are live across the loop boundary.  For example, it turns
// the left into the right code:
// 
// for (...)                for (...)
//   if (c)                   if(c)
//     X1 = ...                 X1 = ...
//   else                     else
//     X2 = ...                 X2 = ...
//   X3 = phi(X1, X2)         X3 = phi(X1, X2)
// ... = X3 + 4              X4 = phi(X3)
//                           ... = X4 + 4
//
// This is still valid LLVM; the extra phi nodes are purely redundant, and will
// be trivially eliminated by InstCombine.  The major benefit of this 
// transformation is that it makes many other loop optimizations, such as 
// LoopUnswitching, simpler.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/CFG.h"
#include <algorithm>
#include <map>

using namespace llvm;

namespace {
  static Statistic<> NumLCSSA("lcssa",
                              "Number of live out of a loop variables");
  
  struct LCSSA : public FunctionPass {
    // Cached analysis information for the current function.
    LoopInfo *LI;
    DominatorTree *DT;
    std::vector<BasicBlock*> LoopBlocks;
    
    virtual bool runOnFunction(Function &F);
    bool visitSubloop(Loop* L);
    void ProcessInstruction(Instruction* Instr,
                            const std::vector<BasicBlock*>& exitBlocks);
    
    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG.  It maintains both of these,
    /// as well as the CFG.  It also requires dominator information.
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequiredID(LoopSimplifyID);
      AU.addPreservedID(LoopSimplifyID);
      AU.addRequired<LoopInfo>();
      AU.addRequired<DominatorTree>();
    }
  private:
    SetVector<Instruction*> getLoopValuesUsedOutsideLoop(Loop *L);

    PHINode *GetValueForBlock(DominatorTree::Node *BB, Instruction *OrigInst,
                              std::map<DominatorTree::Node*, PHINode*> &Phis);

    /// inLoop - returns true if the given block is within the current loop
    const bool inLoop(BasicBlock* B) {
      return std::binary_search(LoopBlocks.begin(), LoopBlocks.end(), B);
    }
  };
  
  RegisterPass<LCSSA> X("lcssa", "Loop-Closed SSA Form Pass");
}

FunctionPass *llvm::createLCSSAPass() { return new LCSSA(); }
const PassInfo *llvm::LCSSAID = X.getPassInfo();

/// runOnFunction - Process all loops in the function, inner-most out.
bool LCSSA::runOnFunction(Function &F) {
  bool changed = false;
  
  LI = &getAnalysis<LoopInfo>();
  DT = &getAnalysis<DominatorTree>();
    
  for (LoopInfo::iterator I = LI->begin(), E = LI->end(); I != E; ++I)
    changed |= visitSubloop(*I);
      
  return changed;
}

/// visitSubloop - Recursively process all subloops, and then process the given
/// loop if it has live-out values.
bool LCSSA::visitSubloop(Loop* L) {
  for (Loop::iterator I = L->begin(), E = L->end(); I != E; ++I)
    visitSubloop(*I);
    
  // Speed up queries by creating a sorted list of blocks
  LoopBlocks.clear();
  LoopBlocks.insert(LoopBlocks.end(), L->block_begin(), L->block_end());
  std::sort(LoopBlocks.begin(), LoopBlocks.end());
  
  SetVector<Instruction*> AffectedValues = getLoopValuesUsedOutsideLoop(L);
  
  // If no values are affected, we can save a lot of work, since we know that
  // nothing will be changed.
  if (AffectedValues.empty())
    return false;
  
  std::vector<BasicBlock*> exitBlocks;
  L->getExitBlocks(exitBlocks);
  
  
  // Iterate over all affected values for this loop and insert Phi nodes
  // for them in the appropriate exit blocks
  
  for (SetVector<Instruction*>::iterator I = AffectedValues.begin(),
       E = AffectedValues.end(); I != E; ++I)
    ProcessInstruction(*I, exitBlocks);
  
  assert(L->isLCSSAForm());
  
  return true;
}

/// processInstruction - Given a live-out instruction, insert LCSSA Phi nodes,
/// eliminate all out-of-loop uses.
void LCSSA::ProcessInstruction(Instruction *Instr,
                               const std::vector<BasicBlock*>& exitBlocks) {
  ++NumLCSSA; // We are applying the transformation

  // Keep track of the blocks that have the value available already.
  std::map<DominatorTree::Node*, PHINode*> Phis;

  DominatorTree::Node *InstrNode = DT->getNode(Instr->getParent());

  // Insert the LCSSA phi's into the exit blocks (dominated by the value), and
  // add them to the Phi's map.
  for (std::vector<BasicBlock*>::const_iterator BBI = exitBlocks.begin(),
      BBE = exitBlocks.end(); BBI != BBE; ++BBI) {
    BasicBlock *BB = *BBI;
    DominatorTree::Node *ExitBBNode = DT->getNode(BB);
    PHINode *&Phi = Phis[ExitBBNode];
    if (!Phi && InstrNode->dominates(ExitBBNode)) {
      Phi = new PHINode(Instr->getType(), Instr->getName()+".lcssa",
                        BB->begin());
      Phi->reserveOperandSpace(std::distance(pred_begin(BB), pred_end(BB)));

      // Add inputs from inside the loop for this PHI.
      for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI)
        Phi->addIncoming(Instr, *PI);
      
      // Remember that this phi makes the value alive in this block.
      Phis[ExitBBNode] = Phi;
    }
  }
  
  
  // Record all uses of Instr outside the loop.  We need to rewrite these.  The
  // LCSSA phis won't be included because they use the value in the loop.
  for (Value::use_iterator UI = Instr->use_begin(), E = Instr->use_end();
       UI != E;) {
    BasicBlock *UserBB = cast<Instruction>(*UI)->getParent();
    if (PHINode *P = dyn_cast<PHINode>(*UI)) {
      unsigned OperandNo = UI.getOperandNo();
      UserBB = P->getIncomingBlock(OperandNo/2);
    }
    
    // If the user is in the loop, don't rewrite it!
    if (UserBB == Instr->getParent() || inLoop(UserBB)) {
      ++UI;
      continue;
    }
    
    // Otherwise, patch up uses of the value with the appropriate LCSSA Phi,
    // inserting PHI nodes into join points where needed.
    Value *Val = GetValueForBlock(DT->getNode(UserBB), Instr, Phis);
    
    // Preincrement the iterator to avoid invalidating it when we change the
    // value.
    Use &U = UI.getUse();
    ++UI;
    U.set(Val);
  }
}

/// getLoopValuesUsedOutsideLoop - Return any values defined in the loop that
/// are used by instructions outside of it.
SetVector<Instruction*> LCSSA::getLoopValuesUsedOutsideLoop(Loop *L) {
  
  // FIXME: For large loops, we may be able to avoid a lot of use-scanning
  // by using dominance information.  In particular, if a block does not
  // dominate any of the loop exits, then none of the values defined in the
  // block could be used outside the loop.
  
  SetVector<Instruction*> AffectedValues;  
  for (Loop::block_iterator BB = L->block_begin(), E = L->block_end();
       BB != E; ++BB) {
    for (BasicBlock::iterator I = (*BB)->begin(), E = (*BB)->end(); I != E; ++I)
      for (Value::use_iterator UI = I->use_begin(), E = I->use_end(); UI != E;
           ++UI) {
        BasicBlock *UserBB = cast<Instruction>(*UI)->getParent();
        if (PHINode* p = dyn_cast<PHINode>(*UI)) {
          unsigned OperandNo = UI.getOperandNo();
          UserBB = p->getIncomingBlock(OperandNo/2);
        }
        
        if (*BB != UserBB && !inLoop(UserBB)) {
          AffectedValues.insert(I);
          break;
        }
      }
  }
  return AffectedValues;
}

/// GetValueForBlock - Get the value to use within the specified basic block.
/// available values are in Phis.
PHINode *LCSSA::GetValueForBlock(DominatorTree::Node *BB, Instruction *OrigInst,
                               std::map<DominatorTree::Node*, PHINode*> &Phis) {
  // If we have already computed this value, return the previously computed val.
  PHINode *&V = Phis[BB];
  if (V) return V;

  DominatorTree::Node *IDom = BB->getIDom();

  // Otherwise, there are two cases: we either have to insert a PHI node or we
  // don't.  We need to insert a PHI node if this block is not dominated by one
  // of the exit nodes from the loop (the loop could have multiple exits, and
  // though the value defined *inside* the loop dominated all its uses, each
  // exit by itself may not dominate all the uses).
  //
  // The simplest way to check for this condition is by checking to see if the
  // idom is in the loop.  If so, we *know* that none of the exit blocks
  // dominate this block.  Note that we *know* that the block defining the
  // original instruction is in the idom chain, because if it weren't, then the
  // original value didn't dominate this use.
  if (!inLoop(IDom->getBlock())) {
    // Idom is not in the loop, we must still be "below" the exit block and must
    // be fully dominated by the value live in the idom.
    return V = GetValueForBlock(IDom, OrigInst, Phis);
  }
  
  BasicBlock *BBN = BB->getBlock();
  
  // Otherwise, the idom is the loop, so we need to insert a PHI node.  Do so
  // now, then get values to fill in the incoming values for the PHI.
  V = new PHINode(OrigInst->getType(), OrigInst->getName()+".lcssa",
                  BBN->begin());
  V->reserveOperandSpace(std::distance(pred_begin(BBN), pred_end(BBN)));

  // Fill in the incoming values for the block.
  for (pred_iterator PI = pred_begin(BBN), E = pred_end(BBN); PI != E; ++PI)
    V->addIncoming(GetValueForBlock(DT->getNode(*PI), OrigInst, Phis), *PI);
  return V;
}

