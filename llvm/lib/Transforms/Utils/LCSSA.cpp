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
#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/CFG.h"
#include <algorithm>
#include <set>
#include <vector>

using namespace llvm;

namespace {
  class LCSSA : public FunctionPass {
  public:
    LoopInfo *LI;  // Loop information
    DominatorTree *DT;       // Dominator Tree for the current Loop...
    DominanceFrontier *DF;   // Current Dominance Frontier
    
    virtual bool runOnFunction(Function &F);
    bool LCSSA::visitSubloop(Loop* L);
    
    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG...
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequiredID(LoopSimplifyID);
      AU.addPreservedID(LoopSimplifyID);
      AU.addRequired<LoopInfo>();
      AU.addPreserved<LoopInfo>();
      AU.addRequired<DominatorTree>(); // Not sure if this one will actually
                                       // be needed.
      AU.addRequired<DominanceFrontier>();
    }
  private:
    std::set<Instruction*> getLoopValuesUsedOutsideLoop(Loop *L,
                                           std::vector<BasicBlock*> LoopBlocks);
  };
  
  RegisterOpt<LCSSA> X("lcssa", "Loop-Closed SSA Form Pass");
}

FunctionPass *llvm::createLCSSAPass() { return new LCSSA(); }

bool LCSSA::runOnFunction(Function &F) {
  bool changed = false;
  LI = &getAnalysis<LoopInfo>();
  DF = &getAnalysis<DominanceFrontier>();
  DT = &getAnalysis<DominatorTree>();
    
  for (LoopInfo::iterator I = LI->begin(), E = LI->end(); I != E; ++I) {
    changed |= visitSubloop(*I);
  }
      
  return changed;
}

bool LCSSA::visitSubloop(Loop* L) {
  for (Loop::iterator I = L->begin(), E = L->end(); I != E; ++I)
    visitSubloop(*I);
  
  // Speed up queries by creating a sorted list of blocks
  std::vector<BasicBlock*> LoopBlocks(L->block_begin(), L->block_end());
  std::sort(LoopBlocks.begin(), LoopBlocks.end());
  
  std::set<Instruction*> AffectedValues = getLoopValuesUsedOutsideLoop(L,
                                           LoopBlocks);
  
  std::vector<BasicBlock*> exitBlocks;
  L->getExitBlocks(exitBlocks);
  
  for (std::set<Instruction*>::iterator I = AffectedValues.begin(),
       E = AffectedValues.end(); I != E; ++I) {
    for (std::vector<BasicBlock*>::iterator BBI = exitBlocks.begin(),
         BBE = exitBlocks.end(); BBI != BBE; ++BBI) {
      PHINode *phi = new PHINode((*I)->getType(), "lcssa");
      (*BBI)->getInstList().insert((*BBI)->front(), phi);
    
      for (pred_iterator PI = pred_begin(*BBI), PE = pred_end(*BBI); PI != PE;
           ++PI)
        phi->addIncoming(*I, *PI);
    }
  
    for (Value::use_iterator UI = (*I)->use_begin(), UE = (*I)->use_end();
         UI != UE; ++UI) {
      BasicBlock *UserBB = cast<Instruction>(*UI)->getParent();
      if (!std::binary_search(LoopBlocks.begin(), LoopBlocks.end(), UserBB))
        ; // FIXME: This should update the SSA form.
    }
  }
  
  return true; // FIXME: Should be more intelligent in our return value.
}

/// getLoopValuesUsedOutsideLoop - Return any values defined in the loop that
/// are used by instructions outside of it.
std::set<Instruction*> LCSSA::getLoopValuesUsedOutsideLoop(Loop *L, 
                                         std::vector<BasicBlock*> LoopBlocks) {

  std::set<Instruction*> AffectedValues;  
  for (Loop::block_iterator BB = L->block_begin(), E = L->block_end();
       BB != E; ++BB) {
    for (BasicBlock::iterator I = (*BB)->begin(), E = (*BB)->end(); I != E; ++I)
      for (Value::use_iterator UI = I->use_begin(), E = I->use_end(); UI != E;
           ++UI) {
        BasicBlock *UserBB = cast<Instruction>(*UI)->getParent();
        if (!std::binary_search(LoopBlocks.begin(), LoopBlocks.end(), UserBB))
          AffectedValues.insert(I);
      }
  }
  return AffectedValues;
}
