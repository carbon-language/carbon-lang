//===-- LCSSA.cpp - Convert loops into loop-closed SSA form         ------===//
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

#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/CFG.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <iostream>
#include <set>
#include <vector>

using namespace llvm;

namespace {
  class LCSSA : public FunctionPass {
  public:
    LoopInfo *LI;  // Loop information
    
    // LoopProcessWorklist - List of loops we need to process.
    std::vector<Loop*> LoopProcessWorklist;
    
    virtual bool runOnFunction(Function &F);
    
    bool visitLoop(Loop *L, Value *V);
    
    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG...
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(LoopSimplifyID);
      AU.addPreservedID(LoopSimplifyID);
      AU.addRequired<LoopInfo>();
      AU.addPreserved<LoopInfo>();
    }
  private:
    void addSubloopsToWorklist(Loop* L);
    std::set<Value*> loopValuesUsedOutsideLoop(Loop *L);
  };
  
  RegisterOpt<LCSSA> X("lcssa", "Loop-Closed SSA Form Pass");
}

FunctionPass *llvm::createLCSSAPass() { return new LCSSA(); }

bool LCSSA::runOnFunction(Function &F) {
  bool changed = false;
  LI = &getAnalysis<LoopInfo>();
    
  for (LoopInfo::iterator I = LI->begin(), E = LI->end(); I != E; ++I) {
    addSubloopsToWorklist(*I);
    LoopProcessWorklist.push_back(*I);
  }
      
  for (std::vector<Loop*>::iterator I = LoopProcessWorklist.begin(),
       E = LoopProcessWorklist.end(); I != E; ++I) {
    std::set<Value*> AffectedValues = loopValuesUsedOutsideLoop(*I);
    if (!AffectedValues.empty()) {
      for (std::set<Value*>::iterator VI = AffectedValues.begin(),
           VE = AffectedValues.end(); VI != VE; ++VI)
        changed |= visitLoop(*I, *VI);
    }
  }
      
  return changed;
}

bool LCSSA::visitLoop(Loop *L, Value* V) {
  // We will be doing lots of "loop contains block" queries.  Loop::contains is
  // linear time, use a set to speed this up.
  std::set<BasicBlock*> LoopBlocks;

  for (Loop::block_iterator BB = L->block_begin(), E = L->block_end();
       BB != E; ++BB)
    LoopBlocks.insert(*BB);
  
  std::vector<BasicBlock*> exitBlocks;
  L->getExitBlocks(exitBlocks);
  
  for (std::vector<BasicBlock*>::iterator BBI = exitBlocks.begin(),
       BBE = exitBlocks.end(); BBI != BBE; ++BBI) {
    PHINode *phi = new PHINode(V->getType(), "lcssa");
    (*BBI)->getInstList().insert((*BBI)->front(), phi);
    
    for (pred_iterator PI = pred_begin(*BBI), PE = pred_end(*BBI); PI != PE;
         ++PI)
      phi->addIncoming(V, *PI);
  }
  
  for (Value::use_iterator UI = V->use_begin(), UE = V->use_end(); UI != UE;
       ++UI) {
    BasicBlock *UserBB = cast<Instruction>(*UI)->getParent();
    if (!LoopBlocks.count(UserBB))
      ; // FIXME: This should update the SSA form through the rest of the graph.
  }
  
  return false;
}

void LCSSA::addSubloopsToWorklist(Loop* L) {
  for (Loop::iterator I = L->begin(), E = L->end(); I != E; ++I) {
    addSubloopsToWorklist(*I);
    LoopProcessWorklist.push_back(*I);
  }
}

/// loopValuesUsedOutsideLoop - Return true if there are any values defined in
/// the loop that are used by instructions outside of it.
std::set<Value*> LCSSA::loopValuesUsedOutsideLoop(Loop *L) {
  std::set<Value*> AffectedValues;

  // We will be doing lots of "loop contains block" queries.  Loop::contains is
  // linear time, use a set to speed this up.
  std::set<BasicBlock*> LoopBlocks;

  for (Loop::block_iterator BB = L->block_begin(), E = L->block_end();
       BB != E; ++BB)
    LoopBlocks.insert(*BB);
  
  for (Loop::block_iterator BB = L->block_begin(), E = L->block_end();
       BB != E; ++BB) {
    for (BasicBlock::iterator I = (*BB)->begin(), E = (*BB)->end(); I != E; ++I)
      for (Value::use_iterator UI = I->use_begin(), E = I->use_end(); UI != E;
           ++UI) {
        BasicBlock *UserBB = cast<Instruction>(*UI)->getParent();
        if (!LoopBlocks.count(UserBB))
          AffectedValues.insert(I);
      }
  }
  return AffectedValues;
}