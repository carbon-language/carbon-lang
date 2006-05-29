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
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/CFG.h"
#include <algorithm>
#include <map>
#include <vector>

using namespace llvm;

namespace {
  static Statistic<> NumLCSSA("lcssa",
                              "Number of live out of a loop variables");
  
  class LCSSA : public FunctionPass {
  public:
    
  
    LoopInfo *LI;  // Loop information
    DominatorTree *DT;       // Dominator Tree for the current Loop...
    DominanceFrontier *DF;   // Current Dominance Frontier
    
    virtual bool runOnFunction(Function &F);
    bool visitSubloop(Loop* L);
    
    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG.  It maintains both of these,
    /// as well as the CFG.  It also requires dominator information.
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequiredID(LoopSimplifyID);
      AU.addPreservedID(LoopSimplifyID);
      AU.addRequired<LoopInfo>();
      AU.addPreserved<LoopInfo>();
      AU.addRequired<DominatorTree>();
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
  
  // Phi nodes that need to be IDF-processed
  std::vector<PHINode*> workList;
  
  // Iterate over all affected values for this loop and insert Phi nodes
  // for them in the appropriate exit blocks
  std::map<BasicBlock*, PHINode*> ExitPhis;
  for (std::set<Instruction*>::iterator I = AffectedValues.begin(),
       E = AffectedValues.end(); I != E; ++I) {
    ++NumLCSSA; // We are applying the transformation
    for (std::vector<BasicBlock*>::iterator BBI = exitBlocks.begin(),
         BBE = exitBlocks.end(); BBI != BBE; ++BBI) {
      PHINode *phi = new PHINode((*I)->getType(), "lcssa");
      (*BBI)->getInstList().insert((*BBI)->front(), phi);
      workList.push_back(phi);
      ExitPhis[*BBI] = phi;
    
      // Since LoopSimplify has been run, we know that all of these predecessors
      // are in the loop, so just hook them up in the obvious manner.
      for (pred_iterator PI = pred_begin(*BBI), PE = pred_end(*BBI); PI != PE;
           ++PI)
        phi->addIncoming(*I, *PI);
    }
  
    // Calculate the IDF of these LCSSA Phi nodes, inserting new Phi's where
    // necessary.  Keep track of these new Phi's in DFPhis.
    std::map<BasicBlock*, PHINode*> DFPhis;
    for (std::vector<PHINode*>::iterator DFI = workList.begin(),
         E = workList.end(); DFI != E; ++DFI) {
    
      // Get the current Phi's DF, and insert Phi nodes.  Add these new
      // nodes to our worklist.
      DominanceFrontier::const_iterator it = DF->find((*DFI)->getParent());
      if (it != DF->end()) {
        const DominanceFrontier::DomSetType &S = it->second;
        for (DominanceFrontier::DomSetType::const_iterator P = S.begin(),
             PE = S.end(); P != PE; ++P) {
          if (DFPhis[*P] == 0) {
            // Still doesn't have operands...
            PHINode *phi = new PHINode((*DFI)->getType(), "lcssa");
            (*P)->getInstList().insert((*P)->front(), phi);
            DFPhis[*P] = phi;
          
            workList.push_back(phi);
          }
        }
      }
    
      // Get the predecessor blocks of the current Phi, and use them to hook up
      // the operands of the current Phi to any members of DFPhis that dominate
      // it.  This is a nop for the Phis inserted directly in the exit blocks,
      // since they are not dominated by any members of DFPhis.
      for (pred_iterator PI = pred_begin((*DFI)->getParent()),
           E = pred_end((*DFI)->getParent()); PI != E; ++PI)
        for (std::map<BasicBlock*, PHINode*>::iterator MI = DFPhis.begin(),
             ME = DFPhis.end(); MI != ME; ++MI)
          if (DT->getNode((*MI).first)->dominates(DT->getNode(*PI))) {
            (*DFI)->addIncoming((*MI).second, *PI);
          
            // Since dominate() is not cheap, don't do it more than we have to.
            break;
          }
    }
    
    
    
    // Find all uses of the affected value, and replace them with the
    // appropriate Phi.
    for (Instruction::use_iterator UI = (*I)->use_begin(), UE=(*I)->use_end();
         UI != UE; ++UI) {
      Instruction* use = cast<Instruction>(*UI);
      
      // Don't need to update uses within the loop body
      if (!std::binary_search(LoopBlocks.begin(), LoopBlocks.end(),
          use->getParent())) {
          
        for (std::map<BasicBlock*, PHINode*>::iterator DI = ExitPhis.begin(),
             DE = ExitPhis.end(); DI != DE; ++DI) {
          if (DT->getNode((*DI).first)->dominates( \
              DT->getNode(use->getParent())) && use != (*DI).second) {
            use->replaceUsesOfWith(*I, (*DI).second);
            break;
          }
        }
          
        for (std::map<BasicBlock*, PHINode*>::iterator DI = DFPhis.begin(),
             DE = DFPhis.end(); DI != DE; ++DI) {
          if (DT->getNode((*DI).first)->dominates( \
              DT->getNode(use->getParent()))) {
            use->replaceUsesOfWith(*I, (*DI).second);
            break;
          }
        }
      }
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
