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
  
  class LCSSA : public FunctionPass {
  public:
    
  
    LoopInfo *LI;  // Loop information
    DominatorTree *DT;       // Dominator Tree for the current Function...
    DominanceFrontier *DF;   // Current Dominance Frontier
    std::vector<BasicBlock*> LoopBlocks;
    
    virtual bool runOnFunction(Function &F);
    bool visitSubloop(Loop* L);
    void processInstruction(Instruction* Instr,
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
      AU.addRequired<DominanceFrontier>();
    }
  private:
    SetVector<Instruction*> getLoopValuesUsedOutsideLoop(Loop *L);
    Value *getValueDominatingBlock(BasicBlock *BB,
                                 std::map<BasicBlock*, Value*>& PotDoms) {
      return getValueDominatingDTNode(DT->getNode(BB), PotDoms);
    }
    Value *getValueDominatingDTNode(DominatorTree::Node *Node,
                                  std::map<BasicBlock*, Value*>& PotDoms);
                                      
    /// inLoop - returns true if the given block is within the current loop
    const bool inLoop(BasicBlock* B) {
      return std::binary_search(LoopBlocks.begin(), LoopBlocks.end(), B);
    }
  };
  
  RegisterOpt<LCSSA> X("lcssa", "Loop-Closed SSA Form Pass");
}

FunctionPass *llvm::createLCSSAPass() { return new LCSSA(); }
const PassInfo *llvm::LCSSAID = X.getPassInfo();

/// runOnFunction - Process all loops in the function, inner-most out.
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
       E = AffectedValues.end(); I != E; ++I) {
    processInstruction(*I, exitBlocks);
  }
  
  assert(L->isLCSSAForm());
  
  return true;
}

/// processInstruction - Given a live-out instruction, insert LCSSA Phi nodes,
/// eliminate all out-of-loop uses.
void LCSSA::processInstruction(Instruction* Instr,
                               const std::vector<BasicBlock*>& exitBlocks)
{
  ++NumLCSSA; // We are applying the transformation
  
  std::map<BasicBlock*, Value*> Phis;
  
  // Add the base instruction to the Phis list.  This makes tracking down
  // the dominating values easier when we're filling in Phi nodes.  This will
  // be removed later, before we perform use replacement.
  Phis[Instr->getParent()] = Instr;
  
  // Phi nodes that need to be IDF-processed
  std::vector<PHINode*> workList;
  
  for (std::vector<BasicBlock*>::const_iterator BBI = exitBlocks.begin(),
      BBE = exitBlocks.end(); BBI != BBE; ++BBI) {
    Value*& phi = Phis[*BBI];
    if (phi == 0 &&
        DT->getNode(Instr->getParent())->dominates(DT->getNode(*BBI))) {
      phi = new PHINode(Instr->getType(), Instr->getName()+".lcssa",
                                 (*BBI)->begin());
      workList.push_back(cast<PHINode>(phi));
    }
  }
  
  // Phi nodes that need to have their incoming values filled.
  std::vector<PHINode*> needIncomingValues;
  
  // Calculate the IDF of these LCSSA Phi nodes, inserting new Phi's where
  // necessary.  Keep track of these new Phi's in the "Phis" map.
  while (!workList.empty()) {
    PHINode *CurPHI = workList.back();
    workList.pop_back();
    
    // Even though we've removed this Phi from the work list, we still need
    // to fill in its incoming values.
    needIncomingValues.push_back(CurPHI);
    
    // Get the current Phi's DF, and insert Phi nodes.  Add these new
    // nodes to our worklist.
    DominanceFrontier::const_iterator it = DF->find(CurPHI->getParent());
    if (it != DF->end()) {
      const DominanceFrontier::DomSetType &S = it->second;
      for (DominanceFrontier::DomSetType::const_iterator P = S.begin(),
           PE = S.end(); P != PE; ++P) {
        if (DT->getNode(Instr->getParent())->dominates(DT->getNode(*P))) {
          Value *&Phi = Phis[*P];
          if (Phi == 0) {
            // Still doesn't have operands...
            Phi = new PHINode(Instr->getType(), Instr->getName()+".lcssa",
                              (*P)->begin());
          
            workList.push_back(cast<PHINode>(Phi));
          }
        }
      }
    }
  }
  
  // Fill in all Phis we've inserted that need their incoming values filled in.
  for (std::vector<PHINode*>::iterator IVI = needIncomingValues.begin(),
       IVE = needIncomingValues.end(); IVI != IVE; ++IVI)
    for (pred_iterator PI = pred_begin((*IVI)->getParent()),
         E = pred_end((*IVI)->getParent()); PI != E; ++PI)
      (*IVI)->addIncoming(getValueDominatingBlock(*PI, Phis),
                          *PI);
  
  // Find all uses of the affected value, and replace them with the
  // appropriate Phi.
  std::vector<Instruction*> Uses;
  for (Instruction::use_iterator UI = Instr->use_begin(), UE = Instr->use_end();
       UI != UE; ++UI) {
    Instruction* use = cast<Instruction>(*UI);
    BasicBlock* UserBB = use->getParent();
    if (PHINode* p = dyn_cast<PHINode>(use)) {
      unsigned OperandNo = UI.getOperandNo();
      UserBB = p->getIncomingBlock(OperandNo/2);
    }
    
    // Don't need to update uses within the loop body.
    if (!inLoop(use->getParent()))
      Uses.push_back(use);
  }
  
  for (std::vector<Instruction*>::iterator II = Uses.begin(), IE = Uses.end();
       II != IE; ++II) {
    if (PHINode* phi = dyn_cast<PHINode>(*II)) {
      for (unsigned int i = 0; i < phi->getNumIncomingValues(); ++i) {
        if (phi->getIncomingValue(i) == Instr) {
          Value* dominator = 
                        getValueDominatingBlock(phi->getIncomingBlock(i), Phis);
          phi->setIncomingValue(i, dominator);
        }
      }
    } else {
      Value *NewVal = getValueDominatingBlock((*II)->getParent(), Phis);
      (*II)->replaceUsesOfWith(Instr, NewVal);
    }
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
        
        if (!inLoop(UserBB)) {
          AffectedValues.insert(I);
          break;
        }
      }
  }
  return AffectedValues;
}

/// getValueDominatingBlock - Return the value within the potential dominators
/// map that dominates the given block.
Value *LCSSA::getValueDominatingDTNode(DominatorTree::Node *Node,
                              std::map<BasicBlock*, Value*>& PotDoms) {
  // FIXME: The following insertion should be in place rather than the if
  // statement.  Currently, this is due to the fact that LCSSA isn't smart 
  // enough to avoid inserting IDF Phis that don't dominate any uses.  In some 
  // of those cases, it could ask us to provide a dominating value for a block
  // that has none, so we need to return undef.
  //assert(Node != 0 && "Didn't find dom value?");
  if (Node == 0) return UndefValue::get(PotDoms.begin()->second->getType());
  
  Value *&CacheSlot = PotDoms[Node->getBlock()];
  if (CacheSlot) return CacheSlot;
  
  // Otherwise, return the value of the idom and remember this for next time.
  return CacheSlot = getValueDominatingDTNode(Node->getIDom(), PotDoms);
}
