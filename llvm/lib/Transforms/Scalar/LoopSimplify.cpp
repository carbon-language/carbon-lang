//===- LoopPreheaders.cpp - Loop Preheader Insertion Pass -----------------===//
//
// Insert Loop pre-headers into the CFG for each function in the module.  This
// pass updates loop information and dominator information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Function.h"
#include "llvm/iTerminators.h"
#include "llvm/iPHINode.h"
#include "llvm/Constant.h"
#include "llvm/Support/CFG.h"
#include "Support/StatisticReporter.h"

namespace {
  Statistic<> NumInserted("preheaders\t- Number of pre-header nodes inserted");

  struct Preheaders : public FunctionPass {
    virtual bool runOnFunction(Function &F);
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      // We need loop information to identify the loops...
      AU.addRequired<LoopInfo>();

      AU.addPreserved<LoopInfo>();
      AU.addPreserved<DominatorSet>();
      AU.addPreserved<ImmediateDominators>();
      AU.addPreserved<DominatorTree>();
      AU.addPreservedID(BreakCriticalEdgesID);  // No crit edges added....
    }
  private:
    bool ProcessLoop(Loop *L);
    void InsertPreheaderForLoop(Loop *L);
  };

  RegisterOpt<Preheaders> X("preheaders", "Natural loop pre-header insertion");
}

// Publically exposed interface to pass...
const PassInfo *LoopPreheadersID = X.getPassInfo();
Pass *createLoopPreheaderInsertionPass() { return new Preheaders(); }


/// runOnFunction - Run down all loops in the CFG (recursively, but we could do
/// it in any convenient order) inserting preheaders...
///
bool Preheaders::runOnFunction(Function &F) {
  bool Changed = false;
  LoopInfo &LI = getAnalysis<LoopInfo>();

  for (unsigned i = 0, e = LI.getTopLevelLoops().size(); i != e; ++i)
    Changed |= ProcessLoop(LI.getTopLevelLoops()[i]);

  return Changed;
}


/// ProcessLoop - Walk the loop structure in depth first order, ensuring that
/// all loops have preheaders.
///
bool Preheaders::ProcessLoop(Loop *L) {
  bool Changed = false;

  // Does the loop already have a preheader?  If so, don't modify the loop...
  if (L->getLoopPreheader() == 0) {
    InsertPreheaderForLoop(L);
    NumInserted++;
    Changed = true;
  }

  const std::vector<Loop*> &SubLoops = L->getSubLoops();
  for (unsigned i = 0, e = SubLoops.size(); i != e; ++i)
    Changed |= ProcessLoop(SubLoops[i]);
  return Changed;
}


/// InsertPreheaderForLoop - Once we discover that a loop doesn't have a
/// preheader, this method is called to insert one.  This method has two phases:
/// preheader insertion and analysis updating.
///
void Preheaders::InsertPreheaderForLoop(Loop *L) {
  BasicBlock *Header = L->getHeader();

  // Compute the set of predecessors of the loop that are not in the loop.
  std::vector<BasicBlock*> OutsideBlocks;
  for (pred_iterator PI = pred_begin(Header), PE = pred_end(Header);
       PI != PE; ++PI)
      if (!L->contains(*PI))           // Coming in from outside the loop?
        OutsideBlocks.push_back(*PI);  // Keep track of it...
  
  assert(OutsideBlocks.size() != 1 && "Loop already has a preheader!");
  
  // Create new basic block, insert right before the header of the loop...
  BasicBlock *NewBB = new BasicBlock(Header->getName()+".preheader", Header);

  // The preheader first gets an unconditional branch to the loop header...
  BranchInst *BI = new BranchInst(Header);
  NewBB->getInstList().push_back(BI);
  
  // For every PHI node in the loop body, insert a PHI node into NewBB where
  // the incoming values from the out of loop edges are moved to NewBB.  We
  // have two possible cases here.  If the loop is dead, we just insert dummy
  // entries into the PHI nodes for the new edge.  If the loop is not dead, we
  // move the incoming edges in Header into new PHI nodes in NewBB.
  //
  if (!OutsideBlocks.empty()) {  // Is the loop not obviously dead?
    for (BasicBlock::iterator I = Header->begin();
         PHINode *PN = dyn_cast<PHINode>(&*I); ++I) {
      
      // Create the new PHI node, insert it into NewBB at the end of the block
      PHINode *NewPHI = new PHINode(PN->getType(), PN->getName()+".ph", BI);
        
      // Move all of the edges from blocks outside the loop to the new PHI
      for (unsigned i = 0, e = OutsideBlocks.size(); i != e; ++i) {
        Value *V = PN->removeIncomingValue(OutsideBlocks[i]);
        NewPHI->addIncoming(V, OutsideBlocks[i]);
      }
      
      // Add an incoming value to the PHI node in the loop for the preheader
      // edge
      PN->addIncoming(NewPHI, NewBB);
    }
    
    // Now that the PHI nodes are updated, actually move the edges from
    // OutsideBlocks to point to NewBB instead of Header.
    //
    for (unsigned i = 0, e = OutsideBlocks.size(); i != e; ++i) {
      TerminatorInst *TI = OutsideBlocks[i]->getTerminator();
      for (unsigned s = 0, e = TI->getNumSuccessors(); s != e; ++s)
        if (TI->getSuccessor(s) == Header)
          TI->setSuccessor(s, NewBB);
    }
    
  } else {                       // Otherwise the loop is dead...
    for (BasicBlock::iterator I = Header->begin();
         PHINode *PN = dyn_cast<PHINode>(&*I); ++I)
      // Insert dummy values as the incoming value...
      PN->addIncoming(Constant::getNullValue(PN->getType()), NewBB);
  }


  //===--------------------------------------------------------------------===//
  //  Update analysis results now that we have preformed the transformation
  //
  
  // We know that we have loop information to update... update it now.
  if (Loop *Parent = L->getParentLoop())
    Parent->addBasicBlockToLoop(NewBB, getAnalysis<LoopInfo>());
  
  // Update dominator information if it is around...
  if (DominatorSet *DS = getAnalysisToUpdate<DominatorSet>()) {
    // We need to add information about the fact that NewBB dominates Header.
    DS->addDominator(Header, NewBB);
    
    // The blocks that dominate NewBB are the blocks that dominate Header,
    // minus Header, plus NewBB.
    DominatorSet::DomSetType DomSet = DS->getDominators(Header);
    DomSet.erase(Header);  // Header does not dominate us...
    DS->addBasicBlock(NewBB, DomSet);
  }
  
  // Update immediate dominator information if we have it...
  if (ImmediateDominators *ID = getAnalysisToUpdate<ImmediateDominators>()) {
    // Whatever i-dominated the header node now immediately dominates NewBB
    ID->addNewBlock(NewBB, ID->get(Header));
    
    // The preheader now is the immediate dominator for the header node...
    ID->setImmediateDominator(Header, NewBB);
  }
  
  // Update DominatorTree information if it is active.
  if (DominatorTree *DT = getAnalysisToUpdate<DominatorTree>()) {
    // The immediate dominator of the preheader is the immediate dominator of
    // the old header.
    //
    DominatorTree::Node *HeaderNode = DT->getNode(Header);
    DominatorTree::Node *PHNode = DT->createNewNode(NewBB, HeaderNode);
    
    // Change the header node so that PNHode is the new immediate dominator
    DT->changeImmediateDominator(HeaderNode, PHNode);
  }
}


