//===- LoopPreheaders.cpp - Loop Preheader Insertion Pass -----------------===//
//
// Insert Loop pre-headers and exit blocks into the CFG for each function in the
// module.  This pass updates loop information and dominator information.
//
// Loop pre-header insertion guarantees that there is a single, non-critical
// entry edge from outside of the loop to the loop header.  This simplifies a
// number of analyses and transformations, such as LICM.
//
// Loop exit-block insertion guarantees that all exit blocks from the loop
// (blocks which are outside of the loop that have predecessors inside of the
// loop) are dominated by the loop header.  This simplifies transformations such
// as store-sinking that is built into LICM.
//
// Note that the simplifycfg pass will clean up blocks which are split out but
// end up being unneccesary, so usage of this pass does not neccesarily
// pessimize generated code.
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
#include "Support/SetOperations.h"
#include "Support/Statistic.h"

namespace {
  Statistic<> NumInserted("preheaders", "Number of pre-header nodes inserted");

  struct Preheaders : public FunctionPass {
    virtual bool runOnFunction(Function &F);
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      // We need loop information to identify the loops...
      AU.addRequired<LoopInfo>();
      AU.addRequired<DominatorSet>();

      AU.addPreserved<LoopInfo>();
      AU.addPreserved<DominatorSet>();
      AU.addPreserved<ImmediateDominators>();
      AU.addPreserved<DominatorTree>();
      AU.addPreserved<DominanceFrontier>();
      AU.addPreservedID(BreakCriticalEdgesID);  // No crit edges added....
    }
  private:
    bool ProcessLoop(Loop *L);
    BasicBlock *SplitBlockPredecessors(BasicBlock *BB, const char *Suffix,
                                       const std::vector<BasicBlock*> &Preds);
    void RewriteLoopExitBlock(Loop *L, BasicBlock *Exit);
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

  DominatorSet &DS = getAnalysis<DominatorSet>();
  BasicBlock *Header = L->getHeader();
  for (unsigned i = 0, e = L->getExitBlocks().size(); i != e; ++i)
    if (!DS.dominates(Header, L->getExitBlocks()[i])) {
      RewriteLoopExitBlock(L, L->getExitBlocks()[i]);
      NumInserted++;
      Changed = true;
    }

  const std::vector<Loop*> &SubLoops = L->getSubLoops();
  for (unsigned i = 0, e = SubLoops.size(); i != e; ++i)
    Changed |= ProcessLoop(SubLoops[i]);
  return Changed;
}

/// SplitBlockPredecessors - Split the specified block into two blocks.  We want
/// to move the predecessors specified in the Preds list to point to the new
/// block, leaving the remaining predecessors pointing to BB.  This method
/// updates the SSA PHINode's, but no other analyses.
///
BasicBlock *Preheaders::SplitBlockPredecessors(BasicBlock *BB,
                                               const char *Suffix,
                                       const std::vector<BasicBlock*> &Preds) {
  
  // Create new basic block, insert right before the original block...
  BasicBlock *NewBB = new BasicBlock(BB->getName()+Suffix, BB);

  // The preheader first gets an unconditional branch to the loop header...
  BranchInst *BI = new BranchInst(BB);
  NewBB->getInstList().push_back(BI);
  
  // For every PHI node in the block, insert a PHI node into NewBB where the
  // incoming values from the out of loop edges are moved to NewBB.  We have two
  // possible cases here.  If the loop is dead, we just insert dummy entries
  // into the PHI nodes for the new edge.  If the loop is not dead, we move the
  // incoming edges in BB into new PHI nodes in NewBB.
  //
  if (!Preds.empty()) {  // Is the loop not obviously dead?
    for (BasicBlock::iterator I = BB->begin();
         PHINode *PN = dyn_cast<PHINode>(&*I); ++I) {
      
      // Create the new PHI node, insert it into NewBB at the end of the block
      PHINode *NewPHI = new PHINode(PN->getType(), PN->getName()+".ph", BI);
        
      // Move all of the edges from blocks outside the loop to the new PHI
      for (unsigned i = 0, e = Preds.size(); i != e; ++i) {
        Value *V = PN->removeIncomingValue(Preds[i]);
        NewPHI->addIncoming(V, Preds[i]);
      }
      
      // Add an incoming value to the PHI node in the loop for the preheader
      // edge
      PN->addIncoming(NewPHI, NewBB);
    }
    
    // Now that the PHI nodes are updated, actually move the edges from
    // Preds to point to NewBB instead of BB.
    //
    for (unsigned i = 0, e = Preds.size(); i != e; ++i) {
      TerminatorInst *TI = Preds[i]->getTerminator();
      for (unsigned s = 0, e = TI->getNumSuccessors(); s != e; ++s)
        if (TI->getSuccessor(s) == BB)
          TI->setSuccessor(s, NewBB);
    }
    
  } else {                       // Otherwise the loop is dead...
    for (BasicBlock::iterator I = BB->begin();
         PHINode *PN = dyn_cast<PHINode>(&*I); ++I)
      // Insert dummy values as the incoming value...
      PN->addIncoming(Constant::getNullValue(PN->getType()), NewBB);
  }  
  return NewBB;
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
  
  // Split out the loop pre-header
  BasicBlock *NewBB =
    SplitBlockPredecessors(Header, ".preheader", OutsideBlocks);
  
  //===--------------------------------------------------------------------===//
  //  Update analysis results now that we have preformed the transformation
  //
  
  // We know that we have loop information to update... update it now.
  if (Loop *Parent = L->getParentLoop())
    Parent->addBasicBlockToLoop(NewBB, getAnalysis<LoopInfo>());
  
  DominatorSet &DS = getAnalysis<DominatorSet>();  // Update dominator info
  {
    // The blocks that dominate NewBB are the blocks that dominate Header,
    // minus Header, plus NewBB.
    DominatorSet::DomSetType DomSet = DS.getDominators(Header);
    DomSet.insert(NewBB);  // We dominate ourself
    DomSet.erase(Header);  // Header does not dominate us...
    DS.addBasicBlock(NewBB, DomSet);

    // The newly created basic block dominates all nodes dominated by Header.
    for (Function::iterator I = Header->getParent()->begin(),
           E = Header->getParent()->end(); I != E; ++I)
      if (DS.dominates(Header, I))
        DS.addDominator(I, NewBB);
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
    DominatorTree::Node *PHNode = DT->createNewNode(NewBB,
                                                    HeaderNode->getIDom());
    
    // Change the header node so that PNHode is the new immediate dominator
    DT->changeImmediateDominator(HeaderNode, PHNode);
  }

  // Update dominance frontier information...
  if (DominanceFrontier *DF = getAnalysisToUpdate<DominanceFrontier>()) {
    // The DF(NewBB) is just (DF(Header)-Header), because NewBB dominates
    // everything that Header does, and it strictly dominates Header in
    // addition.
    assert(DF->find(Header) != DF->end() && "Header node doesn't have DF set?");
    DominanceFrontier::DomSetType NewDFSet = DF->find(Header)->second;
    NewDFSet.erase(Header);
    DF->addBasicBlock(NewBB, NewDFSet);

    // Now we must loop over all of the dominance frontiers in the function,
    // replacing occurances of Header with NewBB in some cases.  If a block
    // dominates a (now) predecessor of NewBB, but did not strictly dominate
    // Header, it will have Header in it's DF set, but should now have NewBB in
    // its set.
    for (unsigned i = 0, e = OutsideBlocks.size(); i != e; ++i) {
      // Get all of the dominators of the predecessor...
      const DominatorSet::DomSetType &PredDoms =
        DS.getDominators(OutsideBlocks[i]);
      for (DominatorSet::DomSetType::const_iterator PDI = PredDoms.begin(),
             PDE = PredDoms.end(); PDI != PDE; ++PDI) {
        BasicBlock *PredDom = *PDI;
        // If the loop header is in DF(PredDom), then PredDom didn't dominate
        // the header but did dominate a predecessor outside of the loop.  Now
        // we change this entry to include the preheader in the DF instead of
        // the header.
        DominanceFrontier::iterator DFI = DF->find(PredDom);
        assert(DFI != DF->end() && "No dominance frontier for node?");
        if (DFI->second.count(Header)) {
          DF->removeFromFrontier(DFI, Header);
          DF->addToFrontier(DFI, NewBB);
        }
      }
    }
  }
}

void Preheaders::RewriteLoopExitBlock(Loop *L, BasicBlock *Exit) {
  DominatorSet &DS = getAnalysis<DominatorSet>();
  assert(!DS.dominates(L->getHeader(), Exit) &&
         "Loop already dominates exit block??");
  
  std::vector<BasicBlock*> LoopBlocks;
  for (pred_iterator I = pred_begin(Exit), E = pred_end(Exit); I != E; ++I)
    if (L->contains(*I))
      LoopBlocks.push_back(*I);

  BasicBlock *NewBB =
    SplitBlockPredecessors(Exit, ".loopexit", LoopBlocks);
  
  // Update Loop Information - we know that the new block will be in the parent
  // loop of L.
  if (Loop *Parent = L->getParentLoop())
    Parent->addBasicBlockToLoop(NewBB, getAnalysis<LoopInfo>());

  // Update dominator information...  The blocks that dominate NewBB are the
  // intersection of the dominators of predecessors, plus the block itself.
  // The newly created basic block does not dominate anything except itself.
  //
  DominatorSet::DomSetType NewBBDomSet = DS.getDominators(LoopBlocks[0]);
  for (unsigned i = 1, e = LoopBlocks.size(); i != e; ++i)
    set_intersect(NewBBDomSet, DS.getDominators(LoopBlocks[i]));
  NewBBDomSet.insert(NewBB);  // All blocks dominate themselves...
  DS.addBasicBlock(NewBB, NewBBDomSet);

  // Update immediate dominator information if we have it...
  BasicBlock *NewBBIDom = 0;
  if (ImmediateDominators *ID = getAnalysisToUpdate<ImmediateDominators>()) {
    // This block does not strictly dominate anything, so it is not an immediate
    // dominator.  To find the immediate dominator of the new exit node, we
    // trace up the immediate dominators of a predecessor until we find a basic
    // block that dominates the exit block.
    //
    BasicBlock *Dom = LoopBlocks[0];  // Some random predecessor...
    while (!NewBBDomSet.count(Dom)) {  // Loop until we find a dominator...
      assert(Dom != 0 && "No shared dominator found???");
      Dom = ID->get(Dom);
    }

    // Set the immediate dominator now...
    ID->addNewBlock(NewBB, Dom);
    NewBBIDom = Dom;   // Reuse this if calculating DominatorTree info...
  }

  // Update DominatorTree information if it is active.
  if (DominatorTree *DT = getAnalysisToUpdate<DominatorTree>()) {
    // NewBB doesn't dominate anything, so just create a node and link it into
    // its immediate dominator.  If we don't have ImmediateDominator info
    // around, calculate the idom as above.
    DominatorTree::Node *NewBBIDomNode;
    if (NewBBIDom) {
      NewBBIDomNode = DT->getNode(NewBBIDom);
    } else {
      NewBBIDomNode = DT->getNode(LoopBlocks[0]); // Random pred
      while (!NewBBDomSet.count(NewBBIDomNode->getNode())) {
        NewBBIDomNode = NewBBIDomNode->getIDom();
        assert(NewBBIDomNode && "No shared dominator found??");
      }
    }

    // Create the new dominator tree node...
    DT->createNewNode(NewBB, NewBBIDomNode);
  }

  // Update dominance frontier information...
  if (DominanceFrontier *DF = getAnalysisToUpdate<DominanceFrontier>()) {
    // DF(NewBB) is {Exit} because NewBB does not strictly dominate Exit, but it
    // does dominate itself (and there is an edge (NewBB -> Exit)).
    DominanceFrontier::DomSetType NewDFSet;
    NewDFSet.insert(Exit);
    DF->addBasicBlock(NewBB, NewDFSet);

    // Now we must loop over all of the dominance frontiers in the function,
    // replacing occurances of Exit with NewBB in some cases.  If a block
    // dominates a (now) predecessor of NewBB, but did not strictly dominate
    // Exit, it will have Exit in it's DF set, but should now have NewBB in its
    // set.
    for (unsigned i = 0, e = LoopBlocks.size(); i != e; ++i) {
      // Get all of the dominators of the predecessor...
      const DominatorSet::DomSetType &PredDoms =DS.getDominators(LoopBlocks[i]);
      for (DominatorSet::DomSetType::const_iterator PDI = PredDoms.begin(),
             PDE = PredDoms.end(); PDI != PDE; ++PDI) {
        BasicBlock *PredDom = *PDI;
        // Make sure to only rewrite blocks that are part of the loop...
        if (L->contains(PredDom)) {
          // If the exit node is in DF(PredDom), then PredDom didn't dominate
          // Exit but did dominate a predecessor inside of the loop.  Now we
          // change this entry to include NewBB in the DF instead of Exit.
          DominanceFrontier::iterator DFI = DF->find(PredDom);
          assert(DFI != DF->end() && "No dominance frontier for node?");
          if (DFI->second.count(Exit)) {
            DF->removeFromFrontier(DFI, Exit);
            DF->addToFrontier(DFI, NewBB);
          }
        }
      }
    }
  }
}
