//===- LoopInfo.cpp - Natural Loop Calculator -----------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the LoopInfo class that is used to identify natural loops
// and determine the loop depth of various nodes of the CFG.  Note that the
// loops identified may actually be several natural loops that share the same
// header node... not just a single natural loop.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Support/CFG.h"
#include "llvm/Assembly/Writer.h"
#include "Support/DepthFirstIterator.h"
#include <algorithm>

namespace llvm {

static RegisterAnalysis<LoopInfo>
X("loops", "Natural Loop Construction", true);

//===----------------------------------------------------------------------===//
// Loop implementation
//
bool Loop::contains(const BasicBlock *BB) const {
  return find(Blocks.begin(), Blocks.end(), BB) != Blocks.end();
}

bool Loop::isLoopExit(const BasicBlock *BB) const {
  for (succ_const_iterator SI = succ_begin(BB), SE = succ_end(BB);
       SI != SE; ++SI) {
    if (!contains(*SI))
      return true;
  }
  return false;
}

/// getNumBackEdges - Calculate the number of back edges to the loop header.
///
unsigned Loop::getNumBackEdges() const {
  unsigned NumBackEdges = 0;
  BasicBlock *H = getHeader();

  for (pred_iterator I = pred_begin(H), E = pred_end(H); I != E; ++I)
    if (contains(*I))
      ++NumBackEdges;

  return NumBackEdges;
}

void Loop::print(std::ostream &OS, unsigned Depth) const {
  OS << std::string(Depth*2, ' ') << "Loop Containing: ";

  for (unsigned i = 0; i < getBlocks().size(); ++i) {
    if (i) OS << ",";
    WriteAsOperand(OS, getBlocks()[i], false);
  }
  if (!ExitBlocks.empty()) {
    OS << "\tExitBlocks: ";
    for (unsigned i = 0; i < getExitBlocks().size(); ++i) {
      if (i) OS << ",";
      WriteAsOperand(OS, getExitBlocks()[i], false);
    }
  }

  OS << "\n";

  for (unsigned i = 0, e = getSubLoops().size(); i != e; ++i)
    getSubLoops()[i]->print(OS, Depth+2);
}

void Loop::dump() const {
  print(std::cerr);
}


//===----------------------------------------------------------------------===//
// LoopInfo implementation
//
void LoopInfo::stub() {}

bool LoopInfo::runOnFunction(Function &) {
  releaseMemory();
  Calculate(getAnalysis<DominatorSet>());    // Update
  return false;
}

void LoopInfo::releaseMemory() {
  for (std::vector<Loop*>::iterator I = TopLevelLoops.begin(),
         E = TopLevelLoops.end(); I != E; ++I)
    delete *I;   // Delete all of the loops...

  BBMap.clear();                             // Reset internal state of analysis
  TopLevelLoops.clear();
}


void LoopInfo::Calculate(const DominatorSet &DS) {
  BasicBlock *RootNode = DS.getRoot();

  for (df_iterator<BasicBlock*> NI = df_begin(RootNode),
	 NE = df_end(RootNode); NI != NE; ++NI)
    if (Loop *L = ConsiderForLoop(*NI, DS))
      TopLevelLoops.push_back(L);

  for (unsigned i = 0; i < TopLevelLoops.size(); ++i)
    TopLevelLoops[i]->setLoopDepth(1);
}

void LoopInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<DominatorSet>();
}

void LoopInfo::print(std::ostream &OS) const {
  for (unsigned i = 0; i < TopLevelLoops.size(); ++i)
    TopLevelLoops[i]->print(OS);
#if 0
  for (std::map<BasicBlock*, Loop*>::const_iterator I = BBMap.begin(),
         E = BBMap.end(); I != E; ++I)
    OS << "BB '" << I->first->getName() << "' level = "
       << I->second->LoopDepth << "\n";
#endif
}

static bool isNotAlreadyContainedIn(Loop *SubLoop, Loop *ParentLoop) {
  if (SubLoop == 0) return true;
  if (SubLoop == ParentLoop) return false;
  return isNotAlreadyContainedIn(SubLoop->getParentLoop(), ParentLoop);
}

Loop *LoopInfo::ConsiderForLoop(BasicBlock *BB, const DominatorSet &DS) {
  if (BBMap.find(BB) != BBMap.end()) return 0;   // Haven't processed this node?

  std::vector<BasicBlock *> TodoStack;

  // Scan the predecessors of BB, checking to see if BB dominates any of
  // them.  This identifies backedges which target this node...
  for (pred_iterator I = pred_begin(BB), E = pred_end(BB); I != E; ++I)
    if (DS.dominates(BB, *I))   // If BB dominates it's predecessor...
      TodoStack.push_back(*I);

  if (TodoStack.empty()) return 0;  // No backedges to this block...

  // Create a new loop to represent this basic block...
  Loop *L = new Loop(BB);
  BBMap[BB] = L;

  BasicBlock *EntryBlock = &BB->getParent()->getEntryBlock();

  while (!TodoStack.empty()) {  // Process all the nodes in the loop
    BasicBlock *X = TodoStack.back();
    TodoStack.pop_back();

    if (!L->contains(X) &&         // As of yet unprocessed??
        DS.dominates(EntryBlock, X)) {   // X is reachable from entry block?
      // Check to see if this block already belongs to a loop.  If this occurs
      // then we have a case where a loop that is supposed to be a child of the
      // current loop was processed before the current loop.  When this occurs,
      // this child loop gets added to a part of the current loop, making it a
      // sibling to the current loop.  We have to reparent this loop.
      if (Loop *SubLoop = const_cast<Loop*>(getLoopFor(X)))
        if (SubLoop->getHeader() == X && isNotAlreadyContainedIn(SubLoop, L)) {
          // Remove the subloop from it's current parent...
          assert(SubLoop->ParentLoop && SubLoop->ParentLoop != L);
          Loop *SLP = SubLoop->ParentLoop;  // SubLoopParent
          std::vector<Loop*>::iterator I =
            std::find(SLP->SubLoops.begin(), SLP->SubLoops.end(), SubLoop);
          assert(I != SLP->SubLoops.end() && "SubLoop not a child of parent?");
          SLP->SubLoops.erase(I);   // Remove from parent...
          
          // Add the subloop to THIS loop...
          SubLoop->ParentLoop = L;
          L->SubLoops.push_back(SubLoop);
        }

      // Normal case, add the block to our loop...
      L->Blocks.push_back(X);
        
      // Add all of the predecessors of X to the end of the work stack...
      TodoStack.insert(TodoStack.end(), pred_begin(X), pred_end(X));
    }
  }

  // If there are any loops nested within this loop, create them now!
  for (std::vector<BasicBlock*>::iterator I = L->Blocks.begin(),
	 E = L->Blocks.end(); I != E; ++I)
    if (Loop *NewLoop = ConsiderForLoop(*I, DS)) {
      L->SubLoops.push_back(NewLoop);
      NewLoop->ParentLoop = L;
    }

  // Add the basic blocks that comprise this loop to the BBMap so that this
  // loop can be found for them.
  //
  for (std::vector<BasicBlock*>::iterator I = L->Blocks.begin(),
	 E = L->Blocks.end(); I != E; ++I) {
    std::map<BasicBlock*, Loop*>::iterator BBMI = BBMap.lower_bound(*I);
    if (BBMI == BBMap.end() || BBMI->first != *I)  // Not in map yet...
      BBMap.insert(BBMI, std::make_pair(*I, L));   // Must be at this level
  }

  // Now that we have a list of all of the child loops of this loop, check to
  // see if any of them should actually be nested inside of each other.  We can
  // accidentally pull loops our of their parents, so we must make sure to
  // organize the loop nests correctly now.
  {
    std::map<BasicBlock*, Loop*> ContainingLoops;
    for (unsigned i = 0; i != L->SubLoops.size(); ++i) {
      Loop *Child = L->SubLoops[i];
      assert(Child->getParentLoop() == L && "Not proper child loop?");

      if (Loop *ContainingLoop = ContainingLoops[Child->getHeader()]) {
        // If there is already a loop which contains this loop, move this loop
        // into the containing loop.
        MoveSiblingLoopInto(Child, ContainingLoop);
        --i;  // The loop got removed from the SubLoops list.
      } else {
        // This is currently considered to be a top-level loop.  Check to see if
        // any of the contained blocks are loop headers for subloops we have
        // already processed.
        for (unsigned b = 0, e = Child->Blocks.size(); b != e; ++b) {
          Loop *&BlockLoop = ContainingLoops[Child->Blocks[b]];
          if (BlockLoop == 0) {   // Child block not processed yet...
            BlockLoop = Child;
          } else if (BlockLoop != Child) {
            Loop *SubLoop = BlockLoop;
            // Reparent all of the blocks which used to belong to BlockLoops
            for (unsigned j = 0, e = SubLoop->Blocks.size(); j != e; ++j)
              ContainingLoops[SubLoop->Blocks[j]] = Child;

            // There is already a loop which contains this block, that means
            // that we should reparent the loop which the block is currently
            // considered to belong to to be a child of this loop.
            MoveSiblingLoopInto(SubLoop, Child);
            --i;  // We just shrunk the SubLoops list.
          }
        }
      }      
    }
  }

  // Now that we know all of the blocks that make up this loop, see if there are
  // any branches to outside of the loop... building the ExitBlocks list.
  for (std::vector<BasicBlock*>::iterator BI = L->Blocks.begin(),
         BE = L->Blocks.end(); BI != BE; ++BI)
    for (succ_iterator I = succ_begin(*BI), E = succ_end(*BI); I != E; ++I)
      if (!L->contains(*I))               // Not in current loop?
        L->ExitBlocks.push_back(*I);      // It must be an exit block...

  return L;
}

/// MoveSiblingLoopInto - This method moves the NewChild loop to live inside of
/// the NewParent Loop, instead of being a sibling of it.
void LoopInfo::MoveSiblingLoopInto(Loop *NewChild, Loop *NewParent) {
  Loop *OldParent = NewChild->getParentLoop();
  assert(OldParent && OldParent == NewParent->getParentLoop() &&
         NewChild != NewParent && "Not sibling loops!");

  // Remove NewChild from being a child of OldParent
  std::vector<Loop*>::iterator I =
    std::find(OldParent->SubLoops.begin(), OldParent->SubLoops.end(), NewChild);
  assert(I != OldParent->SubLoops.end() && "Parent fields incorrect??");
  OldParent->SubLoops.erase(I);   // Remove from parent's subloops list
  NewChild->ParentLoop = 0;
  
  InsertLoopInto(NewChild, NewParent);  
}

/// InsertLoopInto - This inserts loop L into the specified parent loop.  If the
/// parent loop contains a loop which should contain L, the loop gets inserted
/// into L instead.
void LoopInfo::InsertLoopInto(Loop *L, Loop *Parent) {
  BasicBlock *LHeader = L->getHeader();
  assert(Parent->contains(LHeader) && "This loop should not be inserted here!");
  
  // Check to see if it belongs in a child loop...
  for (unsigned i = 0, e = Parent->SubLoops.size(); i != e; ++i)
    if (Parent->SubLoops[i]->contains(LHeader)) {
      InsertLoopInto(L, Parent->SubLoops[i]);
      return;
    }      

  // If not, insert it here!
  Parent->SubLoops.push_back(L);
  L->ParentLoop = Parent;
}



/// getLoopPreheader - If there is a preheader for this loop, return it.  A
/// loop has a preheader if there is only one edge to the header of the loop
/// from outside of the loop.  If this is the case, the block branching to the
/// header of the loop is the preheader node.  The "preheaders" pass can be
/// "Required" to ensure that there is always a preheader node for every loop.
///
/// This method returns null if there is no preheader for the loop (either
/// because the loop is dead or because multiple blocks branch to the header
/// node of this loop).
///
BasicBlock *Loop::getLoopPreheader() const {
  // Keep track of nodes outside the loop branching to the header...
  BasicBlock *Out = 0;

  // Loop over the predecessors of the header node...
  BasicBlock *Header = getHeader();
  for (pred_iterator PI = pred_begin(Header), PE = pred_end(Header);
       PI != PE; ++PI)
    if (!contains(*PI)) {     // If the block is not in the loop...
      if (Out && Out != *PI)
        return 0;             // Multiple predecessors outside the loop
      Out = *PI;
    }
  
  // Make sure there is only one exit out of the preheader...
  succ_iterator SI = succ_begin(Out);
  ++SI;
  if (SI != succ_end(Out))
    return 0;  // Multiple exits from the block, must not be a preheader.


  // If there is exactly one preheader, return it.  If there was zero, then Out
  // is still null.
  return Out;
}

/// addBasicBlockToLoop - This function is used by other analyses to update loop
/// information.  NewBB is set to be a new member of the current loop.  Because
/// of this, it is added as a member of all parent loops, and is added to the
/// specified LoopInfo object as being in the current basic block.  It is not
/// valid to replace the loop header with this method.
///
void Loop::addBasicBlockToLoop(BasicBlock *NewBB, LoopInfo &LI) {
  assert(LI[getHeader()] == this && "Incorrect LI specified for this loop!");
  assert(NewBB && "Cannot add a null basic block to the loop!");
  assert(LI[NewBB] == 0 && "BasicBlock already in the loop!");

  // Add the loop mapping to the LoopInfo object...
  LI.BBMap[NewBB] = this;

  // Add the basic block to this loop and all parent loops...
  Loop *L = this;
  while (L) {
    L->Blocks.push_back(NewBB);
    L = L->getParentLoop();
  }
}

/// changeExitBlock - This method is used to update loop information.  All
/// instances of the specified Old basic block are removed from the exit list
/// and replaced with New.
///
void Loop::changeExitBlock(BasicBlock *Old, BasicBlock *New) {
  assert(Old != New && "Cannot changeExitBlock to the same thing!");
  assert(Old && New && "Cannot changeExitBlock to or from a null node!");
  assert(hasExitBlock(Old) && "Old exit block not found!");
  std::vector<BasicBlock*>::iterator
    I = std::find(ExitBlocks.begin(), ExitBlocks.end(), Old);
  while (I != ExitBlocks.end()) {
    *I = New;
    I = std::find(I+1, ExitBlocks.end(), Old);
  }
}

} // End llvm namespace
