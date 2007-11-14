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
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Streams.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>
#include <ostream>
using namespace llvm;

char LoopInfo::ID = 0;
static RegisterPass<LoopInfo>
X("loops", "Natural Loop Construction", true);

//===----------------------------------------------------------------------===//
// Loop implementation
//

/// getNumBackEdges - Calculate the number of back edges to the loop header.
///

//===----------------------------------------------------------------------===//
// LoopInfo implementation
//
bool LoopInfo::runOnFunction(Function &) {
  releaseMemory();
  Calculate(getAnalysis<DominatorTree>());    // Update
  return false;
}

void LoopInfo::releaseMemory() {
  for (std::vector<Loop*>::iterator I = TopLevelLoops.begin(),
         E = TopLevelLoops.end(); I != E; ++I)
    delete *I;   // Delete all of the loops...

  BBMap.clear();                             // Reset internal state of analysis
  TopLevelLoops.clear();
}

void LoopInfo::Calculate(DominatorTree &DT) {
  BasicBlock *RootNode = DT.getRootNode()->getBlock();

  for (df_iterator<BasicBlock*> NI = df_begin(RootNode),
         NE = df_end(RootNode); NI != NE; ++NI)
    if (Loop *L = ConsiderForLoop(*NI, DT))
      TopLevelLoops.push_back(L);
}

void LoopInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<DominatorTree>();
}

void LoopInfo::print(std::ostream &OS, const Module* ) const {
  for (unsigned i = 0; i < TopLevelLoops.size(); ++i)
    TopLevelLoops[i]->print(OS);
#if 0
  for (std::map<BasicBlock*, Loop*>::const_iterator I = BBMap.begin(),
         E = BBMap.end(); I != E; ++I)
    OS << "BB '" << I->first->getName() << "' level = "
       << I->second->getLoopDepth() << "\n";
#endif
}

static bool isNotAlreadyContainedIn(Loop *SubLoop, Loop *ParentLoop) {
  if (SubLoop == 0) return true;
  if (SubLoop == ParentLoop) return false;
  return isNotAlreadyContainedIn(SubLoop->getParentLoop(), ParentLoop);
}

Loop *LoopInfo::ConsiderForLoop(BasicBlock *BB, DominatorTree &DT) {
  if (BBMap.find(BB) != BBMap.end()) return 0;   // Haven't processed this node?

  std::vector<BasicBlock *> TodoStack;

  // Scan the predecessors of BB, checking to see if BB dominates any of
  // them.  This identifies backedges which target this node...
  for (pred_iterator I = pred_begin(BB), E = pred_end(BB); I != E; ++I)
    if (DT.dominates(BB, *I))   // If BB dominates it's predecessor...
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
        DT.dominates(EntryBlock, X)) {   // X is reachable from entry block?
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
    if (Loop *NewLoop = ConsiderForLoop(*I, DT)) {
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

/// changeLoopFor - Change the top-level loop that contains BB to the
/// specified loop.  This should be used by transformations that restructure
/// the loop hierarchy tree.
void LoopInfo::changeLoopFor(BasicBlock *BB, Loop *L) {
  Loop *&OldLoop = BBMap[BB];
  assert(OldLoop && "Block not in a loop yet!");
  OldLoop = L;
}

/// changeTopLevelLoop - Replace the specified loop in the top-level loops
/// list with the indicated loop.
void LoopInfo::changeTopLevelLoop(Loop *OldLoop, Loop *NewLoop) {
  std::vector<Loop*>::iterator I = std::find(TopLevelLoops.begin(),
                                             TopLevelLoops.end(), OldLoop);
  assert(I != TopLevelLoops.end() && "Old loop not at top level!");
  *I = NewLoop;
  assert(NewLoop->ParentLoop == 0 && OldLoop->ParentLoop == 0 &&
         "Loops already embedded into a subloop!");
}

/// removeLoop - This removes the specified top-level loop from this loop info
/// object.  The loop is not deleted, as it will presumably be inserted into
/// another loop.
Loop *LoopInfo::removeLoop(iterator I) {
  assert(I != end() && "Cannot remove end iterator!");
  Loop *L = *I;
  assert(L->getParentLoop() == 0 && "Not a top-level loop!");
  TopLevelLoops.erase(TopLevelLoops.begin() + (I-begin()));
  return L;
}

/// removeBlock - This method completely removes BB from all data structures,
/// including all of the Loop objects it is nested in and our mapping from
/// BasicBlocks to loops.
void LoopInfo::removeBlock(BasicBlock *BB) {
  std::map<BasicBlock *, Loop*>::iterator I = BBMap.find(BB);
  if (I != BBMap.end()) {
    for (Loop *L = I->second; L; L = L->getParentLoop())
      L->removeBlockFromLoop(BB);

    BBMap.erase(I);
  }
}

// Ensure this file gets linked when LoopInfo.h is used.
DEFINING_FILE_FOR(LoopInfo)
