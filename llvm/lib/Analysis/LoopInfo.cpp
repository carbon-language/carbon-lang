//===- LoopInfo.cpp - Natural Loop Calculator -------------------------------=//
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

static RegisterAnalysis<LoopInfo>
X("loops", "Natural Loop Construction", true);

//===----------------------------------------------------------------------===//
// Loop implementation
//
bool Loop::contains(const BasicBlock *BB) const {
  return find(Blocks.begin(), Blocks.end(), BB) != Blocks.end();
}

bool Loop::isLoopExit(const BasicBlock *BB) const {
  for (BasicBlock::succ_const_iterator SI = succ_begin(BB), SE = succ_end(BB);
       SI != SE; ++SI) {
    if (! contains(*SI))
      return true;
  }
  return false;
}

unsigned Loop::getNumBackEdges() const {
  unsigned numBackEdges = 0;
  BasicBlock *header = Blocks.front();

  for (std::vector<BasicBlock*>::const_iterator I = Blocks.begin(),
         E = Blocks.end(); I != E; ++I) {
    for (BasicBlock::succ_iterator SI = succ_begin(*I), SE = succ_end(*I);
         SI != SE; ++SI)
      if (header == *SI)
	++numBackEdges;
  }
  return numBackEdges;
}

void Loop::print(std::ostream &OS) const {
  OS << std::string(getLoopDepth()*2, ' ') << "Loop Containing: ";

  for (unsigned i = 0; i < getBlocks().size(); ++i) {
    if (i) OS << ",";
    WriteAsOperand(OS, (const Value*)getBlocks()[i]);
  }
  OS << "\n";

  for (unsigned i = 0, e = getSubLoops().size(); i != e; ++i)
    getSubLoops()[i]->print(OS);
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
}

Loop *LoopInfo::ConsiderForLoop(BasicBlock *BB, const DominatorSet &DS) {
  if (BBMap.find(BB) != BBMap.end()) return 0;   // Haven't processed this node?

  std::vector<BasicBlock *> TodoStack;

  // Scan the predecessors of BB, checking to see if BB dominates any of
  // them.
  for (pred_iterator I = pred_begin(BB), E = pred_end(BB); I != E; ++I)
    if (DS.dominates(BB, *I))   // If BB dominates it's predecessor...
      TodoStack.push_back(*I);

  if (TodoStack.empty()) return 0;  // Doesn't dominate any predecessors...

  // Create a new loop to represent this basic block...
  Loop *L = new Loop(BB);
  BBMap[BB] = L;

  while (!TodoStack.empty()) {  // Process all the nodes in the loop
    BasicBlock *X = TodoStack.back();
    TodoStack.pop_back();

    if (!L->contains(X)) {                  // As of yet unprocessed??
      L->Blocks.push_back(X);

      // Add all of the predecessors of X to the end of the work stack...
      TodoStack.insert(TodoStack.end(), pred_begin(X), pred_end(X));
    }
  }

  // Add the basic blocks that comprise this loop to the BBMap so that this
  // loop can be found for them.  Also check subsidary basic blocks to see if
  // they start subloops of their own.
  //
  for (std::vector<BasicBlock*>::reverse_iterator I = L->Blocks.rbegin(),
	 E = L->Blocks.rend(); I != E; ++I)

    // Check to see if this block starts a new loop
    if (*I != BB)
      if (Loop *NewLoop = ConsiderForLoop(*I, DS)) {
        L->SubLoops.push_back(NewLoop);
        NewLoop->ParentLoop = L;
      } else {
        std::map<BasicBlock*, Loop*>::iterator BBMI = BBMap.lower_bound(*I);
        if (BBMI == BBMap.end() || BBMI->first != *I) {  // Not in map yet...
          BBMap.insert(BBMI, std::make_pair(*I, L));
        } else {
          // If this is already in the BBMap then this means that we already
          // added a loop for it, but incorrectly added the loop to a higher
          // level loop instead of the current loop we are creating.  Fix this
          // now by moving the loop into the correct subloop.
          //
          Loop *SubLoop = BBMI->second;
          if (SubLoop->getHeader() == *I) { // Only do this once for the loop...
            Loop *OldSubLoopParent = SubLoop->getParentLoop();
            if (OldSubLoopParent != L) {
              // Remove SubLoop from OldSubLoopParent's list of subloops...
              std::vector<Loop*>::iterator I =
                std::find(OldSubLoopParent->SubLoops.begin(),
                          OldSubLoopParent->SubLoops.end(), SubLoop);
              assert(I != OldSubLoopParent->SubLoops.end()
                     && "Loop parent doesn't contain loop?");
              OldSubLoopParent->SubLoops.erase(I);
              SubLoop->ParentLoop = L;
              L->SubLoops.push_back(SubLoop);
            }
          }
        }
      }

  return L;
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
