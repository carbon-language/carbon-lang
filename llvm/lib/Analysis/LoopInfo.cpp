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
#include "Support/DepthFirstIterator.h"
#include <algorithm>
using namespace llvm;

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
  OS << "\n";

  for (iterator I = begin(), E = end(); I != E; ++I)
    (*I)->print(OS, Depth+2);
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


//===----------------------------------------------------------------------===//
// APIs for simple analysis of the loop.
//

/// getExitBlocks - Return all of the successor blocks of this loop.  These
/// are the blocks _outside of the current loop_ which are branched to.
///
void Loop::getExitBlocks(std::vector<BasicBlock*> &ExitBlocks) const {
  for (std::vector<BasicBlock*>::const_iterator BI = Blocks.begin(),
         BE = Blocks.end(); BI != BE; ++BI)
    for (succ_iterator I = succ_begin(*BI), E = succ_end(*BI); I != E; ++I)
      if (!contains(*I))               // Not in current loop?
        ExitBlocks.push_back(*I);          // It must be an exit block...
}


/// getLoopPreheader - If there is a preheader for this loop, return it.  A
/// loop has a preheader if there is only one edge to the header of the loop
/// from outside of the loop.  If this is the case, the block branching to the
/// header of the loop is the preheader node.
///
/// This method returns null if there is no preheader for the loop.
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

/// getCanonicalInductionVariable - Check to see if the loop has a canonical
/// induction variable: an integer recurrence that starts at 0 and increments by
/// one each time through the loop.  If so, return the phi node that corresponds
/// to it.
///
PHINode *Loop::getCanonicalInductionVariable() const {
  BasicBlock *H = getHeader();

  BasicBlock *Incoming = 0, *Backedge = 0;
  pred_iterator PI = pred_begin(H);
  assert(PI != pred_end(H) && "Loop must have at least one backedge!");
  Backedge = *PI++;
  if (PI == pred_end(H)) return 0;  // dead loop
  Incoming = *PI++;
  if (PI != pred_end(H)) return 0;  // multiple backedges?

  if (contains(Incoming)) {
    if (contains(Backedge))
      return 0;
    std::swap(Incoming, Backedge);
  } else if (!contains(Backedge))
    return 0;

  // Loop over all of the PHI nodes, looking for a canonical indvar.
  for (BasicBlock::iterator I = H->begin();
       PHINode *PN = dyn_cast<PHINode>(I); ++I)
    if (Instruction *Inc =
        dyn_cast<Instruction>(PN->getIncomingValueForBlock(Backedge)))
      if (Inc->getOpcode() == Instruction::Add && Inc->getOperand(0) == PN)
        if (ConstantInt *CI = dyn_cast<ConstantInt>(Inc->getOperand(1)))
          if (CI->equalsInt(1))
            return PN;

  return 0;
}

/// getCanonicalInductionVariableIncrement - Return the LLVM value that holds
/// the canonical induction variable value for the "next" iteration of the loop.
/// This always succeeds if getCanonicalInductionVariable succeeds.
///
Instruction *Loop::getCanonicalInductionVariableIncrement() const {
  if (PHINode *PN = getCanonicalInductionVariable()) {
    bool P1InLoop = contains(PN->getIncomingBlock(1));
    return cast<Instruction>(PN->getIncomingValue(P1InLoop));
  }
  return 0;
}

/// getTripCount - Return a loop-invariant LLVM value indicating the number of
/// times the loop will be executed.  Note that this means that the backedge of
/// the loop executes N-1 times.  If the trip-count cannot be determined, this
/// returns null.
///
Value *Loop::getTripCount() const {
  // Canonical loops will end with a 'setne I, V', where I is the incremented
  // canonical induction variable and V is the trip count of the loop.
  Instruction *Inc = getCanonicalInductionVariableIncrement();
  if (Inc == 0) return 0;
  PHINode *IV = cast<PHINode>(Inc->getOperand(0));
  
  BasicBlock *BackedgeBlock =
    IV->getIncomingBlock(contains(IV->getIncomingBlock(1)));

  if (BranchInst *BI = dyn_cast<BranchInst>(BackedgeBlock->getTerminator()))
    if (SetCondInst *SCI = dyn_cast<SetCondInst>(BI->getCondition()))
      if (SCI->getOperand(0) == Inc)
        if (BI->getSuccessor(0) == getHeader()) {
          if (SCI->getOpcode() == Instruction::SetNE)
            return SCI->getOperand(1);
        } else if (SCI->getOpcode() == Instruction::SetEQ) {
          return SCI->getOperand(1);
        }
  
  return 0;
}


//===-------------------------------------------------------------------===//
// APIs for updating loop information after changing the CFG
//

/// addBasicBlockToLoop - This function is used by other analyses to update loop
/// information.  NewBB is set to be a new member of the current loop.  Because
/// of this, it is added as a member of all parent loops, and is added to the
/// specified LoopInfo object as being in the current basic block.  It is not
/// valid to replace the loop header with this method.
///
void Loop::addBasicBlockToLoop(BasicBlock *NewBB, LoopInfo &LI) {
  assert((Blocks.empty() || LI[getHeader()] == this) &&
         "Incorrect LI specified for this loop!");
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

/// replaceChildLoopWith - This is used when splitting loops up.  It replaces
/// the OldChild entry in our children list with NewChild, and updates the
/// parent pointers of the two loops as appropriate.
void Loop::replaceChildLoopWith(Loop *OldChild, Loop *NewChild) {
  assert(OldChild->ParentLoop == this && "This loop is already broken!");
  assert(NewChild->ParentLoop == 0 && "NewChild already has a parent!");
  std::vector<Loop*>::iterator I = std::find(SubLoops.begin(), SubLoops.end(),
                                             OldChild);
  assert(I != SubLoops.end() && "OldChild not in loop!");
  *I = NewChild;
  OldChild->ParentLoop = 0;
  NewChild->ParentLoop = this;

  // Update the loop depth of the new child.
  NewChild->setLoopDepth(LoopDepth+1);
}

/// addChildLoop - Add the specified loop to be a child of this loop.
///
void Loop::addChildLoop(Loop *NewChild) {
  assert(NewChild->ParentLoop == 0 && "NewChild already has a parent!");
  NewChild->ParentLoop = this;
  SubLoops.push_back(NewChild);

  // Update the loop depth of the new child.
  NewChild->setLoopDepth(LoopDepth+1);
}

template<typename T>
static void RemoveFromVector(std::vector<T*> &V, T *N) {
  typename std::vector<T*>::iterator I = std::find(V.begin(), V.end(), N);
  assert(I != V.end() && "N is not in this list!");
  V.erase(I);
}

/// removeChildLoop - This removes the specified child from being a subloop of
/// this loop.  The loop is not deleted, as it will presumably be inserted
/// into another loop.
Loop *Loop::removeChildLoop(iterator I) {
  assert(I != SubLoops.end() && "Cannot remove end iterator!");
  Loop *Child = *I;
  assert(Child->ParentLoop == this && "Child is not a child of this loop!");
  SubLoops.erase(SubLoops.begin()+(I-begin()));
  Child->ParentLoop = 0;
  return Child;
}


/// removeBlockFromLoop - This removes the specified basic block from the
/// current loop, updating the Blocks and ExitBlocks lists as appropriate.  This
/// does not update the mapping in the LoopInfo class.
void Loop::removeBlockFromLoop(BasicBlock *BB) {
  RemoveFromVector(Blocks, BB);
}
