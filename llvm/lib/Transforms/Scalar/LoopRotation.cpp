//===- LoopRotation.cpp - Loop Rotation Pass ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Loop Rotation Pass.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-rotate"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Function.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/SmallVector.h"
using namespace llvm;

#define MAX_HEADER_SIZE 16

STATISTIC(NumRotated, "Number of loops rotated");
namespace {

  class LoopRotate : public LoopPass {
  public:
    static char ID; // Pass ID, replacement for typeid
    LoopRotate() : LoopPass(ID) {
      initializeLoopRotatePass(*PassRegistry::getPassRegistry());
    }

    // Rotate Loop L as many times as possible. Return true if
    // loop is rotated at least once.
    bool runOnLoop(Loop *L, LPPassManager &LPM);

    // LCSSA form makes instruction renaming easier.
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addPreserved<DominatorTree>();
      AU.addPreserved<DominanceFrontier>();
      AU.addRequired<LoopInfo>();
      AU.addPreserved<LoopInfo>();
      AU.addRequiredID(LoopSimplifyID);
      AU.addPreservedID(LoopSimplifyID);
      AU.addRequiredID(LCSSAID);
      AU.addPreservedID(LCSSAID);
      AU.addPreserved<ScalarEvolution>();
    }

    // Helper functions

    /// Do actual work
    bool rotateLoop(Loop *L, LPPassManager &LPM);
    
    /// Initialize local data
    void initialize();

    /// After loop rotation, loop pre-header has multiple sucessors.
    /// Insert one forwarding basic block to ensure that loop pre-header
    /// has only one successor.
    void preserveCanonicalLoopForm(LPPassManager &LPM);

  private:
    Loop *L;
    BasicBlock *OrigHeader;
    BasicBlock *OrigPreHeader;
    BasicBlock *OrigLatch;
    BasicBlock *NewHeader;
    BasicBlock *Exit;
    LPPassManager *LPM_Ptr;
  };
}
  
char LoopRotate::ID = 0;
INITIALIZE_PASS_BEGIN(LoopRotate, "loop-rotate", "Rotate Loops", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(LCSSA)
INITIALIZE_PASS_END(LoopRotate, "loop-rotate", "Rotate Loops", false, false)

Pass *llvm::createLoopRotatePass() { return new LoopRotate(); }

/// Rotate Loop L as many times as possible. Return true if
/// the loop is rotated at least once.
bool LoopRotate::runOnLoop(Loop *Lp, LPPassManager &LPM) {

  bool RotatedOneLoop = false;
  initialize();
  LPM_Ptr = &LPM;

  // One loop can be rotated multiple times.
  while (rotateLoop(Lp,LPM)) {
    RotatedOneLoop = true;
    initialize();
  }

  return RotatedOneLoop;
}

/// Rotate loop LP. Return true if the loop is rotated.
bool LoopRotate::rotateLoop(Loop *Lp, LPPassManager &LPM) {
  L = Lp;

  OrigPreHeader = L->getLoopPreheader();
  if (!OrigPreHeader) return false;

  OrigLatch = L->getLoopLatch();
  if (!OrigLatch) return false;

  OrigHeader =  L->getHeader();

  // If the loop has only one block then there is not much to rotate.
  if (L->getBlocks().size() == 1)
    return false;

  // If the loop header is not one of the loop exiting blocks then
  // either this loop is already rotated or it is not
  // suitable for loop rotation transformations.
  if (!L->isLoopExiting(OrigHeader))
    return false;

  BranchInst *BI = dyn_cast<BranchInst>(OrigHeader->getTerminator());
  if (!BI)
    return false;
  assert(BI->isConditional() && "Branch Instruction is not conditional");

  // Updating PHInodes in loops with multiple exits adds complexity. 
  // Keep it simple, and restrict loop rotation to loops with one exit only.
  // In future, lift this restriction and support for multiple exits if
  // required.
  SmallVector<BasicBlock*, 8> ExitBlocks;
  L->getExitBlocks(ExitBlocks);
  if (ExitBlocks.size() > 1)
    return false;

  // Check size of original header and reject
  // loop if it is very big.
  unsigned Size = 0;
  
  // FIXME: Use common api to estimate size.
  for (BasicBlock::const_iterator OI = OrigHeader->begin(), 
         OE = OrigHeader->end(); OI != OE; ++OI) {
    if (isa<PHINode>(OI)) 
      continue;           // PHI nodes don't count.
    if (isa<DbgInfoIntrinsic>(OI))
      continue;  // Debug intrinsics don't count as size.
    ++Size;
  }

  if (Size > MAX_HEADER_SIZE)
    return false;

  // Now, this loop is suitable for rotation.

  // Anything ScalarEvolution may know about this loop or the PHI nodes
  // in its header will soon be invalidated.
  if (ScalarEvolution *SE = getAnalysisIfAvailable<ScalarEvolution>())
    SE->forgetLoop(L);

  // Find new Loop header. NewHeader is a Header's one and only successor
  // that is inside loop.  Header's other successor is outside the
  // loop.  Otherwise loop is not suitable for rotation.
  Exit = BI->getSuccessor(0);
  NewHeader = BI->getSuccessor(1);
  if (L->contains(Exit))
    std::swap(Exit, NewHeader);
  assert(NewHeader && "Unable to determine new loop header");
  assert(L->contains(NewHeader) && !L->contains(Exit) && 
         "Unable to determine loop header and exit blocks");
  
  // This code assumes that the new header has exactly one predecessor.
  // Remove any single-entry PHI nodes in it.
  assert(NewHeader->getSinglePredecessor() &&
         "New header doesn't have one pred!");
  FoldSingleEntryPHINodes(NewHeader);

  // Begin by walking OrigHeader and populating ValueMap with an entry for
  // each Instruction.
  BasicBlock::iterator I = OrigHeader->begin(), E = OrigHeader->end();
  DenseMap<const Value *, Value *> ValueMap;

  // For PHI nodes, the value available in OldPreHeader is just the
  // incoming value from OldPreHeader.
  for (; PHINode *PN = dyn_cast<PHINode>(I); ++I)
    ValueMap[PN] = PN->getIncomingValue(PN->getBasicBlockIndex(OrigPreHeader));

  // For the rest of the instructions, either hoist to the OrigPreheader if
  // possible or create a clone in the OldPreHeader if not.
  TerminatorInst *LoopEntryBranch = OrigPreHeader->getTerminator();
  while (I != E) {
    Instruction *Inst = I++;
    
    // If the instruction's operands are invariant and it doesn't read or write
    // memory, then it is safe to hoist.  Doing this doesn't change the order of
    // execution in the preheader, but does prevent the instruction from
    // executing in each iteration of the loop.  This means it is safe to hoist
    // something that might trap, but isn't safe to hoist something that reads
    // memory (without proving that the loop doesn't write).
    if (L->hasLoopInvariantOperands(Inst) &&
        !Inst->mayReadFromMemory() && !Inst->mayWriteToMemory() &&
        !isa<TerminatorInst>(Inst)) {
      Inst->moveBefore(LoopEntryBranch);
      continue;
    }
    
    // Otherwise, create a duplicate of the instruction.
    Instruction *C = Inst->clone();
    C->setName(Inst->getName());
    C->insertBefore(LoopEntryBranch);
    ValueMap[Inst] = C;
  }

  // Along with all the other instructions, we just cloned OrigHeader's
  // terminator into OrigPreHeader. Fix up the PHI nodes in each of OrigHeader's
  // successors by duplicating their incoming values for OrigHeader.
  TerminatorInst *TI = OrigHeader->getTerminator();
  for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
    for (BasicBlock::iterator BI = TI->getSuccessor(i)->begin();
         PHINode *PN = dyn_cast<PHINode>(BI); ++BI)
      PN->addIncoming(PN->getIncomingValueForBlock(OrigHeader), OrigPreHeader);

  // Now that OrigPreHeader has a clone of OrigHeader's terminator, remove
  // OrigPreHeader's old terminator (the original branch into the loop), and
  // remove the corresponding incoming values from the PHI nodes in OrigHeader.
  LoopEntryBranch->eraseFromParent();
  for (I = OrigHeader->begin(); PHINode *PN = dyn_cast<PHINode>(I); ++I)
    PN->removeIncomingValue(PN->getBasicBlockIndex(OrigPreHeader));

  // Now fix up users of the instructions in OrigHeader, inserting PHI nodes
  // as necessary.
  SSAUpdater SSA;
  for (I = OrigHeader->begin(); I != E; ++I) {
    Value *OrigHeaderVal = I;
    Value *OrigPreHeaderVal = ValueMap[OrigHeaderVal];

    // The value now exits in two versions: the initial value in the preheader
    // and the loop "next" value in the original header.
    SSA.Initialize(OrigHeaderVal->getType(), OrigHeaderVal->getName());
    SSA.AddAvailableValue(OrigHeader, OrigHeaderVal);
    SSA.AddAvailableValue(OrigPreHeader, OrigPreHeaderVal);

    // Visit each use of the OrigHeader instruction.
    for (Value::use_iterator UI = OrigHeaderVal->use_begin(),
         UE = OrigHeaderVal->use_end(); UI != UE; ) {
      // Grab the use before incrementing the iterator.
      Use &U = UI.getUse();

      // Increment the iterator before removing the use from the list.
      ++UI;

      // SSAUpdater can't handle a non-PHI use in the same block as an
      // earlier def. We can easily handle those cases manually.
      Instruction *UserInst = cast<Instruction>(U.getUser());
      if (!isa<PHINode>(UserInst)) {
        BasicBlock *UserBB = UserInst->getParent();

        // The original users in the OrigHeader are already using the
        // original definitions.
        if (UserBB == OrigHeader)
          continue;

        // Users in the OrigPreHeader need to use the value to which the
        // original definitions are mapped.
        if (UserBB == OrigPreHeader) {
          U = OrigPreHeaderVal;
          continue;
        }
      }

      // Anything else can be handled by SSAUpdater.
      SSA.RewriteUse(U);
    }
  }

  // NewHeader is now the header of the loop.
  L->moveToHeader(NewHeader);

  // Move the original header to the bottom of the loop, where it now more
  // naturally belongs. This isn't necessary for correctness, and CodeGen can
  // usually reorder blocks on its own to fix things like this up, but it's
  // still nice to keep the IR readable.
  //
  // The original header should have only one predecessor at this point, since
  // we checked that the loop had a proper preheader and unique backedge before
  // we started.
  assert(OrigHeader->getSinglePredecessor() &&
         "Original loop header has too many predecessors after loop rotation!");
  OrigHeader->moveAfter(OrigHeader->getSinglePredecessor());

  // Also, since this original header only has one predecessor, zap its
  // PHI nodes, which are now trivial.
  FoldSingleEntryPHINodes(OrigHeader);

  // TODO: We could just go ahead and merge OrigHeader into its predecessor
  // at this point, if we don't mind updating dominator info.

  // Establish a new preheader, update dominators, etc.
  preserveCanonicalLoopForm(LPM);

  ++NumRotated;
  return true;
}

/// Initialize local data
void LoopRotate::initialize() {
  L = NULL;
  OrigHeader = NULL;
  OrigPreHeader = NULL;
  NewHeader = NULL;
  Exit = NULL;
}

/// After loop rotation, loop pre-header has multiple sucessors.
/// Insert one forwarding basic block to ensure that loop pre-header
/// has only one successor.
void LoopRotate::preserveCanonicalLoopForm(LPPassManager &LPM) {

  // Right now original pre-header has two successors, new header and
  // exit block. Insert new block between original pre-header and
  // new header such that loop's new pre-header has only one successor.
  BasicBlock *NewPreHeader = BasicBlock::Create(OrigHeader->getContext(),
                                                "bb.nph",
                                                OrigHeader->getParent(), 
                                                NewHeader);
  LoopInfo &LI = getAnalysis<LoopInfo>();
  if (Loop *PL = LI.getLoopFor(OrigPreHeader))
    PL->addBasicBlockToLoop(NewPreHeader, LI.getBase());
  BranchInst::Create(NewHeader, NewPreHeader);
  
  BranchInst *OrigPH_BI = cast<BranchInst>(OrigPreHeader->getTerminator());
  if (OrigPH_BI->getSuccessor(0) == NewHeader)
    OrigPH_BI->setSuccessor(0, NewPreHeader);
  else {
    assert(OrigPH_BI->getSuccessor(1) == NewHeader &&
           "Unexpected original pre-header terminator");
    OrigPH_BI->setSuccessor(1, NewPreHeader);
  }

  PHINode *PN;
  for (BasicBlock::iterator I = NewHeader->begin();
       (PN = dyn_cast<PHINode>(I)); ++I) {
    int index = PN->getBasicBlockIndex(OrigPreHeader);
    assert(index != -1 && "Expected incoming value from Original PreHeader");
    PN->setIncomingBlock(index, NewPreHeader);
    assert(PN->getBasicBlockIndex(OrigPreHeader) == -1 && 
           "Expected only one incoming value from Original PreHeader");
  }

  if (DominatorTree *DT = getAnalysisIfAvailable<DominatorTree>()) {
    DT->addNewBlock(NewPreHeader, OrigPreHeader);
    DT->changeImmediateDominator(L->getHeader(), NewPreHeader);
    DT->changeImmediateDominator(Exit, OrigPreHeader);
    for (Loop::block_iterator BI = L->block_begin(), BE = L->block_end();
         BI != BE; ++BI) {
      BasicBlock *B = *BI;
      if (L->getHeader() != B) {
        DomTreeNode *Node = DT->getNode(B);
        if (Node && Node->getBlock() == OrigHeader)
          DT->changeImmediateDominator(*BI, L->getHeader());
      }
    }
    DT->changeImmediateDominator(OrigHeader, OrigLatch);
  }

  if (DominanceFrontier *DF = getAnalysisIfAvailable<DominanceFrontier>()) {
    // New Preheader's dominance frontier is Exit block.
    DominanceFrontier::DomSetType NewPHSet;
    NewPHSet.insert(Exit);
    DF->addBasicBlock(NewPreHeader, NewPHSet);

    // New Header's dominance frontier now includes itself and Exit block
    DominanceFrontier::iterator HeadI = DF->find(L->getHeader());
    if (HeadI != DF->end()) {
      DominanceFrontier::DomSetType & HeaderSet = HeadI->second;
      HeaderSet.clear();
      HeaderSet.insert(L->getHeader());
      HeaderSet.insert(Exit);
    } else {
      DominanceFrontier::DomSetType HeaderSet;
      HeaderSet.insert(L->getHeader());
      HeaderSet.insert(Exit);
      DF->addBasicBlock(L->getHeader(), HeaderSet);
    }

    // Original header (new Loop Latch)'s dominance frontier is Exit.
    DominanceFrontier::iterator LatchI = DF->find(L->getLoopLatch());
    if (LatchI != DF->end()) {
      DominanceFrontier::DomSetType &LatchSet = LatchI->second;
      LatchSet = LatchI->second;
      LatchSet.clear();
      LatchSet.insert(Exit);
    } else {
      DominanceFrontier::DomSetType LatchSet;
      LatchSet.insert(Exit);
      DF->addBasicBlock(L->getHeader(), LatchSet);
    }

    // If a loop block dominates new loop latch then add to its frontiers
    // new header and Exit and remove new latch (which is equal to original
    // header).
    BasicBlock *NewLatch = L->getLoopLatch();

    assert(NewLatch == OrigHeader && "NewLatch is inequal to OrigHeader");

    if (DominatorTree *DT = getAnalysisIfAvailable<DominatorTree>()) {
      for (Loop::block_iterator BI = L->block_begin(), BE = L->block_end();
           BI != BE; ++BI) {
        BasicBlock *B = *BI;
        if (DT->dominates(B, NewLatch)) {
          DominanceFrontier::iterator BDFI = DF->find(B);
          if (BDFI != DF->end()) {
            DominanceFrontier::DomSetType &BSet = BDFI->second;
            BSet.erase(NewLatch);
            BSet.insert(L->getHeader());
            BSet.insert(Exit);
          } else {
            DominanceFrontier::DomSetType BSet;
            BSet.insert(L->getHeader());
            BSet.insert(Exit);
            DF->addBasicBlock(B, BSet);
          }
        }
      }
    }
  }

  // Preserve canonical loop form, which means Exit block should
  // have only one predecessor.
  SplitEdge(L->getLoopLatch(), Exit, this);

  assert(NewHeader && L->getHeader() == NewHeader &&
         "Invalid loop header after loop rotation");
  assert(NewPreHeader && L->getLoopPreheader() == NewPreHeader &&
         "Invalid loop preheader after loop rotation");
  assert(L->getLoopLatch() &&
         "Invalid loop latch after loop rotation");
}
