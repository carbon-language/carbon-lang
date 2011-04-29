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
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
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

    // LCSSA form makes instruction renaming easier.
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addPreserved<DominatorTree>();
      AU.addRequired<LoopInfo>();
      AU.addPreserved<LoopInfo>();
      AU.addRequiredID(LoopSimplifyID);
      AU.addPreservedID(LoopSimplifyID);
      AU.addRequiredID(LCSSAID);
      AU.addPreservedID(LCSSAID);
      AU.addPreserved<ScalarEvolution>();
    }

    bool runOnLoop(Loop *L, LPPassManager &LPM);
    bool rotateLoop(Loop *L);
    
  private:
    LoopInfo *LI;
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
bool LoopRotate::runOnLoop(Loop *L, LPPassManager &LPM) {
  LI = &getAnalysis<LoopInfo>();

  // One loop can be rotated multiple times.
  bool MadeChange = false;
  while (rotateLoop(L))
    MadeChange = true;

  return MadeChange;
}

/// RewriteUsesOfClonedInstructions - We just cloned the instructions from the
/// old header into the preheader.  If there were uses of the values produced by
/// these instruction that were outside of the loop, we have to insert PHI nodes
/// to merge the two values.  Do this now.
static void RewriteUsesOfClonedInstructions(BasicBlock *OrigHeader,
                                            BasicBlock *OrigPreheader,
                                            ValueToValueMapTy &ValueMap) {
  // Remove PHI node entries that are no longer live.
  BasicBlock::iterator I, E = OrigHeader->end();
  for (I = OrigHeader->begin(); PHINode *PN = dyn_cast<PHINode>(I); ++I)
    PN->removeIncomingValue(PN->getBasicBlockIndex(OrigPreheader));
    
  // Now fix up users of the instructions in OrigHeader, inserting PHI nodes
  // as necessary.
  SSAUpdater SSA;
  for (I = OrigHeader->begin(); I != E; ++I) {
    Value *OrigHeaderVal = I;
    
    // If there are no uses of the value (e.g. because it returns void), there
    // is nothing to rewrite.
    if (OrigHeaderVal->use_empty())
      continue;
    
    Value *OrigPreHeaderVal = ValueMap[OrigHeaderVal];

    // The value now exits in two versions: the initial value in the preheader
    // and the loop "next" value in the original header.
    SSA.Initialize(OrigHeaderVal->getType(), OrigHeaderVal->getName());
    SSA.AddAvailableValue(OrigHeader, OrigHeaderVal);
    SSA.AddAvailableValue(OrigPreheader, OrigPreHeaderVal);
    
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
        if (UserBB == OrigPreheader) {
          U = OrigPreHeaderVal;
          continue;
        }
      }
      
      // Anything else can be handled by SSAUpdater.
      SSA.RewriteUse(U);
    }
  }
}  

/// Rotate loop LP. Return true if the loop is rotated.
bool LoopRotate::rotateLoop(Loop *L) {
  // If the loop has only one block then there is not much to rotate.
  if (L->getBlocks().size() == 1)
    return false;
  
  BasicBlock *OrigHeader = L->getHeader();
  
  BranchInst *BI = dyn_cast<BranchInst>(OrigHeader->getTerminator());
  if (BI == 0 || BI->isUnconditional())
    return false;
  
  // If the loop header is not one of the loop exiting blocks then
  // either this loop is already rotated or it is not
  // suitable for loop rotation transformations.
  if (!L->isLoopExiting(OrigHeader))
    return false;

  // Updating PHInodes in loops with multiple exits adds complexity. 
  // Keep it simple, and restrict loop rotation to loops with one exit only.
  // In future, lift this restriction and support for multiple exits if
  // required.
  SmallVector<BasicBlock*, 8> ExitBlocks;
  L->getExitBlocks(ExitBlocks);
  if (ExitBlocks.size() > 1)
    return false;

  // Check size of original header and reject loop if it is very big.
  {
    CodeMetrics Metrics;
    Metrics.analyzeBasicBlock(OrigHeader);
    if (Metrics.NumInsts > MAX_HEADER_SIZE)
      return false;
  }

  // Now, this loop is suitable for rotation.
  BasicBlock *OrigPreheader = L->getLoopPreheader();
  BasicBlock *OrigLatch = L->getLoopLatch();
  
  // If the loop could not be converted to canonical form, it must have an
  // indirectbr in it, just give up.
  if (OrigPreheader == 0 || OrigLatch == 0)
    return false;

  // Anything ScalarEvolution may know about this loop or the PHI nodes
  // in its header will soon be invalidated.
  if (ScalarEvolution *SE = getAnalysisIfAvailable<ScalarEvolution>())
    SE->forgetLoop(L);

  // Find new Loop header. NewHeader is a Header's one and only successor
  // that is inside loop.  Header's other successor is outside the
  // loop.  Otherwise loop is not suitable for rotation.
  BasicBlock *Exit = BI->getSuccessor(0);
  BasicBlock *NewHeader = BI->getSuccessor(1);
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
  ValueToValueMapTy ValueMap;

  // For PHI nodes, the value available in OldPreHeader is just the
  // incoming value from OldPreHeader.
  for (; PHINode *PN = dyn_cast<PHINode>(I); ++I)
    ValueMap[PN] = PN->getIncomingValue(PN->getBasicBlockIndex(OrigPreheader));

  // For the rest of the instructions, either hoist to the OrigPreheader if
  // possible or create a clone in the OldPreHeader if not.
  TerminatorInst *LoopEntryBranch = OrigPreheader->getTerminator();
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
        !isa<TerminatorInst>(Inst) && !isa<DbgInfoIntrinsic>(Inst)) {
      Inst->moveBefore(LoopEntryBranch);
      continue;
    }
    
    // Otherwise, create a duplicate of the instruction.
    Instruction *C = Inst->clone();
    
    // Eagerly remap the operands of the instruction.
    RemapInstruction(C, ValueMap,
                     RF_NoModuleLevelChanges|RF_IgnoreMissingEntries);
    
    // With the operands remapped, see if the instruction constant folds or is
    // otherwise simplifyable.  This commonly occurs because the entry from PHI
    // nodes allows icmps and other instructions to fold.
    Value *V = SimplifyInstruction(C);
    if (V && LI->replacementPreservesLCSSAForm(C, V)) {
      // If so, then delete the temporary instruction and stick the folded value
      // in the map.
      delete C;
      ValueMap[Inst] = V;
    } else {
      // Otherwise, stick the new instruction into the new block!
      C->setName(Inst->getName());
      C->insertBefore(LoopEntryBranch);
      ValueMap[Inst] = C;
    }
  }

  // Along with all the other instructions, we just cloned OrigHeader's
  // terminator into OrigPreHeader. Fix up the PHI nodes in each of OrigHeader's
  // successors by duplicating their incoming values for OrigHeader.
  TerminatorInst *TI = OrigHeader->getTerminator();
  for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
    for (BasicBlock::iterator BI = TI->getSuccessor(i)->begin();
         PHINode *PN = dyn_cast<PHINode>(BI); ++BI)
      PN->addIncoming(PN->getIncomingValueForBlock(OrigHeader), OrigPreheader);

  // Now that OrigPreHeader has a clone of OrigHeader's terminator, remove
  // OrigPreHeader's old terminator (the original branch into the loop), and
  // remove the corresponding incoming values from the PHI nodes in OrigHeader.
  LoopEntryBranch->eraseFromParent();

  // If there were any uses of instructions in the duplicated block outside the
  // loop, update them, inserting PHI nodes as required
  RewriteUsesOfClonedInstructions(OrigHeader, OrigPreheader, ValueMap);

  // NewHeader is now the header of the loop.
  L->moveToHeader(NewHeader);
  assert(L->getHeader() == NewHeader && "Latch block is our new header");

  
  // At this point, we've finished our major CFG changes.  As part of cloning
  // the loop into the preheader we've simplified instructions and the
  // duplicated conditional branch may now be branching on a constant.  If it is
  // branching on a constant and if that constant means that we enter the loop,
  // then we fold away the cond branch to an uncond branch.  This simplifies the
  // loop in cases important for nested loops, and it also means we don't have
  // to split as many edges.
  BranchInst *PHBI = cast<BranchInst>(OrigPreheader->getTerminator());
  assert(PHBI->isConditional() && "Should be clone of BI condbr!");
  if (!isa<ConstantInt>(PHBI->getCondition()) ||
      PHBI->getSuccessor(cast<ConstantInt>(PHBI->getCondition())->isZero())
          != NewHeader) {
    // The conditional branch can't be folded, handle the general case.
    // Update DominatorTree to reflect the CFG change we just made.  Then split
    // edges as necessary to preserve LoopSimplify form.
    if (DominatorTree *DT = getAnalysisIfAvailable<DominatorTree>()) {
      // Since OrigPreheader now has the conditional branch to Exit block, it is
      // the dominator of Exit.
      DT->changeImmediateDominator(Exit, OrigPreheader);
      DT->changeImmediateDominator(NewHeader, OrigPreheader);
      
      // Update OrigHeader to be dominated by the new header block.
      DT->changeImmediateDominator(OrigHeader, OrigLatch);
    }
    
    // Right now OrigPreHeader has two successors, NewHeader and ExitBlock, and
    // thus is not a preheader anymore.  Split the edge to form a real preheader.
    BasicBlock *NewPH = SplitCriticalEdge(OrigPreheader, NewHeader, this);
    NewPH->setName(NewHeader->getName() + ".lr.ph");
    
    // Preserve canonical loop form, which means that 'Exit' should have only one
    // predecessor.
    BasicBlock *ExitSplit = SplitCriticalEdge(L->getLoopLatch(), Exit, this);
    ExitSplit->moveBefore(Exit);
  } else {
    // We can fold the conditional branch in the preheader, this makes things
    // simpler. The first step is to remove the extra edge to the Exit block.
    Exit->removePredecessor(OrigPreheader, true /*preserve LCSSA*/);
    BranchInst *NewBI = BranchInst::Create(NewHeader, PHBI);
    NewBI->setDebugLoc(PHBI->getDebugLoc());
    PHBI->eraseFromParent();
    
    // With our CFG finalized, update DomTree if it is available.
    if (DominatorTree *DT = getAnalysisIfAvailable<DominatorTree>()) {
      // Update OrigHeader to be dominated by the new header block.
      DT->changeImmediateDominator(NewHeader, OrigPreheader);
      DT->changeImmediateDominator(OrigHeader, OrigLatch);
    }
  }
  
  assert(L->getLoopPreheader() && "Invalid loop preheader after loop rotation");
  assert(L->getLoopLatch() && "Invalid loop latch after loop rotation");

  // Now that the CFG and DomTree are in a consistent state again, try to merge
  // the OrigHeader block into OrigLatch.  This will succeed if they are
  // connected by an unconditional branch.  This is just a cleanup so the
  // emitted code isn't too gross in this common case.
  MergeBlockIntoPredecessor(OrigHeader, this);
  
  ++NumRotated;
  return true;
}

