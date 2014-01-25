//===-- LCSSA.cpp - Convert loops into loop-closed SSA form ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass transforms loops by placing phi nodes at the end of the loops for
// all values that are live across the loop boundary.  For example, it turns
// the left into the right code:
// 
// for (...)                for (...)
//   if (c)                   if (c)
//     X1 = ...                 X1 = ...
//   else                     else
//     X2 = ...                 X2 = ...
//   X3 = phi(X1, X2)         X3 = phi(X1, X2)
// ... = X3 + 4             X4 = phi(X3)
//                          ... = X4 + 4
//
// This is still valid LLVM; the extra phi nodes are purely redundant, and will
// be trivially eliminated by InstCombine.  The major benefit of this 
// transformation is that it makes many other loop optimizations, such as 
// LoopUnswitching, simpler.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "lcssa"
#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/PredIteratorCache.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
using namespace llvm;

STATISTIC(NumLCSSA, "Number of live out of a loop variables");

/// Return true if the specified block is in the list.
static bool isExitBlock(BasicBlock *BB,
                        const SmallVectorImpl<BasicBlock *> &ExitBlocks) {
  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i)
    if (ExitBlocks[i] == BB)
      return true;
  return false;
}

/// Given an instruction in the loop, check to see if it has any uses that are
/// outside the current loop.  If so, insert LCSSA PHI nodes and rewrite the
/// uses.
static bool processInstruction(Loop &L, Instruction &Inst, DominatorTree &DT,
                               const SmallVectorImpl<BasicBlock *> &ExitBlocks,
                               PredIteratorCache &PredCache) {
  SmallVector<Use *, 16> UsesToRewrite;

  BasicBlock *InstBB = Inst.getParent();

  for (Value::use_iterator UI = Inst.use_begin(), E = Inst.use_end(); UI != E;
       ++UI) {
    User *U = *UI;
    BasicBlock *UserBB = cast<Instruction>(U)->getParent();
    if (PHINode *PN = dyn_cast<PHINode>(U))
      UserBB = PN->getIncomingBlock(UI);

    if (InstBB != UserBB && !L.contains(UserBB))
      UsesToRewrite.push_back(&UI.getUse());
  }

  // If there are no uses outside the loop, exit with no change.
  if (UsesToRewrite.empty())
    return false;

  ++NumLCSSA; // We are applying the transformation

  // Invoke instructions are special in that their result value is not available
  // along their unwind edge. The code below tests to see whether DomBB
  // dominates
  // the value, so adjust DomBB to the normal destination block, which is
  // effectively where the value is first usable.
  BasicBlock *DomBB = Inst.getParent();
  if (InvokeInst *Inv = dyn_cast<InvokeInst>(&Inst))
    DomBB = Inv->getNormalDest();

  DomTreeNode *DomNode = DT.getNode(DomBB);

  SmallVector<PHINode *, 16> AddedPHIs;

  SSAUpdater SSAUpdate;
  SSAUpdate.Initialize(Inst.getType(), Inst.getName());

  // Insert the LCSSA phi's into all of the exit blocks dominated by the
  // value, and add them to the Phi's map.
  for (SmallVectorImpl<BasicBlock *>::const_iterator BBI = ExitBlocks.begin(),
                                                     BBE = ExitBlocks.end();
       BBI != BBE; ++BBI) {
    BasicBlock *ExitBB = *BBI;
    if (!DT.dominates(DomNode, DT.getNode(ExitBB)))
      continue;

    // If we already inserted something for this BB, don't reprocess it.
    if (SSAUpdate.HasValueForBlock(ExitBB))
      continue;

    PHINode *PN = PHINode::Create(Inst.getType(), PredCache.GetNumPreds(ExitBB),
                                  Inst.getName() + ".lcssa", ExitBB->begin());

    // Add inputs from inside the loop for this PHI.
    for (BasicBlock **PI = PredCache.GetPreds(ExitBB); *PI; ++PI) {
      PN->addIncoming(&Inst, *PI);

      // If the exit block has a predecessor not within the loop, arrange for
      // the incoming value use corresponding to that predecessor to be
      // rewritten in terms of a different LCSSA PHI.
      if (!L.contains(*PI))
        UsesToRewrite.push_back(
            &PN->getOperandUse(PN->getOperandNumForIncomingValue(
                 PN->getNumIncomingValues() - 1)));
    }

    AddedPHIs.push_back(PN);

    // Remember that this phi makes the value alive in this block.
    SSAUpdate.AddAvailableValue(ExitBB, PN);
  }

  // Rewrite all uses outside the loop in terms of the new PHIs we just
  // inserted.
  for (unsigned i = 0, e = UsesToRewrite.size(); i != e; ++i) {
    // If this use is in an exit block, rewrite to use the newly inserted PHI.
    // This is required for correctness because SSAUpdate doesn't handle uses in
    // the same block.  It assumes the PHI we inserted is at the end of the
    // block.
    Instruction *User = cast<Instruction>(UsesToRewrite[i]->getUser());
    BasicBlock *UserBB = User->getParent();
    if (PHINode *PN = dyn_cast<PHINode>(User))
      UserBB = PN->getIncomingBlock(*UsesToRewrite[i]);

    if (isa<PHINode>(UserBB->begin()) && isExitBlock(UserBB, ExitBlocks)) {
      // Tell the VHs that the uses changed. This updates SCEV's caches.
      if (UsesToRewrite[i]->get()->hasValueHandle())
        ValueHandleBase::ValueIsRAUWd(*UsesToRewrite[i], UserBB->begin());
      UsesToRewrite[i]->set(UserBB->begin());
      continue;
    }

    // Otherwise, do full PHI insertion.
    SSAUpdate.RewriteUse(*UsesToRewrite[i]);
  }

  // Remove PHI nodes that did not have any uses rewritten.
  for (unsigned i = 0, e = AddedPHIs.size(); i != e; ++i) {
    if (AddedPHIs[i]->use_empty())
      AddedPHIs[i]->eraseFromParent();
  }

  return true;
}

/// Return true if the specified block dominates at least
/// one of the blocks in the specified list.
static bool
blockDominatesAnExit(BasicBlock *BB,
                     DominatorTree &DT,
                     const SmallVectorImpl<BasicBlock *> &ExitBlocks) {
  DomTreeNode *DomNode = DT.getNode(BB);
  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i)
    if (DT.dominates(DomNode, DT.getNode(ExitBlocks[i])))
      return true;

  return false;
}

bool llvm::formLCSSA(Loop &L, DominatorTree &DT, ScalarEvolution *SE) {
  bool Changed = false;

  // Get the set of exiting blocks.
  SmallVector<BasicBlock *, 8> ExitBlocks;
  L.getExitBlocks(ExitBlocks);

  if (ExitBlocks.empty())
    return false;

  PredIteratorCache PredCache;

  // Look at all the instructions in the loop, checking to see if they have uses
  // outside the loop.  If so, rewrite those uses.
  for (Loop::block_iterator BBI = L.block_begin(), BBE = L.block_end();
       BBI != BBE; ++BBI) {
    BasicBlock *BB = *BBI;

    // For large loops, avoid use-scanning by using dominance information:  In
    // particular, if a block does not dominate any of the loop exits, then none
    // of the values defined in the block could be used outside the loop.
    if (!blockDominatesAnExit(BB, DT, ExitBlocks))
      continue;

    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
      // Reject two common cases fast: instructions with no uses (like stores)
      // and instructions with one use that is in the same block as this.
      if (I->use_empty() ||
          (I->hasOneUse() && I->use_back()->getParent() == BB &&
           !isa<PHINode>(I->use_back())))
        continue;

      Changed |= processInstruction(L, *I, DT, ExitBlocks, PredCache);
    }
  }

  // If we modified the code, remove any caches about the loop from SCEV to
  // avoid dangling entries.
  // FIXME: This is a big hammer, can we clear the cache more selectively?
  if (SE && Changed)
    SE->forgetLoop(&L);

  assert(L.isLCSSAForm(DT));

  return Changed;
}

namespace {
struct LCSSA : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  LCSSA() : FunctionPass(ID) {
    initializeLCSSAPass(*PassRegistry::getPassRegistry());
  }

  // Cached analysis information for the current function.
  DominatorTree *DT;
  LoopInfo *LI;
  ScalarEvolution *SE;

  virtual bool runOnFunction(Function &F);

  /// This transformation requires natural loop information & requires that
  /// loop preheaders be inserted into the CFG.  It maintains both of these,
  /// as well as the CFG.  It also requires dominator information.
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesCFG();

    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfo>();
    AU.addPreservedID(LoopSimplifyID);
    AU.addPreserved<ScalarEvolution>();
  }

private:
  bool processLoop(Loop &L);

  virtual void verifyAnalysis() const;
};
}

char LCSSA::ID = 0;
INITIALIZE_PASS_BEGIN(LCSSA, "lcssa", "Loop-Closed SSA Form Pass", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_END(LCSSA, "lcssa", "Loop-Closed SSA Form Pass", false, false)

Pass *llvm::createLCSSAPass() { return new LCSSA(); }
char &llvm::LCSSAID = LCSSA::ID;


/// Process all loops in the function, inner-most out.
bool LCSSA::runOnFunction(Function &F) {
  bool Changed = false;
  LI = &getAnalysis<LoopInfo>();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  SE = getAnalysisIfAvailable<ScalarEvolution>();

  // Simplify each loop nest in the function.
  for (LoopInfo::iterator I = LI->begin(), E = LI->end(); I != E; ++I)
    Changed |= processLoop(**I);

  return Changed;
}

/// Process a loop nest depth first.
bool LCSSA::processLoop(Loop &L) {
  bool Changed = false;

  // Recurse depth-first through inner loops.
  for (Loop::iterator LI = L.begin(), LE = L.end(); LI != LE; ++LI)
    Changed |= processLoop(**LI);

  Changed |= formLCSSA(L, *DT, SE);
  return Changed;
}

static void verifyLoop(Loop &L, DominatorTree &DT) {
  // Recurse depth-first through inner loops.
  for (Loop::iterator LI = L.begin(), LE = L.end(); LI != LE; ++LI)
    verifyLoop(**LI, DT);

  // Check the special guarantees that LCSSA makes.
  //assert(L.isLCSSAForm(DT) && "LCSSA form not preserved!");
}

void LCSSA::verifyAnalysis() const {
  // Verify each loop nest in the function, assuming LI still points at that
  // function's loop info.
  for (LoopInfo::iterator I = LI->begin(), E = LI->end(); I != E; ++I)
    verifyLoop(**I, *DT);
}
