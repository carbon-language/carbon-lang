//===- ScopHelper.cpp - Some Helper Functions for Scop.  ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Small functions that help with Scop and LLVM-IR.
//
//===----------------------------------------------------------------------===//

#include "polly/Support/ScopHelper.h"
#include "polly/ScopInfo.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "polly-scop-helper"

// Helper function for Scop
// TODO: Add assertion to not allow parameter to be null
//===----------------------------------------------------------------------===//
// Temporary Hack for extended region tree.
// Cast the region to loop if there is a loop have the same header and exit.
Loop *polly::castToLoop(const Region &R, LoopInfo &LI) {
  BasicBlock *entry = R.getEntry();

  if (!LI.isLoopHeader(entry))
    return 0;

  Loop *L = LI.getLoopFor(entry);

  BasicBlock *exit = L->getExitBlock();

  // Is the loop with multiple exits?
  if (!exit)
    return 0;

  if (exit != R.getExit()) {
    // SubRegion/ParentRegion with the same entry.
    assert((R.getNode(R.getEntry())->isSubRegion() ||
            R.getParent()->getEntry() == entry) &&
           "Expect the loop is the smaller or bigger region");
    return 0;
  }

  return L;
}

Value *polly::getPointerOperand(Instruction &Inst) {
  if (LoadInst *load = dyn_cast<LoadInst>(&Inst))
    return load->getPointerOperand();
  else if (StoreInst *store = dyn_cast<StoreInst>(&Inst))
    return store->getPointerOperand();
  else if (GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(&Inst))
    return gep->getPointerOperand();

  return 0;
}

bool polly::hasInvokeEdge(const PHINode *PN) {
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i < e; ++i)
    if (InvokeInst *II = dyn_cast<InvokeInst>(PN->getIncomingValue(i)))
      if (II->getParent() == PN->getIncomingBlock(i))
        return true;

  return false;
}

// Ensures that there is just one predecessor to the entry node from outside the
// region.
// The identity of the region entry node is preserved.
static void simplifyRegionEntry(Region *R, DominatorTree *DT, LoopInfo *LI,
                                RegionInfo *RI) {
  BasicBlock *EnteringBB = R->getEnteringBlock();
  BasicBlock *Entry = R->getEntry();

  // Before (one of):
  //
  //                       \    /            //
  //                      EnteringBB         //
  //                        |    \------>    //
  //   \   /                |                //
  //   Entry <--\         Entry <--\         //
  //   /   \    /         /   \    /         //
  //        ....               ....          //

  // Create single entry edge if the region has multiple entry edges.
  if (!EnteringBB) {
    SmallVector<BasicBlock *, 4> Preds;
    for (BasicBlock *P : predecessors(Entry))
      if (!R->contains(P))
        Preds.push_back(P);

    BasicBlock *NewEntering =
        SplitBlockPredecessors(Entry, Preds, ".region_entering", DT, LI);

    if (RI) {
      // The exit block of predecessing regions must be changed to NewEntering
      for (BasicBlock *ExitPred : predecessors(NewEntering)) {
        Region *RegionOfPred = RI->getRegionFor(ExitPred);
        if (RegionOfPred->getExit() != Entry)
          continue;

        while (!RegionOfPred->isTopLevelRegion() &&
               RegionOfPred->getExit() == Entry) {
          RegionOfPred->replaceExit(NewEntering);
          RegionOfPred = RegionOfPred->getParent();
        }
      }

      // Make all ancestors use EnteringBB as entry; there might be edges to it
      Region *AncestorR = R->getParent();
      RI->setRegionFor(NewEntering, AncestorR);
      while (!AncestorR->isTopLevelRegion() && AncestorR->getEntry() == Entry) {
        AncestorR->replaceEntry(NewEntering);
        AncestorR = AncestorR->getParent();
      }
    }

    EnteringBB = NewEntering;
  }
  assert(R->getEnteringBlock() == EnteringBB);

  // After:
  //
  //    \    /       //
  //  EnteringBB     //
  //      |          //
  //      |          //
  //    Entry <--\   //
  //    /   \    /   //
  //         ....    //
}

// Ensure that the region has a single block that branches to the exit node.
static void simplifyRegionExit(Region *R, DominatorTree *DT, LoopInfo *LI,
                               RegionInfo *RI) {
  BasicBlock *ExitBB = R->getExit();
  BasicBlock *ExitingBB = R->getExitingBlock();

  // Before:
  //
  //   (Region)   ______/  //
  //      \  |   /         //
  //       ExitBB          //
  //       /    \          //

  if (!ExitingBB) {
    SmallVector<BasicBlock *, 4> Preds;
    for (BasicBlock *P : predecessors(ExitBB))
      if (R->contains(P))
        Preds.push_back(P);

    //  Preds[0] Preds[1]      otherBB //
    //         \  |  ________/         //
    //          \ | /                  //
    //           BB                    //
    ExitingBB =
        SplitBlockPredecessors(ExitBB, Preds, ".region_exiting", DT, LI);
    // Preds[0] Preds[1]      otherBB  //
    //        \  /           /         //
    // BB.region_exiting    /          //
    //                  \  /           //
    //                   BB            //

    if (RI)
      RI->setRegionFor(ExitingBB, R);

    // Change the exit of nested regions, but not the region itself,
    R->replaceExitRecursive(ExitingBB);
    R->replaceExit(ExitBB);
  }
  assert(ExitingBB == R->getExitingBlock());

  // After:
  //
  //     \   /                //
  //    ExitingBB     _____/  //
  //          \      /        //
  //           ExitBB         //
  //           /    \         //
}

void polly::simplifyRegion(Region *R, DominatorTree *DT, LoopInfo *LI,
                           RegionInfo *RI) {
  assert(R && !R->isTopLevelRegion());
  assert(!RI || RI == R->getRegionInfo());
  assert((!RI || DT) &&
         "RegionInfo requires DominatorTree to be updated as well");

  simplifyRegionEntry(R, DT, LI, RI);
  simplifyRegionExit(R, DT, LI, RI);
  assert(R->isSimple());
}

// Split the block into two successive blocks.
//
// Like llvm::SplitBlock, but also preserves RegionInfo
static BasicBlock *splitBlock(BasicBlock *Old, Instruction *SplitPt,
                              DominatorTree *DT, llvm::LoopInfo *LI,
                              RegionInfo *RI) {
  assert(Old && SplitPt);

  // Before:
  //
  //  \   /  //
  //   Old   //
  //  /   \  //

  BasicBlock *NewBlock = llvm::SplitBlock(Old, SplitPt, DT, LI);

  if (RI) {
    Region *R = RI->getRegionFor(Old);
    RI->setRegionFor(NewBlock, R);
  }

  // After:
  //
  //   \   /    //
  //    Old     //
  //     |      //
  //  NewBlock  //
  //   /   \    //

  return NewBlock;
}

void polly::splitEntryBlockForAlloca(BasicBlock *EntryBlock, Pass *P) {
  // Find first non-alloca instruction. Every basic block has a non-alloc
  // instruction, as every well formed basic block has a terminator.
  BasicBlock::iterator I = EntryBlock->begin();
  while (isa<AllocaInst>(I))
    ++I;

  auto *DTWP = P->getAnalysisIfAvailable<DominatorTreeWrapperPass>();
  auto *DT = DTWP ? &DTWP->getDomTree() : nullptr;
  auto *LIWP = P->getAnalysisIfAvailable<LoopInfoWrapperPass>();
  auto *LI = LIWP ? &LIWP->getLoopInfo() : nullptr;
  RegionInfoPass *RIP = P->getAnalysisIfAvailable<RegionInfoPass>();
  RegionInfo *RI = RIP ? &RIP->getRegionInfo() : nullptr;

  // splitBlock updates DT, LI and RI.
  splitBlock(EntryBlock, I, DT, LI, RI);
}
