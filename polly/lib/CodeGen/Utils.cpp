//===--- Utils.cpp - Utility functions for the code generation --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains utility functions for the code generation.
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/Utils.h"
#include "polly/CodeGen/IRBuilder.h"
#include "polly/ScopInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

// Alternative to llvm::SplitCriticalEdge.
//
// Creates a new block which branches to Succ. The edge to split is redirected
// to the new block.
//
// The issue with llvm::SplitCriticalEdge is that it does nothing if the edge is
// not critical.
// The issue with llvm::SplitEdge is that it does not always create the middle
// block, but reuses Prev/Succ if it can. We always want a new middle block.
static BasicBlock *splitEdge(BasicBlock *Prev, BasicBlock *Succ,
                             const char *Suffix, DominatorTree *DT,
                             LoopInfo *LI, RegionInfo *RI) {
  assert(Prev && Succ);

  // Before:
  //   \    /     /   //
  //    Prev     /    //
  //     |  \___/     //
  //     |   ___      //
  //     |  /   \     //
  //    Succ     \    //
  //   /    \     \   //

  // The algorithm to update DominatorTree and LoopInfo of
  // llvm::SplitCriticalEdge is more efficient than
  // llvm::SplitBlockPredecessors, which is more general. In the future we might
  // either modify llvm::SplitCriticalEdge to allow skipping the critical edge
  // check; or Copy&Pase it here.
  BasicBlock *MiddleBlock = SplitBlockPredecessors(
      Succ, ArrayRef<BasicBlock *>(Prev), Suffix, DT, LI);

  if (RI) {
    Region *PrevRegion = RI->getRegionFor(Prev);
    Region *SuccRegion = RI->getRegionFor(Succ);
    if (PrevRegion->contains(MiddleBlock)) {
      RI->setRegionFor(MiddleBlock, PrevRegion);
    } else {
      RI->setRegionFor(MiddleBlock, SuccRegion);
    }
  }

  // After:
  //   \    /     /   //
  //    Prev     /    //
  //     |  \___/     //
  //     |            //
  // MiddleBlock      //
  //     |   ___      //
  //     |  /   \     //
  //    Succ     \    //
  //   /    \     \   //

  return MiddleBlock;
}

BasicBlock *polly::executeScopConditionally(Scop &S, Pass *P, Value *RTC) {
  Region &R = S.getRegion();
  PollyIRBuilder Builder(R.getEntry());
  DominatorTree &DT = P->getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  RegionInfo &RI = P->getAnalysis<RegionInfoPass>().getRegionInfo();
  LoopInfo &LI = P->getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

  // Before:
  //
  //      \   /      //
  //    EnteringBB   //
  //   _____|_____   //
  //  /  EntryBB  \  //
  //  |  (region) |  //
  //  \_ExitingBB_/  //
  //        |        //
  //      ExitBB     //
  //      /    \     //

  // Create a fork block.
  BasicBlock *EnteringBB = R.getEnteringBlock();
  BasicBlock *EntryBB = R.getEntry();
  assert(EnteringBB && "Must be a simple region");
  BasicBlock *SplitBlock =
      splitEdge(EnteringBB, EntryBB, ".split_new_and_old", &DT, &LI, &RI);
  SplitBlock->setName("polly.split_new_and_old");

  // Create a join block
  BasicBlock *ExitingBB = R.getExitingBlock();
  BasicBlock *ExitBB = R.getExit();
  assert(ExitingBB && "Must be a simple region");
  BasicBlock *MergeBlock =
      splitEdge(ExitingBB, ExitBB, ".merge_new_and_old", &DT, &LI, &RI);
  MergeBlock->setName("polly.merge_new_and_old");

  // Exclude the join block from the region.
  R.replaceExitRecursive(MergeBlock);
  RI.setRegionFor(MergeBlock, R.getParent());

  //      \   /      //
  //    EnteringBB   //
  //        |        //
  //    SplitBlock   //
  //   _____|_____   //
  //  /  EntryBB  \  //
  //  |  (region) |  //
  //  \_ExitingBB_/  //
  //        |        //
  //    MergeBlock   //
  //        |        //
  //      ExitBB     //
  //      /    \     //

  // Create the start block.
  Function *F = SplitBlock->getParent();
  BasicBlock *StartBlock =
      BasicBlock::Create(F->getContext(), "polly.start", F);
  SplitBlock->getTerminator()->eraseFromParent();
  Builder.SetInsertPoint(SplitBlock);
  Builder.CreateCondBr(RTC, StartBlock, R.getEntry());
  if (Loop *L = LI.getLoopFor(SplitBlock))
    L->addBasicBlockToLoop(StartBlock, LI);
  DT.addNewBlock(StartBlock, SplitBlock);
  RI.setRegionFor(StartBlock, RI.getRegionFor(SplitBlock));

  //      \   /                    //
  //    EnteringBB                 //
  //        |                      //
  //    SplitBlock---------\       //
  //   _____|_____         |       //
  //  /  EntryBB  \    StartBlock  //
  //  |  (region) |                //
  //  \_ExitingBB_/                //
  //        |                      //
  //    MergeBlock                 //
  //        |                      //
  //      ExitBB                   //
  //      /    \                   //

  // Connect start block to the join block.
  Builder.SetInsertPoint(StartBlock);
  Builder.CreateBr(MergeBlock);
  DT.changeImmediateDominator(MergeBlock, SplitBlock);

  //      \   /                    //
  //    EnteringBB                 //
  //        |                      //
  //    SplitBlock---------\       //
  //   _____|_____         |       //
  //  /  EntryBB  \    StartBlock  //
  //  |  (region) |        |       //
  //  \_ExitingBB_/        |       //
  //        |              |       //
  //    MergeBlock---------/       //
  //        |                      //
  //      ExitBB                   //
  //      /    \                   //

  return StartBlock;
}
