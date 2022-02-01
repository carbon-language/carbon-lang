//===- BasicBlockUtils.cpp - Unit tests for BasicBlockUtils ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/BreakCriticalEdges.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("BasicBlockUtilsTests", errs());
  return Mod;
}

static BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
  for (BasicBlock &BB : F)
    if (BB.getName() == Name)
      return &BB;
  llvm_unreachable("Expected to find basic block!");
}

TEST(BasicBlockUtils, EliminateUnreachableBlocks) {
  LLVMContext C;

  std::unique_ptr<Module> M = parseIR(
    C,
    "define i32 @has_unreachable(i1 %cond) {\n"
    "entry:\n"
    "  br i1 %cond, label %bb0, label %bb1\n"
    "bb0:\n"
    "  br label %bb1\n"
    "bb1:\n"
    "  %phi = phi i32 [ 0, %entry ], [ 1, %bb0 ]"
    "  ret i32 %phi\n"
    "bb2:\n"
    "  ret i32 42\n"
    "}\n"
    "\n"
    );

  auto *F = M->getFunction("has_unreachable");
  DominatorTree DT(*F);
  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Eager);

  EXPECT_EQ(F->size(), (size_t)4);
  bool Result = EliminateUnreachableBlocks(*F, &DTU);
  EXPECT_TRUE(Result);
  EXPECT_EQ(F->size(), (size_t)3);
  EXPECT_TRUE(DT.verify());
}

TEST(BasicBlockUtils, SplitEdge_ex1) {
  LLVMContext C;
  std::unique_ptr<Module> M =
      parseIR(C, "define void @foo(i1 %cond0) {\n"
                 "entry:\n"
                 "  br i1 %cond0, label %bb0, label %bb1\n"
                 "bb0:\n"
                 " %0 = mul i32 1, 2\n"
                 "  br label %bb1\n"
                 "bb1:\n"
                 "  br label %bb2\n"
                 "bb2:\n"
                 "  ret void\n"
                 "}\n"
                 "\n");

  Function *F = M->getFunction("foo");
  DominatorTree DT(*F);
  BasicBlock *SrcBlock;
  BasicBlock *DestBlock;
  BasicBlock *NewBB;

  SrcBlock = getBasicBlockByName(*F, "entry");
  DestBlock = getBasicBlockByName(*F, "bb0");
  NewBB = SplitEdge(SrcBlock, DestBlock, &DT, nullptr, nullptr);

  EXPECT_TRUE(DT.verify());
  EXPECT_EQ(NewBB->getSinglePredecessor(), SrcBlock);
  EXPECT_EQ(NewBB->getSingleSuccessor(), DestBlock);
  EXPECT_EQ(NewBB->getParent(), F);

  bool BBFlag = false;
  for (BasicBlock &BB : *F) {
    if (BB.getName() == NewBB->getName()) {
      BBFlag = true;
    }
  }
  EXPECT_TRUE(BBFlag);
}

TEST(BasicBlockUtils, SplitEdge_ex2) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, "define void @foo() {\n"
                                         "bb0:\n"
                                         "  br label %bb2\n"
                                         "bb1:\n"
                                         "  br label %bb2\n"
                                         "bb2:\n"
                                         "  ret void\n"
                                         "}\n"
                                         "\n");

  Function *F = M->getFunction("foo");
  DominatorTree DT(*F);

  BasicBlock *SrcBlock;
  BasicBlock *DestBlock;
  BasicBlock *NewBB;

  SrcBlock = getBasicBlockByName(*F, "bb0");
  DestBlock = getBasicBlockByName(*F, "bb2");
  NewBB = SplitEdge(SrcBlock, DestBlock, &DT, nullptr, nullptr);

  EXPECT_TRUE(DT.verify());
  EXPECT_EQ(NewBB->getSinglePredecessor(), SrcBlock);
  EXPECT_EQ(NewBB->getSingleSuccessor(), DestBlock);
  EXPECT_EQ(NewBB->getParent(), F);

  bool BBFlag = false;
  for (BasicBlock &BB : *F) {
    if (BB.getName() == NewBB->getName()) {
      BBFlag = true;
    }
  }
  EXPECT_TRUE(BBFlag);
}

TEST(BasicBlockUtils, SplitEdge_ex3) {
  LLVMContext C;
  std::unique_ptr<Module> M =
      parseIR(C, "define i32 @foo(i32 %n) {\n"
                 "entry:\n"
                 " br label %header\n"
                 "header:\n"
                 " %sum.02 = phi i32 [ 0, %entry ], [ %sum.1, %bb3 ]\n"
                 " %0 = phi i32 [ 0, %entry ], [ %4, %bb3 ] \n"
                 " %1 = icmp slt i32 %0, %n \n"
                 " br i1 %1, label %bb0, label %bb1\n"
                 "bb0:\n"
                 "  %2 = add nsw i32 %sum.02, 2\n"
                 "  br label %bb2\n"
                 "bb1:\n"
                 "  %3 = add nsw i32 %sum.02, 1\n"
                 "  br label %bb2\n"
                 "bb2:\n"
                 "  %sum.1 = phi i32 [ %2, %bb0 ], [ %3, %bb1 ]\n"
                 "  br label %bb3\n"
                 "bb3:\n"
                 "  %4 = add nsw i32 %0, 1 \n"
                 "  %5 = icmp slt i32 %4, 100\n"
                 "  br i1 %5, label %header, label %bb4\n"
                 "bb4:\n"
                 " %sum.0.lcssa = phi i32 [ %sum.1, %bb3 ]\n"
                 " ret i32 %sum.0.lcssa\n"
                 "}\n"
                 "\n");

  Function *F = M->getFunction("foo");
  DominatorTree DT(*F);

  LoopInfo LI(DT);

  DataLayout DL("e-i64:64-f80:128-n8:16:32:64-S128");
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  AssumptionCache AC(*F);
  AAResults AA(TLI);

  BasicAAResult BAA(DL, *F, TLI, AC, &DT);
  AA.addAAResult(BAA);

  MemorySSA MSSA(*F, &AA, &DT);
  MemorySSAUpdater Updater(&MSSA);

  BasicBlock *SrcBlock;
  BasicBlock *DestBlock;
  BasicBlock *NewBB;

  SrcBlock = getBasicBlockByName(*F, "header");
  DestBlock = getBasicBlockByName(*F, "bb0");
  NewBB = SplitEdge(SrcBlock, DestBlock, &DT, &LI, &Updater);

  Updater.getMemorySSA()->verifyMemorySSA();
  EXPECT_TRUE(DT.verify());
  EXPECT_NE(LI.getLoopFor(SrcBlock), nullptr);
  EXPECT_NE(LI.getLoopFor(DestBlock), nullptr);
  EXPECT_NE(LI.getLoopFor(NewBB), nullptr);
  EXPECT_EQ(NewBB->getSinglePredecessor(), SrcBlock);
  EXPECT_EQ(NewBB->getSingleSuccessor(), DestBlock);
  EXPECT_EQ(NewBB->getParent(), F);

  bool BBFlag = false;
  for (BasicBlock &BB : *F) {
    if (BB.getName() == NewBB->getName()) {
      BBFlag = true;
    }
  }
  EXPECT_TRUE(BBFlag);
}

TEST(BasicBlockUtils, SplitEdge_ex4) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(
      C, "define void @bar(i32 %cond) personality i8 0 {\n"
         "entry:\n"
         "  switch i32 %cond, label %exit [\n"
         "    i32 -1, label %continue\n"
         "    i32 0, label %continue\n"
         "    i32 1, label %continue_alt\n"
         "    i32 2, label %continue_alt\n"
         "  ]\n"
         "exit:\n"
         "  ret void\n"
         "continue:\n"
         "  invoke void @sink() to label %normal unwind label %exception\n"
         "continue_alt:\n"
         "  invoke void @sink_alt() to label %normal unwind label %exception\n"
         "exception:\n"
         "  %cleanup = landingpad i8 cleanup\n"
         "  br label %trivial-eh-handler\n"
         "trivial-eh-handler:\n"
         "  call void @sideeffect(i32 1)\n"
         "  br label %normal\n"
         "normal:\n"
         "  call void @sideeffect(i32 0)\n"
         "  ret void\n"
         "}\n"
         "\n"
         "declare void @sideeffect(i32)\n"
         "declare void @sink() cold\n"
         "declare void @sink_alt() cold\n");

  Function *F = M->getFunction("bar");

  DominatorTree DT(*F);

  LoopInfo LI(DT);

  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);

  AAResults AA(TLI);

  MemorySSA MSSA(*F, &AA, &DT);
  MemorySSAUpdater MSSAU(&MSSA);

  BasicBlock *SrcBlock;
  BasicBlock *DestBlock;

  SrcBlock = getBasicBlockByName(*F, "continue");
  DestBlock = getBasicBlockByName(*F, "exception");

  unsigned SuccNum = GetSuccessorNumber(SrcBlock, DestBlock);
  Instruction *LatchTerm = SrcBlock->getTerminator();

  const CriticalEdgeSplittingOptions Options =
      CriticalEdgeSplittingOptions(&DT, &LI, &MSSAU);

  // Check that the following edge is both critical and the destination block is
  // an exception block. These must be handled differently by SplitEdge
  bool CriticalEdge =
      isCriticalEdge(LatchTerm, SuccNum, Options.MergeIdenticalEdges);
  EXPECT_TRUE(CriticalEdge);

  bool Ehpad = DestBlock->isEHPad();
  EXPECT_TRUE(Ehpad);

  BasicBlock *NewBB = SplitEdge(SrcBlock, DestBlock, &DT, &LI, &MSSAU, "");

  MSSA.verifyMemorySSA();
  EXPECT_TRUE(DT.verify());
  EXPECT_NE(NewBB, nullptr);
  EXPECT_EQ(NewBB->getSinglePredecessor(), SrcBlock);
  EXPECT_EQ(NewBB, SrcBlock->getTerminator()->getSuccessor(SuccNum));
  EXPECT_EQ(NewBB->getParent(), F);

  bool BBFlag = false;
  for (BasicBlock &BB : *F) {
    if (BB.getName() == NewBB->getName()) {
      BBFlag = true;
      break;
    }
  }
  EXPECT_TRUE(BBFlag);
}

TEST(BasicBlockUtils, splitBasicBlockBefore_ex1) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, "define void @foo() {\n"
                                         "bb0:\n"
                                         " %0 = mul i32 1, 2\n"
                                         "  br label %bb2\n"
                                         "bb1:\n"
                                         "  br label %bb3\n"
                                         "bb2:\n"
                                         "  %1 = phi  i32 [ %0, %bb0 ]\n"
                                         "  br label %bb3\n"
                                         "bb3:\n"
                                         "  ret void\n"
                                         "}\n"
                                         "\n");

  Function *F = M->getFunction("foo");
  DominatorTree DT(*F);

  BasicBlock *DestBlock;
  BasicBlock *NewBB;

  DestBlock = getBasicBlockByName(*F, "bb2");

  NewBB = DestBlock->splitBasicBlockBefore(DestBlock->front().getIterator(),
                                           "test");

  PHINode *PN = dyn_cast<PHINode>(&(DestBlock->front()));
  EXPECT_EQ(PN->getIncomingBlock(0), NewBB);
  EXPECT_EQ(NewBB->getName(), "test");
  EXPECT_EQ(NewBB->getSingleSuccessor(), DestBlock);
  EXPECT_EQ(DestBlock->getSinglePredecessor(), NewBB);
}

#ifndef NDEBUG
TEST(BasicBlockUtils, splitBasicBlockBefore_ex2) {
  LLVMContext C;
  std::unique_ptr<Module> M =
      parseIR(C, "define void @foo() {\n"
                 "bb0:\n"
                 " %0 = mul i32 1, 2\n"
                 "  br label %bb2\n"
                 "bb1:\n"
                 "  br label %bb2\n"
                 "bb2:\n"
                 "  %1 = phi  i32 [ %0, %bb0 ], [ 1, %bb1 ]\n"
                 "  br label %bb3\n"
                 "bb3:\n"
                 "  ret void\n"
                 "}\n"
                 "\n");

  Function *F = M->getFunction("foo");
  DominatorTree DT(*F);

  BasicBlock *DestBlock;

  DestBlock = getBasicBlockByName(*F, "bb2");

  ASSERT_DEATH(
      {
        DestBlock->splitBasicBlockBefore(DestBlock->front().getIterator(),
                                         "test");
      },
      "cannot split on multi incoming phis");
}
#endif

TEST(BasicBlockUtils, NoUnreachableBlocksToEliminate) {
  LLVMContext C;

  std::unique_ptr<Module> M = parseIR(
    C,
    "define i32 @no_unreachable(i1 %cond) {\n"
    "entry:\n"
    "  br i1 %cond, label %bb0, label %bb1\n"
    "bb0:\n"
    "  br label %bb1\n"
    "bb1:\n"
    "  %phi = phi i32 [ 0, %entry ], [ 1, %bb0 ]"
    "  ret i32 %phi\n"
    "}\n"
    "\n"
    );

  auto *F = M->getFunction("no_unreachable");
  DominatorTree DT(*F);
  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Eager);

  EXPECT_EQ(F->size(), (size_t)3);
  bool Result = EliminateUnreachableBlocks(*F, &DTU);
  EXPECT_FALSE(Result);
  EXPECT_EQ(F->size(), (size_t)3);
  EXPECT_TRUE(DT.verify());
}

TEST(BasicBlockUtils, SplitBlockPredecessors) {
  LLVMContext C;

  std::unique_ptr<Module> M = parseIR(
    C,
    "define i32 @basic_func(i1 %cond) {\n"
    "entry:\n"
    "  br i1 %cond, label %bb0, label %bb1\n"
    "bb0:\n"
    "  br label %bb1\n"
    "bb1:\n"
    "  %phi = phi i32 [ 0, %entry ], [ 1, %bb0 ]"
    "  ret i32 %phi\n"
    "}\n"
    "\n"
    );

  auto *F = M->getFunction("basic_func");
  DominatorTree DT(*F);

  // Make sure the dominator tree is properly updated if calling this on the
  // entry block.
  SplitBlockPredecessors(&F->getEntryBlock(), {}, "split.entry", &DT);
  EXPECT_TRUE(DT.verify());
}

TEST(BasicBlockUtils, SplitCriticalEdge) {
  LLVMContext C;

  std::unique_ptr<Module> M = parseIR(
    C,
    "define void @crit_edge(i1 %cond0, i1 %cond1) {\n"
    "entry:\n"
    "  br i1 %cond0, label %bb0, label %bb1\n"
    "bb0:\n"
    "  br label %bb1\n"
    "bb1:\n"
    "  br label %bb2\n"
    "bb2:\n"
    "  ret void\n"
    "}\n"
    "\n"
    );

  auto *F = M->getFunction("crit_edge");
  DominatorTree DT(*F);
  PostDominatorTree PDT(*F);

  CriticalEdgeSplittingOptions CESO(&DT, nullptr, nullptr, &PDT);
  EXPECT_EQ(1u, SplitAllCriticalEdges(*F, CESO));
  EXPECT_TRUE(DT.verify());
  EXPECT_TRUE(PDT.verify());
}

TEST(BasicBlockUtils, SplitIndirectBrCriticalEdge) {
  LLVMContext C;

  std::unique_ptr<Module> M =
      parseIR(C, "define void @crit_edge(i8* %cond0, i1 %cond1) {\n"
                 "entry:\n"
                 "  indirectbr i8* %cond0, [label %bb0, label %bb1]\n"
                 "bb0:\n"
                 "  br label %bb1\n"
                 "bb1:\n"
                 "  %p = phi i32 [0, %bb0], [0, %entry]\n"
                 "  br i1 %cond1, label %bb2, label %bb3\n"
                 "bb2:\n"
                 "  ret void\n"
                 "bb3:\n"
                 "  ret void\n"
                 "}\n");

  auto *F = M->getFunction("crit_edge");
  DominatorTree DT(*F);
  LoopInfo LI(DT);
  BranchProbabilityInfo BPI(*F, LI);
  BlockFrequencyInfo BFI(*F, BPI, LI);

  auto Block = [&F](StringRef BBName) -> const BasicBlock & {
    for (auto &BB : *F)
      if (BB.getName() == BBName)
        return BB;
    llvm_unreachable("Block not found");
  };

  bool Split = SplitIndirectBrCriticalEdges(*F, &BPI, &BFI);

  EXPECT_TRUE(Split);

  // Check that successors of the split block get their probability correct.
  BasicBlock *SplitBB = Block("bb1").getTerminator()->getSuccessor(0);
  EXPECT_EQ(2u, SplitBB->getTerminator()->getNumSuccessors());
  EXPECT_EQ(BranchProbability(1, 2), BPI.getEdgeProbability(SplitBB, 0u));
  EXPECT_EQ(BranchProbability(1, 2), BPI.getEdgeProbability(SplitBB, 1u));
}

TEST(BasicBlockUtils, SetEdgeProbability) {
  LLVMContext C;

  std::unique_ptr<Module> M = parseIR(
      C, "define void @edge_probability(i32 %0) {\n"
         "entry:\n"
         "switch i32 %0, label %LD [\n"
         "  i32 700, label %L0\n"
         "  i32 701, label %L1\n"
         "  i32 702, label %L2\n"
         "  i32 703, label %L3\n"
         "  i32 704, label %L4\n"
         "  i32 705, label %L5\n"
         "  i32 706, label %L6\n"
         "  i32 707, label %L7\n"
         "  i32 708, label %L8\n"
         "  i32 709, label %L9\n"
         "  i32 710, label %L10\n"
         "  i32 711, label %L11\n"
         "  i32 712, label %L12\n"
         "  i32 713, label %L13\n"
         "  i32 714, label %L14\n"
         "  i32 715, label %L15\n"
         "  i32 716, label %L16\n"
         "  i32 717, label %L17\n"
         "  i32 718, label %L18\n"
         "  i32 719, label %L19\n"
         "], !prof !{!\"branch_weights\", i32 1, i32 1, i32 1, i32 1, i32 1, "
         "i32 451, i32 1, i32 12, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, "
         "i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1}\n"
         "LD:\n"
         "  unreachable\n"
         "L0:\n"
         "  ret void\n"
         "L1:\n"
         "  ret void\n"
         "L2:\n"
         "  ret void\n"
         "L3:\n"
         "  ret void\n"
         "L4:\n"
         "  ret void\n"
         "L5:\n"
         "  ret void\n"
         "L6:\n"
         "  ret void\n"
         "L7:\n"
         "  ret void\n"
         "L8:\n"
         "  ret void\n"
         "L9:\n"
         "  ret void\n"
         "L10:\n"
         "  ret void\n"
         "L11:\n"
         "  ret void\n"
         "L12:\n"
         "  ret void\n"
         "L13:\n"
         "  ret void\n"
         "L14:\n"
         "  ret void\n"
         "L15:\n"
         "  ret void\n"
         "L16:\n"
         "  ret void\n"
         "L17:\n"
         "  ret void\n"
         "L18:\n"
         "  ret void\n"
         "L19:\n"
         "  ret void\n"
         "}\n");

  auto *F = M->getFunction("edge_probability");
  DominatorTree DT(*F);
  LoopInfo LI(DT);
  BranchProbabilityInfo BPI(*F, LI);

  auto Block = [&F](StringRef BBName) -> const BasicBlock & {
    for (auto &BB : *F)
      if (BB.getName() == BBName)
        return BB;
    llvm_unreachable("Block not found");
  };

  // Check that the unreachable block has the minimal probability.
  const BasicBlock &EntryBB = Block("entry");
  const BasicBlock &UnreachableBB = Block("LD");
  EXPECT_EQ(BranchProbability::getRaw(1),
            BPI.getEdgeProbability(&EntryBB, &UnreachableBB));
}
