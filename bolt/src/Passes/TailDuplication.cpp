//===--------- Passes/TailDuplication.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "TailDuplication.h"

#include <numeric>

using namespace llvm;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

static cl::opt<bool> TailDuplicationAggressive(
    "tail-duplication-aggressive",
    cl::desc("tail duplication should act aggressively in duplicating multiple "
             "blocks per tail"),
    cl::ZeroOrMore, cl::ReallyHidden, cl::init(false),
    cl::cat(BoltOptCategory));

static cl::opt<unsigned>
    TailDuplicationMinimumOffset("tail-duplication-minimum-offset",
                                 cl::desc("minimum offset needed between block "
                                          "and successor to allow duplication"),
                                 cl::ZeroOrMore, cl::ReallyHidden, cl::init(64),
                                 cl::cat(BoltOptCategory));

static cl::opt<unsigned> TailDuplicationMaximumDuplication(
    "tail-duplication-maximum-duplication",
    cl::desc("maximum size of duplicated blocks (in bytes)"), cl::ZeroOrMore,
    cl::ReallyHidden, cl::init(64), cl::cat(BoltOptCategory));

} // namespace opts

namespace llvm {
namespace bolt {
bool TailDuplication::isInCacheLine(const BinaryBasicBlock &BB,
                                    const BinaryBasicBlock &Succ) const {
  if (&BB == &Succ)
    return true;

  BinaryFunction::BasicBlockOrderType BlockLayout =
      BB.getFunction()->getLayout();
  uint64_t Distance = 0;
  int Direction = (Succ.getLayoutIndex() > BB.getLayoutIndex()) ? 1 : -1;

  for (unsigned I = BB.getLayoutIndex() + Direction; I != Succ.getLayoutIndex();
       I += Direction) {
    Distance += BlockLayout[I]->getOriginalSize();
    if (Distance > opts::TailDuplicationMinimumOffset)
      return false;
  }
  return true;
}

std::vector<BinaryBasicBlock *>
TailDuplication::moderateCodeToDuplicate(BinaryBasicBlock &BB) const {
  std::vector<BinaryBasicBlock *> BlocksToDuplicate;
  if (BB.hasJumpTable())
    return BlocksToDuplicate;
  if (BB.getOriginalSize() > opts::TailDuplicationMaximumDuplication)
    return BlocksToDuplicate;
  for (auto Itr = BB.succ_begin(); Itr != BB.succ_end(); ++Itr) {
    if ((*Itr)->getLayoutIndex() == BB.getLayoutIndex() + 1)
      // If duplicating would introduce a new branch, don't duplicate
      return BlocksToDuplicate;
  }
  BlocksToDuplicate.push_back(&BB);
  return BlocksToDuplicate;
}

std::vector<BinaryBasicBlock *>
TailDuplication::aggressiveCodeToDuplicate(BinaryBasicBlock &BB) const {
  std::vector<BinaryBasicBlock *> BlocksToDuplicate;
  BinaryBasicBlock *CurrBB = &BB;
  while (CurrBB) {
    BlocksToDuplicate.push_back(CurrBB);

    if (BB.hasJumpTable()) {
      BlocksToDuplicate.clear();
      break;
    }

    // With no successors, we've reached the end and should duplicate all of
    // BlocksToDuplicate
    if (CurrBB->succ_size() == 0)
      break;

    // With two successors, if they're both a jump, we should duplicate all
    // blocks in BlocksToDuplicate. Otherwise, we cannot find a simple stream of
    // blocks to copy
    if (CurrBB->succ_size() >= 2) {
      if (CurrBB->getConditionalSuccessor(false)->getLayoutIndex() ==
              CurrBB->getLayoutIndex() + 1 ||
          CurrBB->getConditionalSuccessor(true)->getLayoutIndex() ==
              CurrBB->getLayoutIndex() + 1)
        BlocksToDuplicate.clear();
      break;
    }

    // With one successor, if its a jump, we should duplicate all blocks in
    // BlocksToDuplicate. Otherwise, we should keep going
    BinaryBasicBlock *Succ = CurrBB->getSuccessor();
    if (Succ->getLayoutIndex() != CurrBB->getLayoutIndex() + 1)
      break;
    CurrBB = Succ;
  }
  // Don't duplicate if its too much code
  unsigned DuplicationByteCount = std::accumulate(
      std::begin(BlocksToDuplicate), std::end(BlocksToDuplicate), 0,
      [](int value, BinaryBasicBlock *p) {
        return value + p->getOriginalSize();
      });
  if (DuplicationByteCount > opts::TailDuplicationMaximumDuplication)
    BlocksToDuplicate.clear();
  return BlocksToDuplicate;
}

void TailDuplication::tailDuplicate(
    BinaryBasicBlock &BB,
    const std::vector<BinaryBasicBlock *> &BlocksToDuplicate) const {
  BinaryFunction *BF = BB.getFunction();
  BinaryContext &BC = BF->getBinaryContext();

  // Ratio of this new branches execution count to the total size of the
  // successor's execution count.  Used to set this new branches execution count
  // and lower the old successor's execution count
  double ExecutionCountRatio =
      BB.getExecutionCount() > BB.getSuccessor()->getExecutionCount()
          ? 1.0
          : (double)BB.getExecutionCount() /
                BB.getSuccessor()->getExecutionCount();

  // Use the last branch info when adding a successor to LastBB
  BinaryBasicBlock::BinaryBranchInfo &LastBI =
      BB.getBranchInfo(*(BB.getSuccessor()));

  BinaryBasicBlock *LastOriginalBB = &BB;
  BinaryBasicBlock *LastDuplicatedBB = &BB;
  assert(LastDuplicatedBB->succ_size() == 1 &&
         "tail duplication cannot act on a block with more than 1 successor");
  LastDuplicatedBB->removeSuccessor(LastDuplicatedBB->getSuccessor());

  std::vector<std::unique_ptr<BinaryBasicBlock>> DuplicatedBlocks;

  for (BinaryBasicBlock *CurrBB : BlocksToDuplicate) {
    DuplicatedBlocks.emplace_back(
        BF->createBasicBlock(0, (BC.Ctx)->createNamedTempSymbol("tail-dup")));
    BinaryBasicBlock *NewBB = DuplicatedBlocks.back().get();

    NewBB->addInstructions(CurrBB->begin(), CurrBB->end());
    // Set execution count as if it was just a copy of the original
    NewBB->setExecutionCount(
        std::max((uint64_t)1, CurrBB->getExecutionCount()));
    LastDuplicatedBB->addSuccessor(NewBB, LastBI);

    // As long as its not the first block, adjust both original and duplicated
    // to what they should be
    if (LastDuplicatedBB != &BB) {
      LastOriginalBB->adjustExecutionCount(1.0 - ExecutionCountRatio);
      LastDuplicatedBB->adjustExecutionCount(ExecutionCountRatio);
    }

    if (CurrBB->succ_size() == 1)
      LastBI = CurrBB->getBranchInfo(*(CurrBB->getSuccessor()));

    LastOriginalBB = CurrBB;
    LastDuplicatedBB = NewBB;
  }

  LastDuplicatedBB->addSuccessors(
      LastOriginalBB->succ_begin(), LastOriginalBB->succ_end(),
      LastOriginalBB->branch_info_begin(), LastOriginalBB->branch_info_end());

  LastOriginalBB->adjustExecutionCount(1.0 - ExecutionCountRatio);
  LastDuplicatedBB->adjustExecutionCount(ExecutionCountRatio);

  BF->insertBasicBlocks(&BB, std::move(DuplicatedBlocks));
}

void TailDuplication::runOnFunction(BinaryFunction &Function) {
  // New blocks will be added and layout will change,
  // so make a copy here to iterate over the original layout
  BinaryFunction::BasicBlockOrderType BlockLayout = Function.getLayout();
  for (BinaryBasicBlock *BB : BlockLayout) {
    if (BB->succ_size() == 1 &&
        BB->getSuccessor()->getLayoutIndex() != BB->getLayoutIndex() + 1)
      UnconditionalBranchDynamicCount += BB->getExecutionCount();
    if (BB->succ_size() == 2 &&
        BB->getFallthrough()->getLayoutIndex() != BB->getLayoutIndex() + 1)
      UnconditionalBranchDynamicCount += BB->getFallthroughBranchInfo().Count;
    AllBlocksDynamicCount += BB->getExecutionCount();

    // The block must be hot
    if (BB->getExecutionCount() == 0)
      continue;
    // with one successor
    if (BB->succ_size() != 1)
      continue;

    // no jump table
    if (BB->hasJumpTable())
      continue;

    // and we are estimating that this sucessor is not already in the same cache
    // line
    BinaryBasicBlock *Succ = BB->getSuccessor();
    if (isInCacheLine(*BB, *Succ))
      continue;
    std::vector<BinaryBasicBlock *> BlocksToDuplicate;
    if (opts::TailDuplicationAggressive)
      BlocksToDuplicate = aggressiveCodeToDuplicate(*Succ);
    else
      BlocksToDuplicate = moderateCodeToDuplicate(*Succ);
    if (BlocksToDuplicate.size() > 0) {
      PossibleDuplications++;
      PossibleDuplicationsDynamicCount += BB->getExecutionCount();
      tailDuplicate(*BB, BlocksToDuplicate);
    }
  }
}

void TailDuplication::runOnFunctions(BinaryContext &BC) {
  for (auto &It : BC.getBinaryFunctions()) {
    BinaryFunction &Function = It.second;
    runOnFunction(Function);
  }

  outs() << "BOLT-INFO: tail duplication possible duplications: "
         << PossibleDuplications << "\n";
  outs() << "BOLT-INFO: tail duplication possible dynamic reductions: "
         << PossibleDuplicationsDynamicCount << "\n";
  outs() << "BOLT-INFO: tail duplication possible dynamic reductions to "
            "unconditional branch execution : "
         << format("%.1f", ((float)PossibleDuplicationsDynamicCount * 100.0f) /
                               UnconditionalBranchDynamicCount)
         << "%\n";
  outs() << "BOLT-INFO: tail duplication possible dynamic reductions to all "
            "blocks execution : "
         << format("%.1f", ((float)PossibleDuplicationsDynamicCount * 100.0f) /
                               AllBlocksDynamicCount)
         << "%\n";
}

} // end namespace bolt
} // end namespace llvm
