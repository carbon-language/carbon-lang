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
  for (auto Itr = BB.succ_begin(); Itr != BB.succ_end(); ++Itr) {
    if ((*Itr)->getLayoutIndex() == BB.getLayoutIndex() + 1) {
      // If duplicating would introduce a new branch, don't duplicate
      return BlocksToDuplicate;
    }
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
        return value + p->getOutputSize();
      });
  if (DuplicationByteCount < opts::TailDuplicationMaximumDuplication)
    BlocksToDuplicate.clear();
  return BlocksToDuplicate;
}

void TailDuplication::runOnFunction(BinaryFunction &Function) {
  for (BinaryBasicBlock *BB : Function.layout()) {
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
    // and that one successor is not a direct fallthrough
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
