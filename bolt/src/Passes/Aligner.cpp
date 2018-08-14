//===--- Aligner.cpp ------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "Aligner.h"

#define DEBUG_TYPE "bolt-aligner"

using namespace llvm;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

extern cl::opt<bool> AlignBlocks;
extern cl::opt<bool> PreserveBlocksAlignment;

cl::opt<unsigned>
AlignBlocksMinSize("align-blocks-min-size",
  cl::desc("minimal size of the basic block that should be aligned"),
  cl::init(0),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

cl::opt<unsigned>
AlignBlocksThreshold("align-blocks-threshold",
  cl::desc("align only blocks with frequency larger than containing function "
           "execution frequency specified in percent. E.g. 1000 means aligning "
           "blocks that are 10 times more frequently executed than the "
           "containing function."),
  cl::init(800),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

cl::opt<unsigned>
AlignFunctions("align-functions",
  cl::desc("align functions at a given value (relocation mode)"),
  cl::init(64),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

cl::opt<unsigned>
AlignFunctionsMaxBytes("align-functions-max-bytes",
  cl::desc("maximum number of bytes to use to align functions"),
  cl::init(32),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

cl::opt<unsigned>
BlockAlignment("block-alignment",
  cl::desc("boundary to use for alignment of basic blocks"),
  cl::init(16),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

cl::opt<bool>
UseCompactAligner("use-compact-aligner",
  cl::desc("Use compact approach for aligning functions"),
  cl::init(true),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

} // end namespace opts

namespace llvm {
namespace bolt {

namespace {

// Align function to the specified byte-boundary (typically, 64) offsetting
// the fuction by not more than the corresponding value
void alignMaxBytes(BinaryFunction &Function) {
  Function.setAlignment(opts::AlignFunctions);
  Function.setMaxAlignmentBytes(opts::AlignFunctionsMaxBytes);
  Function.setMaxColdAlignmentBytes(opts::AlignFunctionsMaxBytes);
}

// Align function to the specified byte-boundary (typically, 64) offsetting
// the fuction by not more than the minimum over
// -- the size of the function
// -- the specified number of bytes
void alignCompact(BinaryFunction &Function) {
  const auto &BC = Function.getBinaryContext();
  size_t HotSize = 0;
  size_t ColdSize = 0;

  for (const auto *BB : Function.layout()) {
    if (BB->isCold())
      ColdSize += BC.computeCodeSize(BB->begin(), BB->end());
    else
      HotSize += BC.computeCodeSize(BB->begin(), BB->end());
  }

  Function.setAlignment(opts::AlignFunctions);
  if (HotSize > 0)
    Function.setMaxAlignmentBytes(
      std::min(size_t(opts::AlignFunctionsMaxBytes), HotSize));

  // using the same option, max-align-bytes, both for cold and hot parts of the
  // functions, as aligning cold functions typically does not affect performance
  if (ColdSize > 0)
    Function.setMaxColdAlignmentBytes(
      std::min(size_t(opts::AlignFunctionsMaxBytes), ColdSize));
}

} // end anonymous namespace

void AlignerPass::alignBlocks(BinaryFunction &Function) {
  if (!Function.hasValidProfile() || !Function.isSimple())
    return;

  const auto &BC = Function.getBinaryContext();

  const auto FuncCount = std::max(1UL, Function.getKnownExecutionCount());
  BinaryBasicBlock *PrevBB{nullptr};
  for (auto *BB : Function.layout()) {
    auto Count = BB->getKnownExecutionCount();

    if (Count <= FuncCount * opts::AlignBlocksThreshold / 100) {
      PrevBB = BB;
      continue;
    }

    uint64_t FTCount = 0;
    if (PrevBB && PrevBB->getFallthrough() == BB) {
      FTCount = PrevBB->getBranchInfo(*BB).Count;
    }
    PrevBB = BB;

    if (Count < FTCount * 2)
      continue;

    const auto BlockSize = BC.computeCodeSize(BB->begin(), BB->end());
    const auto BytesToUse = std::min(opts::BlockAlignment - 1UL, BlockSize);

    if (opts::AlignBlocksMinSize && BlockSize < opts::AlignBlocksMinSize)
      continue;

    BB->setAlignment(opts::BlockAlignment);
    BB->setAlignmentMaxBytes(BytesToUse);

    // Update stats.
    AlignHistogram[BytesToUse]++;
    AlignedBlocksCount += BB->getKnownExecutionCount();
  }
}

void AlignerPass::runOnFunctions(BinaryContext &BC,
                                 std::map<uint64_t, BinaryFunction> &BFs,
                                 std::set<uint64_t> &LargeFunctions) {
  if (!BC.HasRelocations)
    return;

  AlignHistogram.resize(opts::BlockAlignment);

  for (auto &It : BFs) {
    auto &Function = It.second;

    if (opts::UseCompactAligner)
      alignCompact(Function);
    else
      alignMaxBytes(Function);

    if (opts::AlignBlocks && !opts::PreserveBlocksAlignment)
      alignBlocks(Function);
  }

  DEBUG(
    dbgs() << "BOLT-DEBUG: max bytes per basic block alignment distribution:\n";
    for (unsigned I = 1; I < AlignHistogram.size(); ++I) {
      dbgs() << "  " << I << " : " << AlignHistogram[I] << '\n';
    }
    dbgs() << "BOLT-DEBUG: total execution count of aligned blocks: "
           << AlignedBlocksCount << '\n';
  );
}

} // end namespace bolt
} // end namespace llvm
