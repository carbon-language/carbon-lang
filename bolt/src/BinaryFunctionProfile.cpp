//===--- BinaryFunctionProfile.cpp                                 --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//


#include "BinaryBasicBlock.h"
#include "BinaryFunction.h"
#include "DataReader.h"
#include "Passes/MCF.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt-prof"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

extern cl::opt<IndirectCallPromotionType> IndirectCallPromotion;
extern cl::opt<JumpTableSupportLevel> JumpTables;

static cl::opt<MCFCostFunction>
DoMCF("mcf",
  cl::desc("solve a min cost flow problem on the CFG to fix edge counts "
           "(default=disable)"),
  cl::init(MCF_DISABLE),
  cl::values(
    clEnumValN(MCF_DISABLE, "none",
               "disable MCF"),
    clEnumValN(MCF_LINEAR, "linear",
               "cost function is inversely proportional to edge count"),
    clEnumValN(MCF_QUADRATIC, "quadratic",
               "cost function is inversely proportional to edge count squared"),
    clEnumValN(MCF_LOG, "log",
               "cost function is inversely proportional to log of edge count"),
    clEnumValN(MCF_BLAMEFTS, "blamefts",
               "tune cost to blame fall-through edges for surplus flow")),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
FixFuncCounts("fix-func-counts",
  cl::desc("adjust function counts based on basic blocks execution count"),
  cl::init(false),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
FixBlockCounts("fix-block-counts",
  cl::desc("adjust block counts based on outgoing branch counts"),
  cl::init(true),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
InferFallThroughs("infer-fall-throughs",
  cl::desc("infer execution count for fall-through blocks"),
  cl::init(false),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

} // namespace opts

namespace llvm {
namespace bolt {

void BinaryFunction::postProcessProfile() {
  if (!hasValidProfile()) {
    clearProfile();
    return;
  }

  if (!(getProfileFlags() & PF_LBR)) {
    // Check if MCF post-processing was requested.
    if (opts::DoMCF != MCF_DISABLE) {
      removeTagsFromProfile();
      solveMCF(*this, opts::DoMCF);
    }
    return;
  }

  // If we have at least some branch data for the function indicate that it
  // was executed.
  if (opts::FixFuncCounts && ExecutionCount == 0) {
    ExecutionCount = 1;
  }

  // Compute preliminary execution count for each basic block.
  for (auto *BB : BasicBlocks) {
    if ((!BB->isEntryPoint() && !BB->isLandingPad()) ||
        BB->ExecutionCount == BinaryBasicBlock::COUNT_NO_PROFILE)
      BB->ExecutionCount = 0;
  }
  for (auto *BB : BasicBlocks) {
    auto SuccBIIter = BB->branch_info_begin();
    for (auto Succ : BB->successors()) {
      // All incoming edges to the primary entry have been accounted for, thus
      // we skip the update here.
      if (SuccBIIter->Count != BinaryBasicBlock::COUNT_NO_PROFILE &&
          Succ != BasicBlocks.front())
        Succ->setExecutionCount(Succ->getExecutionCount() + SuccBIIter->Count);
      ++SuccBIIter;
    }
  }

  if (opts::FixBlockCounts) {
    for (auto *BB : BasicBlocks) {
      // Make sure that execution count of a block is at least the branch count
      // of an incoming/outgoing jump.
      auto SuccBIIter = BB->branch_info_begin();
      for (auto Succ : BB->successors()) {
        auto Count = SuccBIIter->Count;
        if (Count != BinaryBasicBlock::COUNT_NO_PROFILE && Count > 0) {
          Succ->setExecutionCount(std::max(Succ->getExecutionCount(), Count));
          BB->setExecutionCount(std::max(BB->getExecutionCount(), Count));
        }
        ++SuccBIIter;
      }
      // Make sure that execution count of a block is at least the number of
      // function calls from the block.
      for (auto &Inst : *BB) {
        // Ignore non-call instruction
        if (!BC.MIA->isCall(Inst))
          continue;

        auto CountAnnt = BC.MIB->tryGetAnnotationAs<uint64_t>(Inst, "Count");
        if (CountAnnt) {
          BB->setExecutionCount(std::max(BB->getExecutionCount(), *CountAnnt));
        }
      }
    }
  }

  if (opts::InferFallThroughs)
    inferFallThroughCounts();

  // Check if MCF post-processing was requested.
  if (opts::DoMCF != MCF_DISABLE) {
    removeTagsFromProfile();
    solveMCF(*this, opts::DoMCF);
  }

  // Update profile information for jump tables based on CFG branch data.
  for (auto *BB : BasicBlocks) {
    const auto *LastInstr = BB->getLastNonPseudoInstr();
    if (!LastInstr)
      continue;
    const auto JTAddress = BC.MIB->getJumpTable(*LastInstr);
    if (!JTAddress)
      continue;
    auto *JT = getJumpTableContainingAddress(JTAddress);
    if (!JT)
      continue;

    uint64_t TotalBranchCount = 0;
    for (const auto &BranchInfo : BB->branch_info()) {
      TotalBranchCount += BranchInfo.Count;
    }
    JT->Count += TotalBranchCount;

    if (opts::IndirectCallPromotion < ICP_JUMP_TABLES &&
        opts::JumpTables < JTS_AGGRESSIVE)
      continue;

    if (JT->Counts.empty())
      JT->Counts.resize(JT->Entries.size());
    auto EI = JT->Entries.begin();
    auto Delta = (JTAddress - JT->getAddress()) / JT->EntrySize;
    EI += Delta;
    while (EI != JT->Entries.end()) {
      const auto *TargetBB = getBasicBlockForLabel(*EI);
      if (TargetBB) {
        const auto &BranchInfo = BB->getBranchInfo(*TargetBB);
        assert(Delta < JT->Counts.size());
        JT->Counts[Delta].Count += BranchInfo.Count;
        JT->Counts[Delta].Mispreds += BranchInfo.MispredictedCount;
      }
      ++Delta;
      ++EI;
      // A label marks the start of another jump table.
      if (JT->Labels.count(Delta * JT->EntrySize))
        break;
    }
  }
}

void BinaryFunction::mergeProfileDataInto(BinaryFunction &BF) const {
  // No reason to merge invalid or empty profiles into BF.
  if (!hasValidProfile())
    return;

  // Update function execution count.
  if (getExecutionCount() != BinaryFunction::COUNT_NO_PROFILE) {
    BF.setExecutionCount(BF.getKnownExecutionCount() + getExecutionCount());
  }

  // Since we are merging a valid profile, the new profile should be valid too.
  // It has either already been valid, or it has been cleaned up.
  BF.ProfileMatchRatio = 1.0f;

  // Update basic block and edge counts.
  auto BBMergeI = BF.begin();
  for (BinaryBasicBlock *BB : BasicBlocks) {
    BinaryBasicBlock *BBMerge = &*BBMergeI;
    assert(getIndex(BB) == BF.getIndex(BBMerge));

    // Update basic block count.
    if (BB->getExecutionCount() != BinaryBasicBlock::COUNT_NO_PROFILE) {
      BBMerge->setExecutionCount(
          BBMerge->getKnownExecutionCount() + BB->getExecutionCount());
    }

    // Update edge count for successors of this basic block.
    auto BBMergeSI = BBMerge->succ_begin();
    auto BIMergeI = BBMerge->branch_info_begin();
    auto BII = BB->branch_info_begin();
    for (const auto *BBSucc : BB->successors()) {
      (void)BBSucc;
      assert(getIndex(BBSucc) == BF.getIndex(*BBMergeSI));

      // At this point no branch count should be set to COUNT_NO_PROFILE.
      assert(BII->Count != BinaryBasicBlock::COUNT_NO_PROFILE &&
             "unexpected unknown branch profile");
      assert(BIMergeI->Count != BinaryBasicBlock::COUNT_NO_PROFILE &&
             "unexpected unknown branch profile");

      BIMergeI->Count += BII->Count;

      // When we merge inferred and real fall-through branch data, the merged
      // data is considered inferred.
      if (BII->MispredictedCount != BinaryBasicBlock::COUNT_INFERRED &&
          BIMergeI->MispredictedCount != BinaryBasicBlock::COUNT_INFERRED) {
        BIMergeI->MispredictedCount += BII->MispredictedCount;
      } else {
        BIMergeI->MispredictedCount = BinaryBasicBlock::COUNT_INFERRED;
      }

      ++BBMergeSI;
      ++BII;
      ++BIMergeI;
    }
    assert(BBMergeSI == BBMerge->succ_end());

    ++BBMergeI;
  }
  assert(BBMergeI == BF.end());

  // Merge jump tables profile info.
  auto JTMergeI = BF.JumpTables.begin();
  for (const auto &JTEntry : JumpTables) {
    if (JTMergeI->second->Counts.empty())
      JTMergeI->second->Counts.resize(JTEntry.second->Counts.size());
    auto CountMergeI = JTMergeI->second->Counts.begin();
    for (const auto &JI : JTEntry.second->Counts) {
      CountMergeI->Count += JI.Count;
      CountMergeI->Mispreds += JI.Mispreds;
      ++CountMergeI;
    }
    assert(CountMergeI == JTMergeI->second->Counts.end());

    ++JTMergeI;
  }
  assert(JTMergeI == BF.JumpTables.end());
}

void BinaryFunction::inferFallThroughCounts() {
  // Work on a basic block at a time, propagating frequency information
  // forwards.
  // It is important to walk in the layout order.
  for (auto *BB : BasicBlocks) {
    const uint64_t BBExecCount = BB->getExecutionCount();

    // Propagate this information to successors, filling in fall-through edges
    // with frequency information
    if (BB->succ_size() == 0)
      continue;

    // Calculate frequency of outgoing branches from this node according to
    // LBR data.
    uint64_t ReportedBranches = 0;
    for (const auto &SuccBI : BB->branch_info()) {
      if (SuccBI.Count != BinaryBasicBlock::COUNT_NO_PROFILE)
        ReportedBranches += SuccBI.Count;
    }

    // Get taken count of conditional tail call if the block ends with one.
    uint64_t CTCTakenCount = 0;
    const auto CTCInstr = BB->getLastNonPseudoInstr();
    if (CTCInstr && BC.MIB->getConditionalTailCall(*CTCInstr)) {
      CTCTakenCount =
        BC.MIB->getAnnotationWithDefault<uint64_t>(*CTCInstr, "CTCTakenCount");
    }

    // Calculate frequency of throws from this node according to LBR data
    // for branching into associated landing pads. Since it is possible
    // for a landing pad to be associated with more than one basic blocks,
    // we may overestimate the frequency of throws for such blocks.
    uint64_t ReportedThrows = 0;
    for (const auto *LP: BB->landing_pads()) {
      ReportedThrows += LP->getExecutionCount();
    }

    const uint64_t TotalReportedJumps =
      ReportedBranches + CTCTakenCount + ReportedThrows;

    // Infer the frequency of the fall-through edge, representing not taking the
    // branch.
    uint64_t Inferred = 0;
    if (BBExecCount > TotalReportedJumps)
      Inferred = BBExecCount - TotalReportedJumps;

    DEBUG(
      if (BBExecCount < TotalReportedJumps)
        dbgs()
            << "Fall-through inference is slightly inconsistent. "
               "exec frequency is less than the outgoing edges frequency ("
            << BBExecCount << " < " << ReportedBranches
            << ") for  BB at offset 0x"
            << Twine::utohexstr(getAddress() + BB->getOffset()) << '\n';
    );

    if (BB->succ_size() <= 2) {
      // Skip if the last instruction is an unconditional jump.
      const auto *LastInstr = BB->getLastNonPseudoInstr();
      if (LastInstr &&
          (BC.MIB->isUnconditionalBranch(*LastInstr) ||
           BC.MIB->isIndirectBranch(*LastInstr)))
        continue;
      // If there is an FT it will be the last successor.
      auto &SuccBI = *BB->branch_info_rbegin();
      auto &Succ = *BB->succ_rbegin();
      if (SuccBI.Count == 0) {
        SuccBI.Count = Inferred;
        SuccBI.MispredictedCount = BinaryBasicBlock::COUNT_INFERRED;
        Succ->ExecutionCount += Inferred;
      }
    }
  }

  return;
}

void BinaryFunction::clearProfile() {
  // Keep function execution profile the same. Only clear basic block and edge
  // counts.
  for (auto *BB : BasicBlocks) {
    BB->ExecutionCount = 0;
    for (auto &BI : BB->branch_info()) {
      BI.Count = 0;
      BI.MispredictedCount = 0;
    }
  }
}

} // namespace bolt
} // namespace llvm
