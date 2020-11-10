//===--- DynoStats.cpp ----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//


#include "DynoStats.h"
#include "BinaryBasicBlock.h"
#include "BinaryFunction.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <numeric>
#include <string>

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::OptionCategory BoltCategory;

static cl::opt<uint32_t>
DynoStatsScale("dyno-stats-scale",
  cl::desc("scale to be applied while reporting dyno stats"),
  cl::Optional,
  cl::init(1),
  cl::Hidden,
  cl::cat(BoltCategory));

} // namespace opts

namespace llvm {
namespace bolt {

constexpr const char *DynoStats::Desc[];

bool DynoStats::operator<(const DynoStats &Other) const {
  return std::lexicographical_compare(
    &Stats[FIRST_DYNO_STAT], &Stats[LAST_DYNO_STAT],
    &Other.Stats[FIRST_DYNO_STAT], &Other.Stats[LAST_DYNO_STAT]
  );
}

bool DynoStats::operator==(const DynoStats &Other) const {
  return std::equal(
    &Stats[FIRST_DYNO_STAT], &Stats[LAST_DYNO_STAT],
    &Other.Stats[FIRST_DYNO_STAT]
  );
}

bool DynoStats::lessThan(const DynoStats &Other,
                         ArrayRef<Category> Keys) const {
  return std::lexicographical_compare(
    Keys.begin(), Keys.end(),
    Keys.begin(), Keys.end(),
    [this,&Other](const Category A, const Category) {
      return Stats[A] < Other.Stats[A];
    }
  );
}

void DynoStats::print(raw_ostream &OS, const DynoStats *Other) const {
  auto printStatWithDelta = [&](const std::string &Name, uint64_t Stat,
                                uint64_t OtherStat) {
    OS << format("%'20lld : ", Stat * opts::DynoStatsScale) << Name;
    if (Other) {
      if (Stat != OtherStat) {
       OtherStat = std::max(OtherStat, uint64_t(1)); // to prevent divide by 0
       OS << format(" (%+.1f%%)",
                    ( (float) Stat - (float) OtherStat ) * 100.0 /
                      (float) (OtherStat) );
      } else {
        OS << " (=)";
      }
    }
    OS << '\n';
  };

  for (auto Stat = DynoStats::FIRST_DYNO_STAT + 1;
       Stat < DynoStats::LAST_DYNO_STAT;
       ++Stat) {

    if (!PrintAArch64Stats && Stat == DynoStats::VENEER_CALLS_AARCH64)
      continue;

    printStatWithDelta(Desc[Stat], Stats[Stat], Other ? (*Other)[Stat] : 0);
  }
}

void DynoStats::operator+=(const DynoStats &Other) {
  for (auto Stat = DynoStats::FIRST_DYNO_STAT + 1;
       Stat < DynoStats::LAST_DYNO_STAT;
       ++Stat) {
    Stats[Stat] += Other[Stat];
  }
}

DynoStats getDynoStats(const BinaryFunction &BF) {
  auto &BC = BF.getBinaryContext();

  DynoStats Stats(/*PrintAArch64Stats*/ BC.isAArch64());

  // Return empty-stats about the function we don't completely understand.
  if (!BF.isSimple() || !BF.hasValidProfile() || !BF.hasCanonicalCFG())
    return Stats;

  // Update enumeration of basic blocks for correct detection of branch'
  // direction.
  BF.updateLayoutIndices();

  for (const auto &BB : BF.layout()) {
    // The basic block execution count equals to the sum of incoming branch
    // frequencies. This may deviate from the sum of outgoing branches of the
    // basic block especially since the block may contain a function that
    // does not return or a function that throws an exception.
    const uint64_t BBExecutionCount =  BB->getKnownExecutionCount();

    // Ignore empty blocks and blocks that were not executed.
    if (BB->getNumNonPseudos() == 0 || BBExecutionCount == 0)
      continue;

    // Count AArch64 linker-inserted veneers
    if(BF.isAArch64Veneer())
        Stats[DynoStats::VENEER_CALLS_AARCH64] += BF.getKnownExecutionCount();

    // Count the number of calls by iterating through all instructions.
    for (const auto &Instr : *BB) {
      if (BC.MIB->isStore(Instr)) {
        Stats[DynoStats::STORES] += BBExecutionCount;
      }
      if (BC.MIB->isLoad(Instr)) {
        Stats[DynoStats::LOADS] += BBExecutionCount;
      }

      if (!BC.MIB->isCall(Instr))
        continue;

      uint64_t CallFreq = BBExecutionCount;
      if (BC.MIB->getConditionalTailCall(Instr)) {
        CallFreq =
          BC.MIB->getAnnotationWithDefault<uint64_t>(Instr, "CTCTakenCount");
      }
      Stats[DynoStats::FUNCTION_CALLS] += CallFreq;
      if (BC.MIB->isIndirectCall(Instr)) {
        Stats[DynoStats::INDIRECT_CALLS] += CallFreq;
      } else if (const auto *CallSymbol = BC.MIB->getTargetSymbol(Instr)) {
        const auto *BF = BC.getFunctionForSymbol(CallSymbol);
        if (BF && BF->isPLTFunction()) {
          Stats[DynoStats::PLT_CALLS] += CallFreq;

          // We don't process PLT functions and hence have to adjust relevant
          // dynostats here for:
          //
          //   jmp *GOT_ENTRY(%rip)
          //
          // NOTE: this is arch-specific.
          Stats[DynoStats::FUNCTION_CALLS] += CallFreq;
          Stats[DynoStats::INDIRECT_CALLS] += CallFreq;
          Stats[DynoStats::LOADS] += CallFreq;
          Stats[DynoStats::INSTRUCTIONS] += CallFreq;
        }
      }
    }

    Stats[DynoStats::INSTRUCTIONS] += BB->getNumNonPseudos() * BBExecutionCount;

    // Jump tables.
    const auto *LastInstr = BB->getLastNonPseudoInstr();
    if (BC.MIB->getJumpTable(*LastInstr)) {
      Stats[DynoStats::JUMP_TABLE_BRANCHES] += BBExecutionCount;
      DEBUG(
        static uint64_t MostFrequentJT;
        if (BBExecutionCount > MostFrequentJT) {
          MostFrequentJT = BBExecutionCount;
          dbgs() << "BOLT-INFO: most frequently executed jump table is in "
                 << "function " << BF << " in basic block " << BB->getName()
                 << " executed totally " << BBExecutionCount << " times.\n";
        }
      );
      continue;
    }

    if (BC.MIB->isIndirectBranch(*LastInstr) && !BC.MIB->isCall(*LastInstr)) {
      Stats[DynoStats::UNKNOWN_INDIRECT_BRANCHES] += BBExecutionCount;
      continue;
    }

    // Update stats for branches.
    const MCSymbol *TBB = nullptr;
    const MCSymbol *FBB = nullptr;
    MCInst *CondBranch = nullptr;
    MCInst *UncondBranch = nullptr;
    if (!BB->analyzeBranch(TBB, FBB, CondBranch, UncondBranch)) {
      continue;
    }

    if (!CondBranch && !UncondBranch) {
      continue;
    }

    // Simple unconditional branch.
    if (!CondBranch) {
      Stats[DynoStats::UNCOND_BRANCHES] += BBExecutionCount;
      continue;
    }

    // CTCs: instruction annotations could be stripped, hence check the number
    // of successors to identify conditional tail calls.
    if (BB->succ_size() == 1) {
      if (BB->branch_info_begin() != BB->branch_info_end())
        Stats[DynoStats::UNCOND_BRANCHES] += BB->branch_info_begin()->Count;
      continue;
    }

    // Conditional branch that could be followed by an unconditional branch.
    auto TakenCount = BB->getTakenBranchInfo().Count;
    if (TakenCount == BinaryBasicBlock::COUNT_NO_PROFILE)
      TakenCount = 0;

    auto NonTakenCount = BB->getFallthroughBranchInfo().Count;
    if (NonTakenCount == BinaryBasicBlock::COUNT_NO_PROFILE)
      NonTakenCount = 0;

    if (BF.isForwardBranch(BB, BB->getConditionalSuccessor(true))) {
      Stats[DynoStats::FORWARD_COND_BRANCHES] += BBExecutionCount;
      Stats[DynoStats::FORWARD_COND_BRANCHES_TAKEN] += TakenCount;
    } else {
      Stats[DynoStats::BACKWARD_COND_BRANCHES] += BBExecutionCount;
      Stats[DynoStats::BACKWARD_COND_BRANCHES_TAKEN] += TakenCount;
    }

    if (UncondBranch) {
      Stats[DynoStats::UNCOND_BRANCHES] += NonTakenCount;
    }
  }

  return Stats;
}

} // namespace bolt
} // namespace llvm
