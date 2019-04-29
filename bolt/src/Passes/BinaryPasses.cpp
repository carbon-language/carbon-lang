//===--- BinaryPasses.cpp - Binary-level analysis/optimization passes -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "BinaryPasses.h"
#include "Passes/ReorderAlgorithm.h"
#include "llvm/Support/Options.h"
#include <numeric>

#define DEBUG_TYPE "bolt-opts"

using namespace llvm;

namespace {

const char* dynoStatsOptName(const bolt::DynoStats::Category C) {
  if (C == bolt::DynoStats::FIRST_DYNO_STAT)
    return "none";
  else if (C == bolt::DynoStats::LAST_DYNO_STAT)
    return "all";

  static std::string OptNames[bolt::DynoStats::LAST_DYNO_STAT+1];

  OptNames[C] = bolt::DynoStats::Description(C);

  std::replace(OptNames[C].begin(), OptNames[C].end(), ' ', '-');

  return OptNames[C].c_str();
}

const char* dynoStatsOptDesc(const bolt::DynoStats::Category C) {
  if (C == bolt::DynoStats::FIRST_DYNO_STAT)
    return "unsorted";
  else if (C == bolt::DynoStats::LAST_DYNO_STAT)
    return "sorted by all stats";

  return bolt::DynoStats::Description(C);
}

}

namespace opts {

extern cl::OptionCategory BoltCategory;
extern cl::OptionCategory BoltOptCategory;

extern cl::opt<bolt::MacroFusionType> AlignMacroOpFusion;
extern cl::opt<unsigned> Verbosity;
extern cl::opt<bool> SplitEH;
extern cl::opt<bolt::BinaryFunction::SplittingType> SplitFunctions;
extern bool shouldProcess(const bolt::BinaryFunction &Function);
extern bool isHotTextMover(const bolt::BinaryFunction &Function);

enum DynoStatsSortOrder : char {
  Ascending,
  Descending
};

static cl::opt<bool>
AggressiveSplitting("split-all-cold",
  cl::desc("outline as many cold basic blocks as possible"),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<DynoStatsSortOrder>
DynoStatsSortOrderOpt("print-sorted-by-order",
  cl::desc("use ascending or descending order when printing functions "
           "ordered by dyno stats"),
  cl::ZeroOrMore,
  cl::init(DynoStatsSortOrder::Descending),
  cl::cat(BoltOptCategory));

static cl::opt<bool>
MinBranchClusters("min-branch-clusters",
  cl::desc("use a modified clustering algorithm geared towards minimizing "
           "branches"),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

enum PeepholeOpts : char {
  PEEP_NONE             = 0x0,
  PEEP_SHORTEN          = 0x1,
  PEEP_DOUBLE_JUMPS     = 0x2,
  PEEP_TAILCALL_TRAPS   = 0x4,
  PEEP_USELESS_BRANCHES = 0x8,
  PEEP_ALL              = 0xf
};

static cl::list<PeepholeOpts>
Peepholes("peepholes",
  cl::CommaSeparated,
  cl::desc("enable peephole optimizations"),
  cl::value_desc("opt1,opt2,opt3,..."),
  cl::values(
    clEnumValN(PEEP_NONE, "none", "disable peepholes"),
    clEnumValN(PEEP_SHORTEN, "shorten", "perform instruction shortening"),
    clEnumValN(PEEP_DOUBLE_JUMPS, "double-jumps",
               "remove double jumps when able"),
    clEnumValN(PEEP_TAILCALL_TRAPS, "tailcall-traps", "insert tail call traps"),
    clEnumValN(PEEP_USELESS_BRANCHES, "useless-branches",
               "remove useless conditional branches"),
    clEnumValN(PEEP_ALL, "all", "enable all peephole optimizations")),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<unsigned>
PrintFuncStat("print-function-statistics",
  cl::desc("print statistics about basic block ordering"),
  cl::init(0),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::list<bolt::DynoStats::Category>
PrintSortedBy("print-sorted-by",
  cl::CommaSeparated,
  cl::desc("print functions sorted by order of dyno stats"),
  cl::value_desc("key1,key2,key3,..."),
  cl::values(
#define D(name, ...)                                        \
    clEnumValN(bolt::DynoStats::name,                     \
               dynoStatsOptName(bolt::DynoStats::name),   \
               dynoStatsOptDesc(bolt::DynoStats::name)),
    DYNO_STATS
#undef D
    clEnumValN(0xffff, ".", ".")
    ),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<bolt::ReorderBasicBlocks::LayoutType>
ReorderBlocks("reorder-blocks",
  cl::desc("change layout of basic blocks in a function"),
  cl::init(bolt::ReorderBasicBlocks::LT_NONE),
  cl::values(
    clEnumValN(bolt::ReorderBasicBlocks::LT_NONE,
      "none",
      "do not reorder basic blocks"),
    clEnumValN(bolt::ReorderBasicBlocks::LT_REVERSE,
      "reverse",
      "layout blocks in reverse order"),
    clEnumValN(bolt::ReorderBasicBlocks::LT_OPTIMIZE,
      "normal",
      "perform optimal layout based on profile"),
    clEnumValN(bolt::ReorderBasicBlocks::LT_OPTIMIZE_BRANCH,
      "branch-predictor",
      "perform optimal layout prioritizing branch "
      "predictions"),
    clEnumValN(bolt::ReorderBasicBlocks::LT_OPTIMIZE_CACHE,
      "cache",
      "perform optimal layout prioritizing I-cache "
      "behavior"),
    clEnumValN(bolt::ReorderBasicBlocks::LT_OPTIMIZE_CACHE_PLUS,
      "cache+",
      "perform layout optimizing I-cache behavior"),
    clEnumValN(bolt::ReorderBasicBlocks::LT_OPTIMIZE_SHUFFLE,
      "cluster-shuffle",
      "perform random layout of clusters")),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<unsigned>
ReportBadLayout("report-bad-layout",
  cl::desc("print top <uint> functions with suboptimal code layout on input"),
  cl::init(0),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
ReportStaleFuncs("report-stale",
  cl::desc("print the list of functions with stale profile"),
  cl::init(false),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

enum SctcModes : char {
  SctcAlways,
  SctcPreserveDirection,
  SctcHeuristic
};

static cl::opt<SctcModes>
SctcMode("sctc-mode",
  cl::desc("mode for simplify conditional tail calls"),
  cl::init(SctcAlways),
  cl::values(clEnumValN(SctcAlways, "always", "always perform sctc"),
    clEnumValN(SctcPreserveDirection,
      "preserve",
      "only perform sctc when branch direction is "
      "preserved"),
    clEnumValN(SctcHeuristic,
      "heuristic",
      "use branch prediction data to control sctc")),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<unsigned>
SplitAlignThreshold("split-align-threshold",
  cl::desc("when deciding to split a function, apply this alignment "
           "while doing the size comparison (see -split-threshold). "
           "Default value: 2."),
  cl::init(2),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<unsigned>
SplitThreshold("split-threshold",
  cl::desc("split function only if its main size is reduced by more than "
           "given amount of bytes. Default value: 0, i.e. split iff the "
           "size is reduced. Note that on some architectures the size can "
           "increase after splitting."),
  cl::init(0),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<unsigned>
TSPThreshold("tsp-threshold",
  cl::desc("maximum number of hot basic blocks in a function for which to use "
           "a precise TSP solution while re-ordering basic blocks"),
  cl::init(10),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<unsigned>
TopCalledLimit("top-called-limit",
  cl::desc("maximum number of functions to print in top called "
           "functions section"),
  cl::init(100),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltCategory));

} // namespace opts

namespace llvm {
namespace bolt {

bool BinaryFunctionPass::shouldOptimize(const BinaryFunction &BF) const {
  return BF.isSimple() &&
         BF.getState() == BinaryFunction::State::CFG &&
         opts::shouldProcess(BF) &&
         (BF.getSize() > 0);
}

bool BinaryFunctionPass::shouldPrint(const BinaryFunction &BF) const {
  return BF.isSimple() && opts::shouldProcess(BF);
}

void EliminateUnreachableBlocks::runOnFunction(BinaryFunction& Function) {
  if (Function.layout_size() > 0) {
    unsigned Count;
    uint64_t Bytes;
    Function.markUnreachableBlocks();
    DEBUG({
      for (auto *BB : Function.layout()) {
        if (!BB->isValid()) {
          dbgs() << "BOLT-INFO: UCE found unreachable block " << BB->getName()
                 << " in function " << Function << "\n";
          BB->dump();
        }
      }
    });
    std::tie(Count, Bytes) = Function.eraseInvalidBBs();
    DeletedBlocks += Count;
    DeletedBytes += Bytes;
    if (Count) {
      Modified.insert(&Function);
      if (opts::Verbosity > 0) {
        outs() << "BOLT-INFO: Removed " << Count
               << " dead basic block(s) accounting for " << Bytes
               << " bytes in function " << Function << '\n';
      }
    }
  }
}

void EliminateUnreachableBlocks::runOnFunctions(BinaryContext &BC) {
  for (auto &It : BC.getBinaryFunctions()) {
    auto &Function = It.second;
    if (shouldOptimize(Function)) {
      runOnFunction(Function);
    }
  }
  outs() << "BOLT-INFO: UCE removed " << DeletedBlocks << " blocks and "
         << DeletedBytes << " bytes of code.\n";
}

bool ReorderBasicBlocks::shouldPrint(const BinaryFunction &BF) const {
  return (BinaryFunctionPass::shouldPrint(BF) &&
          opts::ReorderBlocks != ReorderBasicBlocks::LT_NONE);
}

void ReorderBasicBlocks::runOnFunctions(BinaryContext &BC) {
  if (opts::ReorderBlocks == ReorderBasicBlocks::LT_NONE)
    return;

  IsAArch64 = BC.isAArch64();

  uint64_t ModifiedFuncCount = 0;
  for (auto &It : BC.getBinaryFunctions()) {
    auto &Function = It.second;

    if (!shouldOptimize(Function))
      continue;

    const bool ShouldSplit =
            (opts::SplitFunctions == BinaryFunction::ST_ALL) ||
            (opts::SplitFunctions == BinaryFunction::ST_EH &&
             Function.hasEHRanges()) ||
             Function.shouldSplit();
    modifyFunctionLayout(Function, opts::ReorderBlocks, opts::MinBranchClusters,
                         ShouldSplit);

    if (Function.hasLayoutChanged()) {
      ++ModifiedFuncCount;
    }
  }

  outs() << "BOLT-INFO: basic block reordering modified layout of "
         << format("%zu (%.2lf%%) functions\n",
                   ModifiedFuncCount,
                   100.0 * ModifiedFuncCount / BC.getBinaryFunctions().size());

  if (opts::PrintFuncStat > 0) {
    raw_ostream &OS = outs();
    // Copy all the values into vector in order to sort them
    std::map<uint64_t, BinaryFunction &> ScoreMap;
    auto &BFs = BC.getBinaryFunctions();
    for (auto It = BFs.begin(); It != BFs.end(); ++It) {
      ScoreMap.insert(std::pair<uint64_t, BinaryFunction &>(
          It->second.getFunctionScore(), It->second));
    }

    OS << "\nBOLT-INFO: Printing Function Statistics:\n\n";
    OS << "           There are " << BFs.size() << " functions in total. \n";
    OS << "           Number of functions being modified: " << ModifiedFuncCount
       << "\n";
    OS << "           User asks for detailed information on top "
       << opts::PrintFuncStat << " functions. (Ranked by function score)"
       << "\n\n";
    uint64_t I = 0;
    for (std::map<uint64_t, BinaryFunction &>::reverse_iterator
             Rit = ScoreMap.rbegin();
         Rit != ScoreMap.rend() && I < opts::PrintFuncStat; ++Rit, ++I) {
      auto &Function = Rit->second;

      OS << "           Information for function of top: " << (I + 1) << ": \n";
      OS << "             Function Score is: " << Function.getFunctionScore()
         << "\n";
      OS << "             There are " << Function.size()
         << " number of blocks in this function.\n";
      OS << "             There are " << Function.getInstructionCount()
         << " number of instructions in this function.\n";
      OS << "             The edit distance for this function is: "
         << Function.getEditDistance() << "\n\n";
    }
  }
}

void ReorderBasicBlocks::modifyFunctionLayout(BinaryFunction &BF,
    LayoutType Type, bool MinBranchClusters, bool Split) const {
  if (BF.size() == 0 || Type == LT_NONE)
    return;

  BinaryFunction::BasicBlockOrderType NewLayout;
  std::unique_ptr<ReorderAlgorithm> Algo;

  // Cannot do optimal layout without profile.
  if (Type != LT_REVERSE && !BF.hasValidProfile())
    return;

  if (Type == LT_REVERSE) {
    Algo.reset(new ReverseReorderAlgorithm());
  } else if (BF.size() <= opts::TSPThreshold && Type != LT_OPTIMIZE_SHUFFLE) {
    // Work on optimal solution if problem is small enough
    DEBUG(dbgs() << "finding optimal block layout for " << BF << "\n");
    Algo.reset(new OptimalReorderAlgorithm());
  } else {
    DEBUG(dbgs() << "running block layout heuristics on " << BF << "\n");

    std::unique_ptr<ClusterAlgorithm> CAlgo;
    if (MinBranchClusters)
      CAlgo.reset(new MinBranchGreedyClusterAlgorithm());
    else
      CAlgo.reset(new PHGreedyClusterAlgorithm());

    switch(Type) {
    case LT_OPTIMIZE:
      Algo.reset(new OptimizeReorderAlgorithm(std::move(CAlgo)));
      break;

    case LT_OPTIMIZE_BRANCH:
      Algo.reset(new OptimizeBranchReorderAlgorithm(std::move(CAlgo)));
      break;

    case LT_OPTIMIZE_CACHE:
      Algo.reset(new OptimizeCacheReorderAlgorithm(std::move(CAlgo)));
      break;

    case LT_OPTIMIZE_CACHE_PLUS:
      Algo.reset(new CachePlusReorderAlgorithm());
      break;

    case LT_OPTIMIZE_SHUFFLE:
      Algo.reset(new RandomClusterReorderAlgorithm(std::move(CAlgo)));
      break;

    default:
      llvm_unreachable("unexpected layout type");
    }
  }

  Algo->reorderBasicBlocks(BF, NewLayout);

  BF.updateBasicBlockLayout(NewLayout);

  if (Split)
    splitFunction(BF);
}

void ReorderBasicBlocks::splitFunction(BinaryFunction &BF) const {
  if (!BF.size())
    return;

  bool AllCold = true;
  for (auto *BB : BF.layout()) {
    auto ExecCount = BB->getExecutionCount();
    if (ExecCount == BinaryBasicBlock::COUNT_NO_PROFILE)
      return;
    if (ExecCount != 0)
      AllCold = false;
  }

  if (AllCold)
    return;

  auto PreSplitLayout = BF.getLayout();

  auto &BC = BF.getBinaryContext();
  size_t OriginalHotSize;
  size_t HotSize;
  size_t ColdSize;
  if (BC.isX86())
    std::tie(OriginalHotSize, ColdSize) = BC.calculateEmittedSize(BF);
  DEBUG(dbgs() << "Estimated size for function " << BF << " pre-split is <0x"
               << Twine::utohexstr(OriginalHotSize) << ", 0x"
               << Twine::utohexstr(ColdSize) << ">\n");

  // Never outline the first basic block.
  BF.layout_front()->setCanOutline(false);
  for (auto *BB : BF.layout()) {
    if (!BB->canOutline())
      continue;
    if (BB->getExecutionCount() != 0) {
      BB->setCanOutline(false);
      continue;
    }
    // Do not split extra entry points in aarch64. They can be referred by
    // using ADRs and when this happens, these blocks cannot be placed far
    // away due to the limited range in ADR instruction.
    if (IsAArch64 && BB->isEntryPoint()) {
      BB->setCanOutline(false);
      continue;
    }
    if (BF.hasEHRanges() && !opts::SplitEH) {
      // We cannot move landing pads (or rather entry points for landing
      // pads).
      if (BB->isLandingPad()) {
        BB->setCanOutline(false);
        continue;
      }
      // We cannot move a block that can throw since exception-handling
      // runtime cannot deal with split functions. However, if we can guarantee
      // that the block never throws, it is safe to move the block to
      // decrease the size of the function.
      for (auto &Instr : *BB) {
        if (BF.getBinaryContext().MIB->isInvoke(Instr)) {
          BB->setCanOutline(false);
          break;
        }
      }
    }
  }

  if (opts::AggressiveSplitting) {
    // All blocks with 0 count that we can move go to the end of the function.
    // Even if they were natural to cluster formation and were seen in-between
    // hot basic blocks.
    std::stable_sort(BF.layout_begin(), BF.layout_end(),
        [&] (BinaryBasicBlock *A, BinaryBasicBlock *B) {
          return A->canOutline() < B->canOutline();
        });
  } else if (BF.hasEHRanges() && !opts::SplitEH) {
    // Typically functions with exception handling have landing pads at the end.
    // We cannot move beginning of landing pads, but we can move 0-count blocks
    // comprising landing pads to the end and thus facilitate splitting.
    auto FirstLP = BF.layout_begin();
    while ((*FirstLP)->isLandingPad())
      ++FirstLP;

    std::stable_sort(FirstLP, BF.layout_end(),
        [&] (BinaryBasicBlock *A, BinaryBasicBlock *B) {
          return A->canOutline() < B->canOutline();
        });
  }

  // Separate hot from cold starting from the bottom.
  for (auto I = BF.layout_rbegin(), E = BF.layout_rend();
       I != E; ++I) {
    BinaryBasicBlock *BB = *I;
    if (!BB->canOutline())
      break;
    BB->setIsCold(true);
  }

  // Check the new size to see if it's worth splitting the function.
  if (BC.isX86() && BF.isSplit()) {
    std::tie(HotSize, ColdSize) = BC.calculateEmittedSize(BF);
    DEBUG(dbgs() << "Estimated size for function " << BF << " post-split is <0x"
                 << Twine::utohexstr(HotSize) << ", 0x"
                 << Twine::utohexstr(ColdSize) << ">\n");
    if (alignTo(OriginalHotSize, opts::SplitAlignThreshold) <=
        alignTo(HotSize, opts::SplitAlignThreshold) + opts::SplitThreshold) {
      DEBUG(dbgs() << "Reversing splitting of function " << BF << ":\n  0x"
                   << Twine::utohexstr(HotSize) << ", 0x"
                   << Twine::utohexstr(ColdSize) << " -> 0x"
                   << Twine::utohexstr(OriginalHotSize) << '\n');

      BF.updateBasicBlockLayout(PreSplitLayout);
      for (auto &BB : BF) {
        BB.setIsCold(false);
      }
    }
  }
}

void FixupBranches::runOnFunctions(BinaryContext &BC) {
  for (auto &It : BC.getBinaryFunctions()) {
    auto &Function = It.second;
    if (BC.HasRelocations || shouldOptimize(Function)) {
      if (BC.HasRelocations && !Function.isSimple())
        continue;
      Function.fixBranches();
    }
  }
}

void FinalizeFunctions::runOnFunctions(BinaryContext &BC) {
  for (auto &It : BC.getBinaryFunctions()) {
    auto &Function = It.second;
    const auto ShouldOptimize = shouldOptimize(Function);

    // Always fix functions in relocation mode.
    if (!BC.HasRelocations && !ShouldOptimize)
      continue;

    // Fix the CFI state.
    if (ShouldOptimize && !Function.finalizeCFIState()) {
      if (BC.HasRelocations) {
        errs() << "BOLT-ERROR: unable to fix CFI state for function "
               << Function << ". Exiting.\n";
        exit(1);
      }
      Function.setSimple(false);
      continue;
    }

    Function.setFinalized();

    // Update exception handling information.
    Function.updateEHRanges();
  }
}

void LowerAnnotations::runOnFunctions(BinaryContext &BC) {
  for (auto &It : BC.getBinaryFunctions()) {
    auto &BF = It.second;
    int64_t CurrentGnuArgsSize = 0;

    // Have we crossed hot/cold border for split functions?
    bool SeenCold = false;

    for (auto *BB : BF.layout()) {
      if (BB->isCold() && !SeenCold) {
        SeenCold = true;
        CurrentGnuArgsSize = 0;
      }

      for (auto II = BB->begin(); II != BB->end(); ++II) {
        // Convert GnuArgsSize annotations into CFIs.
        if (BF.usesGnuArgsSize() && BC.MIB->isInvoke(*II)) {
          const auto NewGnuArgsSize = BC.MIB->getGnuArgsSize(*II);
          assert(NewGnuArgsSize >= 0 && "expected non-negative GNU_args_size");
          if (NewGnuArgsSize != CurrentGnuArgsSize) {
            auto InsertII = BF.addCFIInstruction(BB, II,
                MCCFIInstruction::createGnuArgsSize(nullptr, NewGnuArgsSize));
            CurrentGnuArgsSize = NewGnuArgsSize;
            II = std::next(InsertII);
          }
        }
        BC.MIB->removeAllAnnotations(*II);
      }
    }
  }

  // Release all memory taken by annotations.
  BC.MIB->freeAnnotations();
}

namespace {

// This peephole fixes jump instructions that jump to another basic
// block with a single jump instruction, e.g.
//
// B0: ...
//     jmp  B1   (or jcc B1)
//
// B1: jmp  B2
//
// ->
//
// B0: ...
//     jmp  B2   (or jcc B2)
//
uint64_t fixDoubleJumps(BinaryContext &BC,
                        BinaryFunction &Function,
                        bool MarkInvalid) {
  uint64_t NumDoubleJumps = 0;

  for (auto &BB : Function) {
    auto checkAndPatch = [&](BinaryBasicBlock *Pred,
                             BinaryBasicBlock *Succ,
                             const MCSymbol *SuccSym) {
      // Ignore infinite loop jumps or fallthrough tail jumps.
      if (Pred == Succ || Succ == &BB)
        return false;

      if (Succ) {
        const MCSymbol *TBB = nullptr;
        const MCSymbol *FBB = nullptr;
        MCInst *CondBranch = nullptr;
        MCInst *UncondBranch = nullptr;
        auto Res = Pred->analyzeBranch(TBB, FBB, CondBranch, UncondBranch);
        if(!Res) {
          DEBUG(dbgs() << "analyzeBranch failed in peepholes in block:\n";
                Pred->dump());
          return false;
        }
        Pred->replaceSuccessor(&BB, Succ);

        // We must patch up any existing branch instructions to match up
        // with the new successor.
        auto *Ctx = BC.Ctx.get();
        assert((CondBranch || (!CondBranch && Pred->succ_size() == 1)) &&
               "Predecessor block has inconsistent number of successors");
        if (CondBranch &&
            BC.MIB->getTargetSymbol(*CondBranch) == BB.getLabel()) {
          BC.MIB->replaceBranchTarget(*CondBranch, Succ->getLabel(), Ctx);
        } else if (UncondBranch &&
                   BC.MIB->getTargetSymbol(*UncondBranch) == BB.getLabel()) {
          BC.MIB->replaceBranchTarget(*UncondBranch, Succ->getLabel(), Ctx);
        } else if (!UncondBranch) {
          assert(Function.getBasicBlockAfter(Pred, false) != Succ &&
                 "Don't add an explicit jump to a fallthrough block.");
          Pred->addBranchInstruction(Succ);
        }
      } else {
        // Succ will be null in the tail call case.  In this case we
        // need to explicitly add a tail call instruction.
        auto *Branch = Pred->getLastNonPseudoInstr();
        if (Branch && BC.MIB->isUnconditionalBranch(*Branch)) {
          assert(BC.MIB->getTargetSymbol(*Branch) == BB.getLabel());
          Pred->removeSuccessor(&BB);
          Pred->eraseInstruction(Pred->findInstruction(Branch));
          Pred->addTailCallInstruction(SuccSym);
        } else {
          return false;
        }
      }

      ++NumDoubleJumps;
      DEBUG(dbgs() << "Removed double jump in " << Function << " from "
                   << Pred->getName() << " -> " << BB.getName() << " to "
                   << Pred->getName() << " -> " << SuccSym->getName()
                   << (!Succ ? " (tail)\n" : "\n"));

      return true;
    };

    if (BB.getNumNonPseudos() != 1 || BB.isLandingPad())
      continue;

    auto *Inst = BB.getFirstNonPseudoInstr();
    const bool IsTailCall = BC.MIB->isTailCall(*Inst);

    if (!BC.MIB->isUnconditionalBranch(*Inst) && !IsTailCall)
      continue;

    // If we operate after SCTC make sure it's not a conditional tail call.
    if (IsTailCall && BC.MIB->isConditionalBranch(*Inst))
      continue;

    const auto *SuccSym = BC.MIB->getTargetSymbol(*Inst);
    auto *Succ = BB.getSuccessor();

    if (((!Succ || &BB == Succ) && !IsTailCall) || (IsTailCall && !SuccSym))
      continue;

    std::vector<BinaryBasicBlock *> Preds{BB.pred_begin(), BB.pred_end()};

    for (auto *Pred : Preds) {
      if (Pred->isLandingPad())
        continue;

      if (Pred->getSuccessor() == &BB ||
          (Pred->getConditionalSuccessor(true) == &BB && !IsTailCall) ||
          Pred->getConditionalSuccessor(false) == &BB) {
        if (checkAndPatch(Pred, Succ, SuccSym) && MarkInvalid) {
          BB.markValid(BB.pred_size() != 0 ||
                       BB.isLandingPad() ||
                       BB.isEntryPoint());
        }
      }
    }
  }

  return NumDoubleJumps;
}

}

bool SimplifyConditionalTailCalls::shouldRewriteBranch(
    const BinaryBasicBlock *PredBB,
    const MCInst &CondBranch,
    const BinaryBasicBlock *BB,
    const bool DirectionFlag
) {
  if (BeenOptimized.count(PredBB))
    return false;

  const bool IsForward = BinaryFunction::isForwardBranch(PredBB, BB);

  if (IsForward)
    ++NumOrigForwardBranches;
  else
    ++NumOrigBackwardBranches;

  if (opts::SctcMode == opts::SctcAlways)
    return true;

  if (opts::SctcMode == opts::SctcPreserveDirection)
    return IsForward == DirectionFlag;

  const auto Frequency = PredBB->getBranchStats(BB);

  // It's ok to rewrite the conditional branch if the new target will be
  // a backward branch.

  // If no data available for these branches, then it should be ok to
  // do the optimization since it will reduce code size.
  if (Frequency.getError())
    return true;

  // TODO: should this use misprediction frequency instead?
  const bool Result =
    (IsForward && Frequency.get().first >= 0.5) ||
    (!IsForward && Frequency.get().first <= 0.5);

  return Result == DirectionFlag;
}

uint64_t SimplifyConditionalTailCalls::fixTailCalls(BinaryContext &BC,
                                                    BinaryFunction &BF) {
  // Need updated indices to correctly detect branch' direction.
  BF.updateLayoutIndices();
  BF.markUnreachableBlocks();

  auto &MIB = BC.MIB;
  uint64_t NumLocalCTCCandidates = 0;
  uint64_t NumLocalCTCs = 0;
  uint64_t LocalCTCTakenCount = 0;
  uint64_t LocalCTCExecCount = 0;
  std::vector<std::pair<BinaryBasicBlock *,
                        const BinaryBasicBlock *>> NeedsUncondBranch;

  // Will block be deleted by UCE?
  auto isValid = [](const BinaryBasicBlock *BB) {
    return (BB->pred_size() != 0 ||
            BB->isLandingPad() ||
            BB->isEntryPoint());
  };

  for (auto *BB : BF.layout()) {
    // Locate BB with a single direct tail-call instruction.
    if (BB->getNumNonPseudos() != 1)
      continue;

    auto *Instr = BB->getFirstNonPseudoInstr();
    if (!MIB->isTailCall(*Instr) || BC.MIB->isConditionalBranch(*Instr))
      continue;

    auto *CalleeSymbol = MIB->getTargetSymbol(*Instr);
    if (!CalleeSymbol)
      continue;

    // Detect direction of the possible conditional tail call.
    const bool IsForwardCTC = BF.isForwardCall(CalleeSymbol);

    // Iterate through all predecessors.
    for (auto *PredBB : BB->predecessors()) {
      auto *CondSucc = PredBB->getConditionalSuccessor(true);
      if (!CondSucc)
        continue;

      ++NumLocalCTCCandidates;

      const MCSymbol *TBB = nullptr;
      const MCSymbol *FBB = nullptr;
      MCInst *CondBranch = nullptr;
      MCInst *UncondBranch = nullptr;
      auto Result = PredBB->analyzeBranch(TBB, FBB, CondBranch, UncondBranch);

      // analyzeBranch() can fail due to unusual branch instructions, e.g. jrcxz
      if (!Result) {
        DEBUG(dbgs() << "analyzeBranch failed in SCTC in block:\n";
              PredBB->dump());
        continue;
      }

      assert(Result && "internal error analyzing conditional branch");
      assert(CondBranch && "conditional branch expected");

      // It's possible that PredBB is also a successor to BB that may have
      // been processed by a previous iteration of the SCTC loop, in which
      // case it may have been marked invalid.  We should skip rewriting in
      // this case.
      if (!PredBB->isValid()) {
        assert(PredBB->isSuccessor(BB) &&
               "PredBB should be valid if it is not a successor to BB");
        continue;
      }

      // We don't want to reverse direction of the branch in new order
      // without further profile analysis.
      const bool DirectionFlag = CondSucc == BB ? IsForwardCTC : !IsForwardCTC;
      if (!shouldRewriteBranch(PredBB, *CondBranch, BB, DirectionFlag))
        continue;

      // Record this block so that we don't try to optimize it twice.
      BeenOptimized.insert(PredBB);

      uint64_t Count = 0;
      if (CondSucc != BB) {
        // Patch the new target address into the conditional branch.
        MIB->reverseBranchCondition(*CondBranch, CalleeSymbol, BC.Ctx.get());
        // Since we reversed the condition on the branch we need to change
        // the target for the unconditional branch or add a unconditional
        // branch to the old target.  This has to be done manually since
        // fixupBranches is not called after SCTC.
        NeedsUncondBranch.emplace_back(std::make_pair(PredBB, CondSucc));
        Count = PredBB->getFallthroughBranchInfo().Count;
      } else {
        // Change destination of the conditional branch.
        MIB->replaceBranchTarget(*CondBranch, CalleeSymbol, BC.Ctx.get());
        Count = PredBB->getTakenBranchInfo().Count;
      }
      const uint64_t CTCTakenFreq =
        Count == BinaryBasicBlock::COUNT_NO_PROFILE ? 0 : Count;

      // Annotate it, so "isCall" returns true for this jcc
      MIB->setConditionalTailCall(*CondBranch);
      // Add info abount the conditional tail call frequency, otherwise this
      // info will be lost when we delete the associated BranchInfo entry
      auto &CTCAnnotation = BC.MIB->getOrCreateAnnotationAs<uint64_t>(
          *CondBranch, "CTCTakenCount");
      CTCAnnotation = CTCTakenFreq;

      // Remove the unused successor which may be eliminated later
      // if there are no other users.
      PredBB->removeSuccessor(BB);
      // Update BB execution count
      if (CTCTakenFreq && CTCTakenFreq <= BB->getKnownExecutionCount()) {
        BB->setExecutionCount(BB->getExecutionCount() - CTCTakenFreq);
      } else if (CTCTakenFreq > BB->getKnownExecutionCount()) {
        BB->setExecutionCount(0);
      }

      ++NumLocalCTCs;
      LocalCTCTakenCount += CTCTakenFreq;
      LocalCTCExecCount += PredBB->getKnownExecutionCount();
    }

    // Remove the block from CFG if all predecessors were removed.
    BB->markValid(isValid(BB));
  }

  // Add unconditional branches at the end of BBs to new successors
  // as long as the successor is not a fallthrough.
  for (auto &Entry : NeedsUncondBranch) {
    auto *PredBB = Entry.first;
    auto *CondSucc = Entry.second;

    const MCSymbol *TBB = nullptr;
    const MCSymbol *FBB = nullptr;
    MCInst *CondBranch = nullptr;
    MCInst *UncondBranch = nullptr;
    PredBB->analyzeBranch(TBB, FBB, CondBranch, UncondBranch);

    // Find the next valid block.  Invalid blocks will be deleted
    // so they shouldn't be considered fallthrough targets.
    const auto *NextBlock = BF.getBasicBlockAfter(PredBB, false);
    while (NextBlock && !isValid(NextBlock)) {
      NextBlock = BF.getBasicBlockAfter(NextBlock, false);
    }

    // Get the unconditional successor to this block.
    const auto *PredSucc = PredBB->getSuccessor();
    assert(PredSucc && "The other branch should be a tail call");

    const bool HasFallthrough = (NextBlock && PredSucc == NextBlock);

    if (UncondBranch) {
      if (HasFallthrough)
        PredBB->eraseInstruction(PredBB->findInstruction(UncondBranch));
      else
        MIB->replaceBranchTarget(*UncondBranch,
                                 CondSucc->getLabel(),
                                 BC.Ctx.get());
    } else if (!HasFallthrough) {
      MCInst Branch;
      MIB->createUncondBranch(Branch, CondSucc->getLabel(), BC.Ctx.get());
      PredBB->addInstruction(Branch);
    }
  }

  if (NumLocalCTCs > 0) {
    NumDoubleJumps += fixDoubleJumps(BC, BF, true);
    // Clean-up unreachable tail-call blocks.
    const auto Stats = BF.eraseInvalidBBs();
    DeletedBlocks += Stats.first;
    DeletedBytes += Stats.second;

    assert(BF.validateCFG());
  }

  DEBUG(dbgs() << "BOLT: created " << NumLocalCTCs
               << " conditional tail calls from a total of "
               << NumLocalCTCCandidates << " candidates in function " << BF
               << ". CTCs execution count for this function is "
               << LocalCTCExecCount << " and CTC taken count is "
               << LocalCTCTakenCount << "\n";);

  NumTailCallsPatched += NumLocalCTCs;
  NumCandidateTailCalls += NumLocalCTCCandidates;
  CTCExecCount += LocalCTCExecCount;
  CTCTakenCount += LocalCTCTakenCount;

  return NumLocalCTCs > 0;
}

void SimplifyConditionalTailCalls::runOnFunctions(BinaryContext &BC) {
  if (!BC.isX86())
    return;

  for (auto &It : BC.getBinaryFunctions()) {
    auto &Function = It.second;

    if (!shouldOptimize(Function))
      continue;

    if (fixTailCalls(BC, Function)) {
      Modified.insert(&Function);
    }
  }

  outs() << "BOLT-INFO: SCTC: patched " << NumTailCallsPatched
         << " tail calls (" << NumOrigForwardBranches << " forward)"
         << " tail calls (" << NumOrigBackwardBranches << " backward)"
         << " from a total of " << NumCandidateTailCalls << " while removing "
         << NumDoubleJumps << " double jumps"
         << " and removing " << DeletedBlocks << " basic blocks"
         << " totalling " << DeletedBytes
         << " bytes of code. CTCs total execution count is " << CTCExecCount
         << " and the number of times CTCs are taken is " << CTCTakenCount
         << ".\n";
}

uint64_t Peepholes::shortenInstructions(BinaryContext &BC,
                                        BinaryFunction &Function) {
  MCInst DebugInst;
  uint64_t Count = 0;
  for (auto &BB : Function) {
    for (auto &Inst : BB) {
      if (opts::Verbosity > 1) {
        DebugInst = Inst;
      }
      if (BC.MIB->shortenInstruction(Inst)) {
        if (opts::Verbosity > 1) {
          outs() << "BOLT-INFO: peephole, shortening:\n"
                 << "BOLT-INFO:    ";
          BC.printInstruction(outs(), DebugInst, 0, &Function);
          outs() << "BOLT-INFO: to:";
          BC.printInstruction(outs(), Inst, 0, &Function);
        }
        ++Count;
      }
    }
  }
  return Count;
}

void Peepholes::addTailcallTraps(BinaryContext &BC,
                                 BinaryFunction &Function) {
  for (auto &BB : Function) {
    auto *Inst = BB.getLastNonPseudoInstr();
    if (Inst && BC.MIB->isTailCall(*Inst) && BC.MIB->isIndirectBranch(*Inst)) {
      MCInst Trap;
      if (BC.MIB->createTrap(Trap)) {
        BB.addInstruction(Trap);
        ++TailCallTraps;
      }
    }
  }
}

void Peepholes::removeUselessCondBranches(BinaryContext &BC,
                                          BinaryFunction &Function) {
  for (auto &BB : Function) {
    if (BB.succ_size() != 2)
      continue;

    auto *CondBB = BB.getConditionalSuccessor(true);
    auto *UncondBB = BB.getConditionalSuccessor(false);
    if (CondBB != UncondBB)
      continue;

    const MCSymbol *TBB = nullptr;
    const MCSymbol *FBB = nullptr;
    MCInst *CondBranch = nullptr;
    MCInst *UncondBranch = nullptr;
    auto Result = BB.analyzeBranch(TBB, FBB, CondBranch, UncondBranch);

    // analyzeBranch() can fail due to unusual branch instructions,
    // e.g. jrcxz, or jump tables (indirect jump).
    if (!Result || !CondBranch)
      continue;

    BB.removeDuplicateConditionalSuccessor(CondBranch);
    ++NumUselessCondBranches;
  }
}

void Peepholes::runOnFunctions(BinaryContext &BC) {
  const char Opts =
    std::accumulate(opts::Peepholes.begin(),
                    opts::Peepholes.end(),
                    0,
                    [](const char A, const opts::PeepholeOpts B) {
                      return A | B;
                    });
  if (Opts == opts::PEEP_NONE || !BC.isX86())
    return;

  for (auto &It : BC.getBinaryFunctions()) {
    auto &Function = It.second;
    if (shouldOptimize(Function)) {
      if (Opts & opts::PEEP_SHORTEN)
        NumShortened += shortenInstructions(BC, Function);
      if (Opts & opts::PEEP_DOUBLE_JUMPS)
        NumDoubleJumps += fixDoubleJumps(BC, Function, false);
      if (Opts & opts::PEEP_TAILCALL_TRAPS)
        addTailcallTraps(BC, Function);
      if (Opts & opts::PEEP_USELESS_BRANCHES)
        removeUselessCondBranches(BC, Function);
      assert(Function.validateCFG());
    }
  }
  outs() << "BOLT-INFO: Peephole: " << NumShortened
         << " instructions shortened.\n"
         << "BOLT-INFO: Peephole: " << NumDoubleJumps
         << " double jumps patched.\n"
         << "BOLT-INFO: Peephole: " << TailCallTraps
         << " tail call traps inserted.\n"
         << "BOLT-INFO: Peephole: " << NumUselessCondBranches
         << " useless conditional branches removed.\n";
}

bool SimplifyRODataLoads::simplifyRODataLoads(
    BinaryContext &BC, BinaryFunction &BF) {
  auto &MIB = BC.MIB;

  uint64_t NumLocalLoadsSimplified = 0;
  uint64_t NumDynamicLocalLoadsSimplified = 0;
  uint64_t NumLocalLoadsFound = 0;
  uint64_t NumDynamicLocalLoadsFound = 0;

  for (auto *BB : BF.layout()) {
    for (auto &Inst : *BB) {
      unsigned Opcode = Inst.getOpcode();
      const MCInstrDesc &Desc = BC.MII->get(Opcode);

      // Skip instructions that do not load from memory.
      if (!Desc.mayLoad())
        continue;

      // Try to statically evaluate the target memory address;
      uint64_t TargetAddress;

      if (MIB->hasPCRelOperand(Inst)) {
        // Try to find the symbol that corresponds to the PC-relative operand.
        auto DispOpI = MIB->getMemOperandDisp(Inst);
        assert(DispOpI != Inst.end() && "expected PC-relative displacement");
        assert(DispOpI->isExpr() &&
              "found PC-relative with non-symbolic displacement");

        // Get displacement symbol.
        const MCSymbol *DisplSymbol;
        uint64_t DisplOffset;

        std::tie(DisplSymbol, DisplOffset) =
          BC.MIB->getTargetSymbolInfo(DispOpI->getExpr());

        if (!DisplSymbol)
          continue;

        // Look up the symbol address in the global symbols map of the binary
        // context object.
        auto *BD = BC.getBinaryDataByName(DisplSymbol->getName());
        if (!BD)
          continue;
        TargetAddress = BD->getAddress() + DisplOffset;
      } else if (!MIB->evaluateMemOperandTarget(Inst, TargetAddress)) {
        continue;
      }

      // Get the contents of the section containing the target address of the
      // memory operand. We are only interested in read-only sections.
      auto DataSection = BC.getSectionForAddress(TargetAddress);
      if (!DataSection || !DataSection->isReadOnly())
        continue;

      if (BC.getRelocationAt(TargetAddress))
        continue;

      uint32_t Offset = TargetAddress - DataSection->getAddress();
      StringRef ConstantData = DataSection->getContents();

      ++NumLocalLoadsFound;
      if (BB->hasProfile())
        NumDynamicLocalLoadsFound += BB->getExecutionCount();

      if (MIB->replaceMemOperandWithImm(Inst, ConstantData, Offset)) {
        ++NumLocalLoadsSimplified;
        if (BB->hasProfile())
          NumDynamicLocalLoadsSimplified += BB->getExecutionCount();
      }
    }
  }

  NumLoadsFound += NumLocalLoadsFound;
  NumDynamicLoadsFound += NumDynamicLocalLoadsFound;
  NumLoadsSimplified += NumLocalLoadsSimplified;
  NumDynamicLoadsSimplified += NumDynamicLocalLoadsSimplified;

  return NumLocalLoadsSimplified > 0;
}

void SimplifyRODataLoads::runOnFunctions(BinaryContext &BC) {
  for (auto &It : BC.getBinaryFunctions()) {
    auto &Function = It.second;
    if (shouldOptimize(Function) && simplifyRODataLoads(BC, Function)) {
      Modified.insert(&Function);
    }
  }

  outs() << "BOLT-INFO: simplified " << NumLoadsSimplified << " out of "
         << NumLoadsFound << " loads from a statically computed address.\n"
         << "BOLT-INFO: dynamic loads simplified: " << NumDynamicLoadsSimplified
         << "\n"
         << "BOLT-INFO: dynamic loads found: " << NumDynamicLoadsFound << "\n";
}

void AssignSections::runOnFunctions(BinaryContext &BC) {
  for (auto *Function : BC.getInjectedBinaryFunctions()) {
    Function->setCodeSectionName(BC.getInjectedCodeSectionName());
    Function->setColdCodeSectionName(BC.getInjectedColdCodeSectionName());
  }

  // In non-relocation mode functions have pre-assigned section names.
  if (!BC.HasRelocations)
    return;

  const auto UseColdSection = BC.NumProfiledFuncs > 0;
  for (auto &BFI : BC.getBinaryFunctions()) {
    auto &Function = BFI.second;
    if (opts::isHotTextMover(Function)) {
      Function.setCodeSectionName(BC.getHotTextMoverSectionName());
      Function.setColdCodeSectionName(BC.getHotTextMoverSectionName());
      continue;
    }

    if (!UseColdSection ||
        Function.hasValidIndex() ||
        Function.hasValidProfile()) {
      Function.setCodeSectionName(BC.getMainCodeSectionName());
    } else {
      Function.setCodeSectionName(BC.getColdCodeSectionName());
    }

    if (Function.isSplit())
      Function.setColdCodeSectionName(BC.getColdCodeSectionName());
  }
}

void
PrintProgramStats::runOnFunctions(BinaryContext &BC) {
  uint64_t NumSimpleFunctions{0};
  uint64_t NumStaleProfileFunctions{0};
  uint64_t NumNonSimpleProfiledFunctions{0};
  std::vector<BinaryFunction *> ProfiledFunctions;
  const char *StaleFuncsHeader = "BOLT-INFO: Functions with stale profile:\n";
  for (auto &BFI : BC.getBinaryFunctions()) {
    auto &Function = BFI.second;
    if (!Function.isSimple()) {
      if (Function.hasProfile()) {
        ++NumNonSimpleProfiledFunctions;
      }
      continue;
    }
    ++NumSimpleFunctions;
    if (!Function.hasProfile())
      continue;
    if (Function.hasValidProfile()) {
      ProfiledFunctions.push_back(&Function);
    } else {
      if (opts::ReportStaleFuncs) {
        outs() << StaleFuncsHeader;
        StaleFuncsHeader = "";
        outs() << "  " << Function << '\n';
      }
      ++NumStaleProfileFunctions;
    }
  }
  BC.NumProfiledFuncs = ProfiledFunctions.size();

  const auto NumAllProfiledFunctions =
                            ProfiledFunctions.size() + NumStaleProfileFunctions;
  outs() << "BOLT-INFO: "
         << NumAllProfiledFunctions
         << " functions out of " << NumSimpleFunctions << " simple functions ("
         << format("%.1f", NumAllProfiledFunctions /
                                            (float) NumSimpleFunctions * 100.0f)
         << "%) have non-empty execution profile.\n";
  if (NumNonSimpleProfiledFunctions) {
    outs() << "BOLT-INFO: " << NumNonSimpleProfiledFunctions
           << " non-simple function(s) have profile.\n";
  }
  if (NumStaleProfileFunctions) {
    outs() << "BOLT-INFO: " << NumStaleProfileFunctions
           << format(" (%.1f%% of all profiled)",
                     NumStaleProfileFunctions /
                                      (float) NumAllProfiledFunctions * 100.0f)
           << " function" << (NumStaleProfileFunctions == 1 ? "" : "s")
           << " have invalid (possibly stale) profile."
              " Use -report-stale to see the list.\n";
  }

  // Profile is marked as 'Used' if it either matches a function name
  // exactly or if it 100% matches any of functions with matching common
  // LTO names.
  auto getUnusedObjects = [&]() -> Optional<std::vector<StringRef>> {
    std::vector<StringRef> UnusedObjects;
    for (const auto &Func : BC.DR.getAllFuncsData()) {
      if (!Func.getValue().Used) {
        UnusedObjects.emplace_back(Func.getKey());
      }
    }
    if (UnusedObjects.empty())
      return NoneType();
    return UnusedObjects;
  };

  if (const auto UnusedObjects = getUnusedObjects()) {
    outs() << "BOLT-INFO: profile for " << UnusedObjects->size()
           << " objects was ignored\n";
    if (opts::Verbosity >= 1) {
      for (auto Name : *UnusedObjects) {
        outs() << "  " << Name << '\n';
      }
    }
  }

  if (ProfiledFunctions.size() > 10) {
    if (opts::Verbosity >= 1) {
      outs() << "BOLT-INFO: top called functions are:\n";
      std::sort(ProfiledFunctions.begin(), ProfiledFunctions.end(),
                [](BinaryFunction *A, BinaryFunction *B) {
                  return B->getExecutionCount() < A->getExecutionCount();
                }
                );
      auto SFI = ProfiledFunctions.begin();
      auto SFIend = ProfiledFunctions.end();
      for (auto I = 0u; I < opts::TopCalledLimit && SFI != SFIend; ++SFI, ++I) {
        outs() << "  " << **SFI << " : "
               << (*SFI)->getExecutionCount() << '\n';
      }
    }
  }

  if (!opts::PrintSortedBy.empty() &&
      std::find(opts::PrintSortedBy.begin(),
                opts::PrintSortedBy.end(),
                DynoStats::FIRST_DYNO_STAT) == opts::PrintSortedBy.end()) {

    std::vector<const BinaryFunction *> Functions;
    std::map<const BinaryFunction *, DynoStats> Stats;

    for (const auto &BFI : BC.getBinaryFunctions()) {
      const auto &BF = BFI.second;
      if (shouldOptimize(BF) && BF.hasValidProfile()) {
        Functions.push_back(&BF);
        Stats.emplace(&BF, getDynoStats(BF));
      }
    }

    const bool SortAll =
      std::find(opts::PrintSortedBy.begin(),
                opts::PrintSortedBy.end(),
                DynoStats::LAST_DYNO_STAT) != opts::PrintSortedBy.end();

    const bool Ascending =
      opts::DynoStatsSortOrderOpt == opts::DynoStatsSortOrder::Ascending;

    if (SortAll) {
      std::stable_sort(
        Functions.begin(),
        Functions.end(),
        [Ascending,&Stats](const BinaryFunction *A, const BinaryFunction *B) {
          return Ascending ?
            Stats.at(A) < Stats.at(B) : Stats.at(B) < Stats.at(A);
        }
      );
    } else {
      std::stable_sort(
        Functions.begin(),
        Functions.end(),
        [Ascending,&Stats](const BinaryFunction *A, const BinaryFunction *B) {
          const auto &StatsA = Stats.at(A);
          const auto &StatsB = Stats.at(B);
          return Ascending
            ? StatsA.lessThan(StatsB, opts::PrintSortedBy)
            : StatsB.lessThan(StatsA, opts::PrintSortedBy);
        }
      );
    }

    outs() << "BOLT-INFO: top functions sorted by ";
    if (SortAll) {
      outs() << "dyno stats";
    } else {
      outs() << "(";
      bool PrintComma = false;
      for (const auto Category : opts::PrintSortedBy) {
        if (PrintComma) outs() << ", ";
        outs() << DynoStats::Description(Category);
        PrintComma = true;
      }
      outs() << ")";
    }

    outs() << " are:\n";
    auto SFI = Functions.begin();
    for (unsigned I = 0; I < 100 && SFI != Functions.end(); ++SFI, ++I) {
      const auto Stats = getDynoStats(**SFI);
      outs() << "  " << **SFI;
      if (!SortAll) {
        outs() << " (";
        bool PrintComma = false;
        for (const auto Category : opts::PrintSortedBy) {
          if (PrintComma) outs() << ", ";
          outs() << dynoStatsOptName(Category) << "=" << Stats[Category];
          PrintComma = true;
        }
        outs() << ")";
      }
      outs() << "\n";
    }
  }

  if (!BC.TrappedFunctions.empty()) {
    errs() << "BOLT-WARNING: " << BC.TrappedFunctions.size()
           << " functions will trap on entry";
    if (opts::Verbosity >= 1) {
      errs() << ".\n";
      for (const auto *Function : BC.TrappedFunctions)
        errs() << "  " << *Function << '\n';
    } else {
      errs() << " (use -v=1 to see the list).\n";
    }
  }

  // Print information on missed macro-fusion opportunities seen on input.
  if (BC.MissedMacroFusionPairs) {
    outs() << "BOLT-INFO: the input contains "
           << BC.MissedMacroFusionPairs << " (dynamic count : "
           << BC.MissedMacroFusionExecCount
           << ") missed opportunities for macro-fusion optimization";
    switch (opts::AlignMacroOpFusion) {
    case MFT_NONE:
      outs() << ". Use -align-macro-fusion to fix.\n";
      break;
    case MFT_HOT:
      outs() << ". Will fix instances on a hot path.\n";
      break;
    case MFT_ALL:
      outs() << " that are going to be fixed\n";
      break;
    }
  }

  // Collect and print information about suboptimal code layout on input.
  if (opts::ReportBadLayout) {
    std::vector<const BinaryFunction *> SuboptimalFuncs;
    for (auto &BFI : BC.getBinaryFunctions()) {
      const auto &BF = BFI.second;
      if (!BF.hasValidProfile())
        continue;

      const auto HotThreshold =
          std::max<uint64_t>(BF.getKnownExecutionCount(), 1);
      bool HotSeen = false;
      for (const auto *BB : BF.rlayout()) {
        if (!HotSeen && BB->getKnownExecutionCount() > HotThreshold) {
          HotSeen = true;
          continue;
        }
        if (HotSeen && BB->getKnownExecutionCount() == 0) {
          SuboptimalFuncs.push_back(&BF);
          break;
        }
      }
    }

    if (!SuboptimalFuncs.empty()) {
      std::sort(SuboptimalFuncs.begin(), SuboptimalFuncs.end(),
               [](const BinaryFunction *A, const BinaryFunction *B) {
                 return A->getKnownExecutionCount() / A->getSize() >
                        B->getKnownExecutionCount() / B->getSize();
               });

      outs() << "BOLT-INFO: " << SuboptimalFuncs.size() << " functions have "
                "cold code in the middle of hot code. Top functions are:\n";
      for (unsigned I = 0;
           I < std::min(static_cast<size_t>(opts::ReportBadLayout),
                        SuboptimalFuncs.size());
           ++I) {
        SuboptimalFuncs[I]->print(outs());
      }
    }
  }
}

void InstructionLowering::runOnFunctions(BinaryContext &BC) {
  for (auto &BFI : BC.getBinaryFunctions()) {
    for (auto &BB : BFI.second) {
      for (auto &Instruction : BB) {
        BC.MIB->lowerTailCall(Instruction);
      }
    }
  }
}

void StripRepRet::runOnFunctions(BinaryContext &BC) {
  uint64_t NumPrefixesRemoved = 0;
  uint64_t NumBytesSaved = 0;
  for (auto &BFI : BC.getBinaryFunctions()) {
    for (auto &BB : BFI.second) {
      auto LastInstRIter = BB.getLastNonPseudo();
      if (LastInstRIter == BB.rend() ||
          !BC.MIB->isReturn(*LastInstRIter) ||
          !BC.MIB->deleteREPPrefix(*LastInstRIter))
        continue;

      NumPrefixesRemoved += BB.getKnownExecutionCount();
      ++NumBytesSaved;
    }
  }

  if (NumBytesSaved) {
    outs() << "BOLT-INFO: removed " << NumBytesSaved << " 'repz' prefixes"
              " with estimated execution count of " << NumPrefixesRemoved
           << " times.\n";
  }
}

void InlineMemcpy::runOnFunctions(BinaryContext &BC) {
  if (!BC.isX86())
    return;

  uint64_t NumInlined = 0;
  uint64_t NumInlinedDyno = 0;
  for (auto &BFI : BC.getBinaryFunctions()) {
    for (auto &BB : BFI.second) {
      for(auto II = BB.begin(); II != BB.end(); ++II) {
        auto &Inst = *II;

        if (!BC.MIB->isCall(Inst) || MCPlus::getNumPrimeOperands(Inst) != 1 ||
            !Inst.getOperand(0).isExpr())
          continue;

        const auto *CalleeSymbol = BC.MIB->getTargetSymbol(Inst);
        if (CalleeSymbol->getName() != "memcpy" &&
            CalleeSymbol->getName() != "memcpy@PLT" &&
            CalleeSymbol->getName() != "_memcpy8")
          continue;

        const auto IsMemcpy8 = (CalleeSymbol->getName() == "_memcpy8");
        const auto IsTailCall = BC.MIB->isTailCall(Inst);

        const auto NewCode = BC.MIB->createInlineMemcpy(IsMemcpy8);
        II = BB.replaceInstruction(II, NewCode);
        std::advance(II, NewCode.size() - 1);
        if (IsTailCall) {
          MCInst Return;
          BC.MIB->createReturn(Return);
          II = BB.insertInstruction(std::next(II), std::move(Return));
        }

        ++NumInlined;
        NumInlinedDyno += BB.getKnownExecutionCount();
      }
    }
  }

  if (NumInlined) {
    outs() << "BOLT-INFO: inlined " << NumInlined << " memcpy() calls";
    if (NumInlinedDyno)
      outs() << ". The calls were executed " << NumInlinedDyno
             << " times based on profile.";
    outs() << '\n';
  }
}

} // namespace bolt
} // namespace llvm
