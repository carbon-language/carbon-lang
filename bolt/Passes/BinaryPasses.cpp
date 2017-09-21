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
#include "llvm/Support/Options.h"

#define DEBUG_TYPE "bolt"

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

extern cl::OptionCategory BoltOptCategory;

extern cl::opt<unsigned> Verbosity;
extern cl::opt<bool> Relocs;
extern cl::opt<bolt::BinaryFunction::SplittingType> SplitFunctions;
extern bool shouldProcess(const bolt::BinaryFunction &Function);

enum DynoStatsSortOrder : char {
  Ascending,
  Descending
};

static cl::opt<DynoStatsSortOrder>
DynoStatsSortOrderOpt("print-sorted-by-order",
  cl::desc("use ascending or descending order when printing functions "
           "ordered by dyno stats"),
  cl::ZeroOrMore,
  cl::init(DynoStatsSortOrder::Descending),
  cl::cat(BoltOptCategory));

static cl::opt<bool>
ICFUseDFS("icf-dfs",
  cl::desc("use DFS ordering when using -icf option"),
  cl::ReallyHidden,
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
MinBranchClusters("min-branch-clusters",
  cl::desc("use a modified clustering algorithm geared towards minimizing "
           "branches"),
  cl::ZeroOrMore,
  cl::Hidden,
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
    clEnumValEnd),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

cl::opt<unsigned>
PrintFuncStat("print-function-statistics",
  cl::desc("print statistics about basic block ordering"),
  cl::init(0),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<bolt::BinaryFunction::LayoutType>
ReorderBlocks("reorder-blocks",
  cl::desc("change layout of basic blocks in a function"),
  cl::init(bolt::BinaryFunction::LT_NONE),
  cl::values(
    clEnumValN(bolt::BinaryFunction::LT_NONE,
      "none",
      "do not reorder basic blocks"),
    clEnumValN(bolt::BinaryFunction::LT_REVERSE,
      "reverse",
      "layout blocks in reverse order"),
    clEnumValN(bolt::BinaryFunction::LT_OPTIMIZE,
      "normal",
      "perform optimal layout based on profile"),
    clEnumValN(bolt::BinaryFunction::LT_OPTIMIZE_BRANCH,
      "branch-predictor",
      "perform optimal layout prioritizing branch "
      "predictions"),
    clEnumValN(bolt::BinaryFunction::LT_OPTIMIZE_CACHE,
      "cache",
      "perform optimal layout prioritizing I-cache "
      "behavior"),
    clEnumValN(bolt::BinaryFunction::LT_OPTIMIZE_SHUFFLE,
      "cluster-shuffle",
      "perform random layout of clusters"),
    clEnumValEnd),
  cl::ZeroOrMore,
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
      "use branch prediction data to control sctc"),
    clEnumValEnd),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

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

void OptimizeBodylessFunctions::analyze(
    BinaryFunction &BF,
    BinaryContext &BC,
    std::map<uint64_t, BinaryFunction> &BFs) {
  if (BF.size() != 1 || BF.front().getNumNonPseudos() != 1)
    return;

  const auto *FirstInstr = BF.front().getFirstNonPseudoInstr();
  if (!FirstInstr)
    return;
  if (!BC.MIA->isTailCall(*FirstInstr))
    return;
  const auto *TargetSymbol = BC.MIA->getTargetSymbol(*FirstInstr);
  if (!TargetSymbol)
    return;
  const auto *Function = BC.getFunctionForSymbol(TargetSymbol);
  if (!Function)
    return;

  EquivalentCallTarget[BF.getSymbol()] = Function;
}

void OptimizeBodylessFunctions::optimizeCalls(BinaryFunction &BF,
                                              BinaryContext &BC) {
  for (auto *BB : BF.layout()) {
    for (auto &Inst : *BB) {
      if (!BC.MIA->isCall(Inst))
        continue;
      const auto *OriginalTarget = BC.MIA->getTargetSymbol(Inst);
      if (!OriginalTarget)
        continue;
      const auto *Target = OriginalTarget;
      // Iteratively update target since we could have f1() calling f2()
      // calling f3() calling f4() and we want to output f1() directly
      // calling f4().
      unsigned CallSites = 0;
      while (EquivalentCallTarget.count(Target)) {
        Target = EquivalentCallTarget.find(Target)->second->getSymbol();
        ++CallSites;
      }
      if (Target == OriginalTarget)
        continue;
      DEBUG(dbgs() << "BOLT-DEBUG: Optimizing " << BB->getName()
                   << " (executed " << BB->getKnownExecutionCount()
                   << " times) in " << BF
                   << ": replacing call to " << OriginalTarget->getName()
                   << " by call to " << Target->getName()
                   << " while folding " << CallSites << " call sites\n");
      BC.MIA->replaceCallTargetOperand(Inst, Target, BC.Ctx.get());

      NumOptimizedCallSites += CallSites;
      if (BB->hasProfile()) {
        NumEliminatedCalls += CallSites * BB->getExecutionCount();
      }
    }
  }
}

void OptimizeBodylessFunctions::runOnFunctions(
    BinaryContext &BC,
    std::map<uint64_t, BinaryFunction> &BFs,
    std::set<uint64_t> &) {
  for (auto &It : BFs) {
    auto &Function = It.second;
    if (shouldOptimize(Function)) {
      analyze(Function, BC, BFs);
    }
  }
  for (auto &It : BFs) {
    auto &Function = It.second;
    if (shouldOptimize(Function)) {
      optimizeCalls(Function, BC);
    }
  }

  if (NumEliminatedCalls || NumOptimizedCallSites) {
    outs() << "BOLT-INFO: optimized " << NumOptimizedCallSites
           << " redirect call sites to eliminate " << NumEliminatedCalls
           << " dynamic calls.\n";
  }
}

void EliminateUnreachableBlocks::runOnFunction(BinaryFunction& Function) {
  if (Function.layout_size() > 0) {
    unsigned Count;
    uint64_t Bytes;
    Function.markUnreachable();
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
    if (Count && opts::Verbosity > 0) {
      Modified.insert(&Function);
      outs() << "BOLT-INFO: Removed " << Count
             << " dead basic block(s) accounting for " << Bytes
             << " bytes in function " << Function << '\n';
    }
  }
}

void EliminateUnreachableBlocks::runOnFunctions(
  BinaryContext&,
  std::map<uint64_t, BinaryFunction> &BFs,
  std::set<uint64_t> &
) {
  for (auto &It : BFs) {
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
          opts::ReorderBlocks != BinaryFunction::LT_NONE);
}

void ReorderBasicBlocks::runOnFunctions(
        BinaryContext &BC,
        std::map<uint64_t, BinaryFunction> &BFs,
        std::set<uint64_t> &LargeFunctions) {
  if (opts::ReorderBlocks == BinaryFunction::LT_NONE)
    return;

  uint64_t ModifiedFuncCount = 0;
  for (auto &It : BFs) {
    auto &Function = It.second;

    if (!shouldOptimize(Function))
      continue;

    const bool ShouldSplit =
            (opts::SplitFunctions == BinaryFunction::ST_ALL) ||
            (opts::SplitFunctions == BinaryFunction::ST_EH &&
             Function.hasEHRanges()) ||
            (LargeFunctions.find(It.first) != LargeFunctions.end());
    Function.modifyLayout(opts::ReorderBlocks, opts::MinBranchClusters,
                          ShouldSplit);

    if (opts::PrintFuncStat > 0 && Function.hasLayoutChanged()) {
      ++ModifiedFuncCount;
    }
  }

  if (opts::PrintFuncStat > 0) {
    raw_ostream &OS = outs();
    // Copy all the values into vector in order to sort them
    std::map<uint64_t, BinaryFunction &> ScoreMap;
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

void FixupBranches::runOnFunctions(
  BinaryContext &BC,
  std::map<uint64_t, BinaryFunction> &BFs,
  std::set<uint64_t> &) {
  for (auto &It : BFs) {
    auto &Function = It.second;
    if (shouldOptimize(Function)) {
      Function.fixBranches();
    }
  }
}

void FinalizeFunctions::runOnFunctions(
  BinaryContext &BC,
  std::map<uint64_t, BinaryFunction> &BFs,
  std::set<uint64_t> &
) {
  for (auto &It : BFs) {
    auto &Function = It.second;
    const auto ShouldOptimize = shouldOptimize(Function);

    // Always fix functions in relocation mode.
    if (!opts::Relocs && !ShouldOptimize)
      continue;

    // Fix the CFI state.
    if (ShouldOptimize && !Function.fixCFIState()) {
      if (opts::Relocs) {
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
        if (CondBranch &&
            BC.MIA->getTargetSymbol(*CondBranch) == BB.getLabel()) {
          BC.MIA->replaceBranchTarget(*CondBranch, Succ->getLabel(), Ctx);
        } else if (UncondBranch &&
                   BC.MIA->getTargetSymbol(*UncondBranch) == BB.getLabel()) {
          BC.MIA->replaceBranchTarget(*UncondBranch, Succ->getLabel(), Ctx);
        }
      } else {
        // Succ will be null in the tail call case.  In this case we
        // need to explicitly add a tail call instruction.
        auto *Branch = Pred->getLastNonPseudoInstr();
        if (Branch && BC.MIA->isUnconditionalBranch(*Branch)) {
          assert(BC.MIA->getTargetSymbol(*Branch) == BB.getLabel());
          Pred->removeSuccessor(&BB);
          Pred->eraseInstruction(Branch);
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
    const bool IsTailCall = BC.MIA->isTailCall(*Inst);

    if (!BC.MIA->isUnconditionalBranch(*Inst) && !IsTailCall)
      continue;

    // If we operate after SCTC make sure it's not a conditional tail call.
    if (IsTailCall && BC.MIA->isConditionalBranch(*Inst))
      continue;

    const auto *SuccSym = BC.MIA->getTargetSymbol(*Inst);
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
        assert(Function.validateCFG());
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
    const bool DirectionFlag) {
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
  BF.markUnreachable();

  auto &MIA = BC.MIA;
  uint64_t NumLocalCTCCandidates = 0;
  uint64_t NumLocalCTCs = 0;
  uint64_t LocalCTCTakenCount = 0;
  uint64_t LocalCTCExecCount = 0;
  std::vector<std::tuple<BinaryBasicBlock *,
                         BinaryBasicBlock *,
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
    if (!MIA->isTailCall(*Instr) || BC.MIA->isConditionalBranch(*Instr))
      continue;

    auto *CalleeSymbol = MIA->getTargetSymbol(*Instr);
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

      if (CondSucc != BB) {
        // Patch the new target address into the conditional branch.
        MIA->reverseBranchCondition(*CondBranch, CalleeSymbol, BC.Ctx.get());
        // Since we reversed the condition on the branch we need to change
        // the target for the unconditional branch or add a unconditional
        // branch to the old target.  This has to be done manually since
        // fixupBranches is not called after SCTC.
        NeedsUncondBranch.emplace_back(std::make_tuple(BB, PredBB, CondSucc));
        // Swap branch statistics after swapping the branch targets.
        auto BI = PredBB->branch_info_begin();
        std::swap(*BI, *(BI + 1));
      } else {
        // Change destination of the conditional branch.
        MIA->replaceBranchTarget(*CondBranch, CalleeSymbol, BC.Ctx.get());
      }
      const uint64_t CTCTakenFreq = PredBB->getBranchInfo(true).Count ==
                                            BinaryBasicBlock::COUNT_NO_PROFILE
                                        ? 0
                                        : PredBB->getBranchInfo(true).Count;
      // Annotate it, so "isCall" returns true for this jcc
      MIA->addAnnotation(BC.Ctx.get(), *CondBranch, "IsCTC", true);
      // Add info abount the conditional tail call frequency, otherwise this
      // info will be lost when we delete the associated BranchInfo entry
      BC.MIA->addAnnotation(BC.Ctx.get(), *CondBranch, "CTCTakenFreq",
                            CTCTakenFreq);

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
    auto *BB = std::get<0>(Entry);
    auto *PredBB = std::get<1>(Entry);
    auto *CondSucc = std::get<2>(Entry);

    const MCSymbol *TBB = nullptr;
    const MCSymbol *FBB = nullptr;
    MCInst *CondBranch = nullptr;
    MCInst *UncondBranch = nullptr;
    PredBB->analyzeBranch(TBB, FBB, CondBranch, UncondBranch);

    // Only add a new branch if the target is not the fall-through.
    if (BF.getBasicBlockAfter(BB) != CondSucc || isValid(BB) ||
        PredBB->isCold() != CondSucc->isCold()) {
      if (UncondBranch) {
        MIA->replaceBranchTarget(*UncondBranch,
                                 CondSucc->getLabel(),
                                 BC.Ctx.get());
      } else {
        MCInst Branch;
        auto Result = MIA->createUncondBranch(Branch,
                                              CondSucc->getLabel(),
                                              BC.Ctx.get());
        (void)Result;
        assert(Result);
        PredBB->addInstruction(Branch);
      }
    } else if (UncondBranch) {
      PredBB->eraseInstruction(UncondBranch);
    }
  }

  if (NumLocalCTCs > 0) {
    NumDoubleJumps += fixDoubleJumps(BC, BF, true);
    // Clean-up unreachable tail-call blocks.
    const auto Stats = BF.eraseInvalidBBs();
    DeletedBlocks += Stats.first;
    DeletedBytes += Stats.second;
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

void SimplifyConditionalTailCalls::runOnFunctions(
  BinaryContext &BC,
  std::map<uint64_t, BinaryFunction> &BFs,
  std::set<uint64_t> &
) {
  for (auto &It : BFs) {
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

void Peepholes::shortenInstructions(BinaryContext &BC,
                                    BinaryFunction &Function) {
  for (auto &BB : Function) {
    for (auto &Inst : BB) {
      BC.MIA->shortenInstruction(Inst);
    }
  }
}

void Peepholes::addTailcallTraps(BinaryContext &BC,
                                 BinaryFunction &Function) {
  for (auto &BB : Function) {
    auto *Inst = BB.getLastNonPseudoInstr();
    if (Inst && BC.MIA->isTailCall(*Inst) && BC.MIA->isIndirectBranch(*Inst)) {
      MCInst Trap;
      if (BC.MIA->createTrap(Trap)) {
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

void Peepholes::runOnFunctions(BinaryContext &BC,
                               std::map<uint64_t, BinaryFunction> &BFs,
                               std::set<uint64_t> &LargeFunctions) {
  for (auto &It : BFs) {
    auto &Function = It.second;
    if (shouldOptimize(Function)) {
      shortenInstructions(BC, Function);
      NumDoubleJumps += fixDoubleJumps(BC, Function, false);
      addTailcallTraps(BC, Function);
      removeUselessCondBranches(BC, Function);
    }
  }
  outs() << "BOLT-INFO: Peephole: " << NumDoubleJumps
         << " double jumps patched.\n"
         << "BOLT-INFO: Peephole: " << TailCallTraps
         << " tail call traps inserted.\n"
         << "BOLT-INFO: Peephole: " << NumUselessCondBranches
         << " useless conditional branches removed.\n";
}

bool SimplifyRODataLoads::simplifyRODataLoads(
    BinaryContext &BC, BinaryFunction &BF) {
  auto &MIA = BC.MIA;

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

      if (MIA->hasRIPOperand(Inst)) {
        // Try to find the symbol that corresponds to the RIP-relative operand.
        auto DispOpI = MIA->getMemOperandDisp(Inst);
        assert(DispOpI != Inst.end() && "expected RIP-relative displacement");
        assert(DispOpI->isExpr() &&
              "found RIP-relative with non-symbolic displacement");

        // Get displacement symbol.
        const MCSymbolRefExpr *DisplExpr;
        if (!(DisplExpr = dyn_cast<MCSymbolRefExpr>(DispOpI->getExpr())))
          continue;
        const MCSymbol &DisplSymbol = DisplExpr->getSymbol();

        // Look up the symbol address in the global symbols map of the binary
        // context object.
        auto GI = BC.GlobalSymbols.find(DisplSymbol.getName());
        if (GI == BC.GlobalSymbols.end())
          continue;
        TargetAddress = GI->second;
      } else if (!MIA->evaluateMemOperandTarget(Inst, TargetAddress)) {
        continue;
      }

      // Get the contents of the section containing the target address of the
      // memory operand. We are only interested in read-only sections.
      ErrorOr<SectionRef> DataSectionOrErr =
        BC.getSectionForAddress(TargetAddress);
      if (!DataSectionOrErr)
        continue;
      SectionRef DataSection = DataSectionOrErr.get();
      if (!DataSection.isReadOnly())
        continue;
      uint32_t Offset = TargetAddress - DataSection.getAddress();
      StringRef ConstantData;
      if (std::error_code EC = DataSection.getContents(ConstantData)) {
        errs() << "BOLT-ERROR: 'cannot get section contents': "
               << EC.message() << ".\n";
        exit(1);
      }

      ++NumLocalLoadsFound;
      if (BB->hasProfile())
        NumDynamicLocalLoadsFound += BB->getExecutionCount();

      if (MIA->replaceMemOperandWithImm(Inst, ConstantData, Offset)) {
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

void SimplifyRODataLoads::runOnFunctions(
  BinaryContext &BC,
  std::map<uint64_t, BinaryFunction> &BFs,
  std::set<uint64_t> &
) {
  for (auto &It : BFs) {
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

void IdenticalCodeFolding::runOnFunctions(BinaryContext &BC,
                                        std::map<uint64_t, BinaryFunction> &BFs,
                                        std::set<uint64_t> &) {
  const auto OriginalFunctionCount = BFs.size();
  uint64_t NumFunctionsFolded = 0;
  uint64_t NumJTFunctionsFolded = 0;
  uint64_t BytesSavedEstimate = 0;
  uint64_t CallsSavedEstimate = 0;
  static bool UseDFS = opts::ICFUseDFS;

  // This hash table is used to identify identical functions. It maps
  // a function to a bucket of functions identical to it.
  struct KeyHash {
    std::size_t operator()(const BinaryFunction *F) const {
      return F->hash(/*Recompute=*/false);
    }
  };
  struct KeyCongruent {
    bool operator()(const BinaryFunction *A, const BinaryFunction *B) const {
      return A->isIdenticalWith(*B, /*IgnoreSymbols=*/true, /*UseDFS=*/UseDFS);
    }
  };
  struct KeyEqual {
    bool operator()(const BinaryFunction *A, const BinaryFunction *B) const {
      return A->isIdenticalWith(*B, /*IgnoreSymbols=*/false, /*UseDFS=*/UseDFS);
    }
  };

  // Create buckets with congruent functions - functions that potentially could
  // be folded.
  std::unordered_map<BinaryFunction *, std::set<BinaryFunction *>,
                     KeyHash, KeyCongruent> CongruentBuckets;
  for (auto &BFI : BFs) {
    auto &BF = BFI.second;
    if (!shouldOptimize(BF) || BF.isFolded())
      continue;

    // Make sure indices are in-order.
    BF.updateLayoutIndices();

    // Pre-compute hash before pushing into hashtable.
    BF.hash(/*Recompute=*/true, /*UseDFS*/UseDFS);

    CongruentBuckets[&BF].emplace(&BF);
  }

  // We repeat the pass until no new modifications happen.
  unsigned Iteration = 1;
  uint64_t NumFoldedLastIteration;
  do {
    NumFoldedLastIteration = 0;

    DEBUG(dbgs() << "BOLT-DEBUG: ICF iteration " << Iteration << "...\n");

    for (auto &CBI : CongruentBuckets) {
      auto &Candidates = CBI.second;
      if (Candidates.size() < 2)
        continue;

      // Identical functions go into the same bucket.
      std::unordered_map<BinaryFunction *, std::vector<BinaryFunction *>,
                         KeyHash, KeyEqual> IdenticalBuckets;
      for (auto *BF : Candidates) {
        IdenticalBuckets[BF].emplace_back(BF);
      }

      for (auto &IBI : IdenticalBuckets) {
        // Functions identified as identical.
        auto &Twins = IBI.second;
        if (Twins.size() < 2)
          continue;

        // Fold functions. Keep the order consistent across invocations with
        // different options.
        std::stable_sort(Twins.begin(), Twins.end(),
            [](const BinaryFunction *A, const BinaryFunction *B) {
              return A->getFunctionNumber() < B->getFunctionNumber();
            });

        BinaryFunction *ParentBF = Twins[0];
        for (unsigned i = 1; i < Twins.size(); ++i) {
          auto *ChildBF = Twins[i];
          DEBUG(dbgs() << "BOLT-DEBUG: folding " << *ChildBF << " into "
                       << *ParentBF << '\n');

          // Remove child function from the list of candidates.
          auto FI = Candidates.find(ChildBF);
          assert(FI != Candidates.end() &&
                 "function expected to be in the set");
          Candidates.erase(FI);

          // Fold the function and remove from the list of processed functions.
          BytesSavedEstimate += ChildBF->getSize();
          CallsSavedEstimate += std::min(ChildBF->getKnownExecutionCount(),
                                         ParentBF->getKnownExecutionCount());
          BC.foldFunction(*ChildBF, *ParentBF, BFs);

          ++NumFoldedLastIteration;

          if (ParentBF->hasJumpTables())
            ++NumJTFunctionsFolded;
        }
      }

    }
    NumFunctionsFolded += NumFoldedLastIteration;
    ++Iteration;

  } while (NumFoldedLastIteration > 0);

  DEBUG(
    // Print functions that are congruent but not identical.
    for (auto &CBI : CongruentBuckets) {
      auto &Candidates = CBI.second;
      if (Candidates.size() < 2)
        continue;
      dbgs() << "BOLT-DEBUG: the following " << Candidates.size()
             << " functions (each of size " << (*Candidates.begin())->getSize()
             << " bytes) are congruent but not identical:\n";
      for (auto *BF : Candidates) {
        dbgs() << "  " << *BF;
        if (BF->getKnownExecutionCount()) {
          dbgs() << " (executed " << BF->getKnownExecutionCount() << " times)";
        }
        dbgs() << '\n';
      }
    }
  );

  if (NumFunctionsFolded) {
    outs() << "BOLT-INFO: ICF folded " << NumFunctionsFolded
           << " out of " << OriginalFunctionCount << " functions in "
           << Iteration << " passes. "
           << NumJTFunctionsFolded << " functions had jump tables.\n"
           << "BOLT-INFO: Removing all identical functions will save "
           << format("%.2lf", (double) BytesSavedEstimate / 1024)
           << " KB of code space. Folded functions were called "
           << CallsSavedEstimate << " times based on profile.\n";
  }
}

void PrintSortedBy::runOnFunctions(
  BinaryContext &,
  std::map<uint64_t, BinaryFunction> &BFs,
  std::set<uint64_t> &
) {
  if (!opts::PrintSortedBy.empty() &&
      std::find(opts::PrintSortedBy.begin(),
                opts::PrintSortedBy.end(),
                DynoStats::FIRST_DYNO_STAT) == opts::PrintSortedBy.end()) {

    std::vector<const BinaryFunction *> Functions;
    std::map<const BinaryFunction *, DynoStats> Stats;

    for (const auto &BFI : BFs) {
      const auto &BF = BFI.second;
      if (shouldOptimize(BF) && BF.hasValidProfile()) {
        Functions.push_back(&BF);
        Stats.emplace(&BF, BF.getDynoStats());
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
    for (unsigned i = 0; i < 100 && SFI != Functions.end(); ++SFI, ++i) {
      const auto Stats = (*SFI)->getDynoStats();
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
}

void InstructionLowering::runOnFunctions(
    BinaryContext &BC,
    std::map<uint64_t, BinaryFunction> &BFs,
    std::set<uint64_t> &LargeFunctions) {
  for (auto &BFI : BFs) {
    for (auto &BB : BFI.second) {
      for (auto &Instruction : BB) {
        BC.MIA->lowerTailCall(Instruction);
      }
    }
  }
}

void StripRepRet::runOnFunctions(
    BinaryContext &BC,
    std::map<uint64_t, BinaryFunction> &BFs,
    std::set<uint64_t> &LargeFunctions) {
  uint64_t NumPrefixesRemoved = 0;
  uint64_t NumBytesSaved = 0;
  for (auto &BFI : BFs) {
    for (auto &BB : BFI.second) {
      auto LastInstRIter = BB.getLastNonPseudo();
      if (LastInstRIter == BB.rend() ||
          !BC.MIA->isReturn(*LastInstRIter))
        continue;

      auto NextToLastInstRIter = std::next(LastInstRIter);
      if (NextToLastInstRIter == BB.rend() ||
          !BC.MIA->isPrefix(*NextToLastInstRIter))
        continue;

      BB.eraseInstruction(std::next(NextToLastInstRIter).base());

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

} // namespace bolt
} // namespace llvm
