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
#include "HFSort.h"
#include "llvm/Support/Options.h"

#include <fstream>

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
extern cl::opt<uint32_t> RandomSeed;
extern cl::opt<bool> Relocs;
extern cl::opt<bolt::BinaryFunction::SplittingType> SplitFunctions;
extern bool shouldProcess(const bolt::BinaryFunction &Function);
extern size_t padFunction(const bolt::BinaryFunction &Function);

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

static cl::opt<std::string>
FunctionOrderFile("function-order",
  cl::desc("file containing an ordered list of functions to use for function "
           "reordering"),
  cl::cat(BoltOptCategory));

static cl::opt<bool>
ICF("icf",
  cl::desc("fold functions with identical code"),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
ICFUseDFS("icf-dfs",
  cl::desc("use DFS ordering when using -icf option"),
  cl::ReallyHidden,
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::list<std::string>
ICPFuncsList("icp-funcs",
  cl::CommaSeparated,
  cl::desc("list of functions to enable ICP for"),
  cl::value_desc("func1,func2,func3,..."),
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
ICPOldCodeSequence("icp-old-code-sequence",
  cl::desc("use old code sequence for promoted calls"),
  cl::init(false),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<unsigned>
IndirectCallPromotionMispredictThreshold(
  "indirect-call-promotion-mispredict-threshold",
  cl::desc("misprediction threshold for skipping ICP on an "
    "indirect call"),
  cl::init(2),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<unsigned>
IndirectCallPromotionThreshold("indirect-call-promotion-threshold",
  cl::desc("threshold for optimizing a frequently taken indirect call"),
  cl::init(90),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<unsigned>
IndirectCallPromotionTopN("indirect-call-promotion-topn",
  cl::desc("number of targets to consider when doing indirect "
           "call promotion"),
  cl::init(1),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
IndirectCallPromotionUseMispredicts("indirect-call-promotion-use-mispredicts",
  cl::desc("use misprediction frequency for determining whether or not ICP "
           "should be applied at a callsite.  The "
           "-indirect-call-promotion-mispredict-threshold value will be used "
           "by this heuristic"),
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

cl::opt<bolt::BinaryFunction::ReorderType>
ReorderFunctions("reorder-functions",
  cl::desc("reorder and cluster functions (works only with relocations)"),
  cl::init(bolt::BinaryFunction::RT_NONE),
  cl::values(clEnumValN(bolt::BinaryFunction::RT_NONE,
      "none",
      "do not reorder functions"),
    clEnumValN(bolt::BinaryFunction::RT_EXEC_COUNT,
      "exec-count",
      "order by execution count"),
    clEnumValN(bolt::BinaryFunction::RT_HFSORT,
      "hfsort",
      "use hfsort algorithm"),
    clEnumValN(bolt::BinaryFunction::RT_HFSORT_PLUS,
      "hfsort+",
      "use hfsort+ algorithm"),
    clEnumValN(bolt::BinaryFunction::RT_PETTIS_HANSEN,
      "pettis-hansen",
      "use Pettis-Hansen algorithm"),
    clEnumValN(bolt::BinaryFunction::RT_RANDOM,
      "random",
      "reorder functions randomly"),
    clEnumValN(bolt::BinaryFunction::RT_USER,
      "user",
      "use function order specified by -function-order"),
    clEnumValEnd),
  cl::cat(BoltOptCategory));

static cl::opt<bool>
ReorderFunctionsUseHotSize("reorder-functions-use-hot-size",
  cl::desc("use a function's hot size when doing clustering"),
  cl::init(true),
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
  cl::init(SctcHeuristic),
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

static cl::opt<bool>
UseEdgeCounts("use-edge-counts",
  cl::desc("use edge count data when doing clustering"),
  cl::init(true),
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
    if (Count) {
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
  for (auto &It : BFs) {
    auto &Function = It.second;

    if (!shouldOptimize(Function))
      continue;

    if (opts::ReorderBlocks != BinaryFunction::LT_NONE) {
      bool ShouldSplit =
        (opts::SplitFunctions == BinaryFunction::ST_ALL) ||
        (opts::SplitFunctions == BinaryFunction::ST_EH &&
         Function.hasEHRanges()) ||
        (LargeFunctions.find(It.first) != LargeFunctions.end());
      Function.modifyLayout(opts::ReorderBlocks, opts::MinBranchClusters,
                            ShouldSplit);
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

    // Always fix functions in relocation mode.
    if (!opts::Relocs && !shouldOptimize(Function))
      continue;

    // Fix the CFI state.
    if (shouldOptimize(Function) && !Function.fixCFIState()) {
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

bool
SimplifyConditionalTailCalls::shouldRewriteBranch(const BinaryBasicBlock *PredBB,
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

  for (auto *BB : BF.layout()) {
    // Locate BB with a single direct tail-call instruction.
    if (BB->getNumNonPseudos() != 1)
      continue;

    auto *Instr = BB->getFirstNonPseudoInstr();
    if (!MIA->isTailCall(*Instr))
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

      // analyzeBranch can fail due to unusual branch instructions, e.g. jrcxz
      if (!Result) {
        DEBUG(dbgs() << "analyzeBranch failed in SCTC in block:\n";
              PredBB->dump());
        continue;
      }

      assert(Result && "internal error analyzing conditional branch");
      assert(CondBranch && "conditional branch expected");

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
        if (UncondBranch) {
          MIA->replaceBranchTarget(*UncondBranch,
                                   CondSucc->getLabel(),
                                   BC.Ctx.get());
        } else {
          MCInst Branch;
          auto Result = MIA->createUncondBranch(Branch,
                                                CondSucc->getLabel(),
                                                BC.Ctx.get());
          assert(Result);
          PredBB->addInstruction(Branch);
        }
        // Swap branch statistics after swapping the branch targets.
        auto BI = PredBB->branch_info_begin();
        std::swap(*BI, *(BI + 1));
      } else {
        // Change destination of the unconditional branch.
        MIA->replaceBranchTarget(*CondBranch, CalleeSymbol, BC.Ctx.get());
      }

      // Remove the unused successor which may be eliminated later
      // if there are no other users.
      PredBB->removeSuccessor(BB);

      ++NumLocalCTCs;
    }

    // Remove the block from CFG if all predecessors were removed.
    BB->markValid(BB->pred_size() != 0 ||
                  BB->isLandingPad() ||
                  BB->isEntryPoint());
  }

  if (NumLocalCTCs > 0) {
    // Clean-up unreachable tail-call blocks.
    BF.eraseInvalidBBs();
  }

  DEBUG(dbgs() << "BOLT: created " << NumLocalCTCs
          << " conditional tail calls from a total of " << NumLocalCTCCandidates
          << " candidates in function " << BF << "\n";);

  NumTailCallsPatched += NumLocalCTCs;
  NumCandidateTailCalls += NumLocalCTCCandidates;

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

    // Fix tail calls to reduce branch mispredictions.
    if (fixTailCalls(BC, Function)) {
      Modified.insert(&Function);
    }
  }

  outs() << "BOLT-INFO: SCTC: patched " << NumTailCallsPatched
         << " tail calls (" << NumOrigForwardBranches << " forward)"
         << " tail calls (" << NumOrigBackwardBranches << " backward)"
         << " from a total of " << NumCandidateTailCalls << "\n";
}

void Peepholes::shortenInstructions(BinaryContext &BC,
                                    BinaryFunction &Function) {
  for (auto &BB : Function) {
    for (auto &Inst : BB) {
      BC.MIA->shortenInstruction(Inst);
    }
  }
}

void debugDump(BinaryFunction *BF) {
  BF->dump();
}

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
void Peepholes::fixDoubleJumps(BinaryContext &BC,
                               BinaryFunction &Function) {
  for (auto &BB : Function) {
    auto checkAndPatch = [&](BinaryBasicBlock *Pred,
                             BinaryBasicBlock *Succ,
                             const MCSymbol *SuccSym) {
      // Ignore infinite loop jumps or fallthrough tail jumps.
      if (Pred == Succ || Succ == &BB)
        return;

      if (Succ) {
        Pred->replaceSuccessor(&BB, Succ);
      } else {
        // Succ will be null in the tail call case.  In this case we
        // need to explicitly add a tail call instruction.
        auto *Branch = Pred->getLastNonPseudoInstr();
        if (Branch && BC.MIA->isUnconditionalBranch(*Branch)) {
          Pred->removeSuccessor(&BB);
          Pred->eraseInstruction(Branch);
          Pred->addTailCallInstruction(SuccSym);
        } else {
          return;
        }
      }

      ++NumDoubleJumps;
      DEBUG(dbgs() << "Removed double jump in " << Function << " from "
                   << Pred->getName() << " -> " << BB.getName() << " to "
                   << Pred->getName() << " -> " << SuccSym->getName()
                   << (!Succ ? " (tail)\n" : "\n"));
    };

    if (BB.getNumNonPseudos() != 1 || BB.isLandingPad())
      continue;

    auto *Inst = BB.getFirstNonPseudoInstr();
    const bool IsTailCall = BC.MIA->isTailCall(*Inst);

    if (!BC.MIA->isUnconditionalBranch(*Inst) && !IsTailCall)
      continue;

    const auto *SuccSym = BC.MIA->getTargetSymbol(*Inst);
    auto *Succ = BB.getSuccessor();

    if ((!Succ || &BB == Succ) && !IsTailCall)
      continue;

    std::vector<BinaryBasicBlock *> Preds{BB.pred_begin(), BB.pred_end()};

    for (auto *Pred : Preds) {
      if (Pred->isLandingPad())
        continue;

      if (Pred->getSuccessor() == &BB ||
          (Pred->getConditionalSuccessor(true) == &BB && !IsTailCall) ||
          Pred->getConditionalSuccessor(false) == &BB) {
        checkAndPatch(Pred, Succ, SuccSym);
      }
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

void Peepholes::runOnFunctions(BinaryContext &BC,
                               std::map<uint64_t, BinaryFunction> &BFs,
                               std::set<uint64_t> &LargeFunctions) {
  for (auto &It : BFs) {
    auto &Function = It.second;
    if (shouldOptimize(Function)) {
      shortenInstructions(BC, Function);
      fixDoubleJumps(BC, Function);
      addTailcallTraps(BC, Function);
    }
  }
  outs() << "BOLT-INFO: Peephole: " << NumDoubleJumps << " double jumps patched.\n";
  outs() << "BOLT-INFO: Peephole: " << TailCallTraps << " tail call traps inserted.\n";
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
  if (!opts::ICF)
    return;

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

// Get list of targets for a given call sorted by most frequently
// called first.
std::vector<BranchInfo> IndirectCallPromotion::getCallTargets(
  BinaryContext &BC,
  const FuncBranchData &BranchData,
  const MCInst &Inst
) const {
  auto Offset = BC.MIA->getAnnotationAs<uint64_t>(Inst, "IndirectBranchData");
  auto Branches = BranchData.getBranchRange(Offset);
  std::vector<BranchInfo> Targets(Branches.begin(), Branches.end());

  // Sort by most commonly called targets.
  std::sort(Targets.begin(), Targets.end(),
            [](const BranchInfo &A, const BranchInfo &B) {
              return A.Branches > B.Branches;
            });

  // Remove non-symbol targets
  auto Last = std::remove_if(Targets.begin(),
                             Targets.end(),
                             [](const BranchInfo &BI) {
                               return !BI.To.IsSymbol;
                             });
  Targets.erase(Last, Targets.end());

  return Targets;
}

std::vector<std::pair<MCSymbol *, uint64_t>>
IndirectCallPromotion::findCallTargetSymbols(
  BinaryContext &BC,
  const std::vector<BranchInfo> &Targets,
  const size_t N
) const {
  std::vector<std::pair<MCSymbol *, uint64_t>> SymTargets;

  for (size_t I = 0; I < N; ++I) {
    assert(Targets[I].To.IsSymbol && "All ICP targets must be symbols.");
    auto Itr = BC.GlobalSymbols.find(Targets[I].To.Name);
    if (Itr == BC.GlobalSymbols.end()) {
      // punt if we can't find a symbol.
      break;
    }
    MCSymbol* Symbol = BC.getOrCreateGlobalSymbol(Itr->second, "FUNCat");
    assert(Symbol && "All ICP targets must be known symbols.");
    SymTargets.push_back(std::make_pair(Symbol, 0));
  }

  return SymTargets;
}

std::vector<std::unique_ptr<BinaryBasicBlock>>
IndirectCallPromotion::rewriteCall(BinaryContext &BC,
                                   BinaryFunction &Function,
                                   BinaryBasicBlock *IndCallBlock,
                                   const MCInst &CallInst,
                                   MCInstrAnalysis::ICPdata &&ICPcode) const {
  // Create new basic blocks with correct code in each one first.
  std::vector<std::unique_ptr<BinaryBasicBlock>> NewBBs;
  const bool IsTailCall = BC.MIA->isTailCall(CallInst);

  // Move instructions from the tail of the original call block
  // to the merge block.

  // Remember any pseudo instructions following a tail call.  These
  // must be preserved and moved to the original block.
  std::vector<MCInst> TailInsts;
  const auto *TailInst= &CallInst;
  if (IsTailCall) {
    while (TailInst + 1 < &(*IndCallBlock->end()) &&
           BC.MII->get((TailInst + 1)->getOpcode()).isPseudo()) {
      TailInsts.push_back(*++TailInst);
    }
  }

  auto MovedInst = IndCallBlock->splitInstructions(&CallInst);

  IndCallBlock->replaceInstruction(&CallInst, ICPcode.front().second);
  IndCallBlock->addInstructions(TailInsts.begin(), TailInsts.end());

  for (auto Itr = ICPcode.begin() + 1; Itr != ICPcode.end(); ++Itr) {
    auto &Sym = Itr->first;
    auto &Insts = Itr->second;
    assert(Sym);
    auto TBB = Function.createBasicBlock(0, Sym);
    for (auto &Inst : Insts) { // sanitize new instructions.
      if (BC.MIA->isCall(Inst))
        BC.MIA->removeAnnotation(Inst, "IndirectBranchData");
    }
    TBB->addInstructions(Insts.begin(), Insts.end());
    NewBBs.emplace_back(std::move(TBB));
  }

  // Move tail of instructions from after the original call to
  // the merge block.
  if (!IsTailCall) {
    NewBBs.back()->addInstructions(MovedInst.begin(), MovedInst.end());
  }

  return NewBBs;
}

BinaryBasicBlock *IndirectCallPromotion::fixCFG(
  BinaryContext &BC,
  BinaryFunction &Function,
  BinaryBasicBlock *IndCallBlock,
  const bool IsTailCall,
  IndirectCallPromotion::BasicBlocksVector &&NewBBs,
  const std::vector<BranchInfo> &Targets
) const {
  BinaryBasicBlock *MergeBlock = !IsTailCall ? NewBBs.back().get() : nullptr;
  assert(NewBBs.size() >= 2);
  assert(NewBBs.size() % 2 == 1 || IndCallBlock->succ_empty());
  assert(NewBBs.size() % 2 == 1 || IsTailCall);
  using BinaryBranchInfo = BinaryBasicBlock::BinaryBranchInfo;

  if (MergeBlock) {
    std::vector<BinaryBasicBlock*> OldSucc(IndCallBlock->successors().begin(),
                                           IndCallBlock->successors().end());
    std::vector<BinaryBranchInfo> BranchInfo(IndCallBlock->branch_info_begin(),
                                             IndCallBlock->branch_info_end());

    // Remove all successors from block doing the indirect call.
    IndCallBlock->removeSuccessors(OldSucc.begin(), OldSucc.end());
    assert(IndCallBlock->succ_empty());

    // Move them to the merge block.
    MergeBlock->addSuccessors(OldSucc.begin(),
                              OldSucc.end(),
                              BranchInfo.begin(),
                              BranchInfo.end());

    // Update the execution count on the MergeBlock.
    MergeBlock->setExecutionCount(IndCallBlock->getExecutionCount());
  }

  // Scale indirect call counts to the execution count of the original
  // basic block containing the indirect call.
  uint64_t TotalIndirectBranches = 0;
  uint64_t TotalIndirectMispreds = 0;
  for (const auto &BI : Targets) {
    TotalIndirectBranches += BI.Branches;
    TotalIndirectMispreds += BI.Mispreds;
  }

  uint64_t TotalCount = 0;
  uint64_t TotalMispreds = 0;

  if (Function.hasValidProfile()) {
    TotalCount = IndCallBlock->getExecutionCount();
    TotalMispreds =
      TotalCount * ((double)TotalIndirectMispreds / TotalIndirectBranches);
    assert(TotalCount != BinaryBasicBlock::COUNT_NO_PROFILE);
  }

  // New BinaryBranchInfo scaled to the execution count of the original BB.
  std::vector<BinaryBranchInfo> BBI;
  for (auto Itr = Targets.begin(); Itr != Targets.end(); ++Itr) {
    BBI.push_back(
      BinaryBranchInfo{
        uint64_t(TotalCount * ((double)Itr->Branches / TotalIndirectBranches)),
        uint64_t(TotalMispreds * ((double)Itr->Mispreds / TotalIndirectMispreds))
      }
    );
  }
  auto BI = BBI.begin();
  auto updateCurrentBranchInfo = [&]{
    assert(BI < BBI.end());
    TotalCount -= BI->Count;
    TotalMispreds -= BI->MispredictedCount;
    ++BI;
  };

  // Fix up successors and execution counts.
  updateCurrentBranchInfo();
  IndCallBlock->addSuccessor(NewBBs[1].get(), TotalCount); // uncond branch
  IndCallBlock->addSuccessor(NewBBs[0].get(), BBI[0]); // conditional branch

  size_t Adj = 1 + (!IsTailCall ? 1 : 0);
  for (size_t I = 0; I < NewBBs.size() - Adj; ++I) {
    assert(TotalCount <= IndCallBlock->getExecutionCount() ||
           TotalCount <= uint64_t(TotalIndirectBranches));
    uint64_t ExecCount = BBI[(I+1)/2].Count;
    NewBBs[I]->setCanOutline(IndCallBlock->canOutline());
    NewBBs[I]->setIsCold(IndCallBlock->isCold());
    if (I % 2 == 0) {
      if (MergeBlock) {
        NewBBs[I]->addSuccessor(MergeBlock, BBI[(I+1)/2].Count); // uncond
      }
    } else {
      assert(I + 2 < NewBBs.size());
      updateCurrentBranchInfo();
      NewBBs[I]->addSuccessor(NewBBs[I+2].get(), TotalCount); // uncond branch
      NewBBs[I]->addSuccessor(NewBBs[I+1].get(), BBI[(I+1)/2]); // cond. branch
      ExecCount += TotalCount;
    }
    NewBBs[I]->setExecutionCount(ExecCount);
  }

  // Arrange for the MergeBlock to be the fallthrough for the first
  // promoted call block.
  if (MergeBlock) {
    MergeBlock->setCanOutline(IndCallBlock->canOutline());
    MergeBlock->setIsCold(IndCallBlock->isCold());
    std::unique_ptr<BinaryBasicBlock> MBPtr;
    std::swap(MBPtr, NewBBs.back());
    NewBBs.pop_back();
    NewBBs.emplace(NewBBs.begin() + 1, std::move(MBPtr));
    // TODO: is COUNT_FALLTHROUGH_EDGE the right thing here?
    NewBBs.back()->addSuccessor(MergeBlock, TotalCount); // uncond branch
  }

  // cold call block
  // TODO: should be able to outline/cold this block.
  NewBBs.back()->setExecutionCount(TotalCount);
  NewBBs.back()->setCanOutline(IndCallBlock->canOutline());
  NewBBs.back()->setIsCold(IndCallBlock->isCold());

  // update BB and BB layout.
  Function.insertBasicBlocks(IndCallBlock, std::move(NewBBs));
  assert(Function.validateCFG());

  return MergeBlock;
}

size_t
IndirectCallPromotion::canPromoteCallsite(const BinaryBasicBlock *BB,
                                          const MCInst &Inst,
                                          const std::vector<BranchInfo> &Targets,
                                          uint64_t NumCalls) {
  // If we have no targets (or no calls), skip this callsite.
  if (Targets.empty() || !NumCalls) {
    if (opts::Verbosity >= 1) {
      const auto InstIdx = &Inst - &(*BB->begin());
      outs() << "BOLT-INFO: ICP failed in " << *BB->getFunction() << " @ "
             << InstIdx << " in " << BB->getName()
             << ", calls = " << NumCalls
             << ", targets empty or NumCalls == 0.\n";
    }
    return 0;
  }

  const auto TrialN = std::min(size_t(opts::IndirectCallPromotionTopN),
                               Targets.size());

  if (!opts::ICPFuncsList.empty()) {
    for (auto &Name : opts::ICPFuncsList) {
      if (BB->getFunction()->hasName(Name))
        return TrialN;
    }
    return 0;
  }

  // Pick the top N targets.
  uint64_t TotalCallsTopN = 0;
  uint64_t TotalMispredictsTopN = 0;
  size_t N = 0;

  if (opts::IndirectCallPromotionUseMispredicts) {
    // Count total number of mispredictions for (at most) the top N targets.
    // We may choose a smaller N (TrialN vs. N) if the frequency threshold
    // is exceeded by fewer targets.
    double Threshold = double(opts::IndirectCallPromotionMispredictThreshold);
    for (size_t I = 0; I < TrialN && Threshold > 0; ++I, ++N) {
      const auto Frequency = (100.0 * Targets[I].Mispreds) / NumCalls;
      TotalMispredictsTopN += Targets[I].Mispreds;
      TotalNumFrequentCalls += Targets[I].Branches;
      Threshold -= Frequency;
    }

    // Compute the misprediction frequency of the top N call targets.  If this
    // frequency is greater than the threshold, we should try ICP on this callsite.
    const double TopNFrequency = (100.0 * TotalMispredictsTopN) / NumCalls;

    if (TopNFrequency == 0 ||
        TopNFrequency < opts::IndirectCallPromotionMispredictThreshold) {
      if (opts::Verbosity >= 1) {
        const auto InstIdx = &Inst - &(*BB->begin());
        outs() << "BOLT-INFO: ICP failed in " << *BB->getFunction() << " @ "
               << InstIdx << " in " << BB->getName() << ", calls = "
               << NumCalls << ", top N mis. frequency "
               << format("%.1f", TopNFrequency) << "% < "
               << opts::IndirectCallPromotionMispredictThreshold << "%\n";
      }
      return 0;
    }
  } else {
    // Count total number of calls for (at most) the top N targets.
    // We may choose a smaller N (TrialN vs. N) if the frequency threshold
    // is exceeded by fewer targets.
    double Threshold = double(opts::IndirectCallPromotionThreshold);
    for (size_t I = 0; I < TrialN && Threshold > 0; ++I, ++N) {
      const auto Frequency = (100.0 * Targets[I].Branches) / NumCalls;
      TotalCallsTopN += Targets[I].Branches;
      TotalMispredictsTopN += Targets[I].Mispreds;
      TotalNumFrequentCalls += Targets[I].Branches;
      Threshold -= Frequency;
    }

    // Compute the frequency of the top N call targets.  If this frequency
    // is greater than the threshold, we should try ICP on this callsite.
    const double TopNFrequency = (100.0 * TotalCallsTopN) / NumCalls;

    if (TopNFrequency == 0 ||
        TopNFrequency < opts::IndirectCallPromotionThreshold) {
      if (opts::Verbosity >= 1) {
        const auto InstIdx = &Inst - &(*BB->begin());
        outs() << "BOLT-INFO: ICP failed in " << *BB->getFunction() << " @ "
               << InstIdx << " in " << BB->getName() << ", calls = "
               << NumCalls << ", top N frequency "
               << format("%.1f", TopNFrequency) << "% < "
               << opts::IndirectCallPromotionThreshold << "%\n";
      }
      return 0;
    }

    // Compute the misprediction frequency of the top N call targets.  If
    // this frequency is less than the threshold, we should skip ICP at
    // this callsite.
    const double TopNMispredictFrequency =
      (100.0 * TotalMispredictsTopN) / NumCalls;

    if (TopNMispredictFrequency <
        opts::IndirectCallPromotionMispredictThreshold) {
      if (opts::Verbosity >= 1) {
        const auto InstIdx = &Inst - &(*BB->begin());
        outs() << "BOLT-INFO: ICP failed in " <<  *BB->getFunction() << " @ "
               << InstIdx << " in " << BB->getName() << ", calls = "
               << NumCalls << ", top N mispredict frequency "
               << format("%.1f", TopNMispredictFrequency) << "% < "
               << opts::IndirectCallPromotionMispredictThreshold << "%\n";
      }
      return 0;
    }
  }

  return N;
}

void
IndirectCallPromotion::printCallsiteInfo(const BinaryBasicBlock *BB,
                                         const MCInst &Inst,
                                         const std::vector<BranchInfo> &Targets,
                                         const size_t N,
                                         uint64_t NumCalls) const {
  auto &BC = BB->getFunction()->getBinaryContext();
  const auto InstIdx = &Inst - &(*BB->begin());
  bool Separator = false;

  outs() << "BOLT-INFO: ICP candidate branch info: "
         << *BB->getFunction() << " @ " << InstIdx
         << " in " << BB->getName()
         << " -> calls = " << NumCalls
         << (BC.MIA->isTailCall(Inst) ? " (tail)" : "");
  for (size_t I = 0; I < N; I++) {
    const auto Frequency = 100.0 * Targets[I].Branches / NumCalls;
    const auto MisFrequency = 100.0 * Targets[I].Mispreds / NumCalls;
    outs() << (Separator ? " | " : ", ");
    Separator = true;
    outs() << Targets[I].To.Name
           << ", calls = " << Targets[I].Branches
           << ", mispreds = " << Targets[I].Mispreds
           << ", taken freq = " << format("%.1f", Frequency) << "%"
           << ", mis. freq = " << format("%.1f", MisFrequency) << "%";
  }
  outs() << "\n";

  DEBUG({
    dbgs() << "BOLT-INFO: ICP original call instruction:\n";
    BC.printInstruction(dbgs(), Inst, Targets[0].From.Offset, nullptr, true);
  });
}

void IndirectCallPromotion::runOnFunctions(
  BinaryContext &BC,
  std::map<uint64_t, BinaryFunction> &BFs,
  std::set<uint64_t> &LargeFunctions
) {
  for (auto &BFIt : BFs) {
    auto &Function = BFIt.second;

    if (!Function.isSimple() || !opts::shouldProcess(Function))
      continue;

    const auto BranchDataOrErr = BC.DR.getFuncBranchData(Function.getNames());
    if (const auto EC = BranchDataOrErr.getError()) {
      DEBUG(dbgs() << "BOLT-INFO: no branch data found for \""
                   << Function << "\"\n");
      continue;
    }
    const FuncBranchData &BranchData = BranchDataOrErr.get();
    const bool HasLayout = !Function.layout_empty();

    // Note: this is not just counting calls.
    TotalCalls += BranchData.ExecutionCount;

    // Total number of indirect calls issued from the current Function.
    // (a fraction of TotalIndirectCalls)
    uint64_t FuncTotalIndirectCalls = 0;

    std::vector<BinaryBasicBlock *> BBs;
    for (auto &BB : Function) {
      // Skip indirect calls in cold blocks.
      if (!HasLayout || !Function.isSplit() || !BB.isCold()) {
        BBs.push_back(&BB);
      }
    }

    while (!BBs.empty()) {
      auto *BB = BBs.back();
      BBs.pop_back();

      for (unsigned Idx = 0; Idx < BB->size(); ++Idx) {
        auto &Inst = BB->getInstructionAtIndex(Idx);
        const auto InstIdx = &Inst - &(*BB->begin());

        if (!BC.MIA->hasAnnotation(Inst, "IndirectBranchData"))
          continue;

        assert(BC.MIA->isCall(Inst));

        ++TotalIndirectCallsites;

        const auto Targets = getCallTargets(BC, BranchData, Inst);

        // Compute the total number of calls from this particular callsite.
        uint64_t NumCalls = 0;
        for (const auto &BInfo : Targets) {
          NumCalls += BInfo.Branches;
        }
        FuncTotalIndirectCalls += NumCalls;

        // Should this callsite be optimized?  Return the number of targets
        // to use when promoting this call.  A value of zero means to skip
        // this callsite.
        size_t N = canPromoteCallsite(BB, Inst, Targets, NumCalls);

        if (!N)
          continue;

        if (opts::Verbosity >= 1) {
          printCallsiteInfo(BB, Inst, Targets, N, NumCalls);
        }

        // Find MCSymbols or absolute addresses for each call target.
        const auto SymTargets = findCallTargetSymbols(BC, Targets, N);

        // If we can't resolve any of the target symbols, punt on this callsite.
        if (SymTargets.size() < N) {
          const auto LastTarget = SymTargets.size();
          if (opts::Verbosity >= 1) {
            outs() << "BOLT-INFO: ICP failed to find target symbol for "
                   << Targets[LastTarget].To.Name << " in "
                   << Function << " @ " << InstIdx << " in "
                   << BB->getName() << ", calls = " << NumCalls << "\n";
          }
          continue;
        }

        // Generate new promoted call code for this callsite.
        auto ICPcode =
          BC.MIA->indirectCallPromotion(Inst,
                                        SymTargets,
                                        opts::ICPOldCodeSequence,
                                        BC.Ctx.get());

        if (ICPcode.empty()) {
          if (opts::Verbosity >= 1) {
            outs() << "BOLT-INFO: ICP failed in " << Function << " @ "
                   << InstIdx << " in " << BB->getName()
                   << ", calls = " << NumCalls
                   << ", unable to generate promoted call code.\n";
          }
          continue;
        }

        DEBUG({
          auto Offset = Targets[0].From.Offset;
          dbgs() << "BOLT-INFO: ICP indirect call code:\n";
          for (const auto &entry : ICPcode) {
            const auto &Sym = entry.first;
            const auto &Insts = entry.second;
            if (Sym) dbgs() << Sym->getName() << ":\n";
            Offset = BC.printInstructions(dbgs(),
                                          Insts.begin(),
                                          Insts.end(),
                                          Offset);
          }
          dbgs() << "---------------------------------------------------\n";
        });

        // Rewrite the CFG with the newly generated ICP code.
        const bool IsTailCall = BC.MIA->isTailCall(Inst);
        auto NewBBs = rewriteCall(BC, Function, BB, Inst, std::move(ICPcode));

        // Fix the CFG after inserting the new basic blocks.
        auto MergeBlock = fixCFG(BC, Function, BB, IsTailCall,
                                 std::move(NewBBs), Targets);

        // Since the tail of the original block was split off and it may contain
        // additional indirect calls, we must add the merge block to the set of
        // blocks to process.
        if (MergeBlock) {
          BBs.push_back(MergeBlock);
        }

        if (opts::Verbosity >= 1) {
          outs() << "BOLT-INFO: ICP succeeded in "
                 << Function << " @ " << InstIdx
                 << " in " << BB->getName()
                 << " -> calls = " << NumCalls << "\n";
        }

        ++TotalOptimizedIndirectCallsites;

        Modified.insert(&Function);
      }
    }
    TotalIndirectCalls += FuncTotalIndirectCalls;
  }

  outs() << "BOLT-INFO: ICP total indirect callsites = "
         << TotalIndirectCallsites
         << "\n"
         << "BOLT-INFO: ICP total number of calls = "
         << TotalCalls
         << "\n"
         << "BOLT-INFO: ICP percentage of calls that are indirect = "
         << format("%.1f", (100.0 * TotalIndirectCalls) / TotalCalls)
         << "%\n"
         << "BOLT-INFO: ICP percentage of indirect calls that can be "
            "optimized = "
         << format("%.1f", (100.0 * TotalNumFrequentCalls) / TotalIndirectCalls)
         << "%\n"
         << "BOLT-INFO: ICP percentage of indirect calls that are optimized = "
         << format("%.1f", (100.0 * TotalOptimizedIndirectCallsites) /
                   TotalIndirectCallsites)
         << "%\n";
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

void ReorderFunctions::buildCallGraph(BinaryContext &BC,
                                      std::map<uint64_t, BinaryFunction> &BFs) {
  // Add call graph nodes.
  auto lookupNode = [&](BinaryFunction *Function) {
    auto It = FuncToTargetId.find(Function);
    if (It == FuncToTargetId.end()) {
      // It's ok to use the hot size here when the function is split.  This is
      // because emitFunctions will emit the hot part first in the order that is
      // computed by ReorderFunctions.  The cold part will be emitted with the
      // rest of the cold functions and code.
      const auto Size = opts::ReorderFunctionsUseHotSize && Function->isSplit()
        ? Function->estimateHotSize()
        : Function->estimateSize();
      const auto Id = Cg.addTarget(Size);
      assert(size_t(Id) == Funcs.size());
      Funcs.push_back(Function);
      FuncToTargetId[Function] = Id;
      // NOTE: for functions without a profile, we set the number of samples
      // to zero.  This will keep these functions from appearing in the hot
      // section.  This is a little weird because we wouldn't be trying to
      // create a node for a function unless it was the target of a call from
      // a hot block.  The alternative would be to set the count to one or
      // accumulate the number of calls from the callsite into the function
      // samples.  Results from perfomance testing seem to favor the zero
      // count though, so I'm leaving it this way for now.
      Cg.Targets[Id].Samples = Function->hasProfile() ? Function->getExecutionCount() : 0;
      assert(Funcs[Id] == Function);
      return Id;
    } else {
      return It->second;
    }
  };

  // Add call graph edges.
  uint64_t NotProcessed = 0;
  uint64_t TotalCalls = 0;
  for (auto &It : BFs) {
    auto *Function = &It.second;

    if(!shouldOptimize(*Function) || !Function->hasProfile()) {
      continue;
    }

    auto BranchDataOrErr = BC.DR.getFuncBranchData(Function->getNames());
    const auto SrcId = lookupNode(Function);
    uint64_t Offset = Function->getAddress();

    auto recordCall = [&](const MCSymbol *DestSymbol, const uint64_t Count) {
      if (auto *DstFunc = BC.getFunctionForSymbol(DestSymbol)) {
        const auto DstId = lookupNode(DstFunc);
        auto &A = Cg.incArcWeight(SrcId, DstId, Count);
        if (!opts::UseEdgeCounts) {
          A.AvgCallOffset += (Offset - DstFunc->getAddress());
        }
        DEBUG(dbgs() << "BOLT-DEBUG: Reorder functions: call " << *Function
                     << " -> " << *DstFunc << " @ " << Offset << "\n");
        return true;
      }
      return false;
    };

    for (auto *BB : Function->layout()) {
      // Don't count calls from cold blocks
      if (BB->isCold())
        continue;

      for (auto &Inst : *BB) {
        // Find call instructions and extract target symbols from each one.
        if (BC.MIA->isCall(Inst)) {
          ++TotalCalls;
          if (const auto *DstSym = BC.MIA->getTargetSymbol(Inst)) {
            // For direct calls, just use the BB execution count.
            assert(BB->hasProfile());
            const auto Count = opts::UseEdgeCounts ? BB->getExecutionCount() : 1;
            if (!recordCall(DstSym, Count))
              ++NotProcessed;
          } else if (BC.MIA->hasAnnotation(Inst, "EdgeCountData")) {
            // For indirect calls and jump tables, use branch data.
            assert(BranchDataOrErr);
            const FuncBranchData &BranchData = BranchDataOrErr.get();
            const auto DataOffset =
              BC.MIA->getAnnotationAs<uint64_t>(Inst, "EdgeCountData");

            for (const auto &BI : BranchData.getBranchRange(DataOffset)) {
              // Count each target as a separate call.
              ++TotalCalls;

              if (!BI.To.IsSymbol) {
                ++NotProcessed;
                continue;
              }

              auto Itr = BC.GlobalSymbols.find(BI.To.Name);
              if (Itr == BC.GlobalSymbols.end()) {
                ++NotProcessed;
                continue;
              }

              const auto *DstSym =
                BC.getOrCreateGlobalSymbol(Itr->second, "FUNCat");

              if (!recordCall(DstSym, opts::UseEdgeCounts ? BI.Branches : 1))
                ++NotProcessed;
            }
          }
        }

        if (!opts::UseEdgeCounts) {
          Offset += BC.computeCodeSize(&Inst, &Inst + 1);
        }
      }
    }
  }
  outs() << "BOLT-WARNING: ReorderFunctions: " << NotProcessed
         << " callsites not processed out of " << TotalCalls << "\n";

  // Normalize arc weights.
  if (!opts::UseEdgeCounts) {
    for (TargetId FuncId = 0; FuncId < Cg.Targets.size(); ++FuncId) {
      auto& Func = Cg.Targets[FuncId];
      for (auto Caller : Func.Preds) {
        auto& A = *Cg.Arcs.find(Arc(Caller, FuncId));
        A.NormalizedWeight = A.Weight / Func.Samples;
        A.AvgCallOffset /= A.Weight;
        assert(A.AvgCallOffset < Cg.Targets[Caller].Size);
      }
    }
  } else {
    for (TargetId FuncId = 0; FuncId < Cg.Targets.size(); ++FuncId) {
      auto &Func = Cg.Targets[FuncId];
      for (auto Caller : Func.Preds) {
        auto& A = *Cg.Arcs.find(Arc(Caller, FuncId));
        A.NormalizedWeight = A.Weight / Func.Samples;
      }
    }
  }
}

void ReorderFunctions::reorder(std::vector<Cluster> &&Clusters,
                               std::map<uint64_t, BinaryFunction> &BFs) {
  std::vector<uint64_t> FuncAddr(Cg.Targets.size());  // Just for computing stats
  uint64_t TotalSize = 0;
  uint32_t Index = 0;

  // Set order of hot functions based on clusters.
  for (const auto& Cluster : Clusters) {
    for (const auto FuncId : Cluster.Targets) {
      assert(Cg.Targets[FuncId].Samples > 0);
      Funcs[FuncId]->setIndex(Index++);
      FuncAddr[FuncId] = TotalSize;
      TotalSize += Cg.Targets[FuncId].Size;
    }
  }

  if (opts::Verbosity > 0 || (DebugFlag && isCurrentDebugType("hfsort"))) {
    uint64_t TotalSize   = 0;
    uint64_t CurPage     = 0;
    uint64_t Hotfuncs    = 0;
    double TotalDistance = 0;
    double TotalCalls    = 0;
    double TotalCalls64B = 0;
    double TotalCalls4KB = 0;
    double TotalCalls2MB = 0;
    dbgs() << "============== page 0 ==============\n";
    for (auto& Cluster : Clusters) {
      dbgs() <<
        format("-------- density = %.3lf (%u / %u) --------\n",
               (double) Cluster.Samples / Cluster.Size,
               Cluster.Samples, Cluster.Size);

      for (auto FuncId : Cluster.Targets) {
        if (Cg.Targets[FuncId].Samples > 0) {
          Hotfuncs++;

          dbgs() << "BOLT-INFO: hot func " << *Funcs[FuncId]
                 << " (" << Cg.Targets[FuncId].Size << ")\n";

          uint64_t Dist = 0;
          uint64_t Calls = 0;
          for (auto Dst : Cg.Targets[FuncId].Succs) {
            auto& A = *Cg.Arcs.find(Arc(FuncId, Dst));
            auto D =
              std::abs(FuncAddr[A.Dst] - (FuncAddr[FuncId] + A.AvgCallOffset));
            auto W = A.Weight;
            Calls += W;
            if (D < 64)        TotalCalls64B += W;
            if (D < 4096)      TotalCalls4KB += W;
            if (D < (2 << 20)) TotalCalls2MB += W;
            Dist += A.Weight * D;
            dbgs() << format("arc: %u [@%lu+%.1lf] -> %u [@%lu]: "
                             "weight = %.0lf, callDist = %f\n",
                             A.Src, FuncAddr[A.Src], A.AvgCallOffset,
                             A.Dst, FuncAddr[A.Dst], A.Weight, D);
          }
          TotalCalls += Calls;
          TotalDistance += Dist;
          dbgs() << format("start = %6u : avgCallDist = %lu : %s\n",
                           TotalSize,
                           Calls ? Dist / Calls : 0,
                           Funcs[FuncId]->getPrintName().c_str());
          TotalSize += Cg.Targets[FuncId].Size;
          auto NewPage = TotalSize / PageSize;
          if (NewPage != CurPage) {
            CurPage = NewPage;
            dbgs() << format("============== page %u ==============\n", CurPage);
          }
        }
      }
    }
    dbgs() << format("  Number of hot functions: %u\n"
                     "  Number of clusters: %lu\n",
                     Hotfuncs, Clusters.size())
           << format("  Final average call distance = %.1lf (%.0lf / %.0lf)\n",
                     TotalCalls ? TotalDistance / TotalCalls : 0,
                     TotalDistance, TotalCalls)
           << format("  Total Calls = %.0lf\n", TotalCalls);
    if (TotalCalls) {
      dbgs() << format("  Total Calls within 64B = %.0lf (%.2lf%%)\n",
                       TotalCalls64B, 100 * TotalCalls64B / TotalCalls)
             << format("  Total Calls within 4KB = %.0lf (%.2lf%%)\n",
                       TotalCalls4KB, 100 * TotalCalls4KB / TotalCalls)
             << format("  Total Calls within 2MB = %.0lf (%.2lf%%)\n",
                       TotalCalls2MB, 100 * TotalCalls2MB / TotalCalls);
    }
  }
}

namespace {

std::vector<std::string> readFunctionOrderFile() {
  std::vector<std::string> FunctionNames;
  std::ifstream FuncsFile(opts::FunctionOrderFile, std::ios::in);
  if (!FuncsFile) {
    errs() << "Ordered functions file \"" << opts::FunctionOrderFile
           << "\" can't be opened.\n";
    exit(1);
  }
  std::string FuncName;
  while (std::getline(FuncsFile, FuncName)) {
    FunctionNames.push_back(FuncName);
  }
  return FunctionNames;
}

}

void ReorderFunctions::runOnFunctions(BinaryContext &BC,
                                      std::map<uint64_t, BinaryFunction> &BFs,
                                      std::set<uint64_t> &LargeFunctions) {
  if (!opts::Relocs && opts::ReorderFunctions != BinaryFunction::RT_NONE) {
    errs() << "BOLT-ERROR: Function reordering only works when "
           << "relocs are enabled.\n";
    exit(1);
  }

  if (opts::ReorderFunctions != BinaryFunction::RT_NONE &&
      opts::ReorderFunctions != BinaryFunction::RT_EXEC_COUNT &&
      opts::ReorderFunctions != BinaryFunction::RT_USER) {
    buildCallGraph(BC, BFs);
  }

  std::vector<Cluster> Clusters;

  switch(opts::ReorderFunctions) {
  case BinaryFunction::RT_NONE:
    break;
  case BinaryFunction::RT_EXEC_COUNT:
    {
      std::vector<BinaryFunction *> SortedFunctions(BFs.size());
      uint32_t Index = 0;
      std::transform(BFs.begin(),
                     BFs.end(),
                     SortedFunctions.begin(),
                     [](std::pair<const uint64_t, BinaryFunction> &BFI) {
                       return &BFI.second;
                     });
      std::stable_sort(SortedFunctions.begin(), SortedFunctions.end(),
                       [&](const BinaryFunction *A, const BinaryFunction *B) {
                         if (!opts::shouldProcess(*A))
                           return false;
                         const auto PadA = opts::padFunction(*A);
                         const auto PadB = opts::padFunction(*B);
                         if (!PadA || !PadB) {
                           if (PadA)
                             return true;
                           if (PadB)
                             return false;
                         }
                         return !A->hasProfile() &&
                           (B->hasProfile() ||
                            (A->getExecutionCount() > B->getExecutionCount()));
                       });
      for (auto *BF : SortedFunctions) {
        if (BF->hasProfile())
          BF->setIndex(Index++);
      }
    }
    break;
  case BinaryFunction::RT_HFSORT:
    Clusters = clusterize(Cg);
    break;
  case BinaryFunction::RT_HFSORT_PLUS:
    Clusters = hfsortPlus(Cg);
    break;
  case BinaryFunction::RT_PETTIS_HANSEN:
    Clusters = pettisAndHansen(Cg);
    break;
  case BinaryFunction::RT_RANDOM:
    std::srand(opts::RandomSeed);
    Clusters = randomClusters(Cg);
    break;
  case BinaryFunction::RT_USER:
    {
      uint32_t Index = 0;
      for (const auto &Function : readFunctionOrderFile()) {
        std::vector<uint64_t> FuncAddrs;

        auto Itr = BC.GlobalSymbols.find(Function);
        if (Itr == BC.GlobalSymbols.end()) {
          uint32_t LocalID = 1;
          while(1) {
            // If we can't find the main symbol name, look for alternates.
            Itr = BC.GlobalSymbols.find(Function + "/" + std::to_string(LocalID));
            if (Itr != BC.GlobalSymbols.end())
              FuncAddrs.push_back(Itr->second);
            else
              break;
            LocalID++;
          }
        } else {
          FuncAddrs.push_back(Itr->second);
        }

        if (FuncAddrs.empty()) {
          errs() << "BOLT-WARNING: Reorder functions: can't find function for "
                 << Function << "\n";
          continue;
        }

        for (const auto FuncAddr : FuncAddrs) {
          const auto *FuncSym = BC.getOrCreateGlobalSymbol(FuncAddr, "FUNCat");
          assert(FuncSym);

          auto *BF = BC.getFunctionForSymbol(FuncSym);
          if (!BF) {
            errs() << "BOLT-WARNING: Reorder functions: can't find function for "
                   << Function << "\n";
            break;
          }
          if (!BF->hasValidIndex()) {
            BF->setIndex(Index++);
          }
        }
      }
    }
    break;
  }

  reorder(std::move(Clusters), BFs);
}

} // namespace bolt
} // namespace llvm
