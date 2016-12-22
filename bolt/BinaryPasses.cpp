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

extern cl::opt<unsigned> Verbosity;
extern cl::opt<bool> Relocs;
extern cl::opt<llvm::bolt::BinaryFunction::SplittingType> SplitFunctions;
extern bool shouldProcess(const bolt::BinaryFunction &Function);

static cl::list<std::string>
ForceInlineFunctions("force-inline",
                     cl::CommaSeparated,
                     cl::desc("list of functions to always consider "
                              "for inlining"),
                     cl::value_desc("func1,func2,func3,..."),
                     cl::Hidden);

static cl::opt<bool>
AggressiveInlining("aggressive-inlining",
                   cl::desc("perform aggressive inlining"),
                   cl::ZeroOrMore,
                   cl::Hidden);

static cl::opt<bolt::BinaryFunction::LayoutType>
ReorderBlocks(
    "reorder-blocks",
    cl::desc("change layout of basic blocks in a function"),
    cl::init(bolt::BinaryFunction::LT_NONE),
    cl::values(clEnumValN(bolt::BinaryFunction::LT_NONE,
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
    cl::ZeroOrMore);

static cl::opt<bool>
MinBranchClusters(
    "min-branch-clusters",
    cl::desc("use a modified clustering algorithm geared towards "
             "minimizing branches"),
    cl::ZeroOrMore,
    cl::Hidden);

static cl::list<bolt::DynoStats::Category>
PrintSortedBy(
    "print-sorted-by",
    cl::CommaSeparated,
    cl::desc("print functions sorted by order of dyno stats"),
    cl::value_desc("key1,key2,key3,..."),
    cl::values(
#define D(name, ...)                                      \
    clEnumValN(bolt::DynoStats::name,                     \
               dynoStatsOptName(bolt::DynoStats::name),   \
               dynoStatsOptDesc(bolt::DynoStats::name)),
    DYNO_STATS
#undef D
    clEnumValEnd),
    cl::ZeroOrMore);

enum DynoStatsSortOrder : char {
  Ascending,
  Descending
};

static cl::opt<DynoStatsSortOrder>
DynoStatsSortOrderOpt(
    "print-sorted-by-order",
    cl::desc("use ascending or descending order when printing "
             "functions ordered by dyno stats"),
    cl::ZeroOrMore,
    cl::init(DynoStatsSortOrder::Descending));

enum SctcModes : char {
  SctcAlways,
  SctcPreserveDirection,
  SctcHeuristic
};

static cl::opt<SctcModes>
SctcMode(
    "sctc-mode",
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
    cl::ZeroOrMore);

static cl::opt<bool>
IdenticalCodeFolding(
    "icf",
    cl::desc("fold functions with identical code"),
    cl::ZeroOrMore);

static cl::opt<bool>
UseDFSForICF(
    "icf-dfs",
    cl::desc("use DFS ordering when using -icf option"),
    cl::ReallyHidden,
    cl::ZeroOrMore);

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

  const auto *FirstInstr = BF.front().getFirstNonPseudo();
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
                   << " (executed " << (BB->getExecutionCount() ==
                       BinaryFunction::COUNT_NO_PROFILE ?
                        0 : BB->getExecutionCount())
                   << " times) in " << BF
                   << ": replacing call to " << OriginalTarget->getName()
                   << " by call to " << Target->getName()
                   << " while folding " << CallSites << " call sites\n");
      BC.MIA->replaceCallTargetOperand(Inst, Target, BC.Ctx.get());

      NumOptimizedCallSites += CallSites;
      if (BB->getExecutionCount() != BinaryFunction::COUNT_NO_PROFILE) {
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

void InlineSmallFunctions::findInliningCandidates(
    BinaryContext &BC,
    const std::map<uint64_t, BinaryFunction> &BFs) {
  for (const auto &BFIt : BFs) {
    const auto &Function = BFIt.second;
    if (!shouldOptimize(Function) || Function.size() != 1)
      continue;
    auto &BB = *Function.begin();
    const auto &LastInstruction = *BB.rbegin();
    // Check if the function is small enough and doesn't do a tail call.
    if (BB.size() > 0 &&
        BB.getNumNonPseudos() <= kMaxInstructions &&
        BC.MIA->isReturn(LastInstruction) &&
        !BC.MIA->isTailCall(LastInstruction)) {
      InliningCandidates.insert(&Function);
    }
  }

  DEBUG(dbgs() << "BOLT-DEBUG: " << InliningCandidates.size()
               << " inlineable functions.\n");
}

void InlineSmallFunctions::findInliningCandidatesAggressive(
    BinaryContext &BC,
    const std::map<uint64_t, BinaryFunction> &BFs) {
  std::set<std::string> OverwrittenFunctions = {
    "_ZN4HPHP13hash_string_iEPKcj",
    "_ZN4HPHP21hash_string_cs_unsafeEPKcj",
    "_ZN4HPHP14hash_string_csEPKcj",
    "_ZN4HPHP20hash_string_i_unsafeEPKcj",
    "_ZNK4HPHP10StringData10hashHelperEv"
  };
  for (const auto &BFIt : BFs) {
    const auto &Function = BFIt.second;
    if (!shouldOptimize(Function) ||
        OverwrittenFunctions.count(Function.getSymbol()->getName()) ||
        Function.hasEHRanges())
      continue;
    uint64_t FunctionSize = 0;
    for (const auto *BB : Function.layout()) {
      FunctionSize += BC.computeCodeSize(BB->begin(), BB->end());
    }
    assert(FunctionSize > 0 && "found empty function");
    if (FunctionSize > kMaxSize)
      continue;
    bool FoundCFI = false;
    for (const auto BB : Function.layout()) {
      for (const auto &Inst : *BB) {
        if (BC.MIA->isEHLabel(Inst) || BC.MIA->isCFI(Inst)) {
          FoundCFI = true;
          break;
        }
      }
    }
    if (!FoundCFI)
      InliningCandidates.insert(&Function);
  }

  DEBUG(dbgs() << "BOLT-DEBUG: " << InliningCandidates.size()
               << " inlineable functions.\n");
}

namespace {

/// Returns whether a function creates a stack frame for itself or not.
/// If so, we need to manipulate the stack pointer when calling this function.
/// Since we're only inlining very small functions, we return false for now, but
/// we could for instance check if the function starts with 'push ebp'.
/// TODO generalize this.
bool createsStackFrame(const BinaryBasicBlock &) {
  return false;
}

} // namespace

void InlineSmallFunctions::inlineCall(
    BinaryContext &BC,
    BinaryBasicBlock &BB,
    MCInst *CallInst,
    const BinaryBasicBlock &InlinedFunctionBB) {
  assert(BC.MIA->isCall(*CallInst) && "Can only inline a call.");
  assert(BC.MIA->isReturn(*InlinedFunctionBB.rbegin()) &&
         "Inlined function should end with a return.");

  std::vector<MCInst> InlinedInstance;

  bool ShouldAdjustStack = createsStackFrame(InlinedFunctionBB);

  // Move stack like 'call' would if needed.
  if (ShouldAdjustStack) {
    MCInst StackInc;
    BC.MIA->createStackPointerIncrement(StackInc);
    InlinedInstance.push_back(StackInc);
  }

  for (auto Instruction : InlinedFunctionBB) {
    if (BC.MIA->isReturn(Instruction)) {
      break;
    }
    if (!BC.MIA->isEHLabel(Instruction) &&
        !BC.MIA->isCFI(Instruction)) {
      InlinedInstance.push_back(Instruction);
    }
  }

  // Move stack pointer like 'ret' would.
  if (ShouldAdjustStack) {
    MCInst StackDec;
    BC.MIA->createStackPointerDecrement(StackDec);
    InlinedInstance.push_back(StackDec);
  }

  BB.replaceInstruction(CallInst, InlinedInstance);
}

std::pair<BinaryBasicBlock *, unsigned>
InlineSmallFunctions::inlineCall(
    BinaryContext &BC,
    BinaryFunction &CallerFunction,
    BinaryBasicBlock *CallerBB,
    const unsigned CallInstIndex,
    const BinaryFunction &InlinedFunction) {
  // Get the instruction to be replaced with inlined code.
  MCInst &CallInst = CallerBB->getInstructionAtIndex(CallInstIndex);
  assert(BC.MIA->isCall(CallInst) && "Can only inline a call.");

  // Point in the function after the inlined code.
  BinaryBasicBlock *AfterInlinedBB = nullptr;
  unsigned AfterInlinedIstrIndex = 0;

  // In case of a tail call we should not remove any ret instructions from the
  // inlined instance.
  bool IsTailCall = BC.MIA->isTailCall(CallInst);

  // The first block of the function to be inlined can be merged with the caller
  // basic block. This cannot happen if there are jumps to the first block.
  bool CanMergeFirstInlinedBlock = (*InlinedFunction.begin()).pred_size() == 0;

  // If the call to be inlined is not at the end of its basic block and we have
  // to inline more than one basic blocks (or even just one basic block that
  // cannot be merged into the caller block), then the caller's basic block
  // should be split.
  bool ShouldSplitCallerBB =
    CallInstIndex < CallerBB->size() - 1 &&
    (InlinedFunction.size() > 1 || !CanMergeFirstInlinedBlock);

  // Copy inlined function's basic blocks into a vector of basic blocks that
  // will be inserted in the caller function (the inlined instance). Also, we
  // keep a mapping from basic block index to the corresponding block in the
  // inlined instance.
  std::vector<std::unique_ptr<BinaryBasicBlock>> InlinedInstance;
  std::unordered_map<const BinaryBasicBlock *, BinaryBasicBlock *> InlinedBBMap;

  for (const auto InlinedFunctionBB : InlinedFunction.layout()) {
    InlinedInstance.emplace_back(CallerFunction.createBasicBlock(0));
    InlinedBBMap[InlinedFunctionBB] = InlinedInstance.back().get();
    if (InlinedFunction.hasValidProfile()) {
      const auto Count = InlinedFunctionBB->getExecutionCount();
      InlinedInstance.back()->setExecutionCount(Count);
    }
  }
  if (ShouldSplitCallerBB) {
    // Add one extra block at the inlined instance for the removed part of the
    // caller block.
    InlinedInstance.emplace_back(CallerFunction.createBasicBlock(0));
    if (CallerFunction.hasValidProfile()) {
      const auto Count = CallerBB->getExecutionCount();
      InlinedInstance.back()->setExecutionCount(Count);
    }
  }

  // Copy instructions to the basic blocks of the inlined instance.
  bool First = true;
  for (const auto InlinedFunctionBB : InlinedFunction.layout()) {
    // Get the corresponding block of the inlined instance.
    auto *InlinedInstanceBB = InlinedBBMap.at(InlinedFunctionBB);
    bool IsExitingBlock = false;

    // Copy instructions into the inlined instance.
    for (auto Instruction : *InlinedFunctionBB) {
      if (!IsTailCall &&
          BC.MIA->isReturn(Instruction) &&
          !BC.MIA->isTailCall(Instruction)) {
        // Skip returns when the caller does a normal call as opposed to a tail
        // call.
        IsExitingBlock = true;
        continue;
      }
      if (!IsTailCall &&
          BC.MIA->isTailCall(Instruction)) {
        // Convert tail calls to normal calls when the caller does a normal
        // call.
        if (!BC.MIA->convertTailCallToCall(Instruction))
           assert(false && "unexpected tail call opcode found");
        IsExitingBlock = true;
      }
      if (BC.MIA->isBranch(Instruction) &&
          !BC.MIA->isIndirectBranch(Instruction)) {
        // Convert the branch targets in the branch instructions that will be
        // added to the inlined instance.
        const MCSymbol *OldTargetLabel = nullptr;
        const MCSymbol *OldFTLabel = nullptr;
        MCInst *CondBranch = nullptr;
        MCInst *UncondBranch = nullptr;
        const bool Result = BC.MIA->analyzeBranch(Instruction, OldTargetLabel,
                                                  OldFTLabel, CondBranch,
                                                  UncondBranch);
        assert(Result &&
               "analyzeBranch failed on instruction guaranteed to be a branch");
        assert(OldTargetLabel);
        const MCSymbol *NewTargetLabel = nullptr;
        for (const auto SuccBB : InlinedFunctionBB->successors()) {
          if (SuccBB->getLabel() == OldTargetLabel) {
            NewTargetLabel = InlinedBBMap.at(SuccBB)->getLabel();
            break;
          }
        }
        assert(NewTargetLabel);
        BC.MIA->replaceBranchTarget(Instruction, NewTargetLabel, BC.Ctx.get());
      }
      // TODO; Currently we simply ignore CFI instructions but we need to
      // address them for correctness.
      if (!BC.MIA->isEHLabel(Instruction) &&
          !BC.MIA->isCFI(Instruction)) {
        InlinedInstanceBB->addInstruction(std::move(Instruction));
      }
    }

    // Add CFG edges to the basic blocks of the inlined instance.
    std::vector<BinaryBasicBlock *>
      Successors(InlinedFunctionBB->succ_size(), nullptr);

    std::transform(
        InlinedFunctionBB->succ_begin(),
        InlinedFunctionBB->succ_end(),
        Successors.begin(),
        [&InlinedBBMap](const BinaryBasicBlock *BB) {
          return InlinedBBMap.at(BB);
        });

    if (InlinedFunction.hasValidProfile()) {
      InlinedInstanceBB->addSuccessors(
          Successors.begin(),
          Successors.end(),
          InlinedFunctionBB->branch_info_begin(),
          InlinedFunctionBB->branch_info_end());
    } else {
      InlinedInstanceBB->addSuccessors(
          Successors.begin(),
          Successors.end());
    }

    if (IsExitingBlock) {
      assert(Successors.size() == 0);
      if (ShouldSplitCallerBB) {
        if (InlinedFunction.hasValidProfile()) {
          InlinedInstanceBB->addSuccessor(
              InlinedInstance.back().get(),
              InlinedInstanceBB->getExecutionCount());
        } else {
          InlinedInstanceBB->addSuccessor(InlinedInstance.back().get());
        }
        InlinedInstanceBB->addBranchInstruction(InlinedInstance.back().get());
      } else if (!First || !CanMergeFirstInlinedBlock) {
        assert(CallInstIndex == CallerBB->size() - 1);
        assert(CallerBB->succ_size() <= 1);
        if (CallerBB->succ_size() == 1) {
          if (InlinedFunction.hasValidProfile()) {
            InlinedInstanceBB->addSuccessor(
                *CallerBB->succ_begin(),
                InlinedInstanceBB->getExecutionCount());
          } else {
            InlinedInstanceBB->addSuccessor(*CallerBB->succ_begin());
          }
          InlinedInstanceBB->addBranchInstruction(*CallerBB->succ_begin());
        }
      }
    }

    First = false;
  }

  if (ShouldSplitCallerBB) {
    // Split the basic block that contains the call and add the removed
    // instructions in the last block of the inlined instance.
    // (Is it OK to have a basic block with just CFI instructions?)
    std::vector<MCInst> TrailInstructions =
        CallerBB->splitInstructions(&CallInst);
    assert(TrailInstructions.size() > 0);
    InlinedInstance.back()->addInstructions(
        TrailInstructions.begin(),
        TrailInstructions.end());
    // Add CFG edges for the block with the removed instructions.
    if (CallerFunction.hasValidProfile()) {
      InlinedInstance.back()->addSuccessors(
          CallerBB->succ_begin(),
          CallerBB->succ_end(),
          CallerBB->branch_info_begin(),
          CallerBB->branch_info_end());
    } else {
      InlinedInstance.back()->addSuccessors(
          CallerBB->succ_begin(),
          CallerBB->succ_end());
    }
    // Update the after-inlined point.
    AfterInlinedBB = InlinedInstance.back().get();
    AfterInlinedIstrIndex = 0;
  }

  assert(InlinedInstance.size() > 0 && "found function with no basic blocks");
  assert(InlinedInstance.front()->size() > 0 &&
         "found function with empty basic block");

  // If the inlining cannot happen as a simple instruction insertion into
  // CallerBB, we remove the outgoing CFG edges of the caller block.
  if (InlinedInstance.size() > 1 || !CanMergeFirstInlinedBlock) {
    CallerBB->removeSuccessors(CallerBB->succ_begin(), CallerBB->succ_end());
    if (!ShouldSplitCallerBB) {
      // Update the after-inlined point.
      AfterInlinedBB = CallerFunction.getBasicBlockAfter(CallerBB);
      AfterInlinedIstrIndex = 0;
    }
  } else {
    assert(!ShouldSplitCallerBB);
    // Update the after-inlined point.
    if (CallInstIndex < CallerBB->size() - 1) {
      AfterInlinedBB = CallerBB;
      AfterInlinedIstrIndex =
        CallInstIndex + InlinedInstance.front()->size();
    } else {
      AfterInlinedBB = CallerFunction.getBasicBlockAfter(CallerBB);
      AfterInlinedIstrIndex = 0;
    }
  }

  // Do the inlining by merging the first block of the inlined instance into
  // the caller basic block if possible and adding the rest of the inlined
  // instance basic blocks in the caller function.
  if (CanMergeFirstInlinedBlock) {
    CallerBB->replaceInstruction(
        &CallInst,
        InlinedInstance.front()->begin(),
        InlinedInstance.front()->end());
    if (InlinedInstance.size() > 1) {
      auto FirstBB = InlinedInstance.begin()->get();
      if (InlinedFunction.hasValidProfile()) {
        CallerBB->addSuccessors(
            FirstBB->succ_begin(),
            FirstBB->succ_end(),
            FirstBB->branch_info_begin(),
            FirstBB->branch_info_end());
      } else {
        CallerBB->addSuccessors(
            FirstBB->succ_begin(),
            FirstBB->succ_end());
      }
      FirstBB->removeSuccessors(FirstBB->succ_begin(), FirstBB->succ_end());
    }
    InlinedInstance.erase(InlinedInstance.begin());
  } else {
    CallerBB->eraseInstruction(&CallInst);
    if (CallerFunction.hasValidProfile()) {
      CallerBB->addSuccessor(InlinedInstance.front().get(),
                             CallerBB->getExecutionCount());
    } else {
      CallerBB->addSuccessor(InlinedInstance.front().get(),
                             CallerBB->getExecutionCount());
    }
  }
  unsigned NumBlocksToAdd = InlinedInstance.size();
  CallerFunction.insertBasicBlocks(CallerBB, std::move(InlinedInstance));
  CallerFunction.updateLayout(CallerBB, NumBlocksToAdd);
  CallerFunction.fixBranches();

  return std::make_pair(AfterInlinedBB, AfterInlinedIstrIndex);
}

bool InlineSmallFunctions::inlineCallsInFunction(
    BinaryContext &BC,
    BinaryFunction &Function) {
  std::vector<BinaryBasicBlock *> Blocks(Function.layout().begin(),
                                         Function.layout().end());
  std::sort(Blocks.begin(), Blocks.end(),
      [](const BinaryBasicBlock *BB1, const BinaryBasicBlock *BB2) {
        return BB1->getExecutionCount() > BB2->getExecutionCount();
      });
  uint32_t ExtraSize = 0;

  for (auto BB : Blocks) {
    for (auto InstIt = BB->begin(), End = BB->end(); InstIt != End; ++InstIt) {
      auto &Inst = *InstIt;
      if (BC.MIA->isCall(Inst)) {
        TotalDynamicCalls += BB->getExecutionCount();
      }
    }
  }

  bool DidInlining = false;

  for (auto BB : Blocks) {
    if (BB->isCold())
      continue;

    for (auto InstIt = BB->begin(), End = BB->end(); InstIt != End; ) {
      auto &Inst = *InstIt;
      if (BC.MIA->isCall(Inst) &&
          !BC.MIA->isTailCall(Inst) &&
          Inst.size() == 1 &&
          Inst.getOperand(0).isExpr()) {
        const auto *TargetSymbol = BC.MIA->getTargetSymbol(Inst);
        assert(TargetSymbol && "target symbol expected for direct call");
        const auto *TargetFunction = BC.getFunctionForSymbol(TargetSymbol);
        if (TargetFunction) {
          bool CallToInlineableFunction =
            InliningCandidates.count(TargetFunction);

          TotalInlineableCalls +=
            CallToInlineableFunction * BB->getExecutionCount();

          if (CallToInlineableFunction &&
              TargetFunction->getSize() + ExtraSize
                + Function.estimateHotSize() < Function.getMaxSize()) {
            auto NextInstIt = std::next(InstIt);
            inlineCall(BC, *BB, &Inst, *TargetFunction->begin());
            DidInlining = true;
            DEBUG(dbgs() << "BOLT-DEBUG: Inlining call to "
                         << *TargetFunction << " in "
                         << Function << "\n");
            InstIt = NextInstIt;
            ExtraSize += TargetFunction->getSize();
            InlinedDynamicCalls += BB->getExecutionCount();
            continue;
          }
        }
      }

      ++InstIt;
    }
  }

  return DidInlining;
}

bool InlineSmallFunctions::inlineCallsInFunctionAggressive(
    BinaryContext &BC,
    BinaryFunction &Function) {
  std::vector<BinaryBasicBlock *> Blocks(Function.layout().begin(),
                                         Function.layout().end());
  std::sort(Blocks.begin(), Blocks.end(),
      [](const BinaryBasicBlock *BB1, const BinaryBasicBlock *BB2) {
        return BB1->getExecutionCount() > BB2->getExecutionCount();
      });
  uint32_t ExtraSize = 0;

  for (auto BB : Blocks) {
    for (auto InstIt = BB->begin(), End = BB->end(); InstIt != End; ++InstIt) {
      auto &Inst = *InstIt;
      if (BC.MIA->isCall(Inst)) {
        TotalDynamicCalls += BB->getExecutionCount();
      }
    }
  }

  bool DidInlining = false;

  for (auto BB : Blocks) {
    if (BB->isCold())
      continue;

    unsigned InstIndex = 0;
    for (auto InstIt = BB->begin(); InstIt != BB->end(); ) {
      auto &Inst = *InstIt;
      if (BC.MIA->isCall(Inst) &&
          Inst.size() == 1 &&
          Inst.getOperand(0).isExpr()) {
        assert(!BC.MIA->isInvoke(Inst));
        const auto *TargetSymbol = BC.MIA->getTargetSymbol(Inst);
        assert(TargetSymbol && "target symbol expected for direct call");
        const auto *TargetFunction = BC.getFunctionForSymbol(TargetSymbol);
        if (TargetFunction) {
          bool CallToInlineableFunction =
            InliningCandidates.count(TargetFunction);

          TotalInlineableCalls +=
            CallToInlineableFunction * BB->getExecutionCount();

          if (CallToInlineableFunction &&
              TargetFunction->getSize() + ExtraSize
              + Function.estimateHotSize() < Function.getMaxSize()) {
            unsigned NextInstIndex = 0;
            BinaryBasicBlock *NextBB = nullptr;
            std::tie(NextBB, NextInstIndex) =
              inlineCall(BC, Function, BB, InstIndex, *TargetFunction);
            DidInlining = true;
            DEBUG(dbgs() << "BOLT-DEBUG: Inlining call to "
                         << *TargetFunction << " in "
                         << Function << "\n");
            InstIndex = NextBB == BB ? NextInstIndex : BB->size();
            InstIt = NextBB == BB ? BB->begin() + NextInstIndex : BB->end();
            ExtraSize += TargetFunction->getSize();
            InlinedDynamicCalls += BB->getExecutionCount();
            continue;
          }
        }
      }

      ++InstIndex;
      ++InstIt;
    }
  }

  return DidInlining;
}

bool InlineSmallFunctions::mustConsider(const BinaryFunction &BF) {
  for (auto &Name : opts::ForceInlineFunctions) {
    if (BF.hasName(Name))
      return true;
  }
  return false;
}

void InlineSmallFunctions::runOnFunctions(
    BinaryContext &BC,
    std::map<uint64_t, BinaryFunction> &BFs,
    std::set<uint64_t> &) {

  if (opts::AggressiveInlining)
    findInliningCandidatesAggressive(BC, BFs);
  else
    findInliningCandidates(BC, BFs);

  std::vector<BinaryFunction *> ConsideredFunctions;
  for (auto &It : BFs) {
    auto &Function = It.second;
    if (!shouldOptimize(Function) ||
        (Function.getExecutionCount() == BinaryFunction::COUNT_NO_PROFILE &&
         !mustConsider(Function)))
      continue;
    ConsideredFunctions.push_back(&Function);
  }
  std::sort(ConsideredFunctions.begin(), ConsideredFunctions.end(),
            [](BinaryFunction *A, BinaryFunction *B) {
              return B->getExecutionCount() < A->getExecutionCount();
            });
  unsigned ModifiedFunctions = 0;
  for (unsigned i = 0; i < ConsideredFunctions.size() &&
                       ModifiedFunctions <= kMaxFunctions; ++i) {
    auto &Function = *ConsideredFunctions[i];

    const bool DidInline = opts::AggressiveInlining
      ? inlineCallsInFunctionAggressive(BC, Function)
      : inlineCallsInFunction(BC, Function);

    if (DidInline) {
      Modified.insert(&Function);
      ++ModifiedFunctions;
    }
  }

  DEBUG(dbgs() << "BOLT-INFO: Inlined " << InlinedDynamicCalls << " of "
               << TotalDynamicCalls << " function calls in the profile.\n"
               << "BOLT-INFO: Inlined calls represent "
               << format("%.1f",
                         100.0 * InlinedDynamicCalls / TotalInlineableCalls)
               << "% of all inlineable calls in the profile.\n");
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

void FixupFunctions::runOnFunctions(
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
               << Function << ". Aborting.\n";
        abort();
      }
      if (opts::Verbosity >= 1) {
        errs() << "BOLT-WARNING: unable to fix CFI state for function "
               << Function << ". Skipping.\n";
      }
      Function.setSimple(false);
      continue;
    }

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

    auto *Instr = BB->getFirstNonPseudo();
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
        Pred->replaceSuccessor(&BB, Succ, BinaryBasicBlock::COUNT_NO_PROFILE);
      } else {
        // Succ will be null in the tail call case.  In this case we
        // need to explicitly add a tail call instruction.
        auto *Branch = Pred->getLastNonPseudo();
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

    auto *Inst = BB.getFirstNonPseudo();
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
    auto *Inst = BB.getLastNonPseudo();
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
  outs() << "BOLT-INFO: " << NumDoubleJumps << " double jumps patched.\n";
  outs() << "BOLT-INFO: " << TailCallTraps << " tail call traps inserted.\n";
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
      if (BB->getExecutionCount() != BinaryBasicBlock::COUNT_NO_PROFILE)
        NumDynamicLocalLoadsFound += BB->getExecutionCount();

      if (MIA->replaceMemOperandWithImm(Inst, ConstantData, Offset)) {
        ++NumLocalLoadsSimplified;
        if (BB->getExecutionCount() != BinaryBasicBlock::COUNT_NO_PROFILE)
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
  if (!opts::IdenticalCodeFolding)
    return;

  const auto OriginalFunctionCount = BFs.size();
  uint64_t NumFunctionsFolded = 0;
  uint64_t NumJTFunctionsFolded = 0;
  uint64_t BytesSavedEstimate = 0;
  static bool UseDFS = opts::UseDFSForICF;

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
    if (!shouldOptimize(BF))
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
          BC.foldFunction(*ChildBF, *ParentBF, BFs);

          BytesSavedEstimate += ChildBF->getSize();
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
           << " KB of code space.\n";
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

} // namespace bolt
} // namespace llvm
