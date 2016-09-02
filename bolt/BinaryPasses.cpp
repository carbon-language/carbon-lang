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
#include <unordered_map>

#define DEBUG_TYPE "bolt"

using namespace llvm;

namespace opts {

extern cl::opt<unsigned> Verbosity;
extern cl::opt<bool> PrintAll;
extern cl::opt<bool> DumpDotAll;
extern cl::opt<llvm::bolt::BinaryFunction::SplittingType> SplitFunctions;
extern bool shouldProcess(const bolt::BinaryFunction &Function);

static cl::opt<bool>
PrintReordered("print-reordered",
               cl::desc("print functions after layout optimization"),
               cl::Hidden);

static cl::opt<bool>
PrintEHRanges("print-eh-ranges",
              cl::desc("print function with updated exception ranges"),
              cl::Hidden);

static cl::opt<bool>
PrintUCE("print-uce",
         cl::desc("print functions after unreachable code elimination"),
         cl::Hidden);

static cl::opt<bool>
PrintPeepholes("print-peepholes",
               cl::desc("print functions after peephole optimization"),
               cl::Hidden);

static cl::opt<bool>
PrintSimplifyROLoads("print-simplify-rodata-loads",
                     cl::desc("print functions after simplification of RO data"
                              " loads"),
                     cl::Hidden);

static cl::opt<bool>
PrintICF("print-icf",
         cl::desc("print functions after ICF optimization"),
         cl::Hidden);

static cl::opt<bool>
PrintInline("print-inline",
            cl::desc("print functions after inlining optimization"),
            cl::Hidden);

static cl::list<std::string>
ForceInlineFunctions("force-inline",
                     cl::CommaSeparated,
                     cl::desc("list of functions to always consider "
                              "for inlining"),
                     cl::value_desc("func1,func2,func3,..."));

static cl::opt<bool>
AggressiveInlining("aggressive-inlining",
                   cl::desc("perform aggressive inlining"),
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
               clEnumValEnd));

static cl::opt<bool>
MinBranchClusters(
    "min-branch-clusters",
    cl::desc("use a modified clustering algorithm geared towards "
             "minimizing branches"),
    cl::Hidden);

} // namespace opts

namespace llvm {
namespace bolt {

void OptimizeBodylessFunctions::analyze(
    BinaryFunction &BF,
    BinaryContext &BC,
    std::map<uint64_t, BinaryFunction> &BFs) {
  if (BF.size() != 1 || (*BF.begin()).size() == 0)
    return;

  auto &BB = *BF.begin();
  const auto &FirstInst = *BB.begin();
  if (!BC.MIA->isTailCall(FirstInst))
    return;
  auto &Op1 = FirstInst.getOperand(0);
  if (!Op1.isExpr())
    return;
  auto Expr = dyn_cast<MCSymbolRefExpr>(Op1.getExpr());
  if (!Expr)
    return;
  const auto *Function = BC.getFunctionForSymbol(&Expr->getSymbol());
  if (!Function)
    return;

  EquivalentCallTarget[BF.getSymbol()] = Function;
}

void OptimizeBodylessFunctions::optimizeCalls(BinaryFunction &BF,
                                              BinaryContext &BC) {
  for (auto BBIt = BF.begin(), BBEnd = BF.end(); BBIt != BBEnd; ++BBIt) {
    for (auto InstIt = (*BBIt).begin(), InstEnd = (*BBIt).end();
        InstIt != InstEnd; ++InstIt) {
      auto &Inst = *InstIt;
      if (!BC.MIA->isCall(Inst))
        continue;
      auto &Op1 = Inst.getOperand(0);
      if (!Op1.isExpr())
        continue;
      auto Expr = dyn_cast<MCSymbolRefExpr>(Op1.getExpr());
      if (!Expr)
        continue;
      auto *OriginalTarget = &Expr->getSymbol();
      auto *Target = OriginalTarget;
      // Iteratively update target since we could have f1() calling f2()
      // calling f3() calling f4() and we want to output f1() directly
      // calling f4().
      while (EquivalentCallTarget.count(Target)) {
        Target = EquivalentCallTarget.find(Target)->second->getSymbol();
      }
      if (Target == OriginalTarget)
        continue;
      DEBUG(dbgs() << "BOLT-DEBUG: Optimizing " << (*BBIt).getName()
                   << " in " << BF
                   << ": replacing call to " << OriginalTarget->getName()
                   << " by call to " << Target->getName() << "\n");
      BC.MIA->replaceCallTargetOperand(Inst, Target, BC.Ctx.get());
    }
  }
}

void OptimizeBodylessFunctions::runOnFunctions(
    BinaryContext &BC,
    std::map<uint64_t, BinaryFunction> &BFs,
    std::set<uint64_t> &) {
  for (auto &It : BFs) {
    auto &Function = It.second;
    if (Function.isSimple() && opts::shouldProcess(Function)) {
      analyze(Function, BC, BFs);
    }
  }
  for (auto &It : BFs) {
    auto &Function = It.second;
    if (Function.isSimple() && opts::shouldProcess(Function)) {
      optimizeCalls(Function, BC);
    }
  }
}

void InlineSmallFunctions::findInliningCandidates(
    BinaryContext &BC,
    const std::map<uint64_t, BinaryFunction> &BFs) {
  for (const auto &BFIt : BFs) {
    const auto &Function = BFIt.second;
    if (!Function.isSimple() ||
        !opts::shouldProcess(Function) ||
        Function.size() != 1)
      continue;
    auto &BB = *Function.begin();
    const auto &LastInstruction = *BB.rbegin();
    // Check if the function is small enough and doesn't do a tail call.
    if (BB.size() > 0 &&
        (BB.size() - BB.getNumPseudos()) <= kMaxInstructions &&
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
    if (!Function.isSimple() ||
        !opts::shouldProcess(Function) ||
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
  std::vector<BinaryBasicBlock *>
    BBIndexToInlinedInstanceBB(InlinedFunction.size(), nullptr);
  for (const auto InlinedFunctionBB : InlinedFunction.layout()) {
    InlinedInstance.emplace_back(CallerFunction.createBasicBlock(0));
    BBIndexToInlinedInstanceBB[InlinedFunction.getIndex(InlinedFunctionBB)] =
      InlinedInstance.back().get();
    if (InlinedFunction.hasValidProfile())
      InlinedInstance.back()->setExecutionCount(
          InlinedFunctionBB->getExecutionCount());
  }
  if (ShouldSplitCallerBB) {
    // Add one extra block at the inlined instance for the removed part of the
    // caller block.
    InlinedInstance.emplace_back(CallerFunction.createBasicBlock(0));
    BBIndexToInlinedInstanceBB.push_back(InlinedInstance.back().get());
    if (CallerFunction.hasValidProfile())
      InlinedInstance.back()->setExecutionCount(CallerBB->getExecutionCount());
  }

  // Copy instructions to the basic blocks of the inlined instance.
  unsigned InlinedInstanceBBIndex = 0;
  for (const auto InlinedFunctionBB : InlinedFunction.layout()) {
    // Get the corresponding block of the inlined instance.
    auto *InlinedInstanceBB = InlinedInstance[InlinedInstanceBBIndex].get();
    assert(InlinedInstanceBB ==
           BBIndexToInlinedInstanceBB[InlinedFunction.getIndex(InlinedFunctionBB)]);

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
            const auto InlinedInstanceSuccBB =
              BBIndexToInlinedInstanceBB[InlinedFunction.getIndex(SuccBB)];
            NewTargetLabel = InlinedInstanceSuccBB->getLabel();
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
        [&InlinedFunction, &BBIndexToInlinedInstanceBB]
        (const BinaryBasicBlock *BB) {
          return BBIndexToInlinedInstanceBB[InlinedFunction.getIndex(BB)];
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
        MCInst ExitBranchInst;
        const MCSymbol *ExitLabel = InlinedInstance.back().get()->getLabel();
        BC.MIA->createUncondBranch(ExitBranchInst, ExitLabel, BC.Ctx.get());
        InlinedInstanceBB->addInstruction(std::move(ExitBranchInst));
      } else if (InlinedInstanceBBIndex > 0 || !CanMergeFirstInlinedBlock) {
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
          MCInst ExitBranchInst;
          const MCSymbol *ExitLabel = (*CallerBB->succ_begin())->getLabel();
          BC.MIA->createUncondBranch(ExitBranchInst, ExitLabel, BC.Ctx.get());
          InlinedInstanceBB->addInstruction(std::move(ExitBranchInst));
        }
      }
    }

    ++InlinedInstanceBBIndex;
  }

  if (ShouldSplitCallerBB) {
    // Split the basic block that contains the call and add the removed
    // instructions in the last block of the inlined instance.
    // (Is it OK to have a basic block with just CFI instructions?)
    std::vector<MCInst> TrailInstructions =
      std::move(CallerBB->splitInstructions(&CallInst));
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
        auto Target = dyn_cast<MCSymbolRefExpr>(
            Inst.getOperand(0).getExpr());
        assert(Target && "Not MCSymbolRefExpr");
        const auto *TargetFunction =
          BC.getFunctionForSymbol(&Target->getSymbol());
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
        auto Target = dyn_cast<MCSymbolRefExpr>(
            Inst.getOperand(0).getExpr());
        assert(Target && "Not MCSymbolRefExpr");
        const auto *TargetFunction =
          BC.getFunctionForSymbol(&Target->getSymbol());
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
  std::vector<bool> Modified;
  for (auto &It : BFs) {
    auto &Function = It.second;
    if (!Function.isSimple() || !opts::shouldProcess(Function))
      continue;
    if (Function.getExecutionCount() == BinaryFunction::COUNT_NO_PROFILE &&
        !mustConsider(Function))
      continue;
    ConsideredFunctions.push_back(&Function);
    Modified.push_back(false);
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
      Modified[i] = true;
      ++ModifiedFunctions;
    }
  }

  if (opts::PrintAll || opts::PrintInline || opts::DumpDotAll) {
    for (unsigned i = 0; i < ConsideredFunctions.size(); ++i) {
      if (Modified[i]) {
        const auto *Function = ConsideredFunctions[i];
        if (opts::PrintAll || opts::PrintInline)
          Function->print(errs(), "after inlining", true);

        if (opts::DumpDotAll)
          Function->dumpGraphForPass("inlining");
      }
    }
  }

  DEBUG(dbgs() << "BOLT-INFO: Inlined " << InlinedDynamicCalls << " of "
               << TotalDynamicCalls << " function calls in the profile.\n"
               << "BOLT-INFO: Inlined calls represent "
               << format("%.1f", 100.0 * InlinedDynamicCalls / TotalInlineableCalls)
               << "% of all inlineable calls in the profile.\n");
}

void EliminateUnreachableBlocks::runOnFunction(BinaryFunction& Function) {
  if (!Function.isSimple() || !opts::shouldProcess(Function)) return;

  // FIXME: this wouldn't work with C++ exceptions until we implement
  //        support for those as there will be "invisible" edges
  //        in the graph.
  if (Function.layout_size() > 0) {
    if (NagUser) {
      if (opts::Verbosity >= 1) {
        errs()
          << "BOLT-WARNING: Using -eliminate-unreachable is experimental and "
          "unsafe for exceptions\n";
      }
      NagUser = false;
    }

    if (Function.hasEHRanges()) return;

    std::stack<BinaryBasicBlock*> Stack;
    std::map<BinaryBasicBlock *, bool> Reachable;
    BinaryBasicBlock *Entry = *Function.layout_begin();
    Stack.push(Entry);
    Reachable[Entry] = true;
    // Determine reachable BBs from the entry point
    while (!Stack.empty()) {
      auto BB = Stack.top();
      Stack.pop();
      for (auto Succ : BB->successors()) {
        if (Reachable[Succ])
          continue;
        Reachable[Succ] = true;
        Stack.push(Succ);
      }
    }

    auto Count = Function.eraseDeadBBs(Reachable);
    if (Count) {
      DEBUG(dbgs() << "BOLT: Removed " << Count
            << " dead basic block(s) in function " << Function << '\n');
    }

    if (opts::PrintAll || opts::PrintUCE)
      Function.print(outs(), "after unreachable code elimination", true);

    if (opts::DumpDotAll)
      Function.dumpGraphForPass("unreachable-code");
  }
}

void EliminateUnreachableBlocks::runOnFunctions(
  BinaryContext&,
  std::map<uint64_t, BinaryFunction> &BFs,
  std::set<uint64_t> &
) {
  for (auto &It : BFs) {
    runOnFunction(It.second);
  }
}

void ReorderBasicBlocks::runOnFunctions(
    BinaryContext &BC,
    std::map<uint64_t, BinaryFunction> &BFs,
    std::set<uint64_t> &LargeFunctions) {
  for (auto &It : BFs) {
    auto &Function = It.second;

    if (!Function.isSimple())
      continue;

    if (!opts::shouldProcess(Function))
      continue;

    if (opts::ReorderBlocks != BinaryFunction::LT_NONE) {
      bool ShouldSplit =
        (opts::SplitFunctions == BinaryFunction::ST_ALL) ||
        (opts::SplitFunctions == BinaryFunction::ST_EH &&
         Function.hasEHRanges()) ||
        (LargeFunctions.find(It.first) != LargeFunctions.end());
      Function.modifyLayout(opts::ReorderBlocks, opts::MinBranchClusters,
                            ShouldSplit);
      if (opts::PrintAll || opts::PrintReordered)
        Function.print(outs(), "after reordering blocks", true);
      if (opts::DumpDotAll)
        Function.dumpGraphForPass("reordering");
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

    if (!Function.isSimple())
      continue;

    if (!opts::shouldProcess(Function))
      continue;

    // Fix the CFI state.
    if (!Function.fixCFIState()) {
      if (opts::Verbosity >= 1) {
        errs() << "BOLT-WARNING: unable to fix CFI state for function "
               << Function << ". Skipping.\n";
      }
      Function.setSimple(false);
      continue;
    }

    // Update exception handling information.
    Function.updateEHRanges();
    if (opts::PrintAll || opts::PrintEHRanges)
      Function.print(outs(), "after updating EH ranges", true);
    if (opts::DumpDotAll)
      Function.dumpGraphForPass("update-EH-ranges");
  }
}

bool SimplifyConditionalTailCalls::fixTailCalls(BinaryContext &BC,
                                                BinaryFunction &BF) {
  if (BF.layout_size() == 0)
    return false;

  auto &MIA = BC.MIA;
  uint64_t NumLocalTailCalls = 0;
  uint64_t NumLocalPatchedTailCalls = 0;

  for (auto* BB : BF.layout()) {
    const MCSymbol *TBB = nullptr;
    const MCSymbol *FBB = nullptr;
    MCInst *CondBranch = nullptr;
    MCInst *UncondBranch = nullptr;

    // Determine the control flow at the end of each basic block
    if (!BB->analyzeBranch(*MIA, TBB, FBB, CondBranch, UncondBranch)) {
      continue;
    }

    // TODO: do we need to test for other branch patterns?

    // For this particular case, the first basic block ends with
    // a conditional branch and has two successors, one fall-through
    // and one for when the condition is true.
    // The target of the conditional is a basic block with a single
    // unconditional branch (i.e. tail call) to another function.
    // We don't care about the contents of the fall-through block.
    // Note: this code makes the assumption that the fall-through
    // block is the last successor.
    if (CondBranch && !UncondBranch && BB->succ_size() == 2) {
      // Find conditional branch target assuming the fall-through is
      // always the last successor.
      auto *CondTargetBB = *BB->succ_begin();

      // Does the BB contain a single instruction?
      if (CondTargetBB->size() - CondTargetBB->getNumPseudos() == 1) {
        // Check to see if the sole instruction is a tail call.
        auto const &Instr = *CondTargetBB->begin();

        if (MIA->isTailCall(Instr)) {
          ++NumTailCallCandidates;
          ++NumLocalTailCalls;

          auto const &TailTargetSymExpr =
            cast<MCSymbolRefExpr>(Instr.getOperand(0).getExpr());
          auto const &TailTarget = TailTargetSymExpr->getSymbol();

          // Lookup the address for the tail call target.
          auto const TailAddress = BC.GlobalSymbols.find(TailTarget.getName());
          if (TailAddress == BC.GlobalSymbols.end())
            continue;

          // Check to make sure we would be doing a forward jump.
          // This assumes the address range of the current BB and the
          // tail call target address don't overlap.
          if (BF.getAddress() < TailAddress->second) {
            ++NumTailCallsPatched;
            ++NumLocalPatchedTailCalls;

            // Is the original jump forward or backward?
            const bool isForward =
              TailAddress->second > BF.getAddress() + BB->getOffset();

            if (isForward) ++NumOrigForwardBranches;

            // Patch the new target address into the conditional branch.
            CondBranch->getOperand(0).setExpr(TailTargetSymExpr);
            // Remove the unused successor which may be eliminated later
            // if there are no other users.
            BB->removeSuccessor(CondTargetBB);
            DEBUG(dbgs() << "patched " << (isForward ? "(fwd)" : "(back)")
                  << " tail call in " << BF << ".\n";);
          }
        }
      }
    }
  }

  DEBUG(dbgs() << "BOLT: patched " << NumLocalPatchedTailCalls
        << " tail calls (" << NumOrigForwardBranches << " forward)"
        << " from a total of " << NumLocalTailCalls
        << " in function " << BF << "\n";);

  return NumLocalPatchedTailCalls > 0;
}

void SimplifyConditionalTailCalls::runOnFunctions(
  BinaryContext &BC,
  std::map<uint64_t, BinaryFunction> &BFs,
  std::set<uint64_t> &
) {
  for (auto &It : BFs) {
    auto &Function = It.second;

    if (!Function.isSimple())
      continue;

    // Fix tail calls to reduce branch mispredictions.
    if (fixTailCalls(BC, Function)) {
      if (opts::PrintAll || opts::PrintReordered) {
        Function.print(outs(), "after tail call patching", true);
      }
      if (opts::DumpDotAll) {
        Function.dumpGraphForPass("tail-call-patching");
      }
    }
  }

  outs() << "BOLT-INFO: patched " << NumTailCallsPatched
         << " tail calls (" << NumOrigForwardBranches << " forward)"
         << " from a total of " << NumTailCallCandidates << "\n";
}

void Peepholes::shortenInstructions(BinaryContext &BC,
                                    BinaryFunction &Function) {
  for (auto &BB : Function) {
    for (auto &Inst : BB) {
      BC.MIA->shortenInstruction(Inst);
    }
  }
}

void Peepholes::runOnFunctions(BinaryContext &BC,
                               std::map<uint64_t, BinaryFunction> &BFs,
                               std::set<uint64_t> &LargeFunctions) {
  for (auto &It : BFs) {
    auto &Function = It.second;
    if (Function.isSimple() && opts::shouldProcess(Function)) {
      shortenInstructions(BC, Function);

      if (opts::PrintAll || opts::PrintPeepholes) {
        Function.print(outs(), "after peepholes", true);
      }

      if (opts::DumpDotAll) {
        Function.dumpGraphForPass("peepholes");
      }
    }
  }
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
        // Try to find the symbol that corresponds to the rip-relative operand.
        MCOperand DisplOp;
        if (!MIA->getRIPOperandDisp(Inst, DisplOp))
          continue;

        assert(DisplOp.isExpr() &&
              "found rip-relative with non-symbolic displacement");

        // Get displacement symbol.
        const MCSymbolRefExpr *DisplExpr;
        if (!(DisplExpr = dyn_cast<MCSymbolRefExpr>(DisplOp.getExpr())))
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

      // Get the contents of the section containing the target addresss of the
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

    if (!Function.isSimple())
      continue;

    if (simplifyRODataLoads(BC, Function)) {
      if (opts::PrintAll || opts::PrintSimplifyROLoads) {
        Function.print(outs(),
                       "after simplifying read-only section loads",
                       true);
      }
      if (opts::DumpDotAll) {
        Function.dumpGraphForPass("simplify-rodata-loads");
      }
    }
  }

  outs() << "BOLT-INFO: simplified " << NumLoadsSimplified << " out of "
         << NumLoadsFound << " loads from a statically computed address.\n"
         << "BOLT-INFO: dynamic loads simplified: " << NumDynamicLoadsSimplified
         << "\n"
         << "BOLT-INFO: dynamic loads found: " << NumDynamicLoadsFound << "\n";
}

void IdenticalCodeFolding::discoverCallers(
  BinaryContext &BC, std::map<uint64_t, BinaryFunction> &BFs) {
  for (auto &I : BFs) {
    BinaryFunction &Caller = I.second;

    if (!Caller.isSimple())
      continue;

    for (BinaryBasicBlock &BB : Caller) {
      unsigned BlockIndex = Caller.getIndex(&BB);
      unsigned InstrIndex = 0;

      for (MCInst &Inst : BB) {
        if (!BC.MIA->isCall(Inst)) {
          ++InstrIndex;
          continue;
        }

        const MCOperand &TargetOp = Inst.getOperand(0);
        if (!TargetOp.isExpr()) {
          // This is an inderect call, we cannot record
          // a target.
          ++InstrIndex;
          continue;
        }

        // Find the target function for this call.
        const auto *TargetExpr = TargetOp.getExpr();
        assert(TargetExpr->getKind() == MCExpr::SymbolRef);
        const auto &TargetSymbol =
          dyn_cast<MCSymbolRefExpr>(TargetExpr)->getSymbol();
        const auto *Function = BC.getFunctionForSymbol(&TargetSymbol);
        if (!Function) {
          // Call to a function without a BinaryFunction object.
          ++InstrIndex;
          continue;
        }
        // Insert a tuple in the Callers map.
        Callers[Function].emplace_back(
          CallSite(&Caller, BlockIndex, InstrIndex));
        ++InstrIndex;
      }
    }
  }
}

void IdenticalCodeFolding::foldFunction(
  BinaryContext &BC,
  std::map<uint64_t, BinaryFunction> &BFs,
  BinaryFunction *BFToFold,
  BinaryFunction *BFToReplaceWith,
  std::set<BinaryFunction *> &Modified) {

  // Mark BFToFold as identical with BFTOreplaceWith.
  BFToFold->setIdenticalFunctionAddress(BFToReplaceWith->getAddress());

  // Add the size of BFToFold to the total size savings estimate.
  BytesSavedEstimate += BFToFold->getSize();

  // Get callers of BFToFold.
  auto CI = Callers.find(BFToFold);
  if (CI == Callers.end())
    return;
  std::vector<CallSite> &BFToFoldCallers = CI->second;

  // Get callers of BFToReplaceWith.
  std::vector<CallSite> &BFToReplaceWithCallers = Callers[BFToReplaceWith];

  // Get MCSymbol for BFToReplaceWith.
  MCSymbol *SymbolToReplaceWith =
    BC.getOrCreateGlobalSymbol(BFToReplaceWith->getAddress(), "");

  // Traverse callers of BFToFold and replace the calls with calls
  // to BFToReplaceWith.
  for (const CallSite &CS : BFToFoldCallers) {
    // Get call instruction.
    BinaryFunction *Caller = CS.Caller;
    BinaryBasicBlock *CallBB = Caller->getBasicBlockAtIndex(CS.BlockIndex);
    MCInst &CallInst = CallBB->getInstructionAtIndex(CS.InstrIndex);

    // Replace call target with BFToReplaceWith.
    auto Success = BC.MIA->replaceCallTargetOperand(CallInst,
                                                    SymbolToReplaceWith,
                                                    BC.Ctx.get());
    assert(Success && "unexpected call target prevented the replacement");

    // Add this call site to the callers of BFToReplaceWith.
    BFToReplaceWithCallers.emplace_back(CS);

    // Add caller to the set of modified functions.
    Modified.insert(Caller);

    // Update dynamic calls folded stat.
    if (Caller->hasValidProfile() &&
        CallBB->getExecutionCount() != BinaryBasicBlock::COUNT_NO_PROFILE)
      NumDynamicCallsFolded += CallBB->getExecutionCount();
  }

  // Remove all callers of BFToFold.
  BFToFoldCallers.clear();

  ++NumFunctionsFolded;

  // Merge execution counts of BFToFold into those of BFToReplaceWith.
  BFToFold->mergeProfileDataInto(*BFToReplaceWith);
}

void IdenticalCodeFolding::runOnFunctions(
  BinaryContext &BC,
  std::map<uint64_t, BinaryFunction> &BFs,
  std::set<uint64_t> &
) {

  discoverCallers(BC, BFs);

  // This hash table is used to identify identical functions. It maps
  // a function to a bucket of functions identical to it.
  struct KeyHash {
    std::size_t operator()(const BinaryFunction *F) const { return F->hash(); }
  };
  struct KeyEqual {
    bool operator()(const BinaryFunction *A, const BinaryFunction *B) const {
      return A->isIdenticalWith(*B);
    }
  };
  std::unordered_map<BinaryFunction *, std::vector<BinaryFunction *>,
                     KeyHash, KeyEqual> Buckets;

  // Set that holds the functions that were modified by the last pass.
  std::set<BinaryFunction *> Mod;

  // Vector of all the candidate functions to be tested for being identical
  // to each other. Initialized with all simple functions.
  std::vector<BinaryFunction *> Cands;
  for (auto &I : BFs) {
    BinaryFunction *BF = &I.second;
    if (BF->isSimple())
      Cands.emplace_back(BF);
  }

  // We repeat the icf pass until no new modifications happen.
  unsigned Iter = 1;
  do {
    Buckets.clear();
    Mod.clear();

    if (opts::Verbosity >= 1) {
      outs() << "BOLT-INFO: icf pass " << Iter << "...\n";
    }

    uint64_t NumIdenticalFunctions = 0;

    // Compare candidate functions using the Buckets hash table. Identical
    // functions are effiently discovered and added to the same bucket.
    for (BinaryFunction *BF : Cands) {
      Buckets[BF].emplace_back(BF);
    }

    Cands.clear();

    // Go through the functions of each bucket and fold any references to them
    // with the references to the hottest function among them.
    for (auto &I : Buckets) {
      std::vector<BinaryFunction *> &IFs = I.second;
      std::sort(IFs.begin(), IFs.end(),
                [](const BinaryFunction *A, const BinaryFunction *B) {
                  if (!A->hasValidProfile() && !B->hasValidProfile())
                    return false;

                  if (!A->hasValidProfile())
                    return false;

                  if (!B->hasValidProfile())
                    return true;

                  return B->getExecutionCount() < A->getExecutionCount();
                }
      );
      BinaryFunction *Hottest = IFs[0];

      // For the next pass, we consider only one function from each set of
      // identical functions.
      Cands.emplace_back(Hottest);

      if (IFs.size() <= 1)
        continue;

      NumIdenticalFunctions += IFs.size() - 1;
      for (unsigned i = 1; i < IFs.size(); ++i) {
        BinaryFunction *BF = IFs[i];
        foldFunction(BC, BFs, BF, Hottest, Mod);
      }
    }

    if (opts::Verbosity >= 1) {
      outs() << "BOLT-INFO: found " << NumIdenticalFunctions
             << " identical functions.\n"
             << "BOLT-INFO: modified " << Mod.size() << " functions.\n";
    }

    NumIdenticalFunctionsFound += NumIdenticalFunctions;

    ++Iter;
  } while (!Mod.empty());

  outs() << "BOLT-INFO: ICF pass found " << NumIdenticalFunctionsFound
         << " functions identical to some other function.\n"
         << "BOLT-INFO: ICF pass folded references to " << NumFunctionsFolded
         << " functions.\n"
         << "BOLT-INFO: ICF pass folded " << NumDynamicCallsFolded << " dynamic"
         << " function calls.\n"
         << "BOLT-INFO: Removing all identical functions could save "
         << format("%.2lf", (double) BytesSavedEstimate / 1024)
         << " KB of code space.\n";

  if (opts::PrintAll || opts::PrintICF) {
    for (auto &I : BFs) {
      I.second.print(outs(), "after identical code folding", true);
    }
  }
}

} // namespace bolt
} // namespace llvm
