//===--- Passes/Inliner.cpp - Inlining infra for BOLT ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "Inliner.h"
#include "llvm/Support/Options.h"

#define DEBUG_TYPE "bolt-inliner"

using namespace llvm;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

static cl::opt<bool>
AggressiveInlining("aggressive-inlining",
  cl::desc("perform aggressive inlining"),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::list<std::string>
ForceInlineFunctions("force-inline",
  cl::CommaSeparated,
  cl::desc("list of functions to always consider for inlining"),
  cl::value_desc("func1,func2,func3,..."),
  cl::Hidden,
  cl::cat(BoltOptCategory));

}

namespace llvm {
namespace bolt {

void InlineSmallFunctions::findInliningCandidates(
    BinaryContext &BC,
    const std::map<uint64_t, BinaryFunction> &BFs) {
  for (const auto &BFIt : BFs) {
    const auto &Function = BFIt.second;
    if (!shouldOptimize(Function) || Function.size() != 1)
      continue;
    auto &BB = *Function.begin();
    const auto &LastInstruction = *BB.rbegin();
    // Check if the function is small enough, doesn't do a tail call
    // and doesn't throw exceptions.
    if (BB.size() > 0 &&
        BB.getNumNonPseudos() <= kMaxInstructions &&
        BB.lp_empty() &&
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
        (void)Result;
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
  CallerFunction.insertBasicBlocks(CallerBB, std::move(InlinedInstance));

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
          Inst.getNumPrimeOperands() == 1 &&
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
          Inst.getNumPrimeOperands() == 1 &&
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


} // namespace bolt
} // namespace llvm
