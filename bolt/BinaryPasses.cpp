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

#define DEBUG_TYPE "bolt"

namespace llvm {
namespace bolt {

void OptimizeBodylessFunctions::analyze(
    BinaryFunction &BF,
    BinaryContext &BC,
    std::map<uint64_t, BinaryFunction> &BFs) {
  if (BF.size() != 1 || BF.begin()->size() == 0)
    return;

  auto &BB = *BF.begin();
  const auto &FirstInst = *BB.begin();

  if (!BC.MIA->isTailCall(FirstInst))
    return;

  auto &Op1 = FirstInst.getOperand(0);
  if (!Op1.isExpr())
    return;

  if (auto Expr = dyn_cast<MCSymbolRefExpr>(Op1.getExpr())) {
    auto AddressIt = BC.GlobalSymbols.find(Expr->getSymbol().getName());
    if (AddressIt != BC.GlobalSymbols.end()) {
      auto CalleeIt = BFs.find(AddressIt->second);
      if (CalleeIt != BFs.end()) {
        assert(Expr->getSymbol().getName() == CalleeIt->second.getName());
        EquivalentCallTarget[BF.getName()] = &CalleeIt->second;
      }
    }
  }
}

void OptimizeBodylessFunctions::optimizeCalls(BinaryFunction &BF,
                                              BinaryContext &BC) {
  for (auto BBIt = BF.begin(), BBEnd = BF.end(); BBIt != BBEnd; ++BBIt) {
    for (auto InstIt = BBIt->begin(), InstEnd = BBIt->end();
        InstIt != InstEnd; ++InstIt) {
      auto &Inst = *InstIt;
      if (BC.MIA->isCall(Inst)) {
        auto &Op1 = Inst.getOperand(0);
        if (Op1.isExpr()) {
          if (auto Expr = dyn_cast<MCSymbolRefExpr>(Op1.getExpr())) {
            auto OriginalTarget = Expr->getSymbol().getName();
            auto Target = OriginalTarget;
            // Iteratively update target since we could have f1() calling f2()
            // calling f3() calling f4() and we want to output f1() directly
            // calling f4().
            while (EquivalentCallTarget.count(Target)) {
              Target = EquivalentCallTarget.find(Target)->second->getName();
            }
            if (Target != OriginalTarget) {
              DEBUG(errs() << "BOLT-DEBUG: Optimizing " << BF.getName()
                           << ": replacing call to "
                           << OriginalTarget
                           << " by call to " << Target << "\n");
              Inst.clear();
              Inst.addOperand(MCOperand::createExpr(
                    MCSymbolRefExpr::create(
                      BC.Ctx->getOrCreateSymbol(Target), *BC.Ctx)));
            }
          }
        }
      }
    }
  }
}

void OptimizeBodylessFunctions::runOnFunctions(
    BinaryContext &BC,
    std::map<uint64_t, BinaryFunction> &BFs) {
  for (auto &It : BFs) {
    analyze(It.second, BC, BFs);
  }
  for (auto &It : BFs) {
    optimizeCalls(It.second, BC);
  }
}

void InlineSmallFunctions::findInliningCandidates(
    BinaryContext &BC,
    const std::map<uint64_t, BinaryFunction> &BFs) {
  for (const auto &BFIt : BFs) {
    const auto &Function = BFIt.second;
    if (Function.size() != 1)
      continue;
    auto &BB = *Function.begin();
    const auto &LastInstruction = *BB.rbegin();
    // Check if the function is small enough and doesn't do a tail call.
    // The size we use includes pseudo-instructions but here they shouldn't
    // matter. So some opportunities may be missed because of this.
    if (BB.size() > 0 &&
        BB.size() <= kMaxInstructions &&
        BC.MIA->isReturn(LastInstruction) &&
        !BC.MIA->isTailCall(LastInstruction)) {
      InliningCandidates.insert(Function.getName());
    }
  }

  DEBUG(errs() << "BOLT-DEBUG: " << InliningCandidates.size()
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

void InlineSmallFunctions::inlineCallsInFunction(
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
        totalDynamicCalls += BB->getExecutionCount();
      }
    }
  }

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
        auto FunctionIt = FunctionByName.find(Target->getSymbol().getName());
        if (FunctionIt != FunctionByName.end()) {
          auto &TargetFunction = *FunctionIt->second;
          bool CallToInlineableFunction =
            InliningCandidates.count(TargetFunction.getName());

          totalInlineableCalls +=
            CallToInlineableFunction * BB->getExecutionCount();

          if (CallToInlineableFunction &&
              TargetFunction.getSize() + ExtraSize
              + Function.estimateHotSize() < Function.getMaxSize()) {
            auto NextInstIt = std::next(InstIt);
            inlineCall(BC, *BB, &Inst, *TargetFunction.begin());
            DEBUG(errs() << "BOLT-DEBUG: Inlining call to "
                         << TargetFunction.getName() << " in "
                         << Function.getName() << "\n");
            InstIt = NextInstIt;
            ExtraSize += TargetFunction.getSize();
            inlinedDynamicCalls += BB->getExecutionCount();
            continue;
          }
        }
      }

      ++InstIt;
    }
  }
}

void InlineSmallFunctions::runOnFunctions(
    BinaryContext &BC,
    std::map<uint64_t, BinaryFunction> &BFs) {
  for (auto &It : BFs) {
    FunctionByName[It.second.getName()] = &It.second;
  }
  findInliningCandidates(BC, BFs);
  uint32_t ConsideredFunctions = 0;
  for (auto &It : BFs) {
    if (ConsideredFunctions == kMaxFunctions)
      break;
    inlineCallsInFunction(BC, It.second);
    ++ConsideredFunctions;
  }
  DEBUG(errs() << "BOLT-DEBUG: Inlined " << inlinedDynamicCalls << " of "
               << totalDynamicCalls << " function calls in the profile.\n");
  DEBUG(errs() << "BOLT-DEBUG: Inlined calls represent "
               << (100.0 * inlinedDynamicCalls / totalInlineableCalls)
               << "% of all inlineable calls in the profile.\n");
}

} // namespace bolt
} // namespace llvm
