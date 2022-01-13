//===- CostModel.cpp ------ Cost Model Analysis ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the cost model analysis. It provides a very basic cost
// estimation for LLVM-IR. This analysis uses the services of the codegen
// to approximate the cost of any IR instruction when lowered to machine
// instructions. The cost results are unit-less and the cost number represents
// the throughput of the machine assuming that all loads hit the cache, all
// branches are predicted, etc. The cost numbers can be added in order to
// compare two or more transformation alternatives.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CostModel.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

static cl::opt<TargetTransformInfo::TargetCostKind> CostKind(
    "cost-kind", cl::desc("Target cost kind"),
    cl::init(TargetTransformInfo::TCK_RecipThroughput),
    cl::values(clEnumValN(TargetTransformInfo::TCK_RecipThroughput,
                          "throughput", "Reciprocal throughput"),
               clEnumValN(TargetTransformInfo::TCK_Latency,
                          "latency", "Instruction latency"),
               clEnumValN(TargetTransformInfo::TCK_CodeSize,
                          "code-size", "Code size"),
               clEnumValN(TargetTransformInfo::TCK_SizeAndLatency,
                          "size-latency", "Code size and latency")));


#define CM_NAME "cost-model"
#define DEBUG_TYPE CM_NAME

namespace {
  class CostModelAnalysis : public FunctionPass {

  public:
    static char ID; // Class identification, replacement for typeinfo
    CostModelAnalysis() : FunctionPass(ID), F(nullptr), TTI(nullptr) {
      initializeCostModelAnalysisPass(
        *PassRegistry::getPassRegistry());
    }

    /// Returns the expected cost of the instruction.
    /// Returns -1 if the cost is unknown.
    /// Note, this method does not cache the cost calculation and it
    /// can be expensive in some cases.
    InstructionCost getInstructionCost(const Instruction *I) const {
      return TTI->getInstructionCost(I, TargetTransformInfo::TCK_RecipThroughput);
    }

  private:
    void getAnalysisUsage(AnalysisUsage &AU) const override;
    bool runOnFunction(Function &F) override;
    void print(raw_ostream &OS, const Module*) const override;

    /// The function that we analyze.
    Function *F;
    /// Target information.
    const TargetTransformInfo *TTI;
  };
}  // End of anonymous namespace

// Register this pass.
char CostModelAnalysis::ID = 0;
static const char cm_name[] = "Cost Model Analysis";
INITIALIZE_PASS_BEGIN(CostModelAnalysis, CM_NAME, cm_name, false, true)
INITIALIZE_PASS_END  (CostModelAnalysis, CM_NAME, cm_name, false, true)

FunctionPass *llvm::createCostModelAnalysisPass() {
  return new CostModelAnalysis();
}

void
CostModelAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

bool
CostModelAnalysis::runOnFunction(Function &F) {
 this->F = &F;
 auto *TTIWP = getAnalysisIfAvailable<TargetTransformInfoWrapperPass>();
 TTI = TTIWP ? &TTIWP->getTTI(F) : nullptr;

 return false;
}

void CostModelAnalysis::print(raw_ostream &OS, const Module*) const {
  if (!F)
    return;

  for (BasicBlock &B : *F) {
    for (Instruction &Inst : B) {
      InstructionCost Cost = TTI->getInstructionCost(&Inst, CostKind);
      if (auto CostVal = Cost.getValue())
        OS << "Cost Model: Found an estimated cost of " << *CostVal;
      else
        OS << "Cost Model: Invalid cost";

      OS << " for instruction: " << Inst << "\n";
    }
  }
}

PreservedAnalyses CostModelPrinterPass::run(Function &F,
                                            FunctionAnalysisManager &AM) {
  auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  OS << "Cost Model for function '" << F.getName() << "'\n";
  for (BasicBlock &B : F) {
    for (Instruction &Inst : B) {
      // TODO: Use a pass parameter instead of cl::opt CostKind to determine
      // which cost kind to print.
      InstructionCost Cost = TTI.getInstructionCost(&Inst, CostKind);
      if (auto CostVal = Cost.getValue())
        OS << "Cost Model: Found an estimated cost of " << *CostVal;
      else
        OS << "Cost Model: Invalid cost";

      OS << " for instruction: " << Inst << "\n";
    }
  }
  return PreservedAnalyses::all();
}
