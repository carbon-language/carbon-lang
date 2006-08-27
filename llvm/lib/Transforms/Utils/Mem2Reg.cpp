//===- Mem2Reg.cpp - The -mem2reg pass, a wrapper around the Utils lib ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is a simple pass wrapper around the PromoteMemToReg function call
// exposed by the Utils library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;

namespace {
  Statistic<> NumPromoted("mem2reg", "Number of alloca's promoted");

  struct VISIBILITY_HIDDEN PromotePass : public FunctionPass {
    // runOnFunction - To run this pass, first we calculate the alloca
    // instructions that are safe for promotion, then we promote each one.
    //
    virtual bool runOnFunction(Function &F);

    // getAnalysisUsage - We need dominance frontiers
    //
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<DominatorTree>();
      AU.addRequired<DominanceFrontier>();
      AU.addRequired<TargetData>();
      AU.setPreservesCFG();
      // This is a cluster of orthogonal Transforms	
      AU.addPreserved<UnifyFunctionExitNodes>();
      AU.addPreservedID(LowerSelectID);
      AU.addPreservedID(LowerSwitchID);
      AU.addPreservedID(LowerInvokePassID);
      AU.addPreservedID(LowerAllocationsID);
    }
  };

  RegisterPass<PromotePass> X("mem2reg", "Promote Memory to Register");
}  // end of anonymous namespace

bool PromotePass::runOnFunction(Function &F) {
  std::vector<AllocaInst*> Allocas;
  const TargetData &TD = getAnalysis<TargetData>();

  BasicBlock &BB = F.getEntryBlock();  // Get the entry node for the function

  bool Changed  = false;

  DominatorTree     &DT = getAnalysis<DominatorTree>();
  DominanceFrontier &DF = getAnalysis<DominanceFrontier>();

  while (1) {
    Allocas.clear();

    // Find allocas that are safe to promote, by looking at all instructions in
    // the entry node
    for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I)
      if (AllocaInst *AI = dyn_cast<AllocaInst>(I))       // Is it an alloca?
        if (isAllocaPromotable(AI, TD))
          Allocas.push_back(AI);

    if (Allocas.empty()) break;

    PromoteMemToReg(Allocas, DT, DF, TD);
    NumPromoted += Allocas.size();
    Changed = true;
  }

  return Changed;
}

// Publically exposed interface to pass...
const PassInfo *llvm::PromoteMemoryToRegisterID = X.getPassInfo();
// createPromoteMemoryToRegister - Provide an entry point to create this pass.
//
FunctionPass *llvm::createPromoteMemoryToRegisterPass() {
  return new PromotePass();
}
