//===- Mem2Reg.cpp - The -mem2reg pass, a wrapper around the Utils lib ----===//
//
// This pass is a simple pass wrapper around the PromoteMemToReg function call
// exposed by the Utils library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/iMemory.h"
#include "llvm/Function.h"
#include "llvm/Target/TargetData.h"
#include "Support/Statistic.h"

namespace {
  Statistic<> NumPromoted("mem2reg", "Number of alloca's promoted");

  struct PromotePass : public FunctionPass {
    // runOnFunction - To run this pass, first we calculate the alloca
    // instructions that are safe for promotion, then we promote each one.
    //
    virtual bool runOnFunction(Function &F);

    // getAnalysisUsage - We need dominance frontiers
    //
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<DominanceFrontier>();
      AU.addRequired<TargetData>();
      AU.setPreservesCFG();
    }
  };

  RegisterOpt<PromotePass> X("mem2reg", "Promote Memory to Register");
}  // end of anonymous namespace

bool PromotePass::runOnFunction(Function &F) {
  std::vector<AllocaInst*> Allocas;
  const TargetData &TD = getAnalysis<TargetData>();

  BasicBlock &BB = F.getEntryNode();  // Get the entry node for the function

  // Find allocas that are safe to promote, by looking at all instructions in
  // the entry node
  for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I)
    if (AllocaInst *AI = dyn_cast<AllocaInst>(I))       // Is it an alloca?
      if (isAllocaPromotable(AI, TD))
        Allocas.push_back(AI);

  if (!Allocas.empty()) {
    PromoteMemToReg(Allocas, getAnalysis<DominanceFrontier>(), TD);
    NumPromoted += Allocas.size();
    return true;
  }
  return false;
}

// createPromoteMemoryToRegister - Provide an entry point to create this pass.
//
Pass *createPromoteMemoryToRegister() {
  return new PromotePass();
}
