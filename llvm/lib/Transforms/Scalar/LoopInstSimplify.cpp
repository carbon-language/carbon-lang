//===- LoopInstSimplify.cpp - Loop Instruction Simplification Pass --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs lightweight instruction simplification on loop bodies.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-instsimplify"
#include "llvm/Function.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumSimplified, "Number of redundant instructions simplified");

namespace {
  class LoopInstSimplify : public FunctionPass {
  public:
    static char ID; // Pass ID, replacement for typeid
    LoopInstSimplify() : FunctionPass(ID) {
      initializeLoopInstSimplifyPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function&);

    virtual void getAnalysisUsage(AnalysisUsage& AU) const {
      AU.setPreservesCFG();
      AU.addRequired<LoopInfo>();
      AU.addPreserved<LoopInfo>();
      AU.addPreservedID(LCSSAID);
    }
  };
}
  
char LoopInstSimplify::ID = 0;
INITIALIZE_PASS_BEGIN(LoopInstSimplify, "loop-instsimplify",
                "Simplify instructions in loops", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_DEPENDENCY(LCSSA)
INITIALIZE_PASS_END(LoopInstSimplify, "loop-instsimplify",
                "Simplify instructions in loops", false, false)

Pass* llvm::createLoopInstSimplifyPass() {
  return new LoopInstSimplify();
}

bool LoopInstSimplify::runOnFunction(Function& F) {
  DominatorTree* DT = getAnalysisIfAvailable<DominatorTree>();
  LoopInfo* LI = &getAnalysis<LoopInfo>();
  const TargetData* TD = getAnalysisIfAvailable<TargetData>();

  bool Changed = false;
  bool LocalChanged;
  do {
    LocalChanged = false;

    for (df_iterator<BasicBlock*> DI = df_begin(&F.getEntryBlock()),
         DE = df_end(&F.getEntryBlock()); DI != DE; ++DI)
      for (BasicBlock::iterator BI = DI->begin(), BE = DI->end(); BI != BE;) {
        Instruction* I = BI++;
        // Don't bother simplifying unused instructions.
        if (!I->use_empty()) {
          Value* V = SimplifyInstruction(I, TD, DT);
          if (V && LI->replacementPreservesLCSSAForm(I, V)) {
            I->replaceAllUsesWith(V);
            LocalChanged = true;
            ++NumSimplified;
          }
        }
        LocalChanged |= RecursivelyDeleteTriviallyDeadInstructions(I);
      }

    Changed |= LocalChanged;
  } while (LocalChanged);

  return Changed;
}
