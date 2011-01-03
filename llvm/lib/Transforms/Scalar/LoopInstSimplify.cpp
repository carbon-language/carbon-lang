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
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumSimplified, "Number of redundant instructions simplified");

namespace {
  class LoopInstSimplify : public LoopPass {
  public:
    static char ID; // Pass ID, replacement for typeid
    LoopInstSimplify() : LoopPass(ID) {
      initializeLoopInstSimplifyPass(*PassRegistry::getPassRegistry());
    }

    bool runOnLoop(Loop*, LPPassManager&);

    virtual void getAnalysisUsage(AnalysisUsage& AU) const {
      AU.setPreservesCFG();
      AU.addRequired<DominatorTree>();
      AU.addPreserved<DominatorTree>();
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

bool LoopInstSimplify::runOnLoop(Loop* L, LPPassManager& LPM) {
  DominatorTree* DT = &getAnalysis<DominatorTree>();
  const LoopInfo* LI = &getAnalysis<LoopInfo>();
  const TargetData* TD = getAnalysisIfAvailable<TargetData>();

  bool Changed = false;
  bool LocalChanged;
  do {
    LocalChanged = false;

    SmallPtrSet<BasicBlock*, 32> Visited;
    SmallVector<BasicBlock*, 32> VisitStack;

    VisitStack.push_back(L->getHeader());

    while (!VisitStack.empty()) {
      BasicBlock* BB = VisitStack.back();
      VisitStack.pop_back();

      if (Visited.count(BB))
        continue;
      Visited.insert(BB);

      for (BasicBlock::iterator BI = BB->begin(), BE = BB->end(); BI != BE;) {
        Instruction* I = BI++;
        // Don't bother simplifying unused instructions.
        if (!I->use_empty()) {
          if (Value* V = SimplifyInstruction(I, TD, DT)) {
            I->replaceAllUsesWith(V);
            LocalChanged = true;
            ++NumSimplified;
          }
        }
        LocalChanged |= RecursivelyDeleteTriviallyDeadInstructions(I);
      }
      Changed |= LocalChanged;

      DomTreeNode* Node = DT->getNode(BB);
      const std::vector<DomTreeNode*>& Children = Node->getChildren();
      for (unsigned i = 0; i < Children.size(); ++i) {
        // Only visit children that are in the same loop.
        BasicBlock* ChildBB = Children[i]->getBlock();
        if (!Visited.count(ChildBB) && LI->getLoopFor(ChildBB) == L)
          VisitStack.push_back(ChildBB);
      }
    }
  } while (LocalChanged);

  // Nothing that SimplifyInstruction() does should invalidate LCSSA form.
  assert(L->isLCSSAForm(*DT));

  return Changed;
}
