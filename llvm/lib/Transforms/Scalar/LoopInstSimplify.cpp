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
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Support/Debug.h"
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

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<LoopInfo>();
      AU.addRequiredID(LoopSimplifyID);
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

bool LoopInstSimplify::runOnLoop(Loop *L, LPPassManager &LPM) {
  DominatorTree *DT = getAnalysisIfAvailable<DominatorTree>();
  LoopInfo *LI = &getAnalysis<LoopInfo>();
  const TargetData *TD = getAnalysisIfAvailable<TargetData>();

  SmallVector<BasicBlock*, 8> ExitBlocks;
  L->getUniqueExitBlocks(ExitBlocks);
  array_pod_sort(ExitBlocks.begin(), ExitBlocks.end());

  SmallPtrSet<const Instruction*, 8> S1, S2, *ToSimplify = &S1, *Next = &S2;

  SmallVector<BasicBlock*, 16> VisitStack;
  SmallPtrSet<BasicBlock*, 32> Visited;

  bool Changed = false;
  bool LocalChanged;
  do {
    LocalChanged = false;

    VisitStack.clear();
    Visited.clear();

    VisitStack.push_back(L->getHeader());

    while (!VisitStack.empty()) {
      BasicBlock *BB = VisitStack.back();
      VisitStack.pop_back();

      // Simplify instructions in the current basic block.
      for (BasicBlock::iterator BI = BB->begin(), BE = BB->end(); BI != BE;) {
        Instruction *I = BI++;

        // The first time through the loop ToSimplify is empty and we try to
        // simplify all instructions. On later iterations ToSimplify is not
        // empty and we only bother simplifying instructions that are in it.
        if (!ToSimplify->empty() && !ToSimplify->count(I))
          continue;

        // Don't bother simplifying unused instructions.
        if (!I->use_empty()) {
          Value *V = SimplifyInstruction(I, TD, DT);
          if (V && LI->replacementPreservesLCSSAForm(I, V)) {
            // Mark all uses for resimplification next time round the loop.
            for (Value::use_iterator UI = I->use_begin(), UE = I->use_end();
                 UI != UE; ++UI)
              Next->insert(cast<Instruction>(*UI));

            I->replaceAllUsesWith(V);
            LocalChanged = true;
            ++NumSimplified;
          }
        }
        LocalChanged |= RecursivelyDeleteTriviallyDeadInstructions(I);
      }

      // Add all successors to the worklist, except for loop exit blocks.
      for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB); SI != SE;
           ++SI) {
        BasicBlock *SuccBB = *SI;
        bool IsExitBlock = std::binary_search(ExitBlocks.begin(),
                                             ExitBlocks.end(), SuccBB);
        if (!IsExitBlock && Visited.insert(SuccBB))
          VisitStack.push_back(SuccBB);
      }
    }

    // Place the list of instructions to simplify on the next loop iteration
    // into ToSimplify.
    std::swap(ToSimplify, Next);
    Next->clear();

    Changed |= LocalChanged;
  } while (LocalChanged);

  return Changed;
}
