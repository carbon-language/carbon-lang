//===------ SimplifyInstructions.cpp - Remove redundant instructions ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a utility pass used for testing the InstructionSimplify analysis.
// The analysis is applied to every instruction, and if it simplifies then the
// instruction is replaced by the simplification.  If you are looking for a pass
// that performs serious instruction folding, use the instcombine pass instead.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "instsimplify"
#include "llvm/Function.h"
#include "llvm/Pass.h"
#include "llvm/Type.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Scalar.h"
using namespace llvm;

STATISTIC(NumSimplified, "Number of redundant instructions removed");

namespace {
  struct InstSimplifier : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    InstSimplifier() : FunctionPass(ID) {
      initializeInstSimplifierPass(*PassRegistry::getPassRegistry());
    }

    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
    }

    /// runOnFunction - Remove instructions that simplify.
    bool runOnFunction(Function &F) {
      bool Changed = false;
      const TargetData *TD = getAnalysisIfAvailable<TargetData>();
      const DominatorTree *DT = getAnalysisIfAvailable<DominatorTree>();
      for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
        for (BasicBlock::iterator BI = BB->begin(), BE = BB->end(); BI != BE;) {
          Instruction *I = BI++;
          if (Value *V = SimplifyInstruction(I, TD, DT)) {
            I->replaceAllUsesWith(V);
            I->eraseFromParent();
            Changed = true;
            ++NumSimplified;
          }
        }
      return Changed;
    }
  };
}

char InstSimplifier::ID = 0;
INITIALIZE_PASS(InstSimplifier, "instsimplify", "Remove redundant instructions",
                false, false)
char &llvm::InstructionSimplifierID = InstSimplifier::ID;

// Public interface to the simplify instructions pass.
FunctionPass *llvm::createInstructionSimplifierPass() {
  return new InstSimplifier();
}
