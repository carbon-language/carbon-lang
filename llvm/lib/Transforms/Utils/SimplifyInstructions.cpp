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
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
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
      const DominatorTree *DT = getAnalysisIfAvailable<DominatorTree>();
      const TargetData *TD = getAnalysisIfAvailable<TargetData>();
      bool Changed = false;
      bool LocalChanged;

      do {
        LocalChanged = false;

        for (df_iterator<BasicBlock*> DI = df_begin(&F.getEntryBlock()),
             DE = df_end(&F.getEntryBlock()); DI != DE; ++DI)
          for (BasicBlock::iterator BI = DI->begin(), BE = DI->end(); BI != BE;) {
            Instruction *I = BI++;
            // Don't bother simplifying unused instructions.
            if (!I->use_empty())
              if (Value *V = SimplifyInstruction(I, TD, DT)) {
                I->replaceAllUsesWith(V);
                LocalChanged = true;
                ++NumSimplified;
              }
            LocalChanged |= RecursivelyDeleteTriviallyDeadInstructions(I);
          }

        Changed |= LocalChanged;
      } while (LocalChanged);

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
