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
      const TargetData *TD = getAnalysisIfAvailable<TargetData>();
      const DominatorTree *DT = getAnalysisIfAvailable<DominatorTree>();
      bool Changed = false;

      // Add all interesting instructions to the worklist.
      std::set<Instruction*> Worklist;
      for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
        for (BasicBlock::iterator BI = BB->begin(), BE = BB->end(); BI != BE;) {
          Instruction *I = BI++;
          // Zap any dead instructions.
          if (isInstructionTriviallyDead(I)) {
            I->eraseFromParent();
            Changed = true;
            continue;
          }
          // Add all others to the worklist.
          Worklist.insert(I);
        }

      // Simplify everything in the worklist until the cows come home.
      while (!Worklist.empty()) {
        Instruction *I = *Worklist.begin();
        Worklist.erase(Worklist.begin());
        Value *V = SimplifyInstruction(I, TD, DT);
        if (!V) continue;

        // This instruction simplifies!  Replace it with its simplification and
        // add all uses to the worklist, since they may now simplify.
        I->replaceAllUsesWith(V);
        for (Value::use_iterator UI = I->use_begin(), UE = I->use_end();
             UI != UE; ++UI)
          // In unreachable code an instruction can use itself, in which case
          // don't add it to the worklist since we are about to erase it.
          if (*UI != I) Worklist.insert(cast<Instruction>(*UI));
        if (isInstructionTriviallyDead(I))
          I->eraseFromParent();
        ++NumSimplified;
        Changed = true;
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
