//===- ConstantProp.cpp - Code to perform Simple Constant Propagation -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements constant propagation and merging:
//
// Specifically, this:
//   * Converts instructions like "add int 1, 2" into 3
//
// Notice that:
//   * This pass has a habit of making definitions be dead.  It is a good idea
//     to to run a DIE pass sometime after running this pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Constant.h"
#include "llvm/Instruction.h"
#include "llvm/Pass.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/ADT/Statistic.h"
#include <set>
using namespace llvm;

namespace {
  Statistic<> NumInstKilled("constprop", "Number of instructions killed");

  struct ConstantPropagation : public FunctionPass {
    bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
    }
  };

  RegisterPass<ConstantPropagation> X("constprop",
                                      "Simple constant propagation");
}

FunctionPass *llvm::createConstantPropagationPass() {
  return new ConstantPropagation();
}


bool ConstantPropagation::runOnFunction(Function &F) {
  // Initialize the worklist to all of the instructions ready to process...
  std::set<Instruction*> WorkList;
  for(inst_iterator i = inst_begin(F), e = inst_end(F); i != e; ++i) {
      WorkList.insert(&*i);
  }
  bool Changed = false;

  while (!WorkList.empty()) {
    Instruction *I = *WorkList.begin();
    WorkList.erase(WorkList.begin());    // Get an element from the worklist...

    if (!I->use_empty())                 // Don't muck with dead instructions...
      if (Constant *C = ConstantFoldInstruction(I)) {
        // Add all of the users of this instruction to the worklist, they might
        // be constant propagatable now...
        for (Value::use_iterator UI = I->use_begin(), UE = I->use_end();
             UI != UE; ++UI)
          WorkList.insert(cast<Instruction>(*UI));

        // Replace all of the uses of a variable with uses of the constant.
        I->replaceAllUsesWith(C);

        // Remove the dead instruction.
        WorkList.erase(I);
        I->getParent()->getInstList().erase(I);

        // We made a change to the function...
        Changed = true;
        ++NumInstKilled;
      }
  }
  return Changed;
}
