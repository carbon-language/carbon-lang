//===- LowerSelect.cpp - Transform select insts to branches ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass lowers select instructions into conditional branches for targets
// that do not have conditional moves or that have not implemented the select
// instruction yet.
//
// Note that this pass could be improved.  In particular it turns every select
// instruction into a new conditional branch, even though some common cases have
// select instructions on the same predicate next to each other.  It would be
// better to use the same branch for the whole group of selects.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Type.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;

namespace {
  /// LowerSelect - Turn select instructions into conditional branches.
  ///
  class VISIBILITY_HIDDEN LowerSelect : public FunctionPass {
    bool OnlyFP;   // Only lower FP select instructions?
  public:
    static char ID; // Pass identification, replacement for typeid
    LowerSelect(bool onlyfp = false) : FunctionPass((intptr_t)&ID), 
      OnlyFP(onlyfp) {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      // This certainly destroys the CFG.
      // This is a cluster of orthogonal Transforms:
      AU.addPreserved<UnifyFunctionExitNodes>();
      AU.addPreservedID(PromoteMemoryToRegisterID);
      AU.addPreservedID(LowerSwitchID);
      AU.addPreservedID(LowerInvokePassID);
      AU.addPreservedID(LowerAllocationsID);
    }

    bool runOnFunction(Function &F);
  };

  char LowerSelect::ID = 0;
  RegisterPass<LowerSelect>
  X("lowerselect", "Lower select instructions to branches");
}

// Publically exposed interface to pass...
const PassInfo *llvm::LowerSelectID = X.getPassInfo();
//===----------------------------------------------------------------------===//
// This pass converts SelectInst instructions into conditional branch and PHI
// instructions.  If the OnlyFP flag is set to true, then only floating point
// select instructions are lowered.
//
FunctionPass *llvm::createLowerSelectPass(bool OnlyFP) {
  return new LowerSelect(OnlyFP);
}


bool LowerSelect::runOnFunction(Function &F) {
  bool Changed = false;
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
      if (SelectInst *SI = dyn_cast<SelectInst>(I))
        if (!OnlyFP || SI->getType()->isFloatingPoint()) {
          // Split this basic block in half right before the select instruction.
          BasicBlock *NewCont =
            BB->splitBasicBlock(I, BB->getName()+".selectcont");

          // Make the true block, and make it branch to the continue block.
          BasicBlock *NewTrue = new BasicBlock(BB->getName()+".selecttrue",
                                               BB->getParent(), NewCont);
          new BranchInst(NewCont, NewTrue);

          // Make the unconditional branch in the incoming block be a
          // conditional branch on the select predicate.
          BB->getInstList().erase(BB->getTerminator());
          new BranchInst(NewTrue, NewCont, SI->getCondition(), BB);

          // Create a new PHI node in the cont block with the entries we need.
          PHINode *PN = new PHINode(SI->getType(), "", NewCont->begin());
          PN->takeName(SI);
          PN->addIncoming(SI->getTrueValue(), NewTrue);
          PN->addIncoming(SI->getFalseValue(), BB);

          // Use the PHI instead of the select.
          SI->replaceAllUsesWith(PN);
          NewCont->getInstList().erase(SI);

          Changed = true;
          break; // This block is done with.
        }
    }
  return Changed;
}
