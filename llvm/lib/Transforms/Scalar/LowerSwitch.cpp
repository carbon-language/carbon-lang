//===- LowerSwitch.cpp - Eliminate Switch instructions --------------------===//
//
// The LowerSwitch transformation rewrites switch statements with a sequence of
// branches, which allows targets to get away with not implementing the switch
// statement until it is convenient.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Function.h"
#include "llvm/iTerminators.h"
#include "llvm/iOperators.h"
#include "llvm/iPHINode.h"
#include "llvm/Pass.h"
#include "Support/Statistic.h"

namespace {
  Statistic<> NumLowered("lowerswitch", "Number of SwitchInst's replaced");

  /// LowerSwitch Pass - Replace all SwitchInst instructions with chained branch
  /// instructions.  Note that this cannot be a BasicBlock pass because it
  /// modifies the CFG!
  struct LowerSwitch : public FunctionPass {
    bool runOnFunction(Function &F);
    void processSwitchInst(SwitchInst *SI);
  };

  RegisterOpt<LowerSwitch>
  X("lowerswitch", "Lower SwitchInst's to branches");
}

// createLowerSwitchPass - Interface to this file...
Pass *createLowerSwitchPass() {
  return new LowerSwitch();
}

bool LowerSwitch::runOnFunction(Function &F) {
  bool Changed = false;

  for (Function::iterator I = F.begin(), E = F.end(); I != E; ) {
    BasicBlock *Cur = I++; // Advance over block so we don't traverse new blocks

    if (SwitchInst *SI = dyn_cast<SwitchInst>(Cur->getTerminator())) {
      Changed = true;
      processSwitchInst(SI);
    }
  }

  return Changed;
}

// processSwitchInst - Replace the specified switch instruction with a sequence
// of chained basic blocks.  Right now we just insert an incredibly stupid
// linear sequence of branches.  It would be better to do a balanced binary
// search eventually.  FIXME
//
void LowerSwitch::processSwitchInst(SwitchInst *SI) {
  BasicBlock *CurBlock = SI->getParent();
  BasicBlock *OrigBlock = CurBlock;
  Function *F = CurBlock->getParent();
  Value *Val = SI->getOperand(0);  // The value we are switching on...

  // Unlink the switch instruction from it's block.
  CurBlock->getInstList().remove(SI);

  // Expand comparisons for all of the non-default cases...
  for (unsigned i = 2, e = SI->getNumOperands(); i != e; i += 2) {
    // Insert a new basic block after the current one...
    BasicBlock *NextBlock;
    if (i != e-2) {
      NextBlock = new BasicBlock("switchblock");
      F->getBasicBlockList().insert(CurBlock->getNext(), NextBlock);
    } else {   // Last case, if it's not the value, go to default block.
      NextBlock = cast<BasicBlock>(SI->getDefaultDest());
    }

    // Make the seteq instruction...
    Instruction *Comp = new SetCondInst(Instruction::SetEQ, Val,
                                        SI->getOperand(i), "switchcase");
    CurBlock->getInstList().push_back(Comp);

    // Make the conditional branch...
    BasicBlock *Succ = cast<BasicBlock>(SI->getOperand(i+1));
    Instruction *Br = new BranchInst(Succ, NextBlock, Comp);
    CurBlock->getInstList().push_back(Br);

    // If there were any PHI nodes in this success, rewrite one entry from
    // OrigBlock to come from CurBlock.
    for (BasicBlock::iterator I = Succ->begin();
         PHINode *PN = dyn_cast<PHINode>(I); ++I) {
      int BlockIdx = PN->getBasicBlockIndex(OrigBlock);
      assert(BlockIdx != -1 && "Switch didn't go to this successor??");
      PN->setIncomingBlock((unsigned)BlockIdx, CurBlock);
    }

    CurBlock = NextBlock;  // Move on to the next condition
  }


  // We are now done with the switch instruction, delete it.
  delete SI;
}
