//===- ConstantProp.cpp - Code to perform Simple Constant Propogation -----===//
//
// This file implements constant propogation and merging:
//
// Specifically, this:
//   * Converts instructions like "add int 1, 2" into 3
//
// Notice that:
//   * This pass has a habit of making definitions be dead.  It is a good idea
//     to to run a DIE pass sometime after running this pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/ConstantProp.h"
#include "llvm/ConstantHandling.h"
#include "llvm/Function.h"
#include "llvm/BasicBlock.h"
#include "llvm/iTerminators.h"
#include "llvm/Pass.h"
#include "llvm/Support/InstIterator.h"
#include <set>

// FIXME: ConstantFoldInstruction & ConstantFoldTerminator should be moved out
// to the Transformations library.

// ConstantFoldInstruction - If an instruction references constants, try to fold
// them together...
//
bool doConstantPropogation(BasicBlock *BB, BasicBlock::iterator &II) {
  Instruction *Inst = *II;
  if (Constant *C = ConstantFoldInstruction(Inst)) {
    // Replaces all of the uses of a variable with uses of the constant.
    Inst->replaceAllUsesWith(C);
    
    // Remove the instruction from the basic block...
    delete BB->getInstList().remove(II);
    return true;
  }

  return false;
}

// ConstantFoldTerminator - If a terminator instruction is predicated on a
// constant value, convert it into an unconditional branch to the constant
// destination.
//
bool ConstantFoldTerminator(BasicBlock *BB, BasicBlock::iterator &II,
                            TerminatorInst *T) {
  // Branch - See if we are conditional jumping on constant
  if (BranchInst *BI = dyn_cast<BranchInst>(T)) {
    if (BI->isUnconditional()) return false;  // Can't optimize uncond branch
    BasicBlock *Dest1 = cast<BasicBlock>(BI->getOperand(0));
    BasicBlock *Dest2 = cast<BasicBlock>(BI->getOperand(1));

    if (ConstantBool *Cond = dyn_cast<ConstantBool>(BI->getCondition())) {
      // Are we branching on constant?
      // YES.  Change to unconditional branch...
      BasicBlock *Destination = Cond->getValue() ? Dest1 : Dest2;
      BasicBlock *OldDest     = Cond->getValue() ? Dest2 : Dest1;

      //cerr << "Function: " << T->getParent()->getParent() 
      //     << "\nRemoving branch from " << T->getParent() 
      //     << "\n\nTo: " << OldDest << endl;

      // Let the basic block know that we are letting go of it.  Based on this,
      // it will adjust it's PHI nodes.
      assert(BI->getParent() && "Terminator not inserted in block!");
      OldDest->removePredecessor(BI->getParent());

      // Set the unconditional destination, and change the insn to be an
      // unconditional branch.
      BI->setUnconditionalDest(Destination);
      II = BB->end()-1;  // Update instruction iterator!
      return true;
    }
#if 0
    // FIXME: TODO: This doesn't work if the destination has PHI nodes with
    // different incoming values on each branch!
    //
    else if (Dest2 == Dest1) {       // Conditional branch to same location?
      // This branch matches something like this:  
      //     br bool %cond, label %Dest, label %Dest
      // and changes it into:  br label %Dest

      // Let the basic block know that we are letting go of one copy of it.
      assert(BI->getParent() && "Terminator not inserted in block!");
      Dest1->removePredecessor(BI->getParent());

      // Change a conditional branch to unconditional.
      BI->setUnconditionalDest(Dest1);
      return true;
    }
#endif
  }
  return false;
}



namespace {
  struct ConstantPropogation : public FunctionPass {
    const char *getPassName() const { return "Simple Constant Propogation"; }

    inline bool runOnFunction(Function *F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.preservesCFG();
    }
  };
}

Pass *createConstantPropogationPass() {
  return new ConstantPropogation();
}


bool ConstantPropogation::runOnFunction(Function *F) {
  // Initialize the worklist to all of the instructions ready to process...
  std::set<Instruction*> WorkList(inst_begin(F), inst_end(F));
  bool Changed = false;

  while (!WorkList.empty()) {
    Instruction *I = *WorkList.begin();
    WorkList.erase(WorkList.begin());    // Get an element from the worklist...

    if (!I->use_empty())                 // Don't muck with dead instructions...
      if (Constant *C = ConstantFoldInstruction(I)) {
        // Add all of the users of this instruction to the worklist, they might
        // be constant propogatable now...
        for (Value::use_iterator UI = I->use_begin(), UE = I->use_end();
             UI != UE; ++UI)
          WorkList.insert(cast<Instruction>(*UI));
        
        // Replace all of the uses of a variable with uses of the constant.
        I->replaceAllUsesWith(C);
        
        // We made a change to the function...
        Changed = true;
      }
  }
  return Changed;
}
