//===-- Local.cpp - Functions to perform local transformations ------------===//
//
// This family of functions perform various local transformations to the
// program.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Local.h"
#include "llvm/iTerminators.h"
#include "llvm/ConstantHandling.h"

//===----------------------------------------------------------------------===//
//  Local constant propogation...
//

// ConstantFoldInstruction - If an instruction references constants, try to fold
// them together...
//
bool doConstantPropogation(BasicBlock::iterator &II) {
  if (Constant *C = ConstantFoldInstruction(II)) {
    // Replaces all of the uses of a variable with uses of the constant.
    II->replaceAllUsesWith(C);
    
    // Remove the instruction from the basic block...
    II = II->getParent()->getInstList().erase(II);
    return true;
  }

  return false;
}

// ConstantFoldTerminator - If a terminator instruction is predicated on a
// constant value, convert it into an unconditional branch to the constant
// destination.
//
bool ConstantFoldTerminator(BasicBlock *BB) {
  TerminatorInst *T = BB->getTerminator();
      
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



//===----------------------------------------------------------------------===//
//  Local dead code elimination...
//

bool isInstructionTriviallyDead(Instruction *I) {
  return I->use_empty() && !I->mayWriteToMemory() && !isa<TerminatorInst>(I);
}

// dceInstruction - Inspect the instruction at *BBI and figure out if it's
// [trivially] dead.  If so, remove the instruction and update the iterator
// to point to the instruction that immediately succeeded the original
// instruction.
//
bool dceInstruction(BasicBlock::iterator &BBI) {
  // Look for un"used" definitions...
  if (isInstructionTriviallyDead(BBI)) {
    BBI = BBI->getParent()->getInstList().erase(BBI);   // Bye bye
    return true;
  }
  return false;
}
