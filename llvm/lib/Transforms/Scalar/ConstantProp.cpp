//===- ConstantProp.cpp - Code to perform Constant Propogation ------------===//
//
// This file implements constant propogation and merging:
//
// Specifically, this:
//   * Folds multiple identical constants in the constant pool together
//     Note that if one is named and the other is not, that the result gets the
//     original name.
//   * Converts instructions like "add int %1, %2" into a direct def of %3 in
//     the constant pool
//   * Converts conditional branches on a constant boolean value into direct
//     branches.
//   * Converts phi nodes with one incoming def to the incoming def directly
//   . Converts switch statements with one entry into a test & conditional
//     branch
//   . Converts switches on constant values into an unconditional branch.
//
// Notice that:
//   * This pass has a habit of making definitions be dead.  It is a good idea
//     to to run a DCE pass sometime after running this pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/ConstantProp.h"
#include "llvm/ConstantHandling.h"
#include "llvm/Module.h"
#include "llvm/Function.h"
#include "llvm/BasicBlock.h"
#include "llvm/iTerminators.h"
#include "llvm/iPHINode.h"
#include "llvm/iOther.h"
#include "llvm/Pass.h"

inline static bool 
ConstantFoldUnaryInst(BasicBlock *BB, BasicBlock::iterator &II,
                      UnaryOperator *Op, Constant *D) {
  Constant *ReplaceWith = ConstantFoldUnaryInstruction(Op->getOpcode(), D);

  if (!ReplaceWith) return false;   // Nothing new to change...

  // Replaces all of the uses of a variable with uses of the constant.
  Op->replaceAllUsesWith(ReplaceWith);
  
  // Remove the operator from the list of definitions...
  Op->getParent()->getInstList().remove(II);
  
  // The new constant inherits the old name of the operator...
  if (Op->hasName())
    ReplaceWith->setName(Op->getName(), BB->getParent()->getSymbolTableSure());
  
  // Delete the operator now...
  delete Op;
  return true;
}

inline static bool 
ConstantFoldCast(BasicBlock *BB, BasicBlock::iterator &II,
                 CastInst *CI, Constant *D) {
  Constant *ReplaceWith = ConstantFoldCastInstruction(D, CI->getType());

  if (!ReplaceWith) return false;   // Nothing new to change...

  // Replaces all of the uses of a variable with uses of the constant.
  CI->replaceAllUsesWith(ReplaceWith);
  
  // Remove the cast from the list of definitions...
  CI->getParent()->getInstList().remove(II);
  
  // The new constant inherits the old name of the cast...
  if (CI->hasName())
    ReplaceWith->setName(CI->getName(), BB->getParent()->getSymbolTableSure());
  
  // Delete the cast now...
  delete CI;
  return true;
}

inline static bool 
ConstantFoldBinaryInst(BasicBlock *BB, BasicBlock::iterator &II,
		       BinaryOperator *Op,
		       Constant *D1, Constant *D2) {
  Constant *ReplaceWith = ConstantFoldBinaryInstruction(Op->getOpcode(), D1,D2);
  if (!ReplaceWith) return false;   // Nothing new to change...

  // Replaces all of the uses of a variable with uses of the constant.
  Op->replaceAllUsesWith(ReplaceWith);
  
  // Remove the operator from the list of definitions...
  Op->getParent()->getInstList().remove(II);
  
  // The new constant inherits the old name of the operator...
  if (Op->hasName())
    ReplaceWith->setName(Op->getName(), BB->getParent()->getSymbolTableSure());
  
  // Delete the operator now...
  delete Op;
  return true;
}

inline static bool 
ConstantFoldShiftInst(BasicBlock *BB, BasicBlock::iterator &II,
		      ShiftInst *Op,
		      Constant *D1, Constant *D2) {
  Constant *ReplaceWith = ConstantFoldShiftInstruction(Op->getOpcode(), D1,D2);
  if (!ReplaceWith) return false;   // Nothing new to change...

  // Replaces all of the uses of a variable with uses of the constant.
  Op->replaceAllUsesWith(ReplaceWith);
  
  // Remove the operator from the list of definitions...
  Op->getParent()->getInstList().remove(II);
  
  // The new constant inherits the old name of the operator...
  if (Op->hasName())
    ReplaceWith->setName(Op->getName(), BB->getParent()->getSymbolTableSure());
  
  // Delete the operator now...
  delete Op;
  return true;
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

// ConstantFoldInstruction - If an instruction references constants, try to fold
// them together...
//
bool doConstantPropogation(BasicBlock *BB, BasicBlock::iterator &II) {
  Instruction *Inst = *II;
  if (TerminatorInst *TInst = dyn_cast<TerminatorInst>(Inst)) {
    return ConstantFoldTerminator(BB, II, TInst);
  } else if (Constant *C = ConstantFoldInstruction(Inst)) {
    // Replaces all of the uses of a variable with uses of the constant.
    Inst->replaceAllUsesWith(C);
  
    // Remove the instruction from the basic block...
    delete BB->getInstList().remove(II);
    return true;

  }

  return false;
}

// DoConstPropPass - Propogate constants and do constant folding on instructions
// this returns true if something was changed, false if nothing was changed.
//
static bool DoConstPropPass(Function *F) {
  bool SomethingChanged = false;

  for (Function::iterator BBI = F->begin(); BBI != F->end(); ++BBI) {
    BasicBlock *BB = *BBI;
    for (BasicBlock::iterator I = BB->begin(); I != BB->end(); )
      if (doConstantPropogation(BB, I))
	SomethingChanged = true;
      else
	++I;
  }
  return SomethingChanged;
}

namespace {
  struct ConstantPropogation : public FunctionPass {
    const char *getPassName() const { return "Simple Constant Propogation"; }

    inline bool runOnFunction(Function *F) {
      bool Modified = false;

      // Fold constants until we make no progress...
      while (DoConstPropPass(F)) Modified = true;
      
      return Modified;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      // FIXME: This pass does not preserve the CFG because it folds terminator
      // instructions!
      //AU.preservesCFG();
    }
  };
}

Pass *createConstantPropogationPass() {
  return new ConstantPropogation();
}

