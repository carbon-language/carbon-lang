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

#include "llvm/Optimizations/ConstantProp.h"
#include "llvm/Optimizations/ConstantHandling.h"
#include "llvm/Module.h"
#include "llvm/Method.h"
#include "llvm/BasicBlock.h"
#include "llvm/iTerminators.h"
#include "llvm/iOther.h"
#include "llvm/ConstPoolVals.h"

inline static bool 
ConstantFoldUnaryInst(Method *M, Method::inst_iterator &DI,
                      UnaryOperator *Op, ConstPoolVal *D) {
  ConstPoolVal *ReplaceWith = 
    opt::ConstantFoldUnaryInstruction(Op->getOpcode(), D);

  if (!ReplaceWith) return false;   // Nothing new to change...

  // Replaces all of the uses of a variable with uses of the constant.
  Op->replaceAllUsesWith(ReplaceWith);
  
  // Remove the operator from the list of definitions...
  Op->getParent()->getInstList().remove(DI.getInstructionIterator());
  
  // The new constant inherits the old name of the operator...
  if (Op->hasName())
    ReplaceWith->setName(Op->getName(), M->getSymbolTableSure());
  
  // Delete the operator now...
  delete Op;
  return true;
}

inline static bool 
ConstantFoldCast(Method *M, Method::inst_iterator &DI,
                 CastInst *CI, ConstPoolVal *D) {
  ConstPoolVal *ReplaceWith = 
    opt::ConstantFoldCastInstruction(D, CI->getType());

  if (!ReplaceWith) return false;   // Nothing new to change...

  // Replaces all of the uses of a variable with uses of the constant.
  CI->replaceAllUsesWith(ReplaceWith);
  
  // Remove the cast from the list of definitions...
  CI->getParent()->getInstList().remove(DI.getInstructionIterator());
  
  // The new constant inherits the old name of the cast...
  if (CI->hasName())
    ReplaceWith->setName(CI->getName(), M->getSymbolTableSure());
  
  // Delete the cast now...
  delete CI;
  return true;
}

inline static bool 
ConstantFoldBinaryInst(Method *M, Method::inst_iterator &DI,
		       BinaryOperator *Op,
		       ConstPoolVal *D1, ConstPoolVal *D2) {
  ConstPoolVal *ReplaceWith =
    opt::ConstantFoldBinaryInstruction(Op->getOpcode(), D1, D2);
  if (!ReplaceWith) return false;   // Nothing new to change...

  // Replaces all of the uses of a variable with uses of the constant.
  Op->replaceAllUsesWith(ReplaceWith);
  
  // Remove the operator from the list of definitions...
  Op->getParent()->getInstList().remove(DI.getInstructionIterator());
  
  // The new constant inherits the old name of the operator...
  if (Op->hasName())
    ReplaceWith->setName(Op->getName(), M->getSymbolTableSure());
  
  // Delete the operator now...
  delete Op;
  return true;
}

// ConstantFoldTerminator - If a terminator instruction is predicated on a
// constant value, convert it into an unconditional branch to the constant
// destination.
//
bool opt::ConstantFoldTerminator(TerminatorInst *T) {
  // Branch - See if we are conditional jumping on constant
  if (BranchInst *BI = dyn_cast<BranchInst>(T)) {
    if (BI->isUnconditional()) return false;  // Can't optimize uncond branch
    BasicBlock *Dest1 = cast<BasicBlock>(BI->getOperand(0));
    BasicBlock *Dest2 = cast<BasicBlock>(BI->getOperand(1));

    if (ConstPoolBool *Cond = dyn_cast<ConstPoolBool>(BI->getCondition())) {
      // Are we branching on constant?
      // YES.  Change to unconditional branch...
      BasicBlock *Destination = Cond->getValue() ? Dest1 : Dest2;
      BasicBlock *OldDest     = Cond->getValue() ? Dest2 : Dest1;

      //cerr << "Method: " << T->getParent()->getParent() 
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

// ConstantFoldInstruction - If an instruction references constants, try to fold
// them together...
//
inline static bool 
ConstantFoldInstruction(Method *M, Method::inst_iterator &II) {
  Instruction *Inst = *II;
  if (BinaryOperator *BInst = dyn_cast<BinaryOperator>(Inst)) {
    ConstPoolVal *D1 = dyn_cast<ConstPoolVal>(Inst->getOperand(0));
    ConstPoolVal *D2 = dyn_cast<ConstPoolVal>(Inst->getOperand(1));

    if (D1 && D2)
      return ConstantFoldBinaryInst(M, II, cast<BinaryOperator>(Inst), D1, D2);

  } else if (CastInst *CI = dyn_cast<CastInst>(Inst)) {
    ConstPoolVal *D = dyn_cast<ConstPoolVal>(CI->getOperand(0));
    if (D) return ConstantFoldCast(M, II, CI, D);
                                         
  } else if (UnaryOperator *UInst = dyn_cast<UnaryOperator>(Inst)) {
    ConstPoolVal *D = dyn_cast<ConstPoolVal>(UInst->getOperand(0));
    if (D) return ConstantFoldUnaryInst(M, II, UInst, D);
  } else if (TerminatorInst *TInst = dyn_cast<TerminatorInst>(Inst)) {
    return opt::ConstantFoldTerminator(TInst);

  } else if (PHINode *PN = dyn_cast<PHINode>(Inst)) {
    // If it's a PHI node and only has one operand
    // Then replace it directly with that operand.
    assert(PN->getOperand(0) && "PHI Node must have at least one operand!");
    if (PN->getNumOperands() == 1) {    // If the PHI Node has exactly 1 operand
      Value *V = PN->getOperand(0);
      PN->replaceAllUsesWith(V);                 // Replace all uses of this PHI
                                                 // Unlink from basic block
      PN->getParent()->getInstList().remove(II.getInstructionIterator());
      if (PN->hasName())                         // Inherit PHINode name
	V->setName(PN->getName(), M->getSymbolTableSure());
      delete PN;                                 // Finally, delete the node...
      return true;
    }
  }
  return false;
}

// DoConstPropPass - Propogate constants and do constant folding on instructions
// this returns true if something was changed, false if nothing was changed.
//
static bool DoConstPropPass(Method *M) {
  bool SomethingChanged = false;

#if 1
  Method::inst_iterator It = M->inst_begin();
  while (It != M->inst_end())
    if (ConstantFoldInstruction(M, It)) {
      SomethingChanged = true;  // If returned true, iter is already incremented

      // Incrementing the iterator in an unchecked manner could mess up the
      // internals of 'It'.  To make sure everything is happy, tell it we might
      // have broken it.
      It.resyncInstructionIterator();
    } else {
      ++It;
    }
#else
  for (Method::iterator BBIt = M->begin(); BBIt != M->end(); ++BBIt) {
    BasicBlock *BB = *BBIt;

    reduce_apply_bool(BB->begin(), BB->end(),
		      bind1st(ConstantFoldInstruction, M));
  }
#endif
  return SomethingChanged;
}


// returns true on failure, false on success...
//
bool opt::ConstantPropogation::doConstantPropogation(Method *M) {
  bool Modified = false;

  // Fold constants until we make no progress...
  while (DoConstPropPass(M)) Modified = true;

  return Modified;
}
