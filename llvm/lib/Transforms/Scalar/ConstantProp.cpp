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

#include "llvm/Module.h"
#include "llvm/Method.h"
#include "llvm/BasicBlock.h"
#include "llvm/iTerminators.h"
#include "llvm/iOther.h"
#include "llvm/ConstPoolVals.h"
#include "llvm/ConstantPool.h"
#include "llvm/Opt/AllOpts.h"
#include "llvm/Opt/ConstantHandling.h"

// Merge identical constant values in the constant pool.
// 
// TODO: We can do better than this simplistic N^2 algorithm...
//
bool DoConstantPoolMerging(Method *M) {
  return DoConstantPoolMerging(M->getConstantPool());
}

bool DoConstantPoolMerging(ConstantPool &CP) {
  bool Modified = false;
  for (ConstantPool::plane_iterator PI = CP.begin(); PI != CP.end(); ++PI) {
    for (ConstantPool::PlaneType::iterator I = (*PI)->begin(); 
	 I != (*PI)->end(); ++I) {
      ConstPoolVal *C = *I;

      ConstantPool::PlaneType::iterator J = I;
      for (++J; J != (*PI)->end(); ++J) {
	if (C->equals(*J)) {
	  Modified = true;
	  // Okay we know that *I == *J.  So now we need to make all uses of *I
	  // point to *J.
	  //
	  C->replaceAllUsesWith(*J);

	  (*PI)->remove(I); // Remove C from constant pool...
	  
	  if (C->hasName() && !(*J)->hasName())  // The merged constant inherits
	    (*J)->setName(C->getName());         // the old name...
	  
	  delete C;                              // Delete the constant itself.
	  break;  // Break out of inner for loop
	}
      }
    }
  }
  return Modified;
}

inline static bool 
ConstantFoldUnaryInst(Method *M, Method::inst_iterator &DI,
                      UnaryOperator *Op, ConstPoolVal *D) {
  ConstPoolVal *ReplaceWith = 
    ConstantFoldUnaryInstruction(Op->getInstType(), D);

  if (!ReplaceWith) return false;   // Nothing new to change...


  // Add the new value to the constant pool...
  M->getConstantPool().insert(ReplaceWith);
  
  // Replaces all of the uses of a variable with uses of the constant.
  Op->replaceAllUsesWith(ReplaceWith);
  
  // Remove the operator from the list of definitions...
  Op->getParent()->getInstList().remove(DI.getInstructionIterator());
  
  // The new constant inherits the old name of the operator...
  if (Op->hasName()) ReplaceWith->setName(Op->getName());
  
  // Delete the operator now...
  delete Op;
  return true;
}

inline static bool 
ConstantFoldBinaryInst(Method *M, Method::inst_iterator &DI,
		       BinaryOperator *Op,
		       ConstPoolVal *D1, ConstPoolVal *D2) {
  ConstPoolVal *ReplaceWith =
    ConstantFoldBinaryInstruction(Op->getInstType(), D1, D2);
  if (!ReplaceWith) return false;   // Nothing new to change...

  // Add the new value to the constant pool...
  M->getConstantPool().insert(ReplaceWith);
  
  // Replaces all of the uses of a variable with uses of the constant.
  Op->replaceAllUsesWith(ReplaceWith);
  
  // Remove the operator from the list of definitions...
  Op->getParent()->getInstList().remove(DI.getInstructionIterator());
  
  // The new constant inherits the old name of the operator...
  if (Op->hasName()) ReplaceWith->setName(Op->getName());
  
  // Delete the operator now...
  delete Op;
  return true;
}

// ConstantFoldTerminator - If a terminator instruction is predicated on a
// constant value, convert it into an unconditional branch to the constant
// destination.
//
bool ConstantFoldTerminator(TerminatorInst *T) {
  // Branch - See if we are conditional jumping on constant
  if (T->getInstType() == Instruction::Br) {
    BranchInst *BI = (BranchInst*)T;
    if (BI->isUnconditional()) return false;  // Can't optimize uncond branch
    BasicBlock *Dest1 = BI->getOperand(0)->castBasicBlockAsserting();
    BasicBlock *Dest2 = BI->getOperand(1)->castBasicBlockAsserting();

    if (BI->getOperand(2)->isConstant()) {    // Are we branching on constant?
      // YES.  Change to unconditional branch...
      ConstPoolBool *Cond = (ConstPoolBool*)BI->getOperand(2);
      BasicBlock *Destination = Cond->getValue() ? Dest1 : Dest2;
      BasicBlock *OldDest     = Cond->getValue() ? Dest2 : Dest1;

      //cerr << "Method: " << T->getParent()->getParent() 
      //     << "\nRemoving branch from " << T->getParent() 
      //     << "\n\nTo: " << OldDest << endl;

      // Let the basic block know that we are letting go of it.  Based on this,
      // it will adjust it's PHI nodes.
      assert(BI->getParent() && "Terminator not inserted in block!");
      OldDest->removePredecessor(BI->getParent());

      BI->setOperand(0, Destination);  // Set the unconditional destination
      BI->setOperand(1, 0);            // Clear the conditional destination
      BI->setOperand(2, 0);            // Clear the condition...
      return true;
    } else if (Dest2 == Dest1) {       // Conditional branch to same location?
      // This branch matches something like this:  
      //     br bool %cond, label %Dest, label %Dest
      // and changes it into:  br label %Dest

      // Let the basic block know that we are letting go of one copy of it.
      assert(BI->getParent() && "Terminator not inserted in block!");
      Dest1->removePredecessor(BI->getParent());

      // Nuke the second destination, and the use of the condition variable
      BI->setOperand(1, 0);            // Clear the conditional destination
      BI->setOperand(2, 0);            // Clear the condition...
      return true;
    }
  }
  return false;
}

// ConstantFoldInstruction - If an instruction references constants, try to fold
// them together...
//
inline static bool 
ConstantFoldInstruction(Method *M, Method::inst_iterator &II) {
  Instruction *Inst = *II;
  if (Inst->isBinaryOp()) {
    ConstPoolVal *D1 = Inst->getOperand(0)->castConstant();
    ConstPoolVal *D2 = Inst->getOperand(1)->castConstant();

    if (D1 && D2)
      return ConstantFoldBinaryInst(M, II, (BinaryOperator*)Inst, D1, D2);

  } else if (Inst->isUnaryOp()) {
    ConstPoolVal *D = Inst->getOperand(0)->castConstant();
    if (D) return ConstantFoldUnaryInst(M, II, (UnaryOperator*)Inst, D);
  } else if (Inst->isTerminator()) {
    return ConstantFoldTerminator((TerminatorInst*)Inst);

  } else if (Inst->isPHINode()) {
    PHINode *PN = (PHINode*)Inst; // If it's a PHI node and only has one operand
                                  // Then replace it directly with that operand.
    assert(PN->getOperand(0) && "PHI Node must have at least one operand!");
    if (PN->getOperand(1) == 0) {       // If the PHI Node has exactly 1 operand
      Value *V = PN->getOperand(0);
      PN->replaceAllUsesWith(V);                 // Replace all uses of this PHI
                                                 // Unlink from basic block
      PN->getParent()->getInstList().remove(II.getInstructionIterator());
      if (PN->hasName()) V->setName(PN->getName()); // Inherit PHINode name
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
bool DoConstantPropogation(Method *M) {
  bool Modified = false;

  // Fold constants until we make no progress...
  while (DoConstPropPass(M)) Modified = true;

  // Merge identical constants last: this is important because we may have just
  // introduced constants that already exist!
  //
  Modified |= DoConstantPoolMerging(M->getConstantPool());

  return Modified;
}
