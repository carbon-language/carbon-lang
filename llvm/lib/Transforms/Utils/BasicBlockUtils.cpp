//===-- BasicBlockUtils.cpp - BasicBlock Utilities -------------------------==//
//
// This family of functions perform manipulations on basic blocks, and
// instructions contained within basic blocks.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Function.h"
#include "llvm/iTerminators.h"
#include "llvm/Constant.h"
#include "llvm/Type.h"
#include <algorithm>

// ReplaceInstWithValue - Replace all uses of an instruction (specified by BI)
// with a value, then remove and delete the original instruction.
//
void ReplaceInstWithValue(BasicBlock::InstListType &BIL,
                          BasicBlock::iterator &BI, Value *V) {
  Instruction &I = *BI;
  // Replaces all of the uses of the instruction with uses of the value
  I.replaceAllUsesWith(V);

  std::string OldName = I.getName();
  
  // Delete the unnecessary instruction now...
  BI = BIL.erase(BI);

  // Make sure to propagate a name if there is one already...
  if (OldName.size() && !V->hasName())
    V->setName(OldName, &BIL.getParent()->getSymbolTable());
}


// ReplaceInstWithInst - Replace the instruction specified by BI with the
// instruction specified by I.  The original instruction is deleted and BI is
// updated to point to the new instruction.
//
void ReplaceInstWithInst(BasicBlock::InstListType &BIL,
                         BasicBlock::iterator &BI, Instruction *I) {
  assert(I->getParent() == 0 &&
         "ReplaceInstWithInst: Instruction already inserted into basic block!");

  // Insert the new instruction into the basic block...
  BasicBlock::iterator New = BIL.insert(BI, I);

  // Replace all uses of the old instruction, and delete it.
  ReplaceInstWithValue(BIL, BI, I);

  // Move BI back to point to the newly inserted instruction
  BI = New;
}

// ReplaceInstWithInst - Replace the instruction specified by From with the
// instruction specified by To.
//
void ReplaceInstWithInst(Instruction *From, Instruction *To) {
  BasicBlock::iterator BI(From);
  ReplaceInstWithInst(From->getParent()->getInstList(), BI, To);
}

// RemoveSuccessor - Change the specified terminator instruction such that its
// successor #SuccNum no longer exists.  Because this reduces the outgoing
// degree of the current basic block, the actual terminator instruction itself
// may have to be changed.  In the case where the last successor of the block is
// deleted, a return instruction is inserted in its place which can cause a
// surprising change in program behavior if it is not expected.
//
void RemoveSuccessor(TerminatorInst *TI, unsigned SuccNum) {
  assert(SuccNum < TI->getNumSuccessors() &&
         "Trying to remove a nonexistant successor!");

  // If our old successor block contains any PHI nodes, remove the entry in the
  // PHI nodes that comes from this branch...
  //
  BasicBlock *BB = TI->getParent();
  TI->getSuccessor(SuccNum)->removePredecessor(BB);

  TerminatorInst *NewTI = 0;
  switch (TI->getOpcode()) {
  case Instruction::Br:
    // If this is a conditional branch... convert to unconditional branch.
    if (TI->getNumSuccessors() == 2) {
      cast<BranchInst>(TI)->setUnconditionalDest(TI->getSuccessor(1-SuccNum));
    } else {                    // Otherwise convert to a return instruction...
      Value *RetVal = 0;
      
      // Create a value to return... if the function doesn't return null...
      if (BB->getParent()->getReturnType() != Type::VoidTy)
        RetVal = Constant::getNullValue(BB->getParent()->getReturnType());

      // Create the return...
      NewTI = new ReturnInst(RetVal);
    }
    break;   

  case Instruction::Invoke:    // Should convert to call
  case Instruction::Switch:    // Should remove entry
  default:
  case Instruction::Ret:       // Cannot happen, has no successors!
    assert(0 && "Unhandled terminator instruction type in RemoveSuccessor!");
    abort();
  }

  if (NewTI)   // If it's a different instruction, replace.
    ReplaceInstWithInst(TI, NewTI);
}
