//===- DemoteRegToStack.cpp - Move a virtual register to the stack --------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// This file provide the function DemoteRegToStack().  This function takes a
// virtual register computed by an Instruction and replaces it with a slot in
// the stack frame, allocated via alloca. It returns the pointer to the
// AllocaInst inserted.  After this function is called on an instruction, we are
// guaranteed that the only user of the instruction is a store that is
// immediately after it.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
using namespace llvm;

/// DemoteRegToStack - This function takes a virtual register computed by an
/// Instruction and replaces it with a slot in the stack frame, allocated via
/// alloca.  This allows the CFG to be changed around without fear of
/// invalidating the SSA information for the value.  It returns the pointer to
/// the alloca inserted to create a stack slot for I.
///
AllocaInst* llvm::DemoteRegToStack(Instruction &I) {
  if (I.use_empty()) return 0;                // nothing to do!

  // Create a stack slot to hold the value.
  Function *F = I.getParent()->getParent();
  AllocaInst *Slot = new AllocaInst(I.getType(), 0, I.getName(),
                                    F->getEntryBlock().begin());

  // Change all of the users of the instruction to read from the stack slot
  // instead.
  while (!I.use_empty()) {
    Instruction *U = cast<Instruction>(I.use_back());
    if (PHINode *PN = dyn_cast<PHINode>(U)) {
      // If this is a PHI node, we can't insert a load of the value before the
      // use.  Instead, insert the load in the predecessor block corresponding
      // to the incoming value.
      //
      // Note that if there are multiple edges from a basic block to this PHI
      // node that we'll insert multiple loads.  Since DemoteRegToStack requires
      // a mem2reg pass after it (to produce reasonable code), we don't care.
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
        if (PN->getIncomingValue(i) == &I) {
          // Insert the load into the predecessor block
          Value *V = new LoadInst(Slot, I.getName()+".reload",
                                  PN->getIncomingBlock(i)->getTerminator());
          PN->setIncomingValue(i, V);
        }

    } else {
      // If this is a normal instruction, just insert a load.
      Value *V = new LoadInst(Slot, I.getName()+".reload", U);
      U->replaceUsesOfWith(&I, V);
    }
  }


  // Insert stores of the computed value into the stack slot.  We have to be
  // careful is I is an invoke instruction though, because we can't insert the
  // store AFTER the terminator instruction.
  if (!isa<TerminatorInst>(I)) {
    BasicBlock::iterator InsertPt = &I;
    for (++InsertPt; isa<PHINode>(InsertPt); ++InsertPt)
      /* empty */;   // Don't insert before any PHI nodes.
    new StoreInst(&I, Slot, InsertPt);
  } else {
    // FIXME: We cannot yet demote invoke instructions to the stack, because
    // doing so would require breaking critical edges.  This should be fixed
    // eventually.
    assert(0 &&
           "Cannot demote the value computed by an invoke instruction yet!");
  }

  return Slot;
}
