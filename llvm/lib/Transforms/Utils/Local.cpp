//===-- Local.cpp - Functions to perform local transformations ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of functions perform various local transformations to the
// program.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Constants.h"
#include "llvm/GlobalAlias.h"
#include "llvm/GlobalVariable.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/LLVMContext.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/Analysis/ProfileInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//  Local analysis.
//

/// isSafeToLoadUnconditionally - Return true if we know that executing a load
/// from this value cannot trap.  If it is not obviously safe to load from the
/// specified pointer, we do a quick local scan of the basic block containing
/// ScanFrom, to determine if the address is already accessed.
bool llvm::isSafeToLoadUnconditionally(Value *V, Instruction *ScanFrom) {
  // If it is an alloca it is always safe to load from.
  if (isa<AllocaInst>(V)) return true;

  // If it is a global variable it is mostly safe to load from.
  if (const GlobalValue *GV = dyn_cast<GlobalVariable>(V))
    // Don't try to evaluate aliases.  External weak GV can be null.
    return !isa<GlobalAlias>(GV) && !GV->hasExternalWeakLinkage();

  // Otherwise, be a little bit agressive by scanning the local block where we
  // want to check to see if the pointer is already being loaded or stored
  // from/to.  If so, the previous load or store would have already trapped,
  // so there is no harm doing an extra load (also, CSE will later eliminate
  // the load entirely).
  BasicBlock::iterator BBI = ScanFrom, E = ScanFrom->getParent()->begin();

  while (BBI != E) {
    --BBI;

    // If we see a free or a call which may write to memory (i.e. which might do
    // a free) the pointer could be marked invalid.
    if (isa<FreeInst>(BBI) || 
        (isa<CallInst>(BBI) && BBI->mayWriteToMemory() &&
         !isa<DbgInfoIntrinsic>(BBI)))
      return false;

    if (LoadInst *LI = dyn_cast<LoadInst>(BBI)) {
      if (LI->getOperand(0) == V) return true;
    } else if (StoreInst *SI = dyn_cast<StoreInst>(BBI)) {
      if (SI->getOperand(1) == V) return true;
    }
  }
  return false;
}


//===----------------------------------------------------------------------===//
//  Local constant propagation.
//

// ConstantFoldTerminator - If a terminator instruction is predicated on a
// constant value, convert it into an unconditional branch to the constant
// destination.
//
bool llvm::ConstantFoldTerminator(BasicBlock *BB) {
  TerminatorInst *T = BB->getTerminator();

  // Branch - See if we are conditional jumping on constant
  if (BranchInst *BI = dyn_cast<BranchInst>(T)) {
    if (BI->isUnconditional()) return false;  // Can't optimize uncond branch
    BasicBlock *Dest1 = BI->getSuccessor(0);
    BasicBlock *Dest2 = BI->getSuccessor(1);

    if (ConstantInt *Cond = dyn_cast<ConstantInt>(BI->getCondition())) {
      // Are we branching on constant?
      // YES.  Change to unconditional branch...
      BasicBlock *Destination = Cond->getZExtValue() ? Dest1 : Dest2;
      BasicBlock *OldDest     = Cond->getZExtValue() ? Dest2 : Dest1;

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
    } else if (Dest2 == Dest1) {       // Conditional branch to same location?
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
  } else if (SwitchInst *SI = dyn_cast<SwitchInst>(T)) {
    // If we are switching on a constant, we can convert the switch into a
    // single branch instruction!
    ConstantInt *CI = dyn_cast<ConstantInt>(SI->getCondition());
    BasicBlock *TheOnlyDest = SI->getSuccessor(0);  // The default dest
    BasicBlock *DefaultDest = TheOnlyDest;
    assert(TheOnlyDest == SI->getDefaultDest() &&
           "Default destination is not successor #0?");

    // Figure out which case it goes to...
    for (unsigned i = 1, e = SI->getNumSuccessors(); i != e; ++i) {
      // Found case matching a constant operand?
      if (SI->getSuccessorValue(i) == CI) {
        TheOnlyDest = SI->getSuccessor(i);
        break;
      }

      // Check to see if this branch is going to the same place as the default
      // dest.  If so, eliminate it as an explicit compare.
      if (SI->getSuccessor(i) == DefaultDest) {
        // Remove this entry...
        DefaultDest->removePredecessor(SI->getParent());
        SI->removeCase(i);
        --i; --e;  // Don't skip an entry...
        continue;
      }

      // Otherwise, check to see if the switch only branches to one destination.
      // We do this by reseting "TheOnlyDest" to null when we find two non-equal
      // destinations.
      if (SI->getSuccessor(i) != TheOnlyDest) TheOnlyDest = 0;
    }

    if (CI && !TheOnlyDest) {
      // Branching on a constant, but not any of the cases, go to the default
      // successor.
      TheOnlyDest = SI->getDefaultDest();
    }

    // If we found a single destination that we can fold the switch into, do so
    // now.
    if (TheOnlyDest) {
      // Insert the new branch..
      BranchInst::Create(TheOnlyDest, SI);
      BasicBlock *BB = SI->getParent();

      // Remove entries from PHI nodes which we no longer branch to...
      for (unsigned i = 0, e = SI->getNumSuccessors(); i != e; ++i) {
        // Found case matching a constant operand?
        BasicBlock *Succ = SI->getSuccessor(i);
        if (Succ == TheOnlyDest)
          TheOnlyDest = 0;  // Don't modify the first branch to TheOnlyDest
        else
          Succ->removePredecessor(BB);
      }

      // Delete the old switch...
      BB->getInstList().erase(SI);
      return true;
    } else if (SI->getNumSuccessors() == 2) {
      // Otherwise, we can fold this switch into a conditional branch
      // instruction if it has only one non-default destination.
      Value *Cond = new ICmpInst(SI, ICmpInst::ICMP_EQ, SI->getCondition(),
                                 SI->getSuccessorValue(1), "cond");
      // Insert the new branch...
      BranchInst::Create(SI->getSuccessor(1), SI->getSuccessor(0), Cond, SI);

      // Delete the old switch...
      SI->eraseFromParent();
      return true;
    }
  }
  return false;
}


//===----------------------------------------------------------------------===//
//  Local dead code elimination...
//

/// isInstructionTriviallyDead - Return true if the result produced by the
/// instruction is not used, and the instruction has no side effects.
///
bool llvm::isInstructionTriviallyDead(Instruction *I) {
  if (!I->use_empty() || isa<TerminatorInst>(I)) return false;

  // We don't want debug info removed by anything this general.
  if (isa<DbgInfoIntrinsic>(I)) return false;

  if (!I->mayHaveSideEffects()) return true;

  // Special case intrinsics that "may have side effects" but can be deleted
  // when dead.
  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I))
    // Safe to delete llvm.stacksave if dead.
    if (II->getIntrinsicID() == Intrinsic::stacksave)
      return true;
  return false;
}

/// RecursivelyDeleteTriviallyDeadInstructions - If the specified value is a
/// trivially dead instruction, delete it.  If that makes any of its operands
/// trivially dead, delete them too, recursively.
void llvm::RecursivelyDeleteTriviallyDeadInstructions(Value *V) {
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I || !I->use_empty() || !isInstructionTriviallyDead(I))
    return;
  
  SmallVector<Instruction*, 16> DeadInsts;
  DeadInsts.push_back(I);
  
  while (!DeadInsts.empty()) {
    I = DeadInsts.pop_back_val();

    // Null out all of the instruction's operands to see if any operand becomes
    // dead as we go.
    for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
      Value *OpV = I->getOperand(i);
      I->setOperand(i, 0);
      
      if (!OpV->use_empty()) continue;
    
      // If the operand is an instruction that became dead as we nulled out the
      // operand, and if it is 'trivially' dead, delete it in a future loop
      // iteration.
      if (Instruction *OpI = dyn_cast<Instruction>(OpV))
        if (isInstructionTriviallyDead(OpI))
          DeadInsts.push_back(OpI);
    }
    
    I->eraseFromParent();
  }
}

/// RecursivelyDeleteDeadPHINode - If the specified value is an effectively
/// dead PHI node, due to being a def-use chain of single-use nodes that
/// either forms a cycle or is terminated by a trivially dead instruction,
/// delete it.  If that makes any of its operands trivially dead, delete them
/// too, recursively.
void
llvm::RecursivelyDeleteDeadPHINode(PHINode *PN) {
  // We can remove a PHI if it is on a cycle in the def-use graph
  // where each node in the cycle has degree one, i.e. only one use,
  // and is an instruction with no side effects.
  if (!PN->hasOneUse())
    return;

  SmallPtrSet<PHINode *, 4> PHIs;
  PHIs.insert(PN);
  for (Instruction *J = cast<Instruction>(*PN->use_begin());
       J->hasOneUse() && !J->mayHaveSideEffects();
       J = cast<Instruction>(*J->use_begin()))
    // If we find a PHI more than once, we're on a cycle that
    // won't prove fruitful.
    if (PHINode *JP = dyn_cast<PHINode>(J))
      if (!PHIs.insert(cast<PHINode>(JP))) {
        // Break the cycle and delete the PHI and its operands.
        JP->replaceAllUsesWith(UndefValue::get(JP->getType()));
        RecursivelyDeleteTriviallyDeadInstructions(JP);
        break;
      }
}

//===----------------------------------------------------------------------===//
//  Control Flow Graph Restructuring...
//

/// MergeBasicBlockIntoOnlyPred - DestBB is a block with one predecessor and its
/// predecessor is known to have one successor (DestBB!).  Eliminate the edge
/// between them, moving the instructions in the predecessor into DestBB and
/// deleting the predecessor block.
///
void llvm::MergeBasicBlockIntoOnlyPred(BasicBlock *DestBB, Pass *P) {
  // If BB has single-entry PHI nodes, fold them.
  while (PHINode *PN = dyn_cast<PHINode>(DestBB->begin())) {
    Value *NewVal = PN->getIncomingValue(0);
    // Replace self referencing PHI with undef, it must be dead.
    if (NewVal == PN) NewVal = UndefValue::get(PN->getType());
    PN->replaceAllUsesWith(NewVal);
    PN->eraseFromParent();
  }
  
  BasicBlock *PredBB = DestBB->getSinglePredecessor();
  assert(PredBB && "Block doesn't have a single predecessor!");
  
  // Splice all the instructions from PredBB to DestBB.
  PredBB->getTerminator()->eraseFromParent();
  DestBB->getInstList().splice(DestBB->begin(), PredBB->getInstList());
  
  // Anything that branched to PredBB now branches to DestBB.
  PredBB->replaceAllUsesWith(DestBB);
  
  if (P) {
    ProfileInfo *PI = P->getAnalysisIfAvailable<ProfileInfo>();
    if (PI) {
      PI->replaceAllUses(PredBB, DestBB);
      PI->removeEdge(ProfileInfo::getEdge(PredBB, DestBB));
    }
  }
  // Nuke BB.
  PredBB->eraseFromParent();
}

/// OnlyUsedByDbgIntrinsics - Return true if the instruction I is only used
/// by DbgIntrinsics. If DbgInUses is specified then the vector is filled 
/// with the DbgInfoIntrinsic that use the instruction I.
bool llvm::OnlyUsedByDbgInfoIntrinsics(Instruction *I, 
                               SmallVectorImpl<DbgInfoIntrinsic *> *DbgInUses) {
  if (DbgInUses)
    DbgInUses->clear();

  for (Value::use_iterator UI = I->use_begin(), UE = I->use_end(); UI != UE; 
       ++UI) {
    if (DbgInfoIntrinsic *DI = dyn_cast<DbgInfoIntrinsic>(*UI)) {
      if (DbgInUses)
        DbgInUses->push_back(DI);
    } else {
      if (DbgInUses)
        DbgInUses->clear();
      return false;
    }
  }
  return true;
}

