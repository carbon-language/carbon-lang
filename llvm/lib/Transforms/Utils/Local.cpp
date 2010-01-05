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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/ProfileInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Support/raw_ostream.h"
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
    if (isa<CallInst>(BBI) && BBI->mayWriteToMemory() &&
        !isa<DbgInfoIntrinsic>(BBI))
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
    }
    
    if (Dest2 == Dest1) {       // Conditional branch to same location?
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
    return false;
  }
  
  if (SwitchInst *SI = dyn_cast<SwitchInst>(T)) {
    // If we are switching on a constant, we can convert the switch into a
    // single branch instruction!
    ConstantInt *CI = dyn_cast<ConstantInt>(SI->getCondition());
    BasicBlock *TheOnlyDest = SI->getSuccessor(0);  // The default dest
    BasicBlock *DefaultDest = TheOnlyDest;
    assert(TheOnlyDest == SI->getDefaultDest() &&
           "Default destination is not successor #0?");

    // Figure out which case it goes to.
    for (unsigned i = 1, e = SI->getNumSuccessors(); i != e; ++i) {
      // Found case matching a constant operand?
      if (SI->getSuccessorValue(i) == CI) {
        TheOnlyDest = SI->getSuccessor(i);
        break;
      }

      // Check to see if this branch is going to the same place as the default
      // dest.  If so, eliminate it as an explicit compare.
      if (SI->getSuccessor(i) == DefaultDest) {
        // Remove this entry.
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
      // Insert the new branch.
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

      // Delete the old switch.
      BB->getInstList().erase(SI);
      return true;
    }
    
    if (SI->getNumSuccessors() == 2) {
      // Otherwise, we can fold this switch into a conditional branch
      // instruction if it has only one non-default destination.
      Value *Cond = new ICmpInst(SI, ICmpInst::ICMP_EQ, SI->getCondition(),
                                 SI->getSuccessorValue(1), "cond");
      // Insert the new branch.
      BranchInst::Create(SI->getSuccessor(1), SI->getSuccessor(0), Cond, SI);

      // Delete the old switch.
      SI->eraseFromParent();
      return true;
    }
    return false;
  }

  if (IndirectBrInst *IBI = dyn_cast<IndirectBrInst>(T)) {
    // indirectbr blockaddress(@F, @BB) -> br label @BB
    if (BlockAddress *BA =
          dyn_cast<BlockAddress>(IBI->getAddress()->stripPointerCasts())) {
      BasicBlock *TheOnlyDest = BA->getBasicBlock();
      // Insert the new branch.
      BranchInst::Create(TheOnlyDest, IBI);
      
      for (unsigned i = 0, e = IBI->getNumDestinations(); i != e; ++i) {
        if (IBI->getDestination(i) == TheOnlyDest)
          TheOnlyDest = 0;
        else
          IBI->getDestination(i)->removePredecessor(IBI->getParent());
      }
      IBI->eraseFromParent();
      
      // If we didn't find our destination in the IBI successor list, then we
      // have undefined behavior.  Replace the unconditional branch with an
      // 'unreachable' instruction.
      if (TheOnlyDest) {
        BB->getTerminator()->eraseFromParent();
        new UnreachableInst(BB->getContext(), BB);
      }
      
      return true;
    }
  }
  
  return false;
}


//===----------------------------------------------------------------------===//
//  Local dead code elimination.
//

/// isInstructionTriviallyDead - Return true if the result produced by the
/// instruction is not used, and the instruction has no side effects.
///
bool llvm::isInstructionTriviallyDead(Instruction *I) {
  if (!I->use_empty() || isa<TerminatorInst>(I)) return false;

  // We don't want debug info removed by anything this general.
  if (isa<DbgInfoIntrinsic>(I)) return false;

  // Likewise for memory use markers.
  if (isa<MemoryUseIntrinsic>(I)) return false;

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
/// trivially dead, delete them too, recursively.  Return true if any
/// instructions were deleted.
bool llvm::RecursivelyDeleteTriviallyDeadInstructions(Value *V) {
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I || !I->use_empty() || !isInstructionTriviallyDead(I))
    return false;
  
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

  return true;
}

/// RecursivelyDeleteDeadPHINode - If the specified value is an effectively
/// dead PHI node, due to being a def-use chain of single-use nodes that
/// either forms a cycle or is terminated by a trivially dead instruction,
/// delete it.  If that makes any of its operands trivially dead, delete them
/// too, recursively.  Return true if the PHI node is actually deleted.
bool
llvm::RecursivelyDeleteDeadPHINode(PHINode *PN) {
  // We can remove a PHI if it is on a cycle in the def-use graph
  // where each node in the cycle has degree one, i.e. only one use,
  // and is an instruction with no side effects.
  if (!PN->hasOneUse())
    return false;

  bool Changed = false;
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
        Changed |= RecursivelyDeleteTriviallyDeadInstructions(JP);
        break;
      }
  return Changed;
}

//===----------------------------------------------------------------------===//
//  Control Flow Graph Restructuring.
//


/// RemovePredecessorAndSimplify - Like BasicBlock::removePredecessor, this
/// method is called when we're about to delete Pred as a predecessor of BB.  If
/// BB contains any PHI nodes, this drops the entries in the PHI nodes for Pred.
///
/// Unlike the removePredecessor method, this attempts to simplify uses of PHI
/// nodes that collapse into identity values.  For example, if we have:
///   x = phi(1, 0, 0, 0)
///   y = and x, z
///
/// .. and delete the predecessor corresponding to the '1', this will attempt to
/// recursively fold the and to 0.
void llvm::RemovePredecessorAndSimplify(BasicBlock *BB, BasicBlock *Pred,
                                        TargetData *TD) {
  // This only adjusts blocks with PHI nodes.
  if (!isa<PHINode>(BB->begin()))
    return;
  
  // Remove the entries for Pred from the PHI nodes in BB, but do not simplify
  // them down.  This will leave us with single entry phi nodes and other phis
  // that can be removed.
  BB->removePredecessor(Pred, true);
  
  WeakVH PhiIt = &BB->front();
  while (PHINode *PN = dyn_cast<PHINode>(PhiIt)) {
    PhiIt = &*++BasicBlock::iterator(cast<Instruction>(PhiIt));
    
    Value *PNV = PN->hasConstantValue();
    if (PNV == 0) continue;
    
    // If we're able to simplify the phi to a single value, substitute the new
    // value into all of its uses.
    assert(PNV != PN && "hasConstantValue broken");
    
    ReplaceAndSimplifyAllUses(PN, PNV, TD);
    
    // If recursive simplification ended up deleting the next PHI node we would
    // iterate to, then our iterator is invalid, restart scanning from the top
    // of the block.
    if (PhiIt == 0) PhiIt = &BB->front();
  }
}


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

/// CanPropagatePredecessorsForPHIs - Return true if we can fold BB, an
/// almost-empty BB ending in an unconditional branch to Succ, into succ.
///
/// Assumption: Succ is the single successor for BB.
///
static bool CanPropagatePredecessorsForPHIs(BasicBlock *BB, BasicBlock *Succ) {
  assert(*succ_begin(BB) == Succ && "Succ is not successor of BB!");

  DEBUG(dbgs() << "Looking to fold " << BB->getName() << " into " 
        << Succ->getName() << "\n");
  // Shortcut, if there is only a single predecessor it must be BB and merging
  // is always safe
  if (Succ->getSinglePredecessor()) return true;

  // Make a list of the predecessors of BB
  typedef SmallPtrSet<BasicBlock*, 16> BlockSet;
  BlockSet BBPreds(pred_begin(BB), pred_end(BB));

  // Use that list to make another list of common predecessors of BB and Succ
  BlockSet CommonPreds;
  for (pred_iterator PI = pred_begin(Succ), PE = pred_end(Succ);
        PI != PE; ++PI)
    if (BBPreds.count(*PI))
      CommonPreds.insert(*PI);

  // Shortcut, if there are no common predecessors, merging is always safe
  if (CommonPreds.empty())
    return true;
  
  // Look at all the phi nodes in Succ, to see if they present a conflict when
  // merging these blocks
  for (BasicBlock::iterator I = Succ->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);

    // If the incoming value from BB is again a PHINode in
    // BB which has the same incoming value for *PI as PN does, we can
    // merge the phi nodes and then the blocks can still be merged
    PHINode *BBPN = dyn_cast<PHINode>(PN->getIncomingValueForBlock(BB));
    if (BBPN && BBPN->getParent() == BB) {
      for (BlockSet::iterator PI = CommonPreds.begin(), PE = CommonPreds.end();
            PI != PE; PI++) {
        if (BBPN->getIncomingValueForBlock(*PI) 
              != PN->getIncomingValueForBlock(*PI)) {
          DEBUG(dbgs() << "Can't fold, phi node " << PN->getName() << " in " 
                << Succ->getName() << " is conflicting with " 
                << BBPN->getName() << " with regard to common predecessor "
                << (*PI)->getName() << "\n");
          return false;
        }
      }
    } else {
      Value* Val = PN->getIncomingValueForBlock(BB);
      for (BlockSet::iterator PI = CommonPreds.begin(), PE = CommonPreds.end();
            PI != PE; PI++) {
        // See if the incoming value for the common predecessor is equal to the
        // one for BB, in which case this phi node will not prevent the merging
        // of the block.
        if (Val != PN->getIncomingValueForBlock(*PI)) {
          DEBUG(dbgs() << "Can't fold, phi node " << PN->getName() << " in " 
                << Succ->getName() << " is conflicting with regard to common "
                << "predecessor " << (*PI)->getName() << "\n");
          return false;
        }
      }
    }
  }

  return true;
}

/// TryToSimplifyUncondBranchFromEmptyBlock - BB is known to contain an
/// unconditional branch, and contains no instructions other than PHI nodes,
/// potential debug intrinsics and the branch.  If possible, eliminate BB by
/// rewriting all the predecessors to branch to the successor block and return
/// true.  If we can't transform, return false.
bool llvm::TryToSimplifyUncondBranchFromEmptyBlock(BasicBlock *BB) {
  // We can't eliminate infinite loops.
  BasicBlock *Succ = cast<BranchInst>(BB->getTerminator())->getSuccessor(0);
  if (BB == Succ) return false;
  
  // Check to see if merging these blocks would cause conflicts for any of the
  // phi nodes in BB or Succ. If not, we can safely merge.
  if (!CanPropagatePredecessorsForPHIs(BB, Succ)) return false;

  // Check for cases where Succ has multiple predecessors and a PHI node in BB
  // has uses which will not disappear when the PHI nodes are merged.  It is
  // possible to handle such cases, but difficult: it requires checking whether
  // BB dominates Succ, which is non-trivial to calculate in the case where
  // Succ has multiple predecessors.  Also, it requires checking whether
  // constructing the necessary self-referential PHI node doesn't intoduce any
  // conflicts; this isn't too difficult, but the previous code for doing this
  // was incorrect.
  //
  // Note that if this check finds a live use, BB dominates Succ, so BB is
  // something like a loop pre-header (or rarely, a part of an irreducible CFG);
  // folding the branch isn't profitable in that case anyway.
  if (!Succ->getSinglePredecessor()) {
    BasicBlock::iterator BBI = BB->begin();
    while (isa<PHINode>(*BBI)) {
      for (Value::use_iterator UI = BBI->use_begin(), E = BBI->use_end();
           UI != E; ++UI) {
        if (PHINode* PN = dyn_cast<PHINode>(*UI)) {
          if (PN->getIncomingBlock(UI) != BB)
            return false;
        } else {
          return false;
        }
      }
      ++BBI;
    }
  }

  DEBUG(dbgs() << "Killing Trivial BB: \n" << *BB);
  
  if (isa<PHINode>(Succ->begin())) {
    // If there is more than one pred of succ, and there are PHI nodes in
    // the successor, then we need to add incoming edges for the PHI nodes
    //
    const SmallVector<BasicBlock*, 16> BBPreds(pred_begin(BB), pred_end(BB));
    
    // Loop over all of the PHI nodes in the successor of BB.
    for (BasicBlock::iterator I = Succ->begin(); isa<PHINode>(I); ++I) {
      PHINode *PN = cast<PHINode>(I);
      Value *OldVal = PN->removeIncomingValue(BB, false);
      assert(OldVal && "No entry in PHI for Pred BB!");
      
      // If this incoming value is one of the PHI nodes in BB, the new entries
      // in the PHI node are the entries from the old PHI.
      if (isa<PHINode>(OldVal) && cast<PHINode>(OldVal)->getParent() == BB) {
        PHINode *OldValPN = cast<PHINode>(OldVal);
        for (unsigned i = 0, e = OldValPN->getNumIncomingValues(); i != e; ++i)
          // Note that, since we are merging phi nodes and BB and Succ might
          // have common predecessors, we could end up with a phi node with
          // identical incoming branches. This will be cleaned up later (and
          // will trigger asserts if we try to clean it up now, without also
          // simplifying the corresponding conditional branch).
          PN->addIncoming(OldValPN->getIncomingValue(i),
                          OldValPN->getIncomingBlock(i));
      } else {
        // Add an incoming value for each of the new incoming values.
        for (unsigned i = 0, e = BBPreds.size(); i != e; ++i)
          PN->addIncoming(OldVal, BBPreds[i]);
      }
    }
  }
  
  while (PHINode *PN = dyn_cast<PHINode>(&BB->front())) {
    if (Succ->getSinglePredecessor()) {
      // BB is the only predecessor of Succ, so Succ will end up with exactly
      // the same predecessors BB had.
      Succ->getInstList().splice(Succ->begin(),
                                 BB->getInstList(), BB->begin());
    } else {
      // We explicitly check for such uses in CanPropagatePredecessorsForPHIs.
      assert(PN->use_empty() && "There shouldn't be any uses here!");
      PN->eraseFromParent();
    }
  }
    
  // Everything that jumped to BB now goes to Succ.
  BB->replaceAllUsesWith(Succ);
  if (!Succ->hasName()) Succ->takeName(BB);
  BB->eraseFromParent();              // Delete the old basic block.
  return true;
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

/// EliminateDuplicatePHINodes - Check for and eliminate duplicate PHI
/// nodes in this block. This doesn't try to be clever about PHI nodes
/// which differ only in the order of the incoming values, but instcombine
/// orders them so it usually won't matter.
///
bool llvm::EliminateDuplicatePHINodes(BasicBlock *BB) {
  bool Changed = false;

  // This implementation doesn't currently consider undef operands
  // specially. Theroetically, two phis which are identical except for
  // one having an undef where the other doesn't could be collapsed.

  // Map from PHI hash values to PHI nodes. If multiple PHIs have
  // the same hash value, the element is the first PHI in the
  // linked list in CollisionMap.
  DenseMap<uintptr_t, PHINode *> HashMap;

  // Maintain linked lists of PHI nodes with common hash values.
  DenseMap<PHINode *, PHINode *> CollisionMap;

  // Examine each PHI.
  for (BasicBlock::iterator I = BB->begin();
       PHINode *PN = dyn_cast<PHINode>(I++); ) {
    // Compute a hash value on the operands. Instcombine will likely have sorted
    // them, which helps expose duplicates, but we have to check all the
    // operands to be safe in case instcombine hasn't run.
    uintptr_t Hash = 0;
    for (User::op_iterator I = PN->op_begin(), E = PN->op_end(); I != E; ++I) {
      // This hash algorithm is quite weak as hash functions go, but it seems
      // to do a good enough job for this particular purpose, and is very quick.
      Hash ^= reinterpret_cast<uintptr_t>(static_cast<Value *>(*I));
      Hash = (Hash << 7) | (Hash >> (sizeof(uintptr_t) * CHAR_BIT - 7));
    }
    // If we've never seen this hash value before, it's a unique PHI.
    std::pair<DenseMap<uintptr_t, PHINode *>::iterator, bool> Pair =
      HashMap.insert(std::make_pair(Hash, PN));
    if (Pair.second) continue;
    // Otherwise it's either a duplicate or a hash collision.
    for (PHINode *OtherPN = Pair.first->second; ; ) {
      if (OtherPN->isIdenticalTo(PN)) {
        // A duplicate. Replace this PHI with its duplicate.
        PN->replaceAllUsesWith(OtherPN);
        PN->eraseFromParent();
        Changed = true;
        break;
      }
      // A non-duplicate hash collision.
      DenseMap<PHINode *, PHINode *>::iterator I = CollisionMap.find(OtherPN);
      if (I == CollisionMap.end()) {
        // Set this PHI to be the head of the linked list of colliding PHIs.
        PHINode *Old = Pair.first->second;
        Pair.first->second = PN;
        CollisionMap[PN] = Old;
        break;
      }
      // Procede to the next PHI in the list.
      OtherPN = I->second;
    }
  }

  return Changed;
}
