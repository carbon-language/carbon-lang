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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

STATISTIC(NumRemoved, "Number of unreachable basic blocks removed");

//===----------------------------------------------------------------------===//
//  Local constant propagation.
//

/// ConstantFoldTerminator - If a terminator instruction is predicated on a
/// constant value, convert it into an unconditional branch to the constant
/// destination.  This is a nontrivial operation because the successors of this
/// basic block must have their PHI nodes updated.
/// Also calls RecursivelyDeleteTriviallyDeadInstructions() on any branch/switch
/// conditions and indirectbr addresses this might make dead if
/// DeleteDeadConditions is true.
bool llvm::ConstantFoldTerminator(BasicBlock *BB, bool DeleteDeadConditions,
                                  const TargetLibraryInfo *TLI) {
  TerminatorInst *T = BB->getTerminator();
  IRBuilder<> Builder(T);

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
      OldDest->removePredecessor(BB);

      // Replace the conditional branch with an unconditional one.
      Builder.CreateBr(Destination);
      BI->eraseFromParent();
      return true;
    }

    if (Dest2 == Dest1) {       // Conditional branch to same location?
      // This branch matches something like this:
      //     br bool %cond, label %Dest, label %Dest
      // and changes it into:  br label %Dest

      // Let the basic block know that we are letting go of one copy of it.
      assert(BI->getParent() && "Terminator not inserted in block!");
      Dest1->removePredecessor(BI->getParent());

      // Replace the conditional branch with an unconditional one.
      Builder.CreateBr(Dest1);
      Value *Cond = BI->getCondition();
      BI->eraseFromParent();
      if (DeleteDeadConditions)
        RecursivelyDeleteTriviallyDeadInstructions(Cond, TLI);
      return true;
    }
    return false;
  }

  if (SwitchInst *SI = dyn_cast<SwitchInst>(T)) {
    // If we are switching on a constant, we can convert the switch into a
    // single branch instruction!
    ConstantInt *CI = dyn_cast<ConstantInt>(SI->getCondition());
    BasicBlock *TheOnlyDest = SI->getDefaultDest();
    BasicBlock *DefaultDest = TheOnlyDest;

    // Figure out which case it goes to.
    for (SwitchInst::CaseIt i = SI->case_begin(), e = SI->case_end();
         i != e; ++i) {
      // Found case matching a constant operand?
      if (i.getCaseValue() == CI) {
        TheOnlyDest = i.getCaseSuccessor();
        break;
      }

      // Check to see if this branch is going to the same place as the default
      // dest.  If so, eliminate it as an explicit compare.
      if (i.getCaseSuccessor() == DefaultDest) {
        MDNode* MD = SI->getMetadata(LLVMContext::MD_prof);
        unsigned NCases = SI->getNumCases();
        // Fold the case metadata into the default if there will be any branches
        // left, unless the metadata doesn't match the switch.
        if (NCases > 1 && MD && MD->getNumOperands() == 2 + NCases) {
          // Collect branch weights into a vector.
          SmallVector<uint32_t, 8> Weights;
          for (unsigned MD_i = 1, MD_e = MD->getNumOperands(); MD_i < MD_e;
               ++MD_i) {
            ConstantInt* CI = dyn_cast<ConstantInt>(MD->getOperand(MD_i));
            assert(CI);
            Weights.push_back(CI->getValue().getZExtValue());
          }
          // Merge weight of this case to the default weight.
          unsigned idx = i.getCaseIndex();
          Weights[0] += Weights[idx+1];
          // Remove weight for this case.
          std::swap(Weights[idx+1], Weights.back());
          Weights.pop_back();
          SI->setMetadata(LLVMContext::MD_prof,
                          MDBuilder(BB->getContext()).
                          createBranchWeights(Weights));
        }
        // Remove this entry.
        DefaultDest->removePredecessor(SI->getParent());
        SI->removeCase(i);
        --i; --e;
        continue;
      }

      // Otherwise, check to see if the switch only branches to one destination.
      // We do this by reseting "TheOnlyDest" to null when we find two non-equal
      // destinations.
      if (i.getCaseSuccessor() != TheOnlyDest) TheOnlyDest = 0;
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
      Builder.CreateBr(TheOnlyDest);
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
      Value *Cond = SI->getCondition();
      SI->eraseFromParent();
      if (DeleteDeadConditions)
        RecursivelyDeleteTriviallyDeadInstructions(Cond, TLI);
      return true;
    }

    if (SI->getNumCases() == 1) {
      // Otherwise, we can fold this switch into a conditional branch
      // instruction if it has only one non-default destination.
      SwitchInst::CaseIt FirstCase = SI->case_begin();
      Value *Cond = Builder.CreateICmpEQ(SI->getCondition(),
          FirstCase.getCaseValue(), "cond");

      // Insert the new branch.
      BranchInst *NewBr = Builder.CreateCondBr(Cond,
                                               FirstCase.getCaseSuccessor(),
                                               SI->getDefaultDest());
      MDNode* MD = SI->getMetadata(LLVMContext::MD_prof);
      if (MD && MD->getNumOperands() == 3) {
        ConstantInt *SICase = dyn_cast<ConstantInt>(MD->getOperand(2));
        ConstantInt *SIDef = dyn_cast<ConstantInt>(MD->getOperand(1));
        assert(SICase && SIDef);
        // The TrueWeight should be the weight for the single case of SI.
        NewBr->setMetadata(LLVMContext::MD_prof,
                        MDBuilder(BB->getContext()).
                        createBranchWeights(SICase->getValue().getZExtValue(),
                                            SIDef->getValue().getZExtValue()));
      }

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
      Builder.CreateBr(TheOnlyDest);

      for (unsigned i = 0, e = IBI->getNumDestinations(); i != e; ++i) {
        if (IBI->getDestination(i) == TheOnlyDest)
          TheOnlyDest = 0;
        else
          IBI->getDestination(i)->removePredecessor(IBI->getParent());
      }
      Value *Address = IBI->getAddress();
      IBI->eraseFromParent();
      if (DeleteDeadConditions)
        RecursivelyDeleteTriviallyDeadInstructions(Address, TLI);

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
bool llvm::isInstructionTriviallyDead(Instruction *I,
                                      const TargetLibraryInfo *TLI) {
  if (!I->use_empty() || isa<TerminatorInst>(I)) return false;

  // We don't want the landingpad instruction removed by anything this general.
  if (isa<LandingPadInst>(I))
    return false;

  // We don't want debug info removed by anything this general, unless
  // debug info is empty.
  if (DbgDeclareInst *DDI = dyn_cast<DbgDeclareInst>(I)) {
    if (DDI->getAddress())
      return false;
    return true;
  }
  if (DbgValueInst *DVI = dyn_cast<DbgValueInst>(I)) {
    if (DVI->getValue())
      return false;
    return true;
  }

  if (!I->mayHaveSideEffects()) return true;

  // Special case intrinsics that "may have side effects" but can be deleted
  // when dead.
  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
    // Safe to delete llvm.stacksave if dead.
    if (II->getIntrinsicID() == Intrinsic::stacksave)
      return true;

    // Lifetime intrinsics are dead when their right-hand is undef.
    if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
        II->getIntrinsicID() == Intrinsic::lifetime_end)
      return isa<UndefValue>(II->getArgOperand(1));
  }

  if (isAllocLikeFn(I, TLI)) return true;

  if (CallInst *CI = isFreeCall(I, TLI))
    if (Constant *C = dyn_cast<Constant>(CI->getArgOperand(0)))
      return C->isNullValue() || isa<UndefValue>(C);

  return false;
}

/// RecursivelyDeleteTriviallyDeadInstructions - If the specified value is a
/// trivially dead instruction, delete it.  If that makes any of its operands
/// trivially dead, delete them too, recursively.  Return true if any
/// instructions were deleted.
bool
llvm::RecursivelyDeleteTriviallyDeadInstructions(Value *V,
                                                 const TargetLibraryInfo *TLI) {
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I || !I->use_empty() || !isInstructionTriviallyDead(I, TLI))
    return false;

  SmallVector<Instruction*, 16> DeadInsts;
  DeadInsts.push_back(I);

  do {
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
        if (isInstructionTriviallyDead(OpI, TLI))
          DeadInsts.push_back(OpI);
    }

    I->eraseFromParent();
  } while (!DeadInsts.empty());

  return true;
}

/// areAllUsesEqual - Check whether the uses of a value are all the same.
/// This is similar to Instruction::hasOneUse() except this will also return
/// true when there are no uses or multiple uses that all refer to the same
/// value.
static bool areAllUsesEqual(Instruction *I) {
  Value::use_iterator UI = I->use_begin();
  Value::use_iterator UE = I->use_end();
  if (UI == UE)
    return true;

  User *TheUse = *UI;
  for (++UI; UI != UE; ++UI) {
    if (*UI != TheUse)
      return false;
  }
  return true;
}

/// RecursivelyDeleteDeadPHINode - If the specified value is an effectively
/// dead PHI node, due to being a def-use chain of single-use nodes that
/// either forms a cycle or is terminated by a trivially dead instruction,
/// delete it.  If that makes any of its operands trivially dead, delete them
/// too, recursively.  Return true if a change was made.
bool llvm::RecursivelyDeleteDeadPHINode(PHINode *PN,
                                        const TargetLibraryInfo *TLI) {
  SmallPtrSet<Instruction*, 4> Visited;
  for (Instruction *I = PN; areAllUsesEqual(I) && !I->mayHaveSideEffects();
       I = cast<Instruction>(*I->use_begin())) {
    if (I->use_empty())
      return RecursivelyDeleteTriviallyDeadInstructions(I, TLI);

    // If we find an instruction more than once, we're on a cycle that
    // won't prove fruitful.
    if (!Visited.insert(I)) {
      // Break the cycle and delete the instruction and its operands.
      I->replaceAllUsesWith(UndefValue::get(I->getType()));
      (void)RecursivelyDeleteTriviallyDeadInstructions(I, TLI);
      return true;
    }
  }
  return false;
}

/// SimplifyInstructionsInBlock - Scan the specified basic block and try to
/// simplify any instructions in it and recursively delete dead instructions.
///
/// This returns true if it changed the code, note that it can delete
/// instructions in other blocks as well in this block.
bool llvm::SimplifyInstructionsInBlock(BasicBlock *BB, const DataLayout *TD,
                                       const TargetLibraryInfo *TLI) {
  bool MadeChange = false;

#ifndef NDEBUG
  // In debug builds, ensure that the terminator of the block is never replaced
  // or deleted by these simplifications. The idea of simplification is that it
  // cannot introduce new instructions, and there is no way to replace the
  // terminator of a block without introducing a new instruction.
  AssertingVH<Instruction> TerminatorVH(--BB->end());
#endif

  for (BasicBlock::iterator BI = BB->begin(), E = --BB->end(); BI != E; ) {
    assert(!BI->isTerminator());
    Instruction *Inst = BI++;

    WeakVH BIHandle(BI);
    if (recursivelySimplifyInstruction(Inst, TD, TLI)) {
      MadeChange = true;
      if (BIHandle != BI)
        BI = BB->begin();
      continue;
    }

    MadeChange |= RecursivelyDeleteTriviallyDeadInstructions(Inst, TLI);
    if (BIHandle != BI)
      BI = BB->begin();
  }
  return MadeChange;
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
                                        DataLayout *TD) {
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
    Value *OldPhiIt = PhiIt;

    if (!recursivelySimplifyInstruction(PN, TD))
      continue;

    // If recursive simplification ended up deleting the next PHI node we would
    // iterate to, then our iterator is invalid, restart scanning from the top
    // of the block.
    if (PhiIt != OldPhiIt) PhiIt = &BB->front();
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

  // Zap anything that took the address of DestBB.  Not doing this will give the
  // address an invalid value.
  if (DestBB->hasAddressTaken()) {
    BlockAddress *BA = BlockAddress::get(DestBB);
    Constant *Replacement =
      ConstantInt::get(llvm::Type::getInt32Ty(BA->getContext()), 1);
    BA->replaceAllUsesWith(ConstantExpr::getIntToPtr(Replacement,
                                                     BA->getType()));
    BA->destroyConstant();
  }

  // Anything that branched to PredBB now branches to DestBB.
  PredBB->replaceAllUsesWith(DestBB);

  // Splice all the instructions from PredBB to DestBB.
  PredBB->getTerminator()->eraseFromParent();
  DestBB->getInstList().splice(DestBB->begin(), PredBB->getInstList());

  if (P) {
    if (DominatorTreeWrapperPass *DTWP =
            P->getAnalysisIfAvailable<DominatorTreeWrapperPass>()) {
      DominatorTree &DT = DTWP->getDomTree();
      BasicBlock *PredBBIDom = DT.getNode(PredBB)->getIDom()->getBlock();
      DT.changeImmediateDominator(DestBB, PredBBIDom);
      DT.eraseNode(PredBB);
    }
  }
  // Nuke BB.
  PredBB->eraseFromParent();
}

/// CanMergeValues - Return true if we can choose one of these values to use
/// in place of the other. Note that we will always choose the non-undef
/// value to keep.
static bool CanMergeValues(Value *First, Value *Second) {
  return First == Second || isa<UndefValue>(First) || isa<UndefValue>(Second);
}

/// CanPropagatePredecessorsForPHIs - Return true if we can fold BB, an
/// almost-empty BB ending in an unconditional branch to Succ, into Succ.
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
  SmallPtrSet<BasicBlock*, 16> BBPreds(pred_begin(BB), pred_end(BB));

  // Look at all the phi nodes in Succ, to see if they present a conflict when
  // merging these blocks
  for (BasicBlock::iterator I = Succ->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);

    // If the incoming value from BB is again a PHINode in
    // BB which has the same incoming value for *PI as PN does, we can
    // merge the phi nodes and then the blocks can still be merged
    PHINode *BBPN = dyn_cast<PHINode>(PN->getIncomingValueForBlock(BB));
    if (BBPN && BBPN->getParent() == BB) {
      for (unsigned PI = 0, PE = PN->getNumIncomingValues(); PI != PE; ++PI) {
        BasicBlock *IBB = PN->getIncomingBlock(PI);
        if (BBPreds.count(IBB) &&
            !CanMergeValues(BBPN->getIncomingValueForBlock(IBB),
                            PN->getIncomingValue(PI))) {
          DEBUG(dbgs() << "Can't fold, phi node " << PN->getName() << " in "
                << Succ->getName() << " is conflicting with "
                << BBPN->getName() << " with regard to common predecessor "
                << IBB->getName() << "\n");
          return false;
        }
      }
    } else {
      Value* Val = PN->getIncomingValueForBlock(BB);
      for (unsigned PI = 0, PE = PN->getNumIncomingValues(); PI != PE; ++PI) {
        // See if the incoming value for the common predecessor is equal to the
        // one for BB, in which case this phi node will not prevent the merging
        // of the block.
        BasicBlock *IBB = PN->getIncomingBlock(PI);
        if (BBPreds.count(IBB) &&
            !CanMergeValues(Val, PN->getIncomingValue(PI))) {
          DEBUG(dbgs() << "Can't fold, phi node " << PN->getName() << " in "
                << Succ->getName() << " is conflicting with regard to common "
                << "predecessor " << IBB->getName() << "\n");
          return false;
        }
      }
    }
  }

  return true;
}

typedef SmallVector<BasicBlock *, 16> PredBlockVector;
typedef DenseMap<BasicBlock *, Value *> IncomingValueMap;

/// \brief Determines the value to use as the phi node input for a block.
///
/// Select between \p OldVal any value that we know flows from \p BB
/// to a particular phi on the basis of which one (if either) is not
/// undef. Update IncomingValues based on the selected value.
///
/// \param OldVal The value we are considering selecting.
/// \param BB The block that the value flows in from.
/// \param IncomingValues A map from block-to-value for other phi inputs
/// that we have examined.
///
/// \returns the selected value.
static Value *selectIncomingValueForBlock(Value *OldVal, BasicBlock *BB,
                                          IncomingValueMap &IncomingValues) {
  if (!isa<UndefValue>(OldVal)) {
    assert((!IncomingValues.count(BB) ||
            IncomingValues.find(BB)->second == OldVal) &&
           "Expected OldVal to match incoming value from BB!");

    IncomingValues.insert(std::make_pair(BB, OldVal));
    return OldVal;
  }

  IncomingValueMap::const_iterator It = IncomingValues.find(BB);
  if (It != IncomingValues.end()) return It->second;

  return OldVal;
}

/// \brief Create a map from block to value for the operands of a
/// given phi.
///
/// Create a map from block to value for each non-undef value flowing
/// into \p PN.
///
/// \param PN The phi we are collecting the map for.
/// \param IncomingValues [out] The map from block to value for this phi.
static void gatherIncomingValuesToPhi(PHINode *PN,
                                      IncomingValueMap &IncomingValues) {
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    BasicBlock *BB = PN->getIncomingBlock(i);
    Value *V = PN->getIncomingValue(i);

    if (!isa<UndefValue>(V))
      IncomingValues.insert(std::make_pair(BB, V));
  }
}

/// \brief Replace the incoming undef values to a phi with the values
/// from a block-to-value map.
///
/// \param PN The phi we are replacing the undefs in.
/// \param IncomingValues A map from block to value.
static void replaceUndefValuesInPhi(PHINode *PN,
                                    const IncomingValueMap &IncomingValues) {
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    Value *V = PN->getIncomingValue(i);

    if (!isa<UndefValue>(V)) continue;

    BasicBlock *BB = PN->getIncomingBlock(i);
    IncomingValueMap::const_iterator It = IncomingValues.find(BB);
    if (It == IncomingValues.end()) continue;

    PN->setIncomingValue(i, It->second);
  }
}

/// \brief Replace a value flowing from a block to a phi with
/// potentially multiple instances of that value flowing from the
/// block's predecessors to the phi.
///
/// \param BB The block with the value flowing into the phi.
/// \param BBPreds The predecessors of BB.
/// \param PN The phi that we are updating.
static void redirectValuesFromPredecessorsToPhi(BasicBlock *BB,
                                                const PredBlockVector &BBPreds,
                                                PHINode *PN) {
  Value *OldVal = PN->removeIncomingValue(BB, false);
  assert(OldVal && "No entry in PHI for Pred BB!");

  IncomingValueMap IncomingValues;

  // We are merging two blocks - BB, and the block containing PN - and
  // as a result we need to redirect edges from the predecessors of BB
  // to go to the block containing PN, and update PN
  // accordingly. Since we allow merging blocks in the case where the
  // predecessor and successor blocks both share some predecessors,
  // and where some of those common predecessors might have undef
  // values flowing into PN, we want to rewrite those values to be
  // consistent with the non-undef values.

  gatherIncomingValuesToPhi(PN, IncomingValues);

  // If this incoming value is one of the PHI nodes in BB, the new entries
  // in the PHI node are the entries from the old PHI.
  if (isa<PHINode>(OldVal) && cast<PHINode>(OldVal)->getParent() == BB) {
    PHINode *OldValPN = cast<PHINode>(OldVal);
    for (unsigned i = 0, e = OldValPN->getNumIncomingValues(); i != e; ++i) {
      // Note that, since we are merging phi nodes and BB and Succ might
      // have common predecessors, we could end up with a phi node with
      // identical incoming branches. This will be cleaned up later (and
      // will trigger asserts if we try to clean it up now, without also
      // simplifying the corresponding conditional branch).
      BasicBlock *PredBB = OldValPN->getIncomingBlock(i);
      Value *PredVal = OldValPN->getIncomingValue(i);
      Value *Selected = selectIncomingValueForBlock(PredVal, PredBB,
                                                    IncomingValues);

      // And add a new incoming value for this predecessor for the
      // newly retargeted branch.
      PN->addIncoming(Selected, PredBB);
    }
  } else {
    for (unsigned i = 0, e = BBPreds.size(); i != e; ++i) {
      // Update existing incoming values in PN for this
      // predecessor of BB.
      BasicBlock *PredBB = BBPreds[i];
      Value *Selected = selectIncomingValueForBlock(OldVal, PredBB,
                                                    IncomingValues);

      // And add a new incoming value for this predecessor for the
      // newly retargeted branch.
      PN->addIncoming(Selected, PredBB);
    }
  }

  replaceUndefValuesInPhi(PN, IncomingValues);
}

/// TryToSimplifyUncondBranchFromEmptyBlock - BB is known to contain an
/// unconditional branch, and contains no instructions other than PHI nodes,
/// potential side-effect free intrinsics and the branch.  If possible,
/// eliminate BB by rewriting all the predecessors to branch to the successor
/// block and return true.  If we can't transform, return false.
bool llvm::TryToSimplifyUncondBranchFromEmptyBlock(BasicBlock *BB) {
  assert(BB != &BB->getParent()->getEntryBlock() &&
         "TryToSimplifyUncondBranchFromEmptyBlock called on entry block!");

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
  // constructing the necessary self-referential PHI node doesn't introduce any
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
    const PredBlockVector BBPreds(pred_begin(BB), pred_end(BB));

    // Loop over all of the PHI nodes in the successor of BB.
    for (BasicBlock::iterator I = Succ->begin(); isa<PHINode>(I); ++I) {
      PHINode *PN = cast<PHINode>(I);

      redirectValuesFromPredecessorsToPhi(BB, BBPreds, PN);
    }
  }

  if (Succ->getSinglePredecessor()) {
    // BB is the only predecessor of Succ, so Succ will end up with exactly
    // the same predecessors BB had.

    // Copy over any phi, debug or lifetime instruction.
    BB->getTerminator()->eraseFromParent();
    Succ->getInstList().splice(Succ->getFirstNonPHI(), BB->getInstList());
  } else {
    while (PHINode *PN = dyn_cast<PHINode>(&BB->front())) {
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

/// EliminateDuplicatePHINodes - Check for and eliminate duplicate PHI
/// nodes in this block. This doesn't try to be clever about PHI nodes
/// which differ only in the order of the incoming values, but instcombine
/// orders them so it usually won't matter.
///
bool llvm::EliminateDuplicatePHINodes(BasicBlock *BB) {
  bool Changed = false;

  // This implementation doesn't currently consider undef operands
  // specially. Theoretically, two phis which are identical except for
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
    // This hash algorithm is quite weak as hash functions go, but it seems
    // to do a good enough job for this particular purpose, and is very quick.
    for (User::op_iterator I = PN->op_begin(), E = PN->op_end(); I != E; ++I) {
      Hash ^= reinterpret_cast<uintptr_t>(static_cast<Value *>(*I));
      Hash = (Hash << 7) | (Hash >> (sizeof(uintptr_t) * CHAR_BIT - 7));
    }
    for (PHINode::block_iterator I = PN->block_begin(), E = PN->block_end();
         I != E; ++I) {
      Hash ^= reinterpret_cast<uintptr_t>(static_cast<BasicBlock *>(*I));
      Hash = (Hash << 7) | (Hash >> (sizeof(uintptr_t) * CHAR_BIT - 7));
    }
    // Avoid colliding with the DenseMap sentinels ~0 and ~0-1.
    Hash >>= 1;
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
      // Proceed to the next PHI in the list.
      OtherPN = I->second;
    }
  }

  return Changed;
}

/// enforceKnownAlignment - If the specified pointer points to an object that
/// we control, modify the object's alignment to PrefAlign. This isn't
/// often possible though. If alignment is important, a more reliable approach
/// is to simply align all global variables and allocation instructions to
/// their preferred alignment from the beginning.
///
static unsigned enforceKnownAlignment(Value *V, unsigned Align,
                                      unsigned PrefAlign, const DataLayout *TD) {
  V = V->stripPointerCasts();

  if (AllocaInst *AI = dyn_cast<AllocaInst>(V)) {
    // If the preferred alignment is greater than the natural stack alignment
    // then don't round up. This avoids dynamic stack realignment.
    if (TD && TD->exceedsNaturalStackAlignment(PrefAlign))
      return Align;
    // If there is a requested alignment and if this is an alloca, round up.
    if (AI->getAlignment() >= PrefAlign)
      return AI->getAlignment();
    AI->setAlignment(PrefAlign);
    return PrefAlign;
  }

  if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
    // If there is a large requested alignment and we can, bump up the alignment
    // of the global.
    if (GV->isDeclaration()) return Align;
    // If the memory we set aside for the global may not be the memory used by
    // the final program then it is impossible for us to reliably enforce the
    // preferred alignment.
    if (GV->isWeakForLinker()) return Align;

    if (GV->getAlignment() >= PrefAlign)
      return GV->getAlignment();
    // We can only increase the alignment of the global if it has no alignment
    // specified or if it is not assigned a section.  If it is assigned a
    // section, the global could be densely packed with other objects in the
    // section, increasing the alignment could cause padding issues.
    if (!GV->hasSection() || GV->getAlignment() == 0)
      GV->setAlignment(PrefAlign);
    return GV->getAlignment();
  }

  return Align;
}

/// getOrEnforceKnownAlignment - If the specified pointer has an alignment that
/// we can determine, return it, otherwise return 0.  If PrefAlign is specified,
/// and it is more than the alignment of the ultimate object, see if we can
/// increase the alignment of the ultimate object, making this check succeed.
unsigned llvm::getOrEnforceKnownAlignment(Value *V, unsigned PrefAlign,
                                          const DataLayout *DL) {
  assert(V->getType()->isPointerTy() &&
         "getOrEnforceKnownAlignment expects a pointer!");
  unsigned BitWidth = DL ? DL->getPointerTypeSizeInBits(V->getType()) : 64;

  APInt KnownZero(BitWidth, 0), KnownOne(BitWidth, 0);
  ComputeMaskedBits(V, KnownZero, KnownOne, DL);
  unsigned TrailZ = KnownZero.countTrailingOnes();

  // Avoid trouble with ridiculously large TrailZ values, such as
  // those computed from a null pointer.
  TrailZ = std::min(TrailZ, unsigned(sizeof(unsigned) * CHAR_BIT - 1));

  unsigned Align = 1u << std::min(BitWidth - 1, TrailZ);

  // LLVM doesn't support alignments larger than this currently.
  Align = std::min(Align, +Value::MaximumAlignment);

  if (PrefAlign > Align)
    Align = enforceKnownAlignment(V, Align, PrefAlign, DL);

  // We don't need to make any adjustment.
  return Align;
}

///===---------------------------------------------------------------------===//
///  Dbg Intrinsic utilities
///

/// See if there is a dbg.value intrinsic for DIVar before I.
static bool LdStHasDebugValue(DIVariable &DIVar, Instruction *I) {
  // Since we can't guarantee that the original dbg.declare instrinsic
  // is removed by LowerDbgDeclare(), we need to make sure that we are
  // not inserting the same dbg.value intrinsic over and over.
  llvm::BasicBlock::InstListType::iterator PrevI(I);
  if (PrevI != I->getParent()->getInstList().begin()) {
    --PrevI;
    if (DbgValueInst *DVI = dyn_cast<DbgValueInst>(PrevI))
      if (DVI->getValue() == I->getOperand(0) &&
          DVI->getOffset() == 0 &&
          DVI->getVariable() == DIVar)
        return true;
  }
  return false;
}

/// Inserts a llvm.dbg.value intrinsic before a store to an alloca'd value
/// that has an associated llvm.dbg.decl intrinsic.
bool llvm::ConvertDebugDeclareToDebugValue(DbgDeclareInst *DDI,
                                           StoreInst *SI, DIBuilder &Builder) {
  DIVariable DIVar(DDI->getVariable());
  assert((!DIVar || DIVar.isVariable()) &&
         "Variable in DbgDeclareInst should be either null or a DIVariable.");
  if (!DIVar)
    return false;

  if (LdStHasDebugValue(DIVar, SI))
    return true;

  Instruction *DbgVal = NULL;
  // If an argument is zero extended then use argument directly. The ZExt
  // may be zapped by an optimization pass in future.
  Argument *ExtendedArg = NULL;
  if (ZExtInst *ZExt = dyn_cast<ZExtInst>(SI->getOperand(0)))
    ExtendedArg = dyn_cast<Argument>(ZExt->getOperand(0));
  if (SExtInst *SExt = dyn_cast<SExtInst>(SI->getOperand(0)))
    ExtendedArg = dyn_cast<Argument>(SExt->getOperand(0));
  if (ExtendedArg)
    DbgVal = Builder.insertDbgValueIntrinsic(ExtendedArg, 0, DIVar, SI);
  else
    DbgVal = Builder.insertDbgValueIntrinsic(SI->getOperand(0), 0, DIVar, SI);

  // Propagate any debug metadata from the store onto the dbg.value.
  DebugLoc SIDL = SI->getDebugLoc();
  if (!SIDL.isUnknown())
    DbgVal->setDebugLoc(SIDL);
  // Otherwise propagate debug metadata from dbg.declare.
  else
    DbgVal->setDebugLoc(DDI->getDebugLoc());
  return true;
}

/// Inserts a llvm.dbg.value intrinsic before a load of an alloca'd value
/// that has an associated llvm.dbg.decl intrinsic.
bool llvm::ConvertDebugDeclareToDebugValue(DbgDeclareInst *DDI,
                                           LoadInst *LI, DIBuilder &Builder) {
  DIVariable DIVar(DDI->getVariable());
  assert((!DIVar || DIVar.isVariable()) &&
         "Variable in DbgDeclareInst should be either null or a DIVariable.");
  if (!DIVar)
    return false;

  if (LdStHasDebugValue(DIVar, LI))
    return true;

  Instruction *DbgVal =
    Builder.insertDbgValueIntrinsic(LI->getOperand(0), 0,
                                    DIVar, LI);

  // Propagate any debug metadata from the store onto the dbg.value.
  DebugLoc LIDL = LI->getDebugLoc();
  if (!LIDL.isUnknown())
    DbgVal->setDebugLoc(LIDL);
  // Otherwise propagate debug metadata from dbg.declare.
  else
    DbgVal->setDebugLoc(DDI->getDebugLoc());
  return true;
}

/// LowerDbgDeclare - Lowers llvm.dbg.declare intrinsics into appropriate set
/// of llvm.dbg.value intrinsics.
bool llvm::LowerDbgDeclare(Function &F) {
  DIBuilder DIB(*F.getParent());
  SmallVector<DbgDeclareInst *, 4> Dbgs;
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI)
    for (BasicBlock::iterator BI = FI->begin(), BE = FI->end(); BI != BE; ++BI) {
      if (DbgDeclareInst *DDI = dyn_cast<DbgDeclareInst>(BI))
        Dbgs.push_back(DDI);
    }
  if (Dbgs.empty())
    return false;

  for (SmallVectorImpl<DbgDeclareInst *>::iterator I = Dbgs.begin(),
         E = Dbgs.end(); I != E; ++I) {
    DbgDeclareInst *DDI = *I;
    AllocaInst *AI = dyn_cast_or_null<AllocaInst>(DDI->getAddress());
    // If this is an alloca for a scalar variable, insert a dbg.value
    // at each load and store to the alloca and erase the dbg.declare.
    if (AI && !AI->isArrayAllocation()) {

      // We only remove the dbg.declare intrinsic if all uses are
      // converted to dbg.value intrinsics.
      bool RemoveDDI = true;
      for (Value::use_iterator UI = AI->use_begin(), E = AI->use_end();
           UI != E; ++UI)
        if (StoreInst *SI = dyn_cast<StoreInst>(*UI))
          ConvertDebugDeclareToDebugValue(DDI, SI, DIB);
        else if (LoadInst *LI = dyn_cast<LoadInst>(*UI))
          ConvertDebugDeclareToDebugValue(DDI, LI, DIB);
        else
          RemoveDDI = false;
      if (RemoveDDI)
        DDI->eraseFromParent();
    }
  }
  return true;
}

/// FindAllocaDbgDeclare - Finds the llvm.dbg.declare intrinsic describing the
/// alloca 'V', if any.
DbgDeclareInst *llvm::FindAllocaDbgDeclare(Value *V) {
  if (MDNode *DebugNode = MDNode::getIfExists(V->getContext(), V))
    for (Value::use_iterator UI = DebugNode->use_begin(),
         E = DebugNode->use_end(); UI != E; ++UI)
      if (DbgDeclareInst *DDI = dyn_cast<DbgDeclareInst>(*UI))
        return DDI;

  return 0;
}

bool llvm::replaceDbgDeclareForAlloca(AllocaInst *AI, Value *NewAllocaAddress,
                                      DIBuilder &Builder) {
  DbgDeclareInst *DDI = FindAllocaDbgDeclare(AI);
  if (!DDI)
    return false;
  DIVariable DIVar(DDI->getVariable());
  assert((!DIVar || DIVar.isVariable()) &&
         "Variable in DbgDeclareInst should be either null or a DIVariable.");
  if (!DIVar)
    return false;

  // Create a copy of the original DIDescriptor for user variable, appending
  // "deref" operation to a list of address elements, as new llvm.dbg.declare
  // will take a value storing address of the memory for variable, not
  // alloca itself.
  Type *Int64Ty = Type::getInt64Ty(AI->getContext());
  SmallVector<Value*, 4> NewDIVarAddress;
  if (DIVar.hasComplexAddress()) {
    for (unsigned i = 0, n = DIVar.getNumAddrElements(); i < n; ++i) {
      NewDIVarAddress.push_back(
          ConstantInt::get(Int64Ty, DIVar.getAddrElement(i)));
    }
  }
  NewDIVarAddress.push_back(ConstantInt::get(Int64Ty, DIBuilder::OpDeref));
  DIVariable NewDIVar = Builder.createComplexVariable(
      DIVar.getTag(), DIVar.getContext(), DIVar.getName(),
      DIVar.getFile(), DIVar.getLineNumber(), DIVar.getType(),
      NewDIVarAddress, DIVar.getArgNumber());

  // Insert llvm.dbg.declare in the same basic block as the original alloca,
  // and remove old llvm.dbg.declare.
  BasicBlock *BB = AI->getParent();
  Builder.insertDeclare(NewAllocaAddress, NewDIVar, BB);
  DDI->eraseFromParent();
  return true;
}

/// changeToUnreachable - Insert an unreachable instruction before the specified
/// instruction, making it and the rest of the code in the block dead.
static void changeToUnreachable(Instruction *I, bool UseLLVMTrap) {
  BasicBlock *BB = I->getParent();
  // Loop over all of the successors, removing BB's entry from any PHI
  // nodes.
  for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB); SI != SE; ++SI)
    (*SI)->removePredecessor(BB);

  // Insert a call to llvm.trap right before this.  This turns the undefined
  // behavior into a hard fail instead of falling through into random code.
  if (UseLLVMTrap) {
    Function *TrapFn =
      Intrinsic::getDeclaration(BB->getParent()->getParent(), Intrinsic::trap);
    CallInst *CallTrap = CallInst::Create(TrapFn, "", I);
    CallTrap->setDebugLoc(I->getDebugLoc());
  }
  new UnreachableInst(I->getContext(), I);

  // All instructions after this are dead.
  BasicBlock::iterator BBI = I, BBE = BB->end();
  while (BBI != BBE) {
    if (!BBI->use_empty())
      BBI->replaceAllUsesWith(UndefValue::get(BBI->getType()));
    BB->getInstList().erase(BBI++);
  }
}

/// changeToCall - Convert the specified invoke into a normal call.
static void changeToCall(InvokeInst *II) {
  SmallVector<Value*, 8> Args(II->op_begin(), II->op_end() - 3);
  CallInst *NewCall = CallInst::Create(II->getCalledValue(), Args, "", II);
  NewCall->takeName(II);
  NewCall->setCallingConv(II->getCallingConv());
  NewCall->setAttributes(II->getAttributes());
  NewCall->setDebugLoc(II->getDebugLoc());
  II->replaceAllUsesWith(NewCall);

  // Follow the call by a branch to the normal destination.
  BranchInst::Create(II->getNormalDest(), II);

  // Update PHI nodes in the unwind destination
  II->getUnwindDest()->removePredecessor(II->getParent());
  II->eraseFromParent();
}

static bool markAliveBlocks(BasicBlock *BB,
                            SmallPtrSet<BasicBlock*, 128> &Reachable) {

  SmallVector<BasicBlock*, 128> Worklist;
  Worklist.push_back(BB);
  Reachable.insert(BB);
  bool Changed = false;
  do {
    BB = Worklist.pop_back_val();

    // Do a quick scan of the basic block, turning any obviously unreachable
    // instructions into LLVM unreachable insts.  The instruction combining pass
    // canonicalizes unreachable insts into stores to null or undef.
    for (BasicBlock::iterator BBI = BB->begin(), E = BB->end(); BBI != E;++BBI){
      if (CallInst *CI = dyn_cast<CallInst>(BBI)) {
        if (CI->doesNotReturn()) {
          // If we found a call to a no-return function, insert an unreachable
          // instruction after it.  Make sure there isn't *already* one there
          // though.
          ++BBI;
          if (!isa<UnreachableInst>(BBI)) {
            // Don't insert a call to llvm.trap right before the unreachable.
            changeToUnreachable(BBI, false);
            Changed = true;
          }
          break;
        }
      }

      // Store to undef and store to null are undefined and used to signal that
      // they should be changed to unreachable by passes that can't modify the
      // CFG.
      if (StoreInst *SI = dyn_cast<StoreInst>(BBI)) {
        // Don't touch volatile stores.
        if (SI->isVolatile()) continue;

        Value *Ptr = SI->getOperand(1);

        if (isa<UndefValue>(Ptr) ||
            (isa<ConstantPointerNull>(Ptr) &&
             SI->getPointerAddressSpace() == 0)) {
          changeToUnreachable(SI, true);
          Changed = true;
          break;
        }
      }
    }

    // Turn invokes that call 'nounwind' functions into ordinary calls.
    if (InvokeInst *II = dyn_cast<InvokeInst>(BB->getTerminator())) {
      Value *Callee = II->getCalledValue();
      if (isa<ConstantPointerNull>(Callee) || isa<UndefValue>(Callee)) {
        changeToUnreachable(II, true);
        Changed = true;
      } else if (II->doesNotThrow()) {
        if (II->use_empty() && II->onlyReadsMemory()) {
          // jump to the normal destination branch.
          BranchInst::Create(II->getNormalDest(), II);
          II->getUnwindDest()->removePredecessor(II->getParent());
          II->eraseFromParent();
        } else
          changeToCall(II);
        Changed = true;
      }
    }

    Changed |= ConstantFoldTerminator(BB, true);
    for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB); SI != SE; ++SI)
      if (Reachable.insert(*SI))
        Worklist.push_back(*SI);
  } while (!Worklist.empty());
  return Changed;
}

/// removeUnreachableBlocksFromFn - Remove blocks that are not reachable, even
/// if they are in a dead cycle.  Return true if a change was made, false
/// otherwise.
bool llvm::removeUnreachableBlocks(Function &F) {
  SmallPtrSet<BasicBlock*, 128> Reachable;
  bool Changed = markAliveBlocks(F.begin(), Reachable);

  // If there are unreachable blocks in the CFG...
  if (Reachable.size() == F.size())
    return Changed;

  assert(Reachable.size() < F.size());
  NumRemoved += F.size()-Reachable.size();

  // Loop over all of the basic blocks that are not reachable, dropping all of
  // their internal references...
  for (Function::iterator BB = ++F.begin(), E = F.end(); BB != E; ++BB) {
    if (Reachable.count(BB))
      continue;

    for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB); SI != SE; ++SI)
      if (Reachable.count(*SI))
        (*SI)->removePredecessor(BB);
    BB->dropAllReferences();
  }

  for (Function::iterator I = ++F.begin(); I != F.end();)
    if (!Reachable.count(I))
      I = F.getBasicBlockList().erase(I);
    else
      ++I;

  return true;
}
