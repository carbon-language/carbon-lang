//===- SimplifyCFG.cpp - Code to perform CFG simplification ---------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Peephole optimize the CFG.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/Support/CFG.h"
#include <algorithm>
#include <functional>
#include <set>
using namespace llvm;

// PropagatePredecessorsForPHIs - This gets "Succ" ready to have the
// predecessors from "BB".  This is a little tricky because "Succ" has PHI
// nodes, which need to have extra slots added to them to hold the merge edges
// from BB's predecessors, and BB itself might have had PHI nodes in it.  This
// function returns true (failure) if the Succ BB already has a predecessor that
// is a predecessor of BB and incoming PHI arguments would not be discernible.
//
// Assumption: Succ is the single successor for BB.
//
static bool PropagatePredecessorsForPHIs(BasicBlock *BB, BasicBlock *Succ) {
  assert(*succ_begin(BB) == Succ && "Succ is not successor of BB!");

  if (!isa<PHINode>(Succ->front()))
    return false;  // We can make the transformation, no problem.

  // If there is more than one predecessor, and there are PHI nodes in
  // the successor, then we need to add incoming edges for the PHI nodes
  //
  const std::vector<BasicBlock*> BBPreds(pred_begin(BB), pred_end(BB));

  // Check to see if one of the predecessors of BB is already a predecessor of
  // Succ.  If so, we cannot do the transformation if there are any PHI nodes
  // with incompatible values coming in from the two edges!
  //
  for (pred_iterator PI = pred_begin(Succ), PE = pred_end(Succ); PI != PE; ++PI)
    if (find(BBPreds.begin(), BBPreds.end(), *PI) != BBPreds.end()) {
      // Loop over all of the PHI nodes checking to see if there are
      // incompatible values coming in.
      for (BasicBlock::iterator I = Succ->begin();
           PHINode *PN = dyn_cast<PHINode>(I); ++I) {
        // Loop up the entries in the PHI node for BB and for *PI if the values
        // coming in are non-equal, we cannot merge these two blocks (instead we
        // should insert a conditional move or something, then merge the
        // blocks).
        int Idx1 = PN->getBasicBlockIndex(BB);
        int Idx2 = PN->getBasicBlockIndex(*PI);
        assert(Idx1 != -1 && Idx2 != -1 &&
               "Didn't have entries for my predecessors??");
        if (PN->getIncomingValue(Idx1) != PN->getIncomingValue(Idx2))
          return true;  // Values are not equal...
      }
    }

  // Loop over all of the PHI nodes in the successor BB
  for (BasicBlock::iterator I = Succ->begin();
       PHINode *PN = dyn_cast<PHINode>(I); ++I) {
    Value *OldVal = PN->removeIncomingValue(BB, false);
    assert(OldVal && "No entry in PHI for Pred BB!");

    // If this incoming value is one of the PHI nodes in BB...
    if (isa<PHINode>(OldVal) && cast<PHINode>(OldVal)->getParent() == BB) {
      PHINode *OldValPN = cast<PHINode>(OldVal);
      for (std::vector<BasicBlock*>::const_iterator PredI = BBPreds.begin(), 
             End = BBPreds.end(); PredI != End; ++PredI) {
        PN->addIncoming(OldValPN->getIncomingValueForBlock(*PredI), *PredI);
      }
    } else {
      for (std::vector<BasicBlock*>::const_iterator PredI = BBPreds.begin(), 
             End = BBPreds.end(); PredI != End; ++PredI) {
        // Add an incoming value for each of the new incoming values...
        PN->addIncoming(OldVal, *PredI);
      }
    }
  }
  return false;
}

/// GetIfCondition - Given a basic block (BB) with two predecessors (and
/// presumably PHI nodes in it), check to see if the merge at this block is due
/// to an "if condition".  If so, return the boolean condition that determines
/// which entry into BB will be taken.  Also, return by references the block
/// that will be entered from if the condition is true, and the block that will
/// be entered if the condition is false.
/// 
///
static Value *GetIfCondition(BasicBlock *BB,
                             BasicBlock *&IfTrue, BasicBlock *&IfFalse) {
  assert(std::distance(pred_begin(BB), pred_end(BB)) == 2 &&
         "Function can only handle blocks with 2 predecessors!");
  BasicBlock *Pred1 = *pred_begin(BB);
  BasicBlock *Pred2 = *++pred_begin(BB);

  // We can only handle branches.  Other control flow will be lowered to
  // branches if possible anyway.
  if (!isa<BranchInst>(Pred1->getTerminator()) ||
      !isa<BranchInst>(Pred2->getTerminator()))
    return 0;
  BranchInst *Pred1Br = cast<BranchInst>(Pred1->getTerminator());
  BranchInst *Pred2Br = cast<BranchInst>(Pred2->getTerminator());

  // Eliminate code duplication by ensuring that Pred1Br is conditional if
  // either are.
  if (Pred2Br->isConditional()) {
    // If both branches are conditional, we don't have an "if statement".  In
    // reality, we could transform this case, but since the condition will be
    // required anyway, we stand no chance of eliminating it, so the xform is
    // probably not profitable.
    if (Pred1Br->isConditional())
      return 0;

    std::swap(Pred1, Pred2);
    std::swap(Pred1Br, Pred2Br);
  }

  if (Pred1Br->isConditional()) {
    // If we found a conditional branch predecessor, make sure that it branches
    // to BB and Pred2Br.  If it doesn't, this isn't an "if statement".
    if (Pred1Br->getSuccessor(0) == BB &&
        Pred1Br->getSuccessor(1) == Pred2) {
      IfTrue = Pred1;
      IfFalse = Pred2;
    } else if (Pred1Br->getSuccessor(0) == Pred2 &&
               Pred1Br->getSuccessor(1) == BB) {
      IfTrue = Pred2;
      IfFalse = Pred1;
    } else {
      // We know that one arm of the conditional goes to BB, so the other must
      // go somewhere unrelated, and this must not be an "if statement".
      return 0;
    }

    // The only thing we have to watch out for here is to make sure that Pred2
    // doesn't have incoming edges from other blocks.  If it does, the condition
    // doesn't dominate BB.
    if (++pred_begin(Pred2) != pred_end(Pred2))
      return 0;

    return Pred1Br->getCondition();
  }

  // Ok, if we got here, both predecessors end with an unconditional branch to
  // BB.  Don't panic!  If both blocks only have a single (identical)
  // predecessor, and THAT is a conditional branch, then we're all ok!
  if (pred_begin(Pred1) == pred_end(Pred1) ||
      ++pred_begin(Pred1) != pred_end(Pred1) ||
      pred_begin(Pred2) == pred_end(Pred2) ||
      ++pred_begin(Pred2) != pred_end(Pred2) ||
      *pred_begin(Pred1) != *pred_begin(Pred2))
    return 0;

  // Otherwise, if this is a conditional branch, then we can use it!
  BasicBlock *CommonPred = *pred_begin(Pred1);
  if (BranchInst *BI = dyn_cast<BranchInst>(CommonPred->getTerminator())) {
    assert(BI->isConditional() && "Two successors but not conditional?");
    if (BI->getSuccessor(0) == Pred1) {
      IfTrue = Pred1;
      IfFalse = Pred2;
    } else {
      IfTrue = Pred2;
      IfFalse = Pred1;
    }
    return BI->getCondition();
  }
  return 0;
}


// If we have a merge point of an "if condition" as accepted above, return true
// if the specified value dominates the block.  We don't handle the true
// generality of domination here, just a special case which works well enough
// for us.
static bool DominatesMergePoint(Value *V, BasicBlock *BB) {
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    BasicBlock *PBB = I->getParent();
    // If this instruction is defined in a block that contains an unconditional
    // branch to BB, then it must be in the 'conditional' part of the "if
    // statement".
    if (isa<BranchInst>(PBB->getTerminator()) && 
        cast<BranchInst>(PBB->getTerminator())->isUnconditional() && 
        cast<BranchInst>(PBB->getTerminator())->getSuccessor(0) == BB)
      return false;

    // We also don't want to allow wierd loops that might have the "if
    // condition" in the bottom of this block.
    if (PBB == BB) return false;
  }

  // Non-instructions all dominate instructions.
  return true;
}

// GatherConstantSetEQs - Given a potentially 'or'd together collection of seteq
// instructions that compare a value against a constant, return the value being
// compared, and stick the constant into the Values vector.
static Value *GatherConstantSetEQs(Value *V, std::vector<Constant*> &Values) {
  if (Instruction *Inst = dyn_cast<Instruction>(V))
    if (Inst->getOpcode() == Instruction::SetEQ) {
      if (Constant *C = dyn_cast<Constant>(Inst->getOperand(1))) {
        Values.push_back(C);
        return Inst->getOperand(0);
      } else if (Constant *C = dyn_cast<Constant>(Inst->getOperand(0))) {
        Values.push_back(C);
        return Inst->getOperand(1);
      }
    } else if (Inst->getOpcode() == Instruction::Or) {
      if (Value *LHS = GatherConstantSetEQs(Inst->getOperand(0), Values))
        if (Value *RHS = GatherConstantSetEQs(Inst->getOperand(1), Values))
          if (LHS == RHS)
            return LHS;
    }
  return 0;
}

// GatherConstantSetNEs - Given a potentially 'and'd together collection of
// setne instructions that compare a value against a constant, return the value
// being compared, and stick the constant into the Values vector.
static Value *GatherConstantSetNEs(Value *V, std::vector<Constant*> &Values) {
  if (Instruction *Inst = dyn_cast<Instruction>(V))
    if (Inst->getOpcode() == Instruction::SetNE) {
      if (Constant *C = dyn_cast<Constant>(Inst->getOperand(1))) {
        Values.push_back(C);
        return Inst->getOperand(0);
      } else if (Constant *C = dyn_cast<Constant>(Inst->getOperand(0))) {
        Values.push_back(C);
        return Inst->getOperand(1);
      }
    } else if (Inst->getOpcode() == Instruction::Cast) {
      // Cast of X to bool is really a comparison against zero.
      assert(Inst->getType() == Type::BoolTy && "Can only handle bool values!");
      Values.push_back(Constant::getNullValue(Inst->getOperand(0)->getType()));
      return Inst->getOperand(0);
    } else if (Inst->getOpcode() == Instruction::And) {
      if (Value *LHS = GatherConstantSetNEs(Inst->getOperand(0), Values))
        if (Value *RHS = GatherConstantSetNEs(Inst->getOperand(1), Values))
          if (LHS == RHS)
            return LHS;
    }
  return 0;
}



/// GatherValueComparisons - If the specified Cond is an 'and' or 'or' of a
/// bunch of comparisons of one value against constants, return the value and
/// the constants being compared.
static bool GatherValueComparisons(Instruction *Cond, Value *&CompVal,
                                   std::vector<Constant*> &Values) {
  if (Cond->getOpcode() == Instruction::Or) {
    CompVal = GatherConstantSetEQs(Cond, Values);

    // Return true to indicate that the condition is true if the CompVal is
    // equal to one of the constants.
    return true;
  } else if (Cond->getOpcode() == Instruction::And) {
    CompVal = GatherConstantSetNEs(Cond, Values);
        
    // Return false to indicate that the condition is false if the CompVal is
    // equal to one of the constants.
    return false;
  }
  return false;
}

/// ErasePossiblyDeadInstructionTree - If the specified instruction is dead and
/// has no side effects, nuke it.  If it uses any instructions that become dead
/// because the instruction is now gone, nuke them too.
static void ErasePossiblyDeadInstructionTree(Instruction *I) {
  if (isInstructionTriviallyDead(I)) {
    std::vector<Value*> Operands(I->op_begin(), I->op_end());
    I->getParent()->getInstList().erase(I);
    for (unsigned i = 0, e = Operands.size(); i != e; ++i)
      if (Instruction *OpI = dyn_cast<Instruction>(Operands[i]))
        ErasePossiblyDeadInstructionTree(OpI);
  }
}

/// SafeToMergeTerminators - Return true if it is safe to merge these two
/// terminator instructions together.
///
static bool SafeToMergeTerminators(TerminatorInst *SI1, TerminatorInst *SI2) {
  if (SI1 == SI2) return false;  // Can't merge with self!

  // It is not safe to merge these two switch instructions if they have a common
  // successor, and if that successor has a PHI node, and if that PHI node has
  // conflicting incoming values from the two switch blocks.
  BasicBlock *SI1BB = SI1->getParent();
  BasicBlock *SI2BB = SI2->getParent();
  std::set<BasicBlock*> SI1Succs(succ_begin(SI1BB), succ_end(SI1BB));

  for (succ_iterator I = succ_begin(SI2BB), E = succ_end(SI2BB); I != E; ++I)
    if (SI1Succs.count(*I))
      for (BasicBlock::iterator BBI = (*I)->begin();
           PHINode *PN = dyn_cast<PHINode>(BBI); ++BBI)
        if (PN->getIncomingValueForBlock(SI1BB) !=
            PN->getIncomingValueForBlock(SI2BB))
          return false;
        
  return true;
}

/// AddPredecessorToBlock - Update PHI nodes in Succ to indicate that there will
/// now be entries in it from the 'NewPred' block.  The values that will be
/// flowing into the PHI nodes will be the same as those coming in from
/// ExistPred, and existing predecessor of Succ.
static void AddPredecessorToBlock(BasicBlock *Succ, BasicBlock *NewPred,
                                  BasicBlock *ExistPred) {
  assert(std::find(succ_begin(ExistPred), succ_end(ExistPred), Succ) !=
         succ_end(ExistPred) && "ExistPred is not a predecessor of Succ!");
  if (!isa<PHINode>(Succ->begin())) return; // Quick exit if nothing to do

  for (BasicBlock::iterator I = Succ->begin();
       PHINode *PN = dyn_cast<PHINode>(I); ++I) {
    Value *V = PN->getIncomingValueForBlock(ExistPred);
    PN->addIncoming(V, NewPred);
  }
}

// SimplifyCFG - This function is used to do simplification of a CFG.  For
// example, it adjusts branches to branches to eliminate the extra hop, it
// eliminates unreachable basic blocks, and does other "peephole" optimization
// of the CFG.  It returns true if a modification was made.
//
// WARNING:  The entry node of a function may not be simplified.
//
bool llvm::SimplifyCFG(BasicBlock *BB) {
  bool Changed = false;
  Function *M = BB->getParent();

  assert(BB && BB->getParent() && "Block not embedded in function!");
  assert(BB->getTerminator() && "Degenerate basic block encountered!");
  assert(&BB->getParent()->front() != BB && "Can't Simplify entry block!");

  // Remove basic blocks that have no predecessors... which are unreachable.
  if (pred_begin(BB) == pred_end(BB) ||
      *pred_begin(BB) == BB && ++pred_begin(BB) == pred_end(BB)) {
    //cerr << "Removing BB: \n" << BB;

    // Loop through all of our successors and make sure they know that one
    // of their predecessors is going away.
    for_each(succ_begin(BB), succ_end(BB),
	     std::bind2nd(std::mem_fun(&BasicBlock::removePredecessor), BB));

    while (!BB->empty()) {
      Instruction &I = BB->back();
      // If this instruction is used, replace uses with an arbitrary
      // constant value.  Because control flow can't get here, we don't care
      // what we replace the value with.  Note that since this block is 
      // unreachable, and all values contained within it must dominate their
      // uses, that all uses will eventually be removed.
      if (!I.use_empty()) 
        // Make all users of this instruction reference the constant instead
        I.replaceAllUsesWith(Constant::getNullValue(I.getType()));
      
      // Remove the instruction from the basic block
      BB->getInstList().pop_back();
    }
    M->getBasicBlockList().erase(BB);
    return true;
  }

  // Check to see if we can constant propagate this terminator instruction
  // away...
  Changed |= ConstantFoldTerminator(BB);

  // Check to see if this block has no non-phi instructions and only a single
  // successor.  If so, replace references to this basic block with references
  // to the successor.
  succ_iterator SI(succ_begin(BB));
  if (SI != succ_end(BB) && ++SI == succ_end(BB)) {  // One succ?

    BasicBlock::iterator BBI = BB->begin();  // Skip over phi nodes...
    while (isa<PHINode>(*BBI)) ++BBI;

    if (BBI->isTerminator()) {   // Terminator is the only non-phi instruction!
      BasicBlock *Succ = *succ_begin(BB); // There is exactly one successor
     
      if (Succ != BB) {   // Arg, don't hurt infinite loops!
        // If our successor has PHI nodes, then we need to update them to
        // include entries for BB's predecessors, not for BB itself.
        // Be careful though, if this transformation fails (returns true) then
        // we cannot do this transformation!
        //
	if (!PropagatePredecessorsForPHIs(BB, Succ)) {
          //cerr << "Killing Trivial BB: \n" << BB;
          std::string OldName = BB->getName();

          std::vector<BasicBlock*>
            OldSuccPreds(pred_begin(Succ), pred_end(Succ));

          // Move all PHI nodes in BB to Succ if they are alive, otherwise
          // delete them.
          while (PHINode *PN = dyn_cast<PHINode>(&BB->front()))
            if (PN->use_empty())
              BB->getInstList().erase(BB->begin());  // Nuke instruction...
            else {
              // The instruction is alive, so this means that Succ must have
              // *ONLY* had BB as a predecessor, and the PHI node is still valid
              // now.  Simply move it into Succ, because we know that BB
              // strictly dominated Succ.
              BB->getInstList().remove(BB->begin());
              Succ->getInstList().push_front(PN);

              // We need to add new entries for the PHI node to account for
              // predecessors of Succ that the PHI node does not take into
              // account.  At this point, since we know that BB dominated succ,
              // this means that we should any newly added incoming edges should
              // use the PHI node as the value for these edges, because they are
              // loop back edges.
              
              for (unsigned i = 0, e = OldSuccPreds.size(); i != e; ++i)
                if (OldSuccPreds[i] != BB)
                  PN->addIncoming(PN, OldSuccPreds[i]);
            }

          // Everything that jumped to BB now goes to Succ...
          BB->replaceAllUsesWith(Succ);

          // Delete the old basic block...
          M->getBasicBlockList().erase(BB);
	
          if (!OldName.empty() && !Succ->hasName())  // Transfer name if we can
            Succ->setName(OldName);
          
          //cerr << "Function after removal: \n" << M;
          return true;
	}
      }
    }
  }

  // If this is a returning block with only PHI nodes in it, fold the return
  // instruction into any unconditional branch predecessors.
  if (ReturnInst *RI = dyn_cast<ReturnInst>(BB->getTerminator())) {
    BasicBlock::iterator BBI = BB->getTerminator();
    if (BBI == BB->begin() || isa<PHINode>(--BBI)) {
      // Find predecessors that end with unconditional branches.
      std::vector<BasicBlock*> UncondBranchPreds;
      for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
        TerminatorInst *PTI = (*PI)->getTerminator();
        if (BranchInst *BI = dyn_cast<BranchInst>(PTI))
          if (BI->isUnconditional())
            UncondBranchPreds.push_back(*PI);
      }
      
      // If we found some, do the transformation!
      if (!UncondBranchPreds.empty()) {
        while (!UncondBranchPreds.empty()) {
          BasicBlock *Pred = UncondBranchPreds.back();
          UncondBranchPreds.pop_back();
          Instruction *UncondBranch = Pred->getTerminator();
          // Clone the return and add it to the end of the predecessor.
          Instruction *NewRet = RI->clone();
          Pred->getInstList().push_back(NewRet);

          // If the return instruction returns a value, and if the value was a
          // PHI node in "BB", propagate the right value into the return.
          if (NewRet->getNumOperands() == 1)
            if (PHINode *PN = dyn_cast<PHINode>(NewRet->getOperand(0)))
              if (PN->getParent() == BB)
                NewRet->setOperand(0, PN->getIncomingValueForBlock(Pred));
          // Update any PHI nodes in the returning block to realize that we no
          // longer branch to them.
          BB->removePredecessor(Pred);
          Pred->getInstList().erase(UncondBranch);
        }

        // If we eliminated all predecessors of the block, delete the block now.
        if (pred_begin(BB) == pred_end(BB))
          // We know there are no successors, so just nuke the block.
          M->getBasicBlockList().erase(BB);

        return true;
      }
    }
  } else if (UnwindInst *UI = dyn_cast<UnwindInst>(BB->begin())) {
    // Check to see if the first instruction in this block is just an unwind.
    // If so, replace any invoke instructions which use this as an exception
    // destination with call instructions.
    //
    std::vector<BasicBlock*> Preds(pred_begin(BB), pred_end(BB));
    while (!Preds.empty()) {
      BasicBlock *Pred = Preds.back();
      if (InvokeInst *II = dyn_cast<InvokeInst>(Pred->getTerminator()))
        if (II->getUnwindDest() == BB) {
          // Insert a new branch instruction before the invoke, because this
          // is now a fall through...
          BranchInst *BI = new BranchInst(II->getNormalDest(), II);
          Pred->getInstList().remove(II);   // Take out of symbol table
          
          // Insert the call now...
          std::vector<Value*> Args(II->op_begin()+3, II->op_end());
          CallInst *CI = new CallInst(II->getCalledValue(), Args,
                                      II->getName(), BI);
          // If the invoke produced a value, the Call now does instead
          II->replaceAllUsesWith(CI);
          delete II;
          Changed = true;
        }
      
      Preds.pop_back();
    }

    // If this block is now dead, remove it.
    if (pred_begin(BB) == pred_end(BB)) {
      // We know there are no successors, so just nuke the block.
      M->getBasicBlockList().erase(BB);
      return true;
    }

  } else if (SwitchInst *SI = dyn_cast<SwitchInst>(BB->begin())) {
    // If the only instruction in this block is a switch instruction, see if we
    // can fold the switch instruction into a switch in a predecessor block.
    std::vector<BasicBlock*> Preds(pred_begin(BB), pred_end(BB));
    while (!Preds.empty()) {
      BasicBlock *Pred = Preds.back();
      Preds.pop_back();

      // If the two blocks are switching on the same value, we can merge this
      // switch into the predecessor's switch.
      if (SwitchInst *PSI = dyn_cast<SwitchInst>(Pred->getTerminator()))
        if (PSI->getCondition() == SI->getCondition() &&
            SafeToMergeTerminators(SI, PSI)) {
          // Figure out which 'cases' to copy from SI to PSI.
          std::vector<std::pair<Constant*, BasicBlock*> > Cases;
          BasicBlock *NewDefault = 0;
          if (PSI->getDefaultDest() == BB) {
            // If this is the default destination from PSI, only the edges in SI
            // that don't occur in PSI, or that branch to BB will be activated.
            std::set<Constant*> PSIHandled;
            for (unsigned i = 1, e = PSI->getNumSuccessors(); i != e; ++i)
              if (PSI->getSuccessor(i) != BB)
                PSIHandled.insert(PSI->getCaseValue(i));
              else {
                // This entry will be replaced.
                PSI->removeCase(i);
                --i; --e;
              }

            NewDefault = SI->getDefaultDest();
            for (unsigned i = 1, e = SI->getNumSuccessors(); i != e; ++i) {
              Constant *C = SI->getCaseValue(i);
              if (!PSIHandled.count(C))
                Cases.push_back(std::make_pair(C, SI->getSuccessor(i)));
            }

          } else {
            // If this is not the default destination from PSI, only the edges
            // in SI that occur in PSI with a destination of BB will be
            // activated.
            std::set<Constant*> PSIHandled;
            for (unsigned i = 1, e = PSI->getNumSuccessors(); i != e; ++i)
              if (PSI->getSuccessor(i) == BB) {
                // We know that BB doesn't have any PHI nodes in it, so just
                // drop the edges.
                PSIHandled.insert(PSI->getCaseValue(i));
                PSI->removeCase(i);
                --i; --e;
              }

            // Okay, now we know which constants were sent to BB from the
            // predecessor.  Figure out where they will all go now.
            for (unsigned i = 1, e = SI->getNumSuccessors(); i != e; ++i) {
              Constant *C = SI->getCaseValue(i);
              if (PSIHandled.count(C)) {
                // If this is one we are capable of getting...
                Cases.push_back(std::make_pair(C, SI->getSuccessor(i)));
                PSIHandled.erase(C);         // This constant is taken care of
              }
            }

            // If there are any constants vectored to BB that SI doesn't handle,
            // they must go to the default destination of SI.
            for (std::set<Constant*>::iterator I = PSIHandled.begin(),
                   E = PSIHandled.end(); I != E; ++I)
              Cases.push_back(std::make_pair(*I, SI->getDefaultDest()));
          }

          // Okay, at this point, we know which cases need to be added to the
          // PSI switch and which destinations they go to.  If PSI needs its
          // default destination changed, NewDefault is set.  Start changing
          // stuff now.
          if (NewDefault) {
            AddPredecessorToBlock(NewDefault, Pred, BB);
            PSI->setSuccessor(0, NewDefault);
          }

          // Okay, add all of the cases now.
          for (unsigned i = 0, e = Cases.size(); i != e; ++i) {
            AddPredecessorToBlock(Cases[i].second, Pred, BB);
            PSI->addCase(Cases[i].first, Cases[i].second);
          }

          // Okay, last check.  If BB is still a successor of PSI, then we must
          // have an infinite loop case.  If so, add an infinitely looping block
          // to handle the case to preserve the behavior of the code.
          BasicBlock *InfLoopBlock = 0;
          for (unsigned i = 0, e = PSI->getNumSuccessors(); i != e; ++i)
            if (PSI->getSuccessor(i) == BB) {
              if (InfLoopBlock == 0) {
                // Insert it at the end of the loop, because it's either code,
                // or it won't matter if it's hot. :)
                InfLoopBlock = new BasicBlock("infloop", BB->getParent());
                new BranchInst(InfLoopBlock, InfLoopBlock);
              }
              PSI->setSuccessor(i, InfLoopBlock);
            }
          
          Changed = true;
        }
    }

    // If we removed all predecessors of this block, recursively call
    // SimplifyCFG to remove it.
    if (pred_begin(BB) == pred_end(BB))
      return SimplifyCFG(BB);
  }

  // Merge basic blocks into their predecessor if there is only one distinct
  // pred, and if there is only one distinct successor of the predecessor, and
  // if there are no PHI nodes.
  //
  pred_iterator PI(pred_begin(BB)), PE(pred_end(BB));
  BasicBlock *OnlyPred = *PI++;
  for (; PI != PE; ++PI)  // Search all predecessors, see if they are all same
    if (*PI != OnlyPred) {
      OnlyPred = 0;       // There are multiple different predecessors...
      break;
    }
  
  BasicBlock *OnlySucc = 0;
  if (OnlyPred && OnlyPred != BB &&    // Don't break self loops
      OnlyPred->getTerminator()->getOpcode() != Instruction::Invoke) {
    // Check to see if there is only one distinct successor...
    succ_iterator SI(succ_begin(OnlyPred)), SE(succ_end(OnlyPred));
    OnlySucc = BB;
    for (; SI != SE; ++SI)
      if (*SI != OnlySucc) {
        OnlySucc = 0;     // There are multiple distinct successors!
        break;
      }
  }

  if (OnlySucc) {
    //cerr << "Merging: " << BB << "into: " << OnlyPred;
    TerminatorInst *Term = OnlyPred->getTerminator();

    // Resolve any PHI nodes at the start of the block.  They are all
    // guaranteed to have exactly one entry if they exist, unless there are
    // multiple duplicate (but guaranteed to be equal) entries for the
    // incoming edges.  This occurs when there are multiple edges from
    // OnlyPred to OnlySucc.
    //
    while (PHINode *PN = dyn_cast<PHINode>(&BB->front())) {
      PN->replaceAllUsesWith(PN->getIncomingValue(0));
      BB->getInstList().pop_front();  // Delete the phi node...
    }

    // Delete the unconditional branch from the predecessor...
    OnlyPred->getInstList().pop_back();
      
    // Move all definitions in the successor to the predecessor...
    OnlyPred->getInstList().splice(OnlyPred->end(), BB->getInstList());
                                     
    // Make all PHI nodes that referred to BB now refer to Pred as their
    // source...
    BB->replaceAllUsesWith(OnlyPred);

    std::string OldName = BB->getName();

    // Erase basic block from the function... 
    M->getBasicBlockList().erase(BB);

    // Inherit predecessors name if it exists...
    if (!OldName.empty() && !OnlyPred->hasName())
      OnlyPred->setName(OldName);
      
    return true;
  }

  // If there is a single predecessor for this block, and if this block is a
  // simple value comparison block (ie, contains X == C), see if we can fold
  // this comparison into the comparison in our predecessor block, making the
  // predecessor block terminator into a switch (or adding cases to a
  // preexisting switch).
  if (OnlyPred) {
    if (SetCondInst *SCI = dyn_cast<SetCondInst>(BB->begin()))
      if (SCI->getOpcode() == Instruction::SetEQ && SCI->hasOneUse() &&
          isa<BranchInst>(SCI->use_back()) &&
          SCI->getNext() == cast<BranchInst>(SCI->use_back())) {
        // Okay, we know we have a block containing (only) a seteq and a
        // conditional branch instruction.  If an integer value is being
        // compared, if the comparison value is a constant, then check the
        // predecessor.
        BranchInst *BBBr = cast<BranchInst>(BB->getTerminator());
        Value *CompVal = SCI->getOperand(0);
        if (ConstantInt *CVal = dyn_cast<ConstantInt>(SCI->getOperand(1))) {
          // We can do the merge if the predecessor contains either a
          // conditional branch or a switch instruction which is operating on
          // the CompVal.
          if (BranchInst *BI = dyn_cast<BranchInst>(OnlyPred->getTerminator())){
            // If it is a branch, then it must be a conditional branch,
            // otherwise we would have merged it in before.  We can only handle
            // this if the block we are looking at is the 'false' branch.
            assert(BI->isConditional() &&
                   "Should have previously merged blocks!");
            if (SetCondInst *PredSCC= dyn_cast<SetCondInst>(BI->getCondition()))
              if (PredSCC->getOperand(0) == CompVal &&
                  PredSCC->getOpcode() == Instruction::SetEQ &&
                  isa<ConstantInt>(PredSCC->getOperand(1)) &&
                  BB == BI->getSuccessor(1) &&
                  SafeToMergeTerminators(BI, BBBr)) {
                // If the constants being compared are the same, then the
                // comparison in this block could never come true.
                if (SCI->getOperand(1) == PredSCC->getOperand(1)) {
                  // Tell the block to skip over us, making us dead.
                  BI->setSuccessor(1, BBBr->getSuccessor(1));
                  AddPredecessorToBlock(BBBr->getSuccessor(1), OnlyPred, BB);
                  return SimplifyCFG(BB);
                }

                // Otherwise, create the switch instruction!
                SwitchInst *SI = new SwitchInst(CompVal, BBBr->getSuccessor(1),
                                                BI);
                // Add the edge from our predecessor, and remove the
                // predecessors (now obsolete branch instruction).  This makes
                // the current block dead.
                SI->addCase(cast<Constant>(PredSCC->getOperand(1)),
                            BI->getSuccessor(0));
                OnlyPred->getInstList().erase(BI);
                if (PredSCC->use_empty())
                  PredSCC->getParent()->getInstList().erase(PredSCC);

                // Add our case...
                SI->addCase(cast<Constant>(SCI->getOperand(1)),
                            BBBr->getSuccessor(0));

                AddPredecessorToBlock(BBBr->getSuccessor(0), OnlyPred, BB);
                AddPredecessorToBlock(BBBr->getSuccessor(1), OnlyPred, BB);

                //std::cerr << "Formed Switch: " << SI;

                // Made a big change!  Now this block is dead, so remove it.
                return SimplifyCFG(BB);
              }

          } else if (SwitchInst *SI =
                     dyn_cast<SwitchInst>(OnlyPred->getTerminator())) {

          }
        }
      }
  }

  for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI)
    if (BranchInst *BI = dyn_cast<BranchInst>((*PI)->getTerminator()))
      // Change br (X == 0 | X == 1), T, F into a switch instruction.
      if (BI->isConditional() && isa<Instruction>(BI->getCondition())) {
        Instruction *Cond = cast<Instruction>(BI->getCondition());
        // If this is a bunch of seteq's or'd together, or if it's a bunch of
        // 'setne's and'ed together, collect them.
        Value *CompVal = 0;
        std::vector<Constant*> Values;
        bool TrueWhenEqual = GatherValueComparisons(Cond, CompVal, Values);
        if (CompVal && CompVal->getType()->isInteger()) {
          // There might be duplicate constants in the list, which the switch
          // instruction can't handle, remove them now.
          std::sort(Values.begin(), Values.end());
          Values.erase(std::unique(Values.begin(), Values.end()), Values.end());
          
          // Figure out which block is which destination.
          BasicBlock *DefaultBB = BI->getSuccessor(1);
          BasicBlock *EdgeBB    = BI->getSuccessor(0);
          if (!TrueWhenEqual) std::swap(DefaultBB, EdgeBB);
          
          // Create the new switch instruction now.
          SwitchInst *New = new SwitchInst(CompVal, DefaultBB, BI);
          
          // Add all of the 'cases' to the switch instruction.
          for (unsigned i = 0, e = Values.size(); i != e; ++i)
            New->addCase(Values[i], EdgeBB);
          
          // We added edges from PI to the EdgeBB.  As such, if there were any
          // PHI nodes in EdgeBB, they need entries to be added corresponding to
          // the number of edges added.
          for (BasicBlock::iterator BBI = EdgeBB->begin();
               PHINode *PN = dyn_cast<PHINode>(BBI); ++BBI) {
            Value *InVal = PN->getIncomingValueForBlock(*PI);
            for (unsigned i = 0, e = Values.size()-1; i != e; ++i)
              PN->addIncoming(InVal, *PI);
          }

          // Erase the old branch instruction.
          (*PI)->getInstList().erase(BI);

          // Erase the potentially condition tree that was used to computed the
          // branch condition.
          ErasePossiblyDeadInstructionTree(Cond);
          return true;
        }
      }

  // If there is a trivial two-entry PHI node in this basic block, and we can
  // eliminate it, do so now.
  if (PHINode *PN = dyn_cast<PHINode>(BB->begin()))
    if (PN->getNumIncomingValues() == 2) {
      // Ok, this is a two entry PHI node.  Check to see if this is a simple "if
      // statement", which has a very simple dominance structure.  Basically, we
      // are trying to find the condition that is being branched on, which
      // subsequently causes this merge to happen.  We really want control
      // dependence information for this check, but simplifycfg can't keep it up
      // to date, and this catches most of the cases we care about anyway.
      //
      BasicBlock *IfTrue, *IfFalse;
      if (Value *IfCond = GetIfCondition(BB, IfTrue, IfFalse)) {
        //std::cerr << "FOUND IF CONDITION!  " << *IfCond << "  T: "
        //       << IfTrue->getName() << "  F: " << IfFalse->getName() << "\n";

        // Figure out where to insert instructions as necessary.
        BasicBlock::iterator AfterPHIIt = BB->begin();
        while (isa<PHINode>(AfterPHIIt)) ++AfterPHIIt;

        BasicBlock::iterator I = BB->begin();
        while (PHINode *PN = dyn_cast<PHINode>(I)) {
          ++I;

          // If we can eliminate this PHI by directly computing it based on the
          // condition, do so now.  We can't eliminate PHI nodes where the
          // incoming values are defined in the conditional parts of the branch,
          // so check for this.
          //
          if (DominatesMergePoint(PN->getIncomingValue(0), BB) &&
              DominatesMergePoint(PN->getIncomingValue(1), BB)) {
            Value *TrueVal =
              PN->getIncomingValue(PN->getIncomingBlock(0) == IfFalse);
            Value *FalseVal =
              PN->getIncomingValue(PN->getIncomingBlock(0) == IfTrue);

            // FIXME: when we have a 'select' statement, we can be completely
            // generic and clean here and let the instcombine pass clean up
            // after us, by folding the select instructions away when possible.
            //
            if (TrueVal == FalseVal) {
              // Degenerate case...
              PN->replaceAllUsesWith(TrueVal);
              BB->getInstList().erase(PN);
              Changed = true;
            } else if (isa<ConstantBool>(TrueVal) &&
                       isa<ConstantBool>(FalseVal)) {
              if (TrueVal == ConstantBool::True) {
                // The PHI node produces the same thing as the condition.
                PN->replaceAllUsesWith(IfCond);
              } else {
                // The PHI node produces the inverse of the condition.  Insert a
                // "NOT" instruction, which is really a XOR.
                Value *InverseCond =
                  BinaryOperator::createNot(IfCond, IfCond->getName()+".inv",
                                            AfterPHIIt);
                PN->replaceAllUsesWith(InverseCond);
              }
              BB->getInstList().erase(PN);
              Changed = true;
            } else if (isa<ConstantInt>(TrueVal) && isa<ConstantInt>(FalseVal)){
              // If this is a PHI of two constant integers, we insert a cast of
              // the boolean to the integer type in question, giving us 0 or 1.
              // Then we multiply this by the difference of the two constants,
              // giving us 0 if false, and the difference if true.  We add this
              // result to the base constant, giving us our final value.  We
              // rely on the instruction combiner to eliminate many special
              // cases, like turning multiplies into shifts when possible.
              std::string Name = PN->getName(); PN->setName("");
              Value *TheCast = new CastInst(IfCond, TrueVal->getType(),
                                            Name, AfterPHIIt);
              Constant *TheDiff = ConstantExpr::get(Instruction::Sub,
                                                    cast<Constant>(TrueVal),
                                                    cast<Constant>(FalseVal));
              Value *V = TheCast;
              if (TheDiff != ConstantInt::get(TrueVal->getType(), 1))
                V = BinaryOperator::create(Instruction::Mul, TheCast,
                                           TheDiff, TheCast->getName()+".scale",
                                           AfterPHIIt);
              if (!cast<Constant>(FalseVal)->isNullValue())
                V = BinaryOperator::create(Instruction::Add, V, FalseVal,
                                           V->getName()+".offs", AfterPHIIt);
              PN->replaceAllUsesWith(V);
              BB->getInstList().erase(PN);
              Changed = true;
            } else if (isa<ConstantInt>(FalseVal) &&
                       cast<Constant>(FalseVal)->isNullValue()) {
              // If the false condition is an integral zero value, we can
              // compute the PHI by multiplying the condition by the other
              // value.
              std::string Name = PN->getName(); PN->setName("");
              Value *TheCast = new CastInst(IfCond, TrueVal->getType(),
                                            Name+".c", AfterPHIIt);
              Value *V = BinaryOperator::create(Instruction::Mul, TrueVal,
                                                TheCast, Name, AfterPHIIt);
              PN->replaceAllUsesWith(V);
              BB->getInstList().erase(PN);
              Changed = true;
            } else if (isa<ConstantInt>(TrueVal) &&
                       cast<Constant>(TrueVal)->isNullValue()) {
              // If the true condition is an integral zero value, we can compute
              // the PHI by multiplying the inverse condition by the other
              // value.
              std::string Name = PN->getName(); PN->setName("");
              Value *NotCond = BinaryOperator::createNot(IfCond, Name+".inv",
                                                         AfterPHIIt);
              Value *TheCast = new CastInst(NotCond, TrueVal->getType(),
                                            Name+".inv", AfterPHIIt);
              Value *V = BinaryOperator::create(Instruction::Mul, FalseVal,
                                                TheCast, Name, AfterPHIIt);
              PN->replaceAllUsesWith(V);
              BB->getInstList().erase(PN);
              Changed = true;
            }
          }
        }
      }
    }
  
  return Changed;
}
