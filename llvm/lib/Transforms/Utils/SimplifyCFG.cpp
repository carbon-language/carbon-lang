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
#include "llvm/Support/CFG.h"
#include <algorithm>
#include <functional>
using namespace llvm;

// PropagatePredecessors - This gets "Succ" ready to have the predecessors from
// "BB".  This is a little tricky because "Succ" has PHI nodes, which need to
// have extra slots added to them to hold the merge edges from BB's
// predecessors, and BB itself might have had PHI nodes in it.  This function
// returns true (failure) if the Succ BB already has a predecessor that is a
// predecessor of BB and incoming PHI arguments would not be discernible.
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

  // Check to see if the first instruction in this block is just an unwind.  If
  // so, replace any invoke instructions which use this as an exception
  // destination with call instructions.
  //
  if (UnwindInst *UI = dyn_cast<UnwindInst>(BB->getTerminator()))
    if (BB->begin() == BasicBlock::iterator(UI)) {  // Empty block?
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
    }

  // Remove basic blocks that have no predecessors... which are unreachable.
  if (pred_begin(BB) == pred_end(BB)) {
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
