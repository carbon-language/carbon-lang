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
#include "llvm/Constant.h"
#include "llvm/Intrinsics.h"
#include "llvm/iPHINode.h"
#include "llvm/iTerminators.h"
#include "llvm/iOther.h"
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

  // Check to see if the first instruction in this block is just an
  // 'llvm.unwind'.  If so, replace any invoke instructions which use this as an
  // exception destination with call instructions.
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
  if (pred_begin(BB) == pred_end(BB) &&
      !BB->hasConstantReferences()) {
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

  // Merge basic blocks into their predecessor if there is only one distinct
  // pred, and if there is only one distinct successor of the predecessor, and
  // if there are no PHI nodes.
  //
  if (!BB->hasConstantReferences()) {
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
  }
  
  return Changed;
}
