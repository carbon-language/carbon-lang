//===- PiNodeInsertion.cpp - Insert Pi nodes into a program ---------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// PiNodeInsertion - This pass inserts single entry Phi nodes into basic blocks
// that are preceded by a conditional branch, where the branch gives
// information about the operands of the condition.  For example, this C code:
//   if (x == 0) { ... = x + 4;
// becomes:
//   if (x == 0) {
//     x2 = phi(x);    // Node that can hold data flow information about X
//     ... = x2 + 4;
//
// Since the direction of the condition branch gives information about X itself
// (whether or not it is zero), some passes (like value numbering or ABCD) can
// use the inserted Phi/Pi nodes as a place to attach information, in this case
// saying that X has a value of 0 in this scope.  The power of this analysis
// information is that "in the scope" translates to "for all uses of x2".
//
// This special form of Phi node is referred to as a Pi node, following the
// terminology defined in the "Array Bounds Checks on Demand" paper.
//
// As a really trivial example of what the Pi nodes are good for, this pass
// replaces values compared for equality with direct constants with the constant
// itself in the branch it's equal to the constant.  In the case above, it would
// change the body to be "... = 0 + 4;"  Real value numbering can do much more.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/iTerminators.h"
#include "llvm/iOperators.h"
#include "llvm/iPHINode.h"
#include "llvm/Support/CFG.h"
#include "Support/Statistic.h"

namespace llvm {

namespace {
  Statistic<> NumInserted("pinodes", "Number of Pi nodes inserted");

  struct PiNodeInserter : public FunctionPass {
    virtual bool runOnFunction(Function &F);
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<DominatorSet>();
    }

    // insertPiNodeFor - Insert a Pi node for V in the successors of BB if our
    // conditions hold.  If Rep is not null, fill in a value of 'Rep' instead of
    // creating a new Pi node itself because we know that the value is a simple
    // constant.
    //
    bool insertPiNodeFor(Value *V, BasicBlock *BB, Value *Rep = 0);
  };

  RegisterOpt<PiNodeInserter> X("pinodes", "Pi Node Insertion");
}

Pass *createPiNodeInsertionPass() { return new PiNodeInserter(); }


bool PiNodeInserter::runOnFunction(Function &F) {
  bool Changed = false;
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    TerminatorInst *TI = I->getTerminator();
    
    // FIXME: Insert PI nodes for switch statements too

    // Look for conditional branch instructions... that branch on a setcc test
    if (BranchInst *BI = dyn_cast<BranchInst>(TI))
      if (BI->isConditional())
        // TODO: we could in theory support logical operations here too...
        if (SetCondInst *SCI = dyn_cast<SetCondInst>(BI->getCondition())) {
          // Calculate replacement values if this is an obvious constant == or
          // != comparison...
          Value *TrueRep = 0, *FalseRep = 0;

          // Make sure the the constant is the second operand if there is one...
          // This fits with our canonicalization patterns used elsewhere in the
          // compiler, without depending on instcombine running before us.
          //
          if (isa<Constant>(SCI->getOperand(0)) &&
              !isa<Constant>(SCI->getOperand(1))) {
            SCI->swapOperands();
            Changed = true;
          }

          if (isa<Constant>(SCI->getOperand(1))) {
            if (SCI->getOpcode() == Instruction::SetEQ)
              TrueRep = SCI->getOperand(1);
            else if (SCI->getOpcode() == Instruction::SetNE)
              FalseRep = SCI->getOperand(1);
          }

          BasicBlock *TB = BI->getSuccessor(0);  // True block
          BasicBlock *FB = BI->getSuccessor(1);  // False block

          // Insert the Pi nodes for the first operand to the comparison...
          Changed |= insertPiNodeFor(SCI->getOperand(0), TB, TrueRep);
          Changed |= insertPiNodeFor(SCI->getOperand(0), FB, FalseRep);

          // Insert the Pi nodes for the second operand to the comparison...
          Changed |= insertPiNodeFor(SCI->getOperand(1), TB);
          Changed |= insertPiNodeFor(SCI->getOperand(1), FB);
        }
  }

  return Changed;
}


// alreadyHasPiNodeFor - Return true if there is already a Pi node in BB for V.
static bool alreadyHasPiNodeFor(Value *V, BasicBlock *BB) {
  for (Value::use_iterator I = V->use_begin(), E = V->use_end(); I != E; ++I)
    if (PHINode *PN = dyn_cast<PHINode>(*I))
      if (PN->getParent() == BB)
        return true;
  return false;
}


// insertPiNodeFor - Insert a Pi node for V in the successors of BB if our
// conditions hold.  If Rep is not null, fill in a value of 'Rep' instead of
// creating a new Pi node itself because we know that the value is a simple
// constant.
//
bool PiNodeInserter::insertPiNodeFor(Value *V, BasicBlock *Succ, Value *Rep) {
  // Do not insert Pi nodes for constants!
  if (isa<Constant>(V)) return false;

  // Check to make sure that there is not already a PI node inserted...
  if (alreadyHasPiNodeFor(V, Succ) && Rep == 0)
    return false;

  // Insert Pi nodes only into successors that the conditional branch dominates.
  // In this simple case, we know that BB dominates a successor as long there
  // are no other incoming edges to the successor.
  //

  // Check to make sure that the successor only has a single predecessor...
  pred_iterator PI = pred_begin(Succ);
  BasicBlock *Pred = *PI;
  if (++PI != pred_end(Succ)) return false;   // Multiple predecessor?  Bail...

  // It seems to be safe to insert the Pi node.  Do so now...
    
  // Create the Pi node...
  Value *Pi = Rep;
  if (Rep == 0)      // Insert the Pi node in the successor basic block...
    Pi = new PHINode(V->getType(), V->getName() + ".pi", Succ->begin());
    
  // Loop over all of the uses of V, replacing ones that the Pi node
  // dominates with references to the Pi node itself.
  //
  DominatorSet &DS = getAnalysis<DominatorSet>();
  for (Value::use_iterator I = V->use_begin(), E = V->use_end(); I != E; )
    if (Instruction *U = dyn_cast<Instruction>(*I++))
      if (U->getParent()->getParent() == Succ->getParent() &&
          DS.dominates(Succ, U->getParent())) {
        // This instruction is dominated by the Pi node, replace reference to V
        // with a reference to the Pi node.
        //
        U->replaceUsesOfWith(V, Pi);
      }
    
  // Set up the incoming value for the Pi node... do this after uses have been
  // replaced, because we don't want the Pi node to refer to itself.
  //
  if (Rep == 0)
    cast<PHINode>(Pi)->addIncoming(V, Pred);
 

  ++NumInserted;
  return true;
}


} // End llvm namespace
