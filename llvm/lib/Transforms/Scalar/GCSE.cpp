//===-- GCSE.cpp - SSA based Global Common Subexpr Elimination ------------===//
//
// This pass is designed to be a very quick global transformation that
// eliminates global common subexpressions from a function.  It does this by
// examining the SSA value graph of the function, instead of doing slow, dense,
// bit-vector computations.
//
// This pass works best if it is proceeded with a simple constant propogation
// pass and an instruction combination pass because this pass does not do any
// value numbering (in order to be speedy).
//
// This pass does not attempt to CSE load instructions, because it does not use
// pointer analysis to determine when it is safe.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/GCSE.h"
#include "llvm/Pass.h"
#include "llvm/InstrTypes.h"
#include "llvm/iMemory.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/InstIterator.h"
#include <set>
#include <algorithm>
using namespace cfg;

namespace {
  class GCSE : public FunctionPass, public InstVisitor<GCSE, bool> {
    set<Instruction*> WorkList;
    DominatorSet        *DomSetInfo;
    ImmediateDominators *ImmDominator;
  public:
    virtual bool runOnFunction(Function *F);

    // Visitation methods, these are invoked depending on the type of
    // instruction being checked.  They should return true if a common
    // subexpression was folded.
    //
    bool visitUnaryOperator(Instruction *I);
    bool visitBinaryOperator(Instruction *I);
    bool visitGetElementPtrInst(GetElementPtrInst *I);
    bool visitCastInst(CastInst *I){return visitUnaryOperator((Instruction*)I);}
    bool visitShiftInst(ShiftInst *I) {
      return visitBinaryOperator((Instruction*)I);
    }
    bool visitInstruction(Instruction *) { return false; }

  private:
    void ReplaceInstWithInst(Instruction *First, BasicBlock::iterator SI);
    void CommonSubExpressionFound(Instruction *I, Instruction *Other);

    // This transformation requires dominator and immediate dominator info
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      //preservesCFG(AU);
      AU.addRequired(DominatorSet::ID);
      AU.addRequired(ImmediateDominators::ID); 
    }
  };
}

// createGCSEPass - The public interface to this file...
Pass *createGCSEPass() { return new GCSE(); }


// GCSE::runOnFunction - This is the main transformation entry point for a
// function.
//
bool GCSE::runOnFunction(Function *F) {
  bool Changed = false;

  DomSetInfo = &getAnalysis<DominatorSet>();
  ImmDominator = &getAnalysis<ImmediateDominators>();

  // Step #1: Add all instructions in the function to the worklist for
  // processing.  All of the instructions are considered to be our
  // subexpressions to eliminate if possible.
  //
  WorkList.insert(inst_begin(F), inst_end(F));

  // Step #2: WorkList processing.  Iterate through all of the instructions,
  // checking to see if there are any additionally defined subexpressions in the
  // program.  If so, eliminate them!
  //
  while (!WorkList.empty()) {
    Instruction *I = *WorkList.begin();  // Get an instruction from the worklist
    WorkList.erase(WorkList.begin());

    // Visit the instruction, dispatching to the correct visit function based on
    // the instruction type.  This does the checking.
    //
    Changed |= visit(I);
  }
  
  // When the worklist is empty, return whether or not we changed anything...
  return Changed;
}


// ReplaceInstWithInst - Destroy the instruction pointed to by SI, making all
// uses of the instruction use First now instead.
//
void GCSE::ReplaceInstWithInst(Instruction *First, BasicBlock::iterator SI) {
  Instruction *Second = *SI;

  // Add the first instruction back to the worklist
  WorkList.insert(First);

  // Add all uses of the second instruction to the worklist
  for (Value::use_iterator UI = Second->use_begin(), UE = Second->use_end();
       UI != UE; ++UI)
    WorkList.insert(cast<Instruction>(*UI));
    
  // Make all users of 'Second' now use 'First'
  Second->replaceAllUsesWith(First);

  // Erase the second instruction from the program
  delete Second->getParent()->getInstList().remove(SI);
}

// CommonSubExpressionFound - The two instruction I & Other have been found to
// be common subexpressions.  This function is responsible for eliminating one
// of them, and for fixing the worklist to be correct.
//
void GCSE::CommonSubExpressionFound(Instruction *I, Instruction *Other) {
  // I has already been removed from the worklist, Other needs to be.
  assert(WorkList.count(I) == 0 && WorkList.count(Other) &&
         "I in worklist or Other not!");
  WorkList.erase(Other);

  // Handle the easy case, where both instructions are in the same basic block
  BasicBlock *BB1 = I->getParent(), *BB2 = Other->getParent();
  if (BB1 == BB2) {
    // Eliminate the second occuring instruction.  Add all uses of the second
    // instruction to the worklist.
    //
    // Scan the basic block looking for the "first" instruction
    BasicBlock::iterator BI = BB1->begin();
    while (*BI != I && *BI != Other) {
      ++BI;
      assert(BI != BB1->end() && "Instructions not found in parent BB!");
    }

    // Keep track of which instructions occurred first & second
    Instruction *First = *BI;
    Instruction *Second = I != First ? I : Other; // Get iterator to second inst
    BI = find(BI, BB1->end(), Second);
    assert(BI != BB1->end() && "Second instruction not found in parent block!");

    // Destroy Second, using First instead.
    ReplaceInstWithInst(First, BI);    

    // Otherwise, the two instructions are in different basic blocks.  If one
    // dominates the other instruction, we can simply use it
    //
  } else if (DomSetInfo->dominates(BB1, BB2)) {    // I dom Other?
    BasicBlock::iterator BI = find(BB2->begin(), BB2->end(), Other);
    assert(BI != BB2->end() && "Other not in parent basic block!");
    ReplaceInstWithInst(I, BI);    
  } else if (DomSetInfo->dominates(BB2, BB1)) {    // Other dom I?
    BasicBlock::iterator BI = find(BB1->begin(), BB1->end(), I);
    assert(BI != BB1->end() && "I not in parent basic block!");
    ReplaceInstWithInst(Other, BI);
  } else {
    // Handle the most general case now.  In this case, neither I dom Other nor
    // Other dom I.  Because we are in SSA form, we are guaranteed that the
    // operands of the two instructions both dominate the uses, so we _know_
    // that there must exist a block that dominates both instructions (if the
    // operands of the instructions are globals or constants, worst case we
    // would get the entry node of the function).  Search for this block now.
    //

    // Search up the immediate dominator chain of BB1 for the shared dominator
    BasicBlock *SharedDom = (*ImmDominator)[BB1];
    while (!DomSetInfo->dominates(SharedDom, BB2))
      SharedDom = (*ImmDominator)[SharedDom];

    // At this point, shared dom must dominate BOTH BB1 and BB2...
    assert(SharedDom && DomSetInfo->dominates(SharedDom, BB1) &&
           DomSetInfo->dominates(SharedDom, BB2) && "Dominators broken!");

    // Rip 'I' out of BB1, and move it to the end of SharedDom.
    BB1->getInstList().remove(I);
    SharedDom->getInstList().insert(SharedDom->end()-1, I);

    // Eliminate 'Other' now.
    BasicBlock::iterator BI = find(BB2->begin(), BB2->end(), Other);
    assert(BI != BB2->end() && "I not in parent basic block!");
    ReplaceInstWithInst(I, BI);
  }
}

//===----------------------------------------------------------------------===//
//
// Visitation methods, these are invoked depending on the type of instruction
// being checked.  They should return true if a common subexpression was folded.
//
//===----------------------------------------------------------------------===//

bool GCSE::visitUnaryOperator(Instruction *I) {
  Value *Op = I->getOperand(0);
  Function *F = I->getParent()->getParent();
  
  for (Value::use_iterator UI = Op->use_begin(), UE = Op->use_end();
       UI != UE; ++UI)
    if (Instruction *Other = dyn_cast<Instruction>(*UI))
      // Check to see if this new binary operator is not I, but same operand...
      if (Other != I && Other->getOpcode() == I->getOpcode() &&
          Other->getOperand(0) == Op &&     // Is the operand the same?
          // Is it embeded in the same function?  (This could be false if LHS
          // is a constant or global!)
          Other->getParent()->getParent() == F &&

          // Check that the types are the same, since this code handles casts...
          Other->getType() == I->getType()) {
        
        // These instructions are identical.  Handle the situation.
        CommonSubExpressionFound(I, Other);
        return true;   // One instruction eliminated!
      }
  
  return false;
}

bool GCSE::visitBinaryOperator(Instruction *I) {
  Value *LHS = I->getOperand(0), *RHS = I->getOperand(1);
  Function *F = I->getParent()->getParent();
  
  for (Value::use_iterator UI = LHS->use_begin(), UE = LHS->use_end();
       UI != UE; ++UI)
    if (Instruction *Other = dyn_cast<Instruction>(*UI))
      // Check to see if this new binary operator is not I, but same operand...
      if (Other != I && Other->getOpcode() == I->getOpcode() &&
          // Are the LHS and RHS the same?
          Other->getOperand(0) == LHS && Other->getOperand(1) == RHS &&
          // Is it embeded in the same function?  (This could be false if LHS
          // is a constant or global!)
          Other->getParent()->getParent() == F) {
        
        // These instructions are identical.  Handle the situation.
        CommonSubExpressionFound(I, Other);
        return true;   // One instruction eliminated!
      }
  
  return false;
}

bool GCSE::visitGetElementPtrInst(GetElementPtrInst *I) {
  Value *Op = I->getOperand(0);
  Function *F = I->getParent()->getParent();
  
  for (Value::use_iterator UI = Op->use_begin(), UE = Op->use_end();
       UI != UE; ++UI)
    if (GetElementPtrInst *Other = dyn_cast<GetElementPtrInst>(*UI))
      // Check to see if this new binary operator is not I, but same operand...
      if (Other != I && Other->getParent()->getParent() == F &&
          Other->getType() == I->getType()) {

        // Check to see that all operators past the 0th are the same...
        unsigned i = 1, e = I->getNumOperands();
        for (; i != e; ++i)
          if (I->getOperand(i) != Other->getOperand(i)) break;
        
        if (i == e) {
          // These instructions are identical.  Handle the situation.
          CommonSubExpressionFound(I, Other);
          return true;   // One instruction eliminated!
        }
      }
  
  return false;
}
