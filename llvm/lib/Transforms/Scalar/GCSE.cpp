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

#include "llvm/Transforms/Scalar.h"
#include "llvm/InstrTypes.h"
#include "llvm/iMemory.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/CFG.h"
#include "Support/StatisticReporter.h"
#include <algorithm>
using std::set;
using std::map;


static Statistic<> NumInstRemoved("gcse\t\t- Number of instructions removed");
static Statistic<> NumLoadRemoved("gcse\t\t- Number of loads removed");

namespace {
  class GCSE : public FunctionPass, public InstVisitor<GCSE, bool> {
    set<Instruction*>       WorkList;
    DominatorSet           *DomSetInfo;
    ImmediateDominators    *ImmDominator;

    // BBContainsStore - Contains a value that indicates whether a basic block
    // has a store or call instruction in it.  This map is demand populated, so
    // not having an entry means that the basic block has not been scanned yet.
    //
    map<BasicBlock*, bool>  BBContainsStore;
  public:
    virtual bool runOnFunction(Function &F);

    // Visitation methods, these are invoked depending on the type of
    // instruction being checked.  They should return true if a common
    // subexpression was folded.
    //
    bool visitUnaryOperator(Instruction &I);
    bool visitBinaryOperator(Instruction &I);
    bool visitGetElementPtrInst(GetElementPtrInst &I);
    bool visitCastInst(CastInst &I){return visitUnaryOperator((Instruction&)I);}
    bool visitShiftInst(ShiftInst &I) {
      return visitBinaryOperator((Instruction&)I);
    }
    bool visitLoadInst(LoadInst &LI);
    bool visitInstruction(Instruction &) { return false; }

  private:
    void ReplaceInstWithInst(Instruction *First, BasicBlock::iterator SI);
    void CommonSubExpressionFound(Instruction *I, Instruction *Other);

    // TryToRemoveALoad - Try to remove one of L1 or L2.  The problem with
    // removing loads is that intervening stores might make otherwise identical
    // load's yield different values.  To ensure that this is not the case, we
    // check that there are no intervening stores or calls between the
    // instructions.
    //
    bool TryToRemoveALoad(LoadInst *L1, LoadInst *L2);

    // CheckForInvalidatingInst - Return true if BB or any of the predecessors
    // of BB (until DestBB) contain a store (or other invalidating) instruction.
    //
    bool CheckForInvalidatingInst(BasicBlock *BB, BasicBlock *DestBB,
                                  set<BasicBlock*> &VisitedSet);

    // This transformation requires dominator and immediate dominator info
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.preservesCFG();
      AU.addRequired(DominatorSet::ID);
      AU.addRequired(ImmediateDominators::ID); 
    }
  };

  RegisterOpt<GCSE> X("gcse", "Global Common Subexpression Elimination");
}

// createGCSEPass - The public interface to this file...
Pass *createGCSEPass() { return new GCSE(); }


// GCSE::runOnFunction - This is the main transformation entry point for a
// function.
//
bool GCSE::runOnFunction(Function &F) {
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
    Instruction &I = **WorkList.begin(); // Get an instruction from the worklist
    WorkList.erase(WorkList.begin());

    // Visit the instruction, dispatching to the correct visit function based on
    // the instruction type.  This does the checking.
    //
    Changed |= visit(I);
  }

  // Clear out data structure so that next function starts fresh
  BBContainsStore.clear();
  
  // When the worklist is empty, return whether or not we changed anything...
  return Changed;
}


// ReplaceInstWithInst - Destroy the instruction pointed to by SI, making all
// uses of the instruction use First now instead.
//
void GCSE::ReplaceInstWithInst(Instruction *First, BasicBlock::iterator SI) {
  Instruction &Second = *SI;
  
  //cerr << "DEL " << (void*)Second << Second;

  // Add the first instruction back to the worklist
  WorkList.insert(First);

  // Add all uses of the second instruction to the worklist
  for (Value::use_iterator UI = Second.use_begin(), UE = Second.use_end();
       UI != UE; ++UI)
    WorkList.insert(cast<Instruction>(*UI));
    
  // Make all users of 'Second' now use 'First'
  Second.replaceAllUsesWith(First);

  // Erase the second instruction from the program
  Second.getParent()->getInstList().erase(SI);
}

// CommonSubExpressionFound - The two instruction I & Other have been found to
// be common subexpressions.  This function is responsible for eliminating one
// of them, and for fixing the worklist to be correct.
//
void GCSE::CommonSubExpressionFound(Instruction *I, Instruction *Other) {
  assert(I != Other);

  WorkList.erase(I);
  WorkList.erase(Other); // Other may not actually be on the worklist anymore...

  ++NumInstRemoved;   // Keep track of number of instructions eliminated

  // Handle the easy case, where both instructions are in the same basic block
  BasicBlock *BB1 = I->getParent(), *BB2 = Other->getParent();
  if (BB1 == BB2) {
    // Eliminate the second occuring instruction.  Add all uses of the second
    // instruction to the worklist.
    //
    // Scan the basic block looking for the "first" instruction
    BasicBlock::iterator BI = BB1->begin();
    while (&*BI != I && &*BI != Other) {
      ++BI;
      assert(BI != BB1->end() && "Instructions not found in parent BB!");
    }

    // Keep track of which instructions occurred first & second
    Instruction *First = BI;
    Instruction *Second = I != First ? I : Other; // Get iterator to second inst
    BI = Second;

    // Destroy Second, using First instead.
    ReplaceInstWithInst(First, BI);    

    // Otherwise, the two instructions are in different basic blocks.  If one
    // dominates the other instruction, we can simply use it
    //
  } else if (DomSetInfo->dominates(BB1, BB2)) {    // I dom Other?
    ReplaceInstWithInst(I, Other);
  } else if (DomSetInfo->dominates(BB2, BB1)) {    // Other dom I?
    ReplaceInstWithInst(Other, I);
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
    SharedDom->getInstList().insert(--SharedDom->end(), I);

    // Eliminate 'Other' now.
    ReplaceInstWithInst(I, Other);
  }
}

//===----------------------------------------------------------------------===//
//
// Visitation methods, these are invoked depending on the type of instruction
// being checked.  They should return true if a common subexpression was folded.
//
//===----------------------------------------------------------------------===//

bool GCSE::visitUnaryOperator(Instruction &I) {
  Value *Op = I.getOperand(0);
  Function *F = I.getParent()->getParent();
  
  for (Value::use_iterator UI = Op->use_begin(), UE = Op->use_end();
       UI != UE; ++UI)
    if (Instruction *Other = dyn_cast<Instruction>(*UI))
      // Check to see if this new binary operator is not I, but same operand...
      if (Other != &I && Other->getOpcode() == I.getOpcode() &&
          Other->getOperand(0) == Op &&     // Is the operand the same?
          // Is it embeded in the same function?  (This could be false if LHS
          // is a constant or global!)
          Other->getParent()->getParent() == F &&

          // Check that the types are the same, since this code handles casts...
          Other->getType() == I.getType()) {
        
        // These instructions are identical.  Handle the situation.
        CommonSubExpressionFound(&I, Other);
        return true;   // One instruction eliminated!
      }
  
  return false;
}

// isIdenticalBinaryInst - Return true if the two binary instructions are
// identical.
//
static inline bool isIdenticalBinaryInst(const Instruction &I1,
                                         const Instruction *I2) {
  // Is it embeded in the same function?  (This could be false if LHS
  // is a constant or global!)
  if (I1.getOpcode() != I2->getOpcode() ||
      I1.getParent()->getParent() != I2->getParent()->getParent())
    return false;
  
  // They are identical if both operands are the same!
  if (I1.getOperand(0) == I2->getOperand(0) &&
      I1.getOperand(1) == I2->getOperand(1))
    return true;
  
  // If the instruction is commutative and associative, the instruction can
  // match if the operands are swapped!
  //
  if ((I1.getOperand(0) == I2->getOperand(1) &&
       I1.getOperand(1) == I2->getOperand(0)) &&
      (I1.getOpcode() == Instruction::Add || 
       I1.getOpcode() == Instruction::Mul ||
       I1.getOpcode() == Instruction::And || 
       I1.getOpcode() == Instruction::Or  ||
       I1.getOpcode() == Instruction::Xor))
    return true;

  return false;
}

bool GCSE::visitBinaryOperator(Instruction &I) {
  Value *LHS = I.getOperand(0), *RHS = I.getOperand(1);
  Function *F = I.getParent()->getParent();
  
  for (Value::use_iterator UI = LHS->use_begin(), UE = LHS->use_end();
       UI != UE; ++UI)
    if (Instruction *Other = dyn_cast<Instruction>(*UI))
      // Check to see if this new binary operator is not I, but same operand...
      if (Other != &I && isIdenticalBinaryInst(I, Other)) {        
        // These instructions are identical.  Handle the situation.
        CommonSubExpressionFound(&I, Other);
        return true;   // One instruction eliminated!
      }
  
  return false;
}

// IdenticalComplexInst - Return true if the two instructions are the same, by
// using a brute force comparison.
//
static bool IdenticalComplexInst(const Instruction *I1, const Instruction *I2) {
  assert(I1->getOpcode() == I2->getOpcode());
  // Equal if they are in the same function...
  return I1->getParent()->getParent() == I2->getParent()->getParent() &&
    // And return the same type...
    I1->getType() == I2->getType() &&
    // And have the same number of operands...
    I1->getNumOperands() == I2->getNumOperands() &&
    // And all of the operands are equal.
    std::equal(I1->op_begin(), I1->op_end(), I2->op_begin());
}

bool GCSE::visitGetElementPtrInst(GetElementPtrInst &I) {
  Value *Op = I.getOperand(0);
  Function *F = I.getParent()->getParent();
  
  for (Value::use_iterator UI = Op->use_begin(), UE = Op->use_end();
       UI != UE; ++UI)
    if (GetElementPtrInst *Other = dyn_cast<GetElementPtrInst>(*UI))
      // Check to see if this new getelementptr is not I, but same operand...
      if (Other != &I && IdenticalComplexInst(&I, Other)) {
        // These instructions are identical.  Handle the situation.
        CommonSubExpressionFound(&I, Other);
        return true;   // One instruction eliminated!
      }
  
  return false;
}

bool GCSE::visitLoadInst(LoadInst &LI) {
  Value *Op = LI.getOperand(0);
  Function *F = LI.getParent()->getParent();
  
  for (Value::use_iterator UI = Op->use_begin(), UE = Op->use_end();
       UI != UE; ++UI)
    if (LoadInst *Other = dyn_cast<LoadInst>(*UI))
      // Check to see if this new load is not LI, but has the same operands...
      if (Other != &LI && IdenticalComplexInst(&LI, Other) &&
          TryToRemoveALoad(&LI, Other))
        return true;   // An instruction was eliminated!
  
  return false;
}

static inline bool isInvalidatingInst(const Instruction &I) {
  return I.getOpcode() == Instruction::Store ||
         I.getOpcode() == Instruction::Call ||
         I.getOpcode() == Instruction::Invoke;
}

// TryToRemoveALoad - Try to remove one of L1 or L2.  The problem with removing
// loads is that intervening stores might make otherwise identical load's yield
// different values.  To ensure that this is not the case, we check that there
// are no intervening stores or calls between the instructions.
//
bool GCSE::TryToRemoveALoad(LoadInst *L1, LoadInst *L2) {
  // Figure out which load dominates the other one.  If neither dominates the
  // other we cannot eliminate one...
  //
  if (DomSetInfo->dominates(L2, L1)) 
    std::swap(L1, L2);   // Make L1 dominate L2
  else if (!DomSetInfo->dominates(L1, L2))
    return false;  // Neither instruction dominates the other one...

  BasicBlock *BB1 = L1->getParent(), *BB2 = L2->getParent();

  BasicBlock::iterator L1I = L1;

  // L1 now dominates L2.  Check to see if the intervening instructions between
  // the two loads include a store or call...
  //
  if (BB1 == BB2) {  // In same basic block?
    // In this degenerate case, no checking of global basic blocks has to occur
    // just check the instructions BETWEEN L1 & L2...
    //
    for (++L1I; &*L1I != L2; ++L1I)
      if (isInvalidatingInst(*L1I))
        return false;   // Cannot eliminate load

    ++NumLoadRemoved;
    CommonSubExpressionFound(L1, L2);
    return true;
  } else {
    // Make sure that there are no store instructions between L1 and the end of
    // it's basic block...
    //
    for (++L1I; L1I != BB1->end(); ++L1I)
      if (isInvalidatingInst(*L1I)) {
        BBContainsStore[BB1] = true;
        return false;   // Cannot eliminate load
      }

    // Make sure that there are no store instructions between the start of BB2
    // and the second load instruction...
    //
    for (BasicBlock::iterator II = BB2->begin(); &*II != L2; ++II)
      if (isInvalidatingInst(*II)) {
        BBContainsStore[BB2] = true;
        return false;   // Cannot eliminate load
      }

    // Do a depth first traversal of the inverse CFG starting at L2's block,
    // looking for L1's block.  The inverse CFG is made up of the predecessor
    // nodes of a block... so all of the edges in the graph are "backward".
    //
    set<BasicBlock*> VisitedSet;
    for (pred_iterator PI = pred_begin(BB2), PE = pred_end(BB2); PI != PE; ++PI)
      if (CheckForInvalidatingInst(*PI, BB1, VisitedSet))
        return false;
    
    ++NumLoadRemoved;
    CommonSubExpressionFound(L1, L2);
    return true;
  }
  return false;
}

// CheckForInvalidatingInst - Return true if BB or any of the predecessors of BB
// (until DestBB) contain a store (or other invalidating) instruction.
//
bool GCSE::CheckForInvalidatingInst(BasicBlock *BB, BasicBlock *DestBB,
                                    set<BasicBlock*> &VisitedSet) {
  // Found the termination point!
  if (BB == DestBB || VisitedSet.count(BB)) return false;

  // Avoid infinite recursion!
  VisitedSet.insert(BB);

  // Have we already checked this block?
  map<BasicBlock*, bool>::iterator MI = BBContainsStore.find(BB);
  
  if (MI != BBContainsStore.end()) {
    // If this block is known to contain a store, exit the recursion early...
    if (MI->second) return true;
    // Otherwise continue checking predecessors...
  } else {
    // We don't know if this basic block contains an invalidating instruction.
    // Check now:
    bool HasStore = std::find_if(BB->begin(), BB->end(),
                                 isInvalidatingInst) != BB->end();
    if ((BBContainsStore[BB] = HasStore))   // Update map
      return true;   // Exit recursion early...
  }

  // Check all of our predecessor blocks...
  for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB); PI != PE; ++PI)
    if (CheckForInvalidatingInst(*PI, DestBB, VisitedSet))
      return true;

  // None of our predecessor blocks contain a store, and we don't either!
  return false;
}
