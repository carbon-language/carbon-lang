//===- InductionVars.cpp - Induction Variable Cannonicalization code --------=//
//
// This file implements induction variable cannonicalization of loops.
//
// Specifically, after this executes, the following is true:
//   - There is a single induction variable for each loop (at least loops that
//     used to contain at least one induction variable)
//   * This induction variable starts at 0 and steps by 1 per iteration
//   * This induction variable is represented by the first PHI node in the
//     Header block, allowing it to be found easily.
//   - All other preexisting induction variables are adjusted to operate in
//     terms of this primary induction variable
//   - Induction variables with a step size of 0 have been eliminated.
//
// This code assumes the following is true to perform its full job:
//   - The CFG has been simplified to not have multiple entrances into an
//     interval header.  Interval headers should only have two predecessors,
//     one from inside of the loop and one from outside of the loop.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/InductionVars.h"
#include "llvm/Constants.h"
#include "llvm/Analysis/IntervalPartition.h"
#include "llvm/iPHINode.h"
#include "llvm/Function.h"
#include "llvm/BasicBlock.h"
#include "llvm/InstrTypes.h"
#include "llvm/Type.h"
#include "llvm/Support/CFG.h"
#include "Support/STLExtras.h"
#include <algorithm>
#include <iostream>
using std::cerr;

// isLoopInvariant - Return true if the specified value/basic block source is 
// an interval invariant computation.
//
static bool isLoopInvariant(Interval *Int, Value *V) {
  assert(isa<Constant>(V) || isa<Instruction>(V) || isa<Argument>(V));

  if (!isa<Instruction>(V))
    return true;  // Constants and arguments are always loop invariant

  BasicBlock *ValueBlock = cast<Instruction>(V)->getParent();
  assert(ValueBlock && "Instruction not embedded in basic block!");

  // For now, only consider values from outside of the interval, regardless of
  // whether the expression could be lifted out of the loop by some LICM.
  //
  // TODO: invoke LICM library if we find out it would be useful.
  //
  return !Int->contains(ValueBlock);
}


// isLinearInductionVariableH - Return isLIV if the expression V is a linear
// expression defined in terms of loop invariant computations, and a single
// instance of the PHI node PN.  Return isLIC if the expression V is a loop
// invariant computation.  Return isNLIV if the expression is a negated linear
// induction variable.  Return isOther if it is neither.
//
// Currently allowed operators are: ADD, SUB, NEG
// TODO: This should allow casts!
//
enum LIVType { isLIV, isLIC, isNLIV, isOther };
//
// neg - Negate the sign of a LIV expression.
inline LIVType neg(LIVType T) { 
  assert(T == isLIV || T == isNLIV && "Negate Only works on LIV expressions");
  return T == isLIV ? isNLIV : isLIV; 
}
//
static LIVType isLinearInductionVariableH(Interval *Int, Value *V,
					  PHINode *PN) {
  if (V == PN) { return isLIV; }  // PHI node references are (0+PHI)
  if (isLoopInvariant(Int, V)) return isLIC;

  // loop variant computations must be instructions!
  Instruction *I = cast<Instruction>(V);
  switch (I->getOpcode()) {       // Handle each instruction seperately
  case Instruction::Add:
  case Instruction::Sub: {
    Value *SubV1 = cast<BinaryOperator>(I)->getOperand(0);
    Value *SubV2 = cast<BinaryOperator>(I)->getOperand(1);
    LIVType SubLIVType1 = isLinearInductionVariableH(Int, SubV1, PN);
    if (SubLIVType1 == isOther) return isOther;  // Early bailout
    LIVType SubLIVType2 = isLinearInductionVariableH(Int, SubV2, PN);

    switch (SubLIVType2) {
    case isOther: return isOther;      // Unknown subexpression type
    case isLIC:   return SubLIVType1;  // Constant offset, return type #1
    case isLIV:
    case isNLIV:
      // So now we know that we have a linear induction variable on the RHS of
      // the ADD or SUB instruction.  SubLIVType1 cannot be isOther, so it is
      // either a Loop Invariant computation, or a LIV type.
      if (SubLIVType1 == isLIC) {
	// Loop invariant computation, we know this is a LIV then.
	return (I->getOpcode() == Instruction::Add) ? 
	               SubLIVType2 : neg(SubLIVType2);
      }

      // If the LHS is also a LIV Expression, we cannot add two LIVs together
      if (I->getOpcode() == Instruction::Add) return isOther;

      // We can only subtract two LIVs if they are the same type, which yields
      // a LIC, because the LIVs cancel each other out.
      return (SubLIVType1 == SubLIVType2) ? isLIC : isOther;
    }
    // NOT REACHED
  }

  default:            // Any other instruction is not a LINEAR induction var
    return isOther;
  }
}

// isLinearInductionVariable - Return true if the specified expression is a
// "linear induction variable", which is an expression involving a single 
// instance of the PHI node and a loop invariant value that is added or
// subtracted to the PHI node.  This is calculated by walking the SSA graph
//
static inline bool isLinearInductionVariable(Interval *Int, Value *V,
					     PHINode *PN) {
  return isLinearInductionVariableH(Int, V, PN) == isLIV;
}


// isSimpleInductionVar - Return true iff the cannonical induction variable PN
// has an initializer of the constant value 0, and has a step size of constant 
// 1.
static inline bool isSimpleInductionVar(PHINode *PN) {
  assert(PN->getNumIncomingValues() == 2 && "Must have cannonical PHI node!");
  Value *Initializer = PN->getIncomingValue(0);
  if (!isa<Constant>(Initializer)) return false;

  if (Initializer->getType()->isSigned()) {  // Signed constant value...
    if (((ConstantSInt*)Initializer)->getValue() != 0) return false;
  } else if (Initializer->getType()->isUnsigned()) {  // Unsigned constant value
    if (((ConstantUInt*)Initializer)->getValue() != 0) return false;
  } else {
    return false;   // Not signed or unsigned?  Must be FP type or something
  }

  Value *StepExpr = PN->getIncomingValue(1);
  if (!isa<Instruction>(StepExpr) ||
      cast<Instruction>(StepExpr)->getOpcode() != Instruction::Add)
    return false;

  BinaryOperator *I = cast<BinaryOperator>(StepExpr);
  assert(isa<PHINode>(I->getOperand(0)) && 
	 "PHI node should be first operand of ADD instruction!");

  // Get the right hand side of the ADD node.  See if it is a constant 1.
  Value *StepSize = I->getOperand(1);
  if (!isa<Constant>(StepSize)) return false;

  if (StepSize->getType()->isSigned()) {  // Signed constant value...
    if (((ConstantSInt*)StepSize)->getValue() != 1) return false;
  } else if (StepSize->getType()->isUnsigned()) {  // Unsigned constant value
    if (((ConstantUInt*)StepSize)->getValue() != 1) return false;
  } else {
    return false;   // Not signed or unsigned?  Must be FP type or something
  }

  // At this point, we know the initializer is a constant value 0 and the step
  // size is a constant value 1.  This is our simple induction variable!
  return true;
}

// InjectSimpleInductionVariable - Insert a cannonical induction variable into
// the interval header Header.  This assumes that the flow graph is in 
// simplified form (so we know that the header block has exactly 2 predecessors)
//
// TODO: This should inherit the largest type that is being used by the already
// present induction variables (instead of always using uint)
//
static PHINode *InjectSimpleInductionVariable(Interval *Int) {
  std::string PHIName, AddName;

  BasicBlock *Header = Int->getHeaderNode();
  Function *M = Header->getParent();

  if (M->hasSymbolTable()) {
    // Only name the induction variable if the function isn't stripped.
    PHIName = "ind_var";
    AddName = "ind_var_next";
  }

  // Create the neccesary instructions...
  PHINode        *PN      = new PHINode(Type::UIntTy, PHIName);
  Constant       *One     = ConstantUInt::get(Type::UIntTy, 1);
  Constant       *Zero    = ConstantUInt::get(Type::UIntTy, 0);
  BinaryOperator *AddNode = BinaryOperator::create(Instruction::Add, 
						   PN, One, AddName);

  // Figure out which predecessors I have to play with... there should be
  // exactly two... one of which is a loop predecessor, and one of which is not.
  //
  pred_iterator PI = pred_begin(Header);
  assert(PI != pred_end(Header) && "Header node should have 2 preds!");
  BasicBlock *Pred1 = *PI; ++PI;
  assert(PI != pred_end(Header) && "Header node should have 2 preds!");
  BasicBlock *Pred2 = *PI;
  assert(++PI == pred_end(Header) && "Header node should have 2 preds!");

  // Make Pred1 be the loop entrance predecessor, Pred2 be the Loop predecessor
  if (Int->contains(Pred1)) std::swap(Pred1, Pred2);

  assert(!Int->contains(Pred1) && "Pred1 should be loop entrance!");
  assert( Int->contains(Pred2) && "Pred2 should be looping edge!");

  // Link the instructions into the PHI node...
  PN->addIncoming(Zero, Pred1);     // The initializer is first argument
  PN->addIncoming(AddNode, Pred2);  // The step size is second PHI argument
  
  // Insert the PHI node into the Header of the loop.  It shall be the first
  // instruction, because the "Simple" Induction Variable must be first in the
  // block.
  //
  BasicBlock::InstListType &IL = Header->getInstList();
  IL.push_front(PN);

  // Insert the Add instruction as the first (non-phi) instruction in the 
  // header node's basic block.
  BasicBlock::iterator I = IL.begin();
  while (isa<PHINode>(*I)) ++I;
  IL.insert(I, AddNode);
  return PN;
}

// ProcessInterval - This function is invoked once for each interval in the 
// IntervalPartition of the program.  It looks for auxilliary induction
// variables in loops.  If it finds one, it:
// * Cannonicalizes the induction variable.  This consists of:
//   A. Making the first element of the PHI node be the loop invariant 
//      computation, and the second element be the linear induction portion.
//   B. Changing the first element of the linear induction portion of the PHI 
//      node to be of the form ADD(PHI, <loop invariant expr>).
// * Add the induction variable PHI to a list of induction variables found.
//
// After this, a list of cannonical induction variables is known.  This list
// is searched to see if there is an induction variable that counts from 
// constant 0 with a step size of constant 1.  If there is not one, one is
// injected into the loop.  Thus a "simple" induction variable is always known
//
// One a simple induction variable is known, all other induction variables are
// modified to refer to the "simple" induction variable.
//
static bool ProcessInterval(Interval *Int) {
  if (!Int->isLoop()) return false;  // Not a loop?  Ignore it!

  std::vector<PHINode *> InductionVars;

  BasicBlock *Header = Int->getHeaderNode();
  // Loop over all of the PHI nodes in the interval header...
  for (BasicBlock::iterator I = Header->begin(), E = Header->end(); 
       I != E && isa<PHINode>(*I); ++I) {
    PHINode *PN = cast<PHINode>(*I);
    if (PN->getNumIncomingValues() != 2) { // These should be eliminated by now.
      cerr << "Found interval header with more than 2 predecessors! Ignoring\n";
      return false;    // Todo, make an assertion.
    }

    // For this to be an induction variable, one of the arguments must be a
    // loop invariant expression, and the other must be an expression involving
    // the PHI node, along with possible additions and subtractions of loop
    // invariant values.
    //
    BasicBlock *BB1 = PN->getIncomingBlock(0);
    Value      *V1  = PN->getIncomingValue(0);
    BasicBlock *BB2 = PN->getIncomingBlock(1);
    Value      *V2  = PN->getIncomingValue(1);

    // Figure out which computation is loop invariant...
    if (!isLoopInvariant(Int, V1)) {
      // V1 is *not* loop invariant.  Check to see if V2 is:
      if (isLoopInvariant(Int, V2)) {
	// They *are* loop invariant.  Exchange BB1/BB2 and V1/V2 so that
	// V1 is always the loop invariant computation.
	std::swap(V1, V2); std::swap(BB1, BB2);
      } else {
	// Neither value is loop invariant.  Must not be an induction variable.
	// This case can happen if there is an unreachable loop in the CFG that
	// has two tail loops in it that was not split by the cleanup phase
	// before.
	continue;
      }      
    }

    // At this point, we know that BB1/V1 are loop invariant.  We don't know
    // anything about BB2/V2.  Check now to see if V2 is a linear induction
    // variable.
    //
    cerr << "Found loop invariant computation: " << V1 << "\n";
    
    if (!isLinearInductionVariable(Int, V2, PN))
      continue;         // No, it is not a linear ind var, ignore the PHI node.
    cerr << "Found linear induction variable: " << V2;

    // TODO: Cannonicalize V2

    // Add this PHI node to the list of induction variables found...
    InductionVars.push_back(PN);    
  }

  // No induction variables found?
  if (InductionVars.empty()) return false;

  // Search to see if there is already a "simple" induction variable.
  std::vector<PHINode*>::iterator It = 
    find_if(InductionVars.begin(), InductionVars.end(), isSimpleInductionVar);
  
  PHINode *PrimaryIndVar;

  // A simple induction variable was not found, inject one now...
  if (It == InductionVars.end()) {
    PrimaryIndVar = InjectSimpleInductionVariable(Int);
  } else {
    // Move the PHI node for this induction variable to the start of the PHI
    // list in HeaderNode... we do not need to do this for the inserted case
    // because the inserted node will always be placed at the beginning of
    // HeaderNode.
    //
    PrimaryIndVar = *It;
    BasicBlock::iterator i =
      find(Header->begin(), Header->end(), PrimaryIndVar);
    assert(i != Header->end() && 
	   "How could Primary IndVar not be in the header!?!!?");

    if (i != Header->begin())
      std::iter_swap(i, Header->begin());
  }

  // Now we know that there is a simple induction variable PrimaryIndVar.
  // Simplify all of the other induction variables to use this induction 
  // variable as their counter, and destroy the PHI nodes that correspond to
  // the old indvars.
  //
  // TODO


  cerr << "Found Interval Header with indvars (primary indvar should be first "
       << "phi): \n" << Header << "\nPrimaryIndVar: " << PrimaryIndVar;

  return false;  // TODO: true;
}


// ProcessIntervalPartition - This function loops over the interval partition
// processing each interval with ProcessInterval
//
static bool ProcessIntervalPartition(IntervalPartition &IP) {
  // This currently just prints out information about the interval structure
  // of the function...
#if 0
  static unsigned N = 0;
  cerr << "\n***********Interval Partition #" << (++N) << "************\n\n";
  copy(IP.begin(), IP.end(), ostream_iterator<Interval*>(cerr, "\n"));

  cerr << "\n*********** PERFORMING WORK ************\n\n";
#endif
  // Loop over all of the intervals in the partition and look for induction
  // variables in intervals that represent loops.
  //
  return reduce_apply(IP.begin(), IP.end(), bitwise_or<bool>(), false,
		      std::ptr_fun(ProcessInterval));
}

// DoInductionVariableCannonicalize - Simplify induction variables in loops.
// This function loops over an interval partition of a program, reducing it
// until the graph is gone.
//
bool InductionVariableCannonicalize::doIt(Function *M, IntervalPartition &IP) {
                                          
  bool Changed = false;

#if 0
  while (!IP->isDegeneratePartition()) {
    Changed |= ProcessIntervalPartition(*IP);

    // Calculate the reduced version of this graph until we get to an 
    // irreducible graph or a degenerate graph...
    //
    IntervalPartition *NewIP = new IntervalPartition(*IP, false);
    if (NewIP->size() == IP->size()) {
      cerr << "IRREDUCIBLE GRAPH FOUND!!!\n";
      return Changed;
    }
    delete IP;
    IP = NewIP;
  }

  delete IP;
#endif
  return Changed;
}


bool InductionVariableCannonicalize::runOnFunction(Function *F) {
  return doIt(F, getAnalysis<IntervalPartition>());
}

// getAnalysisUsage - This function works on the call graph of a module.
// It is capable of updating the call graph to reflect the new state of the
// module.
//
void InductionVariableCannonicalize::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired(IntervalPartition::ID);
}
