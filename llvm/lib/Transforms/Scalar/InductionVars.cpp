//===- InductionVars.cpp - Induction Variable Cannonicalization code --------=//
//
// This file implements induction variable cannonicalization of loops.
//
// Specifically, after this executes, the following is true:
//   - There is a single induction variable for each loop (at least loops that
//     used to contain at least one induction variable)
//   - This induction variable starts at 0 and steps by 1 per iteration
//   - This induction variable is represented by the first PHI node in the
//     Header block, allowing it to be found easily.
//   - All other preexisting induction variables are adjusted to operate in
//     terms of this primary induction variable
//
// This code assumes the following is true to perform its full job:
//   - The CFG has been simplified to not have multiple entrances into an
//     interval header.  Interval headers should only have two predecessors,
//     one from inside of the loop and one from outside of the loop.
//
//===----------------------------------------------------------------------===//

#include "llvm/ConstPoolVals.h"
#include "llvm/Analysis/IntervalPartition.h"
#include "llvm/Opt/AllOpts.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Tools/STLExtras.h"
#include "llvm/iOther.h"
#include <algorithm>

// isLoopInvariant - Return true if the specified value/basic block source is 
// an interval invariant computation.
//
static bool isLoopInvariant(cfg::Interval *Int, Value *V) {
  assert(V->getValueType() == Value::ConstantVal ||
	 V->getValueType() == Value::InstructionVal ||
	 V->getValueType() == Value::MethodArgumentVal);

  if (V->getValueType() != Value::InstructionVal)
    return true;  // Constants and arguments are always loop invariant

  BasicBlock *ValueBlock = ((Instruction*)V)->getParent();
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
static LIVType isLinearInductionVariableH(cfg::Interval *Int, Value *V,
					  PHINode *PN) {
  if (V == PN) { return isLIV; }  // PHI node references are (0+PHI)
  if (isLoopInvariant(Int, V)) return isLIC;

  assert(V->getValueType() == Value::InstructionVal &&
	 "loop noninvariant computations must be instructions!");

  Instruction *I = (Instruction*)V;
  switch (I->getInstType()) {       // Handle each instruction seperately
  case Instruction::Neg: {
    Value *SubV = ((UnaryOperator*)I)->getOperand(0);
    LIVType SubLIVType = isLinearInductionVariableH(Int, SubV, PN);
    switch (SubLIVType) {
    case isLIC:          // Loop invariant & other computations remain the same
    case isOther: return SubLIVType;
    case isLIV:          // Return the opposite signed LIV type
    case isNLIV:  return neg(isLIV);
    }
  }
  case Instruction::Add:
  case Instruction::Sub: {
    Value *SubV1 = ((BinaryOperator*)I)->getOperand(0);
    Value *SubV2 = ((BinaryOperator*)I)->getOperand(1);
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
	return (I->getInstType() == Instruction::Add) ? 
	               SubLIVType2 : neg(SubLIVType2);
      }

      // If the LHS is also a LIV Expression, we cannot add two LIVs together
      if (I->getInstType() == Instruction::Add) return isOther;

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
static inline bool isLinearInductionVariable(cfg::Interval *Int, Value *V,
					     PHINode *PN) {
  return isLinearInductionVariableH(Int, V, PN) == isLIV;
}


// isSimpleInductionVar - Return true iff the cannonical induction variable PN
// has an initializer of the constant value 0, and has a step size of constant 
// 1.
static inline bool isSimpleInductionVar(PHINode *PN) {
  assert(PN->getNumIncomingValues() == 2 && "Must have cannonical PHI node!");
  Value *Initializer = PN->getIncomingValue(0);
  if (Initializer->getValueType() != Value::ConstantVal)
    return false;

  if (Initializer->getType()->isSigned()) {  // Signed constant value...
    if (((ConstPoolSInt*)Initializer)->getValue() != 0) return false;
  } else if (Initializer->getType()->isUnsigned()) {  // Unsigned constant value
    if (((ConstPoolUInt*)Initializer)->getValue() != 0) return false;
  } else {
    return false;   // Not signed or unsigned?  Must be FP type or something
  }

  // How do I check for 0 for any integral value?  Use 
  // ConstPoolVal::getNullConstant?

  Value *StepExpr    = PN->getIncomingValue(1);
  assert(StepExpr->getValueType() == Value::InstructionVal && "No ADD node?");
  assert(((Instruction*)StepExpr)->getInstType() == Instruction::Add &&
	 "No ADD node? Not a cannonical PHI!");
  BinaryOperator *I = (BinaryOperator*)StepExpr;
  assert(I->getOperand(0)->getValueType() == Value::InstructionVal && 
      ((Instruction*)I->getOperand(0))->getInstType() == Instruction::PHINode &&
	 "PHI node should be first operand of ADD instruction!");

  // Get the right hand side of the ADD node.  See if it is a constant 1.
  Value *StepSize = I->getOperand(1);
  if (StepSize->getValueType() != Value::ConstantVal) return false;

  if (StepSize->getType()->isSigned()) {  // Signed constant value...
    if (((ConstPoolSInt*)StepSize)->getValue() != 1) return false;
  } else if (StepSize->getType()->isUnsigned()) {  // Unsigned constant value
    if (((ConstPoolUInt*)StepSize)->getValue() != 1) return false;
  } else {
    return false;   // Not signed or unsigned?  Must be FP type or something
  }

  // At this point, we know the initializer is a constant value 0 and the step
  // size is a constant value 1.  This is our simple induction variable!
  return true;
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
static bool ProcessInterval(cfg::Interval *Int) {
  if (!Int->isLoop()) return false;  // Not a loop?  Ignore it!

  vector<PHINode *> InductionVars;

  BasicBlock *Header = Int->getHeaderNode();
  // Loop over all of the PHI nodes in the interval header...
  for (BasicBlock::InstListType::iterator I = Header->getInstList().begin(),
	 E = Header->getInstList().end(); 
       I != E && (*I)->getInstType() == Instruction::PHINode; ++I) {

    PHINode *PN = (PHINode*)*I;
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
	swap(V1, V2); swap(BB1, BB2);
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
    cerr << "Found loop invariant computation: " << V1;
    
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
  vector<PHINode*>::iterator It = 
    find_if(InductionVars.begin(), InductionVars.end(), isSimpleInductionVar);
  
  PHINode *PrimaryIndVar;

  // A simple induction variable was not found, inject one now...
  if (It == InductionVars.end()) {
    cerr << "WARNING, Induction variable injection not implemented yet!\n";
    // TODO: Inject induction variable
    PrimaryIndVar = 0; // Point it at the new indvar
  } else {
    // Move the PHI node for this induction variable to the start of the PHI
    // list in HeaderNode... we do not need to do this for the inserted case
    // because the inserted node will always be placed at the beginning of
    // HeaderNode.
    //
    PrimaryIndVar = *It;
    BasicBlock::InstListType::iterator i = 
      find(Header->getInstList().begin(), Header->getInstList().end(),
	   PrimaryIndVar);
    assert(i != Header->getInstList().end() && 
	   "How could Primary IndVar not be in the header!?!!?");

    if (i != Header->getInstList().begin())
      iter_swap(i, Header->getInstList().begin());
  }

  // Now we know that there is a simple induction variable PrimaryIndVar.
  // Simplify all of the other induction variables to use this induction 
  // variable as their counter, and destroy the PHI nodes that correspond to
  // the old indvars.
  //
  // TODO


  cerr << "Found Interval Header with indvars (primary indvar should be first "
       << "phi): \n" << Header << "\nPrimaryIndVar = " << PrimaryIndVar;

  return false;  // TODO: true;
}


// ProcessIntervalPartition - This function loops over the interval partition
// processing each interval with ProcessInterval
//
static bool ProcessIntervalPartition(cfg::IntervalPartition &IP) {
  // This currently just prints out information about the interval structure
  // of the method...
  static unsigned N = 0;
  cerr << "\n***********Interval Partition #" << (++N) << "************\n\n";
  copy(IP.begin(), IP.end(), ostream_iterator<cfg::Interval*>(cerr, "\n"));

  cerr << "\n*********** PERFORMING WORK ************\n\n";

  // Loop over all of the intervals in the partition and look for induction
  // variables in intervals that represent loops.
  //
  return reduce_apply(IP.begin(), IP.end(), bitwise_or<bool>(), false,
		      ptr_fun(ProcessInterval));
}


// DoInductionVariableCannonicalize - Simplify induction variables in loops.
// This function loops over an interval partition of a program, reducing it
// until the graph is gone.
//
bool DoInductionVariableCannonicalize(Method *M) {
  cfg::IntervalPartition *IP = new cfg::IntervalPartition(M);
  bool Changed = false;

  while (!IP->isDegeneratePartition()) {
    Changed |= ProcessIntervalPartition(*IP);

    // Calculate the reduced version of this graph until we get to an 
    // irreducible graph or a degenerate graph...
    //
    cfg::IntervalPartition *NewIP = new cfg::IntervalPartition(*IP, false);
    if (NewIP->size() == IP->size()) {
      cerr << "IRREDUCIBLE GRAPH FOUND!!!\n";
      return Changed;
    }
    delete IP;
    IP = NewIP;
  }

  delete IP;
  return Changed;
}
