//===- llvm/Analysis/InductionVariable.h - Induction variable ----*- C++ -*--=//
//
// This interface is used to identify and classify induction variables that
// exist in the program.  Induction variables must contain a PHI node that
// exists in a loop header.  Because of this, they are identified an managed by
// this PHI node.
//
// Induction variables are classified into a type.  Knowing that an induction
// variable is of a specific type can constrain the values of the start and
// step.  For example, a SimpleLinear induction variable must have a start and
// step values that are constants.
//
// Induction variables can be created with or without loop information.  If no
// loop information is available, induction variables cannot be recognized to be
// more than SimpleLinear variables.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/InductionVariable.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/Expressions.h"
#include "llvm/iPHINode.h"
#include "llvm/InstrTypes.h"
#include "llvm/Type.h"
#include "llvm/ConstantVals.h"

using analysis::ExprType;


static bool isLoopInvariant(const Value *V, const cfg::Loop *L) {
  if (isa<Constant>(V) || isa<MethodArgument>(V) || isa<GlobalValue>(V))
    return true;
  
  const Instruction *I = cast<Instruction>(V);
  const BasicBlock *BB = I->getParent();

  return !L->contains(BB);
}

enum InductionVariable::iType
InductionVariable::Classify(const Value *Start, const Value *Step,
			    const cfg::Loop *L = 0) {
  // Check for cannonical and simple linear expressions now...
  if (ConstantInt *CStart = dyn_cast<ConstantInt>(Start))
    if (ConstantInt *CStep = dyn_cast<ConstantInt>(Step)) {
      if (CStart->equalsInt(0) && CStep->equalsInt(1))
	return Cannonical;
      else
	return SimpleLinear;
    }

  // Without loop information, we cannot do any better, so bail now...
  if (L == 0) return Unknown;

  if (isLoopInvariant(Start, L) && isLoopInvariant(Step, L))
    return Linear;
  return Unknown;
}

// Create an induction variable for the specified value.  If it is a PHI, and
// if it's recognizable, classify it and fill in instance variables.
//
InductionVariable::InductionVariable(PHINode *P, cfg::LoopInfo *LoopInfo) {
  InductionType = Unknown;     // Assume the worst
  Phi = P;
  
  // If the PHI node has more than two predecessors, we don't know how to
  // handle it.
  //
  if (Phi->getNumIncomingValues() != 2) return;

  // If we have loop information, make sure that this PHI node is in the header
  // of a loop...
  //
  const cfg::Loop *L = LoopInfo ? LoopInfo->getLoopFor(Phi->getParent()) : 0;
  if (L && L->getHeader() != Phi->getParent())
    return;

  Value *V1 = Phi->getIncomingValue(0);
  Value *V2 = Phi->getIncomingValue(1);

  if (L == 0) {  // No loop information?  Base everything on expression analysis
    ExprType E1 = analysis::ClassifyExpression(V1);
    ExprType E2 = analysis::ClassifyExpression(V2);

    if (E1.ExprTy > E2.ExprTy)        // Make E1 be the simpler expression
      swap(E1, E2);
    
    // E1 must be a constant incoming value, and E2 must be a linear expression
    // with respect to the PHI node.
    //
    if (E1.ExprTy > ExprType::Constant || E2.ExprTy != ExprType::Linear ||
	E2.Var != Phi)
      return;

    // Okay, we have found an induction variable. Save the start and step values
    const Type *ETy = Phi->getType();
    if (ETy->isPointerType()) ETy = Type::ULongTy;

    Start = (Value*)(E1.Offset ? E1.Offset : ConstantInt::get(ETy, 0));
    Step  = (Value*)(E2.Offset ? E2.Offset : ConstantInt::get(ETy, 0));
  } else {
    // Okay, at this point, we know that we have loop information...

    // Make sure that V1 is the incoming value, and V2 is from the backedge of
    // the loop.
    if (L->contains(Phi->getIncomingBlock(0)))     // Wrong order.  Swap now.
      swap(V1, V2);
    
    Start = V1;     // We know that Start has to be loop invariant...
    Step = 0;

    if (V2 == Phi) {  // referencing the PHI directly?  Must have zero step
      Step = Constant::getNullConstant(Phi->getType());
    } else if (BinaryOperator *I = dyn_cast<BinaryOperator>(V2)) {
      // TODO: This could be much better...
      if (I->getOpcode() == Instruction::Add) {
	if (I->getOperand(0) == Phi)
	  Step = I->getOperand(1);
	else if (I->getOperand(1) == Phi)
	  Step = I->getOperand(0);
      }
    }

    if (Step == 0) {                  // Unrecognized step value...
      ExprType StepE = analysis::ClassifyExpression(V2);
      if (StepE.ExprTy != ExprType::Linear ||
	  StepE.Var != Phi) return;

      const Type *ETy = Phi->getType();
      if (ETy->isPointerType()) ETy = Type::ULongTy;
      Step  = (Value*)(StepE.Offset ? StepE.Offset : ConstantInt::get(ETy, 0));
    }
  }

  // Classify the induction variable type now...
  InductionType = InductionVariable::Classify(Start, Step, L);
}
