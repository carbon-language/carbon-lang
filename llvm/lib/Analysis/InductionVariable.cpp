//===- InductionVariable.cpp - Induction variable classification ----------===//
//
// This file implements identification and classification of induction 
// variables.  Induction variables must contain a PHI node that exists in a 
// loop header.  Because of this, they are identified an managed by this PHI 
// node.
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
#include "llvm/BasicBlock.h"
#include "llvm/iPHINode.h"
#include "llvm/iOperators.h"
#include "llvm/iTerminators.h"
#include "llvm/Type.h"
#include "llvm/Constants.h"
#include "llvm/Support/CFG.h"
#include "llvm/Assembly/Writer.h"
#include "Support/Debug.h"

static bool isLoopInvariant(const Value *V, const Loop *L) {
  if (const Instruction *I = dyn_cast<Instruction>(V))
    return !L->contains(I->getParent());
  // non-instructions all dominate instructions/blocks
  return true;
}

enum InductionVariable::iType
InductionVariable::Classify(const Value *Start, const Value *Step,
                            const Loop *L) {
  // Check for canonical and simple linear expressions now...
  if (const ConstantInt *CStart = dyn_cast<ConstantInt>(Start))
    if (const ConstantInt *CStep = dyn_cast<ConstantInt>(Step)) {
      if (CStart->isNullValue() && CStep->equalsInt(1))
        return Canonical;
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
InductionVariable::InductionVariable(PHINode *P, LoopInfo *LoopInfo): End(0) {
  InductionType = Unknown;     // Assume the worst
  Phi = P;
  
  // If the PHI node has more than two predecessors, we don't know how to
  // handle it.
  //
  if (Phi->getNumIncomingValues() != 2) return;

  // FIXME: Handle FP induction variables.
  if (Phi->getType() == Type::FloatTy || Phi->getType() == Type::DoubleTy)
    return;

  // If we have loop information, make sure that this PHI node is in the header
  // of a loop...
  //
  const Loop *L = LoopInfo ? LoopInfo->getLoopFor(Phi->getParent()) : 0;
  if (L && L->getHeader() != Phi->getParent())
    return;

  Value *V1 = Phi->getIncomingValue(0);
  Value *V2 = Phi->getIncomingValue(1);

  if (L == 0) {  // No loop information?  Base everything on expression analysis
    ExprType E1 = ClassifyExpression(V1);
    ExprType E2 = ClassifyExpression(V2);

    if (E1.ExprTy > E2.ExprTy)        // Make E1 be the simpler expression
      std::swap(E1, E2);
    
    // E1 must be a constant incoming value, and E2 must be a linear expression
    // with respect to the PHI node.
    //
    if (E1.ExprTy > ExprType::Constant || E2.ExprTy != ExprType::Linear ||
        E2.Var != Phi)
      return;

    // Okay, we have found an induction variable. Save the start and step values
    const Type *ETy = Phi->getType();
    if (isa<PointerType>(ETy)) ETy = Type::ULongTy;

    Start = (Value*)(E1.Offset ? E1.Offset : ConstantInt::get(ETy, 0));
    Step  = (Value*)(E2.Offset ? E2.Offset : ConstantInt::get(ETy, 0));
  } else {
    // Okay, at this point, we know that we have loop information...

    // Make sure that V1 is the incoming value, and V2 is from the backedge of
    // the loop.
    if (L->contains(Phi->getIncomingBlock(0)))     // Wrong order.  Swap now.
      std::swap(V1, V2);
    
    Start = V1;     // We know that Start has to be loop invariant...
    Step = 0;

    if (V2 == Phi) {  // referencing the PHI directly?  Must have zero step
      Step = Constant::getNullValue(Phi->getType());
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
      ExprType StepE = ClassifyExpression(V2);
      if (StepE.ExprTy != ExprType::Linear ||
          StepE.Var != Phi) return;

      const Type *ETy = Phi->getType();
      if (isa<PointerType>(ETy)) ETy = Type::ULongTy;
      Step  = (Value*)(StepE.Offset ? StepE.Offset : ConstantInt::get(ETy, 0));
    } else {   // We were able to get a step value, simplify with expr analysis
      ExprType StepE = ClassifyExpression(Step);
      if (StepE.ExprTy == ExprType::Linear && StepE.Offset == 0) {
        // No offset from variable?  Grab the variable
        Step = StepE.Var;
      } else if (StepE.ExprTy == ExprType::Constant) {
        if (StepE.Offset)
          Step = (Value*)StepE.Offset;
        else
          Step = Constant::getNullValue(Step->getType());
        const Type *ETy = Phi->getType();
        if (isa<PointerType>(ETy)) ETy = Type::ULongTy;
        Step  = (Value*)(StepE.Offset ? StepE.Offset : ConstantInt::get(ETy,0));
      }
    }
  }

  // Classify the induction variable type now...
  InductionType = InductionVariable::Classify(Start, Step, L);
}


Value* InductionVariable::getExecutionCount(LoopInfo *LoopInfo) {
  DEBUG(std::cerr << "entering getExecutionCount\n");

  // Don't recompute if already available
  if (End) {
    DEBUG(std::cerr << "returning cached End value.\n");
    return End;
  }

  const Loop *L = LoopInfo ? LoopInfo->getLoopFor(Phi->getParent()) : 0;
  if (!L) {
    DEBUG(std::cerr << "null loop. oops\n");
    return NULL;
  }

  // >1 backedge => cannot predict number of iterations
  if (Phi->getNumIncomingValues() != 2) {
    DEBUG(std::cerr << ">2 incoming values. oops\n");
    return NULL;
  }

  // Find final node: predecesor of the loop header that's also an exit
  BasicBlock *terminator = 0;
  BasicBlock *header = L->getHeader();
  for (pred_iterator PI = pred_begin(header), PE = pred_end(header);
       PI != PE; ++PI) {
    if (L->isLoopExit(*PI)) {
      terminator = *PI;
      break;
    }
  }

  // Break in the loop => cannot predict number of iterations
  // break: any block which is an exit node whose successor is not in loop,
  // and this block is not marked as the terminator
  //
  const std::vector<BasicBlock*> &blocks = L->getBlocks();
  for (std::vector<BasicBlock*>::const_iterator i = blocks.begin(), e = blocks.end();
       i != e; ++i) {
    if (L->isLoopExit(*i) && (*i != terminator)) {
      for (succ_iterator SI = succ_begin(*i), SE = succ_end(*i); SI != SE; ++SI) {
        if (! L->contains(*SI)) {
          DEBUG(std::cerr << "break found in loop");
          return NULL;
        }
      }
    }
  }

  BranchInst *B = dyn_cast<BranchInst>(terminator->getTerminator());
  if (!B) {
    // this really should not happen
    DEBUG(std::cerr << "no terminator instruction!");
    return NULL; 
  }
  SetCondInst *SCI = dyn_cast<SetCondInst>(B->getCondition());

  if (SCI && InductionType == Canonical) {
    DEBUG(std::cerr << "sci:" << *SCI);
    Value *condVal0 = SCI->getOperand(0);
    Value *condVal1 = SCI->getOperand(1);
    Value *indVar = 0;

    // the induction variable is the one coming from the backedge
    if (L->contains(Phi->getIncomingBlock(0))) {
      indVar = Phi->getIncomingValue(0);
    } else {
      indVar = Phi->getIncomingValue(1);
    }

    // check to see if indVar is one of the parameters in SCI
    // and if the other is loop-invariant, it is the UB
    if (indVar == condVal0) {
      if (isLoopInvariant(condVal1, L)) {
        End = condVal1;
      } else {
        DEBUG(std::cerr << "not loop invariant 1\n");
      }
    } else if (indVar == condVal1) {
      if (isLoopInvariant(condVal0, L)) {
        End = condVal0;
      } else {
        DEBUG(std::cerr << "not loop invariant 0\n");
      }
    }

    if (End) {
      switch (SCI->getOpcode()) {
      case Instruction::SetLT:
      case Instruction::SetNE: break; // already done
      case Instruction::SetLE: {
        // if compared to a constant int N, then predict N+1 iterations
        if (ConstantSInt *ubSigned = dyn_cast<ConstantSInt>(End)) {
          End = ConstantSInt::get(ubSigned->getType(), ubSigned->getValue()+1);
          DEBUG(std::cerr << "signed int constant\n");
        } else if (ConstantUInt *ubUnsigned = dyn_cast<ConstantUInt>(End)) {
          End = ConstantUInt::get(ubUnsigned->getType(),
                                  ubUnsigned->getValue()+1);
          DEBUG(std::cerr << "unsigned int constant\n");
        } else {
          DEBUG(std::cerr << "symbolic bound\n");
          //End = NULL;
          // new expression N+1
          End = BinaryOperator::create(Instruction::Add, End, 
                                       ConstantUInt::get(ubUnsigned->getType(),
                                                         1));
        }
        break;
      }
      default: End = NULL; // cannot predict
      }
    }
    return End;
  } else {
    DEBUG(std::cerr << "SCI null or non-canonical ind var\n");
  }
  return NULL;
}


void InductionVariable::print(std::ostream &o) const {
  switch (InductionType) {
  case InductionVariable::Canonical:    o << "Canonical ";    break;
  case InductionVariable::SimpleLinear: o << "SimpleLinear "; break;
  case InductionVariable::Linear:       o << "Linear ";       break;
  case InductionVariable::Unknown:      o << "Unrecognized "; break;
  }
  o << "Induction Variable: ";
  if (Phi) {
    WriteAsOperand(o, Phi);
    o << ":\n" << Phi;
  } else {
    o << "\n";
  }
  if (InductionType == InductionVariable::Unknown) return;

  o << "  Start = "; WriteAsOperand(o, Start);
  o << "  Step = " ; WriteAsOperand(o, Step);
  if (End) { 
    o << "  End = " ; WriteAsOperand(o, End);
  }
  o << "\n";
}
