//===- InstCombineSelect.cpp ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the visitSelect function.
//
//===----------------------------------------------------------------------===//

#include "InstCombine.h"
#include "llvm/Support/PatternMatch.h"
#include "llvm/Analysis/InstructionSimplify.h"
using namespace llvm;
using namespace PatternMatch;

/// MatchSelectPattern - Pattern match integer [SU]MIN, [SU]MAX, and ABS idioms,
/// returning the kind and providing the out parameter results if we
/// successfully match.
static SelectPatternFlavor
MatchSelectPattern(Value *V, Value *&LHS, Value *&RHS) {
  SelectInst *SI = dyn_cast<SelectInst>(V);
  if (SI == 0) return SPF_UNKNOWN;
  
  ICmpInst *ICI = dyn_cast<ICmpInst>(SI->getCondition());
  if (ICI == 0) return SPF_UNKNOWN;
  
  LHS = ICI->getOperand(0);
  RHS = ICI->getOperand(1);
  
  // (icmp X, Y) ? X : Y 
  if (SI->getTrueValue() == ICI->getOperand(0) &&
      SI->getFalseValue() == ICI->getOperand(1)) {
    switch (ICI->getPredicate()) {
    default: return SPF_UNKNOWN; // Equality.
    case ICmpInst::ICMP_UGT:
    case ICmpInst::ICMP_UGE: return SPF_UMAX;
    case ICmpInst::ICMP_SGT:
    case ICmpInst::ICMP_SGE: return SPF_SMAX;
    case ICmpInst::ICMP_ULT:
    case ICmpInst::ICMP_ULE: return SPF_UMIN;
    case ICmpInst::ICMP_SLT:
    case ICmpInst::ICMP_SLE: return SPF_SMIN;
    }
  }
  
  // (icmp X, Y) ? Y : X 
  if (SI->getTrueValue() == ICI->getOperand(1) &&
      SI->getFalseValue() == ICI->getOperand(0)) {
    switch (ICI->getPredicate()) {
      default: return SPF_UNKNOWN; // Equality.
      case ICmpInst::ICMP_UGT:
      case ICmpInst::ICMP_UGE: return SPF_UMIN;
      case ICmpInst::ICMP_SGT:
      case ICmpInst::ICMP_SGE: return SPF_SMIN;
      case ICmpInst::ICMP_ULT:
      case ICmpInst::ICMP_ULE: return SPF_UMAX;
      case ICmpInst::ICMP_SLT:
      case ICmpInst::ICMP_SLE: return SPF_SMAX;
    }
  }
  
  // TODO: (X > 4) ? X : 5   -->  (X >= 5) ? X : 5  -->  MAX(X, 5)
  
  return SPF_UNKNOWN;
}


/// GetSelectFoldableOperands - We want to turn code that looks like this:
///   %C = or %A, %B
///   %D = select %cond, %C, %A
/// into:
///   %C = select %cond, %B, 0
///   %D = or %A, %C
///
/// Assuming that the specified instruction is an operand to the select, return
/// a bitmask indicating which operands of this instruction are foldable if they
/// equal the other incoming value of the select.
///
static unsigned GetSelectFoldableOperands(Instruction *I) {
  switch (I->getOpcode()) {
  case Instruction::Add:
  case Instruction::Mul:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    return 3;              // Can fold through either operand.
  case Instruction::Sub:   // Can only fold on the amount subtracted.
  case Instruction::Shl:   // Can only fold on the shift amount.
  case Instruction::LShr:
  case Instruction::AShr:
    return 1;
  default:
    return 0;              // Cannot fold
  }
}

/// GetSelectFoldableConstant - For the same transformation as the previous
/// function, return the identity constant that goes into the select.
static Constant *GetSelectFoldableConstant(Instruction *I) {
  switch (I->getOpcode()) {
  default: llvm_unreachable("This cannot happen!");
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    return Constant::getNullValue(I->getType());
  case Instruction::And:
    return Constant::getAllOnesValue(I->getType());
  case Instruction::Mul:
    return ConstantInt::get(I->getType(), 1);
  }
}

/// FoldSelectOpOp - Here we have (select c, TI, FI), and we know that TI and FI
/// have the same opcode and only one use each.  Try to simplify this.
Instruction *InstCombiner::FoldSelectOpOp(SelectInst &SI, Instruction *TI,
                                          Instruction *FI) {
  if (TI->getNumOperands() == 1) {
    // If this is a non-volatile load or a cast from the same type,
    // merge.
    if (TI->isCast()) {
      if (TI->getOperand(0)->getType() != FI->getOperand(0)->getType())
        return 0;
    } else {
      return 0;  // unknown unary op.
    }

    // Fold this by inserting a select from the input values.
    SelectInst *NewSI = SelectInst::Create(SI.getCondition(), TI->getOperand(0),
                                          FI->getOperand(0), SI.getName()+".v");
    InsertNewInstBefore(NewSI, SI);
    return CastInst::Create(Instruction::CastOps(TI->getOpcode()), NewSI, 
                            TI->getType());
  }

  // Only handle binary operators here.
  if (!isa<BinaryOperator>(TI))
    return 0;

  // Figure out if the operations have any operands in common.
  Value *MatchOp, *OtherOpT, *OtherOpF;
  bool MatchIsOpZero;
  if (TI->getOperand(0) == FI->getOperand(0)) {
    MatchOp  = TI->getOperand(0);
    OtherOpT = TI->getOperand(1);
    OtherOpF = FI->getOperand(1);
    MatchIsOpZero = true;
  } else if (TI->getOperand(1) == FI->getOperand(1)) {
    MatchOp  = TI->getOperand(1);
    OtherOpT = TI->getOperand(0);
    OtherOpF = FI->getOperand(0);
    MatchIsOpZero = false;
  } else if (!TI->isCommutative()) {
    return 0;
  } else if (TI->getOperand(0) == FI->getOperand(1)) {
    MatchOp  = TI->getOperand(0);
    OtherOpT = TI->getOperand(1);
    OtherOpF = FI->getOperand(0);
    MatchIsOpZero = true;
  } else if (TI->getOperand(1) == FI->getOperand(0)) {
    MatchOp  = TI->getOperand(1);
    OtherOpT = TI->getOperand(0);
    OtherOpF = FI->getOperand(1);
    MatchIsOpZero = true;
  } else {
    return 0;
  }

  // If we reach here, they do have operations in common.
  SelectInst *NewSI = SelectInst::Create(SI.getCondition(), OtherOpT,
                                         OtherOpF, SI.getName()+".v");
  InsertNewInstBefore(NewSI, SI);

  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(TI)) {
    if (MatchIsOpZero)
      return BinaryOperator::Create(BO->getOpcode(), MatchOp, NewSI);
    else
      return BinaryOperator::Create(BO->getOpcode(), NewSI, MatchOp);
  }
  llvm_unreachable("Shouldn't get here");
  return 0;
}

static bool isSelect01(Constant *C1, Constant *C2) {
  ConstantInt *C1I = dyn_cast<ConstantInt>(C1);
  if (!C1I)
    return false;
  ConstantInt *C2I = dyn_cast<ConstantInt>(C2);
  if (!C2I)
    return false;
  return (C1I->isZero() || C1I->isOne()) && (C2I->isZero() || C2I->isOne());
}

/// FoldSelectIntoOp - Try fold the select into one of the operands to
/// facilitate further optimization.
Instruction *InstCombiner::FoldSelectIntoOp(SelectInst &SI, Value *TrueVal,
                                            Value *FalseVal) {
  // See the comment above GetSelectFoldableOperands for a description of the
  // transformation we are doing here.
  if (Instruction *TVI = dyn_cast<Instruction>(TrueVal)) {
    if (TVI->hasOneUse() && TVI->getNumOperands() == 2 &&
        !isa<Constant>(FalseVal)) {
      if (unsigned SFO = GetSelectFoldableOperands(TVI)) {
        unsigned OpToFold = 0;
        if ((SFO & 1) && FalseVal == TVI->getOperand(0)) {
          OpToFold = 1;
        } else  if ((SFO & 2) && FalseVal == TVI->getOperand(1)) {
          OpToFold = 2;
        }

        if (OpToFold) {
          Constant *C = GetSelectFoldableConstant(TVI);
          Value *OOp = TVI->getOperand(2-OpToFold);
          // Avoid creating select between 2 constants unless it's selecting
          // between 0 and 1.
          if (!isa<Constant>(OOp) || isSelect01(C, cast<Constant>(OOp))) {
            Instruction *NewSel = SelectInst::Create(SI.getCondition(), OOp, C);
            InsertNewInstBefore(NewSel, SI);
            NewSel->takeName(TVI);
            if (BinaryOperator *BO = dyn_cast<BinaryOperator>(TVI))
              return BinaryOperator::Create(BO->getOpcode(), FalseVal, NewSel);
            llvm_unreachable("Unknown instruction!!");
          }
        }
      }
    }
  }

  if (Instruction *FVI = dyn_cast<Instruction>(FalseVal)) {
    if (FVI->hasOneUse() && FVI->getNumOperands() == 2 &&
        !isa<Constant>(TrueVal)) {
      if (unsigned SFO = GetSelectFoldableOperands(FVI)) {
        unsigned OpToFold = 0;
        if ((SFO & 1) && TrueVal == FVI->getOperand(0)) {
          OpToFold = 1;
        } else  if ((SFO & 2) && TrueVal == FVI->getOperand(1)) {
          OpToFold = 2;
        }

        if (OpToFold) {
          Constant *C = GetSelectFoldableConstant(FVI);
          Value *OOp = FVI->getOperand(2-OpToFold);
          // Avoid creating select between 2 constants unless it's selecting
          // between 0 and 1.
          if (!isa<Constant>(OOp) || isSelect01(C, cast<Constant>(OOp))) {
            Instruction *NewSel = SelectInst::Create(SI.getCondition(), C, OOp);
            InsertNewInstBefore(NewSel, SI);
            NewSel->takeName(FVI);
            if (BinaryOperator *BO = dyn_cast<BinaryOperator>(FVI))
              return BinaryOperator::Create(BO->getOpcode(), TrueVal, NewSel);
            llvm_unreachable("Unknown instruction!!");
          }
        }
      }
    }
  }

  return 0;
}

/// visitSelectInstWithICmp - Visit a SelectInst that has an
/// ICmpInst as its first operand.
///
Instruction *InstCombiner::visitSelectInstWithICmp(SelectInst &SI,
                                                   ICmpInst *ICI) {
  bool Changed = false;
  ICmpInst::Predicate Pred = ICI->getPredicate();
  Value *CmpLHS = ICI->getOperand(0);
  Value *CmpRHS = ICI->getOperand(1);
  Value *TrueVal = SI.getTrueValue();
  Value *FalseVal = SI.getFalseValue();

  // Check cases where the comparison is with a constant that
  // can be adjusted to fit the min/max idiom. We may edit ICI in
  // place here, so make sure the select is the only user.
  if (ICI->hasOneUse())
    if (ConstantInt *CI = dyn_cast<ConstantInt>(CmpRHS)) {
      switch (Pred) {
      default: break;
      case ICmpInst::ICMP_ULT:
      case ICmpInst::ICMP_SLT: {
        // X < MIN ? T : F  -->  F
        if (CI->isMinValue(Pred == ICmpInst::ICMP_SLT))
          return ReplaceInstUsesWith(SI, FalseVal);
        // X < C ? X : C-1  -->  X > C-1 ? C-1 : X
        Constant *AdjustedRHS =
          ConstantInt::get(CI->getContext(), CI->getValue()-1);
        if ((CmpLHS == TrueVal && AdjustedRHS == FalseVal) ||
            (CmpLHS == FalseVal && AdjustedRHS == TrueVal)) {
          Pred = ICmpInst::getSwappedPredicate(Pred);
          CmpRHS = AdjustedRHS;
          std::swap(FalseVal, TrueVal);
          ICI->setPredicate(Pred);
          ICI->setOperand(1, CmpRHS);
          SI.setOperand(1, TrueVal);
          SI.setOperand(2, FalseVal);
          Changed = true;
        }
        break;
      }
      case ICmpInst::ICMP_UGT:
      case ICmpInst::ICMP_SGT: {
        // X > MAX ? T : F  -->  F
        if (CI->isMaxValue(Pred == ICmpInst::ICMP_SGT))
          return ReplaceInstUsesWith(SI, FalseVal);
        // X > C ? X : C+1  -->  X < C+1 ? C+1 : X
        Constant *AdjustedRHS =
          ConstantInt::get(CI->getContext(), CI->getValue()+1);
        if ((CmpLHS == TrueVal && AdjustedRHS == FalseVal) ||
            (CmpLHS == FalseVal && AdjustedRHS == TrueVal)) {
          Pred = ICmpInst::getSwappedPredicate(Pred);
          CmpRHS = AdjustedRHS;
          std::swap(FalseVal, TrueVal);
          ICI->setPredicate(Pred);
          ICI->setOperand(1, CmpRHS);
          SI.setOperand(1, TrueVal);
          SI.setOperand(2, FalseVal);
          Changed = true;
        }
        break;
      }
      }
    }

  if (CmpLHS == TrueVal && CmpRHS == FalseVal) {
    // Transform (X == Y) ? X : Y  -> Y
    if (Pred == ICmpInst::ICMP_EQ)
      return ReplaceInstUsesWith(SI, FalseVal);
    // Transform (X != Y) ? X : Y  -> X
    if (Pred == ICmpInst::ICMP_NE)
      return ReplaceInstUsesWith(SI, TrueVal);
    /// NOTE: if we wanted to, this is where to detect integer MIN/MAX

  } else if (CmpLHS == FalseVal && CmpRHS == TrueVal) {
    // Transform (X == Y) ? Y : X  -> X
    if (Pred == ICmpInst::ICMP_EQ)
      return ReplaceInstUsesWith(SI, FalseVal);
    // Transform (X != Y) ? Y : X  -> Y
    if (Pred == ICmpInst::ICMP_NE)
      return ReplaceInstUsesWith(SI, TrueVal);
    /// NOTE: if we wanted to, this is where to detect integer MIN/MAX
  }
  return Changed ? &SI : 0;
}


/// CanSelectOperandBeMappingIntoPredBlock - SI is a select whose condition is a
/// PHI node (but the two may be in different blocks).  See if the true/false
/// values (V) are live in all of the predecessor blocks of the PHI.  For
/// example, cases like this cannot be mapped:
///
///   X = phi [ C1, BB1], [C2, BB2]
///   Y = add
///   Z = select X, Y, 0
///
/// because Y is not live in BB1/BB2.
///
static bool CanSelectOperandBeMappingIntoPredBlock(const Value *V,
                                                   const SelectInst &SI) {
  // If the value is a non-instruction value like a constant or argument, it
  // can always be mapped.
  const Instruction *I = dyn_cast<Instruction>(V);
  if (I == 0) return true;
  
  // If V is a PHI node defined in the same block as the condition PHI, we can
  // map the arguments.
  const PHINode *CondPHI = cast<PHINode>(SI.getCondition());
  
  if (const PHINode *VP = dyn_cast<PHINode>(I))
    if (VP->getParent() == CondPHI->getParent())
      return true;
  
  // Otherwise, if the PHI and select are defined in the same block and if V is
  // defined in a different block, then we can transform it.
  if (SI.getParent() == CondPHI->getParent() &&
      I->getParent() != CondPHI->getParent())
    return true;
  
  // Otherwise we have a 'hard' case and we can't tell without doing more
  // detailed dominator based analysis, punt.
  return false;
}

/// FoldSPFofSPF - We have an SPF (e.g. a min or max) of an SPF of the form:
///   SPF2(SPF1(A, B), C) 
Instruction *InstCombiner::FoldSPFofSPF(Instruction *Inner,
                                        SelectPatternFlavor SPF1,
                                        Value *A, Value *B,
                                        Instruction &Outer,
                                        SelectPatternFlavor SPF2, Value *C) {
  if (C == A || C == B) {
    // MAX(MAX(A, B), B) -> MAX(A, B)
    // MIN(MIN(a, b), a) -> MIN(a, b)
    if (SPF1 == SPF2)
      return ReplaceInstUsesWith(Outer, Inner);
    
    // MAX(MIN(a, b), a) -> a
    // MIN(MAX(a, b), a) -> a
    if ((SPF1 == SPF_SMIN && SPF2 == SPF_SMAX) ||
        (SPF1 == SPF_SMAX && SPF2 == SPF_SMIN) ||
        (SPF1 == SPF_UMIN && SPF2 == SPF_UMAX) ||
        (SPF1 == SPF_UMAX && SPF2 == SPF_UMIN))
      return ReplaceInstUsesWith(Outer, C);
  }
  
  // TODO: MIN(MIN(A, 23), 97)
  return 0;
}




Instruction *InstCombiner::visitSelectInst(SelectInst &SI) {
  Value *CondVal = SI.getCondition();
  Value *TrueVal = SI.getTrueValue();
  Value *FalseVal = SI.getFalseValue();

  if (Value *V = SimplifySelectInst(CondVal, TrueVal, FalseVal, TD))
    return ReplaceInstUsesWith(SI, V);

  if (SI.getType()->isIntegerTy(1)) {
    if (ConstantInt *C = dyn_cast<ConstantInt>(TrueVal)) {
      if (C->getZExtValue()) {
        // Change: A = select B, true, C --> A = or B, C
        return BinaryOperator::CreateOr(CondVal, FalseVal);
      }
      // Change: A = select B, false, C --> A = and !B, C
      Value *NotCond =
        InsertNewInstBefore(BinaryOperator::CreateNot(CondVal,
                                           "not."+CondVal->getName()), SI);
      return BinaryOperator::CreateAnd(NotCond, FalseVal);
    } else if (ConstantInt *C = dyn_cast<ConstantInt>(FalseVal)) {
      if (C->getZExtValue() == false) {
        // Change: A = select B, C, false --> A = and B, C
        return BinaryOperator::CreateAnd(CondVal, TrueVal);
      }
      // Change: A = select B, C, true --> A = or !B, C
      Value *NotCond =
        InsertNewInstBefore(BinaryOperator::CreateNot(CondVal,
                                           "not."+CondVal->getName()), SI);
      return BinaryOperator::CreateOr(NotCond, TrueVal);
    }
    
    // select a, b, a  -> a&b
    // select a, a, b  -> a|b
    if (CondVal == TrueVal)
      return BinaryOperator::CreateOr(CondVal, FalseVal);
    else if (CondVal == FalseVal)
      return BinaryOperator::CreateAnd(CondVal, TrueVal);
  }

  // Selecting between two integer constants?
  if (ConstantInt *TrueValC = dyn_cast<ConstantInt>(TrueVal))
    if (ConstantInt *FalseValC = dyn_cast<ConstantInt>(FalseVal)) {
      // select C, 1, 0 -> zext C to int
      if (FalseValC->isZero() && TrueValC->getValue() == 1)
        return new ZExtInst(CondVal, SI.getType());

      // select C, -1, 0 -> sext C to int
      if (FalseValC->isZero() && TrueValC->isAllOnesValue())
        return new SExtInst(CondVal, SI.getType());
      
      // select C, 0, 1 -> zext !C to int
      if (TrueValC->isZero() && FalseValC->getValue() == 1) {
        Value *NotCond = Builder->CreateNot(CondVal, "not."+CondVal->getName());
        return new ZExtInst(NotCond, SI.getType());
      }

      // select C, 0, -1 -> sext !C to int
      if (TrueValC->isZero() && FalseValC->isAllOnesValue()) {
        Value *NotCond = Builder->CreateNot(CondVal, "not."+CondVal->getName());
        return new SExtInst(NotCond, SI.getType());
      }
      
      if (ICmpInst *IC = dyn_cast<ICmpInst>(SI.getCondition())) {
        // If one of the constants is zero (we know they can't both be) and we
        // have an icmp instruction with zero, and we have an 'and' with the
        // non-constant value, eliminate this whole mess.  This corresponds to
        // cases like this: ((X & 27) ? 27 : 0)
        if (TrueValC->isZero() || FalseValC->isZero())
          if (IC->isEquality() && isa<ConstantInt>(IC->getOperand(1)) &&
              cast<Constant>(IC->getOperand(1))->isNullValue())
            if (Instruction *ICA = dyn_cast<Instruction>(IC->getOperand(0)))
              if (ICA->getOpcode() == Instruction::And &&
                  isa<ConstantInt>(ICA->getOperand(1)) &&
                  (ICA->getOperand(1) == TrueValC ||
                   ICA->getOperand(1) == FalseValC) &&
               cast<ConstantInt>(ICA->getOperand(1))->getValue().isPowerOf2()) {
                // Okay, now we know that everything is set up, we just don't
                // know whether we have a icmp_ne or icmp_eq and whether the 
                // true or false val is the zero.
                bool ShouldNotVal = !TrueValC->isZero();
                ShouldNotVal ^= IC->getPredicate() == ICmpInst::ICMP_NE;
                Value *V = ICA;
                if (ShouldNotVal)
                  V = Builder->CreateXor(V, ICA->getOperand(1));
                return ReplaceInstUsesWith(SI, V);
              }
      }
    }

  // See if we are selecting two values based on a comparison of the two values.
  if (FCmpInst *FCI = dyn_cast<FCmpInst>(CondVal)) {
    if (FCI->getOperand(0) == TrueVal && FCI->getOperand(1) == FalseVal) {
      // Transform (X == Y) ? X : Y  -> Y
      if (FCI->getPredicate() == FCmpInst::FCMP_OEQ) {
        // This is not safe in general for floating point:  
        // consider X== -0, Y== +0.
        // It becomes safe if either operand is a nonzero constant.
        ConstantFP *CFPt, *CFPf;
        if (((CFPt = dyn_cast<ConstantFP>(TrueVal)) &&
              !CFPt->getValueAPF().isZero()) ||
            ((CFPf = dyn_cast<ConstantFP>(FalseVal)) &&
             !CFPf->getValueAPF().isZero()))
        return ReplaceInstUsesWith(SI, FalseVal);
      }
      // Transform (X une Y) ? X : Y  -> X
      if (FCI->getPredicate() == FCmpInst::FCMP_UNE) {
        // This is not safe in general for floating point:  
        // consider X== -0, Y== +0.
        // It becomes safe if either operand is a nonzero constant.
        ConstantFP *CFPt, *CFPf;
        if (((CFPt = dyn_cast<ConstantFP>(TrueVal)) &&
              !CFPt->getValueAPF().isZero()) ||
            ((CFPf = dyn_cast<ConstantFP>(FalseVal)) &&
             !CFPf->getValueAPF().isZero()))
        return ReplaceInstUsesWith(SI, TrueVal);
      }
      // NOTE: if we wanted to, this is where to detect MIN/MAX

    } else if (FCI->getOperand(0) == FalseVal && FCI->getOperand(1) == TrueVal){
      // Transform (X == Y) ? Y : X  -> X
      if (FCI->getPredicate() == FCmpInst::FCMP_OEQ) {
        // This is not safe in general for floating point:  
        // consider X== -0, Y== +0.
        // It becomes safe if either operand is a nonzero constant.
        ConstantFP *CFPt, *CFPf;
        if (((CFPt = dyn_cast<ConstantFP>(TrueVal)) &&
              !CFPt->getValueAPF().isZero()) ||
            ((CFPf = dyn_cast<ConstantFP>(FalseVal)) &&
             !CFPf->getValueAPF().isZero()))
          return ReplaceInstUsesWith(SI, FalseVal);
      }
      // Transform (X une Y) ? Y : X  -> Y
      if (FCI->getPredicate() == FCmpInst::FCMP_UNE) {
        // This is not safe in general for floating point:  
        // consider X== -0, Y== +0.
        // It becomes safe if either operand is a nonzero constant.
        ConstantFP *CFPt, *CFPf;
        if (((CFPt = dyn_cast<ConstantFP>(TrueVal)) &&
              !CFPt->getValueAPF().isZero()) ||
            ((CFPf = dyn_cast<ConstantFP>(FalseVal)) &&
             !CFPf->getValueAPF().isZero()))
          return ReplaceInstUsesWith(SI, TrueVal);
      }
      // NOTE: if we wanted to, this is where to detect MIN/MAX
    }
    // NOTE: if we wanted to, this is where to detect ABS
  }

  // See if we are selecting two values based on a comparison of the two values.
  if (ICmpInst *ICI = dyn_cast<ICmpInst>(CondVal))
    if (Instruction *Result = visitSelectInstWithICmp(SI, ICI))
      return Result;

  if (Instruction *TI = dyn_cast<Instruction>(TrueVal))
    if (Instruction *FI = dyn_cast<Instruction>(FalseVal))
      if (TI->hasOneUse() && FI->hasOneUse()) {
        Instruction *AddOp = 0, *SubOp = 0;

        // Turn (select C, (op X, Y), (op X, Z)) -> (op X, (select C, Y, Z))
        if (TI->getOpcode() == FI->getOpcode())
          if (Instruction *IV = FoldSelectOpOp(SI, TI, FI))
            return IV;

        // Turn select C, (X+Y), (X-Y) --> (X+(select C, Y, (-Y))).  This is
        // even legal for FP.
        if ((TI->getOpcode() == Instruction::Sub &&
             FI->getOpcode() == Instruction::Add) ||
            (TI->getOpcode() == Instruction::FSub &&
             FI->getOpcode() == Instruction::FAdd)) {
          AddOp = FI; SubOp = TI;
        } else if ((FI->getOpcode() == Instruction::Sub &&
                    TI->getOpcode() == Instruction::Add) ||
                   (FI->getOpcode() == Instruction::FSub &&
                    TI->getOpcode() == Instruction::FAdd)) {
          AddOp = TI; SubOp = FI;
        }

        if (AddOp) {
          Value *OtherAddOp = 0;
          if (SubOp->getOperand(0) == AddOp->getOperand(0)) {
            OtherAddOp = AddOp->getOperand(1);
          } else if (SubOp->getOperand(0) == AddOp->getOperand(1)) {
            OtherAddOp = AddOp->getOperand(0);
          }

          if (OtherAddOp) {
            // So at this point we know we have (Y -> OtherAddOp):
            //        select C, (add X, Y), (sub X, Z)
            Value *NegVal;  // Compute -Z
            if (Constant *C = dyn_cast<Constant>(SubOp->getOperand(1))) {
              NegVal = ConstantExpr::getNeg(C);
            } else {
              NegVal = InsertNewInstBefore(
                    BinaryOperator::CreateNeg(SubOp->getOperand(1),
                                              "tmp"), SI);
            }

            Value *NewTrueOp = OtherAddOp;
            Value *NewFalseOp = NegVal;
            if (AddOp != TI)
              std::swap(NewTrueOp, NewFalseOp);
            Instruction *NewSel =
              SelectInst::Create(CondVal, NewTrueOp,
                                 NewFalseOp, SI.getName() + ".p");

            NewSel = InsertNewInstBefore(NewSel, SI);
            return BinaryOperator::CreateAdd(SubOp->getOperand(0), NewSel);
          }
        }
      }

  // See if we can fold the select into one of our operands.
  if (SI.getType()->isIntegerTy()) {
    if (Instruction *FoldI = FoldSelectIntoOp(SI, TrueVal, FalseVal))
      return FoldI;
    
    // MAX(MAX(a, b), a) -> MAX(a, b)
    // MIN(MIN(a, b), a) -> MIN(a, b)
    // MAX(MIN(a, b), a) -> a
    // MIN(MAX(a, b), a) -> a
    Value *LHS, *RHS, *LHS2, *RHS2;
    if (SelectPatternFlavor SPF = MatchSelectPattern(&SI, LHS, RHS)) {
      if (SelectPatternFlavor SPF2 = MatchSelectPattern(LHS, LHS2, RHS2))
        if (Instruction *R = FoldSPFofSPF(cast<Instruction>(LHS),SPF2,LHS2,RHS2, 
                                          SI, SPF, RHS))
          return R;
      if (SelectPatternFlavor SPF2 = MatchSelectPattern(RHS, LHS2, RHS2))
        if (Instruction *R = FoldSPFofSPF(cast<Instruction>(RHS),SPF2,LHS2,RHS2,
                                          SI, SPF, LHS))
          return R;
    }

    // TODO.
    // ABS(-X) -> ABS(X)
    // ABS(ABS(X)) -> ABS(X)
  }

  // See if we can fold the select into a phi node if the condition is a select.
  if (isa<PHINode>(SI.getCondition())) 
    // The true/false values have to be live in the PHI predecessor's blocks.
    if (CanSelectOperandBeMappingIntoPredBlock(TrueVal, SI) &&
        CanSelectOperandBeMappingIntoPredBlock(FalseVal, SI))
      if (Instruction *NV = FoldOpIntoPhi(SI))
        return NV;

  if (BinaryOperator::isNot(CondVal)) {
    SI.setOperand(0, BinaryOperator::getNotArgument(CondVal));
    SI.setOperand(1, FalseVal);
    SI.setOperand(2, TrueVal);
    return &SI;
  }

  return 0;
}
