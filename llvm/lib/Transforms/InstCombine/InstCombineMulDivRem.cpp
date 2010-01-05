//===- InstCombineMulDivRem.cpp -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the visit functions for mul, fmul, sdiv, udiv, fdiv,
// srem, urem, frem.
//
//===----------------------------------------------------------------------===//

#include "InstCombine.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Support/PatternMatch.h"
using namespace llvm;
using namespace PatternMatch;

/// SubOne - Subtract one from a ConstantInt.
static Constant *SubOne(ConstantInt *C) {
  return ConstantInt::get(C->getContext(), C->getValue()-1);
}

/// MultiplyOverflows - True if the multiply can not be expressed in an int
/// this size.
static bool MultiplyOverflows(ConstantInt *C1, ConstantInt *C2, bool sign) {
  uint32_t W = C1->getBitWidth();
  APInt LHSExt = C1->getValue(), RHSExt = C2->getValue();
  if (sign) {
    LHSExt.sext(W * 2);
    RHSExt.sext(W * 2);
  } else {
    LHSExt.zext(W * 2);
    RHSExt.zext(W * 2);
  }
  
  APInt MulExt = LHSExt * RHSExt;
  
  if (!sign)
    return MulExt.ugt(APInt::getLowBitsSet(W * 2, W));
  
  APInt Min = APInt::getSignedMinValue(W).sext(W * 2);
  APInt Max = APInt::getSignedMaxValue(W).sext(W * 2);
  return MulExt.slt(Min) || MulExt.sgt(Max);
}

Instruction *InstCombiner::visitMul(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (isa<UndefValue>(Op1))              // undef * X -> 0
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  // Simplify mul instructions with a constant RHS.
  if (Constant *Op1C = dyn_cast<Constant>(Op1)) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1C)) {

      // ((X << C1)*C2) == (X * (C2 << C1))
      if (BinaryOperator *SI = dyn_cast<BinaryOperator>(Op0))
        if (SI->getOpcode() == Instruction::Shl)
          if (Constant *ShOp = dyn_cast<Constant>(SI->getOperand(1)))
            return BinaryOperator::CreateMul(SI->getOperand(0),
                                        ConstantExpr::getShl(CI, ShOp));

      if (CI->isZero())
        return ReplaceInstUsesWith(I, Op1C);  // X * 0  == 0
      if (CI->equalsInt(1))                  // X * 1  == X
        return ReplaceInstUsesWith(I, Op0);
      if (CI->isAllOnesValue())              // X * -1 == 0 - X
        return BinaryOperator::CreateNeg(Op0, I.getName());

      const APInt& Val = cast<ConstantInt>(CI)->getValue();
      if (Val.isPowerOf2()) {          // Replace X*(2^C) with X << C
        return BinaryOperator::CreateShl(Op0,
                 ConstantInt::get(Op0->getType(), Val.logBase2()));
      }
    } else if (isa<VectorType>(Op1C->getType())) {
      if (Op1C->isNullValue())
        return ReplaceInstUsesWith(I, Op1C);

      if (ConstantVector *Op1V = dyn_cast<ConstantVector>(Op1C)) {
        if (Op1V->isAllOnesValue())              // X * -1 == 0 - X
          return BinaryOperator::CreateNeg(Op0, I.getName());

        // As above, vector X*splat(1.0) -> X in all defined cases.
        if (Constant *Splat = Op1V->getSplatValue()) {
          if (ConstantInt *CI = dyn_cast<ConstantInt>(Splat))
            if (CI->equalsInt(1))
              return ReplaceInstUsesWith(I, Op0);
        }
      }
    }
    
    if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0))
      if (Op0I->getOpcode() == Instruction::Add && Op0I->hasOneUse() &&
          isa<ConstantInt>(Op0I->getOperand(1)) && isa<ConstantInt>(Op1C)) {
        // Canonicalize (X+C1)*C2 -> X*C2+C1*C2.
        Value *Add = Builder->CreateMul(Op0I->getOperand(0), Op1C, "tmp");
        Value *C1C2 = Builder->CreateMul(Op1C, Op0I->getOperand(1));
        return BinaryOperator::CreateAdd(Add, C1C2);
        
      }

    // Try to fold constant mul into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;

    if (isa<PHINode>(Op0))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
  }

  if (Value *Op0v = dyn_castNegVal(Op0))     // -X * -Y = X*Y
    if (Value *Op1v = dyn_castNegVal(Op1))
      return BinaryOperator::CreateMul(Op0v, Op1v);

  // (X / Y) *  Y = X - (X % Y)
  // (X / Y) * -Y = (X % Y) - X
  {
    Value *Op1C = Op1;
    BinaryOperator *BO = dyn_cast<BinaryOperator>(Op0);
    if (!BO ||
        (BO->getOpcode() != Instruction::UDiv && 
         BO->getOpcode() != Instruction::SDiv)) {
      Op1C = Op0;
      BO = dyn_cast<BinaryOperator>(Op1);
    }
    Value *Neg = dyn_castNegVal(Op1C);
    if (BO && BO->hasOneUse() &&
        (BO->getOperand(1) == Op1C || BO->getOperand(1) == Neg) &&
        (BO->getOpcode() == Instruction::UDiv ||
         BO->getOpcode() == Instruction::SDiv)) {
      Value *Op0BO = BO->getOperand(0), *Op1BO = BO->getOperand(1);

      // If the division is exact, X % Y is zero.
      if (SDivOperator *SDiv = dyn_cast<SDivOperator>(BO))
        if (SDiv->isExact()) {
          if (Op1BO == Op1C)
            return ReplaceInstUsesWith(I, Op0BO);
          return BinaryOperator::CreateNeg(Op0BO);
        }

      Value *Rem;
      if (BO->getOpcode() == Instruction::UDiv)
        Rem = Builder->CreateURem(Op0BO, Op1BO);
      else
        Rem = Builder->CreateSRem(Op0BO, Op1BO);
      Rem->takeName(BO);

      if (Op1BO == Op1C)
        return BinaryOperator::CreateSub(Op0BO, Rem);
      return BinaryOperator::CreateSub(Rem, Op0BO);
    }
  }

  /// i1 mul -> i1 and.
  if (I.getType() == Type::getInt1Ty(I.getContext()))
    return BinaryOperator::CreateAnd(Op0, Op1);

  // X*(1 << Y) --> X << Y
  // (1 << Y)*X --> X << Y
  {
    Value *Y;
    if (match(Op0, m_Shl(m_One(), m_Value(Y))))
      return BinaryOperator::CreateShl(Op1, Y);
    if (match(Op1, m_Shl(m_One(), m_Value(Y))))
      return BinaryOperator::CreateShl(Op0, Y);
  }
  
  // If one of the operands of the multiply is a cast from a boolean value, then
  // we know the bool is either zero or one, so this is a 'masking' multiply.
  //   X * Y (where Y is 0 or 1) -> X & (0-Y)
  if (!isa<VectorType>(I.getType())) {
    // -2 is "-1 << 1" so it is all bits set except the low one.
    APInt Negative2(I.getType()->getPrimitiveSizeInBits(), (uint64_t)-2, true);
    
    Value *BoolCast = 0, *OtherOp = 0;
    if (MaskedValueIsZero(Op0, Negative2))
      BoolCast = Op0, OtherOp = Op1;
    else if (MaskedValueIsZero(Op1, Negative2))
      BoolCast = Op1, OtherOp = Op0;

    if (BoolCast) {
      Value *V = Builder->CreateSub(Constant::getNullValue(I.getType()),
                                    BoolCast, "tmp");
      return BinaryOperator::CreateAnd(V, OtherOp);
    }
  }

  return Changed ? &I : 0;
}

Instruction *InstCombiner::visitFMul(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // Simplify mul instructions with a constant RHS...
  if (Constant *Op1C = dyn_cast<Constant>(Op1)) {
    if (ConstantFP *Op1F = dyn_cast<ConstantFP>(Op1C)) {
      // "In IEEE floating point, x*1 is not equivalent to x for nans.  However,
      // ANSI says we can drop signals, so we can do this anyway." (from GCC)
      if (Op1F->isExactlyValue(1.0))
        return ReplaceInstUsesWith(I, Op0);  // Eliminate 'mul double %X, 1.0'
    } else if (isa<VectorType>(Op1C->getType())) {
      if (ConstantVector *Op1V = dyn_cast<ConstantVector>(Op1C)) {
        // As above, vector X*splat(1.0) -> X in all defined cases.
        if (Constant *Splat = Op1V->getSplatValue()) {
          if (ConstantFP *F = dyn_cast<ConstantFP>(Splat))
            if (F->isExactlyValue(1.0))
              return ReplaceInstUsesWith(I, Op0);
        }
      }
    }

    // Try to fold constant mul into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;

    if (isa<PHINode>(Op0))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
  }

  if (Value *Op0v = dyn_castFNegVal(Op0))     // -X * -Y = X*Y
    if (Value *Op1v = dyn_castFNegVal(Op1))
      return BinaryOperator::CreateFMul(Op0v, Op1v);

  return Changed ? &I : 0;
}

/// SimplifyDivRemOfSelect - Try to fold a divide or remainder of a select
/// instruction.
bool InstCombiner::SimplifyDivRemOfSelect(BinaryOperator &I) {
  SelectInst *SI = cast<SelectInst>(I.getOperand(1));
  
  // div/rem X, (Cond ? 0 : Y) -> div/rem X, Y
  int NonNullOperand = -1;
  if (Constant *ST = dyn_cast<Constant>(SI->getOperand(1)))
    if (ST->isNullValue())
      NonNullOperand = 2;
  // div/rem X, (Cond ? Y : 0) -> div/rem X, Y
  if (Constant *ST = dyn_cast<Constant>(SI->getOperand(2)))
    if (ST->isNullValue())
      NonNullOperand = 1;
  
  if (NonNullOperand == -1)
    return false;
  
  Value *SelectCond = SI->getOperand(0);
  
  // Change the div/rem to use 'Y' instead of the select.
  I.setOperand(1, SI->getOperand(NonNullOperand));
  
  // Okay, we know we replace the operand of the div/rem with 'Y' with no
  // problem.  However, the select, or the condition of the select may have
  // multiple uses.  Based on our knowledge that the operand must be non-zero,
  // propagate the known value for the select into other uses of it, and
  // propagate a known value of the condition into its other users.
  
  // If the select and condition only have a single use, don't bother with this,
  // early exit.
  if (SI->use_empty() && SelectCond->hasOneUse())
    return true;
  
  // Scan the current block backward, looking for other uses of SI.
  BasicBlock::iterator BBI = &I, BBFront = I.getParent()->begin();
  
  while (BBI != BBFront) {
    --BBI;
    // If we found a call to a function, we can't assume it will return, so
    // information from below it cannot be propagated above it.
    if (isa<CallInst>(BBI) && !isa<IntrinsicInst>(BBI))
      break;
    
    // Replace uses of the select or its condition with the known values.
    for (Instruction::op_iterator I = BBI->op_begin(), E = BBI->op_end();
         I != E; ++I) {
      if (*I == SI) {
        *I = SI->getOperand(NonNullOperand);
        Worklist.Add(BBI);
      } else if (*I == SelectCond) {
        *I = NonNullOperand == 1 ? ConstantInt::getTrue(BBI->getContext()) :
                                   ConstantInt::getFalse(BBI->getContext());
        Worklist.Add(BBI);
      }
    }
    
    // If we past the instruction, quit looking for it.
    if (&*BBI == SI)
      SI = 0;
    if (&*BBI == SelectCond)
      SelectCond = 0;
    
    // If we ran out of things to eliminate, break out of the loop.
    if (SelectCond == 0 && SI == 0)
      break;
    
  }
  return true;
}


/// This function implements the transforms on div instructions that work
/// regardless of the kind of div instruction it is (udiv, sdiv, or fdiv). It is
/// used by the visitors to those instructions.
/// @brief Transforms common to all three div instructions
Instruction *InstCombiner::commonDivTransforms(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // undef / X -> 0        for integer.
  // undef / X -> undef    for FP (the undef could be a snan).
  if (isa<UndefValue>(Op0)) {
    if (Op0->getType()->isFPOrFPVector())
      return ReplaceInstUsesWith(I, Op0);
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
  }

  // X / undef -> undef
  if (isa<UndefValue>(Op1))
    return ReplaceInstUsesWith(I, Op1);

  return 0;
}

/// This function implements the transforms common to both integer division
/// instructions (udiv and sdiv). It is called by the visitors to those integer
/// division instructions.
/// @brief Common integer divide transforms
Instruction *InstCombiner::commonIDivTransforms(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // (sdiv X, X) --> 1     (udiv X, X) --> 1
  if (Op0 == Op1) {
    if (const VectorType *Ty = dyn_cast<VectorType>(I.getType())) {
      Constant *CI = ConstantInt::get(Ty->getElementType(), 1);
      std::vector<Constant*> Elts(Ty->getNumElements(), CI);
      return ReplaceInstUsesWith(I, ConstantVector::get(Elts));
    }

    Constant *CI = ConstantInt::get(I.getType(), 1);
    return ReplaceInstUsesWith(I, CI);
  }
  
  if (Instruction *Common = commonDivTransforms(I))
    return Common;
  
  // Handle cases involving: [su]div X, (select Cond, Y, Z)
  // This does not apply for fdiv.
  if (isa<SelectInst>(Op1) && SimplifyDivRemOfSelect(I))
    return &I;

  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    // div X, 1 == X
    if (RHS->equalsInt(1))
      return ReplaceInstUsesWith(I, Op0);

    // (X / C1) / C2  -> X / (C1*C2)
    if (Instruction *LHS = dyn_cast<Instruction>(Op0))
      if (Instruction::BinaryOps(LHS->getOpcode()) == I.getOpcode())
        if (ConstantInt *LHSRHS = dyn_cast<ConstantInt>(LHS->getOperand(1))) {
          if (MultiplyOverflows(RHS, LHSRHS,
                                I.getOpcode()==Instruction::SDiv))
            return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
          else 
            return BinaryOperator::Create(I.getOpcode(), LHS->getOperand(0),
                                      ConstantExpr::getMul(RHS, LHSRHS));
        }

    if (!RHS->isZero()) { // avoid X udiv 0
      if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
        if (Instruction *R = FoldOpIntoSelect(I, SI))
          return R;
      if (isa<PHINode>(Op0))
        if (Instruction *NV = FoldOpIntoPhi(I))
          return NV;
    }
  }

  // 0 / X == 0, we don't need to preserve faults!
  if (ConstantInt *LHS = dyn_cast<ConstantInt>(Op0))
    if (LHS->equalsInt(0))
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  // It can't be division by zero, hence it must be division by one.
  if (I.getType() == Type::getInt1Ty(I.getContext()))
    return ReplaceInstUsesWith(I, Op0);

  if (ConstantVector *Op1V = dyn_cast<ConstantVector>(Op1)) {
    if (ConstantInt *X = cast_or_null<ConstantInt>(Op1V->getSplatValue()))
      // div X, 1 == X
      if (X->isOne())
        return ReplaceInstUsesWith(I, Op0);
  }

  return 0;
}

Instruction *InstCombiner::visitUDiv(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // Handle the integer div common cases
  if (Instruction *Common = commonIDivTransforms(I))
    return Common;

  if (ConstantInt *C = dyn_cast<ConstantInt>(Op1)) {
    // X udiv C^2 -> X >> C
    // Check to see if this is an unsigned division with an exact power of 2,
    // if so, convert to a right shift.
    if (C->getValue().isPowerOf2())  // 0 not included in isPowerOf2
      return BinaryOperator::CreateLShr(Op0, 
            ConstantInt::get(Op0->getType(), C->getValue().logBase2()));

    // X udiv C, where C >= signbit
    if (C->getValue().isNegative()) {
      Value *IC = Builder->CreateICmpULT( Op0, C);
      return SelectInst::Create(IC, Constant::getNullValue(I.getType()),
                                ConstantInt::get(I.getType(), 1));
    }
  }

  // X udiv (C1 << N), where C1 is "1<<C2"  -->  X >> (N+C2)
  if (BinaryOperator *RHSI = dyn_cast<BinaryOperator>(I.getOperand(1))) {
    if (RHSI->getOpcode() == Instruction::Shl &&
        isa<ConstantInt>(RHSI->getOperand(0))) {
      const APInt& C1 = cast<ConstantInt>(RHSI->getOperand(0))->getValue();
      if (C1.isPowerOf2()) {
        Value *N = RHSI->getOperand(1);
        const Type *NTy = N->getType();
        if (uint32_t C2 = C1.logBase2())
          N = Builder->CreateAdd(N, ConstantInt::get(NTy, C2), "tmp");
        return BinaryOperator::CreateLShr(Op0, N);
      }
    }
  }
  
  // udiv X, (Select Cond, C1, C2) --> Select Cond, (shr X, C1), (shr X, C2)
  // where C1&C2 are powers of two.
  if (SelectInst *SI = dyn_cast<SelectInst>(Op1)) 
    if (ConstantInt *STO = dyn_cast<ConstantInt>(SI->getOperand(1)))
      if (ConstantInt *SFO = dyn_cast<ConstantInt>(SI->getOperand(2)))  {
        const APInt &TVA = STO->getValue(), &FVA = SFO->getValue();
        if (TVA.isPowerOf2() && FVA.isPowerOf2()) {
          // Compute the shift amounts
          uint32_t TSA = TVA.logBase2(), FSA = FVA.logBase2();
          // Construct the "on true" case of the select
          Constant *TC = ConstantInt::get(Op0->getType(), TSA);
          Value *TSI = Builder->CreateLShr(Op0, TC, SI->getName()+".t");
  
          // Construct the "on false" case of the select
          Constant *FC = ConstantInt::get(Op0->getType(), FSA); 
          Value *FSI = Builder->CreateLShr(Op0, FC, SI->getName()+".f");

          // construct the select instruction and return it.
          return SelectInst::Create(SI->getOperand(0), TSI, FSI, SI->getName());
        }
      }
  return 0;
}

Instruction *InstCombiner::visitSDiv(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // Handle the integer div common cases
  if (Instruction *Common = commonIDivTransforms(I))
    return Common;

  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    // sdiv X, -1 == -X
    if (RHS->isAllOnesValue())
      return BinaryOperator::CreateNeg(Op0);

    // sdiv X, C  -->  ashr X, log2(C)
    if (cast<SDivOperator>(&I)->isExact() &&
        RHS->getValue().isNonNegative() &&
        RHS->getValue().isPowerOf2()) {
      Value *ShAmt = llvm::ConstantInt::get(RHS->getType(),
                                            RHS->getValue().exactLogBase2());
      return BinaryOperator::CreateAShr(Op0, ShAmt, I.getName());
    }

    // -X/C  -->  X/-C  provided the negation doesn't overflow.
    if (SubOperator *Sub = dyn_cast<SubOperator>(Op0))
      if (isa<Constant>(Sub->getOperand(0)) &&
          cast<Constant>(Sub->getOperand(0))->isNullValue() &&
          Sub->hasNoSignedWrap())
        return BinaryOperator::CreateSDiv(Sub->getOperand(1),
                                          ConstantExpr::getNeg(RHS));
  }

  // If the sign bits of both operands are zero (i.e. we can prove they are
  // unsigned inputs), turn this into a udiv.
  if (I.getType()->isInteger()) {
    APInt Mask(APInt::getSignBit(I.getType()->getPrimitiveSizeInBits()));
    if (MaskedValueIsZero(Op0, Mask)) {
      if (MaskedValueIsZero(Op1, Mask)) {
        // X sdiv Y -> X udiv Y, iff X and Y don't have sign bit set
        return BinaryOperator::CreateUDiv(Op0, Op1, I.getName());
      }
      ConstantInt *ShiftedInt;
      if (match(Op1, m_Shl(m_ConstantInt(ShiftedInt), m_Value())) &&
          ShiftedInt->getValue().isPowerOf2()) {
        // X sdiv (1 << Y) -> X udiv (1 << Y) ( -> X u>> Y)
        // Safe because the only negative value (1 << Y) can take on is
        // INT_MIN, and X sdiv INT_MIN == X udiv INT_MIN == 0 if X doesn't have
        // the sign bit set.
        return BinaryOperator::CreateUDiv(Op0, Op1, I.getName());
      }
    }
  }
  
  return 0;
}

Instruction *InstCombiner::visitFDiv(BinaryOperator &I) {
  return commonDivTransforms(I);
}

/// This function implements the transforms on rem instructions that work
/// regardless of the kind of rem instruction it is (urem, srem, or frem). It 
/// is used by the visitors to those instructions.
/// @brief Transforms common to all three rem instructions
Instruction *InstCombiner::commonRemTransforms(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (isa<UndefValue>(Op0)) {             // undef % X -> 0
    if (I.getType()->isFPOrFPVector())
      return ReplaceInstUsesWith(I, Op0);  // X % undef -> undef (could be SNaN)
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
  }
  if (isa<UndefValue>(Op1))
    return ReplaceInstUsesWith(I, Op1);  // X % undef -> undef

  // Handle cases involving: rem X, (select Cond, Y, Z)
  if (isa<SelectInst>(Op1) && SimplifyDivRemOfSelect(I))
    return &I;

  return 0;
}

/// This function implements the transforms common to both integer remainder
/// instructions (urem and srem). It is called by the visitors to those integer
/// remainder instructions.
/// @brief Common integer remainder transforms
Instruction *InstCombiner::commonIRemTransforms(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Instruction *common = commonRemTransforms(I))
    return common;

  // 0 % X == 0 for integer, we don't need to preserve faults!
  if (Constant *LHS = dyn_cast<Constant>(Op0))
    if (LHS->isNullValue())
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    // X % 0 == undef, we don't need to preserve faults!
    if (RHS->equalsInt(0))
      return ReplaceInstUsesWith(I, UndefValue::get(I.getType()));
    
    if (RHS->equalsInt(1))  // X % 1 == 0
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

    if (Instruction *Op0I = dyn_cast<Instruction>(Op0)) {
      if (SelectInst *SI = dyn_cast<SelectInst>(Op0I)) {
        if (Instruction *R = FoldOpIntoSelect(I, SI))
          return R;
      } else if (isa<PHINode>(Op0I)) {
        if (Instruction *NV = FoldOpIntoPhi(I))
          return NV;
      }

      // See if we can fold away this rem instruction.
      if (SimplifyDemandedInstructionBits(I))
        return &I;
    }
  }

  return 0;
}

Instruction *InstCombiner::visitURem(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Instruction *common = commonIRemTransforms(I))
    return common;
  
  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    // X urem C^2 -> X and C
    // Check to see if this is an unsigned remainder with an exact power of 2,
    // if so, convert to a bitwise and.
    if (ConstantInt *C = dyn_cast<ConstantInt>(RHS))
      if (C->getValue().isPowerOf2())
        return BinaryOperator::CreateAnd(Op0, SubOne(C));
  }

  if (Instruction *RHSI = dyn_cast<Instruction>(I.getOperand(1))) {
    // Turn A % (C << N), where C is 2^k, into A & ((C << N)-1)  
    if (RHSI->getOpcode() == Instruction::Shl &&
        isa<ConstantInt>(RHSI->getOperand(0))) {
      if (cast<ConstantInt>(RHSI->getOperand(0))->getValue().isPowerOf2()) {
        Constant *N1 = Constant::getAllOnesValue(I.getType());
        Value *Add = Builder->CreateAdd(RHSI, N1, "tmp");
        return BinaryOperator::CreateAnd(Op0, Add);
      }
    }
  }

  // urem X, (select Cond, 2^C1, 2^C2) --> select Cond, (and X, C1), (and X, C2)
  // where C1&C2 are powers of two.
  if (SelectInst *SI = dyn_cast<SelectInst>(Op1)) {
    if (ConstantInt *STO = dyn_cast<ConstantInt>(SI->getOperand(1)))
      if (ConstantInt *SFO = dyn_cast<ConstantInt>(SI->getOperand(2))) {
        // STO == 0 and SFO == 0 handled above.
        if ((STO->getValue().isPowerOf2()) && 
            (SFO->getValue().isPowerOf2())) {
          Value *TrueAnd = Builder->CreateAnd(Op0, SubOne(STO),
                                              SI->getName()+".t");
          Value *FalseAnd = Builder->CreateAnd(Op0, SubOne(SFO),
                                               SI->getName()+".f");
          return SelectInst::Create(SI->getOperand(0), TrueAnd, FalseAnd);
        }
      }
  }
  
  return 0;
}

Instruction *InstCombiner::visitSRem(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // Handle the integer rem common cases
  if (Instruction *Common = commonIRemTransforms(I))
    return Common;
  
  if (Value *RHSNeg = dyn_castNegVal(Op1))
    if (!isa<Constant>(RHSNeg) ||
        (isa<ConstantInt>(RHSNeg) &&
         cast<ConstantInt>(RHSNeg)->getValue().isStrictlyPositive())) {
      // X % -Y -> X % Y
      Worklist.AddValue(I.getOperand(1));
      I.setOperand(1, RHSNeg);
      return &I;
    }

  // If the sign bits of both operands are zero (i.e. we can prove they are
  // unsigned inputs), turn this into a urem.
  if (I.getType()->isInteger()) {
    APInt Mask(APInt::getSignBit(I.getType()->getPrimitiveSizeInBits()));
    if (MaskedValueIsZero(Op1, Mask) && MaskedValueIsZero(Op0, Mask)) {
      // X srem Y -> X urem Y, iff X and Y don't have sign bit set
      return BinaryOperator::CreateURem(Op0, Op1, I.getName());
    }
  }

  // If it's a constant vector, flip any negative values positive.
  if (ConstantVector *RHSV = dyn_cast<ConstantVector>(Op1)) {
    unsigned VWidth = RHSV->getNumOperands();

    bool hasNegative = false;
    for (unsigned i = 0; !hasNegative && i != VWidth; ++i)
      if (ConstantInt *RHS = dyn_cast<ConstantInt>(RHSV->getOperand(i)))
        if (RHS->getValue().isNegative())
          hasNegative = true;

    if (hasNegative) {
      std::vector<Constant *> Elts(VWidth);
      for (unsigned i = 0; i != VWidth; ++i) {
        if (ConstantInt *RHS = dyn_cast<ConstantInt>(RHSV->getOperand(i))) {
          if (RHS->getValue().isNegative())
            Elts[i] = cast<ConstantInt>(ConstantExpr::getNeg(RHS));
          else
            Elts[i] = RHS;
        }
      }

      Constant *NewRHSV = ConstantVector::get(Elts);
      if (NewRHSV != RHSV) {
        Worklist.AddValue(I.getOperand(1));
        I.setOperand(1, NewRHSV);
        return &I;
      }
    }
  }

  return 0;
}

Instruction *InstCombiner::visitFRem(BinaryOperator &I) {
  return commonRemTransforms(I);
}

