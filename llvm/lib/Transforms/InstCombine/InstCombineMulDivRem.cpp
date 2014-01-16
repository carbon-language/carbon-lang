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
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/PatternMatch.h"
using namespace llvm;
using namespace PatternMatch;


/// simplifyValueKnownNonZero - The specific integer value is used in a context
/// where it is known to be non-zero.  If this allows us to simplify the
/// computation, do so and return the new operand, otherwise return null.
static Value *simplifyValueKnownNonZero(Value *V, InstCombiner &IC) {
  // If V has multiple uses, then we would have to do more analysis to determine
  // if this is safe.  For example, the use could be in dynamically unreached
  // code.
  if (!V->hasOneUse()) return 0;

  bool MadeChange = false;

  // ((1 << A) >>u B) --> (1 << (A-B))
  // Because V cannot be zero, we know that B is less than A.
  Value *A = 0, *B = 0, *PowerOf2 = 0;
  if (match(V, m_LShr(m_OneUse(m_Shl(m_Value(PowerOf2), m_Value(A))),
                      m_Value(B))) &&
      // The "1" can be any value known to be a power of 2.
      isKnownToBeAPowerOfTwo(PowerOf2)) {
    A = IC.Builder->CreateSub(A, B);
    return IC.Builder->CreateShl(PowerOf2, A);
  }

  // (PowerOfTwo >>u B) --> isExact since shifting out the result would make it
  // inexact.  Similarly for <<.
  if (BinaryOperator *I = dyn_cast<BinaryOperator>(V))
    if (I->isLogicalShift() && isKnownToBeAPowerOfTwo(I->getOperand(0))) {
      // We know that this is an exact/nuw shift and that the input is a
      // non-zero context as well.
      if (Value *V2 = simplifyValueKnownNonZero(I->getOperand(0), IC)) {
        I->setOperand(0, V2);
        MadeChange = true;
      }

      if (I->getOpcode() == Instruction::LShr && !I->isExact()) {
        I->setIsExact();
        MadeChange = true;
      }

      if (I->getOpcode() == Instruction::Shl && !I->hasNoUnsignedWrap()) {
        I->setHasNoUnsignedWrap();
        MadeChange = true;
      }
    }

  // TODO: Lots more we could do here:
  //    If V is a phi node, we can call this on each of its operands.
  //    "select cond, X, 0" can simplify to "X".

  return MadeChange ? V : 0;
}


/// MultiplyOverflows - True if the multiply can not be expressed in an int
/// this size.
static bool MultiplyOverflows(ConstantInt *C1, ConstantInt *C2, bool sign) {
  uint32_t W = C1->getBitWidth();
  APInt LHSExt = C1->getValue(), RHSExt = C2->getValue();
  if (sign) {
    LHSExt = LHSExt.sext(W * 2);
    RHSExt = RHSExt.sext(W * 2);
  } else {
    LHSExt = LHSExt.zext(W * 2);
    RHSExt = RHSExt.zext(W * 2);
  }

  APInt MulExt = LHSExt * RHSExt;

  if (!sign)
    return MulExt.ugt(APInt::getLowBitsSet(W * 2, W));

  APInt Min = APInt::getSignedMinValue(W).sext(W * 2);
  APInt Max = APInt::getSignedMaxValue(W).sext(W * 2);
  return MulExt.slt(Min) || MulExt.sgt(Max);
}

/// \brief A helper routine of InstCombiner::visitMul().
///
/// If C is a vector of known powers of 2, then this function returns
/// a new vector obtained from C replacing each element with its logBase2.
/// Return a null pointer otherwise.
static Constant *getLogBase2Vector(ConstantDataVector *CV) {
  const APInt *IVal;
  SmallVector<Constant *, 4> Elts;

  for (unsigned I = 0, E = CV->getNumElements(); I != E; ++I) {
    Constant *Elt = CV->getElementAsConstant(I);
    if (!match(Elt, m_APInt(IVal)) || !IVal->isPowerOf2())
      return 0;
    Elts.push_back(ConstantInt::get(Elt->getType(), IVal->logBase2()));
  }

  return ConstantVector::get(Elts);
}

Instruction *InstCombiner::visitMul(BinaryOperator &I) {
  bool Changed = SimplifyAssociativeOrCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyMulInst(Op0, Op1, TD))
    return ReplaceInstUsesWith(I, V);

  if (Value *V = SimplifyUsingDistributiveLaws(I))
    return ReplaceInstUsesWith(I, V);

  if (match(Op1, m_AllOnes()))  // X * -1 == 0 - X
    return BinaryOperator::CreateNeg(Op0, I.getName());

  // Also allow combining multiply instructions on vectors.
  {
    Value *NewOp;
    Constant *C1, *C2;
    const APInt *IVal;
    if (match(&I, m_Mul(m_Shl(m_Value(NewOp), m_Constant(C2)),
                        m_Constant(C1))) &&
        match(C1, m_APInt(IVal)))
      // ((X << C1)*C2) == (X * (C2 << C1))
      return BinaryOperator::CreateMul(NewOp, ConstantExpr::getShl(C1, C2));

    if (match(&I, m_Mul(m_Value(NewOp), m_Constant(C1)))) {
      Constant *NewCst = 0;
      if (match(C1, m_APInt(IVal)) && IVal->isPowerOf2())
        // Replace X*(2^C) with X << C, where C is either a scalar or a splat.
        NewCst = ConstantInt::get(NewOp->getType(), IVal->logBase2());
      else if (ConstantDataVector *CV = dyn_cast<ConstantDataVector>(C1))
        // Replace X*(2^C) with X << C, where C is a vector of known
        // constant powers of 2.
        NewCst = getLogBase2Vector(CV);

      if (NewCst) {
        BinaryOperator *Shl = BinaryOperator::CreateShl(NewOp, NewCst);
        if (I.hasNoSignedWrap()) Shl->setHasNoSignedWrap();
        if (I.hasNoUnsignedWrap()) Shl->setHasNoUnsignedWrap();
        return Shl;
      }
    }
  }

  if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
    // Canonicalize (X+C1)*CI -> X*CI+C1*CI.
    { Value *X; ConstantInt *C1;
      if (Op0->hasOneUse() &&
          match(Op0, m_Add(m_Value(X), m_ConstantInt(C1)))) {
        Value *Add = Builder->CreateMul(X, CI);
        return BinaryOperator::CreateAdd(Add, Builder->CreateMul(C1, CI));
      }
    }

    // (Y - X) * (-(2**n)) -> (X - Y) * (2**n), for positive nonzero n
    // (Y + const) * (-(2**n)) -> (-constY) * (2**n), for positive nonzero n
    // The "* (2**n)" thus becomes a potential shifting opportunity.
    {
      const APInt &   Val = CI->getValue();
      const APInt &PosVal = Val.abs();
      if (Val.isNegative() && PosVal.isPowerOf2()) {
        Value *X = 0, *Y = 0;
        if (Op0->hasOneUse()) {
          ConstantInt *C1;
          Value *Sub = 0;
          if (match(Op0, m_Sub(m_Value(Y), m_Value(X))))
            Sub = Builder->CreateSub(X, Y, "suba");
          else if (match(Op0, m_Add(m_Value(Y), m_ConstantInt(C1))))
            Sub = Builder->CreateSub(Builder->CreateNeg(C1), Y, "subc");
          if (Sub)
            return
              BinaryOperator::CreateMul(Sub,
                                        ConstantInt::get(Y->getType(), PosVal));
        }
      }
    }
  }

  // Simplify mul instructions with a constant RHS.
  if (isa<Constant>(Op1)) {
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

      // If the division is exact, X % Y is zero, so we end up with X or -X.
      if (PossiblyExactOperator *SDiv = dyn_cast<PossiblyExactOperator>(BO))
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
  if (I.getType()->isIntegerTy(1))
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
  if (!I.getType()->isVectorTy()) {
    // -2 is "-1 << 1" so it is all bits set except the low one.
    APInt Negative2(I.getType()->getPrimitiveSizeInBits(), (uint64_t)-2, true);

    Value *BoolCast = 0, *OtherOp = 0;
    if (MaskedValueIsZero(Op0, Negative2))
      BoolCast = Op0, OtherOp = Op1;
    else if (MaskedValueIsZero(Op1, Negative2))
      BoolCast = Op1, OtherOp = Op0;

    if (BoolCast) {
      Value *V = Builder->CreateSub(Constant::getNullValue(I.getType()),
                                    BoolCast);
      return BinaryOperator::CreateAnd(V, OtherOp);
    }
  }

  return Changed ? &I : 0;
}

//
// Detect pattern:
//
// log2(Y*0.5)
//
// And check for corresponding fast math flags
//

static void detectLog2OfHalf(Value *&Op, Value *&Y, IntrinsicInst *&Log2) {

   if (!Op->hasOneUse())
     return;

   IntrinsicInst *II = dyn_cast<IntrinsicInst>(Op);
   if (!II)
     return;
   if (II->getIntrinsicID() != Intrinsic::log2 || !II->hasUnsafeAlgebra())
     return;
   Log2 = II;

   Value *OpLog2Of = II->getArgOperand(0);
   if (!OpLog2Of->hasOneUse())
     return;

   Instruction *I = dyn_cast<Instruction>(OpLog2Of);
   if (!I)
     return;
   if (I->getOpcode() != Instruction::FMul || !I->hasUnsafeAlgebra())
     return;

   ConstantFP *CFP = dyn_cast<ConstantFP>(I->getOperand(0));
   if (CFP && CFP->isExactlyValue(0.5)) {
     Y = I->getOperand(1);
     return;
   }
   CFP = dyn_cast<ConstantFP>(I->getOperand(1));
   if (CFP && CFP->isExactlyValue(0.5))
     Y = I->getOperand(0);
}

/// Helper function of InstCombiner::visitFMul(BinaryOperator(). It returns
/// true iff the given value is FMul or FDiv with one and only one operand
/// being a normal constant (i.e. not Zero/NaN/Infinity).
static bool isFMulOrFDivWithConstant(Value *V) {
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I || (I->getOpcode() != Instruction::FMul &&
             I->getOpcode() != Instruction::FDiv))
    return false;

  ConstantFP *C0 = dyn_cast<ConstantFP>(I->getOperand(0));
  ConstantFP *C1 = dyn_cast<ConstantFP>(I->getOperand(1));

  if (C0 && C1)
    return false;

  return (C0 && C0->getValueAPF().isFiniteNonZero()) ||
         (C1 && C1->getValueAPF().isFiniteNonZero());
}

static bool isNormalFp(const ConstantFP *C) {
  const APFloat &Flt = C->getValueAPF();
  return Flt.isNormal();
}

/// foldFMulConst() is a helper routine of InstCombiner::visitFMul().
/// The input \p FMulOrDiv is a FMul/FDiv with one and only one operand
/// being a constant (i.e. isFMulOrFDivWithConstant(FMulOrDiv) == true).
/// This function is to simplify "FMulOrDiv * C" and returns the
/// resulting expression. Note that this function could return NULL in
/// case the constants cannot be folded into a normal floating-point.
///
Value *InstCombiner::foldFMulConst(Instruction *FMulOrDiv, ConstantFP *C,
                                   Instruction *InsertBefore) {
  assert(isFMulOrFDivWithConstant(FMulOrDiv) && "V is invalid");

  Value *Opnd0 = FMulOrDiv->getOperand(0);
  Value *Opnd1 = FMulOrDiv->getOperand(1);

  ConstantFP *C0 = dyn_cast<ConstantFP>(Opnd0);
  ConstantFP *C1 = dyn_cast<ConstantFP>(Opnd1);

  BinaryOperator *R = 0;

  // (X * C0) * C => X * (C0*C)
  if (FMulOrDiv->getOpcode() == Instruction::FMul) {
    Constant *F = ConstantExpr::getFMul(C1 ? C1 : C0, C);
    if (isNormalFp(cast<ConstantFP>(F)))
      R = BinaryOperator::CreateFMul(C1 ? Opnd0 : Opnd1, F);
  } else {
    if (C0) {
      // (C0 / X) * C => (C0 * C) / X
      if (FMulOrDiv->hasOneUse()) {
        // It would otherwise introduce another div.
        ConstantFP *F = cast<ConstantFP>(ConstantExpr::getFMul(C0, C));
        if (isNormalFp(F))
          R = BinaryOperator::CreateFDiv(F, Opnd1);
      }
    } else {
      // (X / C1) * C => X * (C/C1) if C/C1 is not a denormal
      ConstantFP *F = cast<ConstantFP>(ConstantExpr::getFDiv(C, C1));
      if (isNormalFp(F)) {
        R = BinaryOperator::CreateFMul(Opnd0, F);
      } else {
        // (X / C1) * C => X / (C1/C)
        Constant *F = ConstantExpr::getFDiv(C1, C);
        if (isNormalFp(cast<ConstantFP>(F)))
          R = BinaryOperator::CreateFDiv(Opnd0, F);
      }
    }
  }

  if (R) {
    R->setHasUnsafeAlgebra(true);
    InsertNewInstWith(R, *InsertBefore);
  }

  return R;
}

Instruction *InstCombiner::visitFMul(BinaryOperator &I) {
  bool Changed = SimplifyAssociativeOrCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (isa<Constant>(Op0))
    std::swap(Op0, Op1);

  if (Value *V = SimplifyFMulInst(Op0, Op1, I.getFastMathFlags(), TD))
    return ReplaceInstUsesWith(I, V);

  bool AllowReassociate = I.hasUnsafeAlgebra();

  // Simplify mul instructions with a constant RHS.
  if (isa<Constant>(Op1)) {
    // Try to fold constant mul into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;

    if (isa<PHINode>(Op0))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;

    ConstantFP *C = dyn_cast<ConstantFP>(Op1);

    // (fmul X, -1.0) --> (fsub -0.0, X)
    if (C && C->isExactlyValue(-1.0)) {
      Instruction *RI = BinaryOperator::CreateFSub(
        ConstantFP::getNegativeZero(C->getType()),
        Op0);
      RI->copyFastMathFlags(&I);
      return RI;
    }

    if (C && AllowReassociate && C->getValueAPF().isFiniteNonZero()) {
      // Let MDC denote an expression in one of these forms:
      // X * C, C/X, X/C, where C is a constant.
      //
      // Try to simplify "MDC * Constant"
      if (isFMulOrFDivWithConstant(Op0)) {
        Value *V = foldFMulConst(cast<Instruction>(Op0), C, &I);
        if (V)
          return ReplaceInstUsesWith(I, V);
      }

      // (MDC +/- C1) * C => (MDC * C) +/- (C1 * C)
      Instruction *FAddSub = dyn_cast<Instruction>(Op0);
      if (FAddSub &&
          (FAddSub->getOpcode() == Instruction::FAdd ||
           FAddSub->getOpcode() == Instruction::FSub)) {
        Value *Opnd0 = FAddSub->getOperand(0);
        Value *Opnd1 = FAddSub->getOperand(1);
        ConstantFP *C0 = dyn_cast<ConstantFP>(Opnd0);
        ConstantFP *C1 = dyn_cast<ConstantFP>(Opnd1);
        bool Swap = false;
        if (C0) {
          std::swap(C0, C1);
          std::swap(Opnd0, Opnd1);
          Swap = true;
        }

        if (C1 && C1->getValueAPF().isFiniteNonZero() &&
            isFMulOrFDivWithConstant(Opnd0)) {
          Value *M1 = ConstantExpr::getFMul(C1, C);
          Value *M0 = isNormalFp(cast<ConstantFP>(M1)) ?
                      foldFMulConst(cast<Instruction>(Opnd0), C, &I) :
                      0;
          if (M0 && M1) {
            if (Swap && FAddSub->getOpcode() == Instruction::FSub)
              std::swap(M0, M1);

            Instruction *RI = (FAddSub->getOpcode() == Instruction::FAdd)
                                  ? BinaryOperator::CreateFAdd(M0, M1)
                                  : BinaryOperator::CreateFSub(M0, M1);
            RI->copyFastMathFlags(&I);
            return RI;
          }
        }
      }
    }
  }


  // Under unsafe algebra do:
  // X * log2(0.5*Y) = X*log2(Y) - X
  if (I.hasUnsafeAlgebra()) {
    Value *OpX = NULL;
    Value *OpY = NULL;
    IntrinsicInst *Log2;
    detectLog2OfHalf(Op0, OpY, Log2);
    if (OpY) {
      OpX = Op1;
    } else {
      detectLog2OfHalf(Op1, OpY, Log2);
      if (OpY) {
        OpX = Op0;
      }
    }
    // if pattern detected emit alternate sequence
    if (OpX && OpY) {
      BuilderTy::FastMathFlagGuard Guard(*Builder);
      Builder->SetFastMathFlags(Log2->getFastMathFlags());
      Log2->setArgOperand(0, OpY);
      Value *FMulVal = Builder->CreateFMul(OpX, Log2);
      Value *FSub = Builder->CreateFSub(FMulVal, OpX);
      FSub->takeName(&I);
      return ReplaceInstUsesWith(I, FSub);
    }
  }

  // Handle symmetric situation in a 2-iteration loop
  Value *Opnd0 = Op0;
  Value *Opnd1 = Op1;
  for (int i = 0; i < 2; i++) {
    bool IgnoreZeroSign = I.hasNoSignedZeros();
    if (BinaryOperator::isFNeg(Opnd0, IgnoreZeroSign)) {
      BuilderTy::FastMathFlagGuard Guard(*Builder);
      Builder->SetFastMathFlags(I.getFastMathFlags());

      Value *N0 = dyn_castFNegVal(Opnd0, IgnoreZeroSign);
      Value *N1 = dyn_castFNegVal(Opnd1, IgnoreZeroSign);

      // -X * -Y => X*Y
      if (N1)
        return BinaryOperator::CreateFMul(N0, N1);

      if (Opnd0->hasOneUse()) {
        // -X * Y => -(X*Y) (Promote negation as high as possible)
        Value *T = Builder->CreateFMul(N0, Opnd1);
        Value *Neg = Builder->CreateFNeg(T);
        Neg->takeName(&I);
        return ReplaceInstUsesWith(I, Neg);
      }
    }

    // (X*Y) * X => (X*X) * Y where Y != X
    //  The purpose is two-fold:
    //   1) to form a power expression (of X).
    //   2) potentially shorten the critical path: After transformation, the
    //  latency of the instruction Y is amortized by the expression of X*X,
    //  and therefore Y is in a "less critical" position compared to what it
    //  was before the transformation.
    //
    if (AllowReassociate) {
      Value *Opnd0_0, *Opnd0_1;
      if (Opnd0->hasOneUse() &&
          match(Opnd0, m_FMul(m_Value(Opnd0_0), m_Value(Opnd0_1)))) {
        Value *Y = 0;
        if (Opnd0_0 == Opnd1 && Opnd0_1 != Opnd1)
          Y = Opnd0_1;
        else if (Opnd0_1 == Opnd1 && Opnd0_0 != Opnd1)
          Y = Opnd0_0;

        if (Y) {
          BuilderTy::FastMathFlagGuard Guard(*Builder);
          Builder->SetFastMathFlags(I.getFastMathFlags());
          Value *T = Builder->CreateFMul(Opnd1, Opnd1);

          Value *R = Builder->CreateFMul(T, Y);
          R->takeName(&I);
          return ReplaceInstUsesWith(I, R);
        }
      }
    }

    // B * (uitofp i1 C) -> select C, B, 0
    if (I.hasNoNaNs() && I.hasNoInfs() && I.hasNoSignedZeros()) {
      Value *LHS = Op0, *RHS = Op1;
      Value *B, *C;
      if (!match(RHS, m_UIToFP(m_Value(C))))
        std::swap(LHS, RHS);

      if (match(RHS, m_UIToFP(m_Value(C))) && C->getType()->isIntegerTy(1)) {
        B = LHS;
        Value *Zero = ConstantFP::getNegativeZero(B->getType());
        return SelectInst::Create(C, B, Zero);
      }
    }

    // A * (1 - uitofp i1 C) -> select C, 0, A
    if (I.hasNoNaNs() && I.hasNoInfs() && I.hasNoSignedZeros()) {
      Value *LHS = Op0, *RHS = Op1;
      Value *A, *C;
      if (!match(RHS, m_FSub(m_FPOne(), m_UIToFP(m_Value(C)))))
        std::swap(LHS, RHS);

      if (match(RHS, m_FSub(m_FPOne(), m_UIToFP(m_Value(C)))) &&
          C->getType()->isIntegerTy(1)) {
        A = LHS;
        Value *Zero = ConstantFP::getNegativeZero(A->getType());
        return SelectInst::Create(C, Zero, A);
      }
    }

    if (!isa<Constant>(Op1))
      std::swap(Opnd0, Opnd1);
    else
      break;
  }

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
        *I = Builder->getInt1(NonNullOperand == 1);
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


/// This function implements the transforms common to both integer division
/// instructions (udiv and sdiv). It is called by the visitors to those integer
/// division instructions.
/// @brief Common integer divide transforms
Instruction *InstCombiner::commonIDivTransforms(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // The RHS is known non-zero.
  if (Value *V = simplifyValueKnownNonZero(I.getOperand(1), *this)) {
    I.setOperand(1, V);
    return &I;
  }

  // Handle cases involving: [su]div X, (select Cond, Y, Z)
  // This does not apply for fdiv.
  if (isa<SelectInst>(Op1) && SimplifyDivRemOfSelect(I))
    return &I;

  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    // (X / C1) / C2  -> X / (C1*C2)
    if (Instruction *LHS = dyn_cast<Instruction>(Op0))
      if (Instruction::BinaryOps(LHS->getOpcode()) == I.getOpcode())
        if (ConstantInt *LHSRHS = dyn_cast<ConstantInt>(LHS->getOperand(1))) {
          if (MultiplyOverflows(RHS, LHSRHS,
                                I.getOpcode()==Instruction::SDiv))
            return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
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

  // See if we can fold away this div instruction.
  if (SimplifyDemandedInstructionBits(I))
    return &I;

  // (X - (X rem Y)) / Y -> X / Y; usually originates as ((X / Y) * Y) / Y
  Value *X = 0, *Z = 0;
  if (match(Op0, m_Sub(m_Value(X), m_Value(Z)))) { // (X - Z) / Y; Y = Op1
    bool isSigned = I.getOpcode() == Instruction::SDiv;
    if ((isSigned && match(Z, m_SRem(m_Specific(X), m_Specific(Op1)))) ||
        (!isSigned && match(Z, m_URem(m_Specific(X), m_Specific(Op1)))))
      return BinaryOperator::Create(I.getOpcode(), X, Op1);
  }

  return 0;
}

/// dyn_castZExtVal - Checks if V is a zext or constant that can
/// be truncated to Ty without losing bits.
static Value *dyn_castZExtVal(Value *V, Type *Ty) {
  if (ZExtInst *Z = dyn_cast<ZExtInst>(V)) {
    if (Z->getSrcTy() == Ty)
      return Z->getOperand(0);
  } else if (ConstantInt *C = dyn_cast<ConstantInt>(V)) {
    if (C->getValue().getActiveBits() <= cast<IntegerType>(Ty)->getBitWidth())
      return ConstantExpr::getTrunc(C, Ty);
  }
  return 0;
}

namespace {
const unsigned MaxDepth = 6;
typedef Instruction *(*FoldUDivOperandCb)(Value *Op0, Value *Op1,
                                          const BinaryOperator &I,
                                          InstCombiner &IC);

/// \brief Used to maintain state for visitUDivOperand().
struct UDivFoldAction {
  FoldUDivOperandCb FoldAction; ///< Informs visitUDiv() how to fold this
                                ///< operand.  This can be zero if this action
                                ///< joins two actions together.

  Value *OperandToFold;         ///< Which operand to fold.
  union {
    Instruction *FoldResult;    ///< The instruction returned when FoldAction is
                                ///< invoked.

    size_t SelectLHSIdx;        ///< Stores the LHS action index if this action
                                ///< joins two actions together.
  };

  UDivFoldAction(FoldUDivOperandCb FA, Value *InputOperand)
      : FoldAction(FA), OperandToFold(InputOperand), FoldResult(0) {}
  UDivFoldAction(FoldUDivOperandCb FA, Value *InputOperand, size_t SLHS)
      : FoldAction(FA), OperandToFold(InputOperand), SelectLHSIdx(SLHS) {}
};
}

// X udiv 2^C -> X >> C
static Instruction *foldUDivPow2Cst(Value *Op0, Value *Op1,
                                    const BinaryOperator &I, InstCombiner &IC) {
  const APInt &C = cast<Constant>(Op1)->getUniqueInteger();
  BinaryOperator *LShr = BinaryOperator::CreateLShr(
      Op0, ConstantInt::get(Op0->getType(), C.logBase2()));
  if (I.isExact()) LShr->setIsExact();
  return LShr;
}

// X udiv C, where C >= signbit
static Instruction *foldUDivNegCst(Value *Op0, Value *Op1,
                                   const BinaryOperator &I, InstCombiner &IC) {
  Value *ICI = IC.Builder->CreateICmpULT(Op0, cast<ConstantInt>(Op1));

  return SelectInst::Create(ICI, Constant::getNullValue(I.getType()),
                            ConstantInt::get(I.getType(), 1));
}

// X udiv (C1 << N), where C1 is "1<<C2"  -->  X >> (N+C2)
static Instruction *foldUDivShl(Value *Op0, Value *Op1, const BinaryOperator &I,
                                InstCombiner &IC) {
  Instruction *ShiftLeft = cast<Instruction>(Op1);
  if (isa<ZExtInst>(ShiftLeft))
    ShiftLeft = cast<Instruction>(ShiftLeft->getOperand(0));

  const APInt &CI =
      cast<Constant>(ShiftLeft->getOperand(0))->getUniqueInteger();
  Value *N = ShiftLeft->getOperand(1);
  if (CI != 1)
    N = IC.Builder->CreateAdd(N, ConstantInt::get(N->getType(), CI.logBase2()));
  if (ZExtInst *Z = dyn_cast<ZExtInst>(Op1))
    N = IC.Builder->CreateZExt(N, Z->getDestTy());
  BinaryOperator *LShr = BinaryOperator::CreateLShr(Op0, N);
  if (I.isExact()) LShr->setIsExact();
  return LShr;
}

// \brief Recursively visits the possible right hand operands of a udiv
// instruction, seeing through select instructions, to determine if we can
// replace the udiv with something simpler.  If we find that an operand is not
// able to simplify the udiv, we abort the entire transformation.
static size_t visitUDivOperand(Value *Op0, Value *Op1, const BinaryOperator &I,
                               SmallVectorImpl<UDivFoldAction> &Actions,
                               unsigned Depth = 0) {
  // Check to see if this is an unsigned division with an exact power of 2,
  // if so, convert to a right shift.
  if (match(Op1, m_Power2())) {
    Actions.push_back(UDivFoldAction(foldUDivPow2Cst, Op1));
    return Actions.size();
  }

  if (ConstantInt *C = dyn_cast<ConstantInt>(Op1))
    // X udiv C, where C >= signbit
    if (C->getValue().isNegative()) {
      Actions.push_back(UDivFoldAction(foldUDivNegCst, C));
      return Actions.size();
    }

  // X udiv (C1 << N), where C1 is "1<<C2"  -->  X >> (N+C2)
  if (match(Op1, m_Shl(m_Power2(), m_Value())) ||
      match(Op1, m_ZExt(m_Shl(m_Power2(), m_Value())))) {
    Actions.push_back(UDivFoldAction(foldUDivShl, Op1));
    return Actions.size();
  }

  // The remaining tests are all recursive, so bail out if we hit the limit.
  if (Depth++ == MaxDepth)
    return 0;

  if (SelectInst *SI = dyn_cast<SelectInst>(Op1))
    if (size_t LHSIdx = visitUDivOperand(Op0, SI->getOperand(1), I, Actions))
      if (visitUDivOperand(Op0, SI->getOperand(2), I, Actions)) {
        Actions.push_back(UDivFoldAction((FoldUDivOperandCb)0, Op1, LHSIdx-1));
        return Actions.size();
      }

  return 0;
}

Instruction *InstCombiner::visitUDiv(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyUDivInst(Op0, Op1, TD))
    return ReplaceInstUsesWith(I, V);

  // Handle the integer div common cases
  if (Instruction *Common = commonIDivTransforms(I))
    return Common;

  // (x lshr C1) udiv C2 --> x udiv (C2 << C1)
  if (ConstantInt *C2 = dyn_cast<ConstantInt>(Op1)) {
    Value *X;
    ConstantInt *C1;
    if (match(Op0, m_LShr(m_Value(X), m_ConstantInt(C1)))) {
      APInt NC = C2->getValue().shl(C1->getLimitedValue(C1->getBitWidth()-1));
      return BinaryOperator::CreateUDiv(X, Builder->getInt(NC));
    }
  }

  // (zext A) udiv (zext B) --> zext (A udiv B)
  if (ZExtInst *ZOp0 = dyn_cast<ZExtInst>(Op0))
    if (Value *ZOp1 = dyn_castZExtVal(Op1, ZOp0->getSrcTy()))
      return new ZExtInst(Builder->CreateUDiv(ZOp0->getOperand(0), ZOp1, "div",
                                              I.isExact()),
                          I.getType());

  // (LHS udiv (select (select (...)))) -> (LHS >> (select (select (...))))
  SmallVector<UDivFoldAction, 6> UDivActions;
  if (visitUDivOperand(Op0, Op1, I, UDivActions))
    for (unsigned i = 0, e = UDivActions.size(); i != e; ++i) {
      FoldUDivOperandCb Action = UDivActions[i].FoldAction;
      Value *ActionOp1 = UDivActions[i].OperandToFold;
      Instruction *Inst;
      if (Action)
        Inst = Action(Op0, ActionOp1, I, *this);
      else {
        // This action joins two actions together.  The RHS of this action is
        // simply the last action we processed, we saved the LHS action index in
        // the joining action.
        size_t SelectRHSIdx = i - 1;
        Value *SelectRHS = UDivActions[SelectRHSIdx].FoldResult;
        size_t SelectLHSIdx = UDivActions[i].SelectLHSIdx;
        Value *SelectLHS = UDivActions[SelectLHSIdx].FoldResult;
        Inst = SelectInst::Create(cast<SelectInst>(ActionOp1)->getCondition(),
                                  SelectLHS, SelectRHS);
      }

      // If this is the last action to process, return it to the InstCombiner.
      // Otherwise, we insert it before the UDiv and record it so that we may
      // use it as part of a joining action (i.e., a SelectInst).
      if (e - i != 1) {
        Inst->insertBefore(&I);
        UDivActions[i].FoldResult = Inst;
      } else
        return Inst;
    }

  return 0;
}

Instruction *InstCombiner::visitSDiv(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifySDivInst(Op0, Op1, TD))
    return ReplaceInstUsesWith(I, V);

  // Handle the integer div common cases
  if (Instruction *Common = commonIDivTransforms(I))
    return Common;

  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    // sdiv X, -1 == -X
    if (RHS->isAllOnesValue())
      return BinaryOperator::CreateNeg(Op0);

    // sdiv X, C  -->  ashr exact X, log2(C)
    if (I.isExact() && RHS->getValue().isNonNegative() &&
        RHS->getValue().isPowerOf2()) {
      Value *ShAmt = llvm::ConstantInt::get(RHS->getType(),
                                            RHS->getValue().exactLogBase2());
      return BinaryOperator::CreateExactAShr(Op0, ShAmt, I.getName());
    }

    // -X/C  -->  X/-C  provided the negation doesn't overflow.
    if (SubOperator *Sub = dyn_cast<SubOperator>(Op0))
      if (match(Sub->getOperand(0), m_Zero()) && Sub->hasNoSignedWrap())
        return BinaryOperator::CreateSDiv(Sub->getOperand(1),
                                          ConstantExpr::getNeg(RHS));
  }

  // If the sign bits of both operands are zero (i.e. we can prove they are
  // unsigned inputs), turn this into a udiv.
  if (I.getType()->isIntegerTy()) {
    APInt Mask(APInt::getSignBit(I.getType()->getPrimitiveSizeInBits()));
    if (MaskedValueIsZero(Op0, Mask)) {
      if (MaskedValueIsZero(Op1, Mask)) {
        // X sdiv Y -> X udiv Y, iff X and Y don't have sign bit set
        return BinaryOperator::CreateUDiv(Op0, Op1, I.getName());
      }

      if (match(Op1, m_Shl(m_Power2(), m_Value()))) {
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

/// CvtFDivConstToReciprocal tries to convert X/C into X*1/C if C not a special
/// FP value and:
///    1) 1/C is exact, or
///    2) reciprocal is allowed.
/// If the conversion was successful, the simplified expression "X * 1/C" is
/// returned; otherwise, NULL is returned.
///
static Instruction *CvtFDivConstToReciprocal(Value *Dividend,
                                             ConstantFP *Divisor,
                                             bool AllowReciprocal) {
  const APFloat &FpVal = Divisor->getValueAPF();
  APFloat Reciprocal(FpVal.getSemantics());
  bool Cvt = FpVal.getExactInverse(&Reciprocal);

  if (!Cvt && AllowReciprocal && FpVal.isFiniteNonZero()) {
    Reciprocal = APFloat(FpVal.getSemantics(), 1.0f);
    (void)Reciprocal.divide(FpVal, APFloat::rmNearestTiesToEven);
    Cvt = !Reciprocal.isDenormal();
  }

  if (!Cvt)
    return 0;

  ConstantFP *R;
  R = ConstantFP::get(Dividend->getType()->getContext(), Reciprocal);
  return BinaryOperator::CreateFMul(Dividend, R);
}

Instruction *InstCombiner::visitFDiv(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyFDivInst(Op0, Op1, TD))
    return ReplaceInstUsesWith(I, V);

  if (isa<Constant>(Op0))
    if (SelectInst *SI = dyn_cast<SelectInst>(Op1))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;

  bool AllowReassociate = I.hasUnsafeAlgebra();
  bool AllowReciprocal = I.hasAllowReciprocal();

  if (ConstantFP *Op1C = dyn_cast<ConstantFP>(Op1)) {
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;

    if (AllowReassociate) {
      ConstantFP *C1 = 0;
      ConstantFP *C2 = Op1C;
      Value *X;
      Instruction *Res = 0;

      if (match(Op0, m_FMul(m_Value(X), m_ConstantFP(C1)))) {
        // (X*C1)/C2 => X * (C1/C2)
        //
        Constant *C = ConstantExpr::getFDiv(C1, C2);
        const APFloat &F = cast<ConstantFP>(C)->getValueAPF();
        if (F.isNormal())
          Res = BinaryOperator::CreateFMul(X, C);
      } else if (match(Op0, m_FDiv(m_Value(X), m_ConstantFP(C1)))) {
        // (X/C1)/C2 => X /(C2*C1) [=> X * 1/(C2*C1) if reciprocal is allowed]
        //
        Constant *C = ConstantExpr::getFMul(C1, C2);
        const APFloat &F = cast<ConstantFP>(C)->getValueAPF();
        if (F.isNormal()) {
          Res = CvtFDivConstToReciprocal(X, cast<ConstantFP>(C),
                                         AllowReciprocal);
          if (!Res)
            Res = BinaryOperator::CreateFDiv(X, C);
        }
      }

      if (Res) {
        Res->setFastMathFlags(I.getFastMathFlags());
        return Res;
      }
    }

    // X / C => X * 1/C
    if (Instruction *T = CvtFDivConstToReciprocal(Op0, Op1C, AllowReciprocal))
      return T;

    return 0;
  }

  if (AllowReassociate && isa<ConstantFP>(Op0)) {
    ConstantFP *C1 = cast<ConstantFP>(Op0), *C2;
    Constant *Fold = 0;
    Value *X;
    bool CreateDiv = true;

    // C1 / (X*C2) => (C1/C2) / X
    if (match(Op1, m_FMul(m_Value(X), m_ConstantFP(C2))))
      Fold = ConstantExpr::getFDiv(C1, C2);
    else if (match(Op1, m_FDiv(m_Value(X), m_ConstantFP(C2)))) {
      // C1 / (X/C2) => (C1*C2) / X
      Fold = ConstantExpr::getFMul(C1, C2);
    } else if (match(Op1, m_FDiv(m_ConstantFP(C2), m_Value(X)))) {
      // C1 / (C2/X) => (C1/C2) * X
      Fold = ConstantExpr::getFDiv(C1, C2);
      CreateDiv = false;
    }

    if (Fold) {
      const APFloat &FoldC = cast<ConstantFP>(Fold)->getValueAPF();
      if (FoldC.isNormal()) {
        Instruction *R = CreateDiv ?
                         BinaryOperator::CreateFDiv(Fold, X) :
                         BinaryOperator::CreateFMul(X, Fold);
        R->setFastMathFlags(I.getFastMathFlags());
        return R;
      }
    }
    return 0;
  }

  if (AllowReassociate) {
    Value *X, *Y;
    Value *NewInst = 0;
    Instruction *SimpR = 0;

    if (Op0->hasOneUse() && match(Op0, m_FDiv(m_Value(X), m_Value(Y)))) {
      // (X/Y) / Z => X / (Y*Z)
      //
      if (!isa<ConstantFP>(Y) || !isa<ConstantFP>(Op1)) {
        NewInst = Builder->CreateFMul(Y, Op1);
        SimpR = BinaryOperator::CreateFDiv(X, NewInst);
      }
    } else if (Op1->hasOneUse() && match(Op1, m_FDiv(m_Value(X), m_Value(Y)))) {
      // Z / (X/Y) => Z*Y / X
      //
      if (!isa<ConstantFP>(Y) || !isa<ConstantFP>(Op0)) {
        NewInst = Builder->CreateFMul(Op0, Y);
        SimpR = BinaryOperator::CreateFDiv(NewInst, X);
      }
    }

    if (NewInst) {
      if (Instruction *T = dyn_cast<Instruction>(NewInst))
        T->setDebugLoc(I.getDebugLoc());
      SimpR->setFastMathFlags(I.getFastMathFlags());
      return SimpR;
    }
  }

  return 0;
}

/// This function implements the transforms common to both integer remainder
/// instructions (urem and srem). It is called by the visitors to those integer
/// remainder instructions.
/// @brief Common integer remainder transforms
Instruction *InstCombiner::commonIRemTransforms(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // The RHS is known non-zero.
  if (Value *V = simplifyValueKnownNonZero(I.getOperand(1), *this)) {
    I.setOperand(1, V);
    return &I;
  }

  // Handle cases involving: rem X, (select Cond, Y, Z)
  if (isa<SelectInst>(Op1) && SimplifyDivRemOfSelect(I))
    return &I;

  if (isa<ConstantInt>(Op1)) {
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

  if (Value *V = SimplifyURemInst(Op0, Op1, TD))
    return ReplaceInstUsesWith(I, V);

  if (Instruction *common = commonIRemTransforms(I))
    return common;

  // (zext A) urem (zext B) --> zext (A urem B)
  if (ZExtInst *ZOp0 = dyn_cast<ZExtInst>(Op0))
    if (Value *ZOp1 = dyn_castZExtVal(Op1, ZOp0->getSrcTy()))
      return new ZExtInst(Builder->CreateURem(ZOp0->getOperand(0), ZOp1),
                          I.getType());

  // X urem Y -> X and Y-1, where Y is a power of 2,
  if (isKnownToBeAPowerOfTwo(Op1, /*OrZero*/true)) {
    Constant *N1 = Constant::getAllOnesValue(I.getType());
    Value *Add = Builder->CreateAdd(Op1, N1);
    return BinaryOperator::CreateAnd(Op0, Add);
  }

  // 1 urem X -> zext(X != 1)
  if (match(Op0, m_One())) {
    Value *Cmp = Builder->CreateICmpNE(Op1, Op0);
    Value *Ext = Builder->CreateZExt(Cmp, I.getType());
    return ReplaceInstUsesWith(I, Ext);
  }

  return 0;
}

Instruction *InstCombiner::visitSRem(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifySRemInst(Op0, Op1, TD))
    return ReplaceInstUsesWith(I, V);

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
  if (I.getType()->isIntegerTy()) {
    APInt Mask(APInt::getSignBit(I.getType()->getPrimitiveSizeInBits()));
    if (MaskedValueIsZero(Op1, Mask) && MaskedValueIsZero(Op0, Mask)) {
      // X srem Y -> X urem Y, iff X and Y don't have sign bit set
      return BinaryOperator::CreateURem(Op0, Op1, I.getName());
    }
  }

  // If it's a constant vector, flip any negative values positive.
  if (isa<ConstantVector>(Op1) || isa<ConstantDataVector>(Op1)) {
    Constant *C = cast<Constant>(Op1);
    unsigned VWidth = C->getType()->getVectorNumElements();

    bool hasNegative = false;
    bool hasMissing = false;
    for (unsigned i = 0; i != VWidth; ++i) {
      Constant *Elt = C->getAggregateElement(i);
      if (Elt == 0) {
        hasMissing = true;
        break;
      }

      if (ConstantInt *RHS = dyn_cast<ConstantInt>(Elt))
        if (RHS->isNegative())
          hasNegative = true;
    }

    if (hasNegative && !hasMissing) {
      SmallVector<Constant *, 16> Elts(VWidth);
      for (unsigned i = 0; i != VWidth; ++i) {
        Elts[i] = C->getAggregateElement(i);  // Handle undef, etc.
        if (ConstantInt *RHS = dyn_cast<ConstantInt>(Elts[i])) {
          if (RHS->isNegative())
            Elts[i] = cast<ConstantInt>(ConstantExpr::getNeg(RHS));
        }
      }

      Constant *NewRHSV = ConstantVector::get(Elts);
      if (NewRHSV != C) {  // Don't loop on -MININT
        Worklist.AddValue(I.getOperand(1));
        I.setOperand(1, NewRHSV);
        return &I;
      }
    }
  }

  return 0;
}

Instruction *InstCombiner::visitFRem(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyFRemInst(Op0, Op1, TD))
    return ReplaceInstUsesWith(I, V);

  // Handle cases involving: rem X, (select Cond, Y, Z)
  if (isa<SelectInst>(Op1) && SimplifyDivRemOfSelect(I))
    return &I;

  return 0;
}
