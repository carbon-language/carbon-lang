//===- InstCombineMulDivRem.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the visit functions for mul, fmul, sdiv, udiv, fdiv,
// srem, urem, frem.
//
//===----------------------------------------------------------------------===//

#include "InstCombineInternal.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Transforms/InstCombine/InstCombiner.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

#define DEBUG_TYPE "instcombine"
#include "llvm/Transforms/Utils/InstructionWorklist.h"

using namespace llvm;
using namespace PatternMatch;

/// The specific integer value is used in a context where it is known to be
/// non-zero.  If this allows us to simplify the computation, do so and return
/// the new operand, otherwise return null.
static Value *simplifyValueKnownNonZero(Value *V, InstCombinerImpl &IC,
                                        Instruction &CxtI) {
  // If V has multiple uses, then we would have to do more analysis to determine
  // if this is safe.  For example, the use could be in dynamically unreached
  // code.
  if (!V->hasOneUse()) return nullptr;

  bool MadeChange = false;

  // ((1 << A) >>u B) --> (1 << (A-B))
  // Because V cannot be zero, we know that B is less than A.
  Value *A = nullptr, *B = nullptr, *One = nullptr;
  if (match(V, m_LShr(m_OneUse(m_Shl(m_Value(One), m_Value(A))), m_Value(B))) &&
      match(One, m_One())) {
    A = IC.Builder.CreateSub(A, B);
    return IC.Builder.CreateShl(One, A);
  }

  // (PowerOfTwo >>u B) --> isExact since shifting out the result would make it
  // inexact.  Similarly for <<.
  BinaryOperator *I = dyn_cast<BinaryOperator>(V);
  if (I && I->isLogicalShift() &&
      IC.isKnownToBeAPowerOfTwo(I->getOperand(0), false, 0, &CxtI)) {
    // We know that this is an exact/nuw shift and that the input is a
    // non-zero context as well.
    if (Value *V2 = simplifyValueKnownNonZero(I->getOperand(0), IC, CxtI)) {
      IC.replaceOperand(*I, 0, V2);
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

  return MadeChange ? V : nullptr;
}

// TODO: This is a specific form of a much more general pattern.
//       We could detect a select with any binop identity constant, or we
//       could use SimplifyBinOp to see if either arm of the select reduces.
//       But that needs to be done carefully and/or while removing potential
//       reverse canonicalizations as in InstCombiner::foldSelectIntoOp().
static Value *foldMulSelectToNegate(BinaryOperator &I,
                                    InstCombiner::BuilderTy &Builder) {
  Value *Cond, *OtherOp;

  // mul (select Cond, 1, -1), OtherOp --> select Cond, OtherOp, -OtherOp
  // mul OtherOp, (select Cond, 1, -1) --> select Cond, OtherOp, -OtherOp
  if (match(&I, m_c_Mul(m_OneUse(m_Select(m_Value(Cond), m_One(), m_AllOnes())),
                        m_Value(OtherOp)))) {
    bool HasAnyNoWrap = I.hasNoSignedWrap() || I.hasNoUnsignedWrap();
    Value *Neg = Builder.CreateNeg(OtherOp, "", false, HasAnyNoWrap);
    return Builder.CreateSelect(Cond, OtherOp, Neg);
  }
  // mul (select Cond, -1, 1), OtherOp --> select Cond, -OtherOp, OtherOp
  // mul OtherOp, (select Cond, -1, 1) --> select Cond, -OtherOp, OtherOp
  if (match(&I, m_c_Mul(m_OneUse(m_Select(m_Value(Cond), m_AllOnes(), m_One())),
                        m_Value(OtherOp)))) {
    bool HasAnyNoWrap = I.hasNoSignedWrap() || I.hasNoUnsignedWrap();
    Value *Neg = Builder.CreateNeg(OtherOp, "", false, HasAnyNoWrap);
    return Builder.CreateSelect(Cond, Neg, OtherOp);
  }

  // fmul (select Cond, 1.0, -1.0), OtherOp --> select Cond, OtherOp, -OtherOp
  // fmul OtherOp, (select Cond, 1.0, -1.0) --> select Cond, OtherOp, -OtherOp
  if (match(&I, m_c_FMul(m_OneUse(m_Select(m_Value(Cond), m_SpecificFP(1.0),
                                           m_SpecificFP(-1.0))),
                         m_Value(OtherOp)))) {
    IRBuilder<>::FastMathFlagGuard FMFGuard(Builder);
    Builder.setFastMathFlags(I.getFastMathFlags());
    return Builder.CreateSelect(Cond, OtherOp, Builder.CreateFNeg(OtherOp));
  }

  // fmul (select Cond, -1.0, 1.0), OtherOp --> select Cond, -OtherOp, OtherOp
  // fmul OtherOp, (select Cond, -1.0, 1.0) --> select Cond, -OtherOp, OtherOp
  if (match(&I, m_c_FMul(m_OneUse(m_Select(m_Value(Cond), m_SpecificFP(-1.0),
                                           m_SpecificFP(1.0))),
                         m_Value(OtherOp)))) {
    IRBuilder<>::FastMathFlagGuard FMFGuard(Builder);
    Builder.setFastMathFlags(I.getFastMathFlags());
    return Builder.CreateSelect(Cond, Builder.CreateFNeg(OtherOp), OtherOp);
  }

  return nullptr;
}

Instruction *InstCombinerImpl::visitMul(BinaryOperator &I) {
  if (Value *V = SimplifyMulInst(I.getOperand(0), I.getOperand(1),
                                 SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  if (SimplifyAssociativeOrCommutative(I))
    return &I;

  if (Instruction *X = foldVectorBinop(I))
    return X;

  if (Value *V = SimplifyUsingDistributiveLaws(I))
    return replaceInstUsesWith(I, V);

  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  unsigned BitWidth = I.getType()->getScalarSizeInBits();

  // X * -1 == 0 - X
  if (match(Op1, m_AllOnes())) {
    BinaryOperator *BO = BinaryOperator::CreateNeg(Op0, I.getName());
    if (I.hasNoSignedWrap())
      BO->setHasNoSignedWrap();
    return BO;
  }

  // Also allow combining multiply instructions on vectors.
  {
    Value *NewOp;
    Constant *C1, *C2;
    const APInt *IVal;
    if (match(&I, m_Mul(m_Shl(m_Value(NewOp), m_Constant(C2)),
                        m_Constant(C1))) &&
        match(C1, m_APInt(IVal))) {
      // ((X << C2)*C1) == (X * (C1 << C2))
      Constant *Shl = ConstantExpr::getShl(C1, C2);
      BinaryOperator *Mul = cast<BinaryOperator>(I.getOperand(0));
      BinaryOperator *BO = BinaryOperator::CreateMul(NewOp, Shl);
      if (I.hasNoUnsignedWrap() && Mul->hasNoUnsignedWrap())
        BO->setHasNoUnsignedWrap();
      if (I.hasNoSignedWrap() && Mul->hasNoSignedWrap() &&
          Shl->isNotMinSignedValue())
        BO->setHasNoSignedWrap();
      return BO;
    }

    if (match(&I, m_Mul(m_Value(NewOp), m_Constant(C1)))) {
      // Replace X*(2^C) with X << C, where C is either a scalar or a vector.
      if (Constant *NewCst = ConstantExpr::getExactLogBase2(C1)) {
        BinaryOperator *Shl = BinaryOperator::CreateShl(NewOp, NewCst);

        if (I.hasNoUnsignedWrap())
          Shl->setHasNoUnsignedWrap();
        if (I.hasNoSignedWrap()) {
          const APInt *V;
          if (match(NewCst, m_APInt(V)) && *V != V->getBitWidth() - 1)
            Shl->setHasNoSignedWrap();
        }

        return Shl;
      }
    }
  }

  if (Op0->hasOneUse() && match(Op1, m_NegatedPower2())) {
    // Interpret  X * (-1<<C)  as  (-X) * (1<<C)  and try to sink the negation.
    // The "* (1<<C)" thus becomes a potential shifting opportunity.
    if (Value *NegOp0 = Negator::Negate(/*IsNegation*/ true, Op0, *this))
      return BinaryOperator::CreateMul(
          NegOp0, ConstantExpr::getNeg(cast<Constant>(Op1)), I.getName());
  }

  if (Instruction *FoldedMul = foldBinOpIntoSelectOrPhi(I))
    return FoldedMul;

  if (Value *FoldedMul = foldMulSelectToNegate(I, Builder))
    return replaceInstUsesWith(I, FoldedMul);

  // Simplify mul instructions with a constant RHS.
  if (isa<Constant>(Op1)) {
    // Canonicalize (X+C1)*CI -> X*CI+C1*CI.
    Value *X;
    Constant *C1;
    if (match(Op0, m_OneUse(m_Add(m_Value(X), m_Constant(C1))))) {
      Value *Mul = Builder.CreateMul(C1, Op1);
      // Only go forward with the transform if C1*CI simplifies to a tidier
      // constant.
      if (!match(Mul, m_Mul(m_Value(), m_Value())))
        return BinaryOperator::CreateAdd(Builder.CreateMul(X, Op1), Mul);
    }
  }

  // abs(X) * abs(X) -> X * X
  // nabs(X) * nabs(X) -> X * X
  if (Op0 == Op1) {
    Value *X, *Y;
    SelectPatternFlavor SPF = matchSelectPattern(Op0, X, Y).Flavor;
    if (SPF == SPF_ABS || SPF == SPF_NABS)
      return BinaryOperator::CreateMul(X, X);

    if (match(Op0, m_Intrinsic<Intrinsic::abs>(m_Value(X))))
      return BinaryOperator::CreateMul(X, X);
  }

  // -X * C --> X * -C
  Value *X, *Y;
  Constant *Op1C;
  if (match(Op0, m_Neg(m_Value(X))) && match(Op1, m_Constant(Op1C)))
    return BinaryOperator::CreateMul(X, ConstantExpr::getNeg(Op1C));

  // -X * -Y --> X * Y
  if (match(Op0, m_Neg(m_Value(X))) && match(Op1, m_Neg(m_Value(Y)))) {
    auto *NewMul = BinaryOperator::CreateMul(X, Y);
    if (I.hasNoSignedWrap() &&
        cast<OverflowingBinaryOperator>(Op0)->hasNoSignedWrap() &&
        cast<OverflowingBinaryOperator>(Op1)->hasNoSignedWrap())
      NewMul->setHasNoSignedWrap();
    return NewMul;
  }

  // -X * Y --> -(X * Y)
  // X * -Y --> -(X * Y)
  if (match(&I, m_c_Mul(m_OneUse(m_Neg(m_Value(X))), m_Value(Y))))
    return BinaryOperator::CreateNeg(Builder.CreateMul(X, Y));

  // (X / Y) *  Y = X - (X % Y)
  // (X / Y) * -Y = (X % Y) - X
  {
    Value *Y = Op1;
    BinaryOperator *Div = dyn_cast<BinaryOperator>(Op0);
    if (!Div || (Div->getOpcode() != Instruction::UDiv &&
                 Div->getOpcode() != Instruction::SDiv)) {
      Y = Op0;
      Div = dyn_cast<BinaryOperator>(Op1);
    }
    Value *Neg = dyn_castNegVal(Y);
    if (Div && Div->hasOneUse() &&
        (Div->getOperand(1) == Y || Div->getOperand(1) == Neg) &&
        (Div->getOpcode() == Instruction::UDiv ||
         Div->getOpcode() == Instruction::SDiv)) {
      Value *X = Div->getOperand(0), *DivOp1 = Div->getOperand(1);

      // If the division is exact, X % Y is zero, so we end up with X or -X.
      if (Div->isExact()) {
        if (DivOp1 == Y)
          return replaceInstUsesWith(I, X);
        return BinaryOperator::CreateNeg(X);
      }

      auto RemOpc = Div->getOpcode() == Instruction::UDiv ? Instruction::URem
                                                          : Instruction::SRem;
      Value *Rem = Builder.CreateBinOp(RemOpc, X, DivOp1);
      if (DivOp1 == Y)
        return BinaryOperator::CreateSub(X, Rem);
      return BinaryOperator::CreateSub(Rem, X);
    }
  }

  /// i1 mul -> i1 and.
  if (I.getType()->isIntOrIntVectorTy(1))
    return BinaryOperator::CreateAnd(Op0, Op1);

  // X*(1 << Y) --> X << Y
  // (1 << Y)*X --> X << Y
  {
    Value *Y;
    BinaryOperator *BO = nullptr;
    bool ShlNSW = false;
    if (match(Op0, m_Shl(m_One(), m_Value(Y)))) {
      BO = BinaryOperator::CreateShl(Op1, Y);
      ShlNSW = cast<ShlOperator>(Op0)->hasNoSignedWrap();
    } else if (match(Op1, m_Shl(m_One(), m_Value(Y)))) {
      BO = BinaryOperator::CreateShl(Op0, Y);
      ShlNSW = cast<ShlOperator>(Op1)->hasNoSignedWrap();
    }
    if (BO) {
      if (I.hasNoUnsignedWrap())
        BO->setHasNoUnsignedWrap();
      if (I.hasNoSignedWrap() && ShlNSW)
        BO->setHasNoSignedWrap();
      return BO;
    }
  }

  // (zext bool X) * (zext bool Y) --> zext (and X, Y)
  // (sext bool X) * (sext bool Y) --> zext (and X, Y)
  // Note: -1 * -1 == 1 * 1 == 1 (if the extends match, the result is the same)
  if (((match(Op0, m_ZExt(m_Value(X))) && match(Op1, m_ZExt(m_Value(Y)))) ||
       (match(Op0, m_SExt(m_Value(X))) && match(Op1, m_SExt(m_Value(Y))))) &&
      X->getType()->isIntOrIntVectorTy(1) && X->getType() == Y->getType() &&
      (Op0->hasOneUse() || Op1->hasOneUse() || X == Y)) {
    Value *And = Builder.CreateAnd(X, Y, "mulbool");
    return CastInst::Create(Instruction::ZExt, And, I.getType());
  }
  // (sext bool X) * (zext bool Y) --> sext (and X, Y)
  // (zext bool X) * (sext bool Y) --> sext (and X, Y)
  // Note: -1 * 1 == 1 * -1  == -1
  if (((match(Op0, m_SExt(m_Value(X))) && match(Op1, m_ZExt(m_Value(Y)))) ||
       (match(Op0, m_ZExt(m_Value(X))) && match(Op1, m_SExt(m_Value(Y))))) &&
      X->getType()->isIntOrIntVectorTy(1) && X->getType() == Y->getType() &&
      (Op0->hasOneUse() || Op1->hasOneUse())) {
    Value *And = Builder.CreateAnd(X, Y, "mulbool");
    return CastInst::Create(Instruction::SExt, And, I.getType());
  }

  // (bool X) * Y --> X ? Y : 0
  // Y * (bool X) --> X ? Y : 0
  if (match(Op0, m_ZExt(m_Value(X))) && X->getType()->isIntOrIntVectorTy(1))
    return SelectInst::Create(X, Op1, ConstantInt::get(I.getType(), 0));
  if (match(Op1, m_ZExt(m_Value(X))) && X->getType()->isIntOrIntVectorTy(1))
    return SelectInst::Create(X, Op0, ConstantInt::get(I.getType(), 0));

  // (lshr X, 31) * Y --> (ashr X, 31) & Y
  // Y * (lshr X, 31) --> (ashr X, 31) & Y
  // TODO: We are not checking one-use because the elimination of the multiply
  //       is better for analysis?
  // TODO: Should we canonicalize to '(X < 0) ? Y : 0' instead? That would be
  //       more similar to what we're doing above.
  const APInt *C;
  if (match(Op0, m_LShr(m_Value(X), m_APInt(C))) && *C == C->getBitWidth() - 1)
    return BinaryOperator::CreateAnd(Builder.CreateAShr(X, *C), Op1);
  if (match(Op1, m_LShr(m_Value(X), m_APInt(C))) && *C == C->getBitWidth() - 1)
    return BinaryOperator::CreateAnd(Builder.CreateAShr(X, *C), Op0);

  // ((ashr X, 31) | 1) * X --> abs(X)
  // X * ((ashr X, 31) | 1) --> abs(X)
  if (match(&I, m_c_BinOp(m_Or(m_AShr(m_Value(X),
                                    m_SpecificIntAllowUndef(BitWidth - 1)),
                             m_One()),
                        m_Deferred(X)))) {
    Value *Abs = Builder.CreateBinaryIntrinsic(
        Intrinsic::abs, X,
        ConstantInt::getBool(I.getContext(), I.hasNoSignedWrap()));
    Abs->takeName(&I);
    return replaceInstUsesWith(I, Abs);
  }

  if (Instruction *Ext = narrowMathIfNoOverflow(I))
    return Ext;

  bool Changed = false;
  if (!I.hasNoSignedWrap() && willNotOverflowSignedMul(Op0, Op1, I)) {
    Changed = true;
    I.setHasNoSignedWrap(true);
  }

  if (!I.hasNoUnsignedWrap() && willNotOverflowUnsignedMul(Op0, Op1, I)) {
    Changed = true;
    I.setHasNoUnsignedWrap(true);
  }

  return Changed ? &I : nullptr;
}

Instruction *InstCombinerImpl::foldFPSignBitOps(BinaryOperator &I) {
  BinaryOperator::BinaryOps Opcode = I.getOpcode();
  assert((Opcode == Instruction::FMul || Opcode == Instruction::FDiv) &&
         "Expected fmul or fdiv");

  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  Value *X, *Y;

  // -X * -Y --> X * Y
  // -X / -Y --> X / Y
  if (match(Op0, m_FNeg(m_Value(X))) && match(Op1, m_FNeg(m_Value(Y))))
    return BinaryOperator::CreateWithCopiedFlags(Opcode, X, Y, &I);

  // fabs(X) * fabs(X) -> X * X
  // fabs(X) / fabs(X) -> X / X
  if (Op0 == Op1 && match(Op0, m_FAbs(m_Value(X))))
    return BinaryOperator::CreateWithCopiedFlags(Opcode, X, X, &I);

  // fabs(X) * fabs(Y) --> fabs(X * Y)
  // fabs(X) / fabs(Y) --> fabs(X / Y)
  if (match(Op0, m_FAbs(m_Value(X))) && match(Op1, m_FAbs(m_Value(Y))) &&
      (Op0->hasOneUse() || Op1->hasOneUse())) {
    IRBuilder<>::FastMathFlagGuard FMFGuard(Builder);
    Builder.setFastMathFlags(I.getFastMathFlags());
    Value *XY = Builder.CreateBinOp(Opcode, X, Y);
    Value *Fabs = Builder.CreateUnaryIntrinsic(Intrinsic::fabs, XY);
    Fabs->takeName(&I);
    return replaceInstUsesWith(I, Fabs);
  }

  return nullptr;
}

Instruction *InstCombinerImpl::visitFMul(BinaryOperator &I) {
  if (Value *V = SimplifyFMulInst(I.getOperand(0), I.getOperand(1),
                                  I.getFastMathFlags(),
                                  SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  if (SimplifyAssociativeOrCommutative(I))
    return &I;

  if (Instruction *X = foldVectorBinop(I))
    return X;

  if (Instruction *FoldedMul = foldBinOpIntoSelectOrPhi(I))
    return FoldedMul;

  if (Value *FoldedMul = foldMulSelectToNegate(I, Builder))
    return replaceInstUsesWith(I, FoldedMul);

  if (Instruction *R = foldFPSignBitOps(I))
    return R;

  // X * -1.0 --> -X
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  if (match(Op1, m_SpecificFP(-1.0)))
    return UnaryOperator::CreateFNegFMF(Op0, &I);

  // -X * C --> X * -C
  Value *X, *Y;
  Constant *C;
  if (match(Op0, m_FNeg(m_Value(X))) && match(Op1, m_Constant(C)))
    return BinaryOperator::CreateFMulFMF(X, ConstantExpr::getFNeg(C), &I);

  // (select A, B, C) * (select A, D, E) --> select A, (B*D), (C*E)
  if (Value *V = SimplifySelectsFeedingBinaryOp(I, Op0, Op1))
    return replaceInstUsesWith(I, V);

  if (I.hasAllowReassoc()) {
    // Reassociate constant RHS with another constant to form constant
    // expression.
    if (match(Op1, m_Constant(C)) && C->isFiniteNonZeroFP()) {
      Constant *C1;
      if (match(Op0, m_OneUse(m_FDiv(m_Constant(C1), m_Value(X))))) {
        // (C1 / X) * C --> (C * C1) / X
        Constant *CC1 = ConstantExpr::getFMul(C, C1);
        if (CC1->isNormalFP())
          return BinaryOperator::CreateFDivFMF(CC1, X, &I);
      }
      if (match(Op0, m_FDiv(m_Value(X), m_Constant(C1)))) {
        // (X / C1) * C --> X * (C / C1)
        Constant *CDivC1 = ConstantExpr::getFDiv(C, C1);
        if (CDivC1->isNormalFP())
          return BinaryOperator::CreateFMulFMF(X, CDivC1, &I);

        // If the constant was a denormal, try reassociating differently.
        // (X / C1) * C --> X / (C1 / C)
        Constant *C1DivC = ConstantExpr::getFDiv(C1, C);
        if (Op0->hasOneUse() && C1DivC->isNormalFP())
          return BinaryOperator::CreateFDivFMF(X, C1DivC, &I);
      }

      // We do not need to match 'fadd C, X' and 'fsub X, C' because they are
      // canonicalized to 'fadd X, C'. Distributing the multiply may allow
      // further folds and (X * C) + C2 is 'fma'.
      if (match(Op0, m_OneUse(m_FAdd(m_Value(X), m_Constant(C1))))) {
        // (X + C1) * C --> (X * C) + (C * C1)
        Constant *CC1 = ConstantExpr::getFMul(C, C1);
        Value *XC = Builder.CreateFMulFMF(X, C, &I);
        return BinaryOperator::CreateFAddFMF(XC, CC1, &I);
      }
      if (match(Op0, m_OneUse(m_FSub(m_Constant(C1), m_Value(X))))) {
        // (C1 - X) * C --> (C * C1) - (X * C)
        Constant *CC1 = ConstantExpr::getFMul(C, C1);
        Value *XC = Builder.CreateFMulFMF(X, C, &I);
        return BinaryOperator::CreateFSubFMF(CC1, XC, &I);
      }
    }

    Value *Z;
    if (match(&I, m_c_FMul(m_OneUse(m_FDiv(m_Value(X), m_Value(Y))),
                           m_Value(Z)))) {
      // Sink division: (X / Y) * Z --> (X * Z) / Y
      Value *NewFMul = Builder.CreateFMulFMF(X, Z, &I);
      return BinaryOperator::CreateFDivFMF(NewFMul, Y, &I);
    }

    // sqrt(X) * sqrt(Y) -> sqrt(X * Y)
    // nnan disallows the possibility of returning a number if both operands are
    // negative (in that case, we should return NaN).
    if (I.hasNoNaNs() &&
        match(Op0, m_OneUse(m_Intrinsic<Intrinsic::sqrt>(m_Value(X)))) &&
        match(Op1, m_OneUse(m_Intrinsic<Intrinsic::sqrt>(m_Value(Y))))) {
      Value *XY = Builder.CreateFMulFMF(X, Y, &I);
      Value *Sqrt = Builder.CreateUnaryIntrinsic(Intrinsic::sqrt, XY, &I);
      return replaceInstUsesWith(I, Sqrt);
    }

    // The following transforms are done irrespective of the number of uses
    // for the expression "1.0/sqrt(X)".
    //  1) 1.0/sqrt(X) * X -> X/sqrt(X)
    //  2) X * 1.0/sqrt(X) -> X/sqrt(X)
    // We always expect the backend to reduce X/sqrt(X) to sqrt(X), if it
    // has the necessary (reassoc) fast-math-flags.
    if (I.hasNoSignedZeros() &&
        match(Op0, (m_FDiv(m_SpecificFP(1.0), m_Value(Y)))) &&
        match(Y, m_Intrinsic<Intrinsic::sqrt>(m_Value(X))) && Op1 == X)
      return BinaryOperator::CreateFDivFMF(X, Y, &I);
    if (I.hasNoSignedZeros() &&
        match(Op1, (m_FDiv(m_SpecificFP(1.0), m_Value(Y)))) &&
        match(Y, m_Intrinsic<Intrinsic::sqrt>(m_Value(X))) && Op0 == X)
      return BinaryOperator::CreateFDivFMF(X, Y, &I);

    // Like the similar transform in instsimplify, this requires 'nsz' because
    // sqrt(-0.0) = -0.0, and -0.0 * -0.0 does not simplify to -0.0.
    if (I.hasNoNaNs() && I.hasNoSignedZeros() && Op0 == Op1 &&
        Op0->hasNUses(2)) {
      // Peek through fdiv to find squaring of square root:
      // (X / sqrt(Y)) * (X / sqrt(Y)) --> (X * X) / Y
      if (match(Op0, m_FDiv(m_Value(X),
                            m_Intrinsic<Intrinsic::sqrt>(m_Value(Y))))) {
        Value *XX = Builder.CreateFMulFMF(X, X, &I);
        return BinaryOperator::CreateFDivFMF(XX, Y, &I);
      }
      // (sqrt(Y) / X) * (sqrt(Y) / X) --> Y / (X * X)
      if (match(Op0, m_FDiv(m_Intrinsic<Intrinsic::sqrt>(m_Value(Y)),
                            m_Value(X)))) {
        Value *XX = Builder.CreateFMulFMF(X, X, &I);
        return BinaryOperator::CreateFDivFMF(Y, XX, &I);
      }
    }

    if (I.isOnlyUserOfAnyOperand()) {
      // pow(x, y) * pow(x, z) -> pow(x, y + z)
      if (match(Op0, m_Intrinsic<Intrinsic::pow>(m_Value(X), m_Value(Y))) &&
          match(Op1, m_Intrinsic<Intrinsic::pow>(m_Specific(X), m_Value(Z)))) {
        auto *YZ = Builder.CreateFAddFMF(Y, Z, &I);
        auto *NewPow = Builder.CreateBinaryIntrinsic(Intrinsic::pow, X, YZ, &I);
        return replaceInstUsesWith(I, NewPow);
      }

      // powi(x, y) * powi(x, z) -> powi(x, y + z)
      if (match(Op0, m_Intrinsic<Intrinsic::powi>(m_Value(X), m_Value(Y))) &&
          match(Op1, m_Intrinsic<Intrinsic::powi>(m_Specific(X), m_Value(Z))) &&
          Y->getType() == Z->getType()) {
        auto *YZ = Builder.CreateAdd(Y, Z);
        auto *NewPow = Builder.CreateIntrinsic(
            Intrinsic::powi, {X->getType(), YZ->getType()}, {X, YZ}, &I);
        return replaceInstUsesWith(I, NewPow);
      }

      // exp(X) * exp(Y) -> exp(X + Y)
      if (match(Op0, m_Intrinsic<Intrinsic::exp>(m_Value(X))) &&
          match(Op1, m_Intrinsic<Intrinsic::exp>(m_Value(Y)))) {
        Value *XY = Builder.CreateFAddFMF(X, Y, &I);
        Value *Exp = Builder.CreateUnaryIntrinsic(Intrinsic::exp, XY, &I);
        return replaceInstUsesWith(I, Exp);
      }

      // exp2(X) * exp2(Y) -> exp2(X + Y)
      if (match(Op0, m_Intrinsic<Intrinsic::exp2>(m_Value(X))) &&
          match(Op1, m_Intrinsic<Intrinsic::exp2>(m_Value(Y)))) {
        Value *XY = Builder.CreateFAddFMF(X, Y, &I);
        Value *Exp2 = Builder.CreateUnaryIntrinsic(Intrinsic::exp2, XY, &I);
        return replaceInstUsesWith(I, Exp2);
      }
    }

    // (X*Y) * X => (X*X) * Y where Y != X
    //  The purpose is two-fold:
    //   1) to form a power expression (of X).
    //   2) potentially shorten the critical path: After transformation, the
    //  latency of the instruction Y is amortized by the expression of X*X,
    //  and therefore Y is in a "less critical" position compared to what it
    //  was before the transformation.
    if (match(Op0, m_OneUse(m_c_FMul(m_Specific(Op1), m_Value(Y)))) &&
        Op1 != Y) {
      Value *XX = Builder.CreateFMulFMF(Op1, Op1, &I);
      return BinaryOperator::CreateFMulFMF(XX, Y, &I);
    }
    if (match(Op1, m_OneUse(m_c_FMul(m_Specific(Op0), m_Value(Y)))) &&
        Op0 != Y) {
      Value *XX = Builder.CreateFMulFMF(Op0, Op0, &I);
      return BinaryOperator::CreateFMulFMF(XX, Y, &I);
    }
  }

  // log2(X * 0.5) * Y = log2(X) * Y - Y
  if (I.isFast()) {
    IntrinsicInst *Log2 = nullptr;
    if (match(Op0, m_OneUse(m_Intrinsic<Intrinsic::log2>(
            m_OneUse(m_FMul(m_Value(X), m_SpecificFP(0.5))))))) {
      Log2 = cast<IntrinsicInst>(Op0);
      Y = Op1;
    }
    if (match(Op1, m_OneUse(m_Intrinsic<Intrinsic::log2>(
            m_OneUse(m_FMul(m_Value(X), m_SpecificFP(0.5))))))) {
      Log2 = cast<IntrinsicInst>(Op1);
      Y = Op0;
    }
    if (Log2) {
      Value *Log2 = Builder.CreateUnaryIntrinsic(Intrinsic::log2, X, &I);
      Value *LogXTimesY = Builder.CreateFMulFMF(Log2, Y, &I);
      return BinaryOperator::CreateFSubFMF(LogXTimesY, Y, &I);
    }
  }

  return nullptr;
}

/// Fold a divide or remainder with a select instruction divisor when one of the
/// select operands is zero. In that case, we can use the other select operand
/// because div/rem by zero is undefined.
bool InstCombinerImpl::simplifyDivRemOfSelectWithZeroOp(BinaryOperator &I) {
  SelectInst *SI = dyn_cast<SelectInst>(I.getOperand(1));
  if (!SI)
    return false;

  int NonNullOperand;
  if (match(SI->getTrueValue(), m_Zero()))
    // div/rem X, (Cond ? 0 : Y) -> div/rem X, Y
    NonNullOperand = 2;
  else if (match(SI->getFalseValue(), m_Zero()))
    // div/rem X, (Cond ? Y : 0) -> div/rem X, Y
    NonNullOperand = 1;
  else
    return false;

  // Change the div/rem to use 'Y' instead of the select.
  replaceOperand(I, 1, SI->getOperand(NonNullOperand));

  // Okay, we know we replace the operand of the div/rem with 'Y' with no
  // problem.  However, the select, or the condition of the select may have
  // multiple uses.  Based on our knowledge that the operand must be non-zero,
  // propagate the known value for the select into other uses of it, and
  // propagate a known value of the condition into its other users.

  // If the select and condition only have a single use, don't bother with this,
  // early exit.
  Value *SelectCond = SI->getCondition();
  if (SI->use_empty() && SelectCond->hasOneUse())
    return true;

  // Scan the current block backward, looking for other uses of SI.
  BasicBlock::iterator BBI = I.getIterator(), BBFront = I.getParent()->begin();
  Type *CondTy = SelectCond->getType();
  while (BBI != BBFront) {
    --BBI;
    // If we found an instruction that we can't assume will return, so
    // information from below it cannot be propagated above it.
    if (!isGuaranteedToTransferExecutionToSuccessor(&*BBI))
      break;

    // Replace uses of the select or its condition with the known values.
    for (Use &Op : BBI->operands()) {
      if (Op == SI) {
        replaceUse(Op, SI->getOperand(NonNullOperand));
        Worklist.push(&*BBI);
      } else if (Op == SelectCond) {
        replaceUse(Op, NonNullOperand == 1 ? ConstantInt::getTrue(CondTy)
                                           : ConstantInt::getFalse(CondTy));
        Worklist.push(&*BBI);
      }
    }

    // If we past the instruction, quit looking for it.
    if (&*BBI == SI)
      SI = nullptr;
    if (&*BBI == SelectCond)
      SelectCond = nullptr;

    // If we ran out of things to eliminate, break out of the loop.
    if (!SelectCond && !SI)
      break;

  }
  return true;
}

/// True if the multiply can not be expressed in an int this size.
static bool multiplyOverflows(const APInt &C1, const APInt &C2, APInt &Product,
                              bool IsSigned) {
  bool Overflow;
  Product = IsSigned ? C1.smul_ov(C2, Overflow) : C1.umul_ov(C2, Overflow);
  return Overflow;
}

/// True if C1 is a multiple of C2. Quotient contains C1/C2.
static bool isMultiple(const APInt &C1, const APInt &C2, APInt &Quotient,
                       bool IsSigned) {
  assert(C1.getBitWidth() == C2.getBitWidth() && "Constant widths not equal");

  // Bail if we will divide by zero.
  if (C2.isZero())
    return false;

  // Bail if we would divide INT_MIN by -1.
  if (IsSigned && C1.isMinSignedValue() && C2.isAllOnes())
    return false;

  APInt Remainder(C1.getBitWidth(), /*val=*/0ULL, IsSigned);
  if (IsSigned)
    APInt::sdivrem(C1, C2, Quotient, Remainder);
  else
    APInt::udivrem(C1, C2, Quotient, Remainder);

  return Remainder.isMinValue();
}

/// This function implements the transforms common to both integer division
/// instructions (udiv and sdiv). It is called by the visitors to those integer
/// division instructions.
/// Common integer divide transforms
Instruction *InstCombinerImpl::commonIDivTransforms(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  bool IsSigned = I.getOpcode() == Instruction::SDiv;
  Type *Ty = I.getType();

  // The RHS is known non-zero.
  if (Value *V = simplifyValueKnownNonZero(I.getOperand(1), *this, I))
    return replaceOperand(I, 1, V);

  // Handle cases involving: [su]div X, (select Cond, Y, Z)
  // This does not apply for fdiv.
  if (simplifyDivRemOfSelectWithZeroOp(I))
    return &I;

  // If the divisor is a select-of-constants, try to constant fold all div ops:
  // C / (select Cond, TrueC, FalseC) --> select Cond, (C / TrueC), (C / FalseC)
  // TODO: Adapt simplifyDivRemOfSelectWithZeroOp to allow this and other folds.
  if (match(Op0, m_ImmConstant()) &&
      match(Op1, m_Select(m_Value(), m_ImmConstant(), m_ImmConstant()))) {
    if (Instruction *R = FoldOpIntoSelect(I, cast<SelectInst>(Op1)))
      return R;
  }

  const APInt *C2;
  if (match(Op1, m_APInt(C2))) {
    Value *X;
    const APInt *C1;

    // (X / C1) / C2  -> X / (C1*C2)
    if ((IsSigned && match(Op0, m_SDiv(m_Value(X), m_APInt(C1)))) ||
        (!IsSigned && match(Op0, m_UDiv(m_Value(X), m_APInt(C1))))) {
      APInt Product(C1->getBitWidth(), /*val=*/0ULL, IsSigned);
      if (!multiplyOverflows(*C1, *C2, Product, IsSigned))
        return BinaryOperator::Create(I.getOpcode(), X,
                                      ConstantInt::get(Ty, Product));
    }

    if ((IsSigned && match(Op0, m_NSWMul(m_Value(X), m_APInt(C1)))) ||
        (!IsSigned && match(Op0, m_NUWMul(m_Value(X), m_APInt(C1))))) {
      APInt Quotient(C1->getBitWidth(), /*val=*/0ULL, IsSigned);

      // (X * C1) / C2 -> X / (C2 / C1) if C2 is a multiple of C1.
      if (isMultiple(*C2, *C1, Quotient, IsSigned)) {
        auto *NewDiv = BinaryOperator::Create(I.getOpcode(), X,
                                              ConstantInt::get(Ty, Quotient));
        NewDiv->setIsExact(I.isExact());
        return NewDiv;
      }

      // (X * C1) / C2 -> X * (C1 / C2) if C1 is a multiple of C2.
      if (isMultiple(*C1, *C2, Quotient, IsSigned)) {
        auto *Mul = BinaryOperator::Create(Instruction::Mul, X,
                                           ConstantInt::get(Ty, Quotient));
        auto *OBO = cast<OverflowingBinaryOperator>(Op0);
        Mul->setHasNoUnsignedWrap(!IsSigned && OBO->hasNoUnsignedWrap());
        Mul->setHasNoSignedWrap(OBO->hasNoSignedWrap());
        return Mul;
      }
    }

    if ((IsSigned && match(Op0, m_NSWShl(m_Value(X), m_APInt(C1))) &&
         C1->ult(C1->getBitWidth() - 1)) ||
        (!IsSigned && match(Op0, m_NUWShl(m_Value(X), m_APInt(C1))) &&
         C1->ult(C1->getBitWidth()))) {
      APInt Quotient(C1->getBitWidth(), /*val=*/0ULL, IsSigned);
      APInt C1Shifted = APInt::getOneBitSet(
          C1->getBitWidth(), static_cast<unsigned>(C1->getZExtValue()));

      // (X << C1) / C2 -> X / (C2 >> C1) if C2 is a multiple of 1 << C1.
      if (isMultiple(*C2, C1Shifted, Quotient, IsSigned)) {
        auto *BO = BinaryOperator::Create(I.getOpcode(), X,
                                          ConstantInt::get(Ty, Quotient));
        BO->setIsExact(I.isExact());
        return BO;
      }

      // (X << C1) / C2 -> X * ((1 << C1) / C2) if 1 << C1 is a multiple of C2.
      if (isMultiple(C1Shifted, *C2, Quotient, IsSigned)) {
        auto *Mul = BinaryOperator::Create(Instruction::Mul, X,
                                           ConstantInt::get(Ty, Quotient));
        auto *OBO = cast<OverflowingBinaryOperator>(Op0);
        Mul->setHasNoUnsignedWrap(!IsSigned && OBO->hasNoUnsignedWrap());
        Mul->setHasNoSignedWrap(OBO->hasNoSignedWrap());
        return Mul;
      }
    }

    if (!C2->isZero()) // avoid X udiv 0
      if (Instruction *FoldedDiv = foldBinOpIntoSelectOrPhi(I))
        return FoldedDiv;
  }

  if (match(Op0, m_One())) {
    assert(!Ty->isIntOrIntVectorTy(1) && "i1 divide not removed?");
    if (IsSigned) {
      // If Op1 is 0 then it's undefined behaviour, if Op1 is 1 then the
      // result is one, if Op1 is -1 then the result is minus one, otherwise
      // it's zero.
      Value *Inc = Builder.CreateAdd(Op1, Op0);
      Value *Cmp = Builder.CreateICmpULT(Inc, ConstantInt::get(Ty, 3));
      return SelectInst::Create(Cmp, Op1, ConstantInt::get(Ty, 0));
    } else {
      // If Op1 is 0 then it's undefined behaviour. If Op1 is 1 then the
      // result is one, otherwise it's zero.
      return new ZExtInst(Builder.CreateICmpEQ(Op1, Op0), Ty);
    }
  }

  // See if we can fold away this div instruction.
  if (SimplifyDemandedInstructionBits(I))
    return &I;

  // (X - (X rem Y)) / Y -> X / Y; usually originates as ((X / Y) * Y) / Y
  Value *X, *Z;
  if (match(Op0, m_Sub(m_Value(X), m_Value(Z)))) // (X - Z) / Y; Y = Op1
    if ((IsSigned && match(Z, m_SRem(m_Specific(X), m_Specific(Op1)))) ||
        (!IsSigned && match(Z, m_URem(m_Specific(X), m_Specific(Op1)))))
      return BinaryOperator::Create(I.getOpcode(), X, Op1);

  // (X << Y) / X -> 1 << Y
  Value *Y;
  if (IsSigned && match(Op0, m_NSWShl(m_Specific(Op1), m_Value(Y))))
    return BinaryOperator::CreateNSWShl(ConstantInt::get(Ty, 1), Y);
  if (!IsSigned && match(Op0, m_NUWShl(m_Specific(Op1), m_Value(Y))))
    return BinaryOperator::CreateNUWShl(ConstantInt::get(Ty, 1), Y);

  // X / (X * Y) -> 1 / Y if the multiplication does not overflow.
  if (match(Op1, m_c_Mul(m_Specific(Op0), m_Value(Y)))) {
    bool HasNSW = cast<OverflowingBinaryOperator>(Op1)->hasNoSignedWrap();
    bool HasNUW = cast<OverflowingBinaryOperator>(Op1)->hasNoUnsignedWrap();
    if ((IsSigned && HasNSW) || (!IsSigned && HasNUW)) {
      replaceOperand(I, 0, ConstantInt::get(Ty, 1));
      replaceOperand(I, 1, Y);
      return &I;
    }
  }

  return nullptr;
}

static const unsigned MaxDepth = 6;

namespace {

using FoldUDivOperandCb = Instruction *(*)(Value *Op0, Value *Op1,
                                           const BinaryOperator &I,
                                           InstCombinerImpl &IC);

/// Used to maintain state for visitUDivOperand().
struct UDivFoldAction {
  /// Informs visitUDiv() how to fold this operand.  This can be zero if this
  /// action joins two actions together.
  FoldUDivOperandCb FoldAction;

  /// Which operand to fold.
  Value *OperandToFold;

  union {
    /// The instruction returned when FoldAction is invoked.
    Instruction *FoldResult;

    /// Stores the LHS action index if this action joins two actions together.
    size_t SelectLHSIdx;
  };

  UDivFoldAction(FoldUDivOperandCb FA, Value *InputOperand)
      : FoldAction(FA), OperandToFold(InputOperand), FoldResult(nullptr) {}
  UDivFoldAction(FoldUDivOperandCb FA, Value *InputOperand, size_t SLHS)
      : FoldAction(FA), OperandToFold(InputOperand), SelectLHSIdx(SLHS) {}
};

} // end anonymous namespace

// X udiv 2^C -> X >> C
static Instruction *foldUDivPow2Cst(Value *Op0, Value *Op1,
                                    const BinaryOperator &I,
                                    InstCombinerImpl &IC) {
  Constant *C1 = ConstantExpr::getExactLogBase2(cast<Constant>(Op1));
  if (!C1)
    llvm_unreachable("Failed to constant fold udiv -> logbase2");
  BinaryOperator *LShr = BinaryOperator::CreateLShr(Op0, C1);
  if (I.isExact())
    LShr->setIsExact();
  return LShr;
}

// X udiv (C1 << N), where C1 is "1<<C2"  -->  X >> (N+C2)
// X udiv (zext (C1 << N)), where C1 is "1<<C2"  -->  X >> (N+C2)
static Instruction *foldUDivShl(Value *Op0, Value *Op1, const BinaryOperator &I,
                                InstCombinerImpl &IC) {
  Value *ShiftLeft;
  if (!match(Op1, m_ZExt(m_Value(ShiftLeft))))
    ShiftLeft = Op1;

  Constant *CI;
  Value *N;
  if (!match(ShiftLeft, m_Shl(m_Constant(CI), m_Value(N))))
    llvm_unreachable("match should never fail here!");
  Constant *Log2Base = ConstantExpr::getExactLogBase2(CI);
  if (!Log2Base)
    llvm_unreachable("getLogBase2 should never fail here!");
  N = IC.Builder.CreateAdd(N, Log2Base);
  if (Op1 != ShiftLeft)
    N = IC.Builder.CreateZExt(N, Op1->getType());
  BinaryOperator *LShr = BinaryOperator::CreateLShr(Op0, N);
  if (I.isExact())
    LShr->setIsExact();
  return LShr;
}

// Recursively visits the possible right hand operands of a udiv
// instruction, seeing through select instructions, to determine if we can
// replace the udiv with something simpler.  If we find that an operand is not
// able to simplify the udiv, we abort the entire transformation.
static size_t visitUDivOperand(Value *Op0, Value *Op1, const BinaryOperator &I,
                               SmallVectorImpl<UDivFoldAction> &Actions,
                               unsigned Depth = 0) {
  // FIXME: assert that Op1 isn't/doesn't contain undef.

  // Check to see if this is an unsigned division with an exact power of 2,
  // if so, convert to a right shift.
  if (match(Op1, m_Power2())) {
    Actions.push_back(UDivFoldAction(foldUDivPow2Cst, Op1));
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
    // FIXME: missed optimization: if one of the hands of select is/contains
    //        undef, just directly pick the other one.
    // FIXME: can both hands contain undef?
    if (size_t LHSIdx =
            visitUDivOperand(Op0, SI->getOperand(1), I, Actions, Depth))
      if (visitUDivOperand(Op0, SI->getOperand(2), I, Actions, Depth)) {
        Actions.push_back(UDivFoldAction(nullptr, Op1, LHSIdx - 1));
        return Actions.size();
      }

  return 0;
}

/// If we have zero-extended operands of an unsigned div or rem, we may be able
/// to narrow the operation (sink the zext below the math).
static Instruction *narrowUDivURem(BinaryOperator &I,
                                   InstCombiner::BuilderTy &Builder) {
  Instruction::BinaryOps Opcode = I.getOpcode();
  Value *N = I.getOperand(0);
  Value *D = I.getOperand(1);
  Type *Ty = I.getType();
  Value *X, *Y;
  if (match(N, m_ZExt(m_Value(X))) && match(D, m_ZExt(m_Value(Y))) &&
      X->getType() == Y->getType() && (N->hasOneUse() || D->hasOneUse())) {
    // udiv (zext X), (zext Y) --> zext (udiv X, Y)
    // urem (zext X), (zext Y) --> zext (urem X, Y)
    Value *NarrowOp = Builder.CreateBinOp(Opcode, X, Y);
    return new ZExtInst(NarrowOp, Ty);
  }

  Constant *C;
  if ((match(N, m_OneUse(m_ZExt(m_Value(X)))) && match(D, m_Constant(C))) ||
      (match(D, m_OneUse(m_ZExt(m_Value(X)))) && match(N, m_Constant(C)))) {
    // If the constant is the same in the smaller type, use the narrow version.
    Constant *TruncC = ConstantExpr::getTrunc(C, X->getType());
    if (ConstantExpr::getZExt(TruncC, Ty) != C)
      return nullptr;

    // udiv (zext X), C --> zext (udiv X, C')
    // urem (zext X), C --> zext (urem X, C')
    // udiv C, (zext X) --> zext (udiv C', X)
    // urem C, (zext X) --> zext (urem C', X)
    Value *NarrowOp = isa<Constant>(D) ? Builder.CreateBinOp(Opcode, X, TruncC)
                                       : Builder.CreateBinOp(Opcode, TruncC, X);
    return new ZExtInst(NarrowOp, Ty);
  }

  return nullptr;
}

Instruction *InstCombinerImpl::visitUDiv(BinaryOperator &I) {
  if (Value *V = SimplifyUDivInst(I.getOperand(0), I.getOperand(1),
                                  SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  if (Instruction *X = foldVectorBinop(I))
    return X;

  // Handle the integer div common cases
  if (Instruction *Common = commonIDivTransforms(I))
    return Common;

  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  Value *X;
  const APInt *C1, *C2;
  if (match(Op0, m_LShr(m_Value(X), m_APInt(C1))) && match(Op1, m_APInt(C2))) {
    // (X lshr C1) udiv C2 --> X udiv (C2 << C1)
    bool Overflow;
    APInt C2ShlC1 = C2->ushl_ov(*C1, Overflow);
    if (!Overflow) {
      bool IsExact = I.isExact() && match(Op0, m_Exact(m_Value()));
      BinaryOperator *BO = BinaryOperator::CreateUDiv(
          X, ConstantInt::get(X->getType(), C2ShlC1));
      if (IsExact)
        BO->setIsExact();
      return BO;
    }
  }

  // Op0 / C where C is large (negative) --> zext (Op0 >= C)
  // TODO: Could use isKnownNegative() to handle non-constant values.
  Type *Ty = I.getType();
  if (match(Op1, m_Negative())) {
    Value *Cmp = Builder.CreateICmpUGE(Op0, Op1);
    return CastInst::CreateZExtOrBitCast(Cmp, Ty);
  }
  // Op0 / (sext i1 X) --> zext (Op0 == -1) (if X is 0, the div is undefined)
  if (match(Op1, m_SExt(m_Value(X))) && X->getType()->isIntOrIntVectorTy(1)) {
    Value *Cmp = Builder.CreateICmpEQ(Op0, ConstantInt::getAllOnesValue(Ty));
    return CastInst::CreateZExtOrBitCast(Cmp, Ty);
  }

  if (Instruction *NarrowDiv = narrowUDivURem(I, Builder))
    return NarrowDiv;

  // If the udiv operands are non-overflowing multiplies with a common operand,
  // then eliminate the common factor:
  // (A * B) / (A * X) --> B / X (and commuted variants)
  // TODO: The code would be reduced if we had m_c_NUWMul pattern matching.
  // TODO: If -reassociation handled this generally, we could remove this.
  Value *A, *B;
  if (match(Op0, m_NUWMul(m_Value(A), m_Value(B)))) {
    if (match(Op1, m_NUWMul(m_Specific(A), m_Value(X))) ||
        match(Op1, m_NUWMul(m_Value(X), m_Specific(A))))
      return BinaryOperator::CreateUDiv(B, X);
    if (match(Op1, m_NUWMul(m_Specific(B), m_Value(X))) ||
        match(Op1, m_NUWMul(m_Value(X), m_Specific(B))))
      return BinaryOperator::CreateUDiv(A, X);
  }

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

  return nullptr;
}

Instruction *InstCombinerImpl::visitSDiv(BinaryOperator &I) {
  if (Value *V = SimplifySDivInst(I.getOperand(0), I.getOperand(1),
                                  SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  if (Instruction *X = foldVectorBinop(I))
    return X;

  // Handle the integer div common cases
  if (Instruction *Common = commonIDivTransforms(I))
    return Common;

  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  Type *Ty = I.getType();
  Value *X;
  // sdiv Op0, -1 --> -Op0
  // sdiv Op0, (sext i1 X) --> -Op0 (because if X is 0, the op is undefined)
  if (match(Op1, m_AllOnes()) ||
      (match(Op1, m_SExt(m_Value(X))) && X->getType()->isIntOrIntVectorTy(1)))
    return BinaryOperator::CreateNeg(Op0);

  // X / INT_MIN --> X == INT_MIN
  if (match(Op1, m_SignMask()))
    return new ZExtInst(Builder.CreateICmpEQ(Op0, Op1), Ty);

  // sdiv exact X,  1<<C  -->    ashr exact X, C   iff  1<<C  is non-negative
  // sdiv exact X, -1<<C  -->  -(ashr exact X, C)
  if (I.isExact() && ((match(Op1, m_Power2()) && match(Op1, m_NonNegative())) ||
                      match(Op1, m_NegatedPower2()))) {
    bool DivisorWasNegative = match(Op1, m_NegatedPower2());
    if (DivisorWasNegative)
      Op1 = ConstantExpr::getNeg(cast<Constant>(Op1));
    auto *AShr = BinaryOperator::CreateExactAShr(
        Op0, ConstantExpr::getExactLogBase2(cast<Constant>(Op1)), I.getName());
    if (!DivisorWasNegative)
      return AShr;
    Builder.Insert(AShr);
    AShr->setName(I.getName() + ".neg");
    return BinaryOperator::CreateNeg(AShr, I.getName());
  }

  const APInt *Op1C;
  if (match(Op1, m_APInt(Op1C))) {
    // If the dividend is sign-extended and the constant divisor is small enough
    // to fit in the source type, shrink the division to the narrower type:
    // (sext X) sdiv C --> sext (X sdiv C)
    Value *Op0Src;
    if (match(Op0, m_OneUse(m_SExt(m_Value(Op0Src)))) &&
        Op0Src->getType()->getScalarSizeInBits() >= Op1C->getMinSignedBits()) {

      // In the general case, we need to make sure that the dividend is not the
      // minimum signed value because dividing that by -1 is UB. But here, we
      // know that the -1 divisor case is already handled above.

      Constant *NarrowDivisor =
          ConstantExpr::getTrunc(cast<Constant>(Op1), Op0Src->getType());
      Value *NarrowOp = Builder.CreateSDiv(Op0Src, NarrowDivisor);
      return new SExtInst(NarrowOp, Ty);
    }

    // -X / C --> X / -C (if the negation doesn't overflow).
    // TODO: This could be enhanced to handle arbitrary vector constants by
    //       checking if all elements are not the min-signed-val.
    if (!Op1C->isMinSignedValue() &&
        match(Op0, m_NSWSub(m_Zero(), m_Value(X)))) {
      Constant *NegC = ConstantInt::get(Ty, -(*Op1C));
      Instruction *BO = BinaryOperator::CreateSDiv(X, NegC);
      BO->setIsExact(I.isExact());
      return BO;
    }
  }

  // -X / Y --> -(X / Y)
  Value *Y;
  if (match(&I, m_SDiv(m_OneUse(m_NSWSub(m_Zero(), m_Value(X))), m_Value(Y))))
    return BinaryOperator::CreateNSWNeg(
        Builder.CreateSDiv(X, Y, I.getName(), I.isExact()));

  // abs(X) / X --> X > -1 ? 1 : -1
  // X / abs(X) --> X > -1 ? 1 : -1
  if (match(&I, m_c_BinOp(
                    m_OneUse(m_Intrinsic<Intrinsic::abs>(m_Value(X), m_One())),
                    m_Deferred(X)))) {
    Constant *NegOne = ConstantInt::getAllOnesValue(Ty);
    Value *Cond = Builder.CreateICmpSGT(X, NegOne);
    return SelectInst::Create(Cond, ConstantInt::get(Ty, 1), NegOne);
  }

  // If the sign bits of both operands are zero (i.e. we can prove they are
  // unsigned inputs), turn this into a udiv.
  APInt Mask(APInt::getSignMask(Ty->getScalarSizeInBits()));
  if (MaskedValueIsZero(Op0, Mask, 0, &I)) {
    if (MaskedValueIsZero(Op1, Mask, 0, &I)) {
      // X sdiv Y -> X udiv Y, iff X and Y don't have sign bit set
      auto *BO = BinaryOperator::CreateUDiv(Op0, Op1, I.getName());
      BO->setIsExact(I.isExact());
      return BO;
    }

    if (match(Op1, m_NegatedPower2())) {
      // X sdiv (-(1 << C)) -> -(X sdiv (1 << C)) ->
      //                    -> -(X udiv (1 << C)) -> -(X u>> C)
      return BinaryOperator::CreateNeg(Builder.Insert(foldUDivPow2Cst(
          Op0, ConstantExpr::getNeg(cast<Constant>(Op1)), I, *this)));
    }

    if (isKnownToBeAPowerOfTwo(Op1, /*OrZero*/ true, 0, &I)) {
      // X sdiv (1 << Y) -> X udiv (1 << Y) ( -> X u>> Y)
      // Safe because the only negative value (1 << Y) can take on is
      // INT_MIN, and X sdiv INT_MIN == X udiv INT_MIN == 0 if X doesn't have
      // the sign bit set.
      auto *BO = BinaryOperator::CreateUDiv(Op0, Op1, I.getName());
      BO->setIsExact(I.isExact());
      return BO;
    }
  }

  return nullptr;
}

/// Remove negation and try to convert division into multiplication.
static Instruction *foldFDivConstantDivisor(BinaryOperator &I) {
  Constant *C;
  if (!match(I.getOperand(1), m_Constant(C)))
    return nullptr;

  // -X / C --> X / -C
  Value *X;
  if (match(I.getOperand(0), m_FNeg(m_Value(X))))
    return BinaryOperator::CreateFDivFMF(X, ConstantExpr::getFNeg(C), &I);

  // If the constant divisor has an exact inverse, this is always safe. If not,
  // then we can still create a reciprocal if fast-math-flags allow it and the
  // constant is a regular number (not zero, infinite, or denormal).
  if (!(C->hasExactInverseFP() || (I.hasAllowReciprocal() && C->isNormalFP())))
    return nullptr;

  // Disallow denormal constants because we don't know what would happen
  // on all targets.
  // TODO: Use Intrinsic::canonicalize or let function attributes tell us that
  // denorms are flushed?
  auto *RecipC = ConstantExpr::getFDiv(ConstantFP::get(I.getType(), 1.0), C);
  if (!RecipC->isNormalFP())
    return nullptr;

  // X / C --> X * (1 / C)
  return BinaryOperator::CreateFMulFMF(I.getOperand(0), RecipC, &I);
}

/// Remove negation and try to reassociate constant math.
static Instruction *foldFDivConstantDividend(BinaryOperator &I) {
  Constant *C;
  if (!match(I.getOperand(0), m_Constant(C)))
    return nullptr;

  // C / -X --> -C / X
  Value *X;
  if (match(I.getOperand(1), m_FNeg(m_Value(X))))
    return BinaryOperator::CreateFDivFMF(ConstantExpr::getFNeg(C), X, &I);

  if (!I.hasAllowReassoc() || !I.hasAllowReciprocal())
    return nullptr;

  // Try to reassociate C / X expressions where X includes another constant.
  Constant *C2, *NewC = nullptr;
  if (match(I.getOperand(1), m_FMul(m_Value(X), m_Constant(C2)))) {
    // C / (X * C2) --> (C / C2) / X
    NewC = ConstantExpr::getFDiv(C, C2);
  } else if (match(I.getOperand(1), m_FDiv(m_Value(X), m_Constant(C2)))) {
    // C / (X / C2) --> (C * C2) / X
    NewC = ConstantExpr::getFMul(C, C2);
  }
  // Disallow denormal constants because we don't know what would happen
  // on all targets.
  // TODO: Use Intrinsic::canonicalize or let function attributes tell us that
  // denorms are flushed?
  if (!NewC || !NewC->isNormalFP())
    return nullptr;

  return BinaryOperator::CreateFDivFMF(NewC, X, &I);
}

/// Negate the exponent of pow/exp to fold division-by-pow() into multiply.
static Instruction *foldFDivPowDivisor(BinaryOperator &I,
                                       InstCombiner::BuilderTy &Builder) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  auto *II = dyn_cast<IntrinsicInst>(Op1);
  if (!II || !II->hasOneUse() || !I.hasAllowReassoc() ||
      !I.hasAllowReciprocal())
    return nullptr;

  // Z / pow(X, Y) --> Z * pow(X, -Y)
  // Z / exp{2}(Y) --> Z * exp{2}(-Y)
  // In the general case, this creates an extra instruction, but fmul allows
  // for better canonicalization and optimization than fdiv.
  Intrinsic::ID IID = II->getIntrinsicID();
  SmallVector<Value *> Args;
  switch (IID) {
  case Intrinsic::pow:
    Args.push_back(II->getArgOperand(0));
    Args.push_back(Builder.CreateFNegFMF(II->getArgOperand(1), &I));
    break;
  case Intrinsic::powi: {
    // Require 'ninf' assuming that makes powi(X, -INT_MIN) acceptable.
    // That is, X ** (huge negative number) is 0.0, ~1.0, or INF and so
    // dividing by that is INF, ~1.0, or 0.0. Code that uses powi allows
    // non-standard results, so this corner case should be acceptable if the
    // code rules out INF values.
    if (!I.hasNoInfs())
      return nullptr;
    Args.push_back(II->getArgOperand(0));
    Args.push_back(Builder.CreateNeg(II->getArgOperand(1)));
    Type *Tys[] = {I.getType(), II->getArgOperand(1)->getType()};
    Value *Pow = Builder.CreateIntrinsic(IID, Tys, Args, &I);
    return BinaryOperator::CreateFMulFMF(Op0, Pow, &I);
  }
  case Intrinsic::exp:
  case Intrinsic::exp2:
    Args.push_back(Builder.CreateFNegFMF(II->getArgOperand(0), &I));
    break;
  default:
    return nullptr;
  }
  Value *Pow = Builder.CreateIntrinsic(IID, I.getType(), Args, &I);
  return BinaryOperator::CreateFMulFMF(Op0, Pow, &I);
}

Instruction *InstCombinerImpl::visitFDiv(BinaryOperator &I) {
  if (Value *V = SimplifyFDivInst(I.getOperand(0), I.getOperand(1),
                                  I.getFastMathFlags(),
                                  SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  if (Instruction *X = foldVectorBinop(I))
    return X;

  if (Instruction *R = foldFDivConstantDivisor(I))
    return R;

  if (Instruction *R = foldFDivConstantDividend(I))
    return R;

  if (Instruction *R = foldFPSignBitOps(I))
    return R;

  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  if (isa<Constant>(Op0))
    if (SelectInst *SI = dyn_cast<SelectInst>(Op1))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;

  if (isa<Constant>(Op1))
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;

  if (I.hasAllowReassoc() && I.hasAllowReciprocal()) {
    Value *X, *Y;
    if (match(Op0, m_OneUse(m_FDiv(m_Value(X), m_Value(Y)))) &&
        (!isa<Constant>(Y) || !isa<Constant>(Op1))) {
      // (X / Y) / Z => X / (Y * Z)
      Value *YZ = Builder.CreateFMulFMF(Y, Op1, &I);
      return BinaryOperator::CreateFDivFMF(X, YZ, &I);
    }
    if (match(Op1, m_OneUse(m_FDiv(m_Value(X), m_Value(Y)))) &&
        (!isa<Constant>(Y) || !isa<Constant>(Op0))) {
      // Z / (X / Y) => (Y * Z) / X
      Value *YZ = Builder.CreateFMulFMF(Y, Op0, &I);
      return BinaryOperator::CreateFDivFMF(YZ, X, &I);
    }
    // Z / (1.0 / Y) => (Y * Z)
    //
    // This is a special case of Z / (X / Y) => (Y * Z) / X, with X = 1.0. The
    // m_OneUse check is avoided because even in the case of the multiple uses
    // for 1.0/Y, the number of instructions remain the same and a division is
    // replaced by a multiplication.
    if (match(Op1, m_FDiv(m_SpecificFP(1.0), m_Value(Y))))
      return BinaryOperator::CreateFMulFMF(Y, Op0, &I);
  }

  if (I.hasAllowReassoc() && Op0->hasOneUse() && Op1->hasOneUse()) {
    // sin(X) / cos(X) -> tan(X)
    // cos(X) / sin(X) -> 1/tan(X) (cotangent)
    Value *X;
    bool IsTan = match(Op0, m_Intrinsic<Intrinsic::sin>(m_Value(X))) &&
                 match(Op1, m_Intrinsic<Intrinsic::cos>(m_Specific(X)));
    bool IsCot =
        !IsTan && match(Op0, m_Intrinsic<Intrinsic::cos>(m_Value(X))) &&
                  match(Op1, m_Intrinsic<Intrinsic::sin>(m_Specific(X)));

    if ((IsTan || IsCot) &&
        hasFloatFn(&TLI, I.getType(), LibFunc_tan, LibFunc_tanf, LibFunc_tanl)) {
      IRBuilder<> B(&I);
      IRBuilder<>::FastMathFlagGuard FMFGuard(B);
      B.setFastMathFlags(I.getFastMathFlags());
      AttributeList Attrs =
          cast<CallBase>(Op0)->getCalledFunction()->getAttributes();
      Value *Res = emitUnaryFloatFnCall(X, &TLI, LibFunc_tan, LibFunc_tanf,
                                        LibFunc_tanl, B, Attrs);
      if (IsCot)
        Res = B.CreateFDiv(ConstantFP::get(I.getType(), 1.0), Res);
      return replaceInstUsesWith(I, Res);
    }
  }

  // X / (X * Y) --> 1.0 / Y
  // Reassociate to (X / X -> 1.0) is legal when NaNs are not allowed.
  // We can ignore the possibility that X is infinity because INF/INF is NaN.
  Value *X, *Y;
  if (I.hasNoNaNs() && I.hasAllowReassoc() &&
      match(Op1, m_c_FMul(m_Specific(Op0), m_Value(Y)))) {
    replaceOperand(I, 0, ConstantFP::get(I.getType(), 1.0));
    replaceOperand(I, 1, Y);
    return &I;
  }

  // X / fabs(X) -> copysign(1.0, X)
  // fabs(X) / X -> copysign(1.0, X)
  if (I.hasNoNaNs() && I.hasNoInfs() &&
      (match(&I, m_FDiv(m_Value(X), m_FAbs(m_Deferred(X)))) ||
       match(&I, m_FDiv(m_FAbs(m_Value(X)), m_Deferred(X))))) {
    Value *V = Builder.CreateBinaryIntrinsic(
        Intrinsic::copysign, ConstantFP::get(I.getType(), 1.0), X, &I);
    return replaceInstUsesWith(I, V);
  }

  if (Instruction *Mul = foldFDivPowDivisor(I, Builder))
    return Mul;

  return nullptr;
}

/// This function implements the transforms common to both integer remainder
/// instructions (urem and srem). It is called by the visitors to those integer
/// remainder instructions.
/// Common integer remainder transforms
Instruction *InstCombinerImpl::commonIRemTransforms(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // The RHS is known non-zero.
  if (Value *V = simplifyValueKnownNonZero(I.getOperand(1), *this, I))
    return replaceOperand(I, 1, V);

  // Handle cases involving: rem X, (select Cond, Y, Z)
  if (simplifyDivRemOfSelectWithZeroOp(I))
    return &I;

  // If the divisor is a select-of-constants, try to constant fold all rem ops:
  // C % (select Cond, TrueC, FalseC) --> select Cond, (C % TrueC), (C % FalseC)
  // TODO: Adapt simplifyDivRemOfSelectWithZeroOp to allow this and other folds.
  if (match(Op0, m_ImmConstant()) &&
      match(Op1, m_Select(m_Value(), m_ImmConstant(), m_ImmConstant()))) {
    if (Instruction *R = FoldOpIntoSelect(I, cast<SelectInst>(Op1)))
      return R;
  }

  if (isa<Constant>(Op1)) {
    if (Instruction *Op0I = dyn_cast<Instruction>(Op0)) {
      if (SelectInst *SI = dyn_cast<SelectInst>(Op0I)) {
        if (Instruction *R = FoldOpIntoSelect(I, SI))
          return R;
      } else if (auto *PN = dyn_cast<PHINode>(Op0I)) {
        const APInt *Op1Int;
        if (match(Op1, m_APInt(Op1Int)) && !Op1Int->isMinValue() &&
            (I.getOpcode() == Instruction::URem ||
             !Op1Int->isMinSignedValue())) {
          // foldOpIntoPhi will speculate instructions to the end of the PHI's
          // predecessor blocks, so do this only if we know the srem or urem
          // will not fault.
          if (Instruction *NV = foldOpIntoPhi(I, PN))
            return NV;
        }
      }

      // See if we can fold away this rem instruction.
      if (SimplifyDemandedInstructionBits(I))
        return &I;
    }
  }

  return nullptr;
}

Instruction *InstCombinerImpl::visitURem(BinaryOperator &I) {
  if (Value *V = SimplifyURemInst(I.getOperand(0), I.getOperand(1),
                                  SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  if (Instruction *X = foldVectorBinop(I))
    return X;

  if (Instruction *common = commonIRemTransforms(I))
    return common;

  if (Instruction *NarrowRem = narrowUDivURem(I, Builder))
    return NarrowRem;

  // X urem Y -> X and Y-1, where Y is a power of 2,
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  Type *Ty = I.getType();
  if (isKnownToBeAPowerOfTwo(Op1, /*OrZero*/ true, 0, &I)) {
    // This may increase instruction count, we don't enforce that Y is a
    // constant.
    Constant *N1 = Constant::getAllOnesValue(Ty);
    Value *Add = Builder.CreateAdd(Op1, N1);
    return BinaryOperator::CreateAnd(Op0, Add);
  }

  // 1 urem X -> zext(X != 1)
  if (match(Op0, m_One())) {
    Value *Cmp = Builder.CreateICmpNE(Op1, ConstantInt::get(Ty, 1));
    return CastInst::CreateZExtOrBitCast(Cmp, Ty);
  }

  // X urem C -> X < C ? X : X - C, where C >= signbit.
  if (match(Op1, m_Negative())) {
    Value *Cmp = Builder.CreateICmpULT(Op0, Op1);
    Value *Sub = Builder.CreateSub(Op0, Op1);
    return SelectInst::Create(Cmp, Op0, Sub);
  }

  // If the divisor is a sext of a boolean, then the divisor must be max
  // unsigned value (-1). Therefore, the remainder is Op0 unless Op0 is also
  // max unsigned value. In that case, the remainder is 0:
  // urem Op0, (sext i1 X) --> (Op0 == -1) ? 0 : Op0
  Value *X;
  if (match(Op1, m_SExt(m_Value(X))) && X->getType()->isIntOrIntVectorTy(1)) {
    Value *Cmp = Builder.CreateICmpEQ(Op0, ConstantInt::getAllOnesValue(Ty));
    return SelectInst::Create(Cmp, ConstantInt::getNullValue(Ty), Op0);
  }

  return nullptr;
}

Instruction *InstCombinerImpl::visitSRem(BinaryOperator &I) {
  if (Value *V = SimplifySRemInst(I.getOperand(0), I.getOperand(1),
                                  SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  if (Instruction *X = foldVectorBinop(I))
    return X;

  // Handle the integer rem common cases
  if (Instruction *Common = commonIRemTransforms(I))
    return Common;

  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  {
    const APInt *Y;
    // X % -Y -> X % Y
    if (match(Op1, m_Negative(Y)) && !Y->isMinSignedValue())
      return replaceOperand(I, 1, ConstantInt::get(I.getType(), -*Y));
  }

  // -X srem Y --> -(X srem Y)
  Value *X, *Y;
  if (match(&I, m_SRem(m_OneUse(m_NSWSub(m_Zero(), m_Value(X))), m_Value(Y))))
    return BinaryOperator::CreateNSWNeg(Builder.CreateSRem(X, Y));

  // If the sign bits of both operands are zero (i.e. we can prove they are
  // unsigned inputs), turn this into a urem.
  APInt Mask(APInt::getSignMask(I.getType()->getScalarSizeInBits()));
  if (MaskedValueIsZero(Op1, Mask, 0, &I) &&
      MaskedValueIsZero(Op0, Mask, 0, &I)) {
    // X srem Y -> X urem Y, iff X and Y don't have sign bit set
    return BinaryOperator::CreateURem(Op0, Op1, I.getName());
  }

  // If it's a constant vector, flip any negative values positive.
  if (isa<ConstantVector>(Op1) || isa<ConstantDataVector>(Op1)) {
    Constant *C = cast<Constant>(Op1);
    unsigned VWidth = cast<FixedVectorType>(C->getType())->getNumElements();

    bool hasNegative = false;
    bool hasMissing = false;
    for (unsigned i = 0; i != VWidth; ++i) {
      Constant *Elt = C->getAggregateElement(i);
      if (!Elt) {
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
      if (NewRHSV != C)  // Don't loop on -MININT
        return replaceOperand(I, 1, NewRHSV);
    }
  }

  return nullptr;
}

Instruction *InstCombinerImpl::visitFRem(BinaryOperator &I) {
  if (Value *V = SimplifyFRemInst(I.getOperand(0), I.getOperand(1),
                                  I.getFastMathFlags(),
                                  SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  if (Instruction *X = foldVectorBinop(I))
    return X;

  return nullptr;
}
