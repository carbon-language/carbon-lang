//===- InstCombineCasts.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the visit functions for cast operations.
//
//===----------------------------------------------------------------------===//

#include "InstCombineInternal.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Transforms/InstCombine/InstCombiner.h"
#include <numeric>
using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "instcombine"

/// Analyze 'Val', seeing if it is a simple linear expression.
/// If so, decompose it, returning some value X, such that Val is
/// X*Scale+Offset.
///
static Value *decomposeSimpleLinearExpr(Value *Val, unsigned &Scale,
                                        uint64_t &Offset) {
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Val)) {
    Offset = CI->getZExtValue();
    Scale  = 0;
    return ConstantInt::get(Val->getType(), 0);
  }

  if (BinaryOperator *I = dyn_cast<BinaryOperator>(Val)) {
    // Cannot look past anything that might overflow.
    OverflowingBinaryOperator *OBI = dyn_cast<OverflowingBinaryOperator>(Val);
    if (OBI && !OBI->hasNoUnsignedWrap() && !OBI->hasNoSignedWrap()) {
      Scale = 1;
      Offset = 0;
      return Val;
    }

    if (ConstantInt *RHS = dyn_cast<ConstantInt>(I->getOperand(1))) {
      if (I->getOpcode() == Instruction::Shl) {
        // This is a value scaled by '1 << the shift amt'.
        Scale = UINT64_C(1) << RHS->getZExtValue();
        Offset = 0;
        return I->getOperand(0);
      }

      if (I->getOpcode() == Instruction::Mul) {
        // This value is scaled by 'RHS'.
        Scale = RHS->getZExtValue();
        Offset = 0;
        return I->getOperand(0);
      }

      if (I->getOpcode() == Instruction::Add) {
        // We have X+C.  Check to see if we really have (X*C2)+C1,
        // where C1 is divisible by C2.
        unsigned SubScale;
        Value *SubVal =
          decomposeSimpleLinearExpr(I->getOperand(0), SubScale, Offset);
        Offset += RHS->getZExtValue();
        Scale = SubScale;
        return SubVal;
      }
    }
  }

  // Otherwise, we can't look past this.
  Scale = 1;
  Offset = 0;
  return Val;
}

/// If we find a cast of an allocation instruction, try to eliminate the cast by
/// moving the type information into the alloc.
Instruction *InstCombinerImpl::PromoteCastOfAllocation(BitCastInst &CI,
                                                       AllocaInst &AI) {
  PointerType *PTy = cast<PointerType>(CI.getType());
  // Opaque pointers don't have an element type we could replace with.
  if (PTy->isOpaque())
    return nullptr;

  IRBuilderBase::InsertPointGuard Guard(Builder);
  Builder.SetInsertPoint(&AI);

  // Get the type really allocated and the type casted to.
  Type *AllocElTy = AI.getAllocatedType();
  Type *CastElTy = PTy->getNonOpaquePointerElementType();
  if (!AllocElTy->isSized() || !CastElTy->isSized()) return nullptr;

  // This optimisation does not work for cases where the cast type
  // is scalable and the allocated type is not. This because we need to
  // know how many times the casted type fits into the allocated type.
  // For the opposite case where the allocated type is scalable and the
  // cast type is not this leads to poor code quality due to the
  // introduction of 'vscale' into the calculations. It seems better to
  // bail out for this case too until we've done a proper cost-benefit
  // analysis.
  bool AllocIsScalable = isa<ScalableVectorType>(AllocElTy);
  bool CastIsScalable = isa<ScalableVectorType>(CastElTy);
  if (AllocIsScalable != CastIsScalable) return nullptr;

  Align AllocElTyAlign = DL.getABITypeAlign(AllocElTy);
  Align CastElTyAlign = DL.getABITypeAlign(CastElTy);
  if (CastElTyAlign < AllocElTyAlign) return nullptr;

  // If the allocation has multiple uses, only promote it if we are strictly
  // increasing the alignment of the resultant allocation.  If we keep it the
  // same, we open the door to infinite loops of various kinds.
  if (!AI.hasOneUse() && CastElTyAlign == AllocElTyAlign) return nullptr;

  // The alloc and cast types should be either both fixed or both scalable.
  uint64_t AllocElTySize = DL.getTypeAllocSize(AllocElTy).getKnownMinSize();
  uint64_t CastElTySize = DL.getTypeAllocSize(CastElTy).getKnownMinSize();
  if (CastElTySize == 0 || AllocElTySize == 0) return nullptr;

  // If the allocation has multiple uses, only promote it if we're not
  // shrinking the amount of memory being allocated.
  uint64_t AllocElTyStoreSize = DL.getTypeStoreSize(AllocElTy).getKnownMinSize();
  uint64_t CastElTyStoreSize = DL.getTypeStoreSize(CastElTy).getKnownMinSize();
  if (!AI.hasOneUse() && CastElTyStoreSize < AllocElTyStoreSize) return nullptr;

  // See if we can satisfy the modulus by pulling a scale out of the array
  // size argument.
  unsigned ArraySizeScale;
  uint64_t ArrayOffset;
  Value *NumElements = // See if the array size is a decomposable linear expr.
    decomposeSimpleLinearExpr(AI.getOperand(0), ArraySizeScale, ArrayOffset);

  // If we can now satisfy the modulus, by using a non-1 scale, we really can
  // do the xform.
  if ((AllocElTySize*ArraySizeScale) % CastElTySize != 0 ||
      (AllocElTySize*ArrayOffset   ) % CastElTySize != 0) return nullptr;

  // We don't currently support arrays of scalable types.
  assert(!AllocIsScalable || (ArrayOffset == 1 && ArraySizeScale == 0));

  unsigned Scale = (AllocElTySize*ArraySizeScale)/CastElTySize;
  Value *Amt = nullptr;
  if (Scale == 1) {
    Amt = NumElements;
  } else {
    Amt = ConstantInt::get(AI.getArraySize()->getType(), Scale);
    // Insert before the alloca, not before the cast.
    Amt = Builder.CreateMul(Amt, NumElements);
  }

  if (uint64_t Offset = (AllocElTySize*ArrayOffset)/CastElTySize) {
    Value *Off = ConstantInt::get(AI.getArraySize()->getType(),
                                  Offset, true);
    Amt = Builder.CreateAdd(Amt, Off);
  }

  AllocaInst *New = Builder.CreateAlloca(CastElTy, AI.getAddressSpace(), Amt);
  New->setAlignment(AI.getAlign());
  New->takeName(&AI);
  New->setUsedWithInAlloca(AI.isUsedWithInAlloca());

  // If the allocation has multiple real uses, insert a cast and change all
  // things that used it to use the new cast.  This will also hack on CI, but it
  // will die soon.
  if (!AI.hasOneUse()) {
    // New is the allocation instruction, pointer typed. AI is the original
    // allocation instruction, also pointer typed. Thus, cast to use is BitCast.
    Value *NewCast = Builder.CreateBitCast(New, AI.getType(), "tmpcast");
    replaceInstUsesWith(AI, NewCast);
    eraseInstFromFunction(AI);
  }
  return replaceInstUsesWith(CI, New);
}

/// Given an expression that CanEvaluateTruncated or CanEvaluateSExtd returns
/// true for, actually insert the code to evaluate the expression.
Value *InstCombinerImpl::EvaluateInDifferentType(Value *V, Type *Ty,
                                                 bool isSigned) {
  if (Constant *C = dyn_cast<Constant>(V)) {
    C = ConstantExpr::getIntegerCast(C, Ty, isSigned /*Sext or ZExt*/);
    // If we got a constantexpr back, try to simplify it with DL info.
    return ConstantFoldConstant(C, DL, &TLI);
  }

  // Otherwise, it must be an instruction.
  Instruction *I = cast<Instruction>(V);
  Instruction *Res = nullptr;
  unsigned Opc = I->getOpcode();
  switch (Opc) {
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::AShr:
  case Instruction::LShr:
  case Instruction::Shl:
  case Instruction::UDiv:
  case Instruction::URem: {
    Value *LHS = EvaluateInDifferentType(I->getOperand(0), Ty, isSigned);
    Value *RHS = EvaluateInDifferentType(I->getOperand(1), Ty, isSigned);
    Res = BinaryOperator::Create((Instruction::BinaryOps)Opc, LHS, RHS);
    break;
  }
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
    // If the source type of the cast is the type we're trying for then we can
    // just return the source.  There's no need to insert it because it is not
    // new.
    if (I->getOperand(0)->getType() == Ty)
      return I->getOperand(0);

    // Otherwise, must be the same type of cast, so just reinsert a new one.
    // This also handles the case of zext(trunc(x)) -> zext(x).
    Res = CastInst::CreateIntegerCast(I->getOperand(0), Ty,
                                      Opc == Instruction::SExt);
    break;
  case Instruction::Select: {
    Value *True = EvaluateInDifferentType(I->getOperand(1), Ty, isSigned);
    Value *False = EvaluateInDifferentType(I->getOperand(2), Ty, isSigned);
    Res = SelectInst::Create(I->getOperand(0), True, False);
    break;
  }
  case Instruction::PHI: {
    PHINode *OPN = cast<PHINode>(I);
    PHINode *NPN = PHINode::Create(Ty, OPN->getNumIncomingValues());
    for (unsigned i = 0, e = OPN->getNumIncomingValues(); i != e; ++i) {
      Value *V =
          EvaluateInDifferentType(OPN->getIncomingValue(i), Ty, isSigned);
      NPN->addIncoming(V, OPN->getIncomingBlock(i));
    }
    Res = NPN;
    break;
  }
  default:
    // TODO: Can handle more cases here.
    llvm_unreachable("Unreachable!");
  }

  Res->takeName(I);
  return InsertNewInstWith(Res, *I);
}

Instruction::CastOps
InstCombinerImpl::isEliminableCastPair(const CastInst *CI1,
                                       const CastInst *CI2) {
  Type *SrcTy = CI1->getSrcTy();
  Type *MidTy = CI1->getDestTy();
  Type *DstTy = CI2->getDestTy();

  Instruction::CastOps firstOp = CI1->getOpcode();
  Instruction::CastOps secondOp = CI2->getOpcode();
  Type *SrcIntPtrTy =
      SrcTy->isPtrOrPtrVectorTy() ? DL.getIntPtrType(SrcTy) : nullptr;
  Type *MidIntPtrTy =
      MidTy->isPtrOrPtrVectorTy() ? DL.getIntPtrType(MidTy) : nullptr;
  Type *DstIntPtrTy =
      DstTy->isPtrOrPtrVectorTy() ? DL.getIntPtrType(DstTy) : nullptr;
  unsigned Res = CastInst::isEliminableCastPair(firstOp, secondOp, SrcTy, MidTy,
                                                DstTy, SrcIntPtrTy, MidIntPtrTy,
                                                DstIntPtrTy);

  // We don't want to form an inttoptr or ptrtoint that converts to an integer
  // type that differs from the pointer size.
  if ((Res == Instruction::IntToPtr && SrcTy != DstIntPtrTy) ||
      (Res == Instruction::PtrToInt && DstTy != SrcIntPtrTy))
    Res = 0;

  return Instruction::CastOps(Res);
}

/// Implement the transforms common to all CastInst visitors.
Instruction *InstCombinerImpl::commonCastTransforms(CastInst &CI) {
  Value *Src = CI.getOperand(0);
  Type *Ty = CI.getType();

  // Try to eliminate a cast of a cast.
  if (auto *CSrc = dyn_cast<CastInst>(Src)) {   // A->B->C cast
    if (Instruction::CastOps NewOpc = isEliminableCastPair(CSrc, &CI)) {
      // The first cast (CSrc) is eliminable so we need to fix up or replace
      // the second cast (CI). CSrc will then have a good chance of being dead.
      auto *Res = CastInst::Create(NewOpc, CSrc->getOperand(0), Ty);
      // Point debug users of the dying cast to the new one.
      if (CSrc->hasOneUse())
        replaceAllDbgUsesWith(*CSrc, *Res, CI, DT);
      return Res;
    }
  }

  if (auto *Sel = dyn_cast<SelectInst>(Src)) {
    // We are casting a select. Try to fold the cast into the select if the
    // select does not have a compare instruction with matching operand types
    // or the select is likely better done in a narrow type.
    // Creating a select with operands that are different sizes than its
    // condition may inhibit other folds and lead to worse codegen.
    auto *Cmp = dyn_cast<CmpInst>(Sel->getCondition());
    if (!Cmp || Cmp->getOperand(0)->getType() != Sel->getType() ||
        (CI.getOpcode() == Instruction::Trunc &&
         shouldChangeType(CI.getSrcTy(), CI.getType()))) {
      if (Instruction *NV = FoldOpIntoSelect(CI, Sel)) {
        replaceAllDbgUsesWith(*Sel, *NV, CI, DT);
        return NV;
      }
    }
  }

  // If we are casting a PHI, then fold the cast into the PHI.
  if (auto *PN = dyn_cast<PHINode>(Src)) {
    // Don't do this if it would create a PHI node with an illegal type from a
    // legal type.
    if (!Src->getType()->isIntegerTy() || !CI.getType()->isIntegerTy() ||
        shouldChangeType(CI.getSrcTy(), CI.getType()))
      if (Instruction *NV = foldOpIntoPhi(CI, PN))
        return NV;
  }

  // Canonicalize a unary shuffle after the cast if neither operation changes
  // the size or element size of the input vector.
  // TODO: We could allow size-changing ops if that doesn't harm codegen.
  // cast (shuffle X, Mask) --> shuffle (cast X), Mask
  Value *X;
  ArrayRef<int> Mask;
  if (match(Src, m_OneUse(m_Shuffle(m_Value(X), m_Undef(), m_Mask(Mask))))) {
    // TODO: Allow scalable vectors?
    auto *SrcTy = dyn_cast<FixedVectorType>(X->getType());
    auto *DestTy = dyn_cast<FixedVectorType>(Ty);
    if (SrcTy && DestTy &&
        SrcTy->getNumElements() == DestTy->getNumElements() &&
        SrcTy->getPrimitiveSizeInBits() == DestTy->getPrimitiveSizeInBits()) {
      Value *CastX = Builder.CreateCast(CI.getOpcode(), X, DestTy);
      return new ShuffleVectorInst(CastX, Mask);
    }
  }

  return nullptr;
}

/// Constants and extensions/truncates from the destination type are always
/// free to be evaluated in that type. This is a helper for canEvaluate*.
static bool canAlwaysEvaluateInType(Value *V, Type *Ty) {
  if (isa<Constant>(V))
    return true;
  Value *X;
  if ((match(V, m_ZExtOrSExt(m_Value(X))) || match(V, m_Trunc(m_Value(X)))) &&
      X->getType() == Ty)
    return true;

  return false;
}

/// Filter out values that we can not evaluate in the destination type for free.
/// This is a helper for canEvaluate*.
static bool canNotEvaluateInType(Value *V, Type *Ty) {
  assert(!isa<Constant>(V) && "Constant should already be handled.");
  if (!isa<Instruction>(V))
    return true;
  // We don't extend or shrink something that has multiple uses --  doing so
  // would require duplicating the instruction which isn't profitable.
  if (!V->hasOneUse())
    return true;

  return false;
}

/// Return true if we can evaluate the specified expression tree as type Ty
/// instead of its larger type, and arrive with the same value.
/// This is used by code that tries to eliminate truncates.
///
/// Ty will always be a type smaller than V.  We should return true if trunc(V)
/// can be computed by computing V in the smaller type.  If V is an instruction,
/// then trunc(inst(x,y)) can be computed as inst(trunc(x),trunc(y)), which only
/// makes sense if x and y can be efficiently truncated.
///
/// This function works on both vectors and scalars.
///
static bool canEvaluateTruncated(Value *V, Type *Ty, InstCombinerImpl &IC,
                                 Instruction *CxtI) {
  if (canAlwaysEvaluateInType(V, Ty))
    return true;
  if (canNotEvaluateInType(V, Ty))
    return false;

  auto *I = cast<Instruction>(V);
  Type *OrigTy = V->getType();
  switch (I->getOpcode()) {
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    // These operators can all arbitrarily be extended or truncated.
    return canEvaluateTruncated(I->getOperand(0), Ty, IC, CxtI) &&
           canEvaluateTruncated(I->getOperand(1), Ty, IC, CxtI);

  case Instruction::UDiv:
  case Instruction::URem: {
    // UDiv and URem can be truncated if all the truncated bits are zero.
    uint32_t OrigBitWidth = OrigTy->getScalarSizeInBits();
    uint32_t BitWidth = Ty->getScalarSizeInBits();
    assert(BitWidth < OrigBitWidth && "Unexpected bitwidths!");
    APInt Mask = APInt::getBitsSetFrom(OrigBitWidth, BitWidth);
    if (IC.MaskedValueIsZero(I->getOperand(0), Mask, 0, CxtI) &&
        IC.MaskedValueIsZero(I->getOperand(1), Mask, 0, CxtI)) {
      return canEvaluateTruncated(I->getOperand(0), Ty, IC, CxtI) &&
             canEvaluateTruncated(I->getOperand(1), Ty, IC, CxtI);
    }
    break;
  }
  case Instruction::Shl: {
    // If we are truncating the result of this SHL, and if it's a shift of an
    // inrange amount, we can always perform a SHL in a smaller type.
    uint32_t BitWidth = Ty->getScalarSizeInBits();
    KnownBits AmtKnownBits =
        llvm::computeKnownBits(I->getOperand(1), IC.getDataLayout());
    if (AmtKnownBits.getMaxValue().ult(BitWidth))
      return canEvaluateTruncated(I->getOperand(0), Ty, IC, CxtI) &&
             canEvaluateTruncated(I->getOperand(1), Ty, IC, CxtI);
    break;
  }
  case Instruction::LShr: {
    // If this is a truncate of a logical shr, we can truncate it to a smaller
    // lshr iff we know that the bits we would otherwise be shifting in are
    // already zeros.
    // TODO: It is enough to check that the bits we would be shifting in are
    //       zero - use AmtKnownBits.getMaxValue().
    uint32_t OrigBitWidth = OrigTy->getScalarSizeInBits();
    uint32_t BitWidth = Ty->getScalarSizeInBits();
    KnownBits AmtKnownBits =
        llvm::computeKnownBits(I->getOperand(1), IC.getDataLayout());
    APInt ShiftedBits = APInt::getBitsSetFrom(OrigBitWidth, BitWidth);
    if (AmtKnownBits.getMaxValue().ult(BitWidth) &&
        IC.MaskedValueIsZero(I->getOperand(0), ShiftedBits, 0, CxtI)) {
      return canEvaluateTruncated(I->getOperand(0), Ty, IC, CxtI) &&
             canEvaluateTruncated(I->getOperand(1), Ty, IC, CxtI);
    }
    break;
  }
  case Instruction::AShr: {
    // If this is a truncate of an arithmetic shr, we can truncate it to a
    // smaller ashr iff we know that all the bits from the sign bit of the
    // original type and the sign bit of the truncate type are similar.
    // TODO: It is enough to check that the bits we would be shifting in are
    //       similar to sign bit of the truncate type.
    uint32_t OrigBitWidth = OrigTy->getScalarSizeInBits();
    uint32_t BitWidth = Ty->getScalarSizeInBits();
    KnownBits AmtKnownBits =
        llvm::computeKnownBits(I->getOperand(1), IC.getDataLayout());
    unsigned ShiftedBits = OrigBitWidth - BitWidth;
    if (AmtKnownBits.getMaxValue().ult(BitWidth) &&
        ShiftedBits < IC.ComputeNumSignBits(I->getOperand(0), 0, CxtI))
      return canEvaluateTruncated(I->getOperand(0), Ty, IC, CxtI) &&
             canEvaluateTruncated(I->getOperand(1), Ty, IC, CxtI);
    break;
  }
  case Instruction::Trunc:
    // trunc(trunc(x)) -> trunc(x)
    return true;
  case Instruction::ZExt:
  case Instruction::SExt:
    // trunc(ext(x)) -> ext(x) if the source type is smaller than the new dest
    // trunc(ext(x)) -> trunc(x) if the source type is larger than the new dest
    return true;
  case Instruction::Select: {
    SelectInst *SI = cast<SelectInst>(I);
    return canEvaluateTruncated(SI->getTrueValue(), Ty, IC, CxtI) &&
           canEvaluateTruncated(SI->getFalseValue(), Ty, IC, CxtI);
  }
  case Instruction::PHI: {
    // We can change a phi if we can change all operands.  Note that we never
    // get into trouble with cyclic PHIs here because we only consider
    // instructions with a single use.
    PHINode *PN = cast<PHINode>(I);
    for (Value *IncValue : PN->incoming_values())
      if (!canEvaluateTruncated(IncValue, Ty, IC, CxtI))
        return false;
    return true;
  }
  default:
    // TODO: Can handle more cases here.
    break;
  }

  return false;
}

/// Given a vector that is bitcast to an integer, optionally logically
/// right-shifted, and truncated, convert it to an extractelement.
/// Example (big endian):
///   trunc (lshr (bitcast <4 x i32> %X to i128), 32) to i32
///   --->
///   extractelement <4 x i32> %X, 1
static Instruction *foldVecTruncToExtElt(TruncInst &Trunc,
                                         InstCombinerImpl &IC) {
  Value *TruncOp = Trunc.getOperand(0);
  Type *DestType = Trunc.getType();
  if (!TruncOp->hasOneUse() || !isa<IntegerType>(DestType))
    return nullptr;

  Value *VecInput = nullptr;
  ConstantInt *ShiftVal = nullptr;
  if (!match(TruncOp, m_CombineOr(m_BitCast(m_Value(VecInput)),
                                  m_LShr(m_BitCast(m_Value(VecInput)),
                                         m_ConstantInt(ShiftVal)))) ||
      !isa<VectorType>(VecInput->getType()))
    return nullptr;

  VectorType *VecType = cast<VectorType>(VecInput->getType());
  unsigned VecWidth = VecType->getPrimitiveSizeInBits();
  unsigned DestWidth = DestType->getPrimitiveSizeInBits();
  unsigned ShiftAmount = ShiftVal ? ShiftVal->getZExtValue() : 0;

  if ((VecWidth % DestWidth != 0) || (ShiftAmount % DestWidth != 0))
    return nullptr;

  // If the element type of the vector doesn't match the result type,
  // bitcast it to a vector type that we can extract from.
  unsigned NumVecElts = VecWidth / DestWidth;
  if (VecType->getElementType() != DestType) {
    VecType = FixedVectorType::get(DestType, NumVecElts);
    VecInput = IC.Builder.CreateBitCast(VecInput, VecType, "bc");
  }

  unsigned Elt = ShiftAmount / DestWidth;
  if (IC.getDataLayout().isBigEndian())
    Elt = NumVecElts - 1 - Elt;

  return ExtractElementInst::Create(VecInput, IC.Builder.getInt32(Elt));
}

/// Funnel/Rotate left/right may occur in a wider type than necessary because of
/// type promotion rules. Try to narrow the inputs and convert to funnel shift.
Instruction *InstCombinerImpl::narrowFunnelShift(TruncInst &Trunc) {
  assert((isa<VectorType>(Trunc.getSrcTy()) ||
          shouldChangeType(Trunc.getSrcTy(), Trunc.getType())) &&
         "Don't narrow to an illegal scalar type");

  // Bail out on strange types. It is possible to handle some of these patterns
  // even with non-power-of-2 sizes, but it is not a likely scenario.
  Type *DestTy = Trunc.getType();
  unsigned NarrowWidth = DestTy->getScalarSizeInBits();
  unsigned WideWidth = Trunc.getSrcTy()->getScalarSizeInBits();
  if (!isPowerOf2_32(NarrowWidth))
    return nullptr;

  // First, find an or'd pair of opposite shifts:
  // trunc (or (lshr ShVal0, ShAmt0), (shl ShVal1, ShAmt1))
  BinaryOperator *Or0, *Or1;
  if (!match(Trunc.getOperand(0), m_OneUse(m_Or(m_BinOp(Or0), m_BinOp(Or1)))))
    return nullptr;

  Value *ShVal0, *ShVal1, *ShAmt0, *ShAmt1;
  if (!match(Or0, m_OneUse(m_LogicalShift(m_Value(ShVal0), m_Value(ShAmt0)))) ||
      !match(Or1, m_OneUse(m_LogicalShift(m_Value(ShVal1), m_Value(ShAmt1)))) ||
      Or0->getOpcode() == Or1->getOpcode())
    return nullptr;

  // Canonicalize to or(shl(ShVal0, ShAmt0), lshr(ShVal1, ShAmt1)).
  if (Or0->getOpcode() == BinaryOperator::LShr) {
    std::swap(Or0, Or1);
    std::swap(ShVal0, ShVal1);
    std::swap(ShAmt0, ShAmt1);
  }
  assert(Or0->getOpcode() == BinaryOperator::Shl &&
         Or1->getOpcode() == BinaryOperator::LShr &&
         "Illegal or(shift,shift) pair");

  // Match the shift amount operands for a funnel/rotate pattern. This always
  // matches a subtraction on the R operand.
  auto matchShiftAmount = [&](Value *L, Value *R, unsigned Width) -> Value * {
    // The shift amounts may add up to the narrow bit width:
    // (shl ShVal0, L) | (lshr ShVal1, Width - L)
    // If this is a funnel shift (different operands are shifted), then the
    // shift amount can not over-shift (create poison) in the narrow type.
    unsigned MaxShiftAmountWidth = Log2_32(NarrowWidth);
    APInt HiBitMask = ~APInt::getLowBitsSet(WideWidth, MaxShiftAmountWidth);
    if (ShVal0 == ShVal1 || MaskedValueIsZero(L, HiBitMask))
      if (match(R, m_OneUse(m_Sub(m_SpecificInt(Width), m_Specific(L)))))
        return L;

    // The following patterns currently only work for rotation patterns.
    // TODO: Add more general funnel-shift compatible patterns.
    if (ShVal0 != ShVal1)
      return nullptr;

    // The shift amount may be masked with negation:
    // (shl ShVal0, (X & (Width - 1))) | (lshr ShVal1, ((-X) & (Width - 1)))
    Value *X;
    unsigned Mask = Width - 1;
    if (match(L, m_And(m_Value(X), m_SpecificInt(Mask))) &&
        match(R, m_And(m_Neg(m_Specific(X)), m_SpecificInt(Mask))))
      return X;

    // Same as above, but the shift amount may be extended after masking:
    if (match(L, m_ZExt(m_And(m_Value(X), m_SpecificInt(Mask)))) &&
        match(R, m_ZExt(m_And(m_Neg(m_Specific(X)), m_SpecificInt(Mask)))))
      return X;

    return nullptr;
  };

  Value *ShAmt = matchShiftAmount(ShAmt0, ShAmt1, NarrowWidth);
  bool IsFshl = true; // Sub on LSHR.
  if (!ShAmt) {
    ShAmt = matchShiftAmount(ShAmt1, ShAmt0, NarrowWidth);
    IsFshl = false; // Sub on SHL.
  }
  if (!ShAmt)
    return nullptr;

  // The right-shifted value must have high zeros in the wide type (for example
  // from 'zext', 'and' or 'shift'). High bits of the left-shifted value are
  // truncated, so those do not matter.
  APInt HiBitMask = APInt::getHighBitsSet(WideWidth, WideWidth - NarrowWidth);
  if (!MaskedValueIsZero(ShVal1, HiBitMask, 0, &Trunc))
    return nullptr;

  // We have an unnecessarily wide rotate!
  // trunc (or (shl ShVal0, ShAmt), (lshr ShVal1, BitWidth - ShAmt))
  // Narrow the inputs and convert to funnel shift intrinsic:
  // llvm.fshl.i8(trunc(ShVal), trunc(ShVal), trunc(ShAmt))
  Value *NarrowShAmt = Builder.CreateTrunc(ShAmt, DestTy);
  Value *X, *Y;
  X = Y = Builder.CreateTrunc(ShVal0, DestTy);
  if (ShVal0 != ShVal1)
    Y = Builder.CreateTrunc(ShVal1, DestTy);
  Intrinsic::ID IID = IsFshl ? Intrinsic::fshl : Intrinsic::fshr;
  Function *F = Intrinsic::getDeclaration(Trunc.getModule(), IID, DestTy);
  return CallInst::Create(F, {X, Y, NarrowShAmt});
}

/// Try to narrow the width of math or bitwise logic instructions by pulling a
/// truncate ahead of binary operators.
/// TODO: Transforms for truncated shifts should be moved into here.
Instruction *InstCombinerImpl::narrowBinOp(TruncInst &Trunc) {
  Type *SrcTy = Trunc.getSrcTy();
  Type *DestTy = Trunc.getType();
  if (!isa<VectorType>(SrcTy) && !shouldChangeType(SrcTy, DestTy))
    return nullptr;

  BinaryOperator *BinOp;
  if (!match(Trunc.getOperand(0), m_OneUse(m_BinOp(BinOp))))
    return nullptr;

  Value *BinOp0 = BinOp->getOperand(0);
  Value *BinOp1 = BinOp->getOperand(1);
  switch (BinOp->getOpcode()) {
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul: {
    Constant *C;
    if (match(BinOp0, m_Constant(C))) {
      // trunc (binop C, X) --> binop (trunc C', X)
      Constant *NarrowC = ConstantExpr::getTrunc(C, DestTy);
      Value *TruncX = Builder.CreateTrunc(BinOp1, DestTy);
      return BinaryOperator::Create(BinOp->getOpcode(), NarrowC, TruncX);
    }
    if (match(BinOp1, m_Constant(C))) {
      // trunc (binop X, C) --> binop (trunc X, C')
      Constant *NarrowC = ConstantExpr::getTrunc(C, DestTy);
      Value *TruncX = Builder.CreateTrunc(BinOp0, DestTy);
      return BinaryOperator::Create(BinOp->getOpcode(), TruncX, NarrowC);
    }
    Value *X;
    if (match(BinOp0, m_ZExtOrSExt(m_Value(X))) && X->getType() == DestTy) {
      // trunc (binop (ext X), Y) --> binop X, (trunc Y)
      Value *NarrowOp1 = Builder.CreateTrunc(BinOp1, DestTy);
      return BinaryOperator::Create(BinOp->getOpcode(), X, NarrowOp1);
    }
    if (match(BinOp1, m_ZExtOrSExt(m_Value(X))) && X->getType() == DestTy) {
      // trunc (binop Y, (ext X)) --> binop (trunc Y), X
      Value *NarrowOp0 = Builder.CreateTrunc(BinOp0, DestTy);
      return BinaryOperator::Create(BinOp->getOpcode(), NarrowOp0, X);
    }
    break;
  }

  default: break;
  }

  if (Instruction *NarrowOr = narrowFunnelShift(Trunc))
    return NarrowOr;

  return nullptr;
}

/// Try to narrow the width of a splat shuffle. This could be generalized to any
/// shuffle with a constant operand, but we limit the transform to avoid
/// creating a shuffle type that targets may not be able to lower effectively.
static Instruction *shrinkSplatShuffle(TruncInst &Trunc,
                                       InstCombiner::BuilderTy &Builder) {
  auto *Shuf = dyn_cast<ShuffleVectorInst>(Trunc.getOperand(0));
  if (Shuf && Shuf->hasOneUse() && match(Shuf->getOperand(1), m_Undef()) &&
      is_splat(Shuf->getShuffleMask()) &&
      Shuf->getType() == Shuf->getOperand(0)->getType()) {
    // trunc (shuf X, Undef, SplatMask) --> shuf (trunc X), Poison, SplatMask
    // trunc (shuf X, Poison, SplatMask) --> shuf (trunc X), Poison, SplatMask
    Value *NarrowOp = Builder.CreateTrunc(Shuf->getOperand(0), Trunc.getType());
    return new ShuffleVectorInst(NarrowOp, Shuf->getShuffleMask());
  }

  return nullptr;
}

/// Try to narrow the width of an insert element. This could be generalized for
/// any vector constant, but we limit the transform to insertion into undef to
/// avoid potential backend problems from unsupported insertion widths. This
/// could also be extended to handle the case of inserting a scalar constant
/// into a vector variable.
static Instruction *shrinkInsertElt(CastInst &Trunc,
                                    InstCombiner::BuilderTy &Builder) {
  Instruction::CastOps Opcode = Trunc.getOpcode();
  assert((Opcode == Instruction::Trunc || Opcode == Instruction::FPTrunc) &&
         "Unexpected instruction for shrinking");

  auto *InsElt = dyn_cast<InsertElementInst>(Trunc.getOperand(0));
  if (!InsElt || !InsElt->hasOneUse())
    return nullptr;

  Type *DestTy = Trunc.getType();
  Type *DestScalarTy = DestTy->getScalarType();
  Value *VecOp = InsElt->getOperand(0);
  Value *ScalarOp = InsElt->getOperand(1);
  Value *Index = InsElt->getOperand(2);

  if (match(VecOp, m_Undef())) {
    // trunc   (inselt undef, X, Index) --> inselt undef,   (trunc X), Index
    // fptrunc (inselt undef, X, Index) --> inselt undef, (fptrunc X), Index
    UndefValue *NarrowUndef = UndefValue::get(DestTy);
    Value *NarrowOp = Builder.CreateCast(Opcode, ScalarOp, DestScalarTy);
    return InsertElementInst::Create(NarrowUndef, NarrowOp, Index);
  }

  return nullptr;
}

Instruction *InstCombinerImpl::visitTrunc(TruncInst &Trunc) {
  if (Instruction *Result = commonCastTransforms(Trunc))
    return Result;

  Value *Src = Trunc.getOperand(0);
  Type *DestTy = Trunc.getType(), *SrcTy = Src->getType();
  unsigned DestWidth = DestTy->getScalarSizeInBits();
  unsigned SrcWidth = SrcTy->getScalarSizeInBits();

  // Attempt to truncate the entire input expression tree to the destination
  // type.   Only do this if the dest type is a simple type, don't convert the
  // expression tree to something weird like i93 unless the source is also
  // strange.
  if ((DestTy->isVectorTy() || shouldChangeType(SrcTy, DestTy)) &&
      canEvaluateTruncated(Src, DestTy, *this, &Trunc)) {

    // If this cast is a truncate, evaluting in a different type always
    // eliminates the cast, so it is always a win.
    LLVM_DEBUG(
        dbgs() << "ICE: EvaluateInDifferentType converting expression type"
                  " to avoid cast: "
               << Trunc << '\n');
    Value *Res = EvaluateInDifferentType(Src, DestTy, false);
    assert(Res->getType() == DestTy);
    return replaceInstUsesWith(Trunc, Res);
  }

  // For integer types, check if we can shorten the entire input expression to
  // DestWidth * 2, which won't allow removing the truncate, but reducing the
  // width may enable further optimizations, e.g. allowing for larger
  // vectorization factors.
  if (auto *DestITy = dyn_cast<IntegerType>(DestTy)) {
    if (DestWidth * 2 < SrcWidth) {
      auto *NewDestTy = DestITy->getExtendedType();
      if (shouldChangeType(SrcTy, NewDestTy) &&
          canEvaluateTruncated(Src, NewDestTy, *this, &Trunc)) {
        LLVM_DEBUG(
            dbgs() << "ICE: EvaluateInDifferentType converting expression type"
                      " to reduce the width of operand of"
                   << Trunc << '\n');
        Value *Res = EvaluateInDifferentType(Src, NewDestTy, false);
        return new TruncInst(Res, DestTy);
      }
    }
  }

  // Test if the trunc is the user of a select which is part of a
  // minimum or maximum operation. If so, don't do any more simplification.
  // Even simplifying demanded bits can break the canonical form of a
  // min/max.
  Value *LHS, *RHS;
  if (SelectInst *Sel = dyn_cast<SelectInst>(Src))
    if (matchSelectPattern(Sel, LHS, RHS).Flavor != SPF_UNKNOWN)
      return nullptr;

  // See if we can simplify any instructions used by the input whose sole
  // purpose is to compute bits we don't care about.
  if (SimplifyDemandedInstructionBits(Trunc))
    return &Trunc;

  if (DestWidth == 1) {
    Value *Zero = Constant::getNullValue(SrcTy);
    if (DestTy->isIntegerTy()) {
      // Canonicalize trunc x to i1 -> icmp ne (and x, 1), 0 (scalar only).
      // TODO: We canonicalize to more instructions here because we are probably
      // lacking equivalent analysis for trunc relative to icmp. There may also
      // be codegen concerns. If those trunc limitations were removed, we could
      // remove this transform.
      Value *And = Builder.CreateAnd(Src, ConstantInt::get(SrcTy, 1));
      return new ICmpInst(ICmpInst::ICMP_NE, And, Zero);
    }

    // For vectors, we do not canonicalize all truncs to icmp, so optimize
    // patterns that would be covered within visitICmpInst.
    Value *X;
    Constant *C;
    if (match(Src, m_OneUse(m_LShr(m_Value(X), m_Constant(C))))) {
      // trunc (lshr X, C) to i1 --> icmp ne (and X, C'), 0
      Constant *One = ConstantInt::get(SrcTy, APInt(SrcWidth, 1));
      Constant *MaskC = ConstantExpr::getShl(One, C);
      Value *And = Builder.CreateAnd(X, MaskC);
      return new ICmpInst(ICmpInst::ICMP_NE, And, Zero);
    }
    if (match(Src, m_OneUse(m_c_Or(m_LShr(m_Value(X), m_Constant(C)),
                                   m_Deferred(X))))) {
      // trunc (or (lshr X, C), X) to i1 --> icmp ne (and X, C'), 0
      Constant *One = ConstantInt::get(SrcTy, APInt(SrcWidth, 1));
      Constant *MaskC = ConstantExpr::getShl(One, C);
      MaskC = ConstantExpr::getOr(MaskC, One);
      Value *And = Builder.CreateAnd(X, MaskC);
      return new ICmpInst(ICmpInst::ICMP_NE, And, Zero);
    }
  }

  Value *A, *B;
  Constant *C;
  if (match(Src, m_LShr(m_SExt(m_Value(A)), m_Constant(C)))) {
    unsigned AWidth = A->getType()->getScalarSizeInBits();
    unsigned MaxShiftAmt = SrcWidth - std::max(DestWidth, AWidth);
    auto *OldSh = cast<Instruction>(Src);
    bool IsExact = OldSh->isExact();

    // If the shift is small enough, all zero bits created by the shift are
    // removed by the trunc.
    if (match(C, m_SpecificInt_ICMP(ICmpInst::ICMP_ULE,
                                    APInt(SrcWidth, MaxShiftAmt)))) {
      // trunc (lshr (sext A), C) --> ashr A, C
      if (A->getType() == DestTy) {
        Constant *MaxAmt = ConstantInt::get(SrcTy, DestWidth - 1, false);
        Constant *ShAmt = ConstantExpr::getUMin(C, MaxAmt);
        ShAmt = ConstantExpr::getTrunc(ShAmt, A->getType());
        ShAmt = Constant::mergeUndefsWith(ShAmt, C);
        return IsExact ? BinaryOperator::CreateExactAShr(A, ShAmt)
                       : BinaryOperator::CreateAShr(A, ShAmt);
      }
      // The types are mismatched, so create a cast after shifting:
      // trunc (lshr (sext A), C) --> sext/trunc (ashr A, C)
      if (Src->hasOneUse()) {
        Constant *MaxAmt = ConstantInt::get(SrcTy, AWidth - 1, false);
        Constant *ShAmt = ConstantExpr::getUMin(C, MaxAmt);
        ShAmt = ConstantExpr::getTrunc(ShAmt, A->getType());
        Value *Shift = Builder.CreateAShr(A, ShAmt, "", IsExact);
        return CastInst::CreateIntegerCast(Shift, DestTy, true);
      }
    }
    // TODO: Mask high bits with 'and'.
  }

  // trunc (*shr (trunc A), C) --> trunc(*shr A, C)
  if (match(Src, m_OneUse(m_Shr(m_Trunc(m_Value(A)), m_Constant(C))))) {
    unsigned MaxShiftAmt = SrcWidth - DestWidth;

    // If the shift is small enough, all zero/sign bits created by the shift are
    // removed by the trunc.
    if (match(C, m_SpecificInt_ICMP(ICmpInst::ICMP_ULE,
                                    APInt(SrcWidth, MaxShiftAmt)))) {
      auto *OldShift = cast<Instruction>(Src);
      bool IsExact = OldShift->isExact();
      auto *ShAmt = ConstantExpr::getIntegerCast(C, A->getType(), true);
      ShAmt = Constant::mergeUndefsWith(ShAmt, C);
      Value *Shift =
          OldShift->getOpcode() == Instruction::AShr
              ? Builder.CreateAShr(A, ShAmt, OldShift->getName(), IsExact)
              : Builder.CreateLShr(A, ShAmt, OldShift->getName(), IsExact);
      return CastInst::CreateTruncOrBitCast(Shift, DestTy);
    }
  }

  if (Instruction *I = narrowBinOp(Trunc))
    return I;

  if (Instruction *I = shrinkSplatShuffle(Trunc, Builder))
    return I;

  if (Instruction *I = shrinkInsertElt(Trunc, Builder))
    return I;

  if (Src->hasOneUse() &&
      (isa<VectorType>(SrcTy) || shouldChangeType(SrcTy, DestTy))) {
    // Transform "trunc (shl X, cst)" -> "shl (trunc X), cst" so long as the
    // dest type is native and cst < dest size.
    if (match(Src, m_Shl(m_Value(A), m_Constant(C))) &&
        !match(A, m_Shr(m_Value(), m_Constant()))) {
      // Skip shifts of shift by constants. It undoes a combine in
      // FoldShiftByConstant and is the extend in reg pattern.
      APInt Threshold = APInt(C->getType()->getScalarSizeInBits(), DestWidth);
      if (match(C, m_SpecificInt_ICMP(ICmpInst::ICMP_ULT, Threshold))) {
        Value *NewTrunc = Builder.CreateTrunc(A, DestTy, A->getName() + ".tr");
        return BinaryOperator::Create(Instruction::Shl, NewTrunc,
                                      ConstantExpr::getTrunc(C, DestTy));
      }
    }
  }

  if (Instruction *I = foldVecTruncToExtElt(Trunc, *this))
    return I;

  // Whenever an element is extracted from a vector, and then truncated,
  // canonicalize by converting it to a bitcast followed by an
  // extractelement.
  //
  // Example (little endian):
  //   trunc (extractelement <4 x i64> %X, 0) to i32
  //   --->
  //   extractelement <8 x i32> (bitcast <4 x i64> %X to <8 x i32>), i32 0
  Value *VecOp;
  ConstantInt *Cst;
  if (match(Src, m_OneUse(m_ExtractElt(m_Value(VecOp), m_ConstantInt(Cst))))) {
    auto *VecOpTy = cast<VectorType>(VecOp->getType());
    auto VecElts = VecOpTy->getElementCount();

    // A badly fit destination size would result in an invalid cast.
    if (SrcWidth % DestWidth == 0) {
      uint64_t TruncRatio = SrcWidth / DestWidth;
      uint64_t BitCastNumElts = VecElts.getKnownMinValue() * TruncRatio;
      uint64_t VecOpIdx = Cst->getZExtValue();
      uint64_t NewIdx = DL.isBigEndian() ? (VecOpIdx + 1) * TruncRatio - 1
                                         : VecOpIdx * TruncRatio;
      assert(BitCastNumElts <= std::numeric_limits<uint32_t>::max() &&
             "overflow 32-bits");

      auto *BitCastTo =
          VectorType::get(DestTy, BitCastNumElts, VecElts.isScalable());
      Value *BitCast = Builder.CreateBitCast(VecOp, BitCastTo);
      return ExtractElementInst::Create(BitCast, Builder.getInt32(NewIdx));
    }
  }

  // trunc (ctlz_i32(zext(A), B) --> add(ctlz_i16(A, B), C)
  if (match(Src, m_OneUse(m_Intrinsic<Intrinsic::ctlz>(m_ZExt(m_Value(A)),
                                                       m_Value(B))))) {
    unsigned AWidth = A->getType()->getScalarSizeInBits();
    if (AWidth == DestWidth && AWidth > Log2_32(SrcWidth)) {
      Value *WidthDiff = ConstantInt::get(A->getType(), SrcWidth - AWidth);
      Value *NarrowCtlz =
          Builder.CreateIntrinsic(Intrinsic::ctlz, {Trunc.getType()}, {A, B});
      return BinaryOperator::CreateAdd(NarrowCtlz, WidthDiff);
    }
  }

  if (match(Src, m_VScale(DL))) {
    if (Trunc.getFunction() &&
        Trunc.getFunction()->hasFnAttribute(Attribute::VScaleRange)) {
      Attribute Attr =
          Trunc.getFunction()->getFnAttribute(Attribute::VScaleRange);
      if (Optional<unsigned> MaxVScale = Attr.getVScaleRangeMax()) {
        if (Log2_32(MaxVScale.getValue()) < DestWidth) {
          Value *VScale = Builder.CreateVScale(ConstantInt::get(DestTy, 1));
          return replaceInstUsesWith(Trunc, VScale);
        }
      }
    }
  }

  return nullptr;
}

Instruction *InstCombinerImpl::transformZExtICmp(ICmpInst *Cmp, ZExtInst &Zext) {
  // If we are just checking for a icmp eq of a single bit and zext'ing it
  // to an integer, then shift the bit to the appropriate place and then
  // cast to integer to avoid the comparison.
  const APInt *Op1CV;
  if (match(Cmp->getOperand(1), m_APInt(Op1CV))) {

    // zext (x <s  0) to i32 --> x>>u31      true if signbit set.
    // zext (x >s -1) to i32 --> (x>>u31)^1  true if signbit clear.
    if ((Cmp->getPredicate() == ICmpInst::ICMP_SLT && Op1CV->isZero()) ||
        (Cmp->getPredicate() == ICmpInst::ICMP_SGT && Op1CV->isAllOnes())) {
      Value *In = Cmp->getOperand(0);
      Value *Sh = ConstantInt::get(In->getType(),
                                   In->getType()->getScalarSizeInBits() - 1);
      In = Builder.CreateLShr(In, Sh, In->getName() + ".lobit");
      if (In->getType() != Zext.getType())
        In = Builder.CreateIntCast(In, Zext.getType(), false /*ZExt*/);

      if (Cmp->getPredicate() == ICmpInst::ICMP_SGT) {
        Constant *One = ConstantInt::get(In->getType(), 1);
        In = Builder.CreateXor(In, One, In->getName() + ".not");
      }

      return replaceInstUsesWith(Zext, In);
    }

    // zext (X == 0) to i32 --> X^1      iff X has only the low bit set.
    // zext (X == 0) to i32 --> (X>>1)^1 iff X has only the 2nd bit set.
    // zext (X == 1) to i32 --> X        iff X has only the low bit set.
    // zext (X == 2) to i32 --> X>>1     iff X has only the 2nd bit set.
    // zext (X != 0) to i32 --> X        iff X has only the low bit set.
    // zext (X != 0) to i32 --> X>>1     iff X has only the 2nd bit set.
    // zext (X != 1) to i32 --> X^1      iff X has only the low bit set.
    // zext (X != 2) to i32 --> (X>>1)^1 iff X has only the 2nd bit set.
    if ((Op1CV->isZero() || Op1CV->isPowerOf2()) &&
        // This only works for EQ and NE
        Cmp->isEquality()) {
      // If Op1C some other power of two, convert:
      KnownBits Known = computeKnownBits(Cmp->getOperand(0), 0, &Zext);

      APInt KnownZeroMask(~Known.Zero);
      if (KnownZeroMask.isPowerOf2()) { // Exactly 1 possible 1?
        bool isNE = Cmp->getPredicate() == ICmpInst::ICMP_NE;
        if (!Op1CV->isZero() && (*Op1CV != KnownZeroMask)) {
          // (X&4) == 2 --> false
          // (X&4) != 2 --> true
          Constant *Res = ConstantInt::get(Zext.getType(), isNE);
          return replaceInstUsesWith(Zext, Res);
        }

        uint32_t ShAmt = KnownZeroMask.logBase2();
        Value *In = Cmp->getOperand(0);
        if (ShAmt) {
          // Perform a logical shr by shiftamt.
          // Insert the shift to put the result in the low bit.
          In = Builder.CreateLShr(In, ConstantInt::get(In->getType(), ShAmt),
                                  In->getName() + ".lobit");
        }

        if (!Op1CV->isZero() == isNE) { // Toggle the low bit.
          Constant *One = ConstantInt::get(In->getType(), 1);
          In = Builder.CreateXor(In, One);
        }

        if (Zext.getType() == In->getType())
          return replaceInstUsesWith(Zext, In);

        Value *IntCast = Builder.CreateIntCast(In, Zext.getType(), false);
        return replaceInstUsesWith(Zext, IntCast);
      }
    }
  }

  if (Cmp->isEquality() && Zext.getType() == Cmp->getOperand(0)->getType()) {
    // Test if a bit is clear/set using a shifted-one mask:
    // zext (icmp eq (and X, (1 << ShAmt)), 0) --> and (lshr (not X), ShAmt), 1
    // zext (icmp ne (and X, (1 << ShAmt)), 0) --> and (lshr X, ShAmt), 1
    Value *X, *ShAmt;
    if (Cmp->hasOneUse() && match(Cmp->getOperand(1), m_ZeroInt()) &&
        match(Cmp->getOperand(0),
              m_OneUse(m_c_And(m_Shl(m_One(), m_Value(ShAmt)), m_Value(X))))) {
      if (Cmp->getPredicate() == ICmpInst::ICMP_EQ)
        X = Builder.CreateNot(X);
      Value *Lshr = Builder.CreateLShr(X, ShAmt);
      Value *And1 = Builder.CreateAnd(Lshr, ConstantInt::get(X->getType(), 1));
      return replaceInstUsesWith(Zext, And1);
    }

    // icmp ne A, B is equal to xor A, B when A and B only really have one bit.
    // It is also profitable to transform icmp eq into not(xor(A, B)) because
    // that may lead to additional simplifications.
    if (IntegerType *ITy = dyn_cast<IntegerType>(Zext.getType())) {
      Value *LHS = Cmp->getOperand(0);
      Value *RHS = Cmp->getOperand(1);

      KnownBits KnownLHS = computeKnownBits(LHS, 0, &Zext);
      KnownBits KnownRHS = computeKnownBits(RHS, 0, &Zext);

      if (KnownLHS.Zero == KnownRHS.Zero && KnownLHS.One == KnownRHS.One) {
        APInt KnownBits = KnownLHS.Zero | KnownLHS.One;
        APInt UnknownBit = ~KnownBits;
        if (UnknownBit.countPopulation() == 1) {
          Value *Result = Builder.CreateXor(LHS, RHS);

          // Mask off any bits that are set and won't be shifted away.
          if (KnownLHS.One.uge(UnknownBit))
            Result = Builder.CreateAnd(Result,
                                        ConstantInt::get(ITy, UnknownBit));

          // Shift the bit we're testing down to the lsb.
          Result = Builder.CreateLShr(
               Result, ConstantInt::get(ITy, UnknownBit.countTrailingZeros()));

          if (Cmp->getPredicate() == ICmpInst::ICMP_EQ)
            Result = Builder.CreateXor(Result, ConstantInt::get(ITy, 1));
          Result->takeName(Cmp);
          return replaceInstUsesWith(Zext, Result);
        }
      }
    }
  }

  return nullptr;
}

/// Determine if the specified value can be computed in the specified wider type
/// and produce the same low bits. If not, return false.
///
/// If this function returns true, it can also return a non-zero number of bits
/// (in BitsToClear) which indicates that the value it computes is correct for
/// the zero extend, but that the additional BitsToClear bits need to be zero'd
/// out.  For example, to promote something like:
///
///   %B = trunc i64 %A to i32
///   %C = lshr i32 %B, 8
///   %E = zext i32 %C to i64
///
/// CanEvaluateZExtd for the 'lshr' will return true, and BitsToClear will be
/// set to 8 to indicate that the promoted value needs to have bits 24-31
/// cleared in addition to bits 32-63.  Since an 'and' will be generated to
/// clear the top bits anyway, doing this has no extra cost.
///
/// This function works on both vectors and scalars.
static bool canEvaluateZExtd(Value *V, Type *Ty, unsigned &BitsToClear,
                             InstCombinerImpl &IC, Instruction *CxtI) {
  BitsToClear = 0;
  if (canAlwaysEvaluateInType(V, Ty))
    return true;
  if (canNotEvaluateInType(V, Ty))
    return false;

  auto *I = cast<Instruction>(V);
  unsigned Tmp;
  switch (I->getOpcode()) {
  case Instruction::ZExt:  // zext(zext(x)) -> zext(x).
  case Instruction::SExt:  // zext(sext(x)) -> sext(x).
  case Instruction::Trunc: // zext(trunc(x)) -> trunc(x) or zext(x)
    return true;
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
    if (!canEvaluateZExtd(I->getOperand(0), Ty, BitsToClear, IC, CxtI) ||
        !canEvaluateZExtd(I->getOperand(1), Ty, Tmp, IC, CxtI))
      return false;
    // These can all be promoted if neither operand has 'bits to clear'.
    if (BitsToClear == 0 && Tmp == 0)
      return true;

    // If the operation is an AND/OR/XOR and the bits to clear are zero in the
    // other side, BitsToClear is ok.
    if (Tmp == 0 && I->isBitwiseLogicOp()) {
      // We use MaskedValueIsZero here for generality, but the case we care
      // about the most is constant RHS.
      unsigned VSize = V->getType()->getScalarSizeInBits();
      if (IC.MaskedValueIsZero(I->getOperand(1),
                               APInt::getHighBitsSet(VSize, BitsToClear),
                               0, CxtI)) {
        // If this is an And instruction and all of the BitsToClear are
        // known to be zero we can reset BitsToClear.
        if (I->getOpcode() == Instruction::And)
          BitsToClear = 0;
        return true;
      }
    }

    // Otherwise, we don't know how to analyze this BitsToClear case yet.
    return false;

  case Instruction::Shl: {
    // We can promote shl(x, cst) if we can promote x.  Since shl overwrites the
    // upper bits we can reduce BitsToClear by the shift amount.
    const APInt *Amt;
    if (match(I->getOperand(1), m_APInt(Amt))) {
      if (!canEvaluateZExtd(I->getOperand(0), Ty, BitsToClear, IC, CxtI))
        return false;
      uint64_t ShiftAmt = Amt->getZExtValue();
      BitsToClear = ShiftAmt < BitsToClear ? BitsToClear - ShiftAmt : 0;
      return true;
    }
    return false;
  }
  case Instruction::LShr: {
    // We can promote lshr(x, cst) if we can promote x.  This requires the
    // ultimate 'and' to clear out the high zero bits we're clearing out though.
    const APInt *Amt;
    if (match(I->getOperand(1), m_APInt(Amt))) {
      if (!canEvaluateZExtd(I->getOperand(0), Ty, BitsToClear, IC, CxtI))
        return false;
      BitsToClear += Amt->getZExtValue();
      if (BitsToClear > V->getType()->getScalarSizeInBits())
        BitsToClear = V->getType()->getScalarSizeInBits();
      return true;
    }
    // Cannot promote variable LSHR.
    return false;
  }
  case Instruction::Select:
    if (!canEvaluateZExtd(I->getOperand(1), Ty, Tmp, IC, CxtI) ||
        !canEvaluateZExtd(I->getOperand(2), Ty, BitsToClear, IC, CxtI) ||
        // TODO: If important, we could handle the case when the BitsToClear are
        // known zero in the disagreeing side.
        Tmp != BitsToClear)
      return false;
    return true;

  case Instruction::PHI: {
    // We can change a phi if we can change all operands.  Note that we never
    // get into trouble with cyclic PHIs here because we only consider
    // instructions with a single use.
    PHINode *PN = cast<PHINode>(I);
    if (!canEvaluateZExtd(PN->getIncomingValue(0), Ty, BitsToClear, IC, CxtI))
      return false;
    for (unsigned i = 1, e = PN->getNumIncomingValues(); i != e; ++i)
      if (!canEvaluateZExtd(PN->getIncomingValue(i), Ty, Tmp, IC, CxtI) ||
          // TODO: If important, we could handle the case when the BitsToClear
          // are known zero in the disagreeing input.
          Tmp != BitsToClear)
        return false;
    return true;
  }
  default:
    // TODO: Can handle more cases here.
    return false;
  }
}

Instruction *InstCombinerImpl::visitZExt(ZExtInst &CI) {
  // If this zero extend is only used by a truncate, let the truncate be
  // eliminated before we try to optimize this zext.
  if (CI.hasOneUse() && isa<TruncInst>(CI.user_back()))
    return nullptr;

  // If one of the common conversion will work, do it.
  if (Instruction *Result = commonCastTransforms(CI))
    return Result;

  Value *Src = CI.getOperand(0);
  Type *SrcTy = Src->getType(), *DestTy = CI.getType();

  // Try to extend the entire expression tree to the wide destination type.
  unsigned BitsToClear;
  if (shouldChangeType(SrcTy, DestTy) &&
      canEvaluateZExtd(Src, DestTy, BitsToClear, *this, &CI)) {
    assert(BitsToClear <= SrcTy->getScalarSizeInBits() &&
           "Can't clear more bits than in SrcTy");

    // Okay, we can transform this!  Insert the new expression now.
    LLVM_DEBUG(
        dbgs() << "ICE: EvaluateInDifferentType converting expression type"
                  " to avoid zero extend: "
               << CI << '\n');
    Value *Res = EvaluateInDifferentType(Src, DestTy, false);
    assert(Res->getType() == DestTy);

    // Preserve debug values referring to Src if the zext is its last use.
    if (auto *SrcOp = dyn_cast<Instruction>(Src))
      if (SrcOp->hasOneUse())
        replaceAllDbgUsesWith(*SrcOp, *Res, CI, DT);

    uint32_t SrcBitsKept = SrcTy->getScalarSizeInBits()-BitsToClear;
    uint32_t DestBitSize = DestTy->getScalarSizeInBits();

    // If the high bits are already filled with zeros, just replace this
    // cast with the result.
    if (MaskedValueIsZero(Res,
                          APInt::getHighBitsSet(DestBitSize,
                                                DestBitSize-SrcBitsKept),
                             0, &CI))
      return replaceInstUsesWith(CI, Res);

    // We need to emit an AND to clear the high bits.
    Constant *C = ConstantInt::get(Res->getType(),
                               APInt::getLowBitsSet(DestBitSize, SrcBitsKept));
    return BinaryOperator::CreateAnd(Res, C);
  }

  // If this is a TRUNC followed by a ZEXT then we are dealing with integral
  // types and if the sizes are just right we can convert this into a logical
  // 'and' which will be much cheaper than the pair of casts.
  if (TruncInst *CSrc = dyn_cast<TruncInst>(Src)) {   // A->B->C cast
    // TODO: Subsume this into EvaluateInDifferentType.

    // Get the sizes of the types involved.  We know that the intermediate type
    // will be smaller than A or C, but don't know the relation between A and C.
    Value *A = CSrc->getOperand(0);
    unsigned SrcSize = A->getType()->getScalarSizeInBits();
    unsigned MidSize = CSrc->getType()->getScalarSizeInBits();
    unsigned DstSize = CI.getType()->getScalarSizeInBits();
    // If we're actually extending zero bits, then if
    // SrcSize <  DstSize: zext(a & mask)
    // SrcSize == DstSize: a & mask
    // SrcSize  > DstSize: trunc(a) & mask
    if (SrcSize < DstSize) {
      APInt AndValue(APInt::getLowBitsSet(SrcSize, MidSize));
      Constant *AndConst = ConstantInt::get(A->getType(), AndValue);
      Value *And = Builder.CreateAnd(A, AndConst, CSrc->getName() + ".mask");
      return new ZExtInst(And, CI.getType());
    }

    if (SrcSize == DstSize) {
      APInt AndValue(APInt::getLowBitsSet(SrcSize, MidSize));
      return BinaryOperator::CreateAnd(A, ConstantInt::get(A->getType(),
                                                           AndValue));
    }
    if (SrcSize > DstSize) {
      Value *Trunc = Builder.CreateTrunc(A, CI.getType());
      APInt AndValue(APInt::getLowBitsSet(DstSize, MidSize));
      return BinaryOperator::CreateAnd(Trunc,
                                       ConstantInt::get(Trunc->getType(),
                                                        AndValue));
    }
  }

  if (ICmpInst *Cmp = dyn_cast<ICmpInst>(Src))
    return transformZExtICmp(Cmp, CI);

  // zext(trunc(X) & C) -> (X & zext(C)).
  Constant *C;
  Value *X;
  if (match(Src, m_OneUse(m_And(m_Trunc(m_Value(X)), m_Constant(C)))) &&
      X->getType() == CI.getType())
    return BinaryOperator::CreateAnd(X, ConstantExpr::getZExt(C, CI.getType()));

  // zext((trunc(X) & C) ^ C) -> ((X & zext(C)) ^ zext(C)).
  Value *And;
  if (match(Src, m_OneUse(m_Xor(m_Value(And), m_Constant(C)))) &&
      match(And, m_OneUse(m_And(m_Trunc(m_Value(X)), m_Specific(C)))) &&
      X->getType() == CI.getType()) {
    Constant *ZC = ConstantExpr::getZExt(C, CI.getType());
    return BinaryOperator::CreateXor(Builder.CreateAnd(X, ZC), ZC);
  }

  if (match(Src, m_VScale(DL))) {
    if (CI.getFunction() &&
        CI.getFunction()->hasFnAttribute(Attribute::VScaleRange)) {
      Attribute Attr = CI.getFunction()->getFnAttribute(Attribute::VScaleRange);
      if (Optional<unsigned> MaxVScale = Attr.getVScaleRangeMax()) {
        unsigned TypeWidth = Src->getType()->getScalarSizeInBits();
        if (Log2_32(MaxVScale.getValue()) < TypeWidth) {
          Value *VScale = Builder.CreateVScale(ConstantInt::get(DestTy, 1));
          return replaceInstUsesWith(CI, VScale);
        }
      }
    }
  }

  return nullptr;
}

/// Transform (sext icmp) to bitwise / integer operations to eliminate the icmp.
Instruction *InstCombinerImpl::transformSExtICmp(ICmpInst *ICI,
                                                 Instruction &CI) {
  Value *Op0 = ICI->getOperand(0), *Op1 = ICI->getOperand(1);
  ICmpInst::Predicate Pred = ICI->getPredicate();

  // Don't bother if Op1 isn't of vector or integer type.
  if (!Op1->getType()->isIntOrIntVectorTy())
    return nullptr;

  if ((Pred == ICmpInst::ICMP_SLT && match(Op1, m_ZeroInt())) ||
      (Pred == ICmpInst::ICMP_SGT && match(Op1, m_AllOnes()))) {
    // (x <s  0) ? -1 : 0 -> ashr x, 31        -> all ones if negative
    // (x >s -1) ? -1 : 0 -> not (ashr x, 31)  -> all ones if positive
    Value *Sh = ConstantInt::get(Op0->getType(),
                                 Op0->getType()->getScalarSizeInBits() - 1);
    Value *In = Builder.CreateAShr(Op0, Sh, Op0->getName() + ".lobit");
    if (In->getType() != CI.getType())
      In = Builder.CreateIntCast(In, CI.getType(), true /*SExt*/);

    if (Pred == ICmpInst::ICMP_SGT)
      In = Builder.CreateNot(In, In->getName() + ".not");
    return replaceInstUsesWith(CI, In);
  }

  if (ConstantInt *Op1C = dyn_cast<ConstantInt>(Op1)) {
    // If we know that only one bit of the LHS of the icmp can be set and we
    // have an equality comparison with zero or a power of 2, we can transform
    // the icmp and sext into bitwise/integer operations.
    if (ICI->hasOneUse() &&
        ICI->isEquality() && (Op1C->isZero() || Op1C->getValue().isPowerOf2())){
      KnownBits Known = computeKnownBits(Op0, 0, &CI);

      APInt KnownZeroMask(~Known.Zero);
      if (KnownZeroMask.isPowerOf2()) {
        Value *In = ICI->getOperand(0);

        // If the icmp tests for a known zero bit we can constant fold it.
        if (!Op1C->isZero() && Op1C->getValue() != KnownZeroMask) {
          Value *V = Pred == ICmpInst::ICMP_NE ?
                       ConstantInt::getAllOnesValue(CI.getType()) :
                       ConstantInt::getNullValue(CI.getType());
          return replaceInstUsesWith(CI, V);
        }

        if (!Op1C->isZero() == (Pred == ICmpInst::ICMP_NE)) {
          // sext ((x & 2^n) == 0)   -> (x >> n) - 1
          // sext ((x & 2^n) != 2^n) -> (x >> n) - 1
          unsigned ShiftAmt = KnownZeroMask.countTrailingZeros();
          // Perform a right shift to place the desired bit in the LSB.
          if (ShiftAmt)
            In = Builder.CreateLShr(In,
                                    ConstantInt::get(In->getType(), ShiftAmt));

          // At this point "In" is either 1 or 0. Subtract 1 to turn
          // {1, 0} -> {0, -1}.
          In = Builder.CreateAdd(In,
                                 ConstantInt::getAllOnesValue(In->getType()),
                                 "sext");
        } else {
          // sext ((x & 2^n) != 0)   -> (x << bitwidth-n) a>> bitwidth-1
          // sext ((x & 2^n) == 2^n) -> (x << bitwidth-n) a>> bitwidth-1
          unsigned ShiftAmt = KnownZeroMask.countLeadingZeros();
          // Perform a left shift to place the desired bit in the MSB.
          if (ShiftAmt)
            In = Builder.CreateShl(In,
                                   ConstantInt::get(In->getType(), ShiftAmt));

          // Distribute the bit over the whole bit width.
          In = Builder.CreateAShr(In, ConstantInt::get(In->getType(),
                                  KnownZeroMask.getBitWidth() - 1), "sext");
        }

        if (CI.getType() == In->getType())
          return replaceInstUsesWith(CI, In);
        return CastInst::CreateIntegerCast(In, CI.getType(), true/*SExt*/);
      }
    }
  }

  return nullptr;
}

/// Return true if we can take the specified value and return it as type Ty
/// without inserting any new casts and without changing the value of the common
/// low bits.  This is used by code that tries to promote integer operations to
/// a wider types will allow us to eliminate the extension.
///
/// This function works on both vectors and scalars.
///
static bool canEvaluateSExtd(Value *V, Type *Ty) {
  assert(V->getType()->getScalarSizeInBits() < Ty->getScalarSizeInBits() &&
         "Can't sign extend type to a smaller type");
  if (canAlwaysEvaluateInType(V, Ty))
    return true;
  if (canNotEvaluateInType(V, Ty))
    return false;

  auto *I = cast<Instruction>(V);
  switch (I->getOpcode()) {
  case Instruction::SExt:  // sext(sext(x)) -> sext(x)
  case Instruction::ZExt:  // sext(zext(x)) -> zext(x)
  case Instruction::Trunc: // sext(trunc(x)) -> trunc(x) or sext(x)
    return true;
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
    // These operators can all arbitrarily be extended if their inputs can.
    return canEvaluateSExtd(I->getOperand(0), Ty) &&
           canEvaluateSExtd(I->getOperand(1), Ty);

  //case Instruction::Shl:   TODO
  //case Instruction::LShr:  TODO

  case Instruction::Select:
    return canEvaluateSExtd(I->getOperand(1), Ty) &&
           canEvaluateSExtd(I->getOperand(2), Ty);

  case Instruction::PHI: {
    // We can change a phi if we can change all operands.  Note that we never
    // get into trouble with cyclic PHIs here because we only consider
    // instructions with a single use.
    PHINode *PN = cast<PHINode>(I);
    for (Value *IncValue : PN->incoming_values())
      if (!canEvaluateSExtd(IncValue, Ty)) return false;
    return true;
  }
  default:
    // TODO: Can handle more cases here.
    break;
  }

  return false;
}

Instruction *InstCombinerImpl::visitSExt(SExtInst &CI) {
  // If this sign extend is only used by a truncate, let the truncate be
  // eliminated before we try to optimize this sext.
  if (CI.hasOneUse() && isa<TruncInst>(CI.user_back()))
    return nullptr;

  if (Instruction *I = commonCastTransforms(CI))
    return I;

  Value *Src = CI.getOperand(0);
  Type *SrcTy = Src->getType(), *DestTy = CI.getType();
  unsigned SrcBitSize = SrcTy->getScalarSizeInBits();
  unsigned DestBitSize = DestTy->getScalarSizeInBits();

  // If we know that the value being extended is positive, we can use a zext
  // instead.
  KnownBits Known = computeKnownBits(Src, 0, &CI);
  if (Known.isNonNegative())
    return CastInst::Create(Instruction::ZExt, Src, DestTy);

  // Try to extend the entire expression tree to the wide destination type.
  if (shouldChangeType(SrcTy, DestTy) && canEvaluateSExtd(Src, DestTy)) {
    // Okay, we can transform this!  Insert the new expression now.
    LLVM_DEBUG(
        dbgs() << "ICE: EvaluateInDifferentType converting expression type"
                  " to avoid sign extend: "
               << CI << '\n');
    Value *Res = EvaluateInDifferentType(Src, DestTy, true);
    assert(Res->getType() == DestTy);

    // If the high bits are already filled with sign bit, just replace this
    // cast with the result.
    if (ComputeNumSignBits(Res, 0, &CI) > DestBitSize - SrcBitSize)
      return replaceInstUsesWith(CI, Res);

    // We need to emit a shl + ashr to do the sign extend.
    Value *ShAmt = ConstantInt::get(DestTy, DestBitSize-SrcBitSize);
    return BinaryOperator::CreateAShr(Builder.CreateShl(Res, ShAmt, "sext"),
                                      ShAmt);
  }

  Value *X;
  if (match(Src, m_Trunc(m_Value(X)))) {
    // If the input has more sign bits than bits truncated, then convert
    // directly to final type.
    unsigned XBitSize = X->getType()->getScalarSizeInBits();
    if (ComputeNumSignBits(X, 0, &CI) > XBitSize - SrcBitSize)
      return CastInst::CreateIntegerCast(X, DestTy, /* isSigned */ true);

    // If input is a trunc from the destination type, then convert into shifts.
    if (Src->hasOneUse() && X->getType() == DestTy) {
      // sext (trunc X) --> ashr (shl X, C), C
      Constant *ShAmt = ConstantInt::get(DestTy, DestBitSize - SrcBitSize);
      return BinaryOperator::CreateAShr(Builder.CreateShl(X, ShAmt), ShAmt);
    }

    // If we are replacing shifted-in high zero bits with sign bits, convert
    // the logic shift to arithmetic shift and eliminate the cast to
    // intermediate type:
    // sext (trunc (lshr Y, C)) --> sext/trunc (ashr Y, C)
    Value *Y;
    if (Src->hasOneUse() &&
        match(X, m_LShr(m_Value(Y),
                        m_SpecificIntAllowUndef(XBitSize - SrcBitSize)))) {
      Value *Ashr = Builder.CreateAShr(Y, XBitSize - SrcBitSize);
      return CastInst::CreateIntegerCast(Ashr, DestTy, /* isSigned */ true);
    }
  }

  if (ICmpInst *ICI = dyn_cast<ICmpInst>(Src))
    return transformSExtICmp(ICI, CI);

  // If the input is a shl/ashr pair of a same constant, then this is a sign
  // extension from a smaller value.  If we could trust arbitrary bitwidth
  // integers, we could turn this into a truncate to the smaller bit and then
  // use a sext for the whole extension.  Since we don't, look deeper and check
  // for a truncate.  If the source and dest are the same type, eliminate the
  // trunc and extend and just do shifts.  For example, turn:
  //   %a = trunc i32 %i to i8
  //   %b = shl i8 %a, C
  //   %c = ashr i8 %b, C
  //   %d = sext i8 %c to i32
  // into:
  //   %a = shl i32 %i, 32-(8-C)
  //   %d = ashr i32 %a, 32-(8-C)
  Value *A = nullptr;
  // TODO: Eventually this could be subsumed by EvaluateInDifferentType.
  Constant *BA = nullptr, *CA = nullptr;
  if (match(Src, m_AShr(m_Shl(m_Trunc(m_Value(A)), m_Constant(BA)),
                        m_Constant(CA))) &&
      BA->isElementWiseEqual(CA) && A->getType() == DestTy) {
    Constant *WideCurrShAmt = ConstantExpr::getSExt(CA, DestTy);
    Constant *NumLowbitsLeft = ConstantExpr::getSub(
        ConstantInt::get(DestTy, SrcTy->getScalarSizeInBits()), WideCurrShAmt);
    Constant *NewShAmt = ConstantExpr::getSub(
        ConstantInt::get(DestTy, DestTy->getScalarSizeInBits()),
        NumLowbitsLeft);
    NewShAmt =
        Constant::mergeUndefsWith(Constant::mergeUndefsWith(NewShAmt, BA), CA);
    A = Builder.CreateShl(A, NewShAmt, CI.getName());
    return BinaryOperator::CreateAShr(A, NewShAmt);
  }

  // Splatting a bit of constant-index across a value:
  // sext (ashr (trunc iN X to iM), M-1) to iN --> ashr (shl X, N-M), N-1
  // TODO: If the dest type is different, use a cast (adjust use check).
  if (match(Src, m_OneUse(m_AShr(m_Trunc(m_Value(X)),
                                 m_SpecificInt(SrcBitSize - 1)))) &&
      X->getType() == DestTy) {
    Constant *ShlAmtC = ConstantInt::get(DestTy, DestBitSize - SrcBitSize);
    Constant *AshrAmtC = ConstantInt::get(DestTy, DestBitSize - 1);
    Value *Shl = Builder.CreateShl(X, ShlAmtC);
    return BinaryOperator::CreateAShr(Shl, AshrAmtC);
  }

  if (match(Src, m_VScale(DL))) {
    if (CI.getFunction() &&
        CI.getFunction()->hasFnAttribute(Attribute::VScaleRange)) {
      Attribute Attr = CI.getFunction()->getFnAttribute(Attribute::VScaleRange);
      if (Optional<unsigned> MaxVScale = Attr.getVScaleRangeMax()) {
        if (Log2_32(MaxVScale.getValue()) < (SrcBitSize - 1)) {
          Value *VScale = Builder.CreateVScale(ConstantInt::get(DestTy, 1));
          return replaceInstUsesWith(CI, VScale);
        }
      }
    }
  }

  return nullptr;
}

/// Return a Constant* for the specified floating-point constant if it fits
/// in the specified FP type without changing its value.
static bool fitsInFPType(ConstantFP *CFP, const fltSemantics &Sem) {
  bool losesInfo;
  APFloat F = CFP->getValueAPF();
  (void)F.convert(Sem, APFloat::rmNearestTiesToEven, &losesInfo);
  return !losesInfo;
}

static Type *shrinkFPConstant(ConstantFP *CFP) {
  if (CFP->getType() == Type::getPPC_FP128Ty(CFP->getContext()))
    return nullptr;  // No constant folding of this.
  // See if the value can be truncated to half and then reextended.
  if (fitsInFPType(CFP, APFloat::IEEEhalf()))
    return Type::getHalfTy(CFP->getContext());
  // See if the value can be truncated to float and then reextended.
  if (fitsInFPType(CFP, APFloat::IEEEsingle()))
    return Type::getFloatTy(CFP->getContext());
  if (CFP->getType()->isDoubleTy())
    return nullptr;  // Won't shrink.
  if (fitsInFPType(CFP, APFloat::IEEEdouble()))
    return Type::getDoubleTy(CFP->getContext());
  // Don't try to shrink to various long double types.
  return nullptr;
}

// Determine if this is a vector of ConstantFPs and if so, return the minimal
// type we can safely truncate all elements to.
// TODO: Make these support undef elements.
static Type *shrinkFPConstantVector(Value *V) {
  auto *CV = dyn_cast<Constant>(V);
  auto *CVVTy = dyn_cast<FixedVectorType>(V->getType());
  if (!CV || !CVVTy)
    return nullptr;

  Type *MinType = nullptr;

  unsigned NumElts = CVVTy->getNumElements();

  // For fixed-width vectors we find the minimal type by looking
  // through the constant values of the vector.
  for (unsigned i = 0; i != NumElts; ++i) {
    auto *CFP = dyn_cast_or_null<ConstantFP>(CV->getAggregateElement(i));
    if (!CFP)
      return nullptr;

    Type *T = shrinkFPConstant(CFP);
    if (!T)
      return nullptr;

    // If we haven't found a type yet or this type has a larger mantissa than
    // our previous type, this is our new minimal type.
    if (!MinType || T->getFPMantissaWidth() > MinType->getFPMantissaWidth())
      MinType = T;
  }

  // Make a vector type from the minimal type.
  return FixedVectorType::get(MinType, NumElts);
}

/// Find the minimum FP type we can safely truncate to.
static Type *getMinimumFPType(Value *V) {
  if (auto *FPExt = dyn_cast<FPExtInst>(V))
    return FPExt->getOperand(0)->getType();

  // If this value is a constant, return the constant in the smallest FP type
  // that can accurately represent it.  This allows us to turn
  // (float)((double)X+2.0) into x+2.0f.
  if (auto *CFP = dyn_cast<ConstantFP>(V))
    if (Type *T = shrinkFPConstant(CFP))
      return T;

  // We can only correctly find a minimum type for a scalable vector when it is
  // a splat. For splats of constant values the fpext is wrapped up as a
  // ConstantExpr.
  if (auto *FPCExt = dyn_cast<ConstantExpr>(V))
    if (FPCExt->getOpcode() == Instruction::FPExt)
      return FPCExt->getOperand(0)->getType();

  // Try to shrink a vector of FP constants. This returns nullptr on scalable
  // vectors
  if (Type *T = shrinkFPConstantVector(V))
    return T;

  return V->getType();
}

/// Return true if the cast from integer to FP can be proven to be exact for all
/// possible inputs (the conversion does not lose any precision).
static bool isKnownExactCastIntToFP(CastInst &I) {
  CastInst::CastOps Opcode = I.getOpcode();
  assert((Opcode == CastInst::SIToFP || Opcode == CastInst::UIToFP) &&
         "Unexpected cast");
  Value *Src = I.getOperand(0);
  Type *SrcTy = Src->getType();
  Type *FPTy = I.getType();
  bool IsSigned = Opcode == Instruction::SIToFP;
  int SrcSize = (int)SrcTy->getScalarSizeInBits() - IsSigned;

  // Easy case - if the source integer type has less bits than the FP mantissa,
  // then the cast must be exact.
  int DestNumSigBits = FPTy->getFPMantissaWidth();
  if (SrcSize <= DestNumSigBits)
    return true;

  // Cast from FP to integer and back to FP is independent of the intermediate
  // integer width because of poison on overflow.
  Value *F;
  if (match(Src, m_FPToSI(m_Value(F))) || match(Src, m_FPToUI(m_Value(F)))) {
    // If this is uitofp (fptosi F), the source needs an extra bit to avoid
    // potential rounding of negative FP input values.
    int SrcNumSigBits = F->getType()->getFPMantissaWidth();
    if (!IsSigned && match(Src, m_FPToSI(m_Value())))
      SrcNumSigBits++;

    // [su]itofp (fpto[su]i F) --> exact if the source type has less or equal
    // significant bits than the destination (and make sure neither type is
    // weird -- ppc_fp128).
    if (SrcNumSigBits > 0 && DestNumSigBits > 0 &&
        SrcNumSigBits <= DestNumSigBits)
      return true;
  }

  // TODO:
  // Try harder to find if the source integer type has less significant bits.
  // For example, compute number of sign bits or compute low bit mask.
  return false;
}

Instruction *InstCombinerImpl::visitFPTrunc(FPTruncInst &FPT) {
  if (Instruction *I = commonCastTransforms(FPT))
    return I;

  // If we have fptrunc(OpI (fpextend x), (fpextend y)), we would like to
  // simplify this expression to avoid one or more of the trunc/extend
  // operations if we can do so without changing the numerical results.
  //
  // The exact manner in which the widths of the operands interact to limit
  // what we can and cannot do safely varies from operation to operation, and
  // is explained below in the various case statements.
  Type *Ty = FPT.getType();
  auto *BO = dyn_cast<BinaryOperator>(FPT.getOperand(0));
  if (BO && BO->hasOneUse()) {
    Type *LHSMinType = getMinimumFPType(BO->getOperand(0));
    Type *RHSMinType = getMinimumFPType(BO->getOperand(1));
    unsigned OpWidth = BO->getType()->getFPMantissaWidth();
    unsigned LHSWidth = LHSMinType->getFPMantissaWidth();
    unsigned RHSWidth = RHSMinType->getFPMantissaWidth();
    unsigned SrcWidth = std::max(LHSWidth, RHSWidth);
    unsigned DstWidth = Ty->getFPMantissaWidth();
    switch (BO->getOpcode()) {
      default: break;
      case Instruction::FAdd:
      case Instruction::FSub:
        // For addition and subtraction, the infinitely precise result can
        // essentially be arbitrarily wide; proving that double rounding
        // will not occur because the result of OpI is exact (as we will for
        // FMul, for example) is hopeless.  However, we *can* nonetheless
        // frequently know that double rounding cannot occur (or that it is
        // innocuous) by taking advantage of the specific structure of
        // infinitely-precise results that admit double rounding.
        //
        // Specifically, if OpWidth >= 2*DstWdith+1 and DstWidth is sufficient
        // to represent both sources, we can guarantee that the double
        // rounding is innocuous (See p50 of Figueroa's 2000 PhD thesis,
        // "A Rigorous Framework for Fully Supporting the IEEE Standard ..."
        // for proof of this fact).
        //
        // Note: Figueroa does not consider the case where DstFormat !=
        // SrcFormat.  It's possible (likely even!) that this analysis
        // could be tightened for those cases, but they are rare (the main
        // case of interest here is (float)((double)float + float)).
        if (OpWidth >= 2*DstWidth+1 && DstWidth >= SrcWidth) {
          Value *LHS = Builder.CreateFPTrunc(BO->getOperand(0), Ty);
          Value *RHS = Builder.CreateFPTrunc(BO->getOperand(1), Ty);
          Instruction *RI = BinaryOperator::Create(BO->getOpcode(), LHS, RHS);
          RI->copyFastMathFlags(BO);
          return RI;
        }
        break;
      case Instruction::FMul:
        // For multiplication, the infinitely precise result has at most
        // LHSWidth + RHSWidth significant bits; if OpWidth is sufficient
        // that such a value can be exactly represented, then no double
        // rounding can possibly occur; we can safely perform the operation
        // in the destination format if it can represent both sources.
        if (OpWidth >= LHSWidth + RHSWidth && DstWidth >= SrcWidth) {
          Value *LHS = Builder.CreateFPTrunc(BO->getOperand(0), Ty);
          Value *RHS = Builder.CreateFPTrunc(BO->getOperand(1), Ty);
          return BinaryOperator::CreateFMulFMF(LHS, RHS, BO);
        }
        break;
      case Instruction::FDiv:
        // For division, we use again use the bound from Figueroa's
        // dissertation.  I am entirely certain that this bound can be
        // tightened in the unbalanced operand case by an analysis based on
        // the diophantine rational approximation bound, but the well-known
        // condition used here is a good conservative first pass.
        // TODO: Tighten bound via rigorous analysis of the unbalanced case.
        if (OpWidth >= 2*DstWidth && DstWidth >= SrcWidth) {
          Value *LHS = Builder.CreateFPTrunc(BO->getOperand(0), Ty);
          Value *RHS = Builder.CreateFPTrunc(BO->getOperand(1), Ty);
          return BinaryOperator::CreateFDivFMF(LHS, RHS, BO);
        }
        break;
      case Instruction::FRem: {
        // Remainder is straightforward.  Remainder is always exact, so the
        // type of OpI doesn't enter into things at all.  We simply evaluate
        // in whichever source type is larger, then convert to the
        // destination type.
        if (SrcWidth == OpWidth)
          break;
        Value *LHS, *RHS;
        if (LHSWidth == SrcWidth) {
           LHS = Builder.CreateFPTrunc(BO->getOperand(0), LHSMinType);
           RHS = Builder.CreateFPTrunc(BO->getOperand(1), LHSMinType);
        } else {
           LHS = Builder.CreateFPTrunc(BO->getOperand(0), RHSMinType);
           RHS = Builder.CreateFPTrunc(BO->getOperand(1), RHSMinType);
        }

        Value *ExactResult = Builder.CreateFRemFMF(LHS, RHS, BO);
        return CastInst::CreateFPCast(ExactResult, Ty);
      }
    }
  }

  // (fptrunc (fneg x)) -> (fneg (fptrunc x))
  Value *X;
  Instruction *Op = dyn_cast<Instruction>(FPT.getOperand(0));
  if (Op && Op->hasOneUse()) {
    // FIXME: The FMF should propagate from the fptrunc, not the source op.
    IRBuilder<>::FastMathFlagGuard FMFG(Builder);
    if (isa<FPMathOperator>(Op))
      Builder.setFastMathFlags(Op->getFastMathFlags());

    if (match(Op, m_FNeg(m_Value(X)))) {
      Value *InnerTrunc = Builder.CreateFPTrunc(X, Ty);

      return UnaryOperator::CreateFNegFMF(InnerTrunc, Op);
    }

    // If we are truncating a select that has an extended operand, we can
    // narrow the other operand and do the select as a narrow op.
    Value *Cond, *X, *Y;
    if (match(Op, m_Select(m_Value(Cond), m_FPExt(m_Value(X)), m_Value(Y))) &&
        X->getType() == Ty) {
      // fptrunc (select Cond, (fpext X), Y --> select Cond, X, (fptrunc Y)
      Value *NarrowY = Builder.CreateFPTrunc(Y, Ty);
      Value *Sel = Builder.CreateSelect(Cond, X, NarrowY, "narrow.sel", Op);
      return replaceInstUsesWith(FPT, Sel);
    }
    if (match(Op, m_Select(m_Value(Cond), m_Value(Y), m_FPExt(m_Value(X)))) &&
        X->getType() == Ty) {
      // fptrunc (select Cond, Y, (fpext X) --> select Cond, (fptrunc Y), X
      Value *NarrowY = Builder.CreateFPTrunc(Y, Ty);
      Value *Sel = Builder.CreateSelect(Cond, NarrowY, X, "narrow.sel", Op);
      return replaceInstUsesWith(FPT, Sel);
    }
  }

  if (auto *II = dyn_cast<IntrinsicInst>(FPT.getOperand(0))) {
    switch (II->getIntrinsicID()) {
    default: break;
    case Intrinsic::ceil:
    case Intrinsic::fabs:
    case Intrinsic::floor:
    case Intrinsic::nearbyint:
    case Intrinsic::rint:
    case Intrinsic::round:
    case Intrinsic::roundeven:
    case Intrinsic::trunc: {
      Value *Src = II->getArgOperand(0);
      if (!Src->hasOneUse())
        break;

      // Except for fabs, this transformation requires the input of the unary FP
      // operation to be itself an fpext from the type to which we're
      // truncating.
      if (II->getIntrinsicID() != Intrinsic::fabs) {
        FPExtInst *FPExtSrc = dyn_cast<FPExtInst>(Src);
        if (!FPExtSrc || FPExtSrc->getSrcTy() != Ty)
          break;
      }

      // Do unary FP operation on smaller type.
      // (fptrunc (fabs x)) -> (fabs (fptrunc x))
      Value *InnerTrunc = Builder.CreateFPTrunc(Src, Ty);
      Function *Overload = Intrinsic::getDeclaration(FPT.getModule(),
                                                     II->getIntrinsicID(), Ty);
      SmallVector<OperandBundleDef, 1> OpBundles;
      II->getOperandBundlesAsDefs(OpBundles);
      CallInst *NewCI =
          CallInst::Create(Overload, {InnerTrunc}, OpBundles, II->getName());
      NewCI->copyFastMathFlags(II);
      return NewCI;
    }
    }
  }

  if (Instruction *I = shrinkInsertElt(FPT, Builder))
    return I;

  Value *Src = FPT.getOperand(0);
  if (isa<SIToFPInst>(Src) || isa<UIToFPInst>(Src)) {
    auto *FPCast = cast<CastInst>(Src);
    if (isKnownExactCastIntToFP(*FPCast))
      return CastInst::Create(FPCast->getOpcode(), FPCast->getOperand(0), Ty);
  }

  return nullptr;
}

Instruction *InstCombinerImpl::visitFPExt(CastInst &FPExt) {
  // If the source operand is a cast from integer to FP and known exact, then
  // cast the integer operand directly to the destination type.
  Type *Ty = FPExt.getType();
  Value *Src = FPExt.getOperand(0);
  if (isa<SIToFPInst>(Src) || isa<UIToFPInst>(Src)) {
    auto *FPCast = cast<CastInst>(Src);
    if (isKnownExactCastIntToFP(*FPCast))
      return CastInst::Create(FPCast->getOpcode(), FPCast->getOperand(0), Ty);
  }

  return commonCastTransforms(FPExt);
}

/// fpto{s/u}i({u/s}itofp(X)) --> X or zext(X) or sext(X) or trunc(X)
/// This is safe if the intermediate type has enough bits in its mantissa to
/// accurately represent all values of X.  For example, this won't work with
/// i64 -> float -> i64.
Instruction *InstCombinerImpl::foldItoFPtoI(CastInst &FI) {
  if (!isa<UIToFPInst>(FI.getOperand(0)) && !isa<SIToFPInst>(FI.getOperand(0)))
    return nullptr;

  auto *OpI = cast<CastInst>(FI.getOperand(0));
  Value *X = OpI->getOperand(0);
  Type *XType = X->getType();
  Type *DestType = FI.getType();
  bool IsOutputSigned = isa<FPToSIInst>(FI);

  // Since we can assume the conversion won't overflow, our decision as to
  // whether the input will fit in the float should depend on the minimum
  // of the input range and output range.

  // This means this is also safe for a signed input and unsigned output, since
  // a negative input would lead to undefined behavior.
  if (!isKnownExactCastIntToFP(*OpI)) {
    // The first cast may not round exactly based on the source integer width
    // and FP width, but the overflow UB rules can still allow this to fold.
    // If the destination type is narrow, that means the intermediate FP value
    // must be large enough to hold the source value exactly.
    // For example, (uint8_t)((float)(uint32_t 16777217) is undefined behavior.
    int OutputSize = (int)DestType->getScalarSizeInBits() - IsOutputSigned;
    if (OutputSize > OpI->getType()->getFPMantissaWidth())
      return nullptr;
  }

  if (DestType->getScalarSizeInBits() > XType->getScalarSizeInBits()) {
    bool IsInputSigned = isa<SIToFPInst>(OpI);
    if (IsInputSigned && IsOutputSigned)
      return new SExtInst(X, DestType);
    return new ZExtInst(X, DestType);
  }
  if (DestType->getScalarSizeInBits() < XType->getScalarSizeInBits())
    return new TruncInst(X, DestType);

  assert(XType == DestType && "Unexpected types for int to FP to int casts");
  return replaceInstUsesWith(FI, X);
}

Instruction *InstCombinerImpl::visitFPToUI(FPToUIInst &FI) {
  if (Instruction *I = foldItoFPtoI(FI))
    return I;

  return commonCastTransforms(FI);
}

Instruction *InstCombinerImpl::visitFPToSI(FPToSIInst &FI) {
  if (Instruction *I = foldItoFPtoI(FI))
    return I;

  return commonCastTransforms(FI);
}

Instruction *InstCombinerImpl::visitUIToFP(CastInst &CI) {
  return commonCastTransforms(CI);
}

Instruction *InstCombinerImpl::visitSIToFP(CastInst &CI) {
  return commonCastTransforms(CI);
}

Instruction *InstCombinerImpl::visitIntToPtr(IntToPtrInst &CI) {
  // If the source integer type is not the intptr_t type for this target, do a
  // trunc or zext to the intptr_t type, then inttoptr of it.  This allows the
  // cast to be exposed to other transforms.
  unsigned AS = CI.getAddressSpace();
  if (CI.getOperand(0)->getType()->getScalarSizeInBits() !=
      DL.getPointerSizeInBits(AS)) {
    Type *Ty = CI.getOperand(0)->getType()->getWithNewType(
        DL.getIntPtrType(CI.getContext(), AS));
    Value *P = Builder.CreateZExtOrTrunc(CI.getOperand(0), Ty);
    return new IntToPtrInst(P, CI.getType());
  }

  if (Instruction *I = commonCastTransforms(CI))
    return I;

  return nullptr;
}

/// Implement the transforms for cast of pointer (bitcast/ptrtoint)
Instruction *InstCombinerImpl::commonPointerCastTransforms(CastInst &CI) {
  Value *Src = CI.getOperand(0);

  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Src)) {
    // If casting the result of a getelementptr instruction with no offset, turn
    // this into a cast of the original pointer!
    if (GEP->hasAllZeroIndices() &&
        // If CI is an addrspacecast and GEP changes the poiner type, merging
        // GEP into CI would undo canonicalizing addrspacecast with different
        // pointer types, causing infinite loops.
        (!isa<AddrSpaceCastInst>(CI) ||
         GEP->getType() == GEP->getPointerOperandType())) {
      // Changing the cast operand is usually not a good idea but it is safe
      // here because the pointer operand is being replaced with another
      // pointer operand so the opcode doesn't need to change.
      return replaceOperand(CI, 0, GEP->getOperand(0));
    }
  }

  return commonCastTransforms(CI);
}

Instruction *InstCombinerImpl::visitPtrToInt(PtrToIntInst &CI) {
  // If the destination integer type is not the intptr_t type for this target,
  // do a ptrtoint to intptr_t then do a trunc or zext.  This allows the cast
  // to be exposed to other transforms.
  Value *SrcOp = CI.getPointerOperand();
  Type *SrcTy = SrcOp->getType();
  Type *Ty = CI.getType();
  unsigned AS = CI.getPointerAddressSpace();
  unsigned TySize = Ty->getScalarSizeInBits();
  unsigned PtrSize = DL.getPointerSizeInBits(AS);
  if (TySize != PtrSize) {
    Type *IntPtrTy =
        SrcTy->getWithNewType(DL.getIntPtrType(CI.getContext(), AS));
    Value *P = Builder.CreatePtrToInt(SrcOp, IntPtrTy);
    return CastInst::CreateIntegerCast(P, Ty, /*isSigned=*/false);
  }

  if (auto *GEP = dyn_cast<GetElementPtrInst>(SrcOp)) {
    // Fold ptrtoint(gep null, x) to multiply + constant if the GEP has one use.
    // While this can increase the number of instructions it doesn't actually
    // increase the overall complexity since the arithmetic is just part of
    // the GEP otherwise.
    if (GEP->hasOneUse() &&
        isa<ConstantPointerNull>(GEP->getPointerOperand())) {
      return replaceInstUsesWith(CI,
                                 Builder.CreateIntCast(EmitGEPOffset(GEP), Ty,
                                                       /*isSigned=*/false));
    }
  }

  Value *Vec, *Scalar, *Index;
  if (match(SrcOp, m_OneUse(m_InsertElt(m_IntToPtr(m_Value(Vec)),
                                        m_Value(Scalar), m_Value(Index)))) &&
      Vec->getType() == Ty) {
    assert(Vec->getType()->getScalarSizeInBits() == PtrSize && "Wrong type");
    // Convert the scalar to int followed by insert to eliminate one cast:
    // p2i (ins (i2p Vec), Scalar, Index --> ins Vec, (p2i Scalar), Index
    Value *NewCast = Builder.CreatePtrToInt(Scalar, Ty->getScalarType());
    return InsertElementInst::Create(Vec, NewCast, Index);
  }

  return commonPointerCastTransforms(CI);
}

/// This input value (which is known to have vector type) is being zero extended
/// or truncated to the specified vector type. Since the zext/trunc is done
/// using an integer type, we have a (bitcast(cast(bitcast))) pattern,
/// endianness will impact which end of the vector that is extended or
/// truncated.
///
/// A vector is always stored with index 0 at the lowest address, which
/// corresponds to the most significant bits for a big endian stored integer and
/// the least significant bits for little endian. A trunc/zext of an integer
/// impacts the big end of the integer. Thus, we need to add/remove elements at
/// the front of the vector for big endian targets, and the back of the vector
/// for little endian targets.
///
/// Try to replace it with a shuffle (and vector/vector bitcast) if possible.
///
/// The source and destination vector types may have different element types.
static Instruction *
optimizeVectorResizeWithIntegerBitCasts(Value *InVal, VectorType *DestTy,
                                        InstCombinerImpl &IC) {
  // We can only do this optimization if the output is a multiple of the input
  // element size, or the input is a multiple of the output element size.
  // Convert the input type to have the same element type as the output.
  VectorType *SrcTy = cast<VectorType>(InVal->getType());

  if (SrcTy->getElementType() != DestTy->getElementType()) {
    // The input types don't need to be identical, but for now they must be the
    // same size.  There is no specific reason we couldn't handle things like
    // <4 x i16> -> <4 x i32> by bitcasting to <2 x i32> but haven't gotten
    // there yet.
    if (SrcTy->getElementType()->getPrimitiveSizeInBits() !=
        DestTy->getElementType()->getPrimitiveSizeInBits())
      return nullptr;

    SrcTy =
        FixedVectorType::get(DestTy->getElementType(),
                             cast<FixedVectorType>(SrcTy)->getNumElements());
    InVal = IC.Builder.CreateBitCast(InVal, SrcTy);
  }

  bool IsBigEndian = IC.getDataLayout().isBigEndian();
  unsigned SrcElts = cast<FixedVectorType>(SrcTy)->getNumElements();
  unsigned DestElts = cast<FixedVectorType>(DestTy)->getNumElements();

  assert(SrcElts != DestElts && "Element counts should be different.");

  // Now that the element types match, get the shuffle mask and RHS of the
  // shuffle to use, which depends on whether we're increasing or decreasing the
  // size of the input.
  auto ShuffleMaskStorage = llvm::to_vector<16>(llvm::seq<int>(0, SrcElts));
  ArrayRef<int> ShuffleMask;
  Value *V2;

  if (SrcElts > DestElts) {
    // If we're shrinking the number of elements (rewriting an integer
    // truncate), just shuffle in the elements corresponding to the least
    // significant bits from the input and use poison as the second shuffle
    // input.
    V2 = PoisonValue::get(SrcTy);
    // Make sure the shuffle mask selects the "least significant bits" by
    // keeping elements from back of the src vector for big endian, and from the
    // front for little endian.
    ShuffleMask = ShuffleMaskStorage;
    if (IsBigEndian)
      ShuffleMask = ShuffleMask.take_back(DestElts);
    else
      ShuffleMask = ShuffleMask.take_front(DestElts);
  } else {
    // If we're increasing the number of elements (rewriting an integer zext),
    // shuffle in all of the elements from InVal. Fill the rest of the result
    // elements with zeros from a constant zero.
    V2 = Constant::getNullValue(SrcTy);
    // Use first elt from V2 when indicating zero in the shuffle mask.
    uint32_t NullElt = SrcElts;
    // Extend with null values in the "most significant bits" by adding elements
    // in front of the src vector for big endian, and at the back for little
    // endian.
    unsigned DeltaElts = DestElts - SrcElts;
    if (IsBigEndian)
      ShuffleMaskStorage.insert(ShuffleMaskStorage.begin(), DeltaElts, NullElt);
    else
      ShuffleMaskStorage.append(DeltaElts, NullElt);
    ShuffleMask = ShuffleMaskStorage;
  }

  return new ShuffleVectorInst(InVal, V2, ShuffleMask);
}

static bool isMultipleOfTypeSize(unsigned Value, Type *Ty) {
  return Value % Ty->getPrimitiveSizeInBits() == 0;
}

static unsigned getTypeSizeIndex(unsigned Value, Type *Ty) {
  return Value / Ty->getPrimitiveSizeInBits();
}

/// V is a value which is inserted into a vector of VecEltTy.
/// Look through the value to see if we can decompose it into
/// insertions into the vector.  See the example in the comment for
/// OptimizeIntegerToVectorInsertions for the pattern this handles.
/// The type of V is always a non-zero multiple of VecEltTy's size.
/// Shift is the number of bits between the lsb of V and the lsb of
/// the vector.
///
/// This returns false if the pattern can't be matched or true if it can,
/// filling in Elements with the elements found here.
static bool collectInsertionElements(Value *V, unsigned Shift,
                                     SmallVectorImpl<Value *> &Elements,
                                     Type *VecEltTy, bool isBigEndian) {
  assert(isMultipleOfTypeSize(Shift, VecEltTy) &&
         "Shift should be a multiple of the element type size");

  // Undef values never contribute useful bits to the result.
  if (isa<UndefValue>(V)) return true;

  // If we got down to a value of the right type, we win, try inserting into the
  // right element.
  if (V->getType() == VecEltTy) {
    // Inserting null doesn't actually insert any elements.
    if (Constant *C = dyn_cast<Constant>(V))
      if (C->isNullValue())
        return true;

    unsigned ElementIndex = getTypeSizeIndex(Shift, VecEltTy);
    if (isBigEndian)
      ElementIndex = Elements.size() - ElementIndex - 1;

    // Fail if multiple elements are inserted into this slot.
    if (Elements[ElementIndex])
      return false;

    Elements[ElementIndex] = V;
    return true;
  }

  if (Constant *C = dyn_cast<Constant>(V)) {
    // Figure out the # elements this provides, and bitcast it or slice it up
    // as required.
    unsigned NumElts = getTypeSizeIndex(C->getType()->getPrimitiveSizeInBits(),
                                        VecEltTy);
    // If the constant is the size of a vector element, we just need to bitcast
    // it to the right type so it gets properly inserted.
    if (NumElts == 1)
      return collectInsertionElements(ConstantExpr::getBitCast(C, VecEltTy),
                                      Shift, Elements, VecEltTy, isBigEndian);

    // Okay, this is a constant that covers multiple elements.  Slice it up into
    // pieces and insert each element-sized piece into the vector.
    if (!isa<IntegerType>(C->getType()))
      C = ConstantExpr::getBitCast(C, IntegerType::get(V->getContext(),
                                       C->getType()->getPrimitiveSizeInBits()));
    unsigned ElementSize = VecEltTy->getPrimitiveSizeInBits();
    Type *ElementIntTy = IntegerType::get(C->getContext(), ElementSize);

    for (unsigned i = 0; i != NumElts; ++i) {
      unsigned ShiftI = Shift+i*ElementSize;
      Constant *Piece = ConstantExpr::getLShr(C, ConstantInt::get(C->getType(),
                                                                  ShiftI));
      Piece = ConstantExpr::getTrunc(Piece, ElementIntTy);
      if (!collectInsertionElements(Piece, ShiftI, Elements, VecEltTy,
                                    isBigEndian))
        return false;
    }
    return true;
  }

  if (!V->hasOneUse()) return false;

  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) return false;
  switch (I->getOpcode()) {
  default: return false; // Unhandled case.
  case Instruction::BitCast:
    return collectInsertionElements(I->getOperand(0), Shift, Elements, VecEltTy,
                                    isBigEndian);
  case Instruction::ZExt:
    if (!isMultipleOfTypeSize(
                          I->getOperand(0)->getType()->getPrimitiveSizeInBits(),
                              VecEltTy))
      return false;
    return collectInsertionElements(I->getOperand(0), Shift, Elements, VecEltTy,
                                    isBigEndian);
  case Instruction::Or:
    return collectInsertionElements(I->getOperand(0), Shift, Elements, VecEltTy,
                                    isBigEndian) &&
           collectInsertionElements(I->getOperand(1), Shift, Elements, VecEltTy,
                                    isBigEndian);
  case Instruction::Shl: {
    // Must be shifting by a constant that is a multiple of the element size.
    ConstantInt *CI = dyn_cast<ConstantInt>(I->getOperand(1));
    if (!CI) return false;
    Shift += CI->getZExtValue();
    if (!isMultipleOfTypeSize(Shift, VecEltTy)) return false;
    return collectInsertionElements(I->getOperand(0), Shift, Elements, VecEltTy,
                                    isBigEndian);
  }

  }
}


/// If the input is an 'or' instruction, we may be doing shifts and ors to
/// assemble the elements of the vector manually.
/// Try to rip the code out and replace it with insertelements.  This is to
/// optimize code like this:
///
///    %tmp37 = bitcast float %inc to i32
///    %tmp38 = zext i32 %tmp37 to i64
///    %tmp31 = bitcast float %inc5 to i32
///    %tmp32 = zext i32 %tmp31 to i64
///    %tmp33 = shl i64 %tmp32, 32
///    %ins35 = or i64 %tmp33, %tmp38
///    %tmp43 = bitcast i64 %ins35 to <2 x float>
///
/// Into two insertelements that do "buildvector{%inc, %inc5}".
static Value *optimizeIntegerToVectorInsertions(BitCastInst &CI,
                                                InstCombinerImpl &IC) {
  auto *DestVecTy = cast<FixedVectorType>(CI.getType());
  Value *IntInput = CI.getOperand(0);

  SmallVector<Value*, 8> Elements(DestVecTy->getNumElements());
  if (!collectInsertionElements(IntInput, 0, Elements,
                                DestVecTy->getElementType(),
                                IC.getDataLayout().isBigEndian()))
    return nullptr;

  // If we succeeded, we know that all of the element are specified by Elements
  // or are zero if Elements has a null entry.  Recast this as a set of
  // insertions.
  Value *Result = Constant::getNullValue(CI.getType());
  for (unsigned i = 0, e = Elements.size(); i != e; ++i) {
    if (!Elements[i]) continue;  // Unset element.

    Result = IC.Builder.CreateInsertElement(Result, Elements[i],
                                            IC.Builder.getInt32(i));
  }

  return Result;
}

/// Canonicalize scalar bitcasts of extracted elements into a bitcast of the
/// vector followed by extract element. The backend tends to handle bitcasts of
/// vectors better than bitcasts of scalars because vector registers are
/// usually not type-specific like scalar integer or scalar floating-point.
static Instruction *canonicalizeBitCastExtElt(BitCastInst &BitCast,
                                              InstCombinerImpl &IC) {
  // TODO: Create and use a pattern matcher for ExtractElementInst.
  auto *ExtElt = dyn_cast<ExtractElementInst>(BitCast.getOperand(0));
  if (!ExtElt || !ExtElt->hasOneUse())
    return nullptr;

  // The bitcast must be to a vectorizable type, otherwise we can't make a new
  // type to extract from.
  Type *DestType = BitCast.getType();
  if (!VectorType::isValidElementType(DestType))
    return nullptr;

  auto *NewVecType = VectorType::get(DestType, ExtElt->getVectorOperandType());
  auto *NewBC = IC.Builder.CreateBitCast(ExtElt->getVectorOperand(),
                                         NewVecType, "bc");
  return ExtractElementInst::Create(NewBC, ExtElt->getIndexOperand());
}

/// Change the type of a bitwise logic operation if we can eliminate a bitcast.
static Instruction *foldBitCastBitwiseLogic(BitCastInst &BitCast,
                                            InstCombiner::BuilderTy &Builder) {
  Type *DestTy = BitCast.getType();
  BinaryOperator *BO;
  if (!DestTy->isIntOrIntVectorTy() ||
      !match(BitCast.getOperand(0), m_OneUse(m_BinOp(BO))) ||
      !BO->isBitwiseLogicOp())
    return nullptr;

  // FIXME: This transform is restricted to vector types to avoid backend
  // problems caused by creating potentially illegal operations. If a fix-up is
  // added to handle that situation, we can remove this check.
  if (!DestTy->isVectorTy() || !BO->getType()->isVectorTy())
    return nullptr;

  Value *X;
  if (match(BO->getOperand(0), m_OneUse(m_BitCast(m_Value(X)))) &&
      X->getType() == DestTy && !isa<Constant>(X)) {
    // bitcast(logic(bitcast(X), Y)) --> logic'(X, bitcast(Y))
    Value *CastedOp1 = Builder.CreateBitCast(BO->getOperand(1), DestTy);
    return BinaryOperator::Create(BO->getOpcode(), X, CastedOp1);
  }

  if (match(BO->getOperand(1), m_OneUse(m_BitCast(m_Value(X)))) &&
      X->getType() == DestTy && !isa<Constant>(X)) {
    // bitcast(logic(Y, bitcast(X))) --> logic'(bitcast(Y), X)
    Value *CastedOp0 = Builder.CreateBitCast(BO->getOperand(0), DestTy);
    return BinaryOperator::Create(BO->getOpcode(), CastedOp0, X);
  }

  // Canonicalize vector bitcasts to come before vector bitwise logic with a
  // constant. This eases recognition of special constants for later ops.
  // Example:
  // icmp u/s (a ^ signmask), (b ^ signmask) --> icmp s/u a, b
  Constant *C;
  if (match(BO->getOperand(1), m_Constant(C))) {
    // bitcast (logic X, C) --> logic (bitcast X, C')
    Value *CastedOp0 = Builder.CreateBitCast(BO->getOperand(0), DestTy);
    Value *CastedC = Builder.CreateBitCast(C, DestTy);
    return BinaryOperator::Create(BO->getOpcode(), CastedOp0, CastedC);
  }

  return nullptr;
}

/// Change the type of a select if we can eliminate a bitcast.
static Instruction *foldBitCastSelect(BitCastInst &BitCast,
                                      InstCombiner::BuilderTy &Builder) {
  Value *Cond, *TVal, *FVal;
  if (!match(BitCast.getOperand(0),
             m_OneUse(m_Select(m_Value(Cond), m_Value(TVal), m_Value(FVal)))))
    return nullptr;

  // A vector select must maintain the same number of elements in its operands.
  Type *CondTy = Cond->getType();
  Type *DestTy = BitCast.getType();
  if (auto *CondVTy = dyn_cast<VectorType>(CondTy))
    if (!DestTy->isVectorTy() ||
        CondVTy->getElementCount() !=
            cast<VectorType>(DestTy)->getElementCount())
      return nullptr;

  // FIXME: This transform is restricted from changing the select between
  // scalars and vectors to avoid backend problems caused by creating
  // potentially illegal operations. If a fix-up is added to handle that
  // situation, we can remove this check.
  if (DestTy->isVectorTy() != TVal->getType()->isVectorTy())
    return nullptr;

  auto *Sel = cast<Instruction>(BitCast.getOperand(0));
  Value *X;
  if (match(TVal, m_OneUse(m_BitCast(m_Value(X)))) && X->getType() == DestTy &&
      !isa<Constant>(X)) {
    // bitcast(select(Cond, bitcast(X), Y)) --> select'(Cond, X, bitcast(Y))
    Value *CastedVal = Builder.CreateBitCast(FVal, DestTy);
    return SelectInst::Create(Cond, X, CastedVal, "", nullptr, Sel);
  }

  if (match(FVal, m_OneUse(m_BitCast(m_Value(X)))) && X->getType() == DestTy &&
      !isa<Constant>(X)) {
    // bitcast(select(Cond, Y, bitcast(X))) --> select'(Cond, bitcast(Y), X)
    Value *CastedVal = Builder.CreateBitCast(TVal, DestTy);
    return SelectInst::Create(Cond, CastedVal, X, "", nullptr, Sel);
  }

  return nullptr;
}

/// Check if all users of CI are StoreInsts.
static bool hasStoreUsersOnly(CastInst &CI) {
  for (User *U : CI.users()) {
    if (!isa<StoreInst>(U))
      return false;
  }
  return true;
}

/// This function handles following case
///
///     A  ->  B    cast
///     PHI
///     B  ->  A    cast
///
/// All the related PHI nodes can be replaced by new PHI nodes with type A.
/// The uses of \p CI can be changed to the new PHI node corresponding to \p PN.
Instruction *InstCombinerImpl::optimizeBitCastFromPhi(CastInst &CI,
                                                      PHINode *PN) {
  // BitCast used by Store can be handled in InstCombineLoadStoreAlloca.cpp.
  if (hasStoreUsersOnly(CI))
    return nullptr;

  Value *Src = CI.getOperand(0);
  Type *SrcTy = Src->getType();         // Type B
  Type *DestTy = CI.getType();          // Type A

  SmallVector<PHINode *, 4> PhiWorklist;
  SmallSetVector<PHINode *, 4> OldPhiNodes;

  // Find all of the A->B casts and PHI nodes.
  // We need to inspect all related PHI nodes, but PHIs can be cyclic, so
  // OldPhiNodes is used to track all known PHI nodes, before adding a new
  // PHI to PhiWorklist, it is checked against and added to OldPhiNodes first.
  PhiWorklist.push_back(PN);
  OldPhiNodes.insert(PN);
  while (!PhiWorklist.empty()) {
    auto *OldPN = PhiWorklist.pop_back_val();
    for (Value *IncValue : OldPN->incoming_values()) {
      if (isa<Constant>(IncValue))
        continue;

      if (auto *LI = dyn_cast<LoadInst>(IncValue)) {
        // If there is a sequence of one or more load instructions, each loaded
        // value is used as address of later load instruction, bitcast is
        // necessary to change the value type, don't optimize it. For
        // simplicity we give up if the load address comes from another load.
        Value *Addr = LI->getOperand(0);
        if (Addr == &CI || isa<LoadInst>(Addr))
          return nullptr;
        // Don't tranform "load <256 x i32>, <256 x i32>*" to
        // "load x86_amx, x86_amx*", because x86_amx* is invalid.
        // TODO: Remove this check when bitcast between vector and x86_amx
        // is replaced with a specific intrinsic.
        if (DestTy->isX86_AMXTy())
          return nullptr;
        if (LI->hasOneUse() && LI->isSimple())
          continue;
        // If a LoadInst has more than one use, changing the type of loaded
        // value may create another bitcast.
        return nullptr;
      }

      if (auto *PNode = dyn_cast<PHINode>(IncValue)) {
        if (OldPhiNodes.insert(PNode))
          PhiWorklist.push_back(PNode);
        continue;
      }

      auto *BCI = dyn_cast<BitCastInst>(IncValue);
      // We can't handle other instructions.
      if (!BCI)
        return nullptr;

      // Verify it's a A->B cast.
      Type *TyA = BCI->getOperand(0)->getType();
      Type *TyB = BCI->getType();
      if (TyA != DestTy || TyB != SrcTy)
        return nullptr;
    }
  }

  // Check that each user of each old PHI node is something that we can
  // rewrite, so that all of the old PHI nodes can be cleaned up afterwards.
  for (auto *OldPN : OldPhiNodes) {
    for (User *V : OldPN->users()) {
      if (auto *SI = dyn_cast<StoreInst>(V)) {
        if (!SI->isSimple() || SI->getOperand(0) != OldPN)
          return nullptr;
      } else if (auto *BCI = dyn_cast<BitCastInst>(V)) {
        // Verify it's a B->A cast.
        Type *TyB = BCI->getOperand(0)->getType();
        Type *TyA = BCI->getType();
        if (TyA != DestTy || TyB != SrcTy)
          return nullptr;
      } else if (auto *PHI = dyn_cast<PHINode>(V)) {
        // As long as the user is another old PHI node, then even if we don't
        // rewrite it, the PHI web we're considering won't have any users
        // outside itself, so it'll be dead.
        if (!OldPhiNodes.contains(PHI))
          return nullptr;
      } else {
        return nullptr;
      }
    }
  }

  // For each old PHI node, create a corresponding new PHI node with a type A.
  SmallDenseMap<PHINode *, PHINode *> NewPNodes;
  for (auto *OldPN : OldPhiNodes) {
    Builder.SetInsertPoint(OldPN);
    PHINode *NewPN = Builder.CreatePHI(DestTy, OldPN->getNumOperands());
    NewPNodes[OldPN] = NewPN;
  }

  // Fill in the operands of new PHI nodes.
  for (auto *OldPN : OldPhiNodes) {
    PHINode *NewPN = NewPNodes[OldPN];
    for (unsigned j = 0, e = OldPN->getNumOperands(); j != e; ++j) {
      Value *V = OldPN->getOperand(j);
      Value *NewV = nullptr;
      if (auto *C = dyn_cast<Constant>(V)) {
        NewV = ConstantExpr::getBitCast(C, DestTy);
      } else if (auto *LI = dyn_cast<LoadInst>(V)) {
        // Explicitly perform load combine to make sure no opposing transform
        // can remove the bitcast in the meantime and trigger an infinite loop.
        Builder.SetInsertPoint(LI);
        NewV = combineLoadToNewType(*LI, DestTy);
        // Remove the old load and its use in the old phi, which itself becomes
        // dead once the whole transform finishes.
        replaceInstUsesWith(*LI, PoisonValue::get(LI->getType()));
        eraseInstFromFunction(*LI);
      } else if (auto *BCI = dyn_cast<BitCastInst>(V)) {
        NewV = BCI->getOperand(0);
      } else if (auto *PrevPN = dyn_cast<PHINode>(V)) {
        NewV = NewPNodes[PrevPN];
      }
      assert(NewV);
      NewPN->addIncoming(NewV, OldPN->getIncomingBlock(j));
    }
  }

  // Traverse all accumulated PHI nodes and process its users,
  // which are Stores and BitcCasts. Without this processing
  // NewPHI nodes could be replicated and could lead to extra
  // moves generated after DeSSA.
  // If there is a store with type B, change it to type A.


  // Replace users of BitCast B->A with NewPHI. These will help
  // later to get rid off a closure formed by OldPHI nodes.
  Instruction *RetVal = nullptr;
  for (auto *OldPN : OldPhiNodes) {
    PHINode *NewPN = NewPNodes[OldPN];
    for (User *V : make_early_inc_range(OldPN->users())) {
      if (auto *SI = dyn_cast<StoreInst>(V)) {
        assert(SI->isSimple() && SI->getOperand(0) == OldPN);
        Builder.SetInsertPoint(SI);
        auto *NewBC =
          cast<BitCastInst>(Builder.CreateBitCast(NewPN, SrcTy));
        SI->setOperand(0, NewBC);
        Worklist.push(SI);
        assert(hasStoreUsersOnly(*NewBC));
      }
      else if (auto *BCI = dyn_cast<BitCastInst>(V)) {
        Type *TyB = BCI->getOperand(0)->getType();
        Type *TyA = BCI->getType();
        assert(TyA == DestTy && TyB == SrcTy);
        (void) TyA;
        (void) TyB;
        Instruction *I = replaceInstUsesWith(*BCI, NewPN);
        if (BCI == &CI)
          RetVal = I;
      } else if (auto *PHI = dyn_cast<PHINode>(V)) {
        assert(OldPhiNodes.contains(PHI));
        (void) PHI;
      } else {
        llvm_unreachable("all uses should be handled");
      }
    }
  }

  return RetVal;
}

static Instruction *convertBitCastToGEP(BitCastInst &CI, IRBuilderBase &Builder,
                                        const DataLayout &DL) {
  Value *Src = CI.getOperand(0);
  PointerType *SrcPTy = cast<PointerType>(Src->getType());
  PointerType *DstPTy = cast<PointerType>(CI.getType());

  // Bitcasts involving opaque pointers cannot be converted into a GEP.
  if (SrcPTy->isOpaque() || DstPTy->isOpaque())
    return nullptr;

  Type *DstElTy = DstPTy->getNonOpaquePointerElementType();
  Type *SrcElTy = SrcPTy->getNonOpaquePointerElementType();

  // When the type pointed to is not sized the cast cannot be
  // turned into a gep.
  if (!SrcElTy->isSized())
    return nullptr;

  // If the source and destination are pointers, and this cast is equivalent
  // to a getelementptr X, 0, 0, 0...  turn it into the appropriate gep.
  // This can enhance SROA and other transforms that want type-safe pointers.
  unsigned NumZeros = 0;
  while (SrcElTy && SrcElTy != DstElTy) {
    SrcElTy = GetElementPtrInst::getTypeAtIndex(SrcElTy, (uint64_t)0);
    ++NumZeros;
  }

  // If we found a path from the src to dest, create the getelementptr now.
  if (SrcElTy == DstElTy) {
    SmallVector<Value *, 8> Idxs(NumZeros + 1, Builder.getInt32(0));
    GetElementPtrInst *GEP = GetElementPtrInst::Create(
        SrcPTy->getNonOpaquePointerElementType(), Src, Idxs);

    // If the source pointer is dereferenceable, then assume it points to an
    // allocated object and apply "inbounds" to the GEP.
    bool CanBeNull, CanBeFreed;
    if (Src->getPointerDereferenceableBytes(DL, CanBeNull, CanBeFreed)) {
      // In a non-default address space (not 0), a null pointer can not be
      // assumed inbounds, so ignore that case (dereferenceable_or_null).
      // The reason is that 'null' is not treated differently in these address
      // spaces, and we consequently ignore the 'gep inbounds' special case
      // for 'null' which allows 'inbounds' on 'null' if the indices are
      // zeros.
      if (SrcPTy->getAddressSpace() == 0 || !CanBeNull)
        GEP->setIsInBounds();
    }
    return GEP;
  }
  return nullptr;
}

Instruction *InstCombinerImpl::visitBitCast(BitCastInst &CI) {
  // If the operands are integer typed then apply the integer transforms,
  // otherwise just apply the common ones.
  Value *Src = CI.getOperand(0);
  Type *SrcTy = Src->getType();
  Type *DestTy = CI.getType();

  // Get rid of casts from one type to the same type. These are useless and can
  // be replaced by the operand.
  if (DestTy == Src->getType())
    return replaceInstUsesWith(CI, Src);

  if (isa<PointerType>(SrcTy) && isa<PointerType>(DestTy)) {
    // If we are casting a alloca to a pointer to a type of the same
    // size, rewrite the allocation instruction to allocate the "right" type.
    // There is no need to modify malloc calls because it is their bitcast that
    // needs to be cleaned up.
    if (AllocaInst *AI = dyn_cast<AllocaInst>(Src))
      if (Instruction *V = PromoteCastOfAllocation(CI, *AI))
        return V;

    if (Instruction *I = convertBitCastToGEP(CI, Builder, DL))
      return I;
  }

  if (FixedVectorType *DestVTy = dyn_cast<FixedVectorType>(DestTy)) {
    // Beware: messing with this target-specific oddity may cause trouble.
    if (DestVTy->getNumElements() == 1 && SrcTy->isX86_MMXTy()) {
      Value *Elem = Builder.CreateBitCast(Src, DestVTy->getElementType());
      return InsertElementInst::Create(PoisonValue::get(DestTy), Elem,
                     Constant::getNullValue(Type::getInt32Ty(CI.getContext())));
    }

    if (isa<IntegerType>(SrcTy)) {
      // If this is a cast from an integer to vector, check to see if the input
      // is a trunc or zext of a bitcast from vector.  If so, we can replace all
      // the casts with a shuffle and (potentially) a bitcast.
      if (isa<TruncInst>(Src) || isa<ZExtInst>(Src)) {
        CastInst *SrcCast = cast<CastInst>(Src);
        if (BitCastInst *BCIn = dyn_cast<BitCastInst>(SrcCast->getOperand(0)))
          if (isa<VectorType>(BCIn->getOperand(0)->getType()))
            if (Instruction *I = optimizeVectorResizeWithIntegerBitCasts(
                    BCIn->getOperand(0), cast<VectorType>(DestTy), *this))
              return I;
      }

      // If the input is an 'or' instruction, we may be doing shifts and ors to
      // assemble the elements of the vector manually.  Try to rip the code out
      // and replace it with insertelements.
      if (Value *V = optimizeIntegerToVectorInsertions(CI, *this))
        return replaceInstUsesWith(CI, V);
    }
  }

  if (FixedVectorType *SrcVTy = dyn_cast<FixedVectorType>(SrcTy)) {
    if (SrcVTy->getNumElements() == 1) {
      // If our destination is not a vector, then make this a straight
      // scalar-scalar cast.
      if (!DestTy->isVectorTy()) {
        Value *Elem =
          Builder.CreateExtractElement(Src,
                     Constant::getNullValue(Type::getInt32Ty(CI.getContext())));
        return CastInst::Create(Instruction::BitCast, Elem, DestTy);
      }

      // Otherwise, see if our source is an insert. If so, then use the scalar
      // component directly:
      // bitcast (inselt <1 x elt> V, X, 0) to <n x m> --> bitcast X to <n x m>
      if (auto *InsElt = dyn_cast<InsertElementInst>(Src))
        return new BitCastInst(InsElt->getOperand(1), DestTy);
    }

    // Convert an artificial vector insert into more analyzable bitwise logic.
    unsigned BitWidth = DestTy->getScalarSizeInBits();
    Value *X, *Y;
    uint64_t IndexC;
    if (match(Src, m_OneUse(m_InsertElt(m_OneUse(m_BitCast(m_Value(X))),
                                        m_Value(Y), m_ConstantInt(IndexC)))) &&
        DestTy->isIntegerTy() && X->getType() == DestTy &&
        Y->getType()->isIntegerTy() && isDesirableIntType(BitWidth)) {
      // Adjust for big endian - the LSBs are at the high index.
      if (DL.isBigEndian())
        IndexC = SrcVTy->getNumElements() - 1 - IndexC;

      // We only handle (endian-normalized) insert to index 0. Any other insert
      // would require a left-shift, so that is an extra instruction.
      if (IndexC == 0) {
        // bitcast (inselt (bitcast X), Y, 0) --> or (and X, MaskC), (zext Y)
        unsigned EltWidth = Y->getType()->getScalarSizeInBits();
        APInt MaskC = APInt::getHighBitsSet(BitWidth, BitWidth - EltWidth);
        Value *AndX = Builder.CreateAnd(X, MaskC);
        Value *ZextY = Builder.CreateZExt(Y, DestTy);
        return BinaryOperator::CreateOr(AndX, ZextY);
      }
    }
  }

  if (auto *Shuf = dyn_cast<ShuffleVectorInst>(Src)) {
    // Okay, we have (bitcast (shuffle ..)).  Check to see if this is
    // a bitcast to a vector with the same # elts.
    Value *ShufOp0 = Shuf->getOperand(0);
    Value *ShufOp1 = Shuf->getOperand(1);
    auto ShufElts = cast<VectorType>(Shuf->getType())->getElementCount();
    auto SrcVecElts = cast<VectorType>(ShufOp0->getType())->getElementCount();
    if (Shuf->hasOneUse() && DestTy->isVectorTy() &&
        cast<VectorType>(DestTy)->getElementCount() == ShufElts &&
        ShufElts == SrcVecElts) {
      BitCastInst *Tmp;
      // If either of the operands is a cast from CI.getType(), then
      // evaluating the shuffle in the casted destination's type will allow
      // us to eliminate at least one cast.
      if (((Tmp = dyn_cast<BitCastInst>(ShufOp0)) &&
           Tmp->getOperand(0)->getType() == DestTy) ||
          ((Tmp = dyn_cast<BitCastInst>(ShufOp1)) &&
           Tmp->getOperand(0)->getType() == DestTy)) {
        Value *LHS = Builder.CreateBitCast(ShufOp0, DestTy);
        Value *RHS = Builder.CreateBitCast(ShufOp1, DestTy);
        // Return a new shuffle vector.  Use the same element ID's, as we
        // know the vector types match #elts.
        return new ShuffleVectorInst(LHS, RHS, Shuf->getShuffleMask());
      }
    }

    // A bitcasted-to-scalar and byte-reversing shuffle is better recognized as
    // a byte-swap:
    // bitcast <N x i8> (shuf X, undef, <N, N-1,...0>) --> bswap (bitcast X)
    // TODO: We should match the related pattern for bitreverse.
    if (DestTy->isIntegerTy() &&
        DL.isLegalInteger(DestTy->getScalarSizeInBits()) &&
        SrcTy->getScalarSizeInBits() == 8 &&
        ShufElts.getKnownMinValue() % 2 == 0 && Shuf->hasOneUse() &&
        Shuf->isReverse()) {
      assert(ShufOp0->getType() == SrcTy && "Unexpected shuffle mask");
      assert(match(ShufOp1, m_Undef()) && "Unexpected shuffle op");
      Function *Bswap =
          Intrinsic::getDeclaration(CI.getModule(), Intrinsic::bswap, DestTy);
      Value *ScalarX = Builder.CreateBitCast(ShufOp0, DestTy);
      return CallInst::Create(Bswap, { ScalarX });
    }
  }

  // Handle the A->B->A cast, and there is an intervening PHI node.
  if (PHINode *PN = dyn_cast<PHINode>(Src))
    if (Instruction *I = optimizeBitCastFromPhi(CI, PN))
      return I;

  if (Instruction *I = canonicalizeBitCastExtElt(CI, *this))
    return I;

  if (Instruction *I = foldBitCastBitwiseLogic(CI, Builder))
    return I;

  if (Instruction *I = foldBitCastSelect(CI, Builder))
    return I;

  if (SrcTy->isPointerTy())
    return commonPointerCastTransforms(CI);
  return commonCastTransforms(CI);
}

Instruction *InstCombinerImpl::visitAddrSpaceCast(AddrSpaceCastInst &CI) {
  // If the destination pointer element type is not the same as the source's
  // first do a bitcast to the destination type, and then the addrspacecast.
  // This allows the cast to be exposed to other transforms.
  Value *Src = CI.getOperand(0);
  PointerType *SrcTy = cast<PointerType>(Src->getType()->getScalarType());
  PointerType *DestTy = cast<PointerType>(CI.getType()->getScalarType());

  if (!SrcTy->hasSameElementTypeAs(DestTy)) {
    Type *MidTy =
        PointerType::getWithSamePointeeType(DestTy, SrcTy->getAddressSpace());
    // Handle vectors of pointers.
    if (VectorType *VT = dyn_cast<VectorType>(CI.getType()))
      MidTy = VectorType::get(MidTy, VT->getElementCount());

    Value *NewBitCast = Builder.CreateBitCast(Src, MidTy);
    return new AddrSpaceCastInst(NewBitCast, CI.getType());
  }

  return commonPointerCastTransforms(CI);
}
