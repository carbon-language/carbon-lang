//===- InstructionCombining.cpp - Combine multiple instructions -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// InstructionCombining - Combine instructions to form fewer, simple
// instructions.  This pass does not modify the CFG.  This pass is where
// algebraic simplification happens.
//
// This pass combines things like:
//    %Y = add i32 %X, 1
//    %Z = add i32 %Y, 1
// into:
//    %Z = add i32 %X, 2
//
// This is a simple worklist driven algorithm.
//
// This pass guarantees that the following canonicalizations are performed on
// the program:
//    1. If a binary operator has a constant operand, it is moved to the RHS
//    2. Bitwise operators with constant operands are always grouped so that
//       shifts are performed first, then or's, then and's, then xor's.
//    3. Compare instructions are converted from <,>,<=,>= to ==,!= if possible
//    4. All cmp instructions on boolean values are replaced with logical ops
//    5. add X, X is represented as (X*2) => (X << 1)
//    6. Multiplies with a power-of-two constant argument are transformed into
//       shifts.
//   ... etc.
//
//===----------------------------------------------------------------------===//

#include "InstCombineInternal.h"
#include "llvm-c/Initialization.h"
#include "llvm-c/Transforms/InstCombine.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/EHPersonalities.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LazyBlockFrequencyInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/TargetFolder.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Utils/Local.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#define DEBUG_TYPE "instcombine"
#include "llvm/Transforms/Utils/InstructionWorklist.h"

using namespace llvm;
using namespace llvm::PatternMatch;

STATISTIC(NumWorklistIterations,
          "Number of instruction combining iterations performed");

STATISTIC(NumCombined , "Number of insts combined");
STATISTIC(NumConstProp, "Number of constant folds");
STATISTIC(NumDeadInst , "Number of dead inst eliminated");
STATISTIC(NumSunkInst , "Number of instructions sunk");
STATISTIC(NumExpand,    "Number of expansions");
STATISTIC(NumFactor   , "Number of factorizations");
STATISTIC(NumReassoc  , "Number of reassociations");
DEBUG_COUNTER(VisitCounter, "instcombine-visit",
              "Controls which instructions are visited");

// FIXME: these limits eventually should be as low as 2.
static constexpr unsigned InstCombineDefaultMaxIterations = 1000;
#ifndef NDEBUG
static constexpr unsigned InstCombineDefaultInfiniteLoopThreshold = 100;
#else
static constexpr unsigned InstCombineDefaultInfiniteLoopThreshold = 1000;
#endif

static cl::opt<bool>
EnableCodeSinking("instcombine-code-sinking", cl::desc("Enable code sinking"),
                                              cl::init(true));

static cl::opt<unsigned> LimitMaxIterations(
    "instcombine-max-iterations",
    cl::desc("Limit the maximum number of instruction combining iterations"),
    cl::init(InstCombineDefaultMaxIterations));

static cl::opt<unsigned> InfiniteLoopDetectionThreshold(
    "instcombine-infinite-loop-threshold",
    cl::desc("Number of instruction combining iterations considered an "
             "infinite loop"),
    cl::init(InstCombineDefaultInfiniteLoopThreshold), cl::Hidden);

static cl::opt<unsigned>
MaxArraySize("instcombine-maxarray-size", cl::init(1024),
             cl::desc("Maximum array size considered when doing a combine"));

// FIXME: Remove this flag when it is no longer necessary to convert
// llvm.dbg.declare to avoid inaccurate debug info. Setting this to false
// increases variable availability at the cost of accuracy. Variables that
// cannot be promoted by mem2reg or SROA will be described as living in memory
// for their entire lifetime. However, passes like DSE and instcombine can
// delete stores to the alloca, leading to misleading and inaccurate debug
// information. This flag can be removed when those passes are fixed.
static cl::opt<unsigned> ShouldLowerDbgDeclare("instcombine-lower-dbg-declare",
                                               cl::Hidden, cl::init(true));

Optional<Instruction *>
InstCombiner::targetInstCombineIntrinsic(IntrinsicInst &II) {
  // Handle target specific intrinsics
  if (II.getCalledFunction()->isTargetIntrinsic()) {
    return TTI.instCombineIntrinsic(*this, II);
  }
  return None;
}

Optional<Value *> InstCombiner::targetSimplifyDemandedUseBitsIntrinsic(
    IntrinsicInst &II, APInt DemandedMask, KnownBits &Known,
    bool &KnownBitsComputed) {
  // Handle target specific intrinsics
  if (II.getCalledFunction()->isTargetIntrinsic()) {
    return TTI.simplifyDemandedUseBitsIntrinsic(*this, II, DemandedMask, Known,
                                                KnownBitsComputed);
  }
  return None;
}

Optional<Value *> InstCombiner::targetSimplifyDemandedVectorEltsIntrinsic(
    IntrinsicInst &II, APInt DemandedElts, APInt &UndefElts, APInt &UndefElts2,
    APInt &UndefElts3,
    std::function<void(Instruction *, unsigned, APInt, APInt &)>
        SimplifyAndSetOp) {
  // Handle target specific intrinsics
  if (II.getCalledFunction()->isTargetIntrinsic()) {
    return TTI.simplifyDemandedVectorEltsIntrinsic(
        *this, II, DemandedElts, UndefElts, UndefElts2, UndefElts3,
        SimplifyAndSetOp);
  }
  return None;
}

Value *InstCombinerImpl::EmitGEPOffset(User *GEP) {
  return llvm::EmitGEPOffset(&Builder, DL, GEP);
}

/// Legal integers and common types are considered desirable. This is used to
/// avoid creating instructions with types that may not be supported well by the
/// the backend.
/// NOTE: This treats i8, i16 and i32 specially because they are common
///       types in frontend languages.
bool InstCombinerImpl::isDesirableIntType(unsigned BitWidth) const {
  switch (BitWidth) {
  case 8:
  case 16:
  case 32:
    return true;
  default:
    return DL.isLegalInteger(BitWidth);
  }
}

/// Return true if it is desirable to convert an integer computation from a
/// given bit width to a new bit width.
/// We don't want to convert from a legal to an illegal type or from a smaller
/// to a larger illegal type. A width of '1' is always treated as a desirable
/// type because i1 is a fundamental type in IR, and there are many specialized
/// optimizations for i1 types. Common/desirable widths are equally treated as
/// legal to convert to, in order to open up more combining opportunities.
bool InstCombinerImpl::shouldChangeType(unsigned FromWidth,
                                        unsigned ToWidth) const {
  bool FromLegal = FromWidth == 1 || DL.isLegalInteger(FromWidth);
  bool ToLegal = ToWidth == 1 || DL.isLegalInteger(ToWidth);

  // Convert to desirable widths even if they are not legal types.
  // Only shrink types, to prevent infinite loops.
  if (ToWidth < FromWidth && isDesirableIntType(ToWidth))
    return true;

  // If this is a legal integer from type, and the result would be an illegal
  // type, don't do the transformation.
  if (FromLegal && !ToLegal)
    return false;

  // Otherwise, if both are illegal, do not increase the size of the result. We
  // do allow things like i160 -> i64, but not i64 -> i160.
  if (!FromLegal && !ToLegal && ToWidth > FromWidth)
    return false;

  return true;
}

/// Return true if it is desirable to convert a computation from 'From' to 'To'.
/// We don't want to convert from a legal to an illegal type or from a smaller
/// to a larger illegal type. i1 is always treated as a legal type because it is
/// a fundamental type in IR, and there are many specialized optimizations for
/// i1 types.
bool InstCombinerImpl::shouldChangeType(Type *From, Type *To) const {
  // TODO: This could be extended to allow vectors. Datalayout changes might be
  // needed to properly support that.
  if (!From->isIntegerTy() || !To->isIntegerTy())
    return false;

  unsigned FromWidth = From->getPrimitiveSizeInBits();
  unsigned ToWidth = To->getPrimitiveSizeInBits();
  return shouldChangeType(FromWidth, ToWidth);
}

// Return true, if No Signed Wrap should be maintained for I.
// The No Signed Wrap flag can be kept if the operation "B (I.getOpcode) C",
// where both B and C should be ConstantInts, results in a constant that does
// not overflow. This function only handles the Add and Sub opcodes. For
// all other opcodes, the function conservatively returns false.
static bool maintainNoSignedWrap(BinaryOperator &I, Value *B, Value *C) {
  auto *OBO = dyn_cast<OverflowingBinaryOperator>(&I);
  if (!OBO || !OBO->hasNoSignedWrap())
    return false;

  // We reason about Add and Sub Only.
  Instruction::BinaryOps Opcode = I.getOpcode();
  if (Opcode != Instruction::Add && Opcode != Instruction::Sub)
    return false;

  const APInt *BVal, *CVal;
  if (!match(B, m_APInt(BVal)) || !match(C, m_APInt(CVal)))
    return false;

  bool Overflow = false;
  if (Opcode == Instruction::Add)
    (void)BVal->sadd_ov(*CVal, Overflow);
  else
    (void)BVal->ssub_ov(*CVal, Overflow);

  return !Overflow;
}

static bool hasNoUnsignedWrap(BinaryOperator &I) {
  auto *OBO = dyn_cast<OverflowingBinaryOperator>(&I);
  return OBO && OBO->hasNoUnsignedWrap();
}

static bool hasNoSignedWrap(BinaryOperator &I) {
  auto *OBO = dyn_cast<OverflowingBinaryOperator>(&I);
  return OBO && OBO->hasNoSignedWrap();
}

/// Conservatively clears subclassOptionalData after a reassociation or
/// commutation. We preserve fast-math flags when applicable as they can be
/// preserved.
static void ClearSubclassDataAfterReassociation(BinaryOperator &I) {
  FPMathOperator *FPMO = dyn_cast<FPMathOperator>(&I);
  if (!FPMO) {
    I.clearSubclassOptionalData();
    return;
  }

  FastMathFlags FMF = I.getFastMathFlags();
  I.clearSubclassOptionalData();
  I.setFastMathFlags(FMF);
}

/// Combine constant operands of associative operations either before or after a
/// cast to eliminate one of the associative operations:
/// (op (cast (op X, C2)), C1) --> (cast (op X, op (C1, C2)))
/// (op (cast (op X, C2)), C1) --> (op (cast X), op (C1, C2))
static bool simplifyAssocCastAssoc(BinaryOperator *BinOp1,
                                   InstCombinerImpl &IC) {
  auto *Cast = dyn_cast<CastInst>(BinOp1->getOperand(0));
  if (!Cast || !Cast->hasOneUse())
    return false;

  // TODO: Enhance logic for other casts and remove this check.
  auto CastOpcode = Cast->getOpcode();
  if (CastOpcode != Instruction::ZExt)
    return false;

  // TODO: Enhance logic for other BinOps and remove this check.
  if (!BinOp1->isBitwiseLogicOp())
    return false;

  auto AssocOpcode = BinOp1->getOpcode();
  auto *BinOp2 = dyn_cast<BinaryOperator>(Cast->getOperand(0));
  if (!BinOp2 || !BinOp2->hasOneUse() || BinOp2->getOpcode() != AssocOpcode)
    return false;

  Constant *C1, *C2;
  if (!match(BinOp1->getOperand(1), m_Constant(C1)) ||
      !match(BinOp2->getOperand(1), m_Constant(C2)))
    return false;

  // TODO: This assumes a zext cast.
  // Eg, if it was a trunc, we'd cast C1 to the source type because casting C2
  // to the destination type might lose bits.

  // Fold the constants together in the destination type:
  // (op (cast (op X, C2)), C1) --> (op (cast X), FoldedC)
  Type *DestTy = C1->getType();
  Constant *CastC2 = ConstantExpr::getCast(CastOpcode, C2, DestTy);
  Constant *FoldedC = ConstantExpr::get(AssocOpcode, C1, CastC2);
  IC.replaceOperand(*Cast, 0, BinOp2->getOperand(0));
  IC.replaceOperand(*BinOp1, 1, FoldedC);
  return true;
}

// Simplifies IntToPtr/PtrToInt RoundTrip Cast To BitCast.
// inttoptr ( ptrtoint (x) ) --> x
Value *InstCombinerImpl::simplifyIntToPtrRoundTripCast(Value *Val) {
  auto *IntToPtr = dyn_cast<IntToPtrInst>(Val);
  if (IntToPtr && DL.getPointerTypeSizeInBits(IntToPtr->getDestTy()) ==
                      DL.getTypeSizeInBits(IntToPtr->getSrcTy())) {
    auto *PtrToInt = dyn_cast<PtrToIntInst>(IntToPtr->getOperand(0));
    Type *CastTy = IntToPtr->getDestTy();
    if (PtrToInt &&
        CastTy->getPointerAddressSpace() ==
            PtrToInt->getSrcTy()->getPointerAddressSpace() &&
        DL.getPointerTypeSizeInBits(PtrToInt->getSrcTy()) ==
            DL.getTypeSizeInBits(PtrToInt->getDestTy())) {
      return CastInst::CreateBitOrPointerCast(PtrToInt->getOperand(0), CastTy,
                                              "", PtrToInt);
    }
  }
  return nullptr;
}

/// This performs a few simplifications for operators that are associative or
/// commutative:
///
///  Commutative operators:
///
///  1. Order operands such that they are listed from right (least complex) to
///     left (most complex).  This puts constants before unary operators before
///     binary operators.
///
///  Associative operators:
///
///  2. Transform: "(A op B) op C" ==> "A op (B op C)" if "B op C" simplifies.
///  3. Transform: "A op (B op C)" ==> "(A op B) op C" if "A op B" simplifies.
///
///  Associative and commutative operators:
///
///  4. Transform: "(A op B) op C" ==> "(C op A) op B" if "C op A" simplifies.
///  5. Transform: "A op (B op C)" ==> "B op (C op A)" if "C op A" simplifies.
///  6. Transform: "(A op C1) op (B op C2)" ==> "(A op B) op (C1 op C2)"
///     if C1 and C2 are constants.
bool InstCombinerImpl::SimplifyAssociativeOrCommutative(BinaryOperator &I) {
  Instruction::BinaryOps Opcode = I.getOpcode();
  bool Changed = false;

  do {
    // Order operands such that they are listed from right (least complex) to
    // left (most complex).  This puts constants before unary operators before
    // binary operators.
    if (I.isCommutative() && getComplexity(I.getOperand(0)) <
        getComplexity(I.getOperand(1)))
      Changed = !I.swapOperands();

    BinaryOperator *Op0 = dyn_cast<BinaryOperator>(I.getOperand(0));
    BinaryOperator *Op1 = dyn_cast<BinaryOperator>(I.getOperand(1));

    if (I.isAssociative()) {
      // Transform: "(A op B) op C" ==> "A op (B op C)" if "B op C" simplifies.
      if (Op0 && Op0->getOpcode() == Opcode) {
        Value *A = Op0->getOperand(0);
        Value *B = Op0->getOperand(1);
        Value *C = I.getOperand(1);

        // Does "B op C" simplify?
        if (Value *V = SimplifyBinOp(Opcode, B, C, SQ.getWithInstruction(&I))) {
          // It simplifies to V.  Form "A op V".
          replaceOperand(I, 0, A);
          replaceOperand(I, 1, V);
          bool IsNUW = hasNoUnsignedWrap(I) && hasNoUnsignedWrap(*Op0);
          bool IsNSW = maintainNoSignedWrap(I, B, C) && hasNoSignedWrap(*Op0);

          // Conservatively clear all optional flags since they may not be
          // preserved by the reassociation. Reset nsw/nuw based on the above
          // analysis.
          ClearSubclassDataAfterReassociation(I);

          // Note: this is only valid because SimplifyBinOp doesn't look at
          // the operands to Op0.
          if (IsNUW)
            I.setHasNoUnsignedWrap(true);

          if (IsNSW)
            I.setHasNoSignedWrap(true);

          Changed = true;
          ++NumReassoc;
          continue;
        }
      }

      // Transform: "A op (B op C)" ==> "(A op B) op C" if "A op B" simplifies.
      if (Op1 && Op1->getOpcode() == Opcode) {
        Value *A = I.getOperand(0);
        Value *B = Op1->getOperand(0);
        Value *C = Op1->getOperand(1);

        // Does "A op B" simplify?
        if (Value *V = SimplifyBinOp(Opcode, A, B, SQ.getWithInstruction(&I))) {
          // It simplifies to V.  Form "V op C".
          replaceOperand(I, 0, V);
          replaceOperand(I, 1, C);
          // Conservatively clear the optional flags, since they may not be
          // preserved by the reassociation.
          ClearSubclassDataAfterReassociation(I);
          Changed = true;
          ++NumReassoc;
          continue;
        }
      }
    }

    if (I.isAssociative() && I.isCommutative()) {
      if (simplifyAssocCastAssoc(&I, *this)) {
        Changed = true;
        ++NumReassoc;
        continue;
      }

      // Transform: "(A op B) op C" ==> "(C op A) op B" if "C op A" simplifies.
      if (Op0 && Op0->getOpcode() == Opcode) {
        Value *A = Op0->getOperand(0);
        Value *B = Op0->getOperand(1);
        Value *C = I.getOperand(1);

        // Does "C op A" simplify?
        if (Value *V = SimplifyBinOp(Opcode, C, A, SQ.getWithInstruction(&I))) {
          // It simplifies to V.  Form "V op B".
          replaceOperand(I, 0, V);
          replaceOperand(I, 1, B);
          // Conservatively clear the optional flags, since they may not be
          // preserved by the reassociation.
          ClearSubclassDataAfterReassociation(I);
          Changed = true;
          ++NumReassoc;
          continue;
        }
      }

      // Transform: "A op (B op C)" ==> "B op (C op A)" if "C op A" simplifies.
      if (Op1 && Op1->getOpcode() == Opcode) {
        Value *A = I.getOperand(0);
        Value *B = Op1->getOperand(0);
        Value *C = Op1->getOperand(1);

        // Does "C op A" simplify?
        if (Value *V = SimplifyBinOp(Opcode, C, A, SQ.getWithInstruction(&I))) {
          // It simplifies to V.  Form "B op V".
          replaceOperand(I, 0, B);
          replaceOperand(I, 1, V);
          // Conservatively clear the optional flags, since they may not be
          // preserved by the reassociation.
          ClearSubclassDataAfterReassociation(I);
          Changed = true;
          ++NumReassoc;
          continue;
        }
      }

      // Transform: "(A op C1) op (B op C2)" ==> "(A op B) op (C1 op C2)"
      // if C1 and C2 are constants.
      Value *A, *B;
      Constant *C1, *C2;
      if (Op0 && Op1 &&
          Op0->getOpcode() == Opcode && Op1->getOpcode() == Opcode &&
          match(Op0, m_OneUse(m_BinOp(m_Value(A), m_Constant(C1)))) &&
          match(Op1, m_OneUse(m_BinOp(m_Value(B), m_Constant(C2))))) {
        bool IsNUW = hasNoUnsignedWrap(I) &&
           hasNoUnsignedWrap(*Op0) &&
           hasNoUnsignedWrap(*Op1);
         BinaryOperator *NewBO = (IsNUW && Opcode == Instruction::Add) ?
           BinaryOperator::CreateNUW(Opcode, A, B) :
           BinaryOperator::Create(Opcode, A, B);

         if (isa<FPMathOperator>(NewBO)) {
          FastMathFlags Flags = I.getFastMathFlags();
          Flags &= Op0->getFastMathFlags();
          Flags &= Op1->getFastMathFlags();
          NewBO->setFastMathFlags(Flags);
        }
        InsertNewInstWith(NewBO, I);
        NewBO->takeName(Op1);
        replaceOperand(I, 0, NewBO);
        replaceOperand(I, 1, ConstantExpr::get(Opcode, C1, C2));
        // Conservatively clear the optional flags, since they may not be
        // preserved by the reassociation.
        ClearSubclassDataAfterReassociation(I);
        if (IsNUW)
          I.setHasNoUnsignedWrap(true);

        Changed = true;
        continue;
      }
    }

    // No further simplifications.
    return Changed;
  } while (true);
}

/// Return whether "X LOp (Y ROp Z)" is always equal to
/// "(X LOp Y) ROp (X LOp Z)".
static bool leftDistributesOverRight(Instruction::BinaryOps LOp,
                                     Instruction::BinaryOps ROp) {
  // X & (Y | Z) <--> (X & Y) | (X & Z)
  // X & (Y ^ Z) <--> (X & Y) ^ (X & Z)
  if (LOp == Instruction::And)
    return ROp == Instruction::Or || ROp == Instruction::Xor;

  // X | (Y & Z) <--> (X | Y) & (X | Z)
  if (LOp == Instruction::Or)
    return ROp == Instruction::And;

  // X * (Y + Z) <--> (X * Y) + (X * Z)
  // X * (Y - Z) <--> (X * Y) - (X * Z)
  if (LOp == Instruction::Mul)
    return ROp == Instruction::Add || ROp == Instruction::Sub;

  return false;
}

/// Return whether "(X LOp Y) ROp Z" is always equal to
/// "(X ROp Z) LOp (Y ROp Z)".
static bool rightDistributesOverLeft(Instruction::BinaryOps LOp,
                                     Instruction::BinaryOps ROp) {
  if (Instruction::isCommutative(ROp))
    return leftDistributesOverRight(ROp, LOp);

  // (X {&|^} Y) >> Z <--> (X >> Z) {&|^} (Y >> Z) for all shifts.
  return Instruction::isBitwiseLogicOp(LOp) && Instruction::isShift(ROp);

  // TODO: It would be nice to handle division, aka "(X + Y)/Z = X/Z + Y/Z",
  // but this requires knowing that the addition does not overflow and other
  // such subtleties.
}

/// This function returns identity value for given opcode, which can be used to
/// factor patterns like (X * 2) + X ==> (X * 2) + (X * 1) ==> X * (2 + 1).
static Value *getIdentityValue(Instruction::BinaryOps Opcode, Value *V) {
  if (isa<Constant>(V))
    return nullptr;

  return ConstantExpr::getBinOpIdentity(Opcode, V->getType());
}

/// This function predicates factorization using distributive laws. By default,
/// it just returns the 'Op' inputs. But for special-cases like
/// 'add(shl(X, 5), ...)', this function will have TopOpcode == Instruction::Add
/// and Op = shl(X, 5). The 'shl' is treated as the more general 'mul X, 32' to
/// allow more factorization opportunities.
static Instruction::BinaryOps
getBinOpsForFactorization(Instruction::BinaryOps TopOpcode, BinaryOperator *Op,
                          Value *&LHS, Value *&RHS) {
  assert(Op && "Expected a binary operator");
  LHS = Op->getOperand(0);
  RHS = Op->getOperand(1);
  if (TopOpcode == Instruction::Add || TopOpcode == Instruction::Sub) {
    Constant *C;
    if (match(Op, m_Shl(m_Value(), m_Constant(C)))) {
      // X << C --> X * (1 << C)
      RHS = ConstantExpr::getShl(ConstantInt::get(Op->getType(), 1), C);
      return Instruction::Mul;
    }
    // TODO: We can add other conversions e.g. shr => div etc.
  }
  return Op->getOpcode();
}

/// This tries to simplify binary operations by factorizing out common terms
/// (e. g. "(A*B)+(A*C)" -> "A*(B+C)").
Value *InstCombinerImpl::tryFactorization(BinaryOperator &I,
                                          Instruction::BinaryOps InnerOpcode,
                                          Value *A, Value *B, Value *C,
                                          Value *D) {
  assert(A && B && C && D && "All values must be provided");

  Value *V = nullptr;
  Value *SimplifiedInst = nullptr;
  Value *LHS = I.getOperand(0), *RHS = I.getOperand(1);
  Instruction::BinaryOps TopLevelOpcode = I.getOpcode();

  // Does "X op' Y" always equal "Y op' X"?
  bool InnerCommutative = Instruction::isCommutative(InnerOpcode);

  // Does "X op' (Y op Z)" always equal "(X op' Y) op (X op' Z)"?
  if (leftDistributesOverRight(InnerOpcode, TopLevelOpcode))
    // Does the instruction have the form "(A op' B) op (A op' D)" or, in the
    // commutative case, "(A op' B) op (C op' A)"?
    if (A == C || (InnerCommutative && A == D)) {
      if (A != C)
        std::swap(C, D);
      // Consider forming "A op' (B op D)".
      // If "B op D" simplifies then it can be formed with no cost.
      V = SimplifyBinOp(TopLevelOpcode, B, D, SQ.getWithInstruction(&I));
      // If "B op D" doesn't simplify then only go on if both of the existing
      // operations "A op' B" and "C op' D" will be zapped as no longer used.
      if (!V && LHS->hasOneUse() && RHS->hasOneUse())
        V = Builder.CreateBinOp(TopLevelOpcode, B, D, RHS->getName());
      if (V) {
        SimplifiedInst = Builder.CreateBinOp(InnerOpcode, A, V);
      }
    }

  // Does "(X op Y) op' Z" always equal "(X op' Z) op (Y op' Z)"?
  if (!SimplifiedInst && rightDistributesOverLeft(TopLevelOpcode, InnerOpcode))
    // Does the instruction have the form "(A op' B) op (C op' B)" or, in the
    // commutative case, "(A op' B) op (B op' D)"?
    if (B == D || (InnerCommutative && B == C)) {
      if (B != D)
        std::swap(C, D);
      // Consider forming "(A op C) op' B".
      // If "A op C" simplifies then it can be formed with no cost.
      V = SimplifyBinOp(TopLevelOpcode, A, C, SQ.getWithInstruction(&I));

      // If "A op C" doesn't simplify then only go on if both of the existing
      // operations "A op' B" and "C op' D" will be zapped as no longer used.
      if (!V && LHS->hasOneUse() && RHS->hasOneUse())
        V = Builder.CreateBinOp(TopLevelOpcode, A, C, LHS->getName());
      if (V) {
        SimplifiedInst = Builder.CreateBinOp(InnerOpcode, V, B);
      }
    }

  if (SimplifiedInst) {
    ++NumFactor;
    SimplifiedInst->takeName(&I);

    // Check if we can add NSW/NUW flags to SimplifiedInst. If so, set them.
    if (BinaryOperator *BO = dyn_cast<BinaryOperator>(SimplifiedInst)) {
      if (isa<OverflowingBinaryOperator>(SimplifiedInst)) {
        bool HasNSW = false;
        bool HasNUW = false;
        if (isa<OverflowingBinaryOperator>(&I)) {
          HasNSW = I.hasNoSignedWrap();
          HasNUW = I.hasNoUnsignedWrap();
        }

        if (auto *LOBO = dyn_cast<OverflowingBinaryOperator>(LHS)) {
          HasNSW &= LOBO->hasNoSignedWrap();
          HasNUW &= LOBO->hasNoUnsignedWrap();
        }

        if (auto *ROBO = dyn_cast<OverflowingBinaryOperator>(RHS)) {
          HasNSW &= ROBO->hasNoSignedWrap();
          HasNUW &= ROBO->hasNoUnsignedWrap();
        }

        if (TopLevelOpcode == Instruction::Add &&
            InnerOpcode == Instruction::Mul) {
          // We can propagate 'nsw' if we know that
          //  %Y = mul nsw i16 %X, C
          //  %Z = add nsw i16 %Y, %X
          // =>
          //  %Z = mul nsw i16 %X, C+1
          //
          // iff C+1 isn't INT_MIN
          const APInt *CInt;
          if (match(V, m_APInt(CInt))) {
            if (!CInt->isMinSignedValue())
              BO->setHasNoSignedWrap(HasNSW);
          }

          // nuw can be propagated with any constant or nuw value.
          BO->setHasNoUnsignedWrap(HasNUW);
        }
      }
    }
  }
  return SimplifiedInst;
}

/// This tries to simplify binary operations which some other binary operation
/// distributes over either by factorizing out common terms
/// (eg "(A*B)+(A*C)" -> "A*(B+C)") or expanding out if this results in
/// simplifications (eg: "A & (B | C) -> (A&B) | (A&C)" if this is a win).
/// Returns the simplified value, or null if it didn't simplify.
Value *InstCombinerImpl::SimplifyUsingDistributiveLaws(BinaryOperator &I) {
  Value *LHS = I.getOperand(0), *RHS = I.getOperand(1);
  BinaryOperator *Op0 = dyn_cast<BinaryOperator>(LHS);
  BinaryOperator *Op1 = dyn_cast<BinaryOperator>(RHS);
  Instruction::BinaryOps TopLevelOpcode = I.getOpcode();

  {
    // Factorization.
    Value *A, *B, *C, *D;
    Instruction::BinaryOps LHSOpcode, RHSOpcode;
    if (Op0)
      LHSOpcode = getBinOpsForFactorization(TopLevelOpcode, Op0, A, B);
    if (Op1)
      RHSOpcode = getBinOpsForFactorization(TopLevelOpcode, Op1, C, D);

    // The instruction has the form "(A op' B) op (C op' D)".  Try to factorize
    // a common term.
    if (Op0 && Op1 && LHSOpcode == RHSOpcode)
      if (Value *V = tryFactorization(I, LHSOpcode, A, B, C, D))
        return V;

    // The instruction has the form "(A op' B) op (C)".  Try to factorize common
    // term.
    if (Op0)
      if (Value *Ident = getIdentityValue(LHSOpcode, RHS))
        if (Value *V = tryFactorization(I, LHSOpcode, A, B, RHS, Ident))
          return V;

    // The instruction has the form "(B) op (C op' D)".  Try to factorize common
    // term.
    if (Op1)
      if (Value *Ident = getIdentityValue(RHSOpcode, LHS))
        if (Value *V = tryFactorization(I, RHSOpcode, LHS, Ident, C, D))
          return V;
  }

  // Expansion.
  if (Op0 && rightDistributesOverLeft(Op0->getOpcode(), TopLevelOpcode)) {
    // The instruction has the form "(A op' B) op C".  See if expanding it out
    // to "(A op C) op' (B op C)" results in simplifications.
    Value *A = Op0->getOperand(0), *B = Op0->getOperand(1), *C = RHS;
    Instruction::BinaryOps InnerOpcode = Op0->getOpcode(); // op'

    // Disable the use of undef because it's not safe to distribute undef.
    auto SQDistributive = SQ.getWithInstruction(&I).getWithoutUndef();
    Value *L = SimplifyBinOp(TopLevelOpcode, A, C, SQDistributive);
    Value *R = SimplifyBinOp(TopLevelOpcode, B, C, SQDistributive);

    // Do "A op C" and "B op C" both simplify?
    if (L && R) {
      // They do! Return "L op' R".
      ++NumExpand;
      C = Builder.CreateBinOp(InnerOpcode, L, R);
      C->takeName(&I);
      return C;
    }

    // Does "A op C" simplify to the identity value for the inner opcode?
    if (L && L == ConstantExpr::getBinOpIdentity(InnerOpcode, L->getType())) {
      // They do! Return "B op C".
      ++NumExpand;
      C = Builder.CreateBinOp(TopLevelOpcode, B, C);
      C->takeName(&I);
      return C;
    }

    // Does "B op C" simplify to the identity value for the inner opcode?
    if (R && R == ConstantExpr::getBinOpIdentity(InnerOpcode, R->getType())) {
      // They do! Return "A op C".
      ++NumExpand;
      C = Builder.CreateBinOp(TopLevelOpcode, A, C);
      C->takeName(&I);
      return C;
    }
  }

  if (Op1 && leftDistributesOverRight(TopLevelOpcode, Op1->getOpcode())) {
    // The instruction has the form "A op (B op' C)".  See if expanding it out
    // to "(A op B) op' (A op C)" results in simplifications.
    Value *A = LHS, *B = Op1->getOperand(0), *C = Op1->getOperand(1);
    Instruction::BinaryOps InnerOpcode = Op1->getOpcode(); // op'

    // Disable the use of undef because it's not safe to distribute undef.
    auto SQDistributive = SQ.getWithInstruction(&I).getWithoutUndef();
    Value *L = SimplifyBinOp(TopLevelOpcode, A, B, SQDistributive);
    Value *R = SimplifyBinOp(TopLevelOpcode, A, C, SQDistributive);

    // Do "A op B" and "A op C" both simplify?
    if (L && R) {
      // They do! Return "L op' R".
      ++NumExpand;
      A = Builder.CreateBinOp(InnerOpcode, L, R);
      A->takeName(&I);
      return A;
    }

    // Does "A op B" simplify to the identity value for the inner opcode?
    if (L && L == ConstantExpr::getBinOpIdentity(InnerOpcode, L->getType())) {
      // They do! Return "A op C".
      ++NumExpand;
      A = Builder.CreateBinOp(TopLevelOpcode, A, C);
      A->takeName(&I);
      return A;
    }

    // Does "A op C" simplify to the identity value for the inner opcode?
    if (R && R == ConstantExpr::getBinOpIdentity(InnerOpcode, R->getType())) {
      // They do! Return "A op B".
      ++NumExpand;
      A = Builder.CreateBinOp(TopLevelOpcode, A, B);
      A->takeName(&I);
      return A;
    }
  }

  return SimplifySelectsFeedingBinaryOp(I, LHS, RHS);
}

Value *InstCombinerImpl::SimplifySelectsFeedingBinaryOp(BinaryOperator &I,
                                                        Value *LHS,
                                                        Value *RHS) {
  Value *A, *B, *C, *D, *E, *F;
  bool LHSIsSelect = match(LHS, m_Select(m_Value(A), m_Value(B), m_Value(C)));
  bool RHSIsSelect = match(RHS, m_Select(m_Value(D), m_Value(E), m_Value(F)));
  if (!LHSIsSelect && !RHSIsSelect)
    return nullptr;

  FastMathFlags FMF;
  BuilderTy::FastMathFlagGuard Guard(Builder);
  if (isa<FPMathOperator>(&I)) {
    FMF = I.getFastMathFlags();
    Builder.setFastMathFlags(FMF);
  }

  Instruction::BinaryOps Opcode = I.getOpcode();
  SimplifyQuery Q = SQ.getWithInstruction(&I);

  Value *Cond, *True = nullptr, *False = nullptr;
  if (LHSIsSelect && RHSIsSelect && A == D) {
    // (A ? B : C) op (A ? E : F) -> A ? (B op E) : (C op F)
    Cond = A;
    True = SimplifyBinOp(Opcode, B, E, FMF, Q);
    False = SimplifyBinOp(Opcode, C, F, FMF, Q);

    if (LHS->hasOneUse() && RHS->hasOneUse()) {
      if (False && !True)
        True = Builder.CreateBinOp(Opcode, B, E);
      else if (True && !False)
        False = Builder.CreateBinOp(Opcode, C, F);
    }
  } else if (LHSIsSelect && LHS->hasOneUse()) {
    // (A ? B : C) op Y -> A ? (B op Y) : (C op Y)
    Cond = A;
    True = SimplifyBinOp(Opcode, B, RHS, FMF, Q);
    False = SimplifyBinOp(Opcode, C, RHS, FMF, Q);
  } else if (RHSIsSelect && RHS->hasOneUse()) {
    // X op (D ? E : F) -> D ? (X op E) : (X op F)
    Cond = D;
    True = SimplifyBinOp(Opcode, LHS, E, FMF, Q);
    False = SimplifyBinOp(Opcode, LHS, F, FMF, Q);
  }

  if (!True || !False)
    return nullptr;

  Value *SI = Builder.CreateSelect(Cond, True, False);
  SI->takeName(&I);
  return SI;
}

/// Freely adapt every user of V as-if V was changed to !V.
/// WARNING: only if canFreelyInvertAllUsersOf() said this can be done.
void InstCombinerImpl::freelyInvertAllUsersOf(Value *I) {
  for (User *U : I->users()) {
    switch (cast<Instruction>(U)->getOpcode()) {
    case Instruction::Select: {
      auto *SI = cast<SelectInst>(U);
      SI->swapValues();
      SI->swapProfMetadata();
      break;
    }
    case Instruction::Br:
      cast<BranchInst>(U)->swapSuccessors(); // swaps prof metadata too
      break;
    case Instruction::Xor:
      replaceInstUsesWith(cast<Instruction>(*U), I);
      break;
    default:
      llvm_unreachable("Got unexpected user - out of sync with "
                       "canFreelyInvertAllUsersOf() ?");
    }
  }
}

/// Given a 'sub' instruction, return the RHS of the instruction if the LHS is a
/// constant zero (which is the 'negate' form).
Value *InstCombinerImpl::dyn_castNegVal(Value *V) const {
  Value *NegV;
  if (match(V, m_Neg(m_Value(NegV))))
    return NegV;

  // Constants can be considered to be negated values if they can be folded.
  if (ConstantInt *C = dyn_cast<ConstantInt>(V))
    return ConstantExpr::getNeg(C);

  if (ConstantDataVector *C = dyn_cast<ConstantDataVector>(V))
    if (C->getType()->getElementType()->isIntegerTy())
      return ConstantExpr::getNeg(C);

  if (ConstantVector *CV = dyn_cast<ConstantVector>(V)) {
    for (unsigned i = 0, e = CV->getNumOperands(); i != e; ++i) {
      Constant *Elt = CV->getAggregateElement(i);
      if (!Elt)
        return nullptr;

      if (isa<UndefValue>(Elt))
        continue;

      if (!isa<ConstantInt>(Elt))
        return nullptr;
    }
    return ConstantExpr::getNeg(CV);
  }

  // Negate integer vector splats.
  if (auto *CV = dyn_cast<Constant>(V))
    if (CV->getType()->isVectorTy() &&
        CV->getType()->getScalarType()->isIntegerTy() && CV->getSplatValue())
      return ConstantExpr::getNeg(CV);

  return nullptr;
}

/// A binop with a constant operand and a sign-extended boolean operand may be
/// converted into a select of constants by applying the binary operation to
/// the constant with the two possible values of the extended boolean (0 or -1).
Instruction *InstCombinerImpl::foldBinopOfSextBoolToSelect(BinaryOperator &BO) {
  // TODO: Handle non-commutative binop (constant is operand 0).
  // TODO: Handle zext.
  // TODO: Peek through 'not' of cast.
  Value *BO0 = BO.getOperand(0);
  Value *BO1 = BO.getOperand(1);
  Value *X;
  Constant *C;
  if (!match(BO0, m_SExt(m_Value(X))) || !match(BO1, m_ImmConstant(C)) ||
      !X->getType()->isIntOrIntVectorTy(1))
    return nullptr;

  // bo (sext i1 X), C --> select X, (bo -1, C), (bo 0, C)
  Constant *Ones = ConstantInt::getAllOnesValue(BO.getType());
  Constant *Zero = ConstantInt::getNullValue(BO.getType());
  Constant *TVal = ConstantExpr::get(BO.getOpcode(), Ones, C);
  Constant *FVal = ConstantExpr::get(BO.getOpcode(), Zero, C);
  return SelectInst::Create(X, TVal, FVal);
}

static Value *foldOperationIntoSelectOperand(Instruction &I, Value *SO,
                                             InstCombiner::BuilderTy &Builder) {
  if (auto *Cast = dyn_cast<CastInst>(&I))
    return Builder.CreateCast(Cast->getOpcode(), SO, I.getType());

  if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
    assert(canConstantFoldCallTo(II, cast<Function>(II->getCalledOperand())) &&
           "Expected constant-foldable intrinsic");
    Intrinsic::ID IID = II->getIntrinsicID();
    if (II->arg_size() == 1)
      return Builder.CreateUnaryIntrinsic(IID, SO);

    // This works for real binary ops like min/max (where we always expect the
    // constant operand to be canonicalized as op1) and unary ops with a bonus
    // constant argument like ctlz/cttz.
    // TODO: Handle non-commutative binary intrinsics as below for binops.
    assert(II->arg_size() == 2 && "Expected binary intrinsic");
    assert(isa<Constant>(II->getArgOperand(1)) && "Expected constant operand");
    return Builder.CreateBinaryIntrinsic(IID, SO, II->getArgOperand(1));
  }

  assert(I.isBinaryOp() && "Unexpected opcode for select folding");

  // Figure out if the constant is the left or the right argument.
  bool ConstIsRHS = isa<Constant>(I.getOperand(1));
  Constant *ConstOperand = cast<Constant>(I.getOperand(ConstIsRHS));

  if (auto *SOC = dyn_cast<Constant>(SO)) {
    if (ConstIsRHS)
      return ConstantExpr::get(I.getOpcode(), SOC, ConstOperand);
    return ConstantExpr::get(I.getOpcode(), ConstOperand, SOC);
  }

  Value *Op0 = SO, *Op1 = ConstOperand;
  if (!ConstIsRHS)
    std::swap(Op0, Op1);

  Value *NewBO = Builder.CreateBinOp(cast<BinaryOperator>(&I)->getOpcode(), Op0,
                                     Op1, SO->getName() + ".op");
  if (auto *NewBOI = dyn_cast<Instruction>(NewBO))
    NewBOI->copyIRFlags(&I);
  return NewBO;
}

Instruction *InstCombinerImpl::FoldOpIntoSelect(Instruction &Op,
                                                SelectInst *SI) {
  // Don't modify shared select instructions.
  if (!SI->hasOneUse())
    return nullptr;

  Value *TV = SI->getTrueValue();
  Value *FV = SI->getFalseValue();
  if (!(isa<Constant>(TV) || isa<Constant>(FV)))
    return nullptr;

  // Bool selects with constant operands can be folded to logical ops.
  if (SI->getType()->isIntOrIntVectorTy(1))
    return nullptr;

  // If it's a bitcast involving vectors, make sure it has the same number of
  // elements on both sides.
  if (auto *BC = dyn_cast<BitCastInst>(&Op)) {
    VectorType *DestTy = dyn_cast<VectorType>(BC->getDestTy());
    VectorType *SrcTy = dyn_cast<VectorType>(BC->getSrcTy());

    // Verify that either both or neither are vectors.
    if ((SrcTy == nullptr) != (DestTy == nullptr))
      return nullptr;

    // If vectors, verify that they have the same number of elements.
    if (SrcTy && SrcTy->getElementCount() != DestTy->getElementCount())
      return nullptr;
  }

  // Test if a CmpInst instruction is used exclusively by a select as
  // part of a minimum or maximum operation. If so, refrain from doing
  // any other folding. This helps out other analyses which understand
  // non-obfuscated minimum and maximum idioms, such as ScalarEvolution
  // and CodeGen. And in this case, at least one of the comparison
  // operands has at least one user besides the compare (the select),
  // which would often largely negate the benefit of folding anyway.
  if (auto *CI = dyn_cast<CmpInst>(SI->getCondition())) {
    if (CI->hasOneUse()) {
      Value *Op0 = CI->getOperand(0), *Op1 = CI->getOperand(1);

      // FIXME: This is a hack to avoid infinite looping with min/max patterns.
      //        We have to ensure that vector constants that only differ with
      //        undef elements are treated as equivalent.
      auto areLooselyEqual = [](Value *A, Value *B) {
        if (A == B)
          return true;

        // Test for vector constants.
        Constant *ConstA, *ConstB;
        if (!match(A, m_Constant(ConstA)) || !match(B, m_Constant(ConstB)))
          return false;

        // TODO: Deal with FP constants?
        if (!A->getType()->isIntOrIntVectorTy() || A->getType() != B->getType())
          return false;

        // Compare for equality including undefs as equal.
        auto *Cmp = ConstantExpr::getCompare(ICmpInst::ICMP_EQ, ConstA, ConstB);
        const APInt *C;
        return match(Cmp, m_APIntAllowUndef(C)) && C->isOne();
      };

      if ((areLooselyEqual(TV, Op0) && areLooselyEqual(FV, Op1)) ||
          (areLooselyEqual(FV, Op0) && areLooselyEqual(TV, Op1)))
        return nullptr;
    }
  }

  Value *NewTV = foldOperationIntoSelectOperand(Op, TV, Builder);
  Value *NewFV = foldOperationIntoSelectOperand(Op, FV, Builder);
  return SelectInst::Create(SI->getCondition(), NewTV, NewFV, "", nullptr, SI);
}

static Value *foldOperationIntoPhiValue(BinaryOperator *I, Value *InV,
                                        InstCombiner::BuilderTy &Builder) {
  bool ConstIsRHS = isa<Constant>(I->getOperand(1));
  Constant *C = cast<Constant>(I->getOperand(ConstIsRHS));

  if (auto *InC = dyn_cast<Constant>(InV)) {
    if (ConstIsRHS)
      return ConstantExpr::get(I->getOpcode(), InC, C);
    return ConstantExpr::get(I->getOpcode(), C, InC);
  }

  Value *Op0 = InV, *Op1 = C;
  if (!ConstIsRHS)
    std::swap(Op0, Op1);

  Value *RI = Builder.CreateBinOp(I->getOpcode(), Op0, Op1, "phi.bo");
  auto *FPInst = dyn_cast<Instruction>(RI);
  if (FPInst && isa<FPMathOperator>(FPInst))
    FPInst->copyFastMathFlags(I);
  return RI;
}

Instruction *InstCombinerImpl::foldOpIntoPhi(Instruction &I, PHINode *PN) {
  unsigned NumPHIValues = PN->getNumIncomingValues();
  if (NumPHIValues == 0)
    return nullptr;

  // We normally only transform phis with a single use.  However, if a PHI has
  // multiple uses and they are all the same operation, we can fold *all* of the
  // uses into the PHI.
  if (!PN->hasOneUse()) {
    // Walk the use list for the instruction, comparing them to I.
    for (User *U : PN->users()) {
      Instruction *UI = cast<Instruction>(U);
      if (UI != &I && !I.isIdenticalTo(UI))
        return nullptr;
    }
    // Otherwise, we can replace *all* users with the new PHI we form.
  }

  // Check to see if all of the operands of the PHI are simple constants
  // (constantint/constantfp/undef).  If there is one non-constant value,
  // remember the BB it is in.  If there is more than one or if *it* is a PHI,
  // bail out.  We don't do arbitrary constant expressions here because moving
  // their computation can be expensive without a cost model.
  BasicBlock *NonConstBB = nullptr;
  for (unsigned i = 0; i != NumPHIValues; ++i) {
    Value *InVal = PN->getIncomingValue(i);
    // For non-freeze, require constant operand
    // For freeze, require non-undef, non-poison operand
    if (!isa<FreezeInst>(I) && match(InVal, m_ImmConstant()))
      continue;
    if (isa<FreezeInst>(I) && isGuaranteedNotToBeUndefOrPoison(InVal))
      continue;

    if (isa<PHINode>(InVal)) return nullptr;  // Itself a phi.
    if (NonConstBB) return nullptr;  // More than one non-const value.

    NonConstBB = PN->getIncomingBlock(i);

    // If the InVal is an invoke at the end of the pred block, then we can't
    // insert a computation after it without breaking the edge.
    if (isa<InvokeInst>(InVal))
      if (cast<Instruction>(InVal)->getParent() == NonConstBB)
        return nullptr;

    // If the incoming non-constant value is in I's block, we will remove one
    // instruction, but insert another equivalent one, leading to infinite
    // instcombine.
    if (isPotentiallyReachable(I.getParent(), NonConstBB, nullptr, &DT, LI))
      return nullptr;
  }

  // If there is exactly one non-constant value, we can insert a copy of the
  // operation in that block.  However, if this is a critical edge, we would be
  // inserting the computation on some other paths (e.g. inside a loop).  Only
  // do this if the pred block is unconditionally branching into the phi block.
  // Also, make sure that the pred block is not dead code.
  if (NonConstBB != nullptr) {
    BranchInst *BI = dyn_cast<BranchInst>(NonConstBB->getTerminator());
    if (!BI || !BI->isUnconditional() || !DT.isReachableFromEntry(NonConstBB))
      return nullptr;
  }

  // Okay, we can do the transformation: create the new PHI node.
  PHINode *NewPN = PHINode::Create(I.getType(), PN->getNumIncomingValues());
  InsertNewInstBefore(NewPN, *PN);
  NewPN->takeName(PN);

  // If we are going to have to insert a new computation, do so right before the
  // predecessor's terminator.
  if (NonConstBB)
    Builder.SetInsertPoint(NonConstBB->getTerminator());

  // Next, add all of the operands to the PHI.
  if (SelectInst *SI = dyn_cast<SelectInst>(&I)) {
    // We only currently try to fold the condition of a select when it is a phi,
    // not the true/false values.
    Value *TrueV = SI->getTrueValue();
    Value *FalseV = SI->getFalseValue();
    BasicBlock *PhiTransBB = PN->getParent();
    for (unsigned i = 0; i != NumPHIValues; ++i) {
      BasicBlock *ThisBB = PN->getIncomingBlock(i);
      Value *TrueVInPred = TrueV->DoPHITranslation(PhiTransBB, ThisBB);
      Value *FalseVInPred = FalseV->DoPHITranslation(PhiTransBB, ThisBB);
      Value *InV = nullptr;
      // Beware of ConstantExpr:  it may eventually evaluate to getNullValue,
      // even if currently isNullValue gives false.
      Constant *InC = dyn_cast<Constant>(PN->getIncomingValue(i));
      // For vector constants, we cannot use isNullValue to fold into
      // FalseVInPred versus TrueVInPred. When we have individual nonzero
      // elements in the vector, we will incorrectly fold InC to
      // `TrueVInPred`.
      if (InC && isa<ConstantInt>(InC))
        InV = InC->isNullValue() ? FalseVInPred : TrueVInPred;
      else {
        // Generate the select in the same block as PN's current incoming block.
        // Note: ThisBB need not be the NonConstBB because vector constants
        // which are constants by definition are handled here.
        // FIXME: This can lead to an increase in IR generation because we might
        // generate selects for vector constant phi operand, that could not be
        // folded to TrueVInPred or FalseVInPred as done for ConstantInt. For
        // non-vector phis, this transformation was always profitable because
        // the select would be generated exactly once in the NonConstBB.
        Builder.SetInsertPoint(ThisBB->getTerminator());
        InV = Builder.CreateSelect(PN->getIncomingValue(i), TrueVInPred,
                                   FalseVInPred, "phi.sel");
      }
      NewPN->addIncoming(InV, ThisBB);
    }
  } else if (CmpInst *CI = dyn_cast<CmpInst>(&I)) {
    Constant *C = cast<Constant>(I.getOperand(1));
    for (unsigned i = 0; i != NumPHIValues; ++i) {
      Value *InV = nullptr;
      if (auto *InC = dyn_cast<Constant>(PN->getIncomingValue(i)))
        InV = ConstantExpr::getCompare(CI->getPredicate(), InC, C);
      else
        InV = Builder.CreateCmp(CI->getPredicate(), PN->getIncomingValue(i),
                                C, "phi.cmp");
      NewPN->addIncoming(InV, PN->getIncomingBlock(i));
    }
  } else if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
    for (unsigned i = 0; i != NumPHIValues; ++i) {
      Value *InV = foldOperationIntoPhiValue(BO, PN->getIncomingValue(i),
                                             Builder);
      NewPN->addIncoming(InV, PN->getIncomingBlock(i));
    }
  } else if (isa<FreezeInst>(&I)) {
    for (unsigned i = 0; i != NumPHIValues; ++i) {
      Value *InV;
      if (NonConstBB == PN->getIncomingBlock(i))
        InV = Builder.CreateFreeze(PN->getIncomingValue(i), "phi.fr");
      else
        InV = PN->getIncomingValue(i);
      NewPN->addIncoming(InV, PN->getIncomingBlock(i));
    }
  } else {
    CastInst *CI = cast<CastInst>(&I);
    Type *RetTy = CI->getType();
    for (unsigned i = 0; i != NumPHIValues; ++i) {
      Value *InV;
      if (Constant *InC = dyn_cast<Constant>(PN->getIncomingValue(i)))
        InV = ConstantExpr::getCast(CI->getOpcode(), InC, RetTy);
      else
        InV = Builder.CreateCast(CI->getOpcode(), PN->getIncomingValue(i),
                                 I.getType(), "phi.cast");
      NewPN->addIncoming(InV, PN->getIncomingBlock(i));
    }
  }

  for (User *U : make_early_inc_range(PN->users())) {
    Instruction *User = cast<Instruction>(U);
    if (User == &I) continue;
    replaceInstUsesWith(*User, NewPN);
    eraseInstFromFunction(*User);
  }
  return replaceInstUsesWith(I, NewPN);
}

Instruction *InstCombinerImpl::foldBinopWithPhiOperands(BinaryOperator &BO) {
  // TODO: This should be similar to the incoming values check in foldOpIntoPhi:
  //       we are guarding against replicating the binop in >1 predecessor.
  //       This could miss matching a phi with 2 constant incoming values.
  auto *Phi0 = dyn_cast<PHINode>(BO.getOperand(0));
  auto *Phi1 = dyn_cast<PHINode>(BO.getOperand(1));
  if (!Phi0 || !Phi1 || !Phi0->hasOneUse() || !Phi1->hasOneUse() ||
      Phi0->getNumOperands() != 2 || Phi1->getNumOperands() != 2)
    return nullptr;

  // TODO: Remove the restriction for binop being in the same block as the phis.
  if (BO.getParent() != Phi0->getParent() ||
      BO.getParent() != Phi1->getParent())
    return nullptr;

  // Match a pair of incoming constants for one of the predecessor blocks.
  BasicBlock *ConstBB, *OtherBB;
  Constant *C0, *C1;
  if (match(Phi0->getIncomingValue(0), m_ImmConstant(C0))) {
    ConstBB = Phi0->getIncomingBlock(0);
    OtherBB = Phi0->getIncomingBlock(1);
  } else if (match(Phi0->getIncomingValue(1), m_ImmConstant(C0))) {
    ConstBB = Phi0->getIncomingBlock(1);
    OtherBB = Phi0->getIncomingBlock(0);
  } else {
    return nullptr;
  }
  if (!match(Phi1->getIncomingValueForBlock(ConstBB), m_ImmConstant(C1)))
    return nullptr;

  // The block that we are hoisting to must reach here unconditionally.
  // Otherwise, we could be speculatively executing an expensive or
  // non-speculative op.
  auto *PredBlockBranch = dyn_cast<BranchInst>(OtherBB->getTerminator());
  if (!PredBlockBranch || PredBlockBranch->isConditional() ||
      !DT.isReachableFromEntry(OtherBB))
    return nullptr;

  // TODO: This check could be tightened to only apply to binops (div/rem) that
  //       are not safe to speculatively execute. But that could allow hoisting
  //       potentially expensive instructions (fdiv for example).
  for (auto BBIter = BO.getParent()->begin(); &*BBIter != &BO; ++BBIter)
    if (!isGuaranteedToTransferExecutionToSuccessor(&*BBIter))
      return nullptr;

  // Make a new binop in the predecessor block with the non-constant incoming
  // values.
  Builder.SetInsertPoint(PredBlockBranch);
  Value *NewBO = Builder.CreateBinOp(BO.getOpcode(),
                                     Phi0->getIncomingValueForBlock(OtherBB),
                                     Phi1->getIncomingValueForBlock(OtherBB));
  if (auto *NotFoldedNewBO = dyn_cast<BinaryOperator>(NewBO))
    NotFoldedNewBO->copyIRFlags(&BO);

  // Fold constants for the predecessor block with constant incoming values.
  Constant *NewC = ConstantExpr::get(BO.getOpcode(), C0, C1);

  // Replace the binop with a phi of the new values. The old phis are dead.
  PHINode *NewPhi = PHINode::Create(BO.getType(), 2);
  NewPhi->addIncoming(NewBO, OtherBB);
  NewPhi->addIncoming(NewC, ConstBB);
  return NewPhi;
}

Instruction *InstCombinerImpl::foldBinOpIntoSelectOrPhi(BinaryOperator &I) {
  if (!isa<Constant>(I.getOperand(1)))
    return nullptr;

  if (auto *Sel = dyn_cast<SelectInst>(I.getOperand(0))) {
    if (Instruction *NewSel = FoldOpIntoSelect(I, Sel))
      return NewSel;
  } else if (auto *PN = dyn_cast<PHINode>(I.getOperand(0))) {
    if (Instruction *NewPhi = foldOpIntoPhi(I, PN))
      return NewPhi;
  }
  return nullptr;
}

/// Given a pointer type and a constant offset, determine whether or not there
/// is a sequence of GEP indices into the pointed type that will land us at the
/// specified offset. If so, fill them into NewIndices and return the resultant
/// element type, otherwise return null.
static Type *findElementAtOffset(PointerType *PtrTy, int64_t IntOffset,
                                 SmallVectorImpl<Value *> &NewIndices,
                                 const DataLayout &DL) {
  // Only used by visitGEPOfBitcast(), which is skipped for opaque pointers.
  Type *Ty = PtrTy->getNonOpaquePointerElementType();
  if (!Ty->isSized())
    return nullptr;

  APInt Offset(DL.getIndexTypeSizeInBits(PtrTy), IntOffset);
  SmallVector<APInt> Indices = DL.getGEPIndicesForOffset(Ty, Offset);
  if (!Offset.isZero())
    return nullptr;

  for (const APInt &Index : Indices)
    NewIndices.push_back(ConstantInt::get(PtrTy->getContext(), Index));
  return Ty;
}

static bool shouldMergeGEPs(GEPOperator &GEP, GEPOperator &Src) {
  // If this GEP has only 0 indices, it is the same pointer as
  // Src. If Src is not a trivial GEP too, don't combine
  // the indices.
  if (GEP.hasAllZeroIndices() && !Src.hasAllZeroIndices() &&
      !Src.hasOneUse())
    return false;
  return true;
}

/// Return a value X such that Val = X * Scale, or null if none.
/// If the multiplication is known not to overflow, then NoSignedWrap is set.
Value *InstCombinerImpl::Descale(Value *Val, APInt Scale, bool &NoSignedWrap) {
  assert(isa<IntegerType>(Val->getType()) && "Can only descale integers!");
  assert(cast<IntegerType>(Val->getType())->getBitWidth() ==
         Scale.getBitWidth() && "Scale not compatible with value!");

  // If Val is zero or Scale is one then Val = Val * Scale.
  if (match(Val, m_Zero()) || Scale == 1) {
    NoSignedWrap = true;
    return Val;
  }

  // If Scale is zero then it does not divide Val.
  if (Scale.isMinValue())
    return nullptr;

  // Look through chains of multiplications, searching for a constant that is
  // divisible by Scale.  For example, descaling X*(Y*(Z*4)) by a factor of 4
  // will find the constant factor 4 and produce X*(Y*Z).  Descaling X*(Y*8) by
  // a factor of 4 will produce X*(Y*2).  The principle of operation is to bore
  // down from Val:
  //
  //     Val = M1 * X          ||   Analysis starts here and works down
  //      M1 = M2 * Y          ||   Doesn't descend into terms with more
  //      M2 =  Z * 4          \/   than one use
  //
  // Then to modify a term at the bottom:
  //
  //     Val = M1 * X
  //      M1 =  Z * Y          ||   Replaced M2 with Z
  //
  // Then to work back up correcting nsw flags.

  // Op - the term we are currently analyzing.  Starts at Val then drills down.
  // Replaced with its descaled value before exiting from the drill down loop.
  Value *Op = Val;

  // Parent - initially null, but after drilling down notes where Op came from.
  // In the example above, Parent is (Val, 0) when Op is M1, because M1 is the
  // 0'th operand of Val.
  std::pair<Instruction *, unsigned> Parent;

  // Set if the transform requires a descaling at deeper levels that doesn't
  // overflow.
  bool RequireNoSignedWrap = false;

  // Log base 2 of the scale. Negative if not a power of 2.
  int32_t logScale = Scale.exactLogBase2();

  for (;; Op = Parent.first->getOperand(Parent.second)) { // Drill down
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Op)) {
      // If Op is a constant divisible by Scale then descale to the quotient.
      APInt Quotient(Scale), Remainder(Scale); // Init ensures right bitwidth.
      APInt::sdivrem(CI->getValue(), Scale, Quotient, Remainder);
      if (!Remainder.isMinValue())
        // Not divisible by Scale.
        return nullptr;
      // Replace with the quotient in the parent.
      Op = ConstantInt::get(CI->getType(), Quotient);
      NoSignedWrap = true;
      break;
    }

    if (BinaryOperator *BO = dyn_cast<BinaryOperator>(Op)) {
      if (BO->getOpcode() == Instruction::Mul) {
        // Multiplication.
        NoSignedWrap = BO->hasNoSignedWrap();
        if (RequireNoSignedWrap && !NoSignedWrap)
          return nullptr;

        // There are three cases for multiplication: multiplication by exactly
        // the scale, multiplication by a constant different to the scale, and
        // multiplication by something else.
        Value *LHS = BO->getOperand(0);
        Value *RHS = BO->getOperand(1);

        if (ConstantInt *CI = dyn_cast<ConstantInt>(RHS)) {
          // Multiplication by a constant.
          if (CI->getValue() == Scale) {
            // Multiplication by exactly the scale, replace the multiplication
            // by its left-hand side in the parent.
            Op = LHS;
            break;
          }

          // Otherwise drill down into the constant.
          if (!Op->hasOneUse())
            return nullptr;

          Parent = std::make_pair(BO, 1);
          continue;
        }

        // Multiplication by something else. Drill down into the left-hand side
        // since that's where the reassociate pass puts the good stuff.
        if (!Op->hasOneUse())
          return nullptr;

        Parent = std::make_pair(BO, 0);
        continue;
      }

      if (logScale > 0 && BO->getOpcode() == Instruction::Shl &&
          isa<ConstantInt>(BO->getOperand(1))) {
        // Multiplication by a power of 2.
        NoSignedWrap = BO->hasNoSignedWrap();
        if (RequireNoSignedWrap && !NoSignedWrap)
          return nullptr;

        Value *LHS = BO->getOperand(0);
        int32_t Amt = cast<ConstantInt>(BO->getOperand(1))->
          getLimitedValue(Scale.getBitWidth());
        // Op = LHS << Amt.

        if (Amt == logScale) {
          // Multiplication by exactly the scale, replace the multiplication
          // by its left-hand side in the parent.
          Op = LHS;
          break;
        }
        if (Amt < logScale || !Op->hasOneUse())
          return nullptr;

        // Multiplication by more than the scale.  Reduce the multiplying amount
        // by the scale in the parent.
        Parent = std::make_pair(BO, 1);
        Op = ConstantInt::get(BO->getType(), Amt - logScale);
        break;
      }
    }

    if (!Op->hasOneUse())
      return nullptr;

    if (CastInst *Cast = dyn_cast<CastInst>(Op)) {
      if (Cast->getOpcode() == Instruction::SExt) {
        // Op is sign-extended from a smaller type, descale in the smaller type.
        unsigned SmallSize = Cast->getSrcTy()->getPrimitiveSizeInBits();
        APInt SmallScale = Scale.trunc(SmallSize);
        // Suppose Op = sext X, and we descale X as Y * SmallScale.  We want to
        // descale Op as (sext Y) * Scale.  In order to have
        //   sext (Y * SmallScale) = (sext Y) * Scale
        // some conditions need to hold however: SmallScale must sign-extend to
        // Scale and the multiplication Y * SmallScale should not overflow.
        if (SmallScale.sext(Scale.getBitWidth()) != Scale)
          // SmallScale does not sign-extend to Scale.
          return nullptr;
        assert(SmallScale.exactLogBase2() == logScale);
        // Require that Y * SmallScale must not overflow.
        RequireNoSignedWrap = true;

        // Drill down through the cast.
        Parent = std::make_pair(Cast, 0);
        Scale = SmallScale;
        continue;
      }

      if (Cast->getOpcode() == Instruction::Trunc) {
        // Op is truncated from a larger type, descale in the larger type.
        // Suppose Op = trunc X, and we descale X as Y * sext Scale.  Then
        //   trunc (Y * sext Scale) = (trunc Y) * Scale
        // always holds.  However (trunc Y) * Scale may overflow even if
        // trunc (Y * sext Scale) does not, so nsw flags need to be cleared
        // from this point up in the expression (see later).
        if (RequireNoSignedWrap)
          return nullptr;

        // Drill down through the cast.
        unsigned LargeSize = Cast->getSrcTy()->getPrimitiveSizeInBits();
        Parent = std::make_pair(Cast, 0);
        Scale = Scale.sext(LargeSize);
        if (logScale + 1 == (int32_t)Cast->getType()->getPrimitiveSizeInBits())
          logScale = -1;
        assert(Scale.exactLogBase2() == logScale);
        continue;
      }
    }

    // Unsupported expression, bail out.
    return nullptr;
  }

  // If Op is zero then Val = Op * Scale.
  if (match(Op, m_Zero())) {
    NoSignedWrap = true;
    return Op;
  }

  // We know that we can successfully descale, so from here on we can safely
  // modify the IR.  Op holds the descaled version of the deepest term in the
  // expression.  NoSignedWrap is 'true' if multiplying Op by Scale is known
  // not to overflow.

  if (!Parent.first)
    // The expression only had one term.
    return Op;

  // Rewrite the parent using the descaled version of its operand.
  assert(Parent.first->hasOneUse() && "Drilled down when more than one use!");
  assert(Op != Parent.first->getOperand(Parent.second) &&
         "Descaling was a no-op?");
  replaceOperand(*Parent.first, Parent.second, Op);
  Worklist.push(Parent.first);

  // Now work back up the expression correcting nsw flags.  The logic is based
  // on the following observation: if X * Y is known not to overflow as a signed
  // multiplication, and Y is replaced by a value Z with smaller absolute value,
  // then X * Z will not overflow as a signed multiplication either.  As we work
  // our way up, having NoSignedWrap 'true' means that the descaled value at the
  // current level has strictly smaller absolute value than the original.
  Instruction *Ancestor = Parent.first;
  do {
    if (BinaryOperator *BO = dyn_cast<BinaryOperator>(Ancestor)) {
      // If the multiplication wasn't nsw then we can't say anything about the
      // value of the descaled multiplication, and we have to clear nsw flags
      // from this point on up.
      bool OpNoSignedWrap = BO->hasNoSignedWrap();
      NoSignedWrap &= OpNoSignedWrap;
      if (NoSignedWrap != OpNoSignedWrap) {
        BO->setHasNoSignedWrap(NoSignedWrap);
        Worklist.push(Ancestor);
      }
    } else if (Ancestor->getOpcode() == Instruction::Trunc) {
      // The fact that the descaled input to the trunc has smaller absolute
      // value than the original input doesn't tell us anything useful about
      // the absolute values of the truncations.
      NoSignedWrap = false;
    }
    assert((Ancestor->getOpcode() != Instruction::SExt || NoSignedWrap) &&
           "Failed to keep proper track of nsw flags while drilling down?");

    if (Ancestor == Val)
      // Got to the top, all done!
      return Val;

    // Move up one level in the expression.
    assert(Ancestor->hasOneUse() && "Drilled down when more than one use!");
    Ancestor = Ancestor->user_back();
  } while (true);
}

Instruction *InstCombinerImpl::foldVectorBinop(BinaryOperator &Inst) {
  if (!isa<VectorType>(Inst.getType()))
    return nullptr;

  BinaryOperator::BinaryOps Opcode = Inst.getOpcode();
  Value *LHS = Inst.getOperand(0), *RHS = Inst.getOperand(1);
  assert(cast<VectorType>(LHS->getType())->getElementCount() ==
         cast<VectorType>(Inst.getType())->getElementCount());
  assert(cast<VectorType>(RHS->getType())->getElementCount() ==
         cast<VectorType>(Inst.getType())->getElementCount());

  // If both operands of the binop are vector concatenations, then perform the
  // narrow binop on each pair of the source operands followed by concatenation
  // of the results.
  Value *L0, *L1, *R0, *R1;
  ArrayRef<int> Mask;
  if (match(LHS, m_Shuffle(m_Value(L0), m_Value(L1), m_Mask(Mask))) &&
      match(RHS, m_Shuffle(m_Value(R0), m_Value(R1), m_SpecificMask(Mask))) &&
      LHS->hasOneUse() && RHS->hasOneUse() &&
      cast<ShuffleVectorInst>(LHS)->isConcat() &&
      cast<ShuffleVectorInst>(RHS)->isConcat()) {
    // This transform does not have the speculative execution constraint as
    // below because the shuffle is a concatenation. The new binops are
    // operating on exactly the same elements as the existing binop.
    // TODO: We could ease the mask requirement to allow different undef lanes,
    //       but that requires an analysis of the binop-with-undef output value.
    Value *NewBO0 = Builder.CreateBinOp(Opcode, L0, R0);
    if (auto *BO = dyn_cast<BinaryOperator>(NewBO0))
      BO->copyIRFlags(&Inst);
    Value *NewBO1 = Builder.CreateBinOp(Opcode, L1, R1);
    if (auto *BO = dyn_cast<BinaryOperator>(NewBO1))
      BO->copyIRFlags(&Inst);
    return new ShuffleVectorInst(NewBO0, NewBO1, Mask);
  }

  // It may not be safe to reorder shuffles and things like div, urem, etc.
  // because we may trap when executing those ops on unknown vector elements.
  // See PR20059.
  if (!isSafeToSpeculativelyExecute(&Inst))
    return nullptr;

  auto createBinOpShuffle = [&](Value *X, Value *Y, ArrayRef<int> M) {
    Value *XY = Builder.CreateBinOp(Opcode, X, Y);
    if (auto *BO = dyn_cast<BinaryOperator>(XY))
      BO->copyIRFlags(&Inst);
    return new ShuffleVectorInst(XY, M);
  };

  // If both arguments of the binary operation are shuffles that use the same
  // mask and shuffle within a single vector, move the shuffle after the binop.
  Value *V1, *V2;
  if (match(LHS, m_Shuffle(m_Value(V1), m_Undef(), m_Mask(Mask))) &&
      match(RHS, m_Shuffle(m_Value(V2), m_Undef(), m_SpecificMask(Mask))) &&
      V1->getType() == V2->getType() &&
      (LHS->hasOneUse() || RHS->hasOneUse() || LHS == RHS)) {
    // Op(shuffle(V1, Mask), shuffle(V2, Mask)) -> shuffle(Op(V1, V2), Mask)
    return createBinOpShuffle(V1, V2, Mask);
  }

  // If both arguments of a commutative binop are select-shuffles that use the
  // same mask with commuted operands, the shuffles are unnecessary.
  if (Inst.isCommutative() &&
      match(LHS, m_Shuffle(m_Value(V1), m_Value(V2), m_Mask(Mask))) &&
      match(RHS,
            m_Shuffle(m_Specific(V2), m_Specific(V1), m_SpecificMask(Mask)))) {
    auto *LShuf = cast<ShuffleVectorInst>(LHS);
    auto *RShuf = cast<ShuffleVectorInst>(RHS);
    // TODO: Allow shuffles that contain undefs in the mask?
    //       That is legal, but it reduces undef knowledge.
    // TODO: Allow arbitrary shuffles by shuffling after binop?
    //       That might be legal, but we have to deal with poison.
    if (LShuf->isSelect() &&
        !is_contained(LShuf->getShuffleMask(), UndefMaskElem) &&
        RShuf->isSelect() &&
        !is_contained(RShuf->getShuffleMask(), UndefMaskElem)) {
      // Example:
      // LHS = shuffle V1, V2, <0, 5, 6, 3>
      // RHS = shuffle V2, V1, <0, 5, 6, 3>
      // LHS + RHS --> (V10+V20, V21+V11, V22+V12, V13+V23) --> V1 + V2
      Instruction *NewBO = BinaryOperator::Create(Opcode, V1, V2);
      NewBO->copyIRFlags(&Inst);
      return NewBO;
    }
  }

  // If one argument is a shuffle within one vector and the other is a constant,
  // try moving the shuffle after the binary operation. This canonicalization
  // intends to move shuffles closer to other shuffles and binops closer to
  // other binops, so they can be folded. It may also enable demanded elements
  // transforms.
  Constant *C;
  auto *InstVTy = dyn_cast<FixedVectorType>(Inst.getType());
  if (InstVTy &&
      match(&Inst,
            m_c_BinOp(m_OneUse(m_Shuffle(m_Value(V1), m_Undef(), m_Mask(Mask))),
                      m_ImmConstant(C))) &&
      cast<FixedVectorType>(V1->getType())->getNumElements() <=
          InstVTy->getNumElements()) {
    assert(InstVTy->getScalarType() == V1->getType()->getScalarType() &&
           "Shuffle should not change scalar type");

    // Find constant NewC that has property:
    //   shuffle(NewC, ShMask) = C
    // If such constant does not exist (example: ShMask=<0,0> and C=<1,2>)
    // reorder is not possible. A 1-to-1 mapping is not required. Example:
    // ShMask = <1,1,2,2> and C = <5,5,6,6> --> NewC = <undef,5,6,undef>
    bool ConstOp1 = isa<Constant>(RHS);
    ArrayRef<int> ShMask = Mask;
    unsigned SrcVecNumElts =
        cast<FixedVectorType>(V1->getType())->getNumElements();
    UndefValue *UndefScalar = UndefValue::get(C->getType()->getScalarType());
    SmallVector<Constant *, 16> NewVecC(SrcVecNumElts, UndefScalar);
    bool MayChange = true;
    unsigned NumElts = InstVTy->getNumElements();
    for (unsigned I = 0; I < NumElts; ++I) {
      Constant *CElt = C->getAggregateElement(I);
      if (ShMask[I] >= 0) {
        assert(ShMask[I] < (int)NumElts && "Not expecting narrowing shuffle");
        Constant *NewCElt = NewVecC[ShMask[I]];
        // Bail out if:
        // 1. The constant vector contains a constant expression.
        // 2. The shuffle needs an element of the constant vector that can't
        //    be mapped to a new constant vector.
        // 3. This is a widening shuffle that copies elements of V1 into the
        //    extended elements (extending with undef is allowed).
        if (!CElt || (!isa<UndefValue>(NewCElt) && NewCElt != CElt) ||
            I >= SrcVecNumElts) {
          MayChange = false;
          break;
        }
        NewVecC[ShMask[I]] = CElt;
      }
      // If this is a widening shuffle, we must be able to extend with undef
      // elements. If the original binop does not produce an undef in the high
      // lanes, then this transform is not safe.
      // Similarly for undef lanes due to the shuffle mask, we can only
      // transform binops that preserve undef.
      // TODO: We could shuffle those non-undef constant values into the
      //       result by using a constant vector (rather than an undef vector)
      //       as operand 1 of the new binop, but that might be too aggressive
      //       for target-independent shuffle creation.
      if (I >= SrcVecNumElts || ShMask[I] < 0) {
        Constant *MaybeUndef =
            ConstOp1 ? ConstantExpr::get(Opcode, UndefScalar, CElt)
                     : ConstantExpr::get(Opcode, CElt, UndefScalar);
        if (!match(MaybeUndef, m_Undef())) {
          MayChange = false;
          break;
        }
      }
    }
    if (MayChange) {
      Constant *NewC = ConstantVector::get(NewVecC);
      // It may not be safe to execute a binop on a vector with undef elements
      // because the entire instruction can be folded to undef or create poison
      // that did not exist in the original code.
      if (Inst.isIntDivRem() || (Inst.isShift() && ConstOp1))
        NewC = getSafeVectorConstantForBinop(Opcode, NewC, ConstOp1);

      // Op(shuffle(V1, Mask), C) -> shuffle(Op(V1, NewC), Mask)
      // Op(C, shuffle(V1, Mask)) -> shuffle(Op(NewC, V1), Mask)
      Value *NewLHS = ConstOp1 ? V1 : NewC;
      Value *NewRHS = ConstOp1 ? NewC : V1;
      return createBinOpShuffle(NewLHS, NewRHS, Mask);
    }
  }

  // Try to reassociate to sink a splat shuffle after a binary operation.
  if (Inst.isAssociative() && Inst.isCommutative()) {
    // Canonicalize shuffle operand as LHS.
    if (isa<ShuffleVectorInst>(RHS))
      std::swap(LHS, RHS);

    Value *X;
    ArrayRef<int> MaskC;
    int SplatIndex;
    Value *Y, *OtherOp;
    if (!match(LHS,
               m_OneUse(m_Shuffle(m_Value(X), m_Undef(), m_Mask(MaskC)))) ||
        !match(MaskC, m_SplatOrUndefMask(SplatIndex)) ||
        X->getType() != Inst.getType() ||
        !match(RHS, m_OneUse(m_BinOp(Opcode, m_Value(Y), m_Value(OtherOp)))))
      return nullptr;

    // FIXME: This may not be safe if the analysis allows undef elements. By
    //        moving 'Y' before the splat shuffle, we are implicitly assuming
    //        that it is not undef/poison at the splat index.
    if (isSplatValue(OtherOp, SplatIndex)) {
      std::swap(Y, OtherOp);
    } else if (!isSplatValue(Y, SplatIndex)) {
      return nullptr;
    }

    // X and Y are splatted values, so perform the binary operation on those
    // values followed by a splat followed by the 2nd binary operation:
    // bo (splat X), (bo Y, OtherOp) --> bo (splat (bo X, Y)), OtherOp
    Value *NewBO = Builder.CreateBinOp(Opcode, X, Y);
    SmallVector<int, 8> NewMask(MaskC.size(), SplatIndex);
    Value *NewSplat = Builder.CreateShuffleVector(NewBO, NewMask);
    Instruction *R = BinaryOperator::Create(Opcode, NewSplat, OtherOp);

    // Intersect FMF on both new binops. Other (poison-generating) flags are
    // dropped to be safe.
    if (isa<FPMathOperator>(R)) {
      R->copyFastMathFlags(&Inst);
      R->andIRFlags(RHS);
    }
    if (auto *NewInstBO = dyn_cast<BinaryOperator>(NewBO))
      NewInstBO->copyIRFlags(R);
    return R;
  }

  return nullptr;
}

/// Try to narrow the width of a binop if at least 1 operand is an extend of
/// of a value. This requires a potentially expensive known bits check to make
/// sure the narrow op does not overflow.
Instruction *InstCombinerImpl::narrowMathIfNoOverflow(BinaryOperator &BO) {
  // We need at least one extended operand.
  Value *Op0 = BO.getOperand(0), *Op1 = BO.getOperand(1);

  // If this is a sub, we swap the operands since we always want an extension
  // on the RHS. The LHS can be an extension or a constant.
  if (BO.getOpcode() == Instruction::Sub)
    std::swap(Op0, Op1);

  Value *X;
  bool IsSext = match(Op0, m_SExt(m_Value(X)));
  if (!IsSext && !match(Op0, m_ZExt(m_Value(X))))
    return nullptr;

  // If both operands are the same extension from the same source type and we
  // can eliminate at least one (hasOneUse), this might work.
  CastInst::CastOps CastOpc = IsSext ? Instruction::SExt : Instruction::ZExt;
  Value *Y;
  if (!(match(Op1, m_ZExtOrSExt(m_Value(Y))) && X->getType() == Y->getType() &&
        cast<Operator>(Op1)->getOpcode() == CastOpc &&
        (Op0->hasOneUse() || Op1->hasOneUse()))) {
    // If that did not match, see if we have a suitable constant operand.
    // Truncating and extending must produce the same constant.
    Constant *WideC;
    if (!Op0->hasOneUse() || !match(Op1, m_Constant(WideC)))
      return nullptr;
    Constant *NarrowC = ConstantExpr::getTrunc(WideC, X->getType());
    if (ConstantExpr::getCast(CastOpc, NarrowC, BO.getType()) != WideC)
      return nullptr;
    Y = NarrowC;
  }

  // Swap back now that we found our operands.
  if (BO.getOpcode() == Instruction::Sub)
    std::swap(X, Y);

  // Both operands have narrow versions. Last step: the math must not overflow
  // in the narrow width.
  if (!willNotOverflow(BO.getOpcode(), X, Y, BO, IsSext))
    return nullptr;

  // bo (ext X), (ext Y) --> ext (bo X, Y)
  // bo (ext X), C       --> ext (bo X, C')
  Value *NarrowBO = Builder.CreateBinOp(BO.getOpcode(), X, Y, "narrow");
  if (auto *NewBinOp = dyn_cast<BinaryOperator>(NarrowBO)) {
    if (IsSext)
      NewBinOp->setHasNoSignedWrap();
    else
      NewBinOp->setHasNoUnsignedWrap();
  }
  return CastInst::Create(CastOpc, NarrowBO, BO.getType());
}

static bool isMergedGEPInBounds(GEPOperator &GEP1, GEPOperator &GEP2) {
  // At least one GEP must be inbounds.
  if (!GEP1.isInBounds() && !GEP2.isInBounds())
    return false;

  return (GEP1.isInBounds() || GEP1.hasAllZeroIndices()) &&
         (GEP2.isInBounds() || GEP2.hasAllZeroIndices());
}

/// Thread a GEP operation with constant indices through the constant true/false
/// arms of a select.
static Instruction *foldSelectGEP(GetElementPtrInst &GEP,
                                  InstCombiner::BuilderTy &Builder) {
  if (!GEP.hasAllConstantIndices())
    return nullptr;

  Instruction *Sel;
  Value *Cond;
  Constant *TrueC, *FalseC;
  if (!match(GEP.getPointerOperand(), m_Instruction(Sel)) ||
      !match(Sel,
             m_Select(m_Value(Cond), m_Constant(TrueC), m_Constant(FalseC))))
    return nullptr;

  // gep (select Cond, TrueC, FalseC), IndexC --> select Cond, TrueC', FalseC'
  // Propagate 'inbounds' and metadata from existing instructions.
  // Note: using IRBuilder to create the constants for efficiency.
  SmallVector<Value *, 4> IndexC(GEP.indices());
  bool IsInBounds = GEP.isInBounds();
  Type *Ty = GEP.getSourceElementType();
  Value *NewTrueC = IsInBounds ? Builder.CreateInBoundsGEP(Ty, TrueC, IndexC)
                               : Builder.CreateGEP(Ty, TrueC, IndexC);
  Value *NewFalseC = IsInBounds ? Builder.CreateInBoundsGEP(Ty, FalseC, IndexC)
                                : Builder.CreateGEP(Ty, FalseC, IndexC);
  return SelectInst::Create(Cond, NewTrueC, NewFalseC, "", nullptr, Sel);
}

Instruction *InstCombinerImpl::visitGEPOfGEP(GetElementPtrInst &GEP,
                                             GEPOperator *Src) {
  // Combine Indices - If the source pointer to this getelementptr instruction
  // is a getelementptr instruction with matching element type, combine the
  // indices of the two getelementptr instructions into a single instruction.
  if (Src->getResultElementType() != GEP.getSourceElementType())
    return nullptr;

  if (!shouldMergeGEPs(*cast<GEPOperator>(&GEP), *Src))
    return nullptr;

  if (Src->getNumOperands() == 2 && GEP.getNumOperands() == 2 &&
      Src->hasOneUse()) {
    Value *GO1 = GEP.getOperand(1);
    Value *SO1 = Src->getOperand(1);

    if (LI) {
      // Try to reassociate loop invariant GEP chains to enable LICM.
      if (Loop *L = LI->getLoopFor(GEP.getParent())) {
        // Reassociate the two GEPs if SO1 is variant in the loop and GO1 is
        // invariant: this breaks the dependence between GEPs and allows LICM
        // to hoist the invariant part out of the loop.
        if (L->isLoopInvariant(GO1) && !L->isLoopInvariant(SO1)) {
          // We have to be careful here.
          // We have something like:
          //  %src = getelementptr <ty>, <ty>* %base, <ty> %idx
          //  %gep = getelementptr <ty>, <ty>* %src, <ty> %idx2
          // If we just swap idx & idx2 then we could inadvertantly
          // change %src from a vector to a scalar, or vice versa.
          // Cases:
          //  1) %base a scalar & idx a scalar & idx2 a vector
          //      => Swapping idx & idx2 turns %src into a vector type.
          //  2) %base a scalar & idx a vector & idx2 a scalar
          //      => Swapping idx & idx2 turns %src in a scalar type
          //  3) %base, %idx, and %idx2 are scalars
          //      => %src & %gep are scalars
          //      => swapping idx & idx2 is safe
          //  4) %base a vector
          //      => %src is a vector
          //      => swapping idx & idx2 is safe.
          auto *SO0 = Src->getOperand(0);
          auto *SO0Ty = SO0->getType();
          if (!isa<VectorType>(GEP.getType()) || // case 3
              isa<VectorType>(SO0Ty)) { // case 4
            Src->setOperand(1, GO1);
            GEP.setOperand(1, SO1);
            return &GEP;
          } else {
            // Case 1 or 2
            // -- have to recreate %src & %gep
            // put NewSrc at same location as %src
            Builder.SetInsertPoint(cast<Instruction>(Src));
            Value *NewSrc = Builder.CreateGEP(
                GEP.getSourceElementType(), SO0, GO1, Src->getName());
            // Propagate 'inbounds' if the new source was not constant-folded.
            if (auto *NewSrcGEPI = dyn_cast<GetElementPtrInst>(NewSrc))
              NewSrcGEPI->setIsInBounds(Src->isInBounds());
            GetElementPtrInst *NewGEP = GetElementPtrInst::Create(
                GEP.getSourceElementType(), NewSrc, {SO1});
            NewGEP->setIsInBounds(GEP.isInBounds());
            return NewGEP;
          }
        }
      }
    }
  }

  // Note that if our source is a gep chain itself then we wait for that
  // chain to be resolved before we perform this transformation.  This
  // avoids us creating a TON of code in some cases.
  if (auto *SrcGEP = dyn_cast<GEPOperator>(Src->getOperand(0)))
    if (SrcGEP->getNumOperands() == 2 && shouldMergeGEPs(*Src, *SrcGEP))
      return nullptr;   // Wait until our source is folded to completion.

  SmallVector<Value*, 8> Indices;

  // Find out whether the last index in the source GEP is a sequential idx.
  bool EndsWithSequential = false;
  for (gep_type_iterator I = gep_type_begin(*Src), E = gep_type_end(*Src);
       I != E; ++I)
    EndsWithSequential = I.isSequential();

  // Can we combine the two pointer arithmetics offsets?
  if (EndsWithSequential) {
    // Replace: gep (gep %P, long B), long A, ...
    // With:    T = long A+B; gep %P, T, ...
    Value *SO1 = Src->getOperand(Src->getNumOperands()-1);
    Value *GO1 = GEP.getOperand(1);

    // If they aren't the same type, then the input hasn't been processed
    // by the loop above yet (which canonicalizes sequential index types to
    // intptr_t).  Just avoid transforming this until the input has been
    // normalized.
    if (SO1->getType() != GO1->getType())
      return nullptr;

    Value *Sum =
        SimplifyAddInst(GO1, SO1, false, false, SQ.getWithInstruction(&GEP));
    // Only do the combine when we are sure the cost after the
    // merge is never more than that before the merge.
    if (Sum == nullptr)
      return nullptr;

    // Update the GEP in place if possible.
    if (Src->getNumOperands() == 2) {
      GEP.setIsInBounds(isMergedGEPInBounds(*Src, *cast<GEPOperator>(&GEP)));
      replaceOperand(GEP, 0, Src->getOperand(0));
      replaceOperand(GEP, 1, Sum);
      return &GEP;
    }
    Indices.append(Src->op_begin()+1, Src->op_end()-1);
    Indices.push_back(Sum);
    Indices.append(GEP.op_begin()+2, GEP.op_end());
  } else if (isa<Constant>(*GEP.idx_begin()) &&
             cast<Constant>(*GEP.idx_begin())->isNullValue() &&
             Src->getNumOperands() != 1) {
    // Otherwise we can do the fold if the first index of the GEP is a zero
    Indices.append(Src->op_begin()+1, Src->op_end());
    Indices.append(GEP.idx_begin()+1, GEP.idx_end());
  }

  if (!Indices.empty())
    return isMergedGEPInBounds(*Src, *cast<GEPOperator>(&GEP))
               ? GetElementPtrInst::CreateInBounds(
                     Src->getSourceElementType(), Src->getOperand(0), Indices,
                     GEP.getName())
               : GetElementPtrInst::Create(Src->getSourceElementType(),
                                           Src->getOperand(0), Indices,
                                           GEP.getName());

  return nullptr;
}

// Note that we may have also stripped an address space cast in between.
Instruction *InstCombinerImpl::visitGEPOfBitcast(BitCastInst *BCI,
                                                 GetElementPtrInst &GEP) {
  // With opaque pointers, there is no pointer element type we can use to
  // adjust the GEP type.
  PointerType *SrcType = cast<PointerType>(BCI->getSrcTy());
  if (SrcType->isOpaque())
    return nullptr;

  Type *GEPEltType = GEP.getSourceElementType();
  Type *SrcEltType = SrcType->getNonOpaquePointerElementType();
  Value *SrcOp = BCI->getOperand(0);

  // GEP directly using the source operand if this GEP is accessing an element
  // of a bitcasted pointer to vector or array of the same dimensions:
  // gep (bitcast <c x ty>* X to [c x ty]*), Y, Z --> gep X, Y, Z
  // gep (bitcast [c x ty]* X to <c x ty>*), Y, Z --> gep X, Y, Z
  auto areMatchingArrayAndVecTypes = [](Type *ArrTy, Type *VecTy,
                                        const DataLayout &DL) {
    auto *VecVTy = cast<FixedVectorType>(VecTy);
    return ArrTy->getArrayElementType() == VecVTy->getElementType() &&
           ArrTy->getArrayNumElements() == VecVTy->getNumElements() &&
           DL.getTypeAllocSize(ArrTy) == DL.getTypeAllocSize(VecTy);
  };
  if (GEP.getNumOperands() == 3 &&
      ((GEPEltType->isArrayTy() && isa<FixedVectorType>(SrcEltType) &&
        areMatchingArrayAndVecTypes(GEPEltType, SrcEltType, DL)) ||
       (isa<FixedVectorType>(GEPEltType) && SrcEltType->isArrayTy() &&
        areMatchingArrayAndVecTypes(SrcEltType, GEPEltType, DL)))) {

    // Create a new GEP here, as using `setOperand()` followed by
    // `setSourceElementType()` won't actually update the type of the
    // existing GEP Value. Causing issues if this Value is accessed when
    // constructing an AddrSpaceCastInst
    SmallVector<Value *, 8> Indices(GEP.indices());
    Value *NGEP = GEP.isInBounds()
                      ? Builder.CreateInBoundsGEP(SrcEltType, SrcOp, Indices)
                      : Builder.CreateGEP(SrcEltType, SrcOp, Indices);
    NGEP->takeName(&GEP);

    // Preserve GEP address space to satisfy users
    if (NGEP->getType()->getPointerAddressSpace() != GEP.getAddressSpace())
      return new AddrSpaceCastInst(NGEP, GEP.getType());

    return replaceInstUsesWith(GEP, NGEP);
  }

  // See if we can simplify:
  //   X = bitcast A* to B*
  //   Y = gep X, <...constant indices...>
  // into a gep of the original struct. This is important for SROA and alias
  // analysis of unions. If "A" is also a bitcast, wait for A/X to be merged.
  unsigned OffsetBits = DL.getIndexTypeSizeInBits(GEP.getType());
  APInt Offset(OffsetBits, 0);

  // If the bitcast argument is an allocation, The bitcast is for convertion
  // to actual type of allocation. Removing such bitcasts, results in having
  // GEPs with i8* base and pure byte offsets. That means GEP is not aware of
  // struct or array hierarchy.
  // By avoiding such GEPs, phi translation and MemoryDependencyAnalysis have
  // a better chance to succeed.
  if (!isa<BitCastInst>(SrcOp) && GEP.accumulateConstantOffset(DL, Offset) &&
      !isAllocationFn(SrcOp, &TLI)) {
    // If this GEP instruction doesn't move the pointer, just replace the GEP
    // with a bitcast of the real input to the dest type.
    if (!Offset) {
      // If the bitcast is of an allocation, and the allocation will be
      // converted to match the type of the cast, don't touch this.
      if (isa<AllocaInst>(SrcOp)) {
        // See if the bitcast simplifies, if so, don't nuke this GEP yet.
        if (Instruction *I = visitBitCast(*BCI)) {
          if (I != BCI) {
            I->takeName(BCI);
            BCI->getParent()->getInstList().insert(BCI->getIterator(), I);
            replaceInstUsesWith(*BCI, I);
          }
          return &GEP;
        }
      }

      if (SrcType->getPointerAddressSpace() != GEP.getAddressSpace())
        return new AddrSpaceCastInst(SrcOp, GEP.getType());
      return new BitCastInst(SrcOp, GEP.getType());
    }

    // Otherwise, if the offset is non-zero, we need to find out if there is a
    // field at Offset in 'A's type.  If so, we can pull the cast through the
    // GEP.
    SmallVector<Value*, 8> NewIndices;
    if (findElementAtOffset(SrcType, Offset.getSExtValue(), NewIndices, DL)) {
      Value *NGEP =
          GEP.isInBounds()
              ? Builder.CreateInBoundsGEP(SrcEltType, SrcOp, NewIndices)
              : Builder.CreateGEP(SrcEltType, SrcOp, NewIndices);

      if (NGEP->getType() == GEP.getType())
        return replaceInstUsesWith(GEP, NGEP);
      NGEP->takeName(&GEP);

      if (NGEP->getType()->getPointerAddressSpace() != GEP.getAddressSpace())
        return new AddrSpaceCastInst(NGEP, GEP.getType());
      return new BitCastInst(NGEP, GEP.getType());
    }
  }

  return nullptr;
}

Instruction *InstCombinerImpl::visitGetElementPtrInst(GetElementPtrInst &GEP) {
  Value *PtrOp = GEP.getOperand(0);
  SmallVector<Value *, 8> Indices(GEP.indices());
  Type *GEPType = GEP.getType();
  Type *GEPEltType = GEP.getSourceElementType();
  bool IsGEPSrcEleScalable = isa<ScalableVectorType>(GEPEltType);
  if (Value *V = SimplifyGEPInst(GEPEltType, PtrOp, Indices, GEP.isInBounds(),
                                 SQ.getWithInstruction(&GEP)))
    return replaceInstUsesWith(GEP, V);

  // For vector geps, use the generic demanded vector support.
  // Skip if GEP return type is scalable. The number of elements is unknown at
  // compile-time.
  if (auto *GEPFVTy = dyn_cast<FixedVectorType>(GEPType)) {
    auto VWidth = GEPFVTy->getNumElements();
    APInt UndefElts(VWidth, 0);
    APInt AllOnesEltMask(APInt::getAllOnes(VWidth));
    if (Value *V = SimplifyDemandedVectorElts(&GEP, AllOnesEltMask,
                                              UndefElts)) {
      if (V != &GEP)
        return replaceInstUsesWith(GEP, V);
      return &GEP;
    }

    // TODO: 1) Scalarize splat operands, 2) scalarize entire instruction if
    // possible (decide on canonical form for pointer broadcast), 3) exploit
    // undef elements to decrease demanded bits
  }

  // Eliminate unneeded casts for indices, and replace indices which displace
  // by multiples of a zero size type with zero.
  bool MadeChange = false;

  // Index width may not be the same width as pointer width.
  // Data layout chooses the right type based on supported integer types.
  Type *NewScalarIndexTy =
      DL.getIndexType(GEP.getPointerOperandType()->getScalarType());

  gep_type_iterator GTI = gep_type_begin(GEP);
  for (User::op_iterator I = GEP.op_begin() + 1, E = GEP.op_end(); I != E;
       ++I, ++GTI) {
    // Skip indices into struct types.
    if (GTI.isStruct())
      continue;

    Type *IndexTy = (*I)->getType();
    Type *NewIndexType =
        IndexTy->isVectorTy()
            ? VectorType::get(NewScalarIndexTy,
                              cast<VectorType>(IndexTy)->getElementCount())
            : NewScalarIndexTy;

    // If the element type has zero size then any index over it is equivalent
    // to an index of zero, so replace it with zero if it is not zero already.
    Type *EltTy = GTI.getIndexedType();
    if (EltTy->isSized() && DL.getTypeAllocSize(EltTy).isZero())
      if (!isa<Constant>(*I) || !match(I->get(), m_Zero())) {
        *I = Constant::getNullValue(NewIndexType);
        MadeChange = true;
      }

    if (IndexTy != NewIndexType) {
      // If we are using a wider index than needed for this platform, shrink
      // it to what we need.  If narrower, sign-extend it to what we need.
      // This explicit cast can make subsequent optimizations more obvious.
      *I = Builder.CreateIntCast(*I, NewIndexType, true);
      MadeChange = true;
    }
  }
  if (MadeChange)
    return &GEP;

  // Check to see if the inputs to the PHI node are getelementptr instructions.
  if (auto *PN = dyn_cast<PHINode>(PtrOp)) {
    auto *Op1 = dyn_cast<GetElementPtrInst>(PN->getOperand(0));
    if (!Op1)
      return nullptr;

    // Don't fold a GEP into itself through a PHI node. This can only happen
    // through the back-edge of a loop. Folding a GEP into itself means that
    // the value of the previous iteration needs to be stored in the meantime,
    // thus requiring an additional register variable to be live, but not
    // actually achieving anything (the GEP still needs to be executed once per
    // loop iteration).
    if (Op1 == &GEP)
      return nullptr;

    int DI = -1;

    for (auto I = PN->op_begin()+1, E = PN->op_end(); I !=E; ++I) {
      auto *Op2 = dyn_cast<GetElementPtrInst>(*I);
      if (!Op2 || Op1->getNumOperands() != Op2->getNumOperands() ||
          Op1->getSourceElementType() != Op2->getSourceElementType())
        return nullptr;

      // As for Op1 above, don't try to fold a GEP into itself.
      if (Op2 == &GEP)
        return nullptr;

      // Keep track of the type as we walk the GEP.
      Type *CurTy = nullptr;

      for (unsigned J = 0, F = Op1->getNumOperands(); J != F; ++J) {
        if (Op1->getOperand(J)->getType() != Op2->getOperand(J)->getType())
          return nullptr;

        if (Op1->getOperand(J) != Op2->getOperand(J)) {
          if (DI == -1) {
            // We have not seen any differences yet in the GEPs feeding the
            // PHI yet, so we record this one if it is allowed to be a
            // variable.

            // The first two arguments can vary for any GEP, the rest have to be
            // static for struct slots
            if (J > 1) {
              assert(CurTy && "No current type?");
              if (CurTy->isStructTy())
                return nullptr;
            }

            DI = J;
          } else {
            // The GEP is different by more than one input. While this could be
            // extended to support GEPs that vary by more than one variable it
            // doesn't make sense since it greatly increases the complexity and
            // would result in an R+R+R addressing mode which no backend
            // directly supports and would need to be broken into several
            // simpler instructions anyway.
            return nullptr;
          }
        }

        // Sink down a layer of the type for the next iteration.
        if (J > 0) {
          if (J == 1) {
            CurTy = Op1->getSourceElementType();
          } else {
            CurTy =
                GetElementPtrInst::getTypeAtIndex(CurTy, Op1->getOperand(J));
          }
        }
      }
    }

    // If not all GEPs are identical we'll have to create a new PHI node.
    // Check that the old PHI node has only one use so that it will get
    // removed.
    if (DI != -1 && !PN->hasOneUse())
      return nullptr;

    auto *NewGEP = cast<GetElementPtrInst>(Op1->clone());
    if (DI == -1) {
      // All the GEPs feeding the PHI are identical. Clone one down into our
      // BB so that it can be merged with the current GEP.
    } else {
      // All the GEPs feeding the PHI differ at a single offset. Clone a GEP
      // into the current block so it can be merged, and create a new PHI to
      // set that index.
      PHINode *NewPN;
      {
        IRBuilderBase::InsertPointGuard Guard(Builder);
        Builder.SetInsertPoint(PN);
        NewPN = Builder.CreatePHI(Op1->getOperand(DI)->getType(),
                                  PN->getNumOperands());
      }

      for (auto &I : PN->operands())
        NewPN->addIncoming(cast<GEPOperator>(I)->getOperand(DI),
                           PN->getIncomingBlock(I));

      NewGEP->setOperand(DI, NewPN);
    }

    GEP.getParent()->getInstList().insert(
        GEP.getParent()->getFirstInsertionPt(), NewGEP);
    replaceOperand(GEP, 0, NewGEP);
    PtrOp = NewGEP;
  }

  if (auto *Src = dyn_cast<GEPOperator>(PtrOp))
    if (Instruction *I = visitGEPOfGEP(GEP, Src))
      return I;

  // Skip if GEP source element type is scalable. The type alloc size is unknown
  // at compile-time.
  if (GEP.getNumIndices() == 1 && !IsGEPSrcEleScalable) {
    unsigned AS = GEP.getPointerAddressSpace();
    if (GEP.getOperand(1)->getType()->getScalarSizeInBits() ==
        DL.getIndexSizeInBits(AS)) {
      uint64_t TyAllocSize = DL.getTypeAllocSize(GEPEltType).getFixedSize();

      bool Matched = false;
      uint64_t C;
      Value *V = nullptr;
      if (TyAllocSize == 1) {
        V = GEP.getOperand(1);
        Matched = true;
      } else if (match(GEP.getOperand(1),
                       m_AShr(m_Value(V), m_ConstantInt(C)))) {
        if (TyAllocSize == 1ULL << C)
          Matched = true;
      } else if (match(GEP.getOperand(1),
                       m_SDiv(m_Value(V), m_ConstantInt(C)))) {
        if (TyAllocSize == C)
          Matched = true;
      }

      // Canonicalize (gep i8* X, (ptrtoint Y)-(ptrtoint X)) to (bitcast Y), but
      // only if both point to the same underlying object (otherwise provenance
      // is not necessarily retained).
      Value *Y;
      Value *X = GEP.getOperand(0);
      if (Matched &&
          match(V, m_Sub(m_PtrToInt(m_Value(Y)), m_PtrToInt(m_Specific(X)))) &&
          getUnderlyingObject(X) == getUnderlyingObject(Y))
        return CastInst::CreatePointerBitCastOrAddrSpaceCast(Y, GEPType);
    }
  }

  // We do not handle pointer-vector geps here.
  if (GEPType->isVectorTy())
    return nullptr;

  // Handle gep(bitcast x) and gep(gep x, 0, 0, 0).
  Value *StrippedPtr = PtrOp->stripPointerCasts();
  PointerType *StrippedPtrTy = cast<PointerType>(StrippedPtr->getType());

  // TODO: The basic approach of these folds is not compatible with opaque
  // pointers, because we can't use bitcasts as a hint for a desirable GEP
  // type. Instead, we should perform canonicalization directly on the GEP
  // type. For now, skip these.
  if (StrippedPtr != PtrOp && !StrippedPtrTy->isOpaque()) {
    bool HasZeroPointerIndex = false;
    Type *StrippedPtrEltTy = StrippedPtrTy->getNonOpaquePointerElementType();

    if (auto *C = dyn_cast<ConstantInt>(GEP.getOperand(1)))
      HasZeroPointerIndex = C->isZero();

    // Transform: GEP (bitcast [10 x i8]* X to [0 x i8]*), i32 0, ...
    // into     : GEP [10 x i8]* X, i32 0, ...
    //
    // Likewise, transform: GEP (bitcast i8* X to [0 x i8]*), i32 0, ...
    //           into     : GEP i8* X, ...
    //
    // This occurs when the program declares an array extern like "int X[];"
    if (HasZeroPointerIndex) {
      if (auto *CATy = dyn_cast<ArrayType>(GEPEltType)) {
        // GEP (bitcast i8* X to [0 x i8]*), i32 0, ... ?
        if (CATy->getElementType() == StrippedPtrEltTy) {
          // -> GEP i8* X, ...
          SmallVector<Value *, 8> Idx(drop_begin(GEP.indices()));
          GetElementPtrInst *Res = GetElementPtrInst::Create(
              StrippedPtrEltTy, StrippedPtr, Idx, GEP.getName());
          Res->setIsInBounds(GEP.isInBounds());
          if (StrippedPtrTy->getAddressSpace() == GEP.getAddressSpace())
            return Res;
          // Insert Res, and create an addrspacecast.
          // e.g.,
          // GEP (addrspacecast i8 addrspace(1)* X to [0 x i8]*), i32 0, ...
          // ->
          // %0 = GEP i8 addrspace(1)* X, ...
          // addrspacecast i8 addrspace(1)* %0 to i8*
          return new AddrSpaceCastInst(Builder.Insert(Res), GEPType);
        }

        if (auto *XATy = dyn_cast<ArrayType>(StrippedPtrEltTy)) {
          // GEP (bitcast [10 x i8]* X to [0 x i8]*), i32 0, ... ?
          if (CATy->getElementType() == XATy->getElementType()) {
            // -> GEP [10 x i8]* X, i32 0, ...
            // At this point, we know that the cast source type is a pointer
            // to an array of the same type as the destination pointer
            // array.  Because the array type is never stepped over (there
            // is a leading zero) we can fold the cast into this GEP.
            if (StrippedPtrTy->getAddressSpace() == GEP.getAddressSpace()) {
              GEP.setSourceElementType(XATy);
              return replaceOperand(GEP, 0, StrippedPtr);
            }
            // Cannot replace the base pointer directly because StrippedPtr's
            // address space is different. Instead, create a new GEP followed by
            // an addrspacecast.
            // e.g.,
            // GEP (addrspacecast [10 x i8] addrspace(1)* X to [0 x i8]*),
            //   i32 0, ...
            // ->
            // %0 = GEP [10 x i8] addrspace(1)* X, ...
            // addrspacecast i8 addrspace(1)* %0 to i8*
            SmallVector<Value *, 8> Idx(GEP.indices());
            Value *NewGEP =
                GEP.isInBounds()
                    ? Builder.CreateInBoundsGEP(StrippedPtrEltTy, StrippedPtr,
                                                Idx, GEP.getName())
                    : Builder.CreateGEP(StrippedPtrEltTy, StrippedPtr, Idx,
                                        GEP.getName());
            return new AddrSpaceCastInst(NewGEP, GEPType);
          }
        }
      }
    } else if (GEP.getNumOperands() == 2 && !IsGEPSrcEleScalable) {
      // Skip if GEP source element type is scalable. The type alloc size is
      // unknown at compile-time.
      // Transform things like: %t = getelementptr i32*
      // bitcast ([2 x i32]* %str to i32*), i32 %V into:  %t1 = getelementptr [2
      // x i32]* %str, i32 0, i32 %V; bitcast
      if (StrippedPtrEltTy->isArrayTy() &&
          DL.getTypeAllocSize(StrippedPtrEltTy->getArrayElementType()) ==
              DL.getTypeAllocSize(GEPEltType)) {
        Type *IdxType = DL.getIndexType(GEPType);
        Value *Idx[2] = { Constant::getNullValue(IdxType), GEP.getOperand(1) };
        Value *NewGEP =
            GEP.isInBounds()
                ? Builder.CreateInBoundsGEP(StrippedPtrEltTy, StrippedPtr, Idx,
                                            GEP.getName())
                : Builder.CreateGEP(StrippedPtrEltTy, StrippedPtr, Idx,
                                    GEP.getName());

        // V and GEP are both pointer types --> BitCast
        return CastInst::CreatePointerBitCastOrAddrSpaceCast(NewGEP, GEPType);
      }

      // Transform things like:
      // %V = mul i64 %N, 4
      // %t = getelementptr i8* bitcast (i32* %arr to i8*), i32 %V
      // into:  %t1 = getelementptr i32* %arr, i32 %N; bitcast
      if (GEPEltType->isSized() && StrippedPtrEltTy->isSized()) {
        // Check that changing the type amounts to dividing the index by a scale
        // factor.
        uint64_t ResSize = DL.getTypeAllocSize(GEPEltType).getFixedSize();
        uint64_t SrcSize = DL.getTypeAllocSize(StrippedPtrEltTy).getFixedSize();
        if (ResSize && SrcSize % ResSize == 0) {
          Value *Idx = GEP.getOperand(1);
          unsigned BitWidth = Idx->getType()->getPrimitiveSizeInBits();
          uint64_t Scale = SrcSize / ResSize;

          // Earlier transforms ensure that the index has the right type
          // according to Data Layout, which considerably simplifies the
          // logic by eliminating implicit casts.
          assert(Idx->getType() == DL.getIndexType(GEPType) &&
                 "Index type does not match the Data Layout preferences");

          bool NSW;
          if (Value *NewIdx = Descale(Idx, APInt(BitWidth, Scale), NSW)) {
            // Successfully decomposed Idx as NewIdx * Scale, form a new GEP.
            // If the multiplication NewIdx * Scale may overflow then the new
            // GEP may not be "inbounds".
            Value *NewGEP =
                GEP.isInBounds() && NSW
                    ? Builder.CreateInBoundsGEP(StrippedPtrEltTy, StrippedPtr,
                                                NewIdx, GEP.getName())
                    : Builder.CreateGEP(StrippedPtrEltTy, StrippedPtr, NewIdx,
                                        GEP.getName());

            // The NewGEP must be pointer typed, so must the old one -> BitCast
            return CastInst::CreatePointerBitCastOrAddrSpaceCast(NewGEP,
                                                                 GEPType);
          }
        }
      }

      // Similarly, transform things like:
      // getelementptr i8* bitcast ([100 x double]* X to i8*), i32 %tmp
      //   (where tmp = 8*tmp2) into:
      // getelementptr [100 x double]* %arr, i32 0, i32 %tmp2; bitcast
      if (GEPEltType->isSized() && StrippedPtrEltTy->isSized() &&
          StrippedPtrEltTy->isArrayTy()) {
        // Check that changing to the array element type amounts to dividing the
        // index by a scale factor.
        uint64_t ResSize = DL.getTypeAllocSize(GEPEltType).getFixedSize();
        uint64_t ArrayEltSize =
            DL.getTypeAllocSize(StrippedPtrEltTy->getArrayElementType())
                .getFixedSize();
        if (ResSize && ArrayEltSize % ResSize == 0) {
          Value *Idx = GEP.getOperand(1);
          unsigned BitWidth = Idx->getType()->getPrimitiveSizeInBits();
          uint64_t Scale = ArrayEltSize / ResSize;

          // Earlier transforms ensure that the index has the right type
          // according to the Data Layout, which considerably simplifies
          // the logic by eliminating implicit casts.
          assert(Idx->getType() == DL.getIndexType(GEPType) &&
                 "Index type does not match the Data Layout preferences");

          bool NSW;
          if (Value *NewIdx = Descale(Idx, APInt(BitWidth, Scale), NSW)) {
            // Successfully decomposed Idx as NewIdx * Scale, form a new GEP.
            // If the multiplication NewIdx * Scale may overflow then the new
            // GEP may not be "inbounds".
            Type *IndTy = DL.getIndexType(GEPType);
            Value *Off[2] = {Constant::getNullValue(IndTy), NewIdx};

            Value *NewGEP =
                GEP.isInBounds() && NSW
                    ? Builder.CreateInBoundsGEP(StrippedPtrEltTy, StrippedPtr,
                                                Off, GEP.getName())
                    : Builder.CreateGEP(StrippedPtrEltTy, StrippedPtr, Off,
                                        GEP.getName());
            // The NewGEP must be pointer typed, so must the old one -> BitCast
            return CastInst::CreatePointerBitCastOrAddrSpaceCast(NewGEP,
                                                                 GEPType);
          }
        }
      }
    }
  }

  // addrspacecast between types is canonicalized as a bitcast, then an
  // addrspacecast. To take advantage of the below bitcast + struct GEP, look
  // through the addrspacecast.
  Value *ASCStrippedPtrOp = PtrOp;
  if (auto *ASC = dyn_cast<AddrSpaceCastInst>(PtrOp)) {
    //   X = bitcast A addrspace(1)* to B addrspace(1)*
    //   Y = addrspacecast A addrspace(1)* to B addrspace(2)*
    //   Z = gep Y, <...constant indices...>
    // Into an addrspacecasted GEP of the struct.
    if (auto *BC = dyn_cast<BitCastInst>(ASC->getOperand(0)))
      ASCStrippedPtrOp = BC;
  }

  if (auto *BCI = dyn_cast<BitCastInst>(ASCStrippedPtrOp))
    if (Instruction *I = visitGEPOfBitcast(BCI, GEP))
      return I;

  if (!GEP.isInBounds()) {
    unsigned IdxWidth =
        DL.getIndexSizeInBits(PtrOp->getType()->getPointerAddressSpace());
    APInt BasePtrOffset(IdxWidth, 0);
    Value *UnderlyingPtrOp =
            PtrOp->stripAndAccumulateInBoundsConstantOffsets(DL,
                                                             BasePtrOffset);
    if (auto *AI = dyn_cast<AllocaInst>(UnderlyingPtrOp)) {
      if (GEP.accumulateConstantOffset(DL, BasePtrOffset) &&
          BasePtrOffset.isNonNegative()) {
        APInt AllocSize(
            IdxWidth,
            DL.getTypeAllocSize(AI->getAllocatedType()).getKnownMinSize());
        if (BasePtrOffset.ule(AllocSize)) {
          return GetElementPtrInst::CreateInBounds(
              GEP.getSourceElementType(), PtrOp, Indices, GEP.getName());
        }
      }
    }
  }

  if (Instruction *R = foldSelectGEP(GEP, Builder))
    return R;

  return nullptr;
}

static bool isNeverEqualToUnescapedAlloc(Value *V, const TargetLibraryInfo &TLI,
                                         Instruction *AI) {
  if (isa<ConstantPointerNull>(V))
    return true;
  if (auto *LI = dyn_cast<LoadInst>(V))
    return isa<GlobalVariable>(LI->getPointerOperand());
  // Two distinct allocations will never be equal.
  return isAllocLikeFn(V, &TLI) && V != AI;
}

/// Given a call CB which uses an address UsedV, return true if we can prove the
/// call's only possible effect is storing to V.
static bool isRemovableWrite(CallBase &CB, Value *UsedV,
                             const TargetLibraryInfo &TLI) {
  if (!CB.use_empty())
    // TODO: add recursion if returned attribute is present
    return false;

  if (CB.isTerminator())
    // TODO: remove implementation restriction
    return false;

  if (!CB.willReturn() || !CB.doesNotThrow())
    return false;

  // If the only possible side effect of the call is writing to the alloca,
  // and the result isn't used, we can safely remove any reads implied by the
  // call including those which might read the alloca itself.
  Optional<MemoryLocation> Dest = MemoryLocation::getForDest(&CB, TLI);
  return Dest && Dest->Ptr == UsedV;
}

static bool isAllocSiteRemovable(Instruction *AI,
                                 SmallVectorImpl<WeakTrackingVH> &Users,
                                 const TargetLibraryInfo &TLI) {
  SmallVector<Instruction*, 4> Worklist;
  Worklist.push_back(AI);

  do {
    Instruction *PI = Worklist.pop_back_val();
    for (User *U : PI->users()) {
      Instruction *I = cast<Instruction>(U);
      switch (I->getOpcode()) {
      default:
        // Give up the moment we see something we can't handle.
        return false;

      case Instruction::AddrSpaceCast:
      case Instruction::BitCast:
      case Instruction::GetElementPtr:
        Users.emplace_back(I);
        Worklist.push_back(I);
        continue;

      case Instruction::ICmp: {
        ICmpInst *ICI = cast<ICmpInst>(I);
        // We can fold eq/ne comparisons with null to false/true, respectively.
        // We also fold comparisons in some conditions provided the alloc has
        // not escaped (see isNeverEqualToUnescapedAlloc).
        if (!ICI->isEquality())
          return false;
        unsigned OtherIndex = (ICI->getOperand(0) == PI) ? 1 : 0;
        if (!isNeverEqualToUnescapedAlloc(ICI->getOperand(OtherIndex), TLI, AI))
          return false;
        Users.emplace_back(I);
        continue;
      }

      case Instruction::Call:
        // Ignore no-op and store intrinsics.
        if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
          switch (II->getIntrinsicID()) {
          default:
            return false;

          case Intrinsic::memmove:
          case Intrinsic::memcpy:
          case Intrinsic::memset: {
            MemIntrinsic *MI = cast<MemIntrinsic>(II);
            if (MI->isVolatile() || MI->getRawDest() != PI)
              return false;
            LLVM_FALLTHROUGH;
          }
          case Intrinsic::assume:
          case Intrinsic::invariant_start:
          case Intrinsic::invariant_end:
          case Intrinsic::lifetime_start:
          case Intrinsic::lifetime_end:
          case Intrinsic::objectsize:
            Users.emplace_back(I);
            continue;
          case Intrinsic::launder_invariant_group:
          case Intrinsic::strip_invariant_group:
            Users.emplace_back(I);
            Worklist.push_back(I);
            continue;
          }
        }

        if (isRemovableWrite(*cast<CallBase>(I), PI, TLI)) {
          Users.emplace_back(I);
          continue;
        }

        if (isFreeCall(I, &TLI)) {
          Users.emplace_back(I);
          continue;
        }

        if (isReallocLikeFn(I, &TLI)) {
          Users.emplace_back(I);
          Worklist.push_back(I);
          continue;
        }

        return false;

      case Instruction::Store: {
        StoreInst *SI = cast<StoreInst>(I);
        if (SI->isVolatile() || SI->getPointerOperand() != PI)
          return false;
        Users.emplace_back(I);
        continue;
      }
      }
      llvm_unreachable("missing a return?");
    }
  } while (!Worklist.empty());
  return true;
}

Instruction *InstCombinerImpl::visitAllocSite(Instruction &MI) {
  assert(isa<AllocaInst>(MI) || isAllocRemovable(&cast<CallBase>(MI), &TLI));

  // If we have a malloc call which is only used in any amount of comparisons to
  // null and free calls, delete the calls and replace the comparisons with true
  // or false as appropriate.

  // This is based on the principle that we can substitute our own allocation
  // function (which will never return null) rather than knowledge of the
  // specific function being called. In some sense this can change the permitted
  // outputs of a program (when we convert a malloc to an alloca, the fact that
  // the allocation is now on the stack is potentially visible, for example),
  // but we believe in a permissible manner.
  SmallVector<WeakTrackingVH, 64> Users;

  // If we are removing an alloca with a dbg.declare, insert dbg.value calls
  // before each store.
  SmallVector<DbgVariableIntrinsic *, 8> DVIs;
  std::unique_ptr<DIBuilder> DIB;
  if (isa<AllocaInst>(MI)) {
    findDbgUsers(DVIs, &MI);
    DIB.reset(new DIBuilder(*MI.getModule(), /*AllowUnresolved=*/false));
  }

  if (isAllocSiteRemovable(&MI, Users, TLI)) {
    for (unsigned i = 0, e = Users.size(); i != e; ++i) {
      // Lowering all @llvm.objectsize calls first because they may
      // use a bitcast/GEP of the alloca we are removing.
      if (!Users[i])
       continue;

      Instruction *I = cast<Instruction>(&*Users[i]);

      if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
        if (II->getIntrinsicID() == Intrinsic::objectsize) {
          Value *Result =
              lowerObjectSizeCall(II, DL, &TLI, /*MustSucceed=*/true);
          replaceInstUsesWith(*I, Result);
          eraseInstFromFunction(*I);
          Users[i] = nullptr; // Skip examining in the next loop.
        }
      }
    }
    for (unsigned i = 0, e = Users.size(); i != e; ++i) {
      if (!Users[i])
        continue;

      Instruction *I = cast<Instruction>(&*Users[i]);

      if (ICmpInst *C = dyn_cast<ICmpInst>(I)) {
        replaceInstUsesWith(*C,
                            ConstantInt::get(Type::getInt1Ty(C->getContext()),
                                             C->isFalseWhenEqual()));
      } else if (auto *SI = dyn_cast<StoreInst>(I)) {
        for (auto *DVI : DVIs)
          if (DVI->isAddressOfVariable())
            ConvertDebugDeclareToDebugValue(DVI, SI, *DIB);
      } else {
        // Casts, GEP, or anything else: we're about to delete this instruction,
        // so it can not have any valid uses.
        replaceInstUsesWith(*I, PoisonValue::get(I->getType()));
      }
      eraseInstFromFunction(*I);
    }

    if (InvokeInst *II = dyn_cast<InvokeInst>(&MI)) {
      // Replace invoke with a NOP intrinsic to maintain the original CFG
      Module *M = II->getModule();
      Function *F = Intrinsic::getDeclaration(M, Intrinsic::donothing);
      InvokeInst::Create(F, II->getNormalDest(), II->getUnwindDest(),
                         None, "", II->getParent());
    }

    // Remove debug intrinsics which describe the value contained within the
    // alloca. In addition to removing dbg.{declare,addr} which simply point to
    // the alloca, remove dbg.value(<alloca>, ..., DW_OP_deref)'s as well, e.g.:
    //
    // ```
    //   define void @foo(i32 %0) {
    //     %a = alloca i32                              ; Deleted.
    //     store i32 %0, i32* %a
    //     dbg.value(i32 %0, "arg0")                    ; Not deleted.
    //     dbg.value(i32* %a, "arg0", DW_OP_deref)      ; Deleted.
    //     call void @trivially_inlinable_no_op(i32* %a)
    //     ret void
    //  }
    // ```
    //
    // This may not be required if we stop describing the contents of allocas
    // using dbg.value(<alloca>, ..., DW_OP_deref), but we currently do this in
    // the LowerDbgDeclare utility.
    //
    // If there is a dead store to `%a` in @trivially_inlinable_no_op, the
    // "arg0" dbg.value may be stale after the call. However, failing to remove
    // the DW_OP_deref dbg.value causes large gaps in location coverage.
    for (auto *DVI : DVIs)
      if (DVI->isAddressOfVariable() || DVI->getExpression()->startsWithDeref())
        DVI->eraseFromParent();

    return eraseInstFromFunction(MI);
  }
  return nullptr;
}

/// Move the call to free before a NULL test.
///
/// Check if this free is accessed after its argument has been test
/// against NULL (property 0).
/// If yes, it is legal to move this call in its predecessor block.
///
/// The move is performed only if the block containing the call to free
/// will be removed, i.e.:
/// 1. it has only one predecessor P, and P has two successors
/// 2. it contains the call, noops, and an unconditional branch
/// 3. its successor is the same as its predecessor's successor
///
/// The profitability is out-of concern here and this function should
/// be called only if the caller knows this transformation would be
/// profitable (e.g., for code size).
static Instruction *tryToMoveFreeBeforeNullTest(CallInst &FI,
                                                const DataLayout &DL) {
  Value *Op = FI.getArgOperand(0);
  BasicBlock *FreeInstrBB = FI.getParent();
  BasicBlock *PredBB = FreeInstrBB->getSinglePredecessor();

  // Validate part of constraint #1: Only one predecessor
  // FIXME: We can extend the number of predecessor, but in that case, we
  //        would duplicate the call to free in each predecessor and it may
  //        not be profitable even for code size.
  if (!PredBB)
    return nullptr;

  // Validate constraint #2: Does this block contains only the call to
  //                         free, noops, and an unconditional branch?
  BasicBlock *SuccBB;
  Instruction *FreeInstrBBTerminator = FreeInstrBB->getTerminator();
  if (!match(FreeInstrBBTerminator, m_UnconditionalBr(SuccBB)))
    return nullptr;

  // If there are only 2 instructions in the block, at this point,
  // this is the call to free and unconditional.
  // If there are more than 2 instructions, check that they are noops
  // i.e., they won't hurt the performance of the generated code.
  if (FreeInstrBB->size() != 2) {
    for (const Instruction &Inst : FreeInstrBB->instructionsWithoutDebug()) {
      if (&Inst == &FI || &Inst == FreeInstrBBTerminator)
        continue;
      auto *Cast = dyn_cast<CastInst>(&Inst);
      if (!Cast || !Cast->isNoopCast(DL))
        return nullptr;
    }
  }
  // Validate the rest of constraint #1 by matching on the pred branch.
  Instruction *TI = PredBB->getTerminator();
  BasicBlock *TrueBB, *FalseBB;
  ICmpInst::Predicate Pred;
  if (!match(TI, m_Br(m_ICmp(Pred,
                             m_CombineOr(m_Specific(Op),
                                         m_Specific(Op->stripPointerCasts())),
                             m_Zero()),
                      TrueBB, FalseBB)))
    return nullptr;
  if (Pred != ICmpInst::ICMP_EQ && Pred != ICmpInst::ICMP_NE)
    return nullptr;

  // Validate constraint #3: Ensure the null case just falls through.
  if (SuccBB != (Pred == ICmpInst::ICMP_EQ ? TrueBB : FalseBB))
    return nullptr;
  assert(FreeInstrBB == (Pred == ICmpInst::ICMP_EQ ? FalseBB : TrueBB) &&
         "Broken CFG: missing edge from predecessor to successor");

  // At this point, we know that everything in FreeInstrBB can be moved
  // before TI.
  for (Instruction &Instr : llvm::make_early_inc_range(*FreeInstrBB)) {
    if (&Instr == FreeInstrBBTerminator)
      break;
    Instr.moveBefore(TI);
  }
  assert(FreeInstrBB->size() == 1 &&
         "Only the branch instruction should remain");

  // Now that we've moved the call to free before the NULL check, we have to
  // remove any attributes on its parameter that imply it's non-null, because
  // those attributes might have only been valid because of the NULL check, and
  // we can get miscompiles if we keep them. This is conservative if non-null is
  // also implied by something other than the NULL check, but it's guaranteed to
  // be correct, and the conservativeness won't matter in practice, since the
  // attributes are irrelevant for the call to free itself and the pointer
  // shouldn't be used after the call.
  AttributeList Attrs = FI.getAttributes();
  Attrs = Attrs.removeParamAttribute(FI.getContext(), 0, Attribute::NonNull);
  Attribute Dereferenceable = Attrs.getParamAttr(0, Attribute::Dereferenceable);
  if (Dereferenceable.isValid()) {
    uint64_t Bytes = Dereferenceable.getDereferenceableBytes();
    Attrs = Attrs.removeParamAttribute(FI.getContext(), 0,
                                       Attribute::Dereferenceable);
    Attrs = Attrs.addDereferenceableOrNullParamAttr(FI.getContext(), 0, Bytes);
  }
  FI.setAttributes(Attrs);

  return &FI;
}

Instruction *InstCombinerImpl::visitFree(CallInst &FI) {
  Value *Op = FI.getArgOperand(0);

  // free undef -> unreachable.
  if (isa<UndefValue>(Op)) {
    // Leave a marker since we can't modify the CFG here.
    CreateNonTerminatorUnreachable(&FI);
    return eraseInstFromFunction(FI);
  }

  // If we have 'free null' delete the instruction.  This can happen in stl code
  // when lots of inlining happens.
  if (isa<ConstantPointerNull>(Op))
    return eraseInstFromFunction(FI);

  // If we had free(realloc(...)) with no intervening uses, then eliminate the
  // realloc() entirely.
  if (CallInst *CI = dyn_cast<CallInst>(Op)) {
    if (CI->hasOneUse() && isReallocLikeFn(CI, &TLI)) {
      return eraseInstFromFunction(
          *replaceInstUsesWith(*CI, CI->getOperand(0)));
    }
  }

  // If we optimize for code size, try to move the call to free before the null
  // test so that simplify cfg can remove the empty block and dead code
  // elimination the branch. I.e., helps to turn something like:
  // if (foo) free(foo);
  // into
  // free(foo);
  //
  // Note that we can only do this for 'free' and not for any flavor of
  // 'operator delete'; there is no 'operator delete' symbol for which we are
  // permitted to invent a call, even if we're passing in a null pointer.
  if (MinimizeSize) {
    LibFunc Func;
    if (TLI.getLibFunc(FI, Func) && TLI.has(Func) && Func == LibFunc_free)
      if (Instruction *I = tryToMoveFreeBeforeNullTest(FI, DL))
        return I;
  }

  return nullptr;
}

static bool isMustTailCall(Value *V) {
  if (auto *CI = dyn_cast<CallInst>(V))
    return CI->isMustTailCall();
  return false;
}

Instruction *InstCombinerImpl::visitReturnInst(ReturnInst &RI) {
  if (RI.getNumOperands() == 0) // ret void
    return nullptr;

  Value *ResultOp = RI.getOperand(0);
  Type *VTy = ResultOp->getType();
  if (!VTy->isIntegerTy() || isa<Constant>(ResultOp))
    return nullptr;

  // Don't replace result of musttail calls.
  if (isMustTailCall(ResultOp))
    return nullptr;

  // There might be assume intrinsics dominating this return that completely
  // determine the value. If so, constant fold it.
  KnownBits Known = computeKnownBits(ResultOp, 0, &RI);
  if (Known.isConstant())
    return replaceOperand(RI, 0,
        Constant::getIntegerValue(VTy, Known.getConstant()));

  return nullptr;
}

// WARNING: keep in sync with SimplifyCFGOpt::simplifyUnreachable()!
Instruction *InstCombinerImpl::visitUnreachableInst(UnreachableInst &I) {
  // Try to remove the previous instruction if it must lead to unreachable.
  // This includes instructions like stores and "llvm.assume" that may not get
  // removed by simple dead code elimination.
  while (Instruction *Prev = I.getPrevNonDebugInstruction()) {
    // While we theoretically can erase EH, that would result in a block that
    // used to start with an EH no longer starting with EH, which is invalid.
    // To make it valid, we'd need to fixup predecessors to no longer refer to
    // this block, but that changes CFG, which is not allowed in InstCombine.
    if (Prev->isEHPad())
      return nullptr; // Can not drop any more instructions. We're done here.

    if (!isGuaranteedToTransferExecutionToSuccessor(Prev))
      return nullptr; // Can not drop any more instructions. We're done here.
    // Otherwise, this instruction can be freely erased,
    // even if it is not side-effect free.

    // A value may still have uses before we process it here (for example, in
    // another unreachable block), so convert those to poison.
    replaceInstUsesWith(*Prev, PoisonValue::get(Prev->getType()));
    eraseInstFromFunction(*Prev);
  }
  assert(I.getParent()->sizeWithoutDebug() == 1 && "The block is now empty.");
  // FIXME: recurse into unconditional predecessors?
  return nullptr;
}

Instruction *InstCombinerImpl::visitUnconditionalBranchInst(BranchInst &BI) {
  assert(BI.isUnconditional() && "Only for unconditional branches.");

  // If this store is the second-to-last instruction in the basic block
  // (excluding debug info and bitcasts of pointers) and if the block ends with
  // an unconditional branch, try to move the store to the successor block.

  auto GetLastSinkableStore = [](BasicBlock::iterator BBI) {
    auto IsNoopInstrForStoreMerging = [](BasicBlock::iterator BBI) {
      return BBI->isDebugOrPseudoInst() ||
             (isa<BitCastInst>(BBI) && BBI->getType()->isPointerTy());
    };

    BasicBlock::iterator FirstInstr = BBI->getParent()->begin();
    do {
      if (BBI != FirstInstr)
        --BBI;
    } while (BBI != FirstInstr && IsNoopInstrForStoreMerging(BBI));

    return dyn_cast<StoreInst>(BBI);
  };

  if (StoreInst *SI = GetLastSinkableStore(BasicBlock::iterator(BI)))
    if (mergeStoreIntoSuccessor(*SI))
      return &BI;

  return nullptr;
}

Instruction *InstCombinerImpl::visitBranchInst(BranchInst &BI) {
  if (BI.isUnconditional())
    return visitUnconditionalBranchInst(BI);

  // Change br (not X), label True, label False to: br X, label False, True
  Value *X = nullptr;
  if (match(&BI, m_Br(m_Not(m_Value(X)), m_BasicBlock(), m_BasicBlock())) &&
      !isa<Constant>(X)) {
    // Swap Destinations and condition...
    BI.swapSuccessors();
    return replaceOperand(BI, 0, X);
  }

  // If the condition is irrelevant, remove the use so that other
  // transforms on the condition become more effective.
  if (!isa<ConstantInt>(BI.getCondition()) &&
      BI.getSuccessor(0) == BI.getSuccessor(1))
    return replaceOperand(
        BI, 0, ConstantInt::getFalse(BI.getCondition()->getType()));

  // Canonicalize, for example, fcmp_one -> fcmp_oeq.
  CmpInst::Predicate Pred;
  if (match(&BI, m_Br(m_OneUse(m_FCmp(Pred, m_Value(), m_Value())),
                      m_BasicBlock(), m_BasicBlock())) &&
      !isCanonicalPredicate(Pred)) {
    // Swap destinations and condition.
    CmpInst *Cond = cast<CmpInst>(BI.getCondition());
    Cond->setPredicate(CmpInst::getInversePredicate(Pred));
    BI.swapSuccessors();
    Worklist.push(Cond);
    return &BI;
  }

  return nullptr;
}

Instruction *InstCombinerImpl::visitSwitchInst(SwitchInst &SI) {
  Value *Cond = SI.getCondition();
  Value *Op0;
  ConstantInt *AddRHS;
  if (match(Cond, m_Add(m_Value(Op0), m_ConstantInt(AddRHS)))) {
    // Change 'switch (X+4) case 1:' into 'switch (X) case -3'.
    for (auto Case : SI.cases()) {
      Constant *NewCase = ConstantExpr::getSub(Case.getCaseValue(), AddRHS);
      assert(isa<ConstantInt>(NewCase) &&
             "Result of expression should be constant");
      Case.setValue(cast<ConstantInt>(NewCase));
    }
    return replaceOperand(SI, 0, Op0);
  }

  KnownBits Known = computeKnownBits(Cond, 0, &SI);
  unsigned LeadingKnownZeros = Known.countMinLeadingZeros();
  unsigned LeadingKnownOnes = Known.countMinLeadingOnes();

  // Compute the number of leading bits we can ignore.
  // TODO: A better way to determine this would use ComputeNumSignBits().
  for (auto &C : SI.cases()) {
    LeadingKnownZeros = std::min(
        LeadingKnownZeros, C.getCaseValue()->getValue().countLeadingZeros());
    LeadingKnownOnes = std::min(
        LeadingKnownOnes, C.getCaseValue()->getValue().countLeadingOnes());
  }

  unsigned NewWidth = Known.getBitWidth() - std::max(LeadingKnownZeros, LeadingKnownOnes);

  // Shrink the condition operand if the new type is smaller than the old type.
  // But do not shrink to a non-standard type, because backend can't generate
  // good code for that yet.
  // TODO: We can make it aggressive again after fixing PR39569.
  if (NewWidth > 0 && NewWidth < Known.getBitWidth() &&
      shouldChangeType(Known.getBitWidth(), NewWidth)) {
    IntegerType *Ty = IntegerType::get(SI.getContext(), NewWidth);
    Builder.SetInsertPoint(&SI);
    Value *NewCond = Builder.CreateTrunc(Cond, Ty, "trunc");

    for (auto Case : SI.cases()) {
      APInt TruncatedCase = Case.getCaseValue()->getValue().trunc(NewWidth);
      Case.setValue(ConstantInt::get(SI.getContext(), TruncatedCase));
    }
    return replaceOperand(SI, 0, NewCond);
  }

  return nullptr;
}

Instruction *InstCombinerImpl::visitExtractValueInst(ExtractValueInst &EV) {
  Value *Agg = EV.getAggregateOperand();

  if (!EV.hasIndices())
    return replaceInstUsesWith(EV, Agg);

  if (Value *V = SimplifyExtractValueInst(Agg, EV.getIndices(),
                                          SQ.getWithInstruction(&EV)))
    return replaceInstUsesWith(EV, V);

  if (InsertValueInst *IV = dyn_cast<InsertValueInst>(Agg)) {
    // We're extracting from an insertvalue instruction, compare the indices
    const unsigned *exti, *exte, *insi, *inse;
    for (exti = EV.idx_begin(), insi = IV->idx_begin(),
         exte = EV.idx_end(), inse = IV->idx_end();
         exti != exte && insi != inse;
         ++exti, ++insi) {
      if (*insi != *exti)
        // The insert and extract both reference distinctly different elements.
        // This means the extract is not influenced by the insert, and we can
        // replace the aggregate operand of the extract with the aggregate
        // operand of the insert. i.e., replace
        // %I = insertvalue { i32, { i32 } } %A, { i32 } { i32 42 }, 1
        // %E = extractvalue { i32, { i32 } } %I, 0
        // with
        // %E = extractvalue { i32, { i32 } } %A, 0
        return ExtractValueInst::Create(IV->getAggregateOperand(),
                                        EV.getIndices());
    }
    if (exti == exte && insi == inse)
      // Both iterators are at the end: Index lists are identical. Replace
      // %B = insertvalue { i32, { i32 } } %A, i32 42, 1, 0
      // %C = extractvalue { i32, { i32 } } %B, 1, 0
      // with "i32 42"
      return replaceInstUsesWith(EV, IV->getInsertedValueOperand());
    if (exti == exte) {
      // The extract list is a prefix of the insert list. i.e. replace
      // %I = insertvalue { i32, { i32 } } %A, i32 42, 1, 0
      // %E = extractvalue { i32, { i32 } } %I, 1
      // with
      // %X = extractvalue { i32, { i32 } } %A, 1
      // %E = insertvalue { i32 } %X, i32 42, 0
      // by switching the order of the insert and extract (though the
      // insertvalue should be left in, since it may have other uses).
      Value *NewEV = Builder.CreateExtractValue(IV->getAggregateOperand(),
                                                EV.getIndices());
      return InsertValueInst::Create(NewEV, IV->getInsertedValueOperand(),
                                     makeArrayRef(insi, inse));
    }
    if (insi == inse)
      // The insert list is a prefix of the extract list
      // We can simply remove the common indices from the extract and make it
      // operate on the inserted value instead of the insertvalue result.
      // i.e., replace
      // %I = insertvalue { i32, { i32 } } %A, { i32 } { i32 42 }, 1
      // %E = extractvalue { i32, { i32 } } %I, 1, 0
      // with
      // %E extractvalue { i32 } { i32 42 }, 0
      return ExtractValueInst::Create(IV->getInsertedValueOperand(),
                                      makeArrayRef(exti, exte));
  }
  if (WithOverflowInst *WO = dyn_cast<WithOverflowInst>(Agg)) {
    // We're extracting from an overflow intrinsic, see if we're the only user,
    // which allows us to simplify multiple result intrinsics to simpler
    // things that just get one value.
    if (WO->hasOneUse()) {
      // Check if we're grabbing only the result of a 'with overflow' intrinsic
      // and replace it with a traditional binary instruction.
      if (*EV.idx_begin() == 0) {
        Instruction::BinaryOps BinOp = WO->getBinaryOp();
        Value *LHS = WO->getLHS(), *RHS = WO->getRHS();
        // Replace the old instruction's uses with poison.
        replaceInstUsesWith(*WO, PoisonValue::get(WO->getType()));
        eraseInstFromFunction(*WO);
        return BinaryOperator::Create(BinOp, LHS, RHS);
      }

      assert(*EV.idx_begin() == 1 &&
             "unexpected extract index for overflow inst");

      // If only the overflow result is used, and the right hand side is a
      // constant (or constant splat), we can remove the intrinsic by directly
      // checking for overflow.
      const APInt *C;
      if (match(WO->getRHS(), m_APInt(C))) {
        // Compute the no-wrap range for LHS given RHS=C, then construct an
        // equivalent icmp, potentially using an offset.
        ConstantRange NWR =
          ConstantRange::makeExactNoWrapRegion(WO->getBinaryOp(), *C,
                                               WO->getNoWrapKind());

        CmpInst::Predicate Pred;
        APInt NewRHSC, Offset;
        NWR.getEquivalentICmp(Pred, NewRHSC, Offset);
        auto *OpTy = WO->getRHS()->getType();
        auto *NewLHS = WO->getLHS();
        if (Offset != 0)
          NewLHS = Builder.CreateAdd(NewLHS, ConstantInt::get(OpTy, Offset));
        return new ICmpInst(ICmpInst::getInversePredicate(Pred), NewLHS,
                            ConstantInt::get(OpTy, NewRHSC));
      }
    }
  }
  if (LoadInst *L = dyn_cast<LoadInst>(Agg))
    // If the (non-volatile) load only has one use, we can rewrite this to a
    // load from a GEP. This reduces the size of the load. If a load is used
    // only by extractvalue instructions then this either must have been
    // optimized before, or it is a struct with padding, in which case we
    // don't want to do the transformation as it loses padding knowledge.
    if (L->isSimple() && L->hasOneUse()) {
      // extractvalue has integer indices, getelementptr has Value*s. Convert.
      SmallVector<Value*, 4> Indices;
      // Prefix an i32 0 since we need the first element.
      Indices.push_back(Builder.getInt32(0));
      for (unsigned Idx : EV.indices())
        Indices.push_back(Builder.getInt32(Idx));

      // We need to insert these at the location of the old load, not at that of
      // the extractvalue.
      Builder.SetInsertPoint(L);
      Value *GEP = Builder.CreateInBoundsGEP(L->getType(),
                                             L->getPointerOperand(), Indices);
      Instruction *NL = Builder.CreateLoad(EV.getType(), GEP);
      // Whatever aliasing information we had for the orignal load must also
      // hold for the smaller load, so propagate the annotations.
      NL->setAAMetadata(L->getAAMetadata());
      // Returning the load directly will cause the main loop to insert it in
      // the wrong spot, so use replaceInstUsesWith().
      return replaceInstUsesWith(EV, NL);
    }
  // We could simplify extracts from other values. Note that nested extracts may
  // already be simplified implicitly by the above: extract (extract (insert) )
  // will be translated into extract ( insert ( extract ) ) first and then just
  // the value inserted, if appropriate. Similarly for extracts from single-use
  // loads: extract (extract (load)) will be translated to extract (load (gep))
  // and if again single-use then via load (gep (gep)) to load (gep).
  // However, double extracts from e.g. function arguments or return values
  // aren't handled yet.
  return nullptr;
}

/// Return 'true' if the given typeinfo will match anything.
static bool isCatchAll(EHPersonality Personality, Constant *TypeInfo) {
  switch (Personality) {
  case EHPersonality::GNU_C:
  case EHPersonality::GNU_C_SjLj:
  case EHPersonality::Rust:
    // The GCC C EH and Rust personality only exists to support cleanups, so
    // it's not clear what the semantics of catch clauses are.
    return false;
  case EHPersonality::Unknown:
    return false;
  case EHPersonality::GNU_Ada:
    // While __gnat_all_others_value will match any Ada exception, it doesn't
    // match foreign exceptions (or didn't, before gcc-4.7).
    return false;
  case EHPersonality::GNU_CXX:
  case EHPersonality::GNU_CXX_SjLj:
  case EHPersonality::GNU_ObjC:
  case EHPersonality::MSVC_X86SEH:
  case EHPersonality::MSVC_TableSEH:
  case EHPersonality::MSVC_CXX:
  case EHPersonality::CoreCLR:
  case EHPersonality::Wasm_CXX:
  case EHPersonality::XL_CXX:
    return TypeInfo->isNullValue();
  }
  llvm_unreachable("invalid enum");
}

static bool shorter_filter(const Value *LHS, const Value *RHS) {
  return
    cast<ArrayType>(LHS->getType())->getNumElements()
  <
    cast<ArrayType>(RHS->getType())->getNumElements();
}

Instruction *InstCombinerImpl::visitLandingPadInst(LandingPadInst &LI) {
  // The logic here should be correct for any real-world personality function.
  // However if that turns out not to be true, the offending logic can always
  // be conditioned on the personality function, like the catch-all logic is.
  EHPersonality Personality =
      classifyEHPersonality(LI.getParent()->getParent()->getPersonalityFn());

  // Simplify the list of clauses, eg by removing repeated catch clauses
  // (these are often created by inlining).
  bool MakeNewInstruction = false; // If true, recreate using the following:
  SmallVector<Constant *, 16> NewClauses; // - Clauses for the new instruction;
  bool CleanupFlag = LI.isCleanup();   // - The new instruction is a cleanup.

  SmallPtrSet<Value *, 16> AlreadyCaught; // Typeinfos known caught already.
  for (unsigned i = 0, e = LI.getNumClauses(); i != e; ++i) {
    bool isLastClause = i + 1 == e;
    if (LI.isCatch(i)) {
      // A catch clause.
      Constant *CatchClause = LI.getClause(i);
      Constant *TypeInfo = CatchClause->stripPointerCasts();

      // If we already saw this clause, there is no point in having a second
      // copy of it.
      if (AlreadyCaught.insert(TypeInfo).second) {
        // This catch clause was not already seen.
        NewClauses.push_back(CatchClause);
      } else {
        // Repeated catch clause - drop the redundant copy.
        MakeNewInstruction = true;
      }

      // If this is a catch-all then there is no point in keeping any following
      // clauses or marking the landingpad as having a cleanup.
      if (isCatchAll(Personality, TypeInfo)) {
        if (!isLastClause)
          MakeNewInstruction = true;
        CleanupFlag = false;
        break;
      }
    } else {
      // A filter clause.  If any of the filter elements were already caught
      // then they can be dropped from the filter.  It is tempting to try to
      // exploit the filter further by saying that any typeinfo that does not
      // occur in the filter can't be caught later (and thus can be dropped).
      // However this would be wrong, since typeinfos can match without being
      // equal (for example if one represents a C++ class, and the other some
      // class derived from it).
      assert(LI.isFilter(i) && "Unsupported landingpad clause!");
      Constant *FilterClause = LI.getClause(i);
      ArrayType *FilterType = cast<ArrayType>(FilterClause->getType());
      unsigned NumTypeInfos = FilterType->getNumElements();

      // An empty filter catches everything, so there is no point in keeping any
      // following clauses or marking the landingpad as having a cleanup.  By
      // dealing with this case here the following code is made a bit simpler.
      if (!NumTypeInfos) {
        NewClauses.push_back(FilterClause);
        if (!isLastClause)
          MakeNewInstruction = true;
        CleanupFlag = false;
        break;
      }

      bool MakeNewFilter = false; // If true, make a new filter.
      SmallVector<Constant *, 16> NewFilterElts; // New elements.
      if (isa<ConstantAggregateZero>(FilterClause)) {
        // Not an empty filter - it contains at least one null typeinfo.
        assert(NumTypeInfos > 0 && "Should have handled empty filter already!");
        Constant *TypeInfo =
          Constant::getNullValue(FilterType->getElementType());
        // If this typeinfo is a catch-all then the filter can never match.
        if (isCatchAll(Personality, TypeInfo)) {
          // Throw the filter away.
          MakeNewInstruction = true;
          continue;
        }

        // There is no point in having multiple copies of this typeinfo, so
        // discard all but the first copy if there is more than one.
        NewFilterElts.push_back(TypeInfo);
        if (NumTypeInfos > 1)
          MakeNewFilter = true;
      } else {
        ConstantArray *Filter = cast<ConstantArray>(FilterClause);
        SmallPtrSet<Value *, 16> SeenInFilter; // For uniquing the elements.
        NewFilterElts.reserve(NumTypeInfos);

        // Remove any filter elements that were already caught or that already
        // occurred in the filter.  While there, see if any of the elements are
        // catch-alls.  If so, the filter can be discarded.
        bool SawCatchAll = false;
        for (unsigned j = 0; j != NumTypeInfos; ++j) {
          Constant *Elt = Filter->getOperand(j);
          Constant *TypeInfo = Elt->stripPointerCasts();
          if (isCatchAll(Personality, TypeInfo)) {
            // This element is a catch-all.  Bail out, noting this fact.
            SawCatchAll = true;
            break;
          }

          // Even if we've seen a type in a catch clause, we don't want to
          // remove it from the filter.  An unexpected type handler may be
          // set up for a call site which throws an exception of the same
          // type caught.  In order for the exception thrown by the unexpected
          // handler to propagate correctly, the filter must be correctly
          // described for the call site.
          //
          // Example:
          //
          // void unexpected() { throw 1;}
          // void foo() throw (int) {
          //   std::set_unexpected(unexpected);
          //   try {
          //     throw 2.0;
          //   } catch (int i) {}
          // }

          // There is no point in having multiple copies of the same typeinfo in
          // a filter, so only add it if we didn't already.
          if (SeenInFilter.insert(TypeInfo).second)
            NewFilterElts.push_back(cast<Constant>(Elt));
        }
        // A filter containing a catch-all cannot match anything by definition.
        if (SawCatchAll) {
          // Throw the filter away.
          MakeNewInstruction = true;
          continue;
        }

        // If we dropped something from the filter, make a new one.
        if (NewFilterElts.size() < NumTypeInfos)
          MakeNewFilter = true;
      }
      if (MakeNewFilter) {
        FilterType = ArrayType::get(FilterType->getElementType(),
                                    NewFilterElts.size());
        FilterClause = ConstantArray::get(FilterType, NewFilterElts);
        MakeNewInstruction = true;
      }

      NewClauses.push_back(FilterClause);

      // If the new filter is empty then it will catch everything so there is
      // no point in keeping any following clauses or marking the landingpad
      // as having a cleanup.  The case of the original filter being empty was
      // already handled above.
      if (MakeNewFilter && !NewFilterElts.size()) {
        assert(MakeNewInstruction && "New filter but not a new instruction!");
        CleanupFlag = false;
        break;
      }
    }
  }

  // If several filters occur in a row then reorder them so that the shortest
  // filters come first (those with the smallest number of elements).  This is
  // advantageous because shorter filters are more likely to match, speeding up
  // unwinding, but mostly because it increases the effectiveness of the other
  // filter optimizations below.
  for (unsigned i = 0, e = NewClauses.size(); i + 1 < e; ) {
    unsigned j;
    // Find the maximal 'j' s.t. the range [i, j) consists entirely of filters.
    for (j = i; j != e; ++j)
      if (!isa<ArrayType>(NewClauses[j]->getType()))
        break;

    // Check whether the filters are already sorted by length.  We need to know
    // if sorting them is actually going to do anything so that we only make a
    // new landingpad instruction if it does.
    for (unsigned k = i; k + 1 < j; ++k)
      if (shorter_filter(NewClauses[k+1], NewClauses[k])) {
        // Not sorted, so sort the filters now.  Doing an unstable sort would be
        // correct too but reordering filters pointlessly might confuse users.
        std::stable_sort(NewClauses.begin() + i, NewClauses.begin() + j,
                         shorter_filter);
        MakeNewInstruction = true;
        break;
      }

    // Look for the next batch of filters.
    i = j + 1;
  }

  // If typeinfos matched if and only if equal, then the elements of a filter L
  // that occurs later than a filter F could be replaced by the intersection of
  // the elements of F and L.  In reality two typeinfos can match without being
  // equal (for example if one represents a C++ class, and the other some class
  // derived from it) so it would be wrong to perform this transform in general.
  // However the transform is correct and useful if F is a subset of L.  In that
  // case L can be replaced by F, and thus removed altogether since repeating a
  // filter is pointless.  So here we look at all pairs of filters F and L where
  // L follows F in the list of clauses, and remove L if every element of F is
  // an element of L.  This can occur when inlining C++ functions with exception
  // specifications.
  for (unsigned i = 0; i + 1 < NewClauses.size(); ++i) {
    // Examine each filter in turn.
    Value *Filter = NewClauses[i];
    ArrayType *FTy = dyn_cast<ArrayType>(Filter->getType());
    if (!FTy)
      // Not a filter - skip it.
      continue;
    unsigned FElts = FTy->getNumElements();
    // Examine each filter following this one.  Doing this backwards means that
    // we don't have to worry about filters disappearing under us when removed.
    for (unsigned j = NewClauses.size() - 1; j != i; --j) {
      Value *LFilter = NewClauses[j];
      ArrayType *LTy = dyn_cast<ArrayType>(LFilter->getType());
      if (!LTy)
        // Not a filter - skip it.
        continue;
      // If Filter is a subset of LFilter, i.e. every element of Filter is also
      // an element of LFilter, then discard LFilter.
      SmallVectorImpl<Constant *>::iterator J = NewClauses.begin() + j;
      // If Filter is empty then it is a subset of LFilter.
      if (!FElts) {
        // Discard LFilter.
        NewClauses.erase(J);
        MakeNewInstruction = true;
        // Move on to the next filter.
        continue;
      }
      unsigned LElts = LTy->getNumElements();
      // If Filter is longer than LFilter then it cannot be a subset of it.
      if (FElts > LElts)
        // Move on to the next filter.
        continue;
      // At this point we know that LFilter has at least one element.
      if (isa<ConstantAggregateZero>(LFilter)) { // LFilter only contains zeros.
        // Filter is a subset of LFilter iff Filter contains only zeros (as we
        // already know that Filter is not longer than LFilter).
        if (isa<ConstantAggregateZero>(Filter)) {
          assert(FElts <= LElts && "Should have handled this case earlier!");
          // Discard LFilter.
          NewClauses.erase(J);
          MakeNewInstruction = true;
        }
        // Move on to the next filter.
        continue;
      }
      ConstantArray *LArray = cast<ConstantArray>(LFilter);
      if (isa<ConstantAggregateZero>(Filter)) { // Filter only contains zeros.
        // Since Filter is non-empty and contains only zeros, it is a subset of
        // LFilter iff LFilter contains a zero.
        assert(FElts > 0 && "Should have eliminated the empty filter earlier!");
        for (unsigned l = 0; l != LElts; ++l)
          if (LArray->getOperand(l)->isNullValue()) {
            // LFilter contains a zero - discard it.
            NewClauses.erase(J);
            MakeNewInstruction = true;
            break;
          }
        // Move on to the next filter.
        continue;
      }
      // At this point we know that both filters are ConstantArrays.  Loop over
      // operands to see whether every element of Filter is also an element of
      // LFilter.  Since filters tend to be short this is probably faster than
      // using a method that scales nicely.
      ConstantArray *FArray = cast<ConstantArray>(Filter);
      bool AllFound = true;
      for (unsigned f = 0; f != FElts; ++f) {
        Value *FTypeInfo = FArray->getOperand(f)->stripPointerCasts();
        AllFound = false;
        for (unsigned l = 0; l != LElts; ++l) {
          Value *LTypeInfo = LArray->getOperand(l)->stripPointerCasts();
          if (LTypeInfo == FTypeInfo) {
            AllFound = true;
            break;
          }
        }
        if (!AllFound)
          break;
      }
      if (AllFound) {
        // Discard LFilter.
        NewClauses.erase(J);
        MakeNewInstruction = true;
      }
      // Move on to the next filter.
    }
  }

  // If we changed any of the clauses, replace the old landingpad instruction
  // with a new one.
  if (MakeNewInstruction) {
    LandingPadInst *NLI = LandingPadInst::Create(LI.getType(),
                                                 NewClauses.size());
    for (unsigned i = 0, e = NewClauses.size(); i != e; ++i)
      NLI->addClause(NewClauses[i]);
    // A landing pad with no clauses must have the cleanup flag set.  It is
    // theoretically possible, though highly unlikely, that we eliminated all
    // clauses.  If so, force the cleanup flag to true.
    if (NewClauses.empty())
      CleanupFlag = true;
    NLI->setCleanup(CleanupFlag);
    return NLI;
  }

  // Even if none of the clauses changed, we may nonetheless have understood
  // that the cleanup flag is pointless.  Clear it if so.
  if (LI.isCleanup() != CleanupFlag) {
    assert(!CleanupFlag && "Adding a cleanup, not removing one?!");
    LI.setCleanup(CleanupFlag);
    return &LI;
  }

  return nullptr;
}

Value *
InstCombinerImpl::pushFreezeToPreventPoisonFromPropagating(FreezeInst &OrigFI) {
  // Try to push freeze through instructions that propagate but don't produce
  // poison as far as possible.  If an operand of freeze follows three
  // conditions 1) one-use, 2) does not produce poison, and 3) has all but one
  // guaranteed-non-poison operands then push the freeze through to the one
  // operand that is not guaranteed non-poison.  The actual transform is as
  // follows.
  //   Op1 = ...                        ; Op1 can be posion
  //   Op0 = Inst(Op1, NonPoisonOps...) ; Op0 has only one use and only have
  //                                    ; single guaranteed-non-poison operands
  //   ... = Freeze(Op0)
  // =>
  //   Op1 = ...
  //   Op1.fr = Freeze(Op1)
  //   ... = Inst(Op1.fr, NonPoisonOps...)
  auto *OrigOp = OrigFI.getOperand(0);
  auto *OrigOpInst = dyn_cast<Instruction>(OrigOp);

  // While we could change the other users of OrigOp to use freeze(OrigOp), that
  // potentially reduces their optimization potential, so let's only do this iff
  // the OrigOp is only used by the freeze.
  if (!OrigOpInst || !OrigOpInst->hasOneUse() || isa<PHINode>(OrigOp))
    return nullptr;

  // We can't push the freeze through an instruction which can itself create
  // poison.  If the only source of new poison is flags, we can simply
  // strip them (since we know the only use is the freeze and nothing can
  // benefit from them.)
  if (canCreateUndefOrPoison(cast<Operator>(OrigOp), /*ConsiderFlags*/ false))
    return nullptr;

  // If operand is guaranteed not to be poison, there is no need to add freeze
  // to the operand. So we first find the operand that is not guaranteed to be
  // poison.
  Use *MaybePoisonOperand = nullptr;
  for (Use &U : OrigOpInst->operands()) {
    if (isGuaranteedNotToBeUndefOrPoison(U.get()))
      continue;
    if (!MaybePoisonOperand)
      MaybePoisonOperand = &U;
    else
      return nullptr;
  }

  OrigOpInst->dropPoisonGeneratingFlags();

  // If all operands are guaranteed to be non-poison, we can drop freeze.
  if (!MaybePoisonOperand)
    return OrigOp;

  auto *FrozenMaybePoisonOperand = new FreezeInst(
      MaybePoisonOperand->get(), MaybePoisonOperand->get()->getName() + ".fr");

  replaceUse(*MaybePoisonOperand, FrozenMaybePoisonOperand);
  FrozenMaybePoisonOperand->insertBefore(OrigOpInst);
  return OrigOp;
}

bool InstCombinerImpl::freezeDominatedUses(FreezeInst &FI) {
  Value *Op = FI.getOperand(0);

  if (isa<Constant>(Op))
    return false;

  bool Changed = false;
  Op->replaceUsesWithIf(&FI, [&](Use &U) -> bool {
    bool Dominates = DT.dominates(&FI, U);
    Changed |= Dominates;
    return Dominates;
  });

  return Changed;
}

Instruction *InstCombinerImpl::visitFreeze(FreezeInst &I) {
  Value *Op0 = I.getOperand(0);

  if (Value *V = SimplifyFreezeInst(Op0, SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  // freeze (phi const, x) --> phi const, (freeze x)
  if (auto *PN = dyn_cast<PHINode>(Op0)) {
    if (Instruction *NV = foldOpIntoPhi(I, PN))
      return NV;
  }

  if (Value *NI = pushFreezeToPreventPoisonFromPropagating(I))
    return replaceInstUsesWith(I, NI);

  if (match(Op0, m_Undef())) {
    // If I is freeze(undef), see its uses and fold it to the best constant.
    // - or: pick -1
    // - select's condition: pick the value that leads to choosing a constant
    // - other ops: pick 0
    Constant *BestValue = nullptr;
    Constant *NullValue = Constant::getNullValue(I.getType());
    for (const auto *U : I.users()) {
      Constant *C = NullValue;

      if (match(U, m_Or(m_Value(), m_Value())))
        C = Constant::getAllOnesValue(I.getType());
      else if (const auto *SI = dyn_cast<SelectInst>(U)) {
        if (SI->getCondition() == &I) {
          APInt CondVal(1, isa<Constant>(SI->getFalseValue()) ? 0 : 1);
          C = Constant::getIntegerValue(I.getType(), CondVal);
        }
      }

      if (!BestValue)
        BestValue = C;
      else if (BestValue != C)
        BestValue = NullValue;
    }

    return replaceInstUsesWith(I, BestValue);
  }

  // Replace all dominated uses of Op to freeze(Op).
  if (freezeDominatedUses(I))
    return &I;

  return nullptr;
}

/// Check for case where the call writes to an otherwise dead alloca.  This
/// shows up for unused out-params in idiomatic C/C++ code.   Note that this
/// helper *only* analyzes the write; doesn't check any other legality aspect.
static bool SoleWriteToDeadLocal(Instruction *I, TargetLibraryInfo &TLI) {
  auto *CB = dyn_cast<CallBase>(I);
  if (!CB)
    // TODO: handle e.g. store to alloca here - only worth doing if we extend
    // to allow reload along used path as described below.  Otherwise, this
    // is simply a store to a dead allocation which will be removed.
    return false;
  Optional<MemoryLocation> Dest = MemoryLocation::getForDest(CB, TLI);
  if (!Dest)
    return false;
  auto *AI = dyn_cast<AllocaInst>(getUnderlyingObject(Dest->Ptr));
  if (!AI)
    // TODO: allow malloc?
    return false;
  // TODO: allow memory access dominated by move point?  Note that since AI
  // could have a reference to itself captured by the call, we would need to
  // account for cycles in doing so.
  SmallVector<const User *> AllocaUsers;
  SmallPtrSet<const User *, 4> Visited;
  auto pushUsers = [&](const Instruction &I) {
    for (const User *U : I.users()) {
      if (Visited.insert(U).second)
        AllocaUsers.push_back(U);
    }
  };
  pushUsers(*AI);
  while (!AllocaUsers.empty()) {
    auto *UserI = cast<Instruction>(AllocaUsers.pop_back_val());
    if (isa<BitCastInst>(UserI) || isa<GetElementPtrInst>(UserI) ||
        isa<AddrSpaceCastInst>(UserI)) {
      pushUsers(*UserI);
      continue;
    }
    if (UserI == CB)
      continue;
    // TODO: support lifetime.start/end here
    return false;
  }
  return true;
}

/// Try to move the specified instruction from its current block into the
/// beginning of DestBlock, which can only happen if it's safe to move the
/// instruction past all of the instructions between it and the end of its
/// block.
static bool TryToSinkInstruction(Instruction *I, BasicBlock *DestBlock,
                                 TargetLibraryInfo &TLI) {
  assert(I->getUniqueUndroppableUser() && "Invariants didn't hold!");
  BasicBlock *SrcBlock = I->getParent();

  // Cannot move control-flow-involving, volatile loads, vaarg, etc.
  if (isa<PHINode>(I) || I->isEHPad() || I->mayThrow() || !I->willReturn() ||
      I->isTerminator())
    return false;

  // Do not sink static or dynamic alloca instructions. Static allocas must
  // remain in the entry block, and dynamic allocas must not be sunk in between
  // a stacksave / stackrestore pair, which would incorrectly shorten its
  // lifetime.
  if (isa<AllocaInst>(I))
    return false;

  // Do not sink into catchswitch blocks.
  if (isa<CatchSwitchInst>(DestBlock->getTerminator()))
    return false;

  // Do not sink convergent call instructions.
  if (auto *CI = dyn_cast<CallInst>(I)) {
    if (CI->isConvergent())
      return false;
  }

  // Unless we can prove that the memory write isn't visibile except on the
  // path we're sinking to, we must bail.
  if (I->mayWriteToMemory()) {
    if (!SoleWriteToDeadLocal(I, TLI))
      return false;
  }

  // We can only sink load instructions if there is nothing between the load and
  // the end of block that could change the value.
  if (I->mayReadFromMemory()) {
    // We don't want to do any sophisticated alias analysis, so we only check
    // the instructions after I in I's parent block if we try to sink to its
    // successor block.
    if (DestBlock->getUniquePredecessor() != I->getParent())
      return false;
    for (BasicBlock::iterator Scan = std::next(I->getIterator()),
                              E = I->getParent()->end();
         Scan != E; ++Scan)
      if (Scan->mayWriteToMemory())
        return false;
  }

  I->dropDroppableUses([DestBlock](const Use *U) {
    if (auto *I = dyn_cast<Instruction>(U->getUser()))
      return I->getParent() != DestBlock;
    return true;
  });
  /// FIXME: We could remove droppable uses that are not dominated by
  /// the new position.

  BasicBlock::iterator InsertPos = DestBlock->getFirstInsertionPt();
  I->moveBefore(&*InsertPos);
  ++NumSunkInst;

  // Also sink all related debug uses from the source basic block. Otherwise we
  // get debug use before the def. Attempt to salvage debug uses first, to
  // maximise the range variables have location for. If we cannot salvage, then
  // mark the location undef: we know it was supposed to receive a new location
  // here, but that computation has been sunk.
  SmallVector<DbgVariableIntrinsic *, 2> DbgUsers;
  findDbgUsers(DbgUsers, I);
  // Process the sinking DbgUsers in reverse order, as we only want to clone the
  // last appearing debug intrinsic for each given variable.
  SmallVector<DbgVariableIntrinsic *, 2> DbgUsersToSink;
  for (DbgVariableIntrinsic *DVI : DbgUsers)
    if (DVI->getParent() == SrcBlock)
      DbgUsersToSink.push_back(DVI);
  llvm::sort(DbgUsersToSink,
             [](auto *A, auto *B) { return B->comesBefore(A); });

  SmallVector<DbgVariableIntrinsic *, 2> DIIClones;
  SmallSet<DebugVariable, 4> SunkVariables;
  for (auto User : DbgUsersToSink) {
    // A dbg.declare instruction should not be cloned, since there can only be
    // one per variable fragment. It should be left in the original place
    // because the sunk instruction is not an alloca (otherwise we could not be
    // here).
    if (isa<DbgDeclareInst>(User))
      continue;

    DebugVariable DbgUserVariable =
        DebugVariable(User->getVariable(), User->getExpression(),
                      User->getDebugLoc()->getInlinedAt());

    if (!SunkVariables.insert(DbgUserVariable).second)
      continue;

    DIIClones.emplace_back(cast<DbgVariableIntrinsic>(User->clone()));
    if (isa<DbgDeclareInst>(User) && isa<CastInst>(I))
      DIIClones.back()->replaceVariableLocationOp(I, I->getOperand(0));
    LLVM_DEBUG(dbgs() << "CLONE: " << *DIIClones.back() << '\n');
  }

  // Perform salvaging without the clones, then sink the clones.
  if (!DIIClones.empty()) {
    salvageDebugInfoForDbgValues(*I, DbgUsers);
    // The clones are in reverse order of original appearance, reverse again to
    // maintain the original order.
    for (auto &DIIClone : llvm::reverse(DIIClones)) {
      DIIClone->insertBefore(&*InsertPos);
      LLVM_DEBUG(dbgs() << "SINK: " << *DIIClone << '\n');
    }
  }

  return true;
}

bool InstCombinerImpl::run() {
  while (!Worklist.isEmpty()) {
    // Walk deferred instructions in reverse order, and push them to the
    // worklist, which means they'll end up popped from the worklist in-order.
    while (Instruction *I = Worklist.popDeferred()) {
      // Check to see if we can DCE the instruction. We do this already here to
      // reduce the number of uses and thus allow other folds to trigger.
      // Note that eraseInstFromFunction() may push additional instructions on
      // the deferred worklist, so this will DCE whole instruction chains.
      if (isInstructionTriviallyDead(I, &TLI)) {
        eraseInstFromFunction(*I);
        ++NumDeadInst;
        continue;
      }

      Worklist.push(I);
    }

    Instruction *I = Worklist.removeOne();
    if (I == nullptr) continue;  // skip null values.

    // Check to see if we can DCE the instruction.
    if (isInstructionTriviallyDead(I, &TLI)) {
      eraseInstFromFunction(*I);
      ++NumDeadInst;
      continue;
    }

    if (!DebugCounter::shouldExecute(VisitCounter))
      continue;

    // Instruction isn't dead, see if we can constant propagate it.
    if (!I->use_empty() &&
        (I->getNumOperands() == 0 || isa<Constant>(I->getOperand(0)))) {
      if (Constant *C = ConstantFoldInstruction(I, DL, &TLI)) {
        LLVM_DEBUG(dbgs() << "IC: ConstFold to: " << *C << " from: " << *I
                          << '\n');

        // Add operands to the worklist.
        replaceInstUsesWith(*I, C);
        ++NumConstProp;
        if (isInstructionTriviallyDead(I, &TLI))
          eraseInstFromFunction(*I);
        MadeIRChange = true;
        continue;
      }
    }

    // See if we can trivially sink this instruction to its user if we can
    // prove that the successor is not executed more frequently than our block.
    // Return the UserBlock if successful.
    auto getOptionalSinkBlockForInst =
        [this](Instruction *I) -> Optional<BasicBlock *> {
      if (!EnableCodeSinking)
        return None;
      auto *UserInst = cast_or_null<Instruction>(I->getUniqueUndroppableUser());
      if (!UserInst)
        return None;

      BasicBlock *BB = I->getParent();
      BasicBlock *UserParent = nullptr;

      // Special handling for Phi nodes - get the block the use occurs in.
      if (PHINode *PN = dyn_cast<PHINode>(UserInst)) {
        for (unsigned i = 0; i < PN->getNumIncomingValues(); i++) {
          if (PN->getIncomingValue(i) == I) {
            // Bail out if we have uses in different blocks. We don't do any
            // sophisticated analysis (i.e finding NearestCommonDominator of these
            // use blocks).
            if (UserParent && UserParent != PN->getIncomingBlock(i))
              return None;
            UserParent = PN->getIncomingBlock(i);
          }
        }
        assert(UserParent && "expected to find user block!");
      } else
        UserParent = UserInst->getParent();

      // Try sinking to another block. If that block is unreachable, then do
      // not bother. SimplifyCFG should handle it.
      if (UserParent == BB || !DT.isReachableFromEntry(UserParent))
        return None;

      auto *Term = UserParent->getTerminator();
      // See if the user is one of our successors that has only one
      // predecessor, so that we don't have to split the critical edge.
      // Another option where we can sink is a block that ends with a
      // terminator that does not pass control to other block (such as
      // return or unreachable or resume). In this case:
      //   - I dominates the User (by SSA form);
      //   - the User will be executed at most once.
      // So sinking I down to User is always profitable or neutral.
      if (UserParent->getUniquePredecessor() == BB || succ_empty(Term)) {
        assert(DT.dominates(BB, UserParent) && "Dominance relation broken?");
        return UserParent;
      }
      return None;
    };

    auto OptBB = getOptionalSinkBlockForInst(I);
    if (OptBB) {
      auto *UserParent = *OptBB;
      // Okay, the CFG is simple enough, try to sink this instruction.
      if (TryToSinkInstruction(I, UserParent, TLI)) {
        LLVM_DEBUG(dbgs() << "IC: Sink: " << *I << '\n');
        MadeIRChange = true;
        // We'll add uses of the sunk instruction below, but since
        // sinking can expose opportunities for it's *operands* add
        // them to the worklist
        for (Use &U : I->operands())
          if (Instruction *OpI = dyn_cast<Instruction>(U.get()))
            Worklist.push(OpI);
      }
    }

    // Now that we have an instruction, try combining it to simplify it.
    Builder.SetInsertPoint(I);
    Builder.CollectMetadataToCopy(
        I, {LLVMContext::MD_dbg, LLVMContext::MD_annotation});

#ifndef NDEBUG
    std::string OrigI;
#endif
    LLVM_DEBUG(raw_string_ostream SS(OrigI); I->print(SS); OrigI = SS.str(););
    LLVM_DEBUG(dbgs() << "IC: Visiting: " << OrigI << '\n');

    if (Instruction *Result = visit(*I)) {
      ++NumCombined;
      // Should we replace the old instruction with a new one?
      if (Result != I) {
        LLVM_DEBUG(dbgs() << "IC: Old = " << *I << '\n'
                          << "    New = " << *Result << '\n');

        Result->copyMetadata(*I,
                             {LLVMContext::MD_dbg, LLVMContext::MD_annotation});
        // Everything uses the new instruction now.
        I->replaceAllUsesWith(Result);

        // Move the name to the new instruction first.
        Result->takeName(I);

        // Insert the new instruction into the basic block...
        BasicBlock *InstParent = I->getParent();
        BasicBlock::iterator InsertPos = I->getIterator();

        // Are we replace a PHI with something that isn't a PHI, or vice versa?
        if (isa<PHINode>(Result) != isa<PHINode>(I)) {
          // We need to fix up the insertion point.
          if (isa<PHINode>(I)) // PHI -> Non-PHI
            InsertPos = InstParent->getFirstInsertionPt();
          else // Non-PHI -> PHI
            InsertPos = InstParent->getFirstNonPHI()->getIterator();
        }

        InstParent->getInstList().insert(InsertPos, Result);

        // Push the new instruction and any users onto the worklist.
        Worklist.pushUsersToWorkList(*Result);
        Worklist.push(Result);

        eraseInstFromFunction(*I);
      } else {
        LLVM_DEBUG(dbgs() << "IC: Mod = " << OrigI << '\n'
                          << "    New = " << *I << '\n');

        // If the instruction was modified, it's possible that it is now dead.
        // if so, remove it.
        if (isInstructionTriviallyDead(I, &TLI)) {
          eraseInstFromFunction(*I);
        } else {
          Worklist.pushUsersToWorkList(*I);
          Worklist.push(I);
        }
      }
      MadeIRChange = true;
    }
  }

  Worklist.zap();
  return MadeIRChange;
}

// Track the scopes used by !alias.scope and !noalias. In a function, a
// @llvm.experimental.noalias.scope.decl is only useful if that scope is used
// by both sets. If not, the declaration of the scope can be safely omitted.
// The MDNode of the scope can be omitted as well for the instructions that are
// part of this function. We do not do that at this point, as this might become
// too time consuming to do.
class AliasScopeTracker {
  SmallPtrSet<const MDNode *, 8> UsedAliasScopesAndLists;
  SmallPtrSet<const MDNode *, 8> UsedNoAliasScopesAndLists;

public:
  void analyse(Instruction *I) {
    // This seems to be faster than checking 'mayReadOrWriteMemory()'.
    if (!I->hasMetadataOtherThanDebugLoc())
      return;

    auto Track = [](Metadata *ScopeList, auto &Container) {
      const auto *MDScopeList = dyn_cast_or_null<MDNode>(ScopeList);
      if (!MDScopeList || !Container.insert(MDScopeList).second)
        return;
      for (auto &MDOperand : MDScopeList->operands())
        if (auto *MDScope = dyn_cast<MDNode>(MDOperand))
          Container.insert(MDScope);
    };

    Track(I->getMetadata(LLVMContext::MD_alias_scope), UsedAliasScopesAndLists);
    Track(I->getMetadata(LLVMContext::MD_noalias), UsedNoAliasScopesAndLists);
  }

  bool isNoAliasScopeDeclDead(Instruction *Inst) {
    NoAliasScopeDeclInst *Decl = dyn_cast<NoAliasScopeDeclInst>(Inst);
    if (!Decl)
      return false;

    assert(Decl->use_empty() &&
           "llvm.experimental.noalias.scope.decl in use ?");
    const MDNode *MDSL = Decl->getScopeList();
    assert(MDSL->getNumOperands() == 1 &&
           "llvm.experimental.noalias.scope should refer to a single scope");
    auto &MDOperand = MDSL->getOperand(0);
    if (auto *MD = dyn_cast<MDNode>(MDOperand))
      return !UsedAliasScopesAndLists.contains(MD) ||
             !UsedNoAliasScopesAndLists.contains(MD);

    // Not an MDNode ? throw away.
    return true;
  }
};

/// Populate the IC worklist from a function, by walking it in depth-first
/// order and adding all reachable code to the worklist.
///
/// This has a couple of tricks to make the code faster and more powerful.  In
/// particular, we constant fold and DCE instructions as we go, to avoid adding
/// them to the worklist (this significantly speeds up instcombine on code where
/// many instructions are dead or constant).  Additionally, if we find a branch
/// whose condition is a known constant, we only visit the reachable successors.
static bool prepareICWorklistFromFunction(Function &F, const DataLayout &DL,
                                          const TargetLibraryInfo *TLI,
                                          InstructionWorklist &ICWorklist) {
  bool MadeIRChange = false;
  SmallPtrSet<BasicBlock *, 32> Visited;
  SmallVector<BasicBlock*, 256> Worklist;
  Worklist.push_back(&F.front());

  SmallVector<Instruction *, 128> InstrsForInstructionWorklist;
  DenseMap<Constant *, Constant *> FoldedConstants;
  AliasScopeTracker SeenAliasScopes;

  do {
    BasicBlock *BB = Worklist.pop_back_val();

    // We have now visited this block!  If we've already been here, ignore it.
    if (!Visited.insert(BB).second)
      continue;

    for (Instruction &Inst : llvm::make_early_inc_range(*BB)) {
      // ConstantProp instruction if trivially constant.
      if (!Inst.use_empty() &&
          (Inst.getNumOperands() == 0 || isa<Constant>(Inst.getOperand(0))))
        if (Constant *C = ConstantFoldInstruction(&Inst, DL, TLI)) {
          LLVM_DEBUG(dbgs() << "IC: ConstFold to: " << *C << " from: " << Inst
                            << '\n');
          Inst.replaceAllUsesWith(C);
          ++NumConstProp;
          if (isInstructionTriviallyDead(&Inst, TLI))
            Inst.eraseFromParent();
          MadeIRChange = true;
          continue;
        }

      // See if we can constant fold its operands.
      for (Use &U : Inst.operands()) {
        if (!isa<ConstantVector>(U) && !isa<ConstantExpr>(U))
          continue;

        auto *C = cast<Constant>(U);
        Constant *&FoldRes = FoldedConstants[C];
        if (!FoldRes)
          FoldRes = ConstantFoldConstant(C, DL, TLI);

        if (FoldRes != C) {
          LLVM_DEBUG(dbgs() << "IC: ConstFold operand of: " << Inst
                            << "\n    Old = " << *C
                            << "\n    New = " << *FoldRes << '\n');
          U = FoldRes;
          MadeIRChange = true;
        }
      }

      // Skip processing debug and pseudo intrinsics in InstCombine. Processing
      // these call instructions consumes non-trivial amount of time and
      // provides no value for the optimization.
      if (!Inst.isDebugOrPseudoInst()) {
        InstrsForInstructionWorklist.push_back(&Inst);
        SeenAliasScopes.analyse(&Inst);
      }
    }

    // Recursively visit successors.  If this is a branch or switch on a
    // constant, only visit the reachable successor.
    Instruction *TI = BB->getTerminator();
    if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
      if (BI->isConditional() && isa<ConstantInt>(BI->getCondition())) {
        bool CondVal = cast<ConstantInt>(BI->getCondition())->getZExtValue();
        BasicBlock *ReachableBB = BI->getSuccessor(!CondVal);
        Worklist.push_back(ReachableBB);
        continue;
      }
    } else if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
      if (ConstantInt *Cond = dyn_cast<ConstantInt>(SI->getCondition())) {
        Worklist.push_back(SI->findCaseValue(Cond)->getCaseSuccessor());
        continue;
      }
    }

    append_range(Worklist, successors(TI));
  } while (!Worklist.empty());

  // Remove instructions inside unreachable blocks. This prevents the
  // instcombine code from having to deal with some bad special cases, and
  // reduces use counts of instructions.
  for (BasicBlock &BB : F) {
    if (Visited.count(&BB))
      continue;

    unsigned NumDeadInstInBB;
    unsigned NumDeadDbgInstInBB;
    std::tie(NumDeadInstInBB, NumDeadDbgInstInBB) =
        removeAllNonTerminatorAndEHPadInstructions(&BB);

    MadeIRChange |= NumDeadInstInBB + NumDeadDbgInstInBB > 0;
    NumDeadInst += NumDeadInstInBB;
  }

  // Once we've found all of the instructions to add to instcombine's worklist,
  // add them in reverse order.  This way instcombine will visit from the top
  // of the function down.  This jives well with the way that it adds all uses
  // of instructions to the worklist after doing a transformation, thus avoiding
  // some N^2 behavior in pathological cases.
  ICWorklist.reserve(InstrsForInstructionWorklist.size());
  for (Instruction *Inst : reverse(InstrsForInstructionWorklist)) {
    // DCE instruction if trivially dead. As we iterate in reverse program
    // order here, we will clean up whole chains of dead instructions.
    if (isInstructionTriviallyDead(Inst, TLI) ||
        SeenAliasScopes.isNoAliasScopeDeclDead(Inst)) {
      ++NumDeadInst;
      LLVM_DEBUG(dbgs() << "IC: DCE: " << *Inst << '\n');
      salvageDebugInfo(*Inst);
      Inst->eraseFromParent();
      MadeIRChange = true;
      continue;
    }

    ICWorklist.push(Inst);
  }

  return MadeIRChange;
}

static bool combineInstructionsOverFunction(
    Function &F, InstructionWorklist &Worklist, AliasAnalysis *AA,
    AssumptionCache &AC, TargetLibraryInfo &TLI, TargetTransformInfo &TTI,
    DominatorTree &DT, OptimizationRemarkEmitter &ORE, BlockFrequencyInfo *BFI,
    ProfileSummaryInfo *PSI, unsigned MaxIterations, LoopInfo *LI) {
  auto &DL = F.getParent()->getDataLayout();
  MaxIterations = std::min(MaxIterations, LimitMaxIterations.getValue());

  /// Builder - This is an IRBuilder that automatically inserts new
  /// instructions into the worklist when they are created.
  IRBuilder<TargetFolder, IRBuilderCallbackInserter> Builder(
      F.getContext(), TargetFolder(DL),
      IRBuilderCallbackInserter([&Worklist, &AC](Instruction *I) {
        Worklist.add(I);
        if (auto *Assume = dyn_cast<AssumeInst>(I))
          AC.registerAssumption(Assume);
      }));

  // Lower dbg.declare intrinsics otherwise their value may be clobbered
  // by instcombiner.
  bool MadeIRChange = false;
  if (ShouldLowerDbgDeclare)
    MadeIRChange = LowerDbgDeclare(F);

  // Iterate while there is work to do.
  unsigned Iteration = 0;
  while (true) {
    ++NumWorklistIterations;
    ++Iteration;

    if (Iteration > InfiniteLoopDetectionThreshold) {
      report_fatal_error(
          "Instruction Combining seems stuck in an infinite loop after " +
          Twine(InfiniteLoopDetectionThreshold) + " iterations.");
    }

    if (Iteration > MaxIterations) {
      LLVM_DEBUG(dbgs() << "\n\n[IC] Iteration limit #" << MaxIterations
                        << " on " << F.getName()
                        << " reached; stopping before reaching a fixpoint\n");
      break;
    }

    LLVM_DEBUG(dbgs() << "\n\nINSTCOMBINE ITERATION #" << Iteration << " on "
                      << F.getName() << "\n");

    MadeIRChange |= prepareICWorklistFromFunction(F, DL, &TLI, Worklist);

    InstCombinerImpl IC(Worklist, Builder, F.hasMinSize(), AA, AC, TLI, TTI, DT,
                        ORE, BFI, PSI, DL, LI);
    IC.MaxArraySizeForCombine = MaxArraySize;

    if (!IC.run())
      break;

    MadeIRChange = true;
  }

  return MadeIRChange;
}

InstCombinePass::InstCombinePass() : MaxIterations(LimitMaxIterations) {}

InstCombinePass::InstCombinePass(unsigned MaxIterations)
    : MaxIterations(MaxIterations) {}

PreservedAnalyses InstCombinePass::run(Function &F,
                                       FunctionAnalysisManager &AM) {
  auto &AC = AM.getResult<AssumptionAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  auto &ORE = AM.getResult<OptimizationRemarkEmitterAnalysis>(F);
  auto &TTI = AM.getResult<TargetIRAnalysis>(F);

  auto *LI = AM.getCachedResult<LoopAnalysis>(F);

  auto *AA = &AM.getResult<AAManager>(F);
  auto &MAMProxy = AM.getResult<ModuleAnalysisManagerFunctionProxy>(F);
  ProfileSummaryInfo *PSI =
      MAMProxy.getCachedResult<ProfileSummaryAnalysis>(*F.getParent());
  auto *BFI = (PSI && PSI->hasProfileSummary()) ?
      &AM.getResult<BlockFrequencyAnalysis>(F) : nullptr;

  if (!combineInstructionsOverFunction(F, Worklist, AA, AC, TLI, TTI, DT, ORE,
                                       BFI, PSI, MaxIterations, LI))
    // No changes, all analyses are preserved.
    return PreservedAnalyses::all();

  // Mark all the analyses that instcombine updates as preserved.
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

void InstructionCombiningPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<AAResultsWrapperPass>();
  AU.addRequired<AssumptionCacheTracker>();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
  AU.addRequired<TargetTransformInfoWrapperPass>();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
  AU.addPreserved<DominatorTreeWrapperPass>();
  AU.addPreserved<AAResultsWrapperPass>();
  AU.addPreserved<BasicAAWrapperPass>();
  AU.addPreserved<GlobalsAAWrapperPass>();
  AU.addRequired<ProfileSummaryInfoWrapperPass>();
  LazyBlockFrequencyInfoPass::getLazyBFIAnalysisUsage(AU);
}

bool InstructionCombiningPass::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  // Required analyses.
  auto AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
  auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &ORE = getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();

  // Optional analyses.
  auto *LIWP = getAnalysisIfAvailable<LoopInfoWrapperPass>();
  auto *LI = LIWP ? &LIWP->getLoopInfo() : nullptr;
  ProfileSummaryInfo *PSI =
      &getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI();
  BlockFrequencyInfo *BFI =
      (PSI && PSI->hasProfileSummary()) ?
      &getAnalysis<LazyBlockFrequencyInfoPass>().getBFI() :
      nullptr;

  return combineInstructionsOverFunction(F, Worklist, AA, AC, TLI, TTI, DT, ORE,
                                         BFI, PSI, MaxIterations, LI);
}

char InstructionCombiningPass::ID = 0;

InstructionCombiningPass::InstructionCombiningPass()
    : FunctionPass(ID), MaxIterations(InstCombineDefaultMaxIterations) {
  initializeInstructionCombiningPassPass(*PassRegistry::getPassRegistry());
}

InstructionCombiningPass::InstructionCombiningPass(unsigned MaxIterations)
    : FunctionPass(ID), MaxIterations(MaxIterations) {
  initializeInstructionCombiningPassPass(*PassRegistry::getPassRegistry());
}

INITIALIZE_PASS_BEGIN(InstructionCombiningPass, "instcombine",
                      "Combine redundant instructions", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(GlobalsAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LazyBlockFrequencyInfoPass)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_END(InstructionCombiningPass, "instcombine",
                    "Combine redundant instructions", false, false)

// Initialization Routines
void llvm::initializeInstCombine(PassRegistry &Registry) {
  initializeInstructionCombiningPassPass(Registry);
}

void LLVMInitializeInstCombine(LLVMPassRegistryRef R) {
  initializeInstructionCombiningPassPass(*unwrap(R));
}

FunctionPass *llvm::createInstructionCombiningPass() {
  return new InstructionCombiningPass();
}

FunctionPass *llvm::createInstructionCombiningPass(unsigned MaxIterations) {
  return new InstructionCombiningPass(MaxIterations);
}

void LLVMAddInstructionCombiningPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createInstructionCombiningPass());
}
