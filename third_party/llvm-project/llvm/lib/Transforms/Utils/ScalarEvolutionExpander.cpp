//===- ScalarEvolutionExpander.cpp - Scalar Evolution Analysis ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the scalar evolution expander,
// which is used to generate the code corresponding to a given scalar evolution
// expression.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
#define SCEV_DEBUG_WITH_TYPE(TYPE, X) DEBUG_WITH_TYPE(TYPE, X)
#else
#define SCEV_DEBUG_WITH_TYPE(TYPE, X)
#endif

using namespace llvm;

cl::opt<unsigned> llvm::SCEVCheapExpansionBudget(
    "scev-cheap-expansion-budget", cl::Hidden, cl::init(4),
    cl::desc("When performing SCEV expansion only if it is cheap to do, this "
             "controls the budget that is considered cheap (default = 4)"));

using namespace PatternMatch;

/// ReuseOrCreateCast - Arrange for there to be a cast of V to Ty at IP,
/// reusing an existing cast if a suitable one (= dominating IP) exists, or
/// creating a new one.
Value *SCEVExpander::ReuseOrCreateCast(Value *V, Type *Ty,
                                       Instruction::CastOps Op,
                                       BasicBlock::iterator IP) {
  // This function must be called with the builder having a valid insertion
  // point. It doesn't need to be the actual IP where the uses of the returned
  // cast will be added, but it must dominate such IP.
  // We use this precondition to produce a cast that will dominate all its
  // uses. In particular, this is crucial for the case where the builder's
  // insertion point *is* the point where we were asked to put the cast.
  // Since we don't know the builder's insertion point is actually
  // where the uses will be added (only that it dominates it), we are
  // not allowed to move it.
  BasicBlock::iterator BIP = Builder.GetInsertPoint();

  Value *Ret = nullptr;

  // Check to see if there is already a cast!
  for (User *U : V->users()) {
    if (U->getType() != Ty)
      continue;
    CastInst *CI = dyn_cast<CastInst>(U);
    if (!CI || CI->getOpcode() != Op)
      continue;

    // Found a suitable cast that is at IP or comes before IP. Use it. Note that
    // the cast must also properly dominate the Builder's insertion point.
    if (IP->getParent() == CI->getParent() && &*BIP != CI &&
        (&*IP == CI || CI->comesBefore(&*IP))) {
      Ret = CI;
      break;
    }
  }

  // Create a new cast.
  if (!Ret) {
    SCEVInsertPointGuard Guard(Builder, this);
    Builder.SetInsertPoint(&*IP);
    Ret = Builder.CreateCast(Op, V, Ty, V->getName());
  }

  // We assert at the end of the function since IP might point to an
  // instruction with different dominance properties than a cast
  // (an invoke for example) and not dominate BIP (but the cast does).
  assert(!isa<Instruction>(Ret) ||
         SE.DT.dominates(cast<Instruction>(Ret), &*BIP));

  return Ret;
}

BasicBlock::iterator
SCEVExpander::findInsertPointAfter(Instruction *I,
                                   Instruction *MustDominate) const {
  BasicBlock::iterator IP = ++I->getIterator();
  if (auto *II = dyn_cast<InvokeInst>(I))
    IP = II->getNormalDest()->begin();

  while (isa<PHINode>(IP))
    ++IP;

  if (isa<FuncletPadInst>(IP) || isa<LandingPadInst>(IP)) {
    ++IP;
  } else if (isa<CatchSwitchInst>(IP)) {
    IP = MustDominate->getParent()->getFirstInsertionPt();
  } else {
    assert(!IP->isEHPad() && "unexpected eh pad!");
  }

  // Adjust insert point to be after instructions inserted by the expander, so
  // we can re-use already inserted instructions. Avoid skipping past the
  // original \p MustDominate, in case it is an inserted instruction.
  while (isInsertedInstruction(&*IP) && &*IP != MustDominate)
    ++IP;

  return IP;
}

BasicBlock::iterator
SCEVExpander::GetOptimalInsertionPointForCastOf(Value *V) const {
  // Cast the argument at the beginning of the entry block, after
  // any bitcasts of other arguments.
  if (Argument *A = dyn_cast<Argument>(V)) {
    BasicBlock::iterator IP = A->getParent()->getEntryBlock().begin();
    while ((isa<BitCastInst>(IP) &&
            isa<Argument>(cast<BitCastInst>(IP)->getOperand(0)) &&
            cast<BitCastInst>(IP)->getOperand(0) != A) ||
           isa<DbgInfoIntrinsic>(IP))
      ++IP;
    return IP;
  }

  // Cast the instruction immediately after the instruction.
  if (Instruction *I = dyn_cast<Instruction>(V))
    return findInsertPointAfter(I, &*Builder.GetInsertPoint());

  // Otherwise, this must be some kind of a constant,
  // so let's plop this cast into the function's entry block.
  assert(isa<Constant>(V) &&
         "Expected the cast argument to be a global/constant");
  return Builder.GetInsertBlock()
      ->getParent()
      ->getEntryBlock()
      .getFirstInsertionPt();
}

/// InsertNoopCastOfTo - Insert a cast of V to the specified type,
/// which must be possible with a noop cast, doing what we can to share
/// the casts.
Value *SCEVExpander::InsertNoopCastOfTo(Value *V, Type *Ty) {
  Instruction::CastOps Op = CastInst::getCastOpcode(V, false, Ty, false);
  assert((Op == Instruction::BitCast ||
          Op == Instruction::PtrToInt ||
          Op == Instruction::IntToPtr) &&
         "InsertNoopCastOfTo cannot perform non-noop casts!");
  assert(SE.getTypeSizeInBits(V->getType()) == SE.getTypeSizeInBits(Ty) &&
         "InsertNoopCastOfTo cannot change sizes!");

  // inttoptr only works for integral pointers. For non-integral pointers, we
  // can create a GEP on i8* null  with the integral value as index. Note that
  // it is safe to use GEP of null instead of inttoptr here, because only
  // expressions already based on a GEP of null should be converted to pointers
  // during expansion.
  if (Op == Instruction::IntToPtr) {
    auto *PtrTy = cast<PointerType>(Ty);
    if (DL.isNonIntegralPointerType(PtrTy)) {
      auto *Int8PtrTy = Builder.getInt8PtrTy(PtrTy->getAddressSpace());
      assert(DL.getTypeAllocSize(Int8PtrTy->getElementType()) == 1 &&
             "alloc size of i8 must by 1 byte for the GEP to be correct");
      auto *GEP = Builder.CreateGEP(
          Builder.getInt8Ty(), Constant::getNullValue(Int8PtrTy), V, "uglygep");
      return Builder.CreateBitCast(GEP, Ty);
    }
  }
  // Short-circuit unnecessary bitcasts.
  if (Op == Instruction::BitCast) {
    if (V->getType() == Ty)
      return V;
    if (CastInst *CI = dyn_cast<CastInst>(V)) {
      if (CI->getOperand(0)->getType() == Ty)
        return CI->getOperand(0);
    }
  }
  // Short-circuit unnecessary inttoptr<->ptrtoint casts.
  if ((Op == Instruction::PtrToInt || Op == Instruction::IntToPtr) &&
      SE.getTypeSizeInBits(Ty) == SE.getTypeSizeInBits(V->getType())) {
    if (CastInst *CI = dyn_cast<CastInst>(V))
      if ((CI->getOpcode() == Instruction::PtrToInt ||
           CI->getOpcode() == Instruction::IntToPtr) &&
          SE.getTypeSizeInBits(CI->getType()) ==
          SE.getTypeSizeInBits(CI->getOperand(0)->getType()))
        return CI->getOperand(0);
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
      if ((CE->getOpcode() == Instruction::PtrToInt ||
           CE->getOpcode() == Instruction::IntToPtr) &&
          SE.getTypeSizeInBits(CE->getType()) ==
          SE.getTypeSizeInBits(CE->getOperand(0)->getType()))
        return CE->getOperand(0);
  }

  // Fold a cast of a constant.
  if (Constant *C = dyn_cast<Constant>(V))
    return ConstantExpr::getCast(Op, C, Ty);

  // Try to reuse existing cast, or insert one.
  return ReuseOrCreateCast(V, Ty, Op, GetOptimalInsertionPointForCastOf(V));
}

/// InsertBinop - Insert the specified binary operator, doing a small amount
/// of work to avoid inserting an obviously redundant operation, and hoisting
/// to an outer loop when the opportunity is there and it is safe.
Value *SCEVExpander::InsertBinop(Instruction::BinaryOps Opcode,
                                 Value *LHS, Value *RHS,
                                 SCEV::NoWrapFlags Flags, bool IsSafeToHoist) {
  // Fold a binop with constant operands.
  if (Constant *CLHS = dyn_cast<Constant>(LHS))
    if (Constant *CRHS = dyn_cast<Constant>(RHS))
      return ConstantExpr::get(Opcode, CLHS, CRHS);

  // Do a quick scan to see if we have this binop nearby.  If so, reuse it.
  unsigned ScanLimit = 6;
  BasicBlock::iterator BlockBegin = Builder.GetInsertBlock()->begin();
  // Scanning starts from the last instruction before the insertion point.
  BasicBlock::iterator IP = Builder.GetInsertPoint();
  if (IP != BlockBegin) {
    --IP;
    for (; ScanLimit; --IP, --ScanLimit) {
      // Don't count dbg.value against the ScanLimit, to avoid perturbing the
      // generated code.
      if (isa<DbgInfoIntrinsic>(IP))
        ScanLimit++;

      auto canGenerateIncompatiblePoison = [&Flags](Instruction *I) {
        // Ensure that no-wrap flags match.
        if (isa<OverflowingBinaryOperator>(I)) {
          if (I->hasNoSignedWrap() != (Flags & SCEV::FlagNSW))
            return true;
          if (I->hasNoUnsignedWrap() != (Flags & SCEV::FlagNUW))
            return true;
        }
        // Conservatively, do not use any instruction which has any of exact
        // flags installed.
        if (isa<PossiblyExactOperator>(I) && I->isExact())
          return true;
        return false;
      };
      if (IP->getOpcode() == (unsigned)Opcode && IP->getOperand(0) == LHS &&
          IP->getOperand(1) == RHS && !canGenerateIncompatiblePoison(&*IP))
        return &*IP;
      if (IP == BlockBegin) break;
    }
  }

  // Save the original insertion point so we can restore it when we're done.
  DebugLoc Loc = Builder.GetInsertPoint()->getDebugLoc();
  SCEVInsertPointGuard Guard(Builder, this);

  if (IsSafeToHoist) {
    // Move the insertion point out of as many loops as we can.
    while (const Loop *L = SE.LI.getLoopFor(Builder.GetInsertBlock())) {
      if (!L->isLoopInvariant(LHS) || !L->isLoopInvariant(RHS)) break;
      BasicBlock *Preheader = L->getLoopPreheader();
      if (!Preheader) break;

      // Ok, move up a level.
      Builder.SetInsertPoint(Preheader->getTerminator());
    }
  }

  // If we haven't found this binop, insert it.
  Instruction *BO = cast<Instruction>(Builder.CreateBinOp(Opcode, LHS, RHS));
  BO->setDebugLoc(Loc);
  if (Flags & SCEV::FlagNUW)
    BO->setHasNoUnsignedWrap();
  if (Flags & SCEV::FlagNSW)
    BO->setHasNoSignedWrap();

  return BO;
}

/// FactorOutConstant - Test if S is divisible by Factor, using signed
/// division. If so, update S with Factor divided out and return true.
/// S need not be evenly divisible if a reasonable remainder can be
/// computed.
static bool FactorOutConstant(const SCEV *&S, const SCEV *&Remainder,
                              const SCEV *Factor, ScalarEvolution &SE,
                              const DataLayout &DL) {
  // Everything is divisible by one.
  if (Factor->isOne())
    return true;

  // x/x == 1.
  if (S == Factor) {
    S = SE.getConstant(S->getType(), 1);
    return true;
  }

  // For a Constant, check for a multiple of the given factor.
  if (const SCEVConstant *C = dyn_cast<SCEVConstant>(S)) {
    // 0/x == 0.
    if (C->isZero())
      return true;
    // Check for divisibility.
    if (const SCEVConstant *FC = dyn_cast<SCEVConstant>(Factor)) {
      ConstantInt *CI =
          ConstantInt::get(SE.getContext(), C->getAPInt().sdiv(FC->getAPInt()));
      // If the quotient is zero and the remainder is non-zero, reject
      // the value at this scale. It will be considered for subsequent
      // smaller scales.
      if (!CI->isZero()) {
        const SCEV *Div = SE.getConstant(CI);
        S = Div;
        Remainder = SE.getAddExpr(
            Remainder, SE.getConstant(C->getAPInt().srem(FC->getAPInt())));
        return true;
      }
    }
  }

  // In a Mul, check if there is a constant operand which is a multiple
  // of the given factor.
  if (const SCEVMulExpr *M = dyn_cast<SCEVMulExpr>(S)) {
    // Size is known, check if there is a constant operand which is a multiple
    // of the given factor. If so, we can factor it.
    if (const SCEVConstant *FC = dyn_cast<SCEVConstant>(Factor))
      if (const SCEVConstant *C = dyn_cast<SCEVConstant>(M->getOperand(0)))
        if (!C->getAPInt().srem(FC->getAPInt())) {
          SmallVector<const SCEV *, 4> NewMulOps(M->operands());
          NewMulOps[0] = SE.getConstant(C->getAPInt().sdiv(FC->getAPInt()));
          S = SE.getMulExpr(NewMulOps);
          return true;
        }
  }

  // In an AddRec, check if both start and step are divisible.
  if (const SCEVAddRecExpr *A = dyn_cast<SCEVAddRecExpr>(S)) {
    const SCEV *Step = A->getStepRecurrence(SE);
    const SCEV *StepRem = SE.getConstant(Step->getType(), 0);
    if (!FactorOutConstant(Step, StepRem, Factor, SE, DL))
      return false;
    if (!StepRem->isZero())
      return false;
    const SCEV *Start = A->getStart();
    if (!FactorOutConstant(Start, Remainder, Factor, SE, DL))
      return false;
    S = SE.getAddRecExpr(Start, Step, A->getLoop(),
                         A->getNoWrapFlags(SCEV::FlagNW));
    return true;
  }

  return false;
}

/// SimplifyAddOperands - Sort and simplify a list of add operands. NumAddRecs
/// is the number of SCEVAddRecExprs present, which are kept at the end of
/// the list.
///
static void SimplifyAddOperands(SmallVectorImpl<const SCEV *> &Ops,
                                Type *Ty,
                                ScalarEvolution &SE) {
  unsigned NumAddRecs = 0;
  for (unsigned i = Ops.size(); i > 0 && isa<SCEVAddRecExpr>(Ops[i-1]); --i)
    ++NumAddRecs;
  // Group Ops into non-addrecs and addrecs.
  SmallVector<const SCEV *, 8> NoAddRecs(Ops.begin(), Ops.end() - NumAddRecs);
  SmallVector<const SCEV *, 8> AddRecs(Ops.end() - NumAddRecs, Ops.end());
  // Let ScalarEvolution sort and simplify the non-addrecs list.
  const SCEV *Sum = NoAddRecs.empty() ?
                    SE.getConstant(Ty, 0) :
                    SE.getAddExpr(NoAddRecs);
  // If it returned an add, use the operands. Otherwise it simplified
  // the sum into a single value, so just use that.
  Ops.clear();
  if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(Sum))
    Ops.append(Add->op_begin(), Add->op_end());
  else if (!Sum->isZero())
    Ops.push_back(Sum);
  // Then append the addrecs.
  Ops.append(AddRecs.begin(), AddRecs.end());
}

/// SplitAddRecs - Flatten a list of add operands, moving addrec start values
/// out to the top level. For example, convert {a + b,+,c} to a, b, {0,+,d}.
/// This helps expose more opportunities for folding parts of the expressions
/// into GEP indices.
///
static void SplitAddRecs(SmallVectorImpl<const SCEV *> &Ops,
                         Type *Ty,
                         ScalarEvolution &SE) {
  // Find the addrecs.
  SmallVector<const SCEV *, 8> AddRecs;
  for (unsigned i = 0, e = Ops.size(); i != e; ++i)
    while (const SCEVAddRecExpr *A = dyn_cast<SCEVAddRecExpr>(Ops[i])) {
      const SCEV *Start = A->getStart();
      if (Start->isZero()) break;
      const SCEV *Zero = SE.getConstant(Ty, 0);
      AddRecs.push_back(SE.getAddRecExpr(Zero,
                                         A->getStepRecurrence(SE),
                                         A->getLoop(),
                                         A->getNoWrapFlags(SCEV::FlagNW)));
      if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(Start)) {
        Ops[i] = Zero;
        Ops.append(Add->op_begin(), Add->op_end());
        e += Add->getNumOperands();
      } else {
        Ops[i] = Start;
      }
    }
  if (!AddRecs.empty()) {
    // Add the addrecs onto the end of the list.
    Ops.append(AddRecs.begin(), AddRecs.end());
    // Resort the operand list, moving any constants to the front.
    SimplifyAddOperands(Ops, Ty, SE);
  }
}

/// expandAddToGEP - Expand an addition expression with a pointer type into
/// a GEP instead of using ptrtoint+arithmetic+inttoptr. This helps
/// BasicAliasAnalysis and other passes analyze the result. See the rules
/// for getelementptr vs. inttoptr in
/// http://llvm.org/docs/LangRef.html#pointeraliasing
/// for details.
///
/// Design note: The correctness of using getelementptr here depends on
/// ScalarEvolution not recognizing inttoptr and ptrtoint operators, as
/// they may introduce pointer arithmetic which may not be safely converted
/// into getelementptr.
///
/// Design note: It might seem desirable for this function to be more
/// loop-aware. If some of the indices are loop-invariant while others
/// aren't, it might seem desirable to emit multiple GEPs, keeping the
/// loop-invariant portions of the overall computation outside the loop.
/// However, there are a few reasons this is not done here. Hoisting simple
/// arithmetic is a low-level optimization that often isn't very
/// important until late in the optimization process. In fact, passes
/// like InstructionCombining will combine GEPs, even if it means
/// pushing loop-invariant computation down into loops, so even if the
/// GEPs were split here, the work would quickly be undone. The
/// LoopStrengthReduction pass, which is usually run quite late (and
/// after the last InstructionCombining pass), takes care of hoisting
/// loop-invariant portions of expressions, after considering what
/// can be folded using target addressing modes.
///
Value *SCEVExpander::expandAddToGEP(const SCEV *const *op_begin,
                                    const SCEV *const *op_end,
                                    PointerType *PTy,
                                    Type *Ty,
                                    Value *V) {
  SmallVector<Value *, 4> GepIndices;
  SmallVector<const SCEV *, 8> Ops(op_begin, op_end);
  bool AnyNonZeroIndices = false;

  // Split AddRecs up into parts as either of the parts may be usable
  // without the other.
  SplitAddRecs(Ops, Ty, SE);

  Type *IntIdxTy = DL.getIndexType(PTy);

  // For opaque pointers, always generate i8 GEP.
  if (!PTy->isOpaque()) {
    // Descend down the pointer's type and attempt to convert the other
    // operands into GEP indices, at each level. The first index in a GEP
    // indexes into the array implied by the pointer operand; the rest of
    // the indices index into the element or field type selected by the
    // preceding index.
    Type *ElTy = PTy->getElementType();
    for (;;) {
      // If the scale size is not 0, attempt to factor out a scale for
      // array indexing.
      SmallVector<const SCEV *, 8> ScaledOps;
      if (ElTy->isSized()) {
        const SCEV *ElSize = SE.getSizeOfExpr(IntIdxTy, ElTy);
        if (!ElSize->isZero()) {
          SmallVector<const SCEV *, 8> NewOps;
          for (const SCEV *Op : Ops) {
            const SCEV *Remainder = SE.getConstant(Ty, 0);
            if (FactorOutConstant(Op, Remainder, ElSize, SE, DL)) {
              // Op now has ElSize factored out.
              ScaledOps.push_back(Op);
              if (!Remainder->isZero())
                NewOps.push_back(Remainder);
              AnyNonZeroIndices = true;
            } else {
              // The operand was not divisible, so add it to the list of
              // operands we'll scan next iteration.
              NewOps.push_back(Op);
            }
          }
          // If we made any changes, update Ops.
          if (!ScaledOps.empty()) {
            Ops = NewOps;
            SimplifyAddOperands(Ops, Ty, SE);
          }
        }
      }

      // Record the scaled array index for this level of the type. If
      // we didn't find any operands that could be factored, tentatively
      // assume that element zero was selected (since the zero offset
      // would obviously be folded away).
      Value *Scaled =
          ScaledOps.empty()
              ? Constant::getNullValue(Ty)
              : expandCodeForImpl(SE.getAddExpr(ScaledOps), Ty, false);
      GepIndices.push_back(Scaled);

      // Collect struct field index operands.
      while (StructType *STy = dyn_cast<StructType>(ElTy)) {
        bool FoundFieldNo = false;
        // An empty struct has no fields.
        if (STy->getNumElements() == 0) break;
        // Field offsets are known. See if a constant offset falls within any of
        // the struct fields.
        if (Ops.empty())
          break;
        if (const SCEVConstant *C = dyn_cast<SCEVConstant>(Ops[0]))
          if (SE.getTypeSizeInBits(C->getType()) <= 64) {
            const StructLayout &SL = *DL.getStructLayout(STy);
            uint64_t FullOffset = C->getValue()->getZExtValue();
            if (FullOffset < SL.getSizeInBytes()) {
              unsigned ElIdx = SL.getElementContainingOffset(FullOffset);
              GepIndices.push_back(
                  ConstantInt::get(Type::getInt32Ty(Ty->getContext()), ElIdx));
              ElTy = STy->getTypeAtIndex(ElIdx);
              Ops[0] =
                  SE.getConstant(Ty, FullOffset - SL.getElementOffset(ElIdx));
              AnyNonZeroIndices = true;
              FoundFieldNo = true;
            }
          }
        // If no struct field offsets were found, tentatively assume that
        // field zero was selected (since the zero offset would obviously
        // be folded away).
        if (!FoundFieldNo) {
          ElTy = STy->getTypeAtIndex(0u);
          GepIndices.push_back(
            Constant::getNullValue(Type::getInt32Ty(Ty->getContext())));
        }
      }

      if (ArrayType *ATy = dyn_cast<ArrayType>(ElTy))
        ElTy = ATy->getElementType();
      else
        // FIXME: Handle VectorType.
        // E.g., If ElTy is scalable vector, then ElSize is not a compile-time
        // constant, therefore can not be factored out. The generated IR is less
        // ideal with base 'V' cast to i8* and do ugly getelementptr over that.
        break;
    }
  }

  // If none of the operands were convertible to proper GEP indices, cast
  // the base to i8* and do an ugly getelementptr with that. It's still
  // better than ptrtoint+arithmetic+inttoptr at least.
  if (!AnyNonZeroIndices) {
    // Cast the base to i8*.
    if (!PTy->isOpaque())
      V = InsertNoopCastOfTo(V,
         Type::getInt8PtrTy(Ty->getContext(), PTy->getAddressSpace()));

    assert(!isa<Instruction>(V) ||
           SE.DT.dominates(cast<Instruction>(V), &*Builder.GetInsertPoint()));

    // Expand the operands for a plain byte offset.
    Value *Idx = expandCodeForImpl(SE.getAddExpr(Ops), Ty, false);

    // Fold a GEP with constant operands.
    if (Constant *CLHS = dyn_cast<Constant>(V))
      if (Constant *CRHS = dyn_cast<Constant>(Idx))
        return ConstantExpr::getGetElementPtr(Type::getInt8Ty(Ty->getContext()),
                                              CLHS, CRHS);

    // Do a quick scan to see if we have this GEP nearby.  If so, reuse it.
    unsigned ScanLimit = 6;
    BasicBlock::iterator BlockBegin = Builder.GetInsertBlock()->begin();
    // Scanning starts from the last instruction before the insertion point.
    BasicBlock::iterator IP = Builder.GetInsertPoint();
    if (IP != BlockBegin) {
      --IP;
      for (; ScanLimit; --IP, --ScanLimit) {
        // Don't count dbg.value against the ScanLimit, to avoid perturbing the
        // generated code.
        if (isa<DbgInfoIntrinsic>(IP))
          ScanLimit++;
        if (IP->getOpcode() == Instruction::GetElementPtr &&
            IP->getOperand(0) == V && IP->getOperand(1) == Idx)
          return &*IP;
        if (IP == BlockBegin) break;
      }
    }

    // Save the original insertion point so we can restore it when we're done.
    SCEVInsertPointGuard Guard(Builder, this);

    // Move the insertion point out of as many loops as we can.
    while (const Loop *L = SE.LI.getLoopFor(Builder.GetInsertBlock())) {
      if (!L->isLoopInvariant(V) || !L->isLoopInvariant(Idx)) break;
      BasicBlock *Preheader = L->getLoopPreheader();
      if (!Preheader) break;

      // Ok, move up a level.
      Builder.SetInsertPoint(Preheader->getTerminator());
    }

    // Emit a GEP.
    return Builder.CreateGEP(Builder.getInt8Ty(), V, Idx, "uglygep");
  }

  {
    SCEVInsertPointGuard Guard(Builder, this);

    // Move the insertion point out of as many loops as we can.
    while (const Loop *L = SE.LI.getLoopFor(Builder.GetInsertBlock())) {
      if (!L->isLoopInvariant(V)) break;

      bool AnyIndexNotLoopInvariant = any_of(
          GepIndices, [L](Value *Op) { return !L->isLoopInvariant(Op); });

      if (AnyIndexNotLoopInvariant)
        break;

      BasicBlock *Preheader = L->getLoopPreheader();
      if (!Preheader) break;

      // Ok, move up a level.
      Builder.SetInsertPoint(Preheader->getTerminator());
    }

    // Insert a pretty getelementptr. Note that this GEP is not marked inbounds,
    // because ScalarEvolution may have changed the address arithmetic to
    // compute a value which is beyond the end of the allocated object.
    Value *Casted = V;
    if (V->getType() != PTy)
      Casted = InsertNoopCastOfTo(Casted, PTy);
    Value *GEP = Builder.CreateGEP(PTy->getElementType(), Casted, GepIndices,
                                   "scevgep");
    Ops.push_back(SE.getUnknown(GEP));
  }

  return expand(SE.getAddExpr(Ops));
}

Value *SCEVExpander::expandAddToGEP(const SCEV *Op, PointerType *PTy, Type *Ty,
                                    Value *V) {
  const SCEV *const Ops[1] = {Op};
  return expandAddToGEP(Ops, Ops + 1, PTy, Ty, V);
}

/// PickMostRelevantLoop - Given two loops pick the one that's most relevant for
/// SCEV expansion. If they are nested, this is the most nested. If they are
/// neighboring, pick the later.
static const Loop *PickMostRelevantLoop(const Loop *A, const Loop *B,
                                        DominatorTree &DT) {
  if (!A) return B;
  if (!B) return A;
  if (A->contains(B)) return B;
  if (B->contains(A)) return A;
  if (DT.dominates(A->getHeader(), B->getHeader())) return B;
  if (DT.dominates(B->getHeader(), A->getHeader())) return A;
  return A; // Arbitrarily break the tie.
}

/// getRelevantLoop - Get the most relevant loop associated with the given
/// expression, according to PickMostRelevantLoop.
const Loop *SCEVExpander::getRelevantLoop(const SCEV *S) {
  // Test whether we've already computed the most relevant loop for this SCEV.
  auto Pair = RelevantLoops.insert(std::make_pair(S, nullptr));
  if (!Pair.second)
    return Pair.first->second;

  if (isa<SCEVConstant>(S))
    // A constant has no relevant loops.
    return nullptr;
  if (const SCEVUnknown *U = dyn_cast<SCEVUnknown>(S)) {
    if (const Instruction *I = dyn_cast<Instruction>(U->getValue()))
      return Pair.first->second = SE.LI.getLoopFor(I->getParent());
    // A non-instruction has no relevant loops.
    return nullptr;
  }
  if (const SCEVNAryExpr *N = dyn_cast<SCEVNAryExpr>(S)) {
    const Loop *L = nullptr;
    if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S))
      L = AR->getLoop();
    for (const SCEV *Op : N->operands())
      L = PickMostRelevantLoop(L, getRelevantLoop(Op), SE.DT);
    return RelevantLoops[N] = L;
  }
  if (const SCEVCastExpr *C = dyn_cast<SCEVCastExpr>(S)) {
    const Loop *Result = getRelevantLoop(C->getOperand());
    return RelevantLoops[C] = Result;
  }
  if (const SCEVUDivExpr *D = dyn_cast<SCEVUDivExpr>(S)) {
    const Loop *Result = PickMostRelevantLoop(
        getRelevantLoop(D->getLHS()), getRelevantLoop(D->getRHS()), SE.DT);
    return RelevantLoops[D] = Result;
  }
  llvm_unreachable("Unexpected SCEV type!");
}

namespace {

/// LoopCompare - Compare loops by PickMostRelevantLoop.
class LoopCompare {
  DominatorTree &DT;
public:
  explicit LoopCompare(DominatorTree &dt) : DT(dt) {}

  bool operator()(std::pair<const Loop *, const SCEV *> LHS,
                  std::pair<const Loop *, const SCEV *> RHS) const {
    // Keep pointer operands sorted at the end.
    if (LHS.second->getType()->isPointerTy() !=
        RHS.second->getType()->isPointerTy())
      return LHS.second->getType()->isPointerTy();

    // Compare loops with PickMostRelevantLoop.
    if (LHS.first != RHS.first)
      return PickMostRelevantLoop(LHS.first, RHS.first, DT) != LHS.first;

    // If one operand is a non-constant negative and the other is not,
    // put the non-constant negative on the right so that a sub can
    // be used instead of a negate and add.
    if (LHS.second->isNonConstantNegative()) {
      if (!RHS.second->isNonConstantNegative())
        return false;
    } else if (RHS.second->isNonConstantNegative())
      return true;

    // Otherwise they are equivalent according to this comparison.
    return false;
  }
};

}

Value *SCEVExpander::visitAddExpr(const SCEVAddExpr *S) {
  Type *Ty = SE.getEffectiveSCEVType(S->getType());

  // Collect all the add operands in a loop, along with their associated loops.
  // Iterate in reverse so that constants are emitted last, all else equal, and
  // so that pointer operands are inserted first, which the code below relies on
  // to form more involved GEPs.
  SmallVector<std::pair<const Loop *, const SCEV *>, 8> OpsAndLoops;
  for (std::reverse_iterator<SCEVAddExpr::op_iterator> I(S->op_end()),
       E(S->op_begin()); I != E; ++I)
    OpsAndLoops.push_back(std::make_pair(getRelevantLoop(*I), *I));

  // Sort by loop. Use a stable sort so that constants follow non-constants and
  // pointer operands precede non-pointer operands.
  llvm::stable_sort(OpsAndLoops, LoopCompare(SE.DT));

  // Emit instructions to add all the operands. Hoist as much as possible
  // out of loops, and form meaningful getelementptrs where possible.
  Value *Sum = nullptr;
  for (auto I = OpsAndLoops.begin(), E = OpsAndLoops.end(); I != E;) {
    const Loop *CurLoop = I->first;
    const SCEV *Op = I->second;
    if (!Sum) {
      // This is the first operand. Just expand it.
      Sum = expand(Op);
      ++I;
    } else if (PointerType *PTy = dyn_cast<PointerType>(Sum->getType())) {
      // The running sum expression is a pointer. Try to form a getelementptr
      // at this level with that as the base.
      SmallVector<const SCEV *, 4> NewOps;
      for (; I != E && I->first == CurLoop; ++I) {
        // If the operand is SCEVUnknown and not instructions, peek through
        // it, to enable more of it to be folded into the GEP.
        const SCEV *X = I->second;
        if (const SCEVUnknown *U = dyn_cast<SCEVUnknown>(X))
          if (!isa<Instruction>(U->getValue()))
            X = SE.getSCEV(U->getValue());
        NewOps.push_back(X);
      }
      Sum = expandAddToGEP(NewOps.begin(), NewOps.end(), PTy, Ty, Sum);
    } else if (PointerType *PTy = dyn_cast<PointerType>(Op->getType())) {
      // The running sum is an integer, and there's a pointer at this level.
      // Try to form a getelementptr. If the running sum is instructions,
      // use a SCEVUnknown to avoid re-analyzing them.
      SmallVector<const SCEV *, 4> NewOps;
      NewOps.push_back(isa<Instruction>(Sum) ? SE.getUnknown(Sum) :
                                               SE.getSCEV(Sum));
      for (++I; I != E && I->first == CurLoop; ++I)
        NewOps.push_back(I->second);
      Sum = expandAddToGEP(NewOps.begin(), NewOps.end(), PTy, Ty, expand(Op));
    } else if (Op->isNonConstantNegative()) {
      // Instead of doing a negate and add, just do a subtract.
      Value *W = expandCodeForImpl(SE.getNegativeSCEV(Op), Ty, false);
      Sum = InsertNoopCastOfTo(Sum, Ty);
      Sum = InsertBinop(Instruction::Sub, Sum, W, SCEV::FlagAnyWrap,
                        /*IsSafeToHoist*/ true);
      ++I;
    } else {
      // A simple add.
      Value *W = expandCodeForImpl(Op, Ty, false);
      Sum = InsertNoopCastOfTo(Sum, Ty);
      // Canonicalize a constant to the RHS.
      if (isa<Constant>(Sum)) std::swap(Sum, W);
      Sum = InsertBinop(Instruction::Add, Sum, W, S->getNoWrapFlags(),
                        /*IsSafeToHoist*/ true);
      ++I;
    }
  }

  return Sum;
}

Value *SCEVExpander::visitMulExpr(const SCEVMulExpr *S) {
  Type *Ty = SE.getEffectiveSCEVType(S->getType());

  // Collect all the mul operands in a loop, along with their associated loops.
  // Iterate in reverse so that constants are emitted last, all else equal.
  SmallVector<std::pair<const Loop *, const SCEV *>, 8> OpsAndLoops;
  for (std::reverse_iterator<SCEVMulExpr::op_iterator> I(S->op_end()),
       E(S->op_begin()); I != E; ++I)
    OpsAndLoops.push_back(std::make_pair(getRelevantLoop(*I), *I));

  // Sort by loop. Use a stable sort so that constants follow non-constants.
  llvm::stable_sort(OpsAndLoops, LoopCompare(SE.DT));

  // Emit instructions to mul all the operands. Hoist as much as possible
  // out of loops.
  Value *Prod = nullptr;
  auto I = OpsAndLoops.begin();

  // Expand the calculation of X pow N in the following manner:
  // Let N = P1 + P2 + ... + PK, where all P are powers of 2. Then:
  // X pow N = (X pow P1) * (X pow P2) * ... * (X pow PK).
  const auto ExpandOpBinPowN = [this, &I, &OpsAndLoops, &Ty]() {
    auto E = I;
    // Calculate how many times the same operand from the same loop is included
    // into this power.
    uint64_t Exponent = 0;
    const uint64_t MaxExponent = UINT64_MAX >> 1;
    // No one sane will ever try to calculate such huge exponents, but if we
    // need this, we stop on UINT64_MAX / 2 because we need to exit the loop
    // below when the power of 2 exceeds our Exponent, and we want it to be
    // 1u << 31 at most to not deal with unsigned overflow.
    while (E != OpsAndLoops.end() && *I == *E && Exponent != MaxExponent) {
      ++Exponent;
      ++E;
    }
    assert(Exponent > 0 && "Trying to calculate a zeroth exponent of operand?");

    // Calculate powers with exponents 1, 2, 4, 8 etc. and include those of them
    // that are needed into the result.
    Value *P = expandCodeForImpl(I->second, Ty, false);
    Value *Result = nullptr;
    if (Exponent & 1)
      Result = P;
    for (uint64_t BinExp = 2; BinExp <= Exponent; BinExp <<= 1) {
      P = InsertBinop(Instruction::Mul, P, P, SCEV::FlagAnyWrap,
                      /*IsSafeToHoist*/ true);
      if (Exponent & BinExp)
        Result = Result ? InsertBinop(Instruction::Mul, Result, P,
                                      SCEV::FlagAnyWrap,
                                      /*IsSafeToHoist*/ true)
                        : P;
    }

    I = E;
    assert(Result && "Nothing was expanded?");
    return Result;
  };

  while (I != OpsAndLoops.end()) {
    if (!Prod) {
      // This is the first operand. Just expand it.
      Prod = ExpandOpBinPowN();
    } else if (I->second->isAllOnesValue()) {
      // Instead of doing a multiply by negative one, just do a negate.
      Prod = InsertNoopCastOfTo(Prod, Ty);
      Prod = InsertBinop(Instruction::Sub, Constant::getNullValue(Ty), Prod,
                         SCEV::FlagAnyWrap, /*IsSafeToHoist*/ true);
      ++I;
    } else {
      // A simple mul.
      Value *W = ExpandOpBinPowN();
      Prod = InsertNoopCastOfTo(Prod, Ty);
      // Canonicalize a constant to the RHS.
      if (isa<Constant>(Prod)) std::swap(Prod, W);
      const APInt *RHS;
      if (match(W, m_Power2(RHS))) {
        // Canonicalize Prod*(1<<C) to Prod<<C.
        assert(!Ty->isVectorTy() && "vector types are not SCEVable");
        auto NWFlags = S->getNoWrapFlags();
        // clear nsw flag if shl will produce poison value.
        if (RHS->logBase2() == RHS->getBitWidth() - 1)
          NWFlags = ScalarEvolution::clearFlags(NWFlags, SCEV::FlagNSW);
        Prod = InsertBinop(Instruction::Shl, Prod,
                           ConstantInt::get(Ty, RHS->logBase2()), NWFlags,
                           /*IsSafeToHoist*/ true);
      } else {
        Prod = InsertBinop(Instruction::Mul, Prod, W, S->getNoWrapFlags(),
                           /*IsSafeToHoist*/ true);
      }
    }
  }

  return Prod;
}

Value *SCEVExpander::visitUDivExpr(const SCEVUDivExpr *S) {
  Type *Ty = SE.getEffectiveSCEVType(S->getType());

  Value *LHS = expandCodeForImpl(S->getLHS(), Ty, false);
  if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(S->getRHS())) {
    const APInt &RHS = SC->getAPInt();
    if (RHS.isPowerOf2())
      return InsertBinop(Instruction::LShr, LHS,
                         ConstantInt::get(Ty, RHS.logBase2()),
                         SCEV::FlagAnyWrap, /*IsSafeToHoist*/ true);
  }

  Value *RHS = expandCodeForImpl(S->getRHS(), Ty, false);
  return InsertBinop(Instruction::UDiv, LHS, RHS, SCEV::FlagAnyWrap,
                     /*IsSafeToHoist*/ SE.isKnownNonZero(S->getRHS()));
}

/// Move parts of Base into Rest to leave Base with the minimal
/// expression that provides a pointer operand suitable for a
/// GEP expansion.
static void ExposePointerBase(const SCEV *&Base, const SCEV *&Rest,
                              ScalarEvolution &SE) {
  while (const SCEVAddRecExpr *A = dyn_cast<SCEVAddRecExpr>(Base)) {
    Base = A->getStart();
    Rest = SE.getAddExpr(Rest,
                         SE.getAddRecExpr(SE.getConstant(A->getType(), 0),
                                          A->getStepRecurrence(SE),
                                          A->getLoop(),
                                          A->getNoWrapFlags(SCEV::FlagNW)));
  }
  if (const SCEVAddExpr *A = dyn_cast<SCEVAddExpr>(Base)) {
    Base = A->getOperand(A->getNumOperands()-1);
    SmallVector<const SCEV *, 8> NewAddOps(A->operands());
    NewAddOps.back() = Rest;
    Rest = SE.getAddExpr(NewAddOps);
    ExposePointerBase(Base, Rest, SE);
  }
}

/// Determine if this is a well-behaved chain of instructions leading back to
/// the PHI. If so, it may be reused by expanded expressions.
bool SCEVExpander::isNormalAddRecExprPHI(PHINode *PN, Instruction *IncV,
                                         const Loop *L) {
  if (IncV->getNumOperands() == 0 || isa<PHINode>(IncV) ||
      (isa<CastInst>(IncV) && !isa<BitCastInst>(IncV)))
    return false;
  // If any of the operands don't dominate the insert position, bail.
  // Addrec operands are always loop-invariant, so this can only happen
  // if there are instructions which haven't been hoisted.
  if (L == IVIncInsertLoop) {
    for (Use &Op : llvm::drop_begin(IncV->operands()))
      if (Instruction *OInst = dyn_cast<Instruction>(Op))
        if (!SE.DT.dominates(OInst, IVIncInsertPos))
          return false;
  }
  // Advance to the next instruction.
  IncV = dyn_cast<Instruction>(IncV->getOperand(0));
  if (!IncV)
    return false;

  if (IncV->mayHaveSideEffects())
    return false;

  if (IncV == PN)
    return true;

  return isNormalAddRecExprPHI(PN, IncV, L);
}

/// getIVIncOperand returns an induction variable increment's induction
/// variable operand.
///
/// If allowScale is set, any type of GEP is allowed as long as the nonIV
/// operands dominate InsertPos.
///
/// If allowScale is not set, ensure that a GEP increment conforms to one of the
/// simple patterns generated by getAddRecExprPHILiterally and
/// expandAddtoGEP. If the pattern isn't recognized, return NULL.
Instruction *SCEVExpander::getIVIncOperand(Instruction *IncV,
                                           Instruction *InsertPos,
                                           bool allowScale) {
  if (IncV == InsertPos)
    return nullptr;

  switch (IncV->getOpcode()) {
  default:
    return nullptr;
  // Check for a simple Add/Sub or GEP of a loop invariant step.
  case Instruction::Add:
  case Instruction::Sub: {
    Instruction *OInst = dyn_cast<Instruction>(IncV->getOperand(1));
    if (!OInst || SE.DT.dominates(OInst, InsertPos))
      return dyn_cast<Instruction>(IncV->getOperand(0));
    return nullptr;
  }
  case Instruction::BitCast:
    return dyn_cast<Instruction>(IncV->getOperand(0));
  case Instruction::GetElementPtr:
    for (Use &U : llvm::drop_begin(IncV->operands())) {
      if (isa<Constant>(U))
        continue;
      if (Instruction *OInst = dyn_cast<Instruction>(U)) {
        if (!SE.DT.dominates(OInst, InsertPos))
          return nullptr;
      }
      if (allowScale) {
        // allow any kind of GEP as long as it can be hoisted.
        continue;
      }
      // This must be a pointer addition of constants (pretty), which is already
      // handled, or some number of address-size elements (ugly). Ugly geps
      // have 2 operands. i1* is used by the expander to represent an
      // address-size element.
      if (IncV->getNumOperands() != 2)
        return nullptr;
      unsigned AS = cast<PointerType>(IncV->getType())->getAddressSpace();
      if (IncV->getType() != Type::getInt1PtrTy(SE.getContext(), AS)
          && IncV->getType() != Type::getInt8PtrTy(SE.getContext(), AS))
        return nullptr;
      break;
    }
    return dyn_cast<Instruction>(IncV->getOperand(0));
  }
}

/// If the insert point of the current builder or any of the builders on the
/// stack of saved builders has 'I' as its insert point, update it to point to
/// the instruction after 'I'.  This is intended to be used when the instruction
/// 'I' is being moved.  If this fixup is not done and 'I' is moved to a
/// different block, the inconsistent insert point (with a mismatched
/// Instruction and Block) can lead to an instruction being inserted in a block
/// other than its parent.
void SCEVExpander::fixupInsertPoints(Instruction *I) {
  BasicBlock::iterator It(*I);
  BasicBlock::iterator NewInsertPt = std::next(It);
  if (Builder.GetInsertPoint() == It)
    Builder.SetInsertPoint(&*NewInsertPt);
  for (auto *InsertPtGuard : InsertPointGuards)
    if (InsertPtGuard->GetInsertPoint() == It)
      InsertPtGuard->SetInsertPoint(NewInsertPt);
}

/// hoistStep - Attempt to hoist a simple IV increment above InsertPos to make
/// it available to other uses in this loop. Recursively hoist any operands,
/// until we reach a value that dominates InsertPos.
bool SCEVExpander::hoistIVInc(Instruction *IncV, Instruction *InsertPos) {
  if (SE.DT.dominates(IncV, InsertPos))
      return true;

  // InsertPos must itself dominate IncV so that IncV's new position satisfies
  // its existing users.
  if (isa<PHINode>(InsertPos) ||
      !SE.DT.dominates(InsertPos->getParent(), IncV->getParent()))
    return false;

  if (!SE.LI.movementPreservesLCSSAForm(IncV, InsertPos))
    return false;

  // Check that the chain of IV operands leading back to Phi can be hoisted.
  SmallVector<Instruction*, 4> IVIncs;
  for(;;) {
    Instruction *Oper = getIVIncOperand(IncV, InsertPos, /*allowScale*/true);
    if (!Oper)
      return false;
    // IncV is safe to hoist.
    IVIncs.push_back(IncV);
    IncV = Oper;
    if (SE.DT.dominates(IncV, InsertPos))
      break;
  }
  for (auto I = IVIncs.rbegin(), E = IVIncs.rend(); I != E; ++I) {
    fixupInsertPoints(*I);
    (*I)->moveBefore(InsertPos);
  }
  return true;
}

/// Determine if this cyclic phi is in a form that would have been generated by
/// LSR. We don't care if the phi was actually expanded in this pass, as long
/// as it is in a low-cost form, for example, no implied multiplication. This
/// should match any patterns generated by getAddRecExprPHILiterally and
/// expandAddtoGEP.
bool SCEVExpander::isExpandedAddRecExprPHI(PHINode *PN, Instruction *IncV,
                                           const Loop *L) {
  for(Instruction *IVOper = IncV;
      (IVOper = getIVIncOperand(IVOper, L->getLoopPreheader()->getTerminator(),
                                /*allowScale=*/false));) {
    if (IVOper == PN)
      return true;
  }
  return false;
}

/// expandIVInc - Expand an IV increment at Builder's current InsertPos.
/// Typically this is the LatchBlock terminator or IVIncInsertPos, but we may
/// need to materialize IV increments elsewhere to handle difficult situations.
Value *SCEVExpander::expandIVInc(PHINode *PN, Value *StepV, const Loop *L,
                                 Type *ExpandTy, Type *IntTy,
                                 bool useSubtract) {
  Value *IncV;
  // If the PHI is a pointer, use a GEP, otherwise use an add or sub.
  if (ExpandTy->isPointerTy()) {
    PointerType *GEPPtrTy = cast<PointerType>(ExpandTy);
    // If the step isn't constant, don't use an implicitly scaled GEP, because
    // that would require a multiply inside the loop.
    if (!isa<ConstantInt>(StepV))
      GEPPtrTy = PointerType::get(Type::getInt1Ty(SE.getContext()),
                                  GEPPtrTy->getAddressSpace());
    IncV = expandAddToGEP(SE.getSCEV(StepV), GEPPtrTy, IntTy, PN);
    if (IncV->getType() != PN->getType())
      IncV = Builder.CreateBitCast(IncV, PN->getType());
  } else {
    IncV = useSubtract ?
      Builder.CreateSub(PN, StepV, Twine(IVName) + ".iv.next") :
      Builder.CreateAdd(PN, StepV, Twine(IVName) + ".iv.next");
  }
  return IncV;
}

/// Check whether we can cheaply express the requested SCEV in terms of
/// the available PHI SCEV by truncation and/or inversion of the step.
static bool canBeCheaplyTransformed(ScalarEvolution &SE,
                                    const SCEVAddRecExpr *Phi,
                                    const SCEVAddRecExpr *Requested,
                                    bool &InvertStep) {
  // We can't transform to match a pointer PHI.
  if (Phi->getType()->isPointerTy())
    return false;

  Type *PhiTy = SE.getEffectiveSCEVType(Phi->getType());
  Type *RequestedTy = SE.getEffectiveSCEVType(Requested->getType());

  if (RequestedTy->getIntegerBitWidth() > PhiTy->getIntegerBitWidth())
    return false;

  // Try truncate it if necessary.
  Phi = dyn_cast<SCEVAddRecExpr>(SE.getTruncateOrNoop(Phi, RequestedTy));
  if (!Phi)
    return false;

  // Check whether truncation will help.
  if (Phi == Requested) {
    InvertStep = false;
    return true;
  }

  // Check whether inverting will help: {R,+,-1} == R - {0,+,1}.
  if (SE.getMinusSCEV(Requested->getStart(), Requested) == Phi) {
    InvertStep = true;
    return true;
  }

  return false;
}

static bool IsIncrementNSW(ScalarEvolution &SE, const SCEVAddRecExpr *AR) {
  if (!isa<IntegerType>(AR->getType()))
    return false;

  unsigned BitWidth = cast<IntegerType>(AR->getType())->getBitWidth();
  Type *WideTy = IntegerType::get(AR->getType()->getContext(), BitWidth * 2);
  const SCEV *Step = AR->getStepRecurrence(SE);
  const SCEV *OpAfterExtend = SE.getAddExpr(SE.getSignExtendExpr(Step, WideTy),
                                            SE.getSignExtendExpr(AR, WideTy));
  const SCEV *ExtendAfterOp =
    SE.getSignExtendExpr(SE.getAddExpr(AR, Step), WideTy);
  return ExtendAfterOp == OpAfterExtend;
}

static bool IsIncrementNUW(ScalarEvolution &SE, const SCEVAddRecExpr *AR) {
  if (!isa<IntegerType>(AR->getType()))
    return false;

  unsigned BitWidth = cast<IntegerType>(AR->getType())->getBitWidth();
  Type *WideTy = IntegerType::get(AR->getType()->getContext(), BitWidth * 2);
  const SCEV *Step = AR->getStepRecurrence(SE);
  const SCEV *OpAfterExtend = SE.getAddExpr(SE.getZeroExtendExpr(Step, WideTy),
                                            SE.getZeroExtendExpr(AR, WideTy));
  const SCEV *ExtendAfterOp =
    SE.getZeroExtendExpr(SE.getAddExpr(AR, Step), WideTy);
  return ExtendAfterOp == OpAfterExtend;
}

/// getAddRecExprPHILiterally - Helper for expandAddRecExprLiterally. Expand
/// the base addrec, which is the addrec without any non-loop-dominating
/// values, and return the PHI.
PHINode *
SCEVExpander::getAddRecExprPHILiterally(const SCEVAddRecExpr *Normalized,
                                        const Loop *L,
                                        Type *ExpandTy,
                                        Type *IntTy,
                                        Type *&TruncTy,
                                        bool &InvertStep) {
  assert((!IVIncInsertLoop||IVIncInsertPos) && "Uninitialized insert position");

  // Reuse a previously-inserted PHI, if present.
  BasicBlock *LatchBlock = L->getLoopLatch();
  if (LatchBlock) {
    PHINode *AddRecPhiMatch = nullptr;
    Instruction *IncV = nullptr;
    TruncTy = nullptr;
    InvertStep = false;

    // Only try partially matching scevs that need truncation and/or
    // step-inversion if we know this loop is outside the current loop.
    bool TryNonMatchingSCEV =
        IVIncInsertLoop &&
        SE.DT.properlyDominates(LatchBlock, IVIncInsertLoop->getHeader());

    for (PHINode &PN : L->getHeader()->phis()) {
      if (!SE.isSCEVable(PN.getType()))
        continue;

      // We should not look for a incomplete PHI. Getting SCEV for a incomplete
      // PHI has no meaning at all.
      if (!PN.isComplete()) {
        SCEV_DEBUG_WITH_TYPE(
            DebugType, dbgs() << "One incomplete PHI is found: " << PN << "\n");
        continue;
      }

      const SCEVAddRecExpr *PhiSCEV = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(&PN));
      if (!PhiSCEV)
        continue;

      bool IsMatchingSCEV = PhiSCEV == Normalized;
      // We only handle truncation and inversion of phi recurrences for the
      // expanded expression if the expanded expression's loop dominates the
      // loop we insert to. Check now, so we can bail out early.
      if (!IsMatchingSCEV && !TryNonMatchingSCEV)
          continue;

      // TODO: this possibly can be reworked to avoid this cast at all.
      Instruction *TempIncV =
          dyn_cast<Instruction>(PN.getIncomingValueForBlock(LatchBlock));
      if (!TempIncV)
        continue;

      // Check whether we can reuse this PHI node.
      if (LSRMode) {
        if (!isExpandedAddRecExprPHI(&PN, TempIncV, L))
          continue;
      } else {
        if (!isNormalAddRecExprPHI(&PN, TempIncV, L))
          continue;
      }

      // Stop if we have found an exact match SCEV.
      if (IsMatchingSCEV) {
        IncV = TempIncV;
        TruncTy = nullptr;
        InvertStep = false;
        AddRecPhiMatch = &PN;
        break;
      }

      // Try whether the phi can be translated into the requested form
      // (truncated and/or offset by a constant).
      if ((!TruncTy || InvertStep) &&
          canBeCheaplyTransformed(SE, PhiSCEV, Normalized, InvertStep)) {
        // Record the phi node. But don't stop we might find an exact match
        // later.
        AddRecPhiMatch = &PN;
        IncV = TempIncV;
        TruncTy = SE.getEffectiveSCEVType(Normalized->getType());
      }
    }

    if (AddRecPhiMatch) {
      // Ok, the add recurrence looks usable.
      // Remember this PHI, even in post-inc mode.
      InsertedValues.insert(AddRecPhiMatch);
      // Remember the increment.
      rememberInstruction(IncV);
      // Those values were not actually inserted but re-used.
      ReusedValues.insert(AddRecPhiMatch);
      ReusedValues.insert(IncV);
      return AddRecPhiMatch;
    }
  }

  // Save the original insertion point so we can restore it when we're done.
  SCEVInsertPointGuard Guard(Builder, this);

  // Another AddRec may need to be recursively expanded below. For example, if
  // this AddRec is quadratic, the StepV may itself be an AddRec in this
  // loop. Remove this loop from the PostIncLoops set before expanding such
  // AddRecs. Otherwise, we cannot find a valid position for the step
  // (i.e. StepV can never dominate its loop header).  Ideally, we could do
  // SavedIncLoops.swap(PostIncLoops), but we generally have a single element,
  // so it's not worth implementing SmallPtrSet::swap.
  PostIncLoopSet SavedPostIncLoops = PostIncLoops;
  PostIncLoops.clear();

  // Expand code for the start value into the loop preheader.
  assert(L->getLoopPreheader() &&
         "Can't expand add recurrences without a loop preheader!");
  Value *StartV =
      expandCodeForImpl(Normalized->getStart(), ExpandTy,
                        L->getLoopPreheader()->getTerminator(), false);

  // StartV must have been be inserted into L's preheader to dominate the new
  // phi.
  assert(!isa<Instruction>(StartV) ||
         SE.DT.properlyDominates(cast<Instruction>(StartV)->getParent(),
                                 L->getHeader()));

  // Expand code for the step value. Do this before creating the PHI so that PHI
  // reuse code doesn't see an incomplete PHI.
  const SCEV *Step = Normalized->getStepRecurrence(SE);
  // If the stride is negative, insert a sub instead of an add for the increment
  // (unless it's a constant, because subtracts of constants are canonicalized
  // to adds).
  bool useSubtract = !ExpandTy->isPointerTy() && Step->isNonConstantNegative();
  if (useSubtract)
    Step = SE.getNegativeSCEV(Step);
  // Expand the step somewhere that dominates the loop header.
  Value *StepV = expandCodeForImpl(
      Step, IntTy, &*L->getHeader()->getFirstInsertionPt(), false);

  // The no-wrap behavior proved by IsIncrement(NUW|NSW) is only applicable if
  // we actually do emit an addition.  It does not apply if we emit a
  // subtraction.
  bool IncrementIsNUW = !useSubtract && IsIncrementNUW(SE, Normalized);
  bool IncrementIsNSW = !useSubtract && IsIncrementNSW(SE, Normalized);

  // Create the PHI.
  BasicBlock *Header = L->getHeader();
  Builder.SetInsertPoint(Header, Header->begin());
  pred_iterator HPB = pred_begin(Header), HPE = pred_end(Header);
  PHINode *PN = Builder.CreatePHI(ExpandTy, std::distance(HPB, HPE),
                                  Twine(IVName) + ".iv");

  // Create the step instructions and populate the PHI.
  for (pred_iterator HPI = HPB; HPI != HPE; ++HPI) {
    BasicBlock *Pred = *HPI;

    // Add a start value.
    if (!L->contains(Pred)) {
      PN->addIncoming(StartV, Pred);
      continue;
    }

    // Create a step value and add it to the PHI.
    // If IVIncInsertLoop is non-null and equal to the addrec's loop, insert the
    // instructions at IVIncInsertPos.
    Instruction *InsertPos = L == IVIncInsertLoop ?
      IVIncInsertPos : Pred->getTerminator();
    Builder.SetInsertPoint(InsertPos);
    Value *IncV = expandIVInc(PN, StepV, L, ExpandTy, IntTy, useSubtract);

    if (isa<OverflowingBinaryOperator>(IncV)) {
      if (IncrementIsNUW)
        cast<BinaryOperator>(IncV)->setHasNoUnsignedWrap();
      if (IncrementIsNSW)
        cast<BinaryOperator>(IncV)->setHasNoSignedWrap();
    }
    PN->addIncoming(IncV, Pred);
  }

  // After expanding subexpressions, restore the PostIncLoops set so the caller
  // can ensure that IVIncrement dominates the current uses.
  PostIncLoops = SavedPostIncLoops;

  // Remember this PHI, even in post-inc mode. LSR SCEV-based salvaging is most
  // effective when we are able to use an IV inserted here, so record it.
  InsertedValues.insert(PN);
  InsertedIVs.push_back(PN);
  return PN;
}

Value *SCEVExpander::expandAddRecExprLiterally(const SCEVAddRecExpr *S) {
  Type *STy = S->getType();
  Type *IntTy = SE.getEffectiveSCEVType(STy);
  const Loop *L = S->getLoop();

  // Determine a normalized form of this expression, which is the expression
  // before any post-inc adjustment is made.
  const SCEVAddRecExpr *Normalized = S;
  if (PostIncLoops.count(L)) {
    PostIncLoopSet Loops;
    Loops.insert(L);
    Normalized = cast<SCEVAddRecExpr>(normalizeForPostIncUse(S, Loops, SE));
  }

  // Strip off any non-loop-dominating component from the addrec start.
  const SCEV *Start = Normalized->getStart();
  const SCEV *PostLoopOffset = nullptr;
  if (!SE.properlyDominates(Start, L->getHeader())) {
    PostLoopOffset = Start;
    Start = SE.getConstant(Normalized->getType(), 0);
    Normalized = cast<SCEVAddRecExpr>(
      SE.getAddRecExpr(Start, Normalized->getStepRecurrence(SE),
                       Normalized->getLoop(),
                       Normalized->getNoWrapFlags(SCEV::FlagNW)));
  }

  // Strip off any non-loop-dominating component from the addrec step.
  const SCEV *Step = Normalized->getStepRecurrence(SE);
  const SCEV *PostLoopScale = nullptr;
  if (!SE.dominates(Step, L->getHeader())) {
    PostLoopScale = Step;
    Step = SE.getConstant(Normalized->getType(), 1);
    if (!Start->isZero()) {
        // The normalization below assumes that Start is constant zero, so if
        // it isn't re-associate Start to PostLoopOffset.
        assert(!PostLoopOffset && "Start not-null but PostLoopOffset set?");
        PostLoopOffset = Start;
        Start = SE.getConstant(Normalized->getType(), 0);
    }
    Normalized =
      cast<SCEVAddRecExpr>(SE.getAddRecExpr(
                             Start, Step, Normalized->getLoop(),
                             Normalized->getNoWrapFlags(SCEV::FlagNW)));
  }

  // Expand the core addrec. If we need post-loop scaling, force it to
  // expand to an integer type to avoid the need for additional casting.
  Type *ExpandTy = PostLoopScale ? IntTy : STy;
  // We can't use a pointer type for the addrec if the pointer type is
  // non-integral.
  Type *AddRecPHIExpandTy =
      DL.isNonIntegralPointerType(STy) ? Normalized->getType() : ExpandTy;

  // In some cases, we decide to reuse an existing phi node but need to truncate
  // it and/or invert the step.
  Type *TruncTy = nullptr;
  bool InvertStep = false;
  PHINode *PN = getAddRecExprPHILiterally(Normalized, L, AddRecPHIExpandTy,
                                          IntTy, TruncTy, InvertStep);

  // Accommodate post-inc mode, if necessary.
  Value *Result;
  if (!PostIncLoops.count(L))
    Result = PN;
  else {
    // In PostInc mode, use the post-incremented value.
    BasicBlock *LatchBlock = L->getLoopLatch();
    assert(LatchBlock && "PostInc mode requires a unique loop latch!");
    Result = PN->getIncomingValueForBlock(LatchBlock);

    // We might be introducing a new use of the post-inc IV that is not poison
    // safe, in which case we should drop poison generating flags. Only keep
    // those flags for which SCEV has proven that they always hold.
    if (isa<OverflowingBinaryOperator>(Result)) {
      auto *I = cast<Instruction>(Result);
      if (!S->hasNoUnsignedWrap())
        I->setHasNoUnsignedWrap(false);
      if (!S->hasNoSignedWrap())
        I->setHasNoSignedWrap(false);
    }

    // For an expansion to use the postinc form, the client must call
    // expandCodeFor with an InsertPoint that is either outside the PostIncLoop
    // or dominated by IVIncInsertPos.
    if (isa<Instruction>(Result) &&
        !SE.DT.dominates(cast<Instruction>(Result),
                         &*Builder.GetInsertPoint())) {
      // The induction variable's postinc expansion does not dominate this use.
      // IVUsers tries to prevent this case, so it is rare. However, it can
      // happen when an IVUser outside the loop is not dominated by the latch
      // block. Adjusting IVIncInsertPos before expansion begins cannot handle
      // all cases. Consider a phi outside whose operand is replaced during
      // expansion with the value of the postinc user. Without fundamentally
      // changing the way postinc users are tracked, the only remedy is
      // inserting an extra IV increment. StepV might fold into PostLoopOffset,
      // but hopefully expandCodeFor handles that.
      bool useSubtract =
        !ExpandTy->isPointerTy() && Step->isNonConstantNegative();
      if (useSubtract)
        Step = SE.getNegativeSCEV(Step);
      Value *StepV;
      {
        // Expand the step somewhere that dominates the loop header.
        SCEVInsertPointGuard Guard(Builder, this);
        StepV = expandCodeForImpl(
            Step, IntTy, &*L->getHeader()->getFirstInsertionPt(), false);
      }
      Result = expandIVInc(PN, StepV, L, ExpandTy, IntTy, useSubtract);
    }
  }

  // We have decided to reuse an induction variable of a dominating loop. Apply
  // truncation and/or inversion of the step.
  if (TruncTy) {
    Type *ResTy = Result->getType();
    // Normalize the result type.
    if (ResTy != SE.getEffectiveSCEVType(ResTy))
      Result = InsertNoopCastOfTo(Result, SE.getEffectiveSCEVType(ResTy));
    // Truncate the result.
    if (TruncTy != Result->getType())
      Result = Builder.CreateTrunc(Result, TruncTy);

    // Invert the result.
    if (InvertStep)
      Result = Builder.CreateSub(
          expandCodeForImpl(Normalized->getStart(), TruncTy, false), Result);
  }

  // Re-apply any non-loop-dominating scale.
  if (PostLoopScale) {
    assert(S->isAffine() && "Can't linearly scale non-affine recurrences.");
    Result = InsertNoopCastOfTo(Result, IntTy);
    Result = Builder.CreateMul(Result,
                               expandCodeForImpl(PostLoopScale, IntTy, false));
  }

  // Re-apply any non-loop-dominating offset.
  if (PostLoopOffset) {
    if (PointerType *PTy = dyn_cast<PointerType>(ExpandTy)) {
      if (Result->getType()->isIntegerTy()) {
        Value *Base = expandCodeForImpl(PostLoopOffset, ExpandTy, false);
        Result = expandAddToGEP(SE.getUnknown(Result), PTy, IntTy, Base);
      } else {
        Result = expandAddToGEP(PostLoopOffset, PTy, IntTy, Result);
      }
    } else {
      Result = InsertNoopCastOfTo(Result, IntTy);
      Result = Builder.CreateAdd(
          Result, expandCodeForImpl(PostLoopOffset, IntTy, false));
    }
  }

  return Result;
}

Value *SCEVExpander::visitAddRecExpr(const SCEVAddRecExpr *S) {
  // In canonical mode we compute the addrec as an expression of a canonical IV
  // using evaluateAtIteration and expand the resulting SCEV expression. This
  // way we avoid introducing new IVs to carry on the comutation of the addrec
  // throughout the loop.
  //
  // For nested addrecs evaluateAtIteration might need a canonical IV of a
  // type wider than the addrec itself. Emitting a canonical IV of the
  // proper type might produce non-legal types, for example expanding an i64
  // {0,+,2,+,1} addrec would need an i65 canonical IV. To avoid this just fall
  // back to non-canonical mode for nested addrecs.
  if (!CanonicalMode || (S->getNumOperands() > 2))
    return expandAddRecExprLiterally(S);

  Type *Ty = SE.getEffectiveSCEVType(S->getType());
  const Loop *L = S->getLoop();

  // First check for an existing canonical IV in a suitable type.
  PHINode *CanonicalIV = nullptr;
  if (PHINode *PN = L->getCanonicalInductionVariable())
    if (SE.getTypeSizeInBits(PN->getType()) >= SE.getTypeSizeInBits(Ty))
      CanonicalIV = PN;

  // Rewrite an AddRec in terms of the canonical induction variable, if
  // its type is more narrow.
  if (CanonicalIV &&
      SE.getTypeSizeInBits(CanonicalIV->getType()) > SE.getTypeSizeInBits(Ty) &&
      !S->getType()->isPointerTy()) {
    SmallVector<const SCEV *, 4> NewOps(S->getNumOperands());
    for (unsigned i = 0, e = S->getNumOperands(); i != e; ++i)
      NewOps[i] = SE.getAnyExtendExpr(S->op_begin()[i], CanonicalIV->getType());
    Value *V = expand(SE.getAddRecExpr(NewOps, S->getLoop(),
                                       S->getNoWrapFlags(SCEV::FlagNW)));
    BasicBlock::iterator NewInsertPt =
        findInsertPointAfter(cast<Instruction>(V), &*Builder.GetInsertPoint());
    V = expandCodeForImpl(SE.getTruncateExpr(SE.getUnknown(V), Ty), nullptr,
                          &*NewInsertPt, false);
    return V;
  }

  // {X,+,F} --> X + {0,+,F}
  if (!S->getStart()->isZero()) {
    SmallVector<const SCEV *, 4> NewOps(S->operands());
    NewOps[0] = SE.getConstant(Ty, 0);
    const SCEV *Rest = SE.getAddRecExpr(NewOps, L,
                                        S->getNoWrapFlags(SCEV::FlagNW));

    // Turn things like ptrtoint+arithmetic+inttoptr into GEP. See the
    // comments on expandAddToGEP for details.
    const SCEV *Base = S->getStart();
    // Dig into the expression to find the pointer base for a GEP.
    const SCEV *ExposedRest = Rest;
    ExposePointerBase(Base, ExposedRest, SE);
    // If we found a pointer, expand the AddRec with a GEP.
    if (PointerType *PTy = dyn_cast<PointerType>(Base->getType())) {
      // Make sure the Base isn't something exotic, such as a multiplied
      // or divided pointer value. In those cases, the result type isn't
      // actually a pointer type.
      if (!isa<SCEVMulExpr>(Base) && !isa<SCEVUDivExpr>(Base)) {
        Value *StartV = expand(Base);
        assert(StartV->getType() == PTy && "Pointer type mismatch for GEP!");
        return expandAddToGEP(ExposedRest, PTy, Ty, StartV);
      }
    }

    // Just do a normal add. Pre-expand the operands to suppress folding.
    //
    // The LHS and RHS values are factored out of the expand call to make the
    // output independent of the argument evaluation order.
    const SCEV *AddExprLHS = SE.getUnknown(expand(S->getStart()));
    const SCEV *AddExprRHS = SE.getUnknown(expand(Rest));
    return expand(SE.getAddExpr(AddExprLHS, AddExprRHS));
  }

  // If we don't yet have a canonical IV, create one.
  if (!CanonicalIV) {
    // Create and insert the PHI node for the induction variable in the
    // specified loop.
    BasicBlock *Header = L->getHeader();
    pred_iterator HPB = pred_begin(Header), HPE = pred_end(Header);
    CanonicalIV = PHINode::Create(Ty, std::distance(HPB, HPE), "indvar",
                                  &Header->front());
    rememberInstruction(CanonicalIV);

    SmallSet<BasicBlock *, 4> PredSeen;
    Constant *One = ConstantInt::get(Ty, 1);
    for (pred_iterator HPI = HPB; HPI != HPE; ++HPI) {
      BasicBlock *HP = *HPI;
      if (!PredSeen.insert(HP).second) {
        // There must be an incoming value for each predecessor, even the
        // duplicates!
        CanonicalIV->addIncoming(CanonicalIV->getIncomingValueForBlock(HP), HP);
        continue;
      }

      if (L->contains(HP)) {
        // Insert a unit add instruction right before the terminator
        // corresponding to the back-edge.
        Instruction *Add = BinaryOperator::CreateAdd(CanonicalIV, One,
                                                     "indvar.next",
                                                     HP->getTerminator());
        Add->setDebugLoc(HP->getTerminator()->getDebugLoc());
        rememberInstruction(Add);
        CanonicalIV->addIncoming(Add, HP);
      } else {
        CanonicalIV->addIncoming(Constant::getNullValue(Ty), HP);
      }
    }
  }

  // {0,+,1} --> Insert a canonical induction variable into the loop!
  if (S->isAffine() && S->getOperand(1)->isOne()) {
    assert(Ty == SE.getEffectiveSCEVType(CanonicalIV->getType()) &&
           "IVs with types different from the canonical IV should "
           "already have been handled!");
    return CanonicalIV;
  }

  // {0,+,F} --> {0,+,1} * F

  // If this is a simple linear addrec, emit it now as a special case.
  if (S->isAffine())    // {0,+,F} --> i*F
    return
      expand(SE.getTruncateOrNoop(
        SE.getMulExpr(SE.getUnknown(CanonicalIV),
                      SE.getNoopOrAnyExtend(S->getOperand(1),
                                            CanonicalIV->getType())),
        Ty));

  // If this is a chain of recurrences, turn it into a closed form, using the
  // folders, then expandCodeFor the closed form.  This allows the folders to
  // simplify the expression without having to build a bunch of special code
  // into this folder.
  const SCEV *IH = SE.getUnknown(CanonicalIV);   // Get I as a "symbolic" SCEV.

  // Promote S up to the canonical IV type, if the cast is foldable.
  const SCEV *NewS = S;
  const SCEV *Ext = SE.getNoopOrAnyExtend(S, CanonicalIV->getType());
  if (isa<SCEVAddRecExpr>(Ext))
    NewS = Ext;

  const SCEV *V = cast<SCEVAddRecExpr>(NewS)->evaluateAtIteration(IH, SE);
  //cerr << "Evaluated: " << *this << "\n     to: " << *V << "\n";

  // Truncate the result down to the original type, if needed.
  const SCEV *T = SE.getTruncateOrNoop(V, Ty);
  return expand(T);
}

Value *SCEVExpander::visitPtrToIntExpr(const SCEVPtrToIntExpr *S) {
  Value *V =
      expandCodeForImpl(S->getOperand(), S->getOperand()->getType(), false);
  return ReuseOrCreateCast(V, S->getType(), CastInst::PtrToInt,
                           GetOptimalInsertionPointForCastOf(V));
}

Value *SCEVExpander::visitTruncateExpr(const SCEVTruncateExpr *S) {
  Type *Ty = SE.getEffectiveSCEVType(S->getType());
  Value *V = expandCodeForImpl(
      S->getOperand(), SE.getEffectiveSCEVType(S->getOperand()->getType()),
      false);
  return Builder.CreateTrunc(V, Ty);
}

Value *SCEVExpander::visitZeroExtendExpr(const SCEVZeroExtendExpr *S) {
  Type *Ty = SE.getEffectiveSCEVType(S->getType());
  Value *V = expandCodeForImpl(
      S->getOperand(), SE.getEffectiveSCEVType(S->getOperand()->getType()),
      false);
  return Builder.CreateZExt(V, Ty);
}

Value *SCEVExpander::visitSignExtendExpr(const SCEVSignExtendExpr *S) {
  Type *Ty = SE.getEffectiveSCEVType(S->getType());
  Value *V = expandCodeForImpl(
      S->getOperand(), SE.getEffectiveSCEVType(S->getOperand()->getType()),
      false);
  return Builder.CreateSExt(V, Ty);
}

Value *SCEVExpander::visitSMaxExpr(const SCEVSMaxExpr *S) {
  Value *LHS = expand(S->getOperand(S->getNumOperands()-1));
  Type *Ty = LHS->getType();
  for (int i = S->getNumOperands()-2; i >= 0; --i) {
    // In the case of mixed integer and pointer types, do the
    // rest of the comparisons as integer.
    Type *OpTy = S->getOperand(i)->getType();
    if (OpTy->isIntegerTy() != Ty->isIntegerTy()) {
      Ty = SE.getEffectiveSCEVType(Ty);
      LHS = InsertNoopCastOfTo(LHS, Ty);
    }
    Value *RHS = expandCodeForImpl(S->getOperand(i), Ty, false);
    Value *Sel;
    if (Ty->isIntegerTy())
      Sel = Builder.CreateIntrinsic(Intrinsic::smax, {Ty}, {LHS, RHS},
                                    /*FMFSource=*/nullptr, "smax");
    else {
      Value *ICmp = Builder.CreateICmpSGT(LHS, RHS);
      Sel = Builder.CreateSelect(ICmp, LHS, RHS, "smax");
    }
    LHS = Sel;
  }
  // In the case of mixed integer and pointer types, cast the
  // final result back to the pointer type.
  if (LHS->getType() != S->getType())
    LHS = InsertNoopCastOfTo(LHS, S->getType());
  return LHS;
}

Value *SCEVExpander::visitUMaxExpr(const SCEVUMaxExpr *S) {
  Value *LHS = expand(S->getOperand(S->getNumOperands()-1));
  Type *Ty = LHS->getType();
  for (int i = S->getNumOperands()-2; i >= 0; --i) {
    // In the case of mixed integer and pointer types, do the
    // rest of the comparisons as integer.
    Type *OpTy = S->getOperand(i)->getType();
    if (OpTy->isIntegerTy() != Ty->isIntegerTy()) {
      Ty = SE.getEffectiveSCEVType(Ty);
      LHS = InsertNoopCastOfTo(LHS, Ty);
    }
    Value *RHS = expandCodeForImpl(S->getOperand(i), Ty, false);
    Value *Sel;
    if (Ty->isIntegerTy())
      Sel = Builder.CreateIntrinsic(Intrinsic::umax, {Ty}, {LHS, RHS},
                                    /*FMFSource=*/nullptr, "umax");
    else {
      Value *ICmp = Builder.CreateICmpUGT(LHS, RHS);
      Sel = Builder.CreateSelect(ICmp, LHS, RHS, "umax");
    }
    LHS = Sel;
  }
  // In the case of mixed integer and pointer types, cast the
  // final result back to the pointer type.
  if (LHS->getType() != S->getType())
    LHS = InsertNoopCastOfTo(LHS, S->getType());
  return LHS;
}

Value *SCEVExpander::visitSMinExpr(const SCEVSMinExpr *S) {
  Value *LHS = expand(S->getOperand(S->getNumOperands() - 1));
  Type *Ty = LHS->getType();
  for (int i = S->getNumOperands() - 2; i >= 0; --i) {
    // In the case of mixed integer and pointer types, do the
    // rest of the comparisons as integer.
    Type *OpTy = S->getOperand(i)->getType();
    if (OpTy->isIntegerTy() != Ty->isIntegerTy()) {
      Ty = SE.getEffectiveSCEVType(Ty);
      LHS = InsertNoopCastOfTo(LHS, Ty);
    }
    Value *RHS = expandCodeForImpl(S->getOperand(i), Ty, false);
    Value *Sel;
    if (Ty->isIntegerTy())
      Sel = Builder.CreateIntrinsic(Intrinsic::smin, {Ty}, {LHS, RHS},
                                    /*FMFSource=*/nullptr, "smin");
    else {
      Value *ICmp = Builder.CreateICmpSLT(LHS, RHS);
      Sel = Builder.CreateSelect(ICmp, LHS, RHS, "smin");
    }
    LHS = Sel;
  }
  // In the case of mixed integer and pointer types, cast the
  // final result back to the pointer type.
  if (LHS->getType() != S->getType())
    LHS = InsertNoopCastOfTo(LHS, S->getType());
  return LHS;
}

Value *SCEVExpander::visitUMinExpr(const SCEVUMinExpr *S) {
  Value *LHS = expand(S->getOperand(S->getNumOperands() - 1));
  Type *Ty = LHS->getType();
  for (int i = S->getNumOperands() - 2; i >= 0; --i) {
    // In the case of mixed integer and pointer types, do the
    // rest of the comparisons as integer.
    Type *OpTy = S->getOperand(i)->getType();
    if (OpTy->isIntegerTy() != Ty->isIntegerTy()) {
      Ty = SE.getEffectiveSCEVType(Ty);
      LHS = InsertNoopCastOfTo(LHS, Ty);
    }
    Value *RHS = expandCodeForImpl(S->getOperand(i), Ty, false);
    Value *Sel;
    if (Ty->isIntegerTy())
      Sel = Builder.CreateIntrinsic(Intrinsic::umin, {Ty}, {LHS, RHS},
                                    /*FMFSource=*/nullptr, "umin");
    else {
      Value *ICmp = Builder.CreateICmpULT(LHS, RHS);
      Sel = Builder.CreateSelect(ICmp, LHS, RHS, "umin");
    }
    LHS = Sel;
  }
  // In the case of mixed integer and pointer types, cast the
  // final result back to the pointer type.
  if (LHS->getType() != S->getType())
    LHS = InsertNoopCastOfTo(LHS, S->getType());
  return LHS;
}

Value *SCEVExpander::expandCodeForImpl(const SCEV *SH, Type *Ty,
                                       Instruction *IP, bool Root) {
  setInsertPoint(IP);
  Value *V = expandCodeForImpl(SH, Ty, Root);
  return V;
}

Value *SCEVExpander::expandCodeForImpl(const SCEV *SH, Type *Ty, bool Root) {
  // Expand the code for this SCEV.
  Value *V = expand(SH);

  if (PreserveLCSSA) {
    if (auto *Inst = dyn_cast<Instruction>(V)) {
      // Create a temporary instruction to at the current insertion point, so we
      // can hand it off to the helper to create LCSSA PHIs if required for the
      // new use.
      // FIXME: Ideally formLCSSAForInstructions (used in fixupLCSSAFormFor)
      // would accept a insertion point and return an LCSSA phi for that
      // insertion point, so there is no need to insert & remove the temporary
      // instruction.
      Instruction *Tmp;
      if (Inst->getType()->isIntegerTy())
        Tmp =
            cast<Instruction>(Builder.CreateAdd(Inst, Inst, "tmp.lcssa.user"));
      else {
        assert(Inst->getType()->isPointerTy());
        Tmp = cast<Instruction>(Builder.CreatePtrToInt(
            Inst, Type::getInt32Ty(Inst->getContext()), "tmp.lcssa.user"));
      }
      V = fixupLCSSAFormFor(Tmp, 0);

      // Clean up temporary instruction.
      InsertedValues.erase(Tmp);
      InsertedPostIncValues.erase(Tmp);
      Tmp->eraseFromParent();
    }
  }

  InsertedExpressions[std::make_pair(SH, &*Builder.GetInsertPoint())] = V;
  if (Ty) {
    assert(SE.getTypeSizeInBits(Ty) == SE.getTypeSizeInBits(SH->getType()) &&
           "non-trivial casts should be done with the SCEVs directly!");
    V = InsertNoopCastOfTo(V, Ty);
  }
  return V;
}

ScalarEvolution::ValueOffsetPair
SCEVExpander::FindValueInExprValueMap(const SCEV *S,
                                      const Instruction *InsertPt) {
  auto *Set = SE.getSCEVValues(S);
  // If the expansion is not in CanonicalMode, and the SCEV contains any
  // sub scAddRecExpr type SCEV, it is required to expand the SCEV literally.
  if (CanonicalMode || !SE.containsAddRecurrence(S)) {
    // If S is scConstant, it may be worse to reuse an existing Value.
    if (S->getSCEVType() != scConstant && Set) {
      // Choose a Value from the set which dominates the insertPt.
      // insertPt should be inside the Value's parent loop so as not to break
      // the LCSSA form.
      for (auto const &VOPair : *Set) {
        Value *V = VOPair.first;
        ConstantInt *Offset = VOPair.second;
        Instruction *EntInst = nullptr;
        if (V && isa<Instruction>(V) && (EntInst = cast<Instruction>(V)) &&
            S->getType() == V->getType() &&
            EntInst->getFunction() == InsertPt->getFunction() &&
            SE.DT.dominates(EntInst, InsertPt) &&
            (SE.LI.getLoopFor(EntInst->getParent()) == nullptr ||
             SE.LI.getLoopFor(EntInst->getParent())->contains(InsertPt)))
          return {V, Offset};
      }
    }
  }
  return {nullptr, nullptr};
}

// The expansion of SCEV will either reuse a previous Value in ExprValueMap,
// or expand the SCEV literally. Specifically, if the expansion is in LSRMode,
// and the SCEV contains any sub scAddRecExpr type SCEV, it will be expanded
// literally, to prevent LSR's transformed SCEV from being reverted. Otherwise,
// the expansion will try to reuse Value from ExprValueMap, and only when it
// fails, expand the SCEV literally.
Value *SCEVExpander::expand(const SCEV *S) {
  // Compute an insertion point for this SCEV object. Hoist the instructions
  // as far out in the loop nest as possible.
  Instruction *InsertPt = &*Builder.GetInsertPoint();

  // We can move insertion point only if there is no div or rem operations
  // otherwise we are risky to move it over the check for zero denominator.
  auto SafeToHoist = [](const SCEV *S) {
    return !SCEVExprContains(S, [](const SCEV *S) {
              if (const auto *D = dyn_cast<SCEVUDivExpr>(S)) {
                if (const auto *SC = dyn_cast<SCEVConstant>(D->getRHS()))
                  // Division by non-zero constants can be hoisted.
                  return SC->getValue()->isZero();
                // All other divisions should not be moved as they may be
                // divisions by zero and should be kept within the
                // conditions of the surrounding loops that guard their
                // execution (see PR35406).
                return true;
              }
              return false;
            });
  };
  if (SafeToHoist(S)) {
    for (Loop *L = SE.LI.getLoopFor(Builder.GetInsertBlock());;
         L = L->getParentLoop()) {
      if (SE.isLoopInvariant(S, L)) {
        if (!L) break;
        if (BasicBlock *Preheader = L->getLoopPreheader())
          InsertPt = Preheader->getTerminator();
        else
          // LSR sets the insertion point for AddRec start/step values to the
          // block start to simplify value reuse, even though it's an invalid
          // position. SCEVExpander must correct for this in all cases.
          InsertPt = &*L->getHeader()->getFirstInsertionPt();
      } else {
        // If the SCEV is computable at this level, insert it into the header
        // after the PHIs (and after any other instructions that we've inserted
        // there) so that it is guaranteed to dominate any user inside the loop.
        if (L && SE.hasComputableLoopEvolution(S, L) && !PostIncLoops.count(L))
          InsertPt = &*L->getHeader()->getFirstInsertionPt();

        while (InsertPt->getIterator() != Builder.GetInsertPoint() &&
               (isInsertedInstruction(InsertPt) ||
                isa<DbgInfoIntrinsic>(InsertPt))) {
          InsertPt = &*std::next(InsertPt->getIterator());
        }
        break;
      }
    }
  }

  // Check to see if we already expanded this here.
  auto I = InsertedExpressions.find(std::make_pair(S, InsertPt));
  if (I != InsertedExpressions.end())
    return I->second;

  SCEVInsertPointGuard Guard(Builder, this);
  Builder.SetInsertPoint(InsertPt);

  // Expand the expression into instructions.
  ScalarEvolution::ValueOffsetPair VO = FindValueInExprValueMap(S, InsertPt);
  Value *V = VO.first;

  if (!V)
    V = visit(S);
  else if (VO.second) {
    if (PointerType *Vty = dyn_cast<PointerType>(V->getType())) {
      Type *Ety = Vty->getPointerElementType();
      int64_t Offset = VO.second->getSExtValue();
      int64_t ESize = SE.getTypeSizeInBits(Ety);
      if ((Offset * 8) % ESize == 0) {
        ConstantInt *Idx =
            ConstantInt::getSigned(VO.second->getType(), -(Offset * 8) / ESize);
        V = Builder.CreateGEP(Ety, V, Idx, "scevgep");
      } else {
        ConstantInt *Idx =
            ConstantInt::getSigned(VO.second->getType(), -Offset);
        unsigned AS = Vty->getAddressSpace();
        V = Builder.CreateBitCast(V, Type::getInt8PtrTy(SE.getContext(), AS));
        V = Builder.CreateGEP(Type::getInt8Ty(SE.getContext()), V, Idx,
                              "uglygep");
        V = Builder.CreateBitCast(V, Vty);
      }
    } else {
      V = Builder.CreateSub(V, VO.second);
    }
  }
  // Remember the expanded value for this SCEV at this location.
  //
  // This is independent of PostIncLoops. The mapped value simply materializes
  // the expression at this insertion point. If the mapped value happened to be
  // a postinc expansion, it could be reused by a non-postinc user, but only if
  // its insertion point was already at the head of the loop.
  InsertedExpressions[std::make_pair(S, InsertPt)] = V;
  return V;
}

void SCEVExpander::rememberInstruction(Value *I) {
  auto DoInsert = [this](Value *V) {
    if (!PostIncLoops.empty())
      InsertedPostIncValues.insert(V);
    else
      InsertedValues.insert(V);
  };
  DoInsert(I);

  if (!PreserveLCSSA)
    return;

  if (auto *Inst = dyn_cast<Instruction>(I)) {
    // A new instruction has been added, which might introduce new uses outside
    // a defining loop. Fix LCSSA from for each operand of the new instruction,
    // if required.
    for (unsigned OpIdx = 0, OpEnd = Inst->getNumOperands(); OpIdx != OpEnd;
         OpIdx++)
      fixupLCSSAFormFor(Inst, OpIdx);
  }
}

/// replaceCongruentIVs - Check for congruent phis in this loop header and
/// replace them with their most canonical representative. Return the number of
/// phis eliminated.
///
/// This does not depend on any SCEVExpander state but should be used in
/// the same context that SCEVExpander is used.
unsigned
SCEVExpander::replaceCongruentIVs(Loop *L, const DominatorTree *DT,
                                  SmallVectorImpl<WeakTrackingVH> &DeadInsts,
                                  const TargetTransformInfo *TTI) {
  // Find integer phis in order of increasing width.
  SmallVector<PHINode*, 8> Phis;
  for (PHINode &PN : L->getHeader()->phis())
    Phis.push_back(&PN);

  if (TTI)
    llvm::sort(Phis, [](Value *LHS, Value *RHS) {
      // Put pointers at the back and make sure pointer < pointer = false.
      if (!LHS->getType()->isIntegerTy() || !RHS->getType()->isIntegerTy())
        return RHS->getType()->isIntegerTy() && !LHS->getType()->isIntegerTy();
      return RHS->getType()->getPrimitiveSizeInBits().getFixedSize() <
             LHS->getType()->getPrimitiveSizeInBits().getFixedSize();
    });

  unsigned NumElim = 0;
  DenseMap<const SCEV *, PHINode *> ExprToIVMap;
  // Process phis from wide to narrow. Map wide phis to their truncation
  // so narrow phis can reuse them.
  for (PHINode *Phi : Phis) {
    auto SimplifyPHINode = [&](PHINode *PN) -> Value * {
      if (Value *V = SimplifyInstruction(PN, {DL, &SE.TLI, &SE.DT, &SE.AC}))
        return V;
      if (!SE.isSCEVable(PN->getType()))
        return nullptr;
      auto *Const = dyn_cast<SCEVConstant>(SE.getSCEV(PN));
      if (!Const)
        return nullptr;
      return Const->getValue();
    };

    // Fold constant phis. They may be congruent to other constant phis and
    // would confuse the logic below that expects proper IVs.
    if (Value *V = SimplifyPHINode(Phi)) {
      if (V->getType() != Phi->getType())
        continue;
      Phi->replaceAllUsesWith(V);
      DeadInsts.emplace_back(Phi);
      ++NumElim;
      SCEV_DEBUG_WITH_TYPE(DebugType,
                           dbgs() << "INDVARS: Eliminated constant iv: " << *Phi
                                  << '\n');
      continue;
    }

    if (!SE.isSCEVable(Phi->getType()))
      continue;

    PHINode *&OrigPhiRef = ExprToIVMap[SE.getSCEV(Phi)];
    if (!OrigPhiRef) {
      OrigPhiRef = Phi;
      if (Phi->getType()->isIntegerTy() && TTI &&
          TTI->isTruncateFree(Phi->getType(), Phis.back()->getType())) {
        // This phi can be freely truncated to the narrowest phi type. Map the
        // truncated expression to it so it will be reused for narrow types.
        const SCEV *TruncExpr =
          SE.getTruncateExpr(SE.getSCEV(Phi), Phis.back()->getType());
        ExprToIVMap[TruncExpr] = Phi;
      }
      continue;
    }

    // Replacing a pointer phi with an integer phi or vice-versa doesn't make
    // sense.
    if (OrigPhiRef->getType()->isPointerTy() != Phi->getType()->isPointerTy())
      continue;

    if (BasicBlock *LatchBlock = L->getLoopLatch()) {
      Instruction *OrigInc = dyn_cast<Instruction>(
          OrigPhiRef->getIncomingValueForBlock(LatchBlock));
      Instruction *IsomorphicInc =
          dyn_cast<Instruction>(Phi->getIncomingValueForBlock(LatchBlock));

      if (OrigInc && IsomorphicInc) {
        // If this phi has the same width but is more canonical, replace the
        // original with it. As part of the "more canonical" determination,
        // respect a prior decision to use an IV chain.
        if (OrigPhiRef->getType() == Phi->getType() &&
            !(ChainedPhis.count(Phi) ||
              isExpandedAddRecExprPHI(OrigPhiRef, OrigInc, L)) &&
            (ChainedPhis.count(Phi) ||
             isExpandedAddRecExprPHI(Phi, IsomorphicInc, L))) {
          std::swap(OrigPhiRef, Phi);
          std::swap(OrigInc, IsomorphicInc);
        }
        // Replacing the congruent phi is sufficient because acyclic
        // redundancy elimination, CSE/GVN, should handle the
        // rest. However, once SCEV proves that a phi is congruent,
        // it's often the head of an IV user cycle that is isomorphic
        // with the original phi. It's worth eagerly cleaning up the
        // common case of a single IV increment so that DeleteDeadPHIs
        // can remove cycles that had postinc uses.
        const SCEV *TruncExpr =
            SE.getTruncateOrNoop(SE.getSCEV(OrigInc), IsomorphicInc->getType());
        if (OrigInc != IsomorphicInc &&
            TruncExpr == SE.getSCEV(IsomorphicInc) &&
            SE.LI.replacementPreservesLCSSAForm(IsomorphicInc, OrigInc) &&
            hoistIVInc(OrigInc, IsomorphicInc)) {
          SCEV_DEBUG_WITH_TYPE(
              DebugType, dbgs() << "INDVARS: Eliminated congruent iv.inc: "
                                << *IsomorphicInc << '\n');
          Value *NewInc = OrigInc;
          if (OrigInc->getType() != IsomorphicInc->getType()) {
            Instruction *IP = nullptr;
            if (PHINode *PN = dyn_cast<PHINode>(OrigInc))
              IP = &*PN->getParent()->getFirstInsertionPt();
            else
              IP = OrigInc->getNextNode();

            IRBuilder<> Builder(IP);
            Builder.SetCurrentDebugLocation(IsomorphicInc->getDebugLoc());
            NewInc = Builder.CreateTruncOrBitCast(
                OrigInc, IsomorphicInc->getType(), IVName);
          }
          IsomorphicInc->replaceAllUsesWith(NewInc);
          DeadInsts.emplace_back(IsomorphicInc);
        }
      }
    }
    SCEV_DEBUG_WITH_TYPE(DebugType,
                         dbgs() << "INDVARS: Eliminated congruent iv: " << *Phi
                                << '\n');
    SCEV_DEBUG_WITH_TYPE(
        DebugType, dbgs() << "INDVARS: Original iv: " << *OrigPhiRef << '\n');
    ++NumElim;
    Value *NewIV = OrigPhiRef;
    if (OrigPhiRef->getType() != Phi->getType()) {
      IRBuilder<> Builder(&*L->getHeader()->getFirstInsertionPt());
      Builder.SetCurrentDebugLocation(Phi->getDebugLoc());
      NewIV = Builder.CreateTruncOrBitCast(OrigPhiRef, Phi->getType(), IVName);
    }
    Phi->replaceAllUsesWith(NewIV);
    DeadInsts.emplace_back(Phi);
  }
  return NumElim;
}

Optional<ScalarEvolution::ValueOffsetPair>
SCEVExpander::getRelatedExistingExpansion(const SCEV *S, const Instruction *At,
                                          Loop *L) {
  using namespace llvm::PatternMatch;

  SmallVector<BasicBlock *, 4> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);

  // Look for suitable value in simple conditions at the loop exits.
  for (BasicBlock *BB : ExitingBlocks) {
    ICmpInst::Predicate Pred;
    Instruction *LHS, *RHS;

    if (!match(BB->getTerminator(),
               m_Br(m_ICmp(Pred, m_Instruction(LHS), m_Instruction(RHS)),
                    m_BasicBlock(), m_BasicBlock())))
      continue;

    if (SE.getSCEV(LHS) == S && SE.DT.dominates(LHS, At))
      return ScalarEvolution::ValueOffsetPair(LHS, nullptr);

    if (SE.getSCEV(RHS) == S && SE.DT.dominates(RHS, At))
      return ScalarEvolution::ValueOffsetPair(RHS, nullptr);
  }

  // Use expand's logic which is used for reusing a previous Value in
  // ExprValueMap.
  ScalarEvolution::ValueOffsetPair VO = FindValueInExprValueMap(S, At);
  if (VO.first)
    return VO;

  // There is potential to make this significantly smarter, but this simple
  // heuristic already gets some interesting cases.

  // Can not find suitable value.
  return None;
}

template<typename T> static InstructionCost costAndCollectOperands(
  const SCEVOperand &WorkItem, const TargetTransformInfo &TTI,
  TargetTransformInfo::TargetCostKind CostKind,
  SmallVectorImpl<SCEVOperand> &Worklist) {

  const T *S = cast<T>(WorkItem.S);
  InstructionCost Cost = 0;
  // Object to help map SCEV operands to expanded IR instructions.
  struct OperationIndices {
    OperationIndices(unsigned Opc, size_t min, size_t max) :
      Opcode(Opc), MinIdx(min), MaxIdx(max) { }
    unsigned Opcode;
    size_t MinIdx;
    size_t MaxIdx;
  };

  // Collect the operations of all the instructions that will be needed to
  // expand the SCEVExpr. This is so that when we come to cost the operands,
  // we know what the generated user(s) will be.
  SmallVector<OperationIndices, 2> Operations;

  auto CastCost = [&](unsigned Opcode) -> InstructionCost {
    Operations.emplace_back(Opcode, 0, 0);
    return TTI.getCastInstrCost(Opcode, S->getType(),
                                S->getOperand(0)->getType(),
                                TTI::CastContextHint::None, CostKind);
  };

  auto ArithCost = [&](unsigned Opcode, unsigned NumRequired,
                       unsigned MinIdx = 0,
                       unsigned MaxIdx = 1) -> InstructionCost {
    Operations.emplace_back(Opcode, MinIdx, MaxIdx);
    return NumRequired *
      TTI.getArithmeticInstrCost(Opcode, S->getType(), CostKind);
  };

  auto CmpSelCost = [&](unsigned Opcode, unsigned NumRequired, unsigned MinIdx,
                        unsigned MaxIdx) -> InstructionCost {
    Operations.emplace_back(Opcode, MinIdx, MaxIdx);
    Type *OpType = S->getOperand(0)->getType();
    return NumRequired * TTI.getCmpSelInstrCost(
                             Opcode, OpType, CmpInst::makeCmpResultType(OpType),
                             CmpInst::BAD_ICMP_PREDICATE, CostKind);
  };

  switch (S->getSCEVType()) {
  case scCouldNotCompute:
    llvm_unreachable("Attempt to use a SCEVCouldNotCompute object!");
  case scUnknown:
  case scConstant:
    return 0;
  case scPtrToInt:
    Cost = CastCost(Instruction::PtrToInt);
    break;
  case scTruncate:
    Cost = CastCost(Instruction::Trunc);
    break;
  case scZeroExtend:
    Cost = CastCost(Instruction::ZExt);
    break;
  case scSignExtend:
    Cost = CastCost(Instruction::SExt);
    break;
  case scUDivExpr: {
    unsigned Opcode = Instruction::UDiv;
    if (auto *SC = dyn_cast<SCEVConstant>(S->getOperand(1)))
      if (SC->getAPInt().isPowerOf2())
        Opcode = Instruction::LShr;
    Cost = ArithCost(Opcode, 1);
    break;
  }
  case scAddExpr:
    Cost = ArithCost(Instruction::Add, S->getNumOperands() - 1);
    break;
  case scMulExpr:
    // TODO: this is a very pessimistic cost modelling for Mul,
    // because of Bin Pow algorithm actually used by the expander,
    // see SCEVExpander::visitMulExpr(), ExpandOpBinPowN().
    Cost = ArithCost(Instruction::Mul, S->getNumOperands() - 1);
    break;
  case scSMaxExpr:
  case scUMaxExpr:
  case scSMinExpr:
  case scUMinExpr: {
    // FIXME: should this ask the cost for Intrinsic's?
    Cost += CmpSelCost(Instruction::ICmp, S->getNumOperands() - 1, 0, 1);
    Cost += CmpSelCost(Instruction::Select, S->getNumOperands() - 1, 0, 2);
    break;
  }
  case scAddRecExpr: {
    // In this polynominal, we may have some zero operands, and we shouldn't
    // really charge for those. So how many non-zero coeffients are there?
    int NumTerms = llvm::count_if(S->operands(), [](const SCEV *Op) {
                                    return !Op->isZero();
                                  });

    assert(NumTerms >= 1 && "Polynominal should have at least one term.");
    assert(!(*std::prev(S->operands().end()))->isZero() &&
           "Last operand should not be zero");

    // Ignoring constant term (operand 0), how many of the coeffients are u> 1?
    int NumNonZeroDegreeNonOneTerms =
      llvm::count_if(S->operands(), [](const SCEV *Op) {
                      auto *SConst = dyn_cast<SCEVConstant>(Op);
                      return !SConst || SConst->getAPInt().ugt(1);
                    });

    // Much like with normal add expr, the polynominal will require
    // one less addition than the number of it's terms.
    InstructionCost AddCost = ArithCost(Instruction::Add, NumTerms - 1,
                                        /*MinIdx*/ 1, /*MaxIdx*/ 1);
    // Here, *each* one of those will require a multiplication.
    InstructionCost MulCost =
        ArithCost(Instruction::Mul, NumNonZeroDegreeNonOneTerms);
    Cost = AddCost + MulCost;

    // What is the degree of this polynominal?
    int PolyDegree = S->getNumOperands() - 1;
    assert(PolyDegree >= 1 && "Should be at least affine.");

    // The final term will be:
    //   Op_{PolyDegree} * x ^ {PolyDegree}
    // Where  x ^ {PolyDegree}  will again require PolyDegree-1 mul operations.
    // Note that  x ^ {PolyDegree} = x * x ^ {PolyDegree-1}  so charging for
    // x ^ {PolyDegree}  will give us  x ^ {2} .. x ^ {PolyDegree-1}  for free.
    // FIXME: this is conservatively correct, but might be overly pessimistic.
    Cost += MulCost * (PolyDegree - 1);
    break;
  }
  }

  for (auto &CostOp : Operations) {
    for (auto SCEVOp : enumerate(S->operands())) {
      // Clamp the index to account for multiple IR operations being chained.
      size_t MinIdx = std::max(SCEVOp.index(), CostOp.MinIdx);
      size_t OpIdx = std::min(MinIdx, CostOp.MaxIdx);
      Worklist.emplace_back(CostOp.Opcode, OpIdx, SCEVOp.value());
    }
  }
  return Cost;
}

bool SCEVExpander::isHighCostExpansionHelper(
    const SCEVOperand &WorkItem, Loop *L, const Instruction &At,
    InstructionCost &Cost, unsigned Budget, const TargetTransformInfo &TTI,
    SmallPtrSetImpl<const SCEV *> &Processed,
    SmallVectorImpl<SCEVOperand> &Worklist) {
  if (Cost > Budget)
    return true; // Already run out of budget, give up.

  const SCEV *S = WorkItem.S;
  // Was the cost of expansion of this expression already accounted for?
  if (!isa<SCEVConstant>(S) && !Processed.insert(S).second)
    return false; // We have already accounted for this expression.

  // If we can find an existing value for this scev available at the point "At"
  // then consider the expression cheap.
  if (getRelatedExistingExpansion(S, &At, L))
    return false; // Consider the expression to be free.

  TargetTransformInfo::TargetCostKind CostKind =
      L->getHeader()->getParent()->hasMinSize()
          ? TargetTransformInfo::TCK_CodeSize
          : TargetTransformInfo::TCK_RecipThroughput;

  switch (S->getSCEVType()) {
  case scCouldNotCompute:
    llvm_unreachable("Attempt to use a SCEVCouldNotCompute object!");
  case scUnknown:
    // Assume to be zero-cost.
    return false;
  case scConstant: {
    // Only evalulate the costs of constants when optimizing for size.
    if (CostKind != TargetTransformInfo::TCK_CodeSize)
      return 0;
    const APInt &Imm = cast<SCEVConstant>(S)->getAPInt();
    Type *Ty = S->getType();
    Cost += TTI.getIntImmCostInst(
        WorkItem.ParentOpcode, WorkItem.OperandIdx, Imm, Ty, CostKind);
    return Cost > Budget;
  }
  case scTruncate:
  case scPtrToInt:
  case scZeroExtend:
  case scSignExtend: {
    Cost +=
        costAndCollectOperands<SCEVCastExpr>(WorkItem, TTI, CostKind, Worklist);
    return false; // Will answer upon next entry into this function.
  }
  case scUDivExpr: {
    // UDivExpr is very likely a UDiv that ScalarEvolution's HowFarToZero or
    // HowManyLessThans produced to compute a precise expression, rather than a
    // UDiv from the user's code. If we can't find a UDiv in the code with some
    // simple searching, we need to account for it's cost.

    // At the beginning of this function we already tried to find existing
    // value for plain 'S'. Now try to lookup 'S + 1' since it is common
    // pattern involving division. This is just a simple search heuristic.
    if (getRelatedExistingExpansion(
            SE.getAddExpr(S, SE.getConstant(S->getType(), 1)), &At, L))
      return false; // Consider it to be free.

    Cost +=
        costAndCollectOperands<SCEVUDivExpr>(WorkItem, TTI, CostKind, Worklist);
    return false; // Will answer upon next entry into this function.
  }
  case scAddExpr:
  case scMulExpr:
  case scUMaxExpr:
  case scSMaxExpr:
  case scUMinExpr:
  case scSMinExpr: {
    assert(cast<SCEVNAryExpr>(S)->getNumOperands() > 1 &&
           "Nary expr should have more than 1 operand.");
    // The simple nary expr will require one less op (or pair of ops)
    // than the number of it's terms.
    Cost +=
        costAndCollectOperands<SCEVNAryExpr>(WorkItem, TTI, CostKind, Worklist);
    return Cost > Budget;
  }
  case scAddRecExpr: {
    assert(cast<SCEVAddRecExpr>(S)->getNumOperands() >= 2 &&
           "Polynomial should be at least linear");
    Cost += costAndCollectOperands<SCEVAddRecExpr>(
        WorkItem, TTI, CostKind, Worklist);
    return Cost > Budget;
  }
  }
  llvm_unreachable("Unknown SCEV kind!");
}

Value *SCEVExpander::expandCodeForPredicate(const SCEVPredicate *Pred,
                                            Instruction *IP) {
  assert(IP);
  switch (Pred->getKind()) {
  case SCEVPredicate::P_Union:
    return expandUnionPredicate(cast<SCEVUnionPredicate>(Pred), IP);
  case SCEVPredicate::P_Equal:
    return expandEqualPredicate(cast<SCEVEqualPredicate>(Pred), IP);
  case SCEVPredicate::P_Wrap: {
    auto *AddRecPred = cast<SCEVWrapPredicate>(Pred);
    return expandWrapPredicate(AddRecPred, IP);
  }
  }
  llvm_unreachable("Unknown SCEV predicate type");
}

Value *SCEVExpander::expandEqualPredicate(const SCEVEqualPredicate *Pred,
                                          Instruction *IP) {
  Value *Expr0 =
      expandCodeForImpl(Pred->getLHS(), Pred->getLHS()->getType(), IP, false);
  Value *Expr1 =
      expandCodeForImpl(Pred->getRHS(), Pred->getRHS()->getType(), IP, false);

  Builder.SetInsertPoint(IP);
  auto *I = Builder.CreateICmpNE(Expr0, Expr1, "ident.check");
  return I;
}

Value *SCEVExpander::generateOverflowCheck(const SCEVAddRecExpr *AR,
                                           Instruction *Loc, bool Signed) {
  assert(AR->isAffine() && "Cannot generate RT check for "
                           "non-affine expression");

  SCEVUnionPredicate Pred;
  const SCEV *ExitCount =
      SE.getPredicatedBackedgeTakenCount(AR->getLoop(), Pred);

  assert(!isa<SCEVCouldNotCompute>(ExitCount) && "Invalid loop count");

  const SCEV *Step = AR->getStepRecurrence(SE);
  const SCEV *Start = AR->getStart();

  Type *ARTy = AR->getType();
  unsigned SrcBits = SE.getTypeSizeInBits(ExitCount->getType());
  unsigned DstBits = SE.getTypeSizeInBits(ARTy);

  // The expression {Start,+,Step} has nusw/nssw if
  //   Step < 0, Start - |Step| * Backedge <= Start
  //   Step >= 0, Start + |Step| * Backedge > Start
  // and |Step| * Backedge doesn't unsigned overflow.

  IntegerType *CountTy = IntegerType::get(Loc->getContext(), SrcBits);
  Builder.SetInsertPoint(Loc);
  Value *TripCountVal = expandCodeForImpl(ExitCount, CountTy, Loc, false);

  IntegerType *Ty =
      IntegerType::get(Loc->getContext(), SE.getTypeSizeInBits(ARTy));
  Type *ARExpandTy = DL.isNonIntegralPointerType(ARTy) ? ARTy : Ty;

  Value *StepValue = expandCodeForImpl(Step, Ty, Loc, false);
  Value *NegStepValue =
      expandCodeForImpl(SE.getNegativeSCEV(Step), Ty, Loc, false);
  Value *StartValue = expandCodeForImpl(
      isa<PointerType>(ARExpandTy) ? Start
                                   : SE.getPtrToIntExpr(Start, ARExpandTy),
      ARExpandTy, Loc, false);

  ConstantInt *Zero =
      ConstantInt::get(Loc->getContext(), APInt::getNullValue(DstBits));

  Builder.SetInsertPoint(Loc);
  // Compute |Step|
  Value *StepCompare = Builder.CreateICmp(ICmpInst::ICMP_SLT, StepValue, Zero);
  Value *AbsStep = Builder.CreateSelect(StepCompare, NegStepValue, StepValue);

  // Get the backedge taken count and truncate or extended to the AR type.
  Value *TruncTripCount = Builder.CreateZExtOrTrunc(TripCountVal, Ty);
  auto *MulF = Intrinsic::getDeclaration(Loc->getModule(),
                                         Intrinsic::umul_with_overflow, Ty);

  // Compute |Step| * Backedge
  CallInst *Mul = Builder.CreateCall(MulF, {AbsStep, TruncTripCount}, "mul");
  Value *MulV = Builder.CreateExtractValue(Mul, 0, "mul.result");
  Value *OfMul = Builder.CreateExtractValue(Mul, 1, "mul.overflow");

  // Compute:
  //   Start + |Step| * Backedge < Start
  //   Start - |Step| * Backedge > Start
  Value *Add = nullptr, *Sub = nullptr;
  if (PointerType *ARPtrTy = dyn_cast<PointerType>(ARExpandTy)) {
    const SCEV *MulS = SE.getSCEV(MulV);
    const SCEV *NegMulS = SE.getNegativeSCEV(MulS);
    Add = Builder.CreateBitCast(expandAddToGEP(MulS, ARPtrTy, Ty, StartValue),
                                ARPtrTy);
    Sub = Builder.CreateBitCast(
        expandAddToGEP(NegMulS, ARPtrTy, Ty, StartValue), ARPtrTy);
  } else {
    Add = Builder.CreateAdd(StartValue, MulV);
    Sub = Builder.CreateSub(StartValue, MulV);
  }

  Value *EndCompareGT = Builder.CreateICmp(
      Signed ? ICmpInst::ICMP_SGT : ICmpInst::ICMP_UGT, Sub, StartValue);

  Value *EndCompareLT = Builder.CreateICmp(
      Signed ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT, Add, StartValue);

  // Select the answer based on the sign of Step.
  Value *EndCheck =
      Builder.CreateSelect(StepCompare, EndCompareGT, EndCompareLT);

  // If the backedge taken count type is larger than the AR type,
  // check that we don't drop any bits by truncating it. If we are
  // dropping bits, then we have overflow (unless the step is zero).
  if (SE.getTypeSizeInBits(CountTy) > SE.getTypeSizeInBits(Ty)) {
    auto MaxVal = APInt::getMaxValue(DstBits).zext(SrcBits);
    auto *BackedgeCheck =
        Builder.CreateICmp(ICmpInst::ICMP_UGT, TripCountVal,
                           ConstantInt::get(Loc->getContext(), MaxVal));
    BackedgeCheck = Builder.CreateAnd(
        BackedgeCheck, Builder.CreateICmp(ICmpInst::ICMP_NE, StepValue, Zero));

    EndCheck = Builder.CreateOr(EndCheck, BackedgeCheck);
  }

  return Builder.CreateOr(EndCheck, OfMul);
}

Value *SCEVExpander::expandWrapPredicate(const SCEVWrapPredicate *Pred,
                                         Instruction *IP) {
  const auto *A = cast<SCEVAddRecExpr>(Pred->getExpr());
  Value *NSSWCheck = nullptr, *NUSWCheck = nullptr;

  // Add a check for NUSW
  if (Pred->getFlags() & SCEVWrapPredicate::IncrementNUSW)
    NUSWCheck = generateOverflowCheck(A, IP, false);

  // Add a check for NSSW
  if (Pred->getFlags() & SCEVWrapPredicate::IncrementNSSW)
    NSSWCheck = generateOverflowCheck(A, IP, true);

  if (NUSWCheck && NSSWCheck)
    return Builder.CreateOr(NUSWCheck, NSSWCheck);

  if (NUSWCheck)
    return NUSWCheck;

  if (NSSWCheck)
    return NSSWCheck;

  return ConstantInt::getFalse(IP->getContext());
}

Value *SCEVExpander::expandUnionPredicate(const SCEVUnionPredicate *Union,
                                          Instruction *IP) {
  auto *BoolType = IntegerType::get(IP->getContext(), 1);
  Value *Check = ConstantInt::getNullValue(BoolType);

  // Loop over all checks in this set.
  for (auto Pred : Union->getPredicates()) {
    auto *NextCheck = expandCodeForPredicate(Pred, IP);
    Builder.SetInsertPoint(IP);
    Check = Builder.CreateOr(Check, NextCheck);
  }

  return Check;
}

Value *SCEVExpander::fixupLCSSAFormFor(Instruction *User, unsigned OpIdx) {
  assert(PreserveLCSSA);
  SmallVector<Instruction *, 1> ToUpdate;

  auto *OpV = User->getOperand(OpIdx);
  auto *OpI = dyn_cast<Instruction>(OpV);
  if (!OpI)
    return OpV;

  Loop *DefLoop = SE.LI.getLoopFor(OpI->getParent());
  Loop *UseLoop = SE.LI.getLoopFor(User->getParent());
  if (!DefLoop || UseLoop == DefLoop || DefLoop->contains(UseLoop))
    return OpV;

  ToUpdate.push_back(OpI);
  SmallVector<PHINode *, 16> PHIsToRemove;
  formLCSSAForInstructions(ToUpdate, SE.DT, SE.LI, &SE, Builder, &PHIsToRemove);
  for (PHINode *PN : PHIsToRemove) {
    if (!PN->use_empty())
      continue;
    InsertedValues.erase(PN);
    InsertedPostIncValues.erase(PN);
    PN->eraseFromParent();
  }

  return User->getOperand(OpIdx);
}

namespace {
// Search for a SCEV subexpression that is not safe to expand.  Any expression
// that may expand to a !isSafeToSpeculativelyExecute value is unsafe, namely
// UDiv expressions. We don't know if the UDiv is derived from an IR divide
// instruction, but the important thing is that we prove the denominator is
// nonzero before expansion.
//
// IVUsers already checks that IV-derived expressions are safe. So this check is
// only needed when the expression includes some subexpression that is not IV
// derived.
//
// Currently, we only allow division by a nonzero constant here. If this is
// inadequate, we could easily allow division by SCEVUnknown by using
// ValueTracking to check isKnownNonZero().
//
// We cannot generally expand recurrences unless the step dominates the loop
// header. The expander handles the special case of affine recurrences by
// scaling the recurrence outside the loop, but this technique isn't generally
// applicable. Expanding a nested recurrence outside a loop requires computing
// binomial coefficients. This could be done, but the recurrence has to be in a
// perfectly reduced form, which can't be guaranteed.
struct SCEVFindUnsafe {
  ScalarEvolution &SE;
  bool IsUnsafe;

  SCEVFindUnsafe(ScalarEvolution &se): SE(se), IsUnsafe(false) {}

  bool follow(const SCEV *S) {
    if (const SCEVUDivExpr *D = dyn_cast<SCEVUDivExpr>(S)) {
      const SCEVConstant *SC = dyn_cast<SCEVConstant>(D->getRHS());
      if (!SC || SC->getValue()->isZero()) {
        IsUnsafe = true;
        return false;
      }
    }
    if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S)) {
      const SCEV *Step = AR->getStepRecurrence(SE);
      if (!AR->isAffine() && !SE.dominates(Step, AR->getLoop()->getHeader())) {
        IsUnsafe = true;
        return false;
      }
    }
    return true;
  }
  bool isDone() const { return IsUnsafe; }
};
}

namespace llvm {
bool isSafeToExpand(const SCEV *S, ScalarEvolution &SE) {
  SCEVFindUnsafe Search(SE);
  visitAll(S, Search);
  return !Search.IsUnsafe;
}

bool isSafeToExpandAt(const SCEV *S, const Instruction *InsertionPoint,
                      ScalarEvolution &SE) {
  if (!isSafeToExpand(S, SE))
    return false;
  // We have to prove that the expanded site of S dominates InsertionPoint.
  // This is easy when not in the same block, but hard when S is an instruction
  // to be expanded somewhere inside the same block as our insertion point.
  // What we really need here is something analogous to an OrderedBasicBlock,
  // but for the moment, we paper over the problem by handling two common and
  // cheap to check cases.
  if (SE.properlyDominates(S, InsertionPoint->getParent()))
    return true;
  if (SE.dominates(S, InsertionPoint->getParent())) {
    if (InsertionPoint->getParent()->getTerminator() == InsertionPoint)
      return true;
    if (const SCEVUnknown *U = dyn_cast<SCEVUnknown>(S))
      if (llvm::is_contained(InsertionPoint->operand_values(), U->getValue()))
        return true;
  }
  return false;
}

void SCEVExpanderCleaner::cleanup() {
  // Result is used, nothing to remove.
  if (ResultUsed)
    return;

  auto InsertedInstructions = Expander.getAllInsertedInstructions();
#ifndef NDEBUG
  SmallPtrSet<Instruction *, 8> InsertedSet(InsertedInstructions.begin(),
                                            InsertedInstructions.end());
  (void)InsertedSet;
#endif
  // Remove sets with value handles.
  Expander.clear();

  // Sort so that earlier instructions do not dominate later instructions.
  stable_sort(InsertedInstructions, [this](Instruction *A, Instruction *B) {
    return DT.dominates(B, A);
  });
  // Remove all inserted instructions.
  for (Instruction *I : InsertedInstructions) {

#ifndef NDEBUG
    assert(all_of(I->users(),
                  [&InsertedSet](Value *U) {
                    return InsertedSet.contains(cast<Instruction>(U));
                  }) &&
           "removed instruction should only be used by instructions inserted "
           "during expansion");
#endif
    assert(!I->getType()->isVoidTy() &&
           "inserted instruction should have non-void types");
    I->replaceAllUsesWith(UndefValue::get(I->getType()));
    I->eraseFromParent();
  }
}
}
