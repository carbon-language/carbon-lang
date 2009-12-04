//===- ScalarEvolutionExpander.cpp - Scalar Evolution Analysis --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the scalar evolution expander,
// which is used to generate the code corresponding to a given scalar evolution
// expression.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/LLVMContext.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

/// InsertNoopCastOfTo - Insert a cast of V to the specified type,
/// which must be possible with a noop cast, doing what we can to share
/// the casts.
Value *SCEVExpander::InsertNoopCastOfTo(Value *V, const Type *Ty) {
  Instruction::CastOps Op = CastInst::getCastOpcode(V, false, Ty, false);
  assert((Op == Instruction::BitCast ||
          Op == Instruction::PtrToInt ||
          Op == Instruction::IntToPtr) &&
         "InsertNoopCastOfTo cannot perform non-noop casts!");
  assert(SE.getTypeSizeInBits(V->getType()) == SE.getTypeSizeInBits(Ty) &&
         "InsertNoopCastOfTo cannot change sizes!");

  // Short-circuit unnecessary bitcasts.
  if (Op == Instruction::BitCast && V->getType() == Ty)
    return V;

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

  if (Constant *C = dyn_cast<Constant>(V))
    return ConstantExpr::getCast(Op, C, Ty);

  if (Argument *A = dyn_cast<Argument>(V)) {
    // Check to see if there is already a cast!
    for (Value::use_iterator UI = A->use_begin(), E = A->use_end();
         UI != E; ++UI)
      if ((*UI)->getType() == Ty)
        if (CastInst *CI = dyn_cast<CastInst>(cast<Instruction>(*UI)))
          if (CI->getOpcode() == Op) {
            // If the cast isn't the first instruction of the function, move it.
            if (BasicBlock::iterator(CI) !=
                A->getParent()->getEntryBlock().begin()) {
              // Recreate the cast at the beginning of the entry block.
              // The old cast is left in place in case it is being used
              // as an insert point.
              Instruction *NewCI =
                CastInst::Create(Op, V, Ty, "",
                                 A->getParent()->getEntryBlock().begin());
              NewCI->takeName(CI);
              CI->replaceAllUsesWith(NewCI);
              return NewCI;
            }
            return CI;
          }

    Instruction *I = CastInst::Create(Op, V, Ty, V->getName(),
                                      A->getParent()->getEntryBlock().begin());
    InsertedValues.insert(I);
    return I;
  }

  Instruction *I = cast<Instruction>(V);

  // Check to see if there is already a cast.  If there is, use it.
  for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
       UI != E; ++UI) {
    if ((*UI)->getType() == Ty)
      if (CastInst *CI = dyn_cast<CastInst>(cast<Instruction>(*UI)))
        if (CI->getOpcode() == Op) {
          BasicBlock::iterator It = I; ++It;
          if (isa<InvokeInst>(I))
            It = cast<InvokeInst>(I)->getNormalDest()->begin();
          while (isa<PHINode>(It)) ++It;
          if (It != BasicBlock::iterator(CI)) {
            // Recreate the cast at the beginning of the entry block.
            // The old cast is left in place in case it is being used
            // as an insert point.
            Instruction *NewCI = CastInst::Create(Op, V, Ty, "", It);
            NewCI->takeName(CI);
            CI->replaceAllUsesWith(NewCI);
            return NewCI;
          }
          return CI;
        }
  }
  BasicBlock::iterator IP = I; ++IP;
  if (InvokeInst *II = dyn_cast<InvokeInst>(I))
    IP = II->getNormalDest()->begin();
  while (isa<PHINode>(IP)) ++IP;
  Instruction *CI = CastInst::Create(Op, V, Ty, V->getName(), IP);
  InsertedValues.insert(CI);
  return CI;
}

/// InsertBinop - Insert the specified binary operator, doing a small amount
/// of work to avoid inserting an obviously redundant operation.
Value *SCEVExpander::InsertBinop(Instruction::BinaryOps Opcode,
                                 Value *LHS, Value *RHS) {
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
      if (IP->getOpcode() == (unsigned)Opcode && IP->getOperand(0) == LHS &&
          IP->getOperand(1) == RHS)
        return IP;
      if (IP == BlockBegin) break;
    }
  }

  // If we haven't found this binop, insert it.
  Value *BO = Builder.CreateBinOp(Opcode, LHS, RHS, "tmp");
  InsertedValues.insert(BO);
  return BO;
}

/// FactorOutConstant - Test if S is divisible by Factor, using signed
/// division. If so, update S with Factor divided out and return true.
/// S need not be evenly divisble if a reasonable remainder can be
/// computed.
/// TODO: When ScalarEvolution gets a SCEVSDivExpr, this can be made
/// unnecessary; in its place, just signed-divide Ops[i] by the scale and
/// check to see if the divide was folded.
static bool FactorOutConstant(const SCEV *&S,
                              const SCEV *&Remainder,
                              const SCEV *Factor,
                              ScalarEvolution &SE,
                              const TargetData *TD) {
  // Everything is divisible by one.
  if (Factor->isOne())
    return true;

  // x/x == 1.
  if (S == Factor) {
    S = SE.getIntegerSCEV(1, S->getType());
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
        ConstantInt::get(SE.getContext(),
                         C->getValue()->getValue().sdiv(
                                                   FC->getValue()->getValue()));
      // If the quotient is zero and the remainder is non-zero, reject
      // the value at this scale. It will be considered for subsequent
      // smaller scales.
      if (!CI->isZero()) {
        const SCEV *Div = SE.getConstant(CI);
        S = Div;
        Remainder =
          SE.getAddExpr(Remainder,
                        SE.getConstant(C->getValue()->getValue().srem(
                                                  FC->getValue()->getValue())));
        return true;
      }
    }
  }

  // In a Mul, check if there is a constant operand which is a multiple
  // of the given factor.
  if (const SCEVMulExpr *M = dyn_cast<SCEVMulExpr>(S)) {
    if (TD) {
      // With TargetData, the size is known. Check if there is a constant
      // operand which is a multiple of the given factor. If so, we can
      // factor it.
      const SCEVConstant *FC = cast<SCEVConstant>(Factor);
      if (const SCEVConstant *C = dyn_cast<SCEVConstant>(M->getOperand(0)))
        if (!C->getValue()->getValue().srem(FC->getValue()->getValue())) {
          const SmallVectorImpl<const SCEV *> &MOperands = M->getOperands();
          SmallVector<const SCEV *, 4> NewMulOps(MOperands.begin(),
                                                 MOperands.end());
          NewMulOps[0] =
            SE.getConstant(C->getValue()->getValue().sdiv(
                                                   FC->getValue()->getValue()));
          S = SE.getMulExpr(NewMulOps);
          return true;
        }
    } else {
      // Without TargetData, check if Factor can be factored out of any of the
      // Mul's operands. If so, we can just remove it.
      for (unsigned i = 0, e = M->getNumOperands(); i != e; ++i) {
        const SCEV *SOp = M->getOperand(i);
        const SCEV *Remainder = SE.getIntegerSCEV(0, SOp->getType());
        if (FactorOutConstant(SOp, Remainder, Factor, SE, TD) &&
            Remainder->isZero()) {
          const SmallVectorImpl<const SCEV *> &MOperands = M->getOperands();
          SmallVector<const SCEV *, 4> NewMulOps(MOperands.begin(),
                                                 MOperands.end());
          NewMulOps[i] = SOp;
          S = SE.getMulExpr(NewMulOps);
          return true;
        }
      }
    }
  }

  // In an AddRec, check if both start and step are divisible.
  if (const SCEVAddRecExpr *A = dyn_cast<SCEVAddRecExpr>(S)) {
    const SCEV *Step = A->getStepRecurrence(SE);
    const SCEV *StepRem = SE.getIntegerSCEV(0, Step->getType());
    if (!FactorOutConstant(Step, StepRem, Factor, SE, TD))
      return false;
    if (!StepRem->isZero())
      return false;
    const SCEV *Start = A->getStart();
    if (!FactorOutConstant(Start, Remainder, Factor, SE, TD))
      return false;
    S = SE.getAddRecExpr(Start, Step, A->getLoop());
    return true;
  }

  return false;
}

/// SimplifyAddOperands - Sort and simplify a list of add operands. NumAddRecs
/// is the number of SCEVAddRecExprs present, which are kept at the end of
/// the list.
///
static void SimplifyAddOperands(SmallVectorImpl<const SCEV *> &Ops,
                                const Type *Ty,
                                ScalarEvolution &SE) {
  unsigned NumAddRecs = 0;
  for (unsigned i = Ops.size(); i > 0 && isa<SCEVAddRecExpr>(Ops[i-1]); --i)
    ++NumAddRecs;
  // Group Ops into non-addrecs and addrecs.
  SmallVector<const SCEV *, 8> NoAddRecs(Ops.begin(), Ops.end() - NumAddRecs);
  SmallVector<const SCEV *, 8> AddRecs(Ops.end() - NumAddRecs, Ops.end());
  // Let ScalarEvolution sort and simplify the non-addrecs list.
  const SCEV *Sum = NoAddRecs.empty() ?
                    SE.getIntegerSCEV(0, Ty) :
                    SE.getAddExpr(NoAddRecs);
  // If it returned an add, use the operands. Otherwise it simplified
  // the sum into a single value, so just use that.
  if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(Sum))
    Ops = Add->getOperands();
  else {
    Ops.clear();
    if (!Sum->isZero())
      Ops.push_back(Sum);
  }
  // Then append the addrecs.
  Ops.insert(Ops.end(), AddRecs.begin(), AddRecs.end());
}

/// SplitAddRecs - Flatten a list of add operands, moving addrec start values
/// out to the top level. For example, convert {a + b,+,c} to a, b, {0,+,d}.
/// This helps expose more opportunities for folding parts of the expressions
/// into GEP indices.
///
static void SplitAddRecs(SmallVectorImpl<const SCEV *> &Ops,
                         const Type *Ty,
                         ScalarEvolution &SE) {
  // Find the addrecs.
  SmallVector<const SCEV *, 8> AddRecs;
  for (unsigned i = 0, e = Ops.size(); i != e; ++i)
    while (const SCEVAddRecExpr *A = dyn_cast<SCEVAddRecExpr>(Ops[i])) {
      const SCEV *Start = A->getStart();
      if (Start->isZero()) break;
      const SCEV *Zero = SE.getIntegerSCEV(0, Ty);
      AddRecs.push_back(SE.getAddRecExpr(Zero,
                                         A->getStepRecurrence(SE),
                                         A->getLoop()));
      if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(Start)) {
        Ops[i] = Zero;
        Ops.insert(Ops.end(), Add->op_begin(), Add->op_end());
        e += Add->getNumOperands();
      } else {
        Ops[i] = Start;
      }
    }
  if (!AddRecs.empty()) {
    // Add the addrecs onto the end of the list.
    Ops.insert(Ops.end(), AddRecs.begin(), AddRecs.end());
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
/// Design note: The correctness of using getelmeentptr here depends on
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
                                    const PointerType *PTy,
                                    const Type *Ty,
                                    Value *V) {
  const Type *ElTy = PTy->getElementType();
  SmallVector<Value *, 4> GepIndices;
  SmallVector<const SCEV *, 8> Ops(op_begin, op_end);
  bool AnyNonZeroIndices = false;

  // Split AddRecs up into parts as either of the parts may be usable
  // without the other.
  SplitAddRecs(Ops, Ty, SE);

  // Descend down the pointer's type and attempt to convert the other
  // operands into GEP indices, at each level. The first index in a GEP
  // indexes into the array implied by the pointer operand; the rest of
  // the indices index into the element or field type selected by the
  // preceding index.
  for (;;) {
    const SCEV *ElSize = SE.getAllocSizeExpr(ElTy);
    // If the scale size is not 0, attempt to factor out a scale for
    // array indexing.
    SmallVector<const SCEV *, 8> ScaledOps;
    if (ElTy->isSized() && !ElSize->isZero()) {
      SmallVector<const SCEV *, 8> NewOps;
      for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
        const SCEV *Op = Ops[i];
        const SCEV *Remainder = SE.getIntegerSCEV(0, Ty);
        if (FactorOutConstant(Op, Remainder, ElSize, SE, SE.TD)) {
          // Op now has ElSize factored out.
          ScaledOps.push_back(Op);
          if (!Remainder->isZero())
            NewOps.push_back(Remainder);
          AnyNonZeroIndices = true;
        } else {
          // The operand was not divisible, so add it to the list of operands
          // we'll scan next iteration.
          NewOps.push_back(Ops[i]);
        }
      }
      // If we made any changes, update Ops.
      if (!ScaledOps.empty()) {
        Ops = NewOps;
        SimplifyAddOperands(Ops, Ty, SE);
      }
    }

    // Record the scaled array index for this level of the type. If
    // we didn't find any operands that could be factored, tentatively
    // assume that element zero was selected (since the zero offset
    // would obviously be folded away).
    Value *Scaled = ScaledOps.empty() ?
                    Constant::getNullValue(Ty) :
                    expandCodeFor(SE.getAddExpr(ScaledOps), Ty);
    GepIndices.push_back(Scaled);

    // Collect struct field index operands.
    while (const StructType *STy = dyn_cast<StructType>(ElTy)) {
      bool FoundFieldNo = false;
      // An empty struct has no fields.
      if (STy->getNumElements() == 0) break;
      if (SE.TD) {
        // With TargetData, field offsets are known. See if a constant offset
        // falls within any of the struct fields.
        if (Ops.empty()) break;
        if (const SCEVConstant *C = dyn_cast<SCEVConstant>(Ops[0]))
          if (SE.getTypeSizeInBits(C->getType()) <= 64) {
            const StructLayout &SL = *SE.TD->getStructLayout(STy);
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
      } else {
        // Without TargetData, just check for a SCEVFieldOffsetExpr of the
        // appropriate struct type.
        for (unsigned i = 0, e = Ops.size(); i != e; ++i)
          if (const SCEVFieldOffsetExpr *FO =
                dyn_cast<SCEVFieldOffsetExpr>(Ops[i]))
            if (FO->getStructType() == STy) {
              unsigned FieldNo = FO->getFieldNo();
              GepIndices.push_back(
                  ConstantInt::get(Type::getInt32Ty(Ty->getContext()),
                                   FieldNo));
              ElTy = STy->getTypeAtIndex(FieldNo);
              Ops[i] = SE.getConstant(Ty, 0);
              AnyNonZeroIndices = true;
              FoundFieldNo = true;
              break;
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

    if (const ArrayType *ATy = dyn_cast<ArrayType>(ElTy))
      ElTy = ATy->getElementType();
    else
      break;
  }

  // If none of the operands were convertable to proper GEP indices, cast
  // the base to i8* and do an ugly getelementptr with that. It's still
  // better than ptrtoint+arithmetic+inttoptr at least.
  if (!AnyNonZeroIndices) {
    // Cast the base to i8*.
    V = InsertNoopCastOfTo(V,
       Type::getInt8PtrTy(Ty->getContext(), PTy->getAddressSpace()));

    // Expand the operands for a plain byte offset.
    Value *Idx = expandCodeFor(SE.getAddExpr(Ops), Ty);

    // Fold a GEP with constant operands.
    if (Constant *CLHS = dyn_cast<Constant>(V))
      if (Constant *CRHS = dyn_cast<Constant>(Idx))
        return ConstantExpr::getGetElementPtr(CLHS, &CRHS, 1);

    // Do a quick scan to see if we have this GEP nearby.  If so, reuse it.
    unsigned ScanLimit = 6;
    BasicBlock::iterator BlockBegin = Builder.GetInsertBlock()->begin();
    // Scanning starts from the last instruction before the insertion point.
    BasicBlock::iterator IP = Builder.GetInsertPoint();
    if (IP != BlockBegin) {
      --IP;
      for (; ScanLimit; --IP, --ScanLimit) {
        if (IP->getOpcode() == Instruction::GetElementPtr &&
            IP->getOperand(0) == V && IP->getOperand(1) == Idx)
          return IP;
        if (IP == BlockBegin) break;
      }
    }

    // Emit a GEP.
    Value *GEP = Builder.CreateGEP(V, Idx, "uglygep");
    InsertedValues.insert(GEP);
    return GEP;
  }

  // Insert a pretty getelementptr. Note that this GEP is not marked inbounds,
  // because ScalarEvolution may have changed the address arithmetic to
  // compute a value which is beyond the end of the allocated object.
  Value *GEP = Builder.CreateGEP(V,
                                 GepIndices.begin(),
                                 GepIndices.end(),
                                 "scevgep");
  Ops.push_back(SE.getUnknown(GEP));
  InsertedValues.insert(GEP);
  return expand(SE.getAddExpr(Ops));
}

Value *SCEVExpander::visitAddExpr(const SCEVAddExpr *S) {
  int NumOperands = S->getNumOperands();
  const Type *Ty = SE.getEffectiveSCEVType(S->getType());

  // Find the index of an operand to start with. Choose the operand with
  // pointer type, if there is one, or the last operand otherwise.
  int PIdx = 0;
  for (; PIdx != NumOperands - 1; ++PIdx)
    if (isa<PointerType>(S->getOperand(PIdx)->getType())) break;

  // Expand code for the operand that we chose.
  Value *V = expand(S->getOperand(PIdx));

  // Turn things like ptrtoint+arithmetic+inttoptr into GEP. See the
  // comments on expandAddToGEP for details.
  if (const PointerType *PTy = dyn_cast<PointerType>(V->getType())) {
    // Take the operand at PIdx out of the list.
    const SmallVectorImpl<const SCEV *> &Ops = S->getOperands();
    SmallVector<const SCEV *, 8> NewOps;
    NewOps.insert(NewOps.end(), Ops.begin(), Ops.begin() + PIdx);
    NewOps.insert(NewOps.end(), Ops.begin() + PIdx + 1, Ops.end());
    // Make a GEP.
    return expandAddToGEP(NewOps.begin(), NewOps.end(), PTy, Ty, V);
  }

  // Otherwise, we'll expand the rest of the SCEVAddExpr as plain integer
  // arithmetic.
  V = InsertNoopCastOfTo(V, Ty);

  // Emit a bunch of add instructions
  for (int i = NumOperands-1; i >= 0; --i) {
    if (i == PIdx) continue;
    Value *W = expandCodeFor(S->getOperand(i), Ty);
    V = InsertBinop(Instruction::Add, V, W);
  }
  return V;
}

Value *SCEVExpander::visitMulExpr(const SCEVMulExpr *S) {
  const Type *Ty = SE.getEffectiveSCEVType(S->getType());
  int FirstOp = 0;  // Set if we should emit a subtract.
  if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(S->getOperand(0)))
    if (SC->getValue()->isAllOnesValue())
      FirstOp = 1;

  int i = S->getNumOperands()-2;
  Value *V = expandCodeFor(S->getOperand(i+1), Ty);

  // Emit a bunch of multiply instructions
  for (; i >= FirstOp; --i) {
    Value *W = expandCodeFor(S->getOperand(i), Ty);
    V = InsertBinop(Instruction::Mul, V, W);
  }

  // -1 * ...  --->  0 - ...
  if (FirstOp == 1)
    V = InsertBinop(Instruction::Sub, Constant::getNullValue(Ty), V);
  return V;
}

Value *SCEVExpander::visitUDivExpr(const SCEVUDivExpr *S) {
  const Type *Ty = SE.getEffectiveSCEVType(S->getType());

  Value *LHS = expandCodeFor(S->getLHS(), Ty);
  if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(S->getRHS())) {
    const APInt &RHS = SC->getValue()->getValue();
    if (RHS.isPowerOf2())
      return InsertBinop(Instruction::LShr, LHS,
                         ConstantInt::get(Ty, RHS.logBase2()));
  }

  Value *RHS = expandCodeFor(S->getRHS(), Ty);
  return InsertBinop(Instruction::UDiv, LHS, RHS);
}

/// Move parts of Base into Rest to leave Base with the minimal
/// expression that provides a pointer operand suitable for a
/// GEP expansion.
static void ExposePointerBase(const SCEV *&Base, const SCEV *&Rest,
                              ScalarEvolution &SE) {
  while (const SCEVAddRecExpr *A = dyn_cast<SCEVAddRecExpr>(Base)) {
    Base = A->getStart();
    Rest = SE.getAddExpr(Rest,
                         SE.getAddRecExpr(SE.getIntegerSCEV(0, A->getType()),
                                          A->getStepRecurrence(SE),
                                          A->getLoop()));
  }
  if (const SCEVAddExpr *A = dyn_cast<SCEVAddExpr>(Base)) {
    Base = A->getOperand(A->getNumOperands()-1);
    SmallVector<const SCEV *, 8> NewAddOps(A->op_begin(), A->op_end());
    NewAddOps.back() = Rest;
    Rest = SE.getAddExpr(NewAddOps);
    ExposePointerBase(Base, Rest, SE);
  }
}

Value *SCEVExpander::visitAddRecExpr(const SCEVAddRecExpr *S) {
  const Type *Ty = SE.getEffectiveSCEVType(S->getType());
  const Loop *L = S->getLoop();

  // First check for an existing canonical IV in a suitable type.
  PHINode *CanonicalIV = 0;
  if (PHINode *PN = L->getCanonicalInductionVariable())
    if (SE.isSCEVable(PN->getType()) &&
        isa<IntegerType>(SE.getEffectiveSCEVType(PN->getType())) &&
        SE.getTypeSizeInBits(PN->getType()) >= SE.getTypeSizeInBits(Ty))
      CanonicalIV = PN;

  // Rewrite an AddRec in terms of the canonical induction variable, if
  // its type is more narrow.
  if (CanonicalIV &&
      SE.getTypeSizeInBits(CanonicalIV->getType()) >
      SE.getTypeSizeInBits(Ty)) {
    const SmallVectorImpl<const SCEV *> &Ops = S->getOperands();
    SmallVector<const SCEV *, 4> NewOps(Ops.size());
    for (unsigned i = 0, e = Ops.size(); i != e; ++i)
      NewOps[i] = SE.getAnyExtendExpr(Ops[i], CanonicalIV->getType());
    Value *V = expand(SE.getAddRecExpr(NewOps, S->getLoop()));
    BasicBlock *SaveInsertBB = Builder.GetInsertBlock();
    BasicBlock::iterator SaveInsertPt = Builder.GetInsertPoint();
    BasicBlock::iterator NewInsertPt =
      llvm::next(BasicBlock::iterator(cast<Instruction>(V)));
    while (isa<PHINode>(NewInsertPt)) ++NewInsertPt;
    V = expandCodeFor(SE.getTruncateExpr(SE.getUnknown(V), Ty), 0,
                      NewInsertPt);
    Builder.SetInsertPoint(SaveInsertBB, SaveInsertPt);
    return V;
  }

  // {X,+,F} --> X + {0,+,F}
  if (!S->getStart()->isZero()) {
    const SmallVectorImpl<const SCEV *> &SOperands = S->getOperands();
    SmallVector<const SCEV *, 4> NewOps(SOperands.begin(), SOperands.end());
    NewOps[0] = SE.getIntegerSCEV(0, Ty);
    const SCEV *Rest = SE.getAddRecExpr(NewOps, L);

    // Turn things like ptrtoint+arithmetic+inttoptr into GEP. See the
    // comments on expandAddToGEP for details.
    const SCEV *Base = S->getStart();
    const SCEV *RestArray[1] = { Rest };
    // Dig into the expression to find the pointer base for a GEP.
    ExposePointerBase(Base, RestArray[0], SE);
    // If we found a pointer, expand the AddRec with a GEP.
    if (const PointerType *PTy = dyn_cast<PointerType>(Base->getType())) {
      // Make sure the Base isn't something exotic, such as a multiplied
      // or divided pointer value. In those cases, the result type isn't
      // actually a pointer type.
      if (!isa<SCEVMulExpr>(Base) && !isa<SCEVUDivExpr>(Base)) {
        Value *StartV = expand(Base);
        assert(StartV->getType() == PTy && "Pointer type mismatch for GEP!");
        return expandAddToGEP(RestArray, RestArray+1, PTy, Ty, StartV);
      }
    }

    // Just do a normal add. Pre-expand the operands to suppress folding.
    return expand(SE.getAddExpr(SE.getUnknown(expand(S->getStart())),
                                SE.getUnknown(expand(Rest))));
  }

  // {0,+,1} --> Insert a canonical induction variable into the loop!
  if (S->isAffine() &&
      S->getOperand(1) == SE.getIntegerSCEV(1, Ty)) {
    // If there's a canonical IV, just use it.
    if (CanonicalIV) {
      assert(Ty == SE.getEffectiveSCEVType(CanonicalIV->getType()) &&
             "IVs with types different from the canonical IV should "
             "already have been handled!");
      return CanonicalIV;
    }

    // Create and insert the PHI node for the induction variable in the
    // specified loop.
    BasicBlock *Header = L->getHeader();
    PHINode *PN = PHINode::Create(Ty, "indvar", Header->begin());
    InsertedValues.insert(PN);

    Constant *One = ConstantInt::get(Ty, 1);
    for (pred_iterator HPI = pred_begin(Header), HPE = pred_end(Header);
         HPI != HPE; ++HPI)
      if (L->contains(*HPI)) {
        // Insert a unit add instruction right before the terminator corresponding
        // to the back-edge.
        Instruction *Add = BinaryOperator::CreateAdd(PN, One, "indvar.next",
                                                     (*HPI)->getTerminator());
        InsertedValues.insert(Add);
        PN->addIncoming(Add, *HPI);
      } else {
        PN->addIncoming(Constant::getNullValue(Ty), *HPI);
      }
  }

  // {0,+,F} --> {0,+,1} * F
  // Get the canonical induction variable I for this loop.
  Value *I = CanonicalIV ?
             CanonicalIV :
             getOrInsertCanonicalInductionVariable(L, Ty);

  // If this is a simple linear addrec, emit it now as a special case.
  if (S->isAffine())    // {0,+,F} --> i*F
    return
      expand(SE.getTruncateOrNoop(
        SE.getMulExpr(SE.getUnknown(I),
                      SE.getNoopOrAnyExtend(S->getOperand(1),
                                            I->getType())),
        Ty));

  // If this is a chain of recurrences, turn it into a closed form, using the
  // folders, then expandCodeFor the closed form.  This allows the folders to
  // simplify the expression without having to build a bunch of special code
  // into this folder.
  const SCEV *IH = SE.getUnknown(I);   // Get I as a "symbolic" SCEV.

  // Promote S up to the canonical IV type, if the cast is foldable.
  const SCEV *NewS = S;
  const SCEV *Ext = SE.getNoopOrAnyExtend(S, I->getType());
  if (isa<SCEVAddRecExpr>(Ext))
    NewS = Ext;

  const SCEV *V = cast<SCEVAddRecExpr>(NewS)->evaluateAtIteration(IH, SE);
  //cerr << "Evaluated: " << *this << "\n     to: " << *V << "\n";

  // Truncate the result down to the original type, if needed.
  const SCEV *T = SE.getTruncateOrNoop(V, Ty);
  return expand(T);
}

Value *SCEVExpander::visitTruncateExpr(const SCEVTruncateExpr *S) {
  const Type *Ty = SE.getEffectiveSCEVType(S->getType());
  Value *V = expandCodeFor(S->getOperand(),
                           SE.getEffectiveSCEVType(S->getOperand()->getType()));
  Value *I = Builder.CreateTrunc(V, Ty, "tmp");
  InsertedValues.insert(I);
  return I;
}

Value *SCEVExpander::visitZeroExtendExpr(const SCEVZeroExtendExpr *S) {
  const Type *Ty = SE.getEffectiveSCEVType(S->getType());
  Value *V = expandCodeFor(S->getOperand(),
                           SE.getEffectiveSCEVType(S->getOperand()->getType()));
  Value *I = Builder.CreateZExt(V, Ty, "tmp");
  InsertedValues.insert(I);
  return I;
}

Value *SCEVExpander::visitSignExtendExpr(const SCEVSignExtendExpr *S) {
  const Type *Ty = SE.getEffectiveSCEVType(S->getType());
  Value *V = expandCodeFor(S->getOperand(),
                           SE.getEffectiveSCEVType(S->getOperand()->getType()));
  Value *I = Builder.CreateSExt(V, Ty, "tmp");
  InsertedValues.insert(I);
  return I;
}

Value *SCEVExpander::visitSMaxExpr(const SCEVSMaxExpr *S) {
  Value *LHS = expand(S->getOperand(S->getNumOperands()-1));
  const Type *Ty = LHS->getType();
  for (int i = S->getNumOperands()-2; i >= 0; --i) {
    // In the case of mixed integer and pointer types, do the
    // rest of the comparisons as integer.
    if (S->getOperand(i)->getType() != Ty) {
      Ty = SE.getEffectiveSCEVType(Ty);
      LHS = InsertNoopCastOfTo(LHS, Ty);
    }
    Value *RHS = expandCodeFor(S->getOperand(i), Ty);
    Value *ICmp = Builder.CreateICmpSGT(LHS, RHS, "tmp");
    InsertedValues.insert(ICmp);
    Value *Sel = Builder.CreateSelect(ICmp, LHS, RHS, "smax");
    InsertedValues.insert(Sel);
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
  const Type *Ty = LHS->getType();
  for (int i = S->getNumOperands()-2; i >= 0; --i) {
    // In the case of mixed integer and pointer types, do the
    // rest of the comparisons as integer.
    if (S->getOperand(i)->getType() != Ty) {
      Ty = SE.getEffectiveSCEVType(Ty);
      LHS = InsertNoopCastOfTo(LHS, Ty);
    }
    Value *RHS = expandCodeFor(S->getOperand(i), Ty);
    Value *ICmp = Builder.CreateICmpUGT(LHS, RHS, "tmp");
    InsertedValues.insert(ICmp);
    Value *Sel = Builder.CreateSelect(ICmp, LHS, RHS, "umax");
    InsertedValues.insert(Sel);
    LHS = Sel;
  }
  // In the case of mixed integer and pointer types, cast the
  // final result back to the pointer type.
  if (LHS->getType() != S->getType())
    LHS = InsertNoopCastOfTo(LHS, S->getType());
  return LHS;
}

Value *SCEVExpander::visitFieldOffsetExpr(const SCEVFieldOffsetExpr *S) {
  return ConstantExpr::getOffsetOf(S->getStructType(), S->getFieldNo());
}

Value *SCEVExpander::visitAllocSizeExpr(const SCEVAllocSizeExpr *S) {
  return ConstantExpr::getSizeOf(S->getAllocType());
}

Value *SCEVExpander::expandCodeFor(const SCEV *SH, const Type *Ty) {
  // Expand the code for this SCEV.
  Value *V = expand(SH);
  if (Ty) {
    assert(SE.getTypeSizeInBits(Ty) == SE.getTypeSizeInBits(SH->getType()) &&
           "non-trivial casts should be done with the SCEVs directly!");
    V = InsertNoopCastOfTo(V, Ty);
  }
  return V;
}

Value *SCEVExpander::expand(const SCEV *S) {
  // Compute an insertion point for this SCEV object. Hoist the instructions
  // as far out in the loop nest as possible.
  Instruction *InsertPt = Builder.GetInsertPoint();
  for (Loop *L = SE.LI->getLoopFor(Builder.GetInsertBlock()); ;
       L = L->getParentLoop())
    if (S->isLoopInvariant(L)) {
      if (!L) break;
      if (BasicBlock *Preheader = L->getLoopPreheader())
        InsertPt = Preheader->getTerminator();
    } else {
      // If the SCEV is computable at this level, insert it into the header
      // after the PHIs (and after any other instructions that we've inserted
      // there) so that it is guaranteed to dominate any user inside the loop.
      if (L && S->hasComputableLoopEvolution(L))
        InsertPt = L->getHeader()->getFirstNonPHI();
      while (isInsertedInstruction(InsertPt))
        InsertPt = llvm::next(BasicBlock::iterator(InsertPt));
      break;
    }

  // Check to see if we already expanded this here.
  std::map<std::pair<const SCEV *, Instruction *>,
           AssertingVH<Value> >::iterator I =
    InsertedExpressions.find(std::make_pair(S, InsertPt));
  if (I != InsertedExpressions.end())
    return I->second;

  BasicBlock *SaveInsertBB = Builder.GetInsertBlock();
  BasicBlock::iterator SaveInsertPt = Builder.GetInsertPoint();
  Builder.SetInsertPoint(InsertPt->getParent(), InsertPt);

  // Expand the expression into instructions.
  Value *V = visit(S);

  // Remember the expanded value for this SCEV at this location.
  InsertedExpressions[std::make_pair(S, InsertPt)] = V;

  Builder.SetInsertPoint(SaveInsertBB, SaveInsertPt);
  return V;
}

/// getOrInsertCanonicalInductionVariable - This method returns the
/// canonical induction variable of the specified type for the specified
/// loop (inserting one if there is none).  A canonical induction variable
/// starts at zero and steps by one on each iteration.
Value *
SCEVExpander::getOrInsertCanonicalInductionVariable(const Loop *L,
                                                    const Type *Ty) {
  assert(Ty->isInteger() && "Can only insert integer induction variables!");
  const SCEV *H = SE.getAddRecExpr(SE.getIntegerSCEV(0, Ty),
                                   SE.getIntegerSCEV(1, Ty), L);
  BasicBlock *SaveInsertBB = Builder.GetInsertBlock();
  BasicBlock::iterator SaveInsertPt = Builder.GetInsertPoint();
  Value *V = expandCodeFor(H, 0, L->getHeader()->begin());
  if (SaveInsertBB)
    Builder.SetInsertPoint(SaveInsertBB, SaveInsertPt);
  return V;
}
