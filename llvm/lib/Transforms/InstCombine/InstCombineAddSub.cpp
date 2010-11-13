//===- InstCombineAddSub.cpp ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the visit functions for add, fadd, sub, and fsub.
//
//===----------------------------------------------------------------------===//

#include "InstCombine.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/PatternMatch.h"
using namespace llvm;
using namespace PatternMatch;

/// AddOne - Add one to a ConstantInt.
static Constant *AddOne(Constant *C) {
  return ConstantExpr::getAdd(C, ConstantInt::get(C->getType(), 1));
}
/// SubOne - Subtract one from a ConstantInt.
static Constant *SubOne(ConstantInt *C) {
  return ConstantInt::get(C->getContext(), C->getValue()-1);
}


// dyn_castFoldableMul - If this value is a multiply that can be folded into
// other computations (because it has a constant operand), return the
// non-constant operand of the multiply, and set CST to point to the multiplier.
// Otherwise, return null.
//
static inline Value *dyn_castFoldableMul(Value *V, ConstantInt *&CST) {
  if (!V->hasOneUse() || !V->getType()->isIntegerTy())
    return 0;
  
  Instruction *I = dyn_cast<Instruction>(V);
  if (I == 0) return 0;
  
  if (I->getOpcode() == Instruction::Mul)
    if ((CST = dyn_cast<ConstantInt>(I->getOperand(1))))
      return I->getOperand(0);
  if (I->getOpcode() == Instruction::Shl)
    if ((CST = dyn_cast<ConstantInt>(I->getOperand(1)))) {
      // The multiplier is really 1 << CST.
      uint32_t BitWidth = cast<IntegerType>(V->getType())->getBitWidth();
      uint32_t CSTVal = CST->getLimitedValue(BitWidth);
      CST = ConstantInt::get(V->getType()->getContext(),
                             APInt(BitWidth, 1).shl(CSTVal));
      return I->getOperand(0);
    }
  return 0;
}


/// WillNotOverflowSignedAdd - Return true if we can prove that:
///    (sext (add LHS, RHS))  === (add (sext LHS), (sext RHS))
/// This basically requires proving that the add in the original type would not
/// overflow to change the sign bit or have a carry out.
bool InstCombiner::WillNotOverflowSignedAdd(Value *LHS, Value *RHS) {
  // There are different heuristics we can use for this.  Here are some simple
  // ones.
  
  // Add has the property that adding any two 2's complement numbers can only 
  // have one carry bit which can change a sign.  As such, if LHS and RHS each
  // have at least two sign bits, we know that the addition of the two values
  // will sign extend fine.
  if (ComputeNumSignBits(LHS) > 1 && ComputeNumSignBits(RHS) > 1)
    return true;
  
  
  // If one of the operands only has one non-zero bit, and if the other operand
  // has a known-zero bit in a more significant place than it (not including the
  // sign bit) the ripple may go up to and fill the zero, but won't change the
  // sign.  For example, (X & ~4) + 1.
  
  // TODO: Implement.
  
  return false;
}

Instruction *InstCombiner::visitAdd(BinaryOperator &I) {
  bool Changed = SimplifyAssociativeOrCommutative(I);
  Value *LHS = I.getOperand(0), *RHS = I.getOperand(1);

  if (Value *V = SimplifyAddInst(LHS, RHS, I.hasNoSignedWrap(),
                                 I.hasNoUnsignedWrap(), TD))
    return ReplaceInstUsesWith(I, V);

  
  if (Constant *RHSC = dyn_cast<Constant>(RHS)) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(RHSC)) {
      // X + (signbit) --> X ^ signbit
      const APInt& Val = CI->getValue();
      uint32_t BitWidth = Val.getBitWidth();
      if (Val == APInt::getSignBit(BitWidth))
        return BinaryOperator::CreateXor(LHS, RHS);
      
      // See if SimplifyDemandedBits can simplify this.  This handles stuff like
      // (X & 254)+1 -> (X&254)|1
      if (SimplifyDemandedInstructionBits(I))
        return &I;

      // zext(bool) + C -> bool ? C + 1 : C
      if (ZExtInst *ZI = dyn_cast<ZExtInst>(LHS))
        if (ZI->getSrcTy() == Type::getInt1Ty(I.getContext()))
          return SelectInst::Create(ZI->getOperand(0), AddOne(CI), CI);
    }

    if (isa<PHINode>(LHS))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
    
    ConstantInt *XorRHS = 0;
    Value *XorLHS = 0;
    if (isa<ConstantInt>(RHSC) &&
        match(LHS, m_Xor(m_Value(XorLHS), m_ConstantInt(XorRHS)))) {
      uint32_t TySizeBits = I.getType()->getScalarSizeInBits();
      const APInt& RHSVal = cast<ConstantInt>(RHSC)->getValue();
      unsigned ExtendAmt = 0;
      // If we have ADD(XOR(AND(X, 0xFF), 0x80), 0xF..F80), it's a sext.
      // If we have ADD(XOR(AND(X, 0xFF), 0xF..F80), 0x80), it's a sext.
      if (XorRHS->getValue() == -RHSVal) {
        if (RHSVal.isPowerOf2())
          ExtendAmt = TySizeBits - RHSVal.logBase2() - 1;
        else if (XorRHS->getValue().isPowerOf2())
          ExtendAmt = TySizeBits - XorRHS->getValue().logBase2() - 1;
      }

      if (ExtendAmt) {
        APInt Mask = APInt::getHighBitsSet(TySizeBits, ExtendAmt);
        if (!MaskedValueIsZero(XorLHS, Mask))
          ExtendAmt = 0;
      }

      if (ExtendAmt) {
        Constant *ShAmt = ConstantInt::get(I.getType(), ExtendAmt);
        Value *NewShl = Builder->CreateShl(XorLHS, ShAmt, "sext");
        return BinaryOperator::CreateAShr(NewShl, ShAmt);
      }
    }
  }

  if (I.getType()->isIntegerTy(1))
    return BinaryOperator::CreateXor(LHS, RHS);

  if (I.getType()->isIntegerTy()) {
    // X + X --> X << 1
    if (LHS == RHS)
      return BinaryOperator::CreateShl(LHS, ConstantInt::get(I.getType(), 1));

    if (Instruction *RHSI = dyn_cast<Instruction>(RHS)) {
      if (RHSI->getOpcode() == Instruction::Sub)
        if (LHS == RHSI->getOperand(1))                   // A + (B - A) --> B
          return ReplaceInstUsesWith(I, RHSI->getOperand(0));
    }
    if (Instruction *LHSI = dyn_cast<Instruction>(LHS)) {
      if (LHSI->getOpcode() == Instruction::Sub)
        if (RHS == LHSI->getOperand(1))                   // (B - A) + A --> B
          return ReplaceInstUsesWith(I, LHSI->getOperand(0));
    }
  }

  // -A + B  -->  B - A
  // -A + -B  -->  -(A + B)
  if (Value *LHSV = dyn_castNegVal(LHS)) {
    if (LHS->getType()->isIntOrIntVectorTy()) {
      if (Value *RHSV = dyn_castNegVal(RHS)) {
        Value *NewAdd = Builder->CreateAdd(LHSV, RHSV, "sum");
        return BinaryOperator::CreateNeg(NewAdd);
      }
    }
    
    return BinaryOperator::CreateSub(RHS, LHSV);
  }

  // A + -B  -->  A - B
  if (!isa<Constant>(RHS))
    if (Value *V = dyn_castNegVal(RHS))
      return BinaryOperator::CreateSub(LHS, V);


  ConstantInt *C2;
  if (Value *X = dyn_castFoldableMul(LHS, C2)) {
    if (X == RHS)   // X*C + X --> X * (C+1)
      return BinaryOperator::CreateMul(RHS, AddOne(C2));

    // X*C1 + X*C2 --> X * (C1+C2)
    ConstantInt *C1;
    if (X == dyn_castFoldableMul(RHS, C1))
      return BinaryOperator::CreateMul(X, ConstantExpr::getAdd(C1, C2));
  }

  // X + X*C --> X * (C+1)
  if (dyn_castFoldableMul(RHS, C2) == LHS)
    return BinaryOperator::CreateMul(LHS, AddOne(C2));

  // X + ~X --> -1   since   ~X = -X-1
  if (match(LHS, m_Not(m_Specific(RHS))) ||
      match(RHS, m_Not(m_Specific(LHS))))
    return ReplaceInstUsesWith(I, Constant::getAllOnesValue(I.getType()));

  // A+B --> A|B iff A and B have no bits set in common.
  if (const IntegerType *IT = dyn_cast<IntegerType>(I.getType())) {
    APInt Mask = APInt::getAllOnesValue(IT->getBitWidth());
    APInt LHSKnownOne(IT->getBitWidth(), 0);
    APInt LHSKnownZero(IT->getBitWidth(), 0);
    ComputeMaskedBits(LHS, Mask, LHSKnownZero, LHSKnownOne);
    if (LHSKnownZero != 0) {
      APInt RHSKnownOne(IT->getBitWidth(), 0);
      APInt RHSKnownZero(IT->getBitWidth(), 0);
      ComputeMaskedBits(RHS, Mask, RHSKnownZero, RHSKnownOne);
      
      // No bits in common -> bitwise or.
      if ((LHSKnownZero|RHSKnownZero).isAllOnesValue())
        return BinaryOperator::CreateOr(LHS, RHS);
    }
  }

  // W*X + Y*Z --> W * (X+Z)  iff W == Y
  if (I.getType()->isIntOrIntVectorTy()) {
    Value *W, *X, *Y, *Z;
    if (match(LHS, m_Mul(m_Value(W), m_Value(X))) &&
        match(RHS, m_Mul(m_Value(Y), m_Value(Z)))) {
      if (W != Y) {
        if (W == Z) {
          std::swap(Y, Z);
        } else if (Y == X) {
          std::swap(W, X);
        } else if (X == Z) {
          std::swap(Y, Z);
          std::swap(W, X);
        }
      }

      if (W == Y) {
        Value *NewAdd = Builder->CreateAdd(X, Z, LHS->getName());
        return BinaryOperator::CreateMul(W, NewAdd);
      }
    }
  }

  if (ConstantInt *CRHS = dyn_cast<ConstantInt>(RHS)) {
    Value *X = 0;
    if (match(LHS, m_Not(m_Value(X))))    // ~X + C --> (C-1) - X
      return BinaryOperator::CreateSub(SubOne(CRHS), X);

    // (X & FF00) + xx00  -> (X+xx00) & FF00
    if (LHS->hasOneUse() &&
        match(LHS, m_And(m_Value(X), m_ConstantInt(C2)))) {
      Constant *Anded = ConstantExpr::getAnd(CRHS, C2);
      if (Anded == CRHS) {
        // See if all bits from the first bit set in the Add RHS up are included
        // in the mask.  First, get the rightmost bit.
        const APInt &AddRHSV = CRHS->getValue();

        // Form a mask of all bits from the lowest bit added through the top.
        APInt AddRHSHighBits(~((AddRHSV & -AddRHSV)-1));

        // See if the and mask includes all of these bits.
        APInt AddRHSHighBitsAnd(AddRHSHighBits & C2->getValue());

        if (AddRHSHighBits == AddRHSHighBitsAnd) {
          // Okay, the xform is safe.  Insert the new add pronto.
          Value *NewAdd = Builder->CreateAdd(X, CRHS, LHS->getName());
          return BinaryOperator::CreateAnd(NewAdd, C2);
        }
      }
    }

    // Try to fold constant add into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(LHS))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;
  }

  // add (select X 0 (sub n A)) A  -->  select X A n
  {
    SelectInst *SI = dyn_cast<SelectInst>(LHS);
    Value *A = RHS;
    if (!SI) {
      SI = dyn_cast<SelectInst>(RHS);
      A = LHS;
    }
    if (SI && SI->hasOneUse()) {
      Value *TV = SI->getTrueValue();
      Value *FV = SI->getFalseValue();
      Value *N;

      // Can we fold the add into the argument of the select?
      // We check both true and false select arguments for a matching subtract.
      if (match(FV, m_Zero()) &&
          match(TV, m_Sub(m_Value(N), m_Specific(A))))
        // Fold the add into the true select value.
        return SelectInst::Create(SI->getCondition(), N, A);
      if (match(TV, m_Zero()) &&
          match(FV, m_Sub(m_Value(N), m_Specific(A))))
        // Fold the add into the false select value.
        return SelectInst::Create(SI->getCondition(), A, N);
    }
  }

  // Check for (add (sext x), y), see if we can merge this into an
  // integer add followed by a sext.
  if (SExtInst *LHSConv = dyn_cast<SExtInst>(LHS)) {
    // (add (sext x), cst) --> (sext (add x, cst'))
    if (ConstantInt *RHSC = dyn_cast<ConstantInt>(RHS)) {
      Constant *CI = 
        ConstantExpr::getTrunc(RHSC, LHSConv->getOperand(0)->getType());
      if (LHSConv->hasOneUse() &&
          ConstantExpr::getSExt(CI, I.getType()) == RHSC &&
          WillNotOverflowSignedAdd(LHSConv->getOperand(0), CI)) {
        // Insert the new, smaller add.
        Value *NewAdd = Builder->CreateNSWAdd(LHSConv->getOperand(0), 
                                              CI, "addconv");
        return new SExtInst(NewAdd, I.getType());
      }
    }
    
    // (add (sext x), (sext y)) --> (sext (add int x, y))
    if (SExtInst *RHSConv = dyn_cast<SExtInst>(RHS)) {
      // Only do this if x/y have the same type, if at last one of them has a
      // single use (so we don't increase the number of sexts), and if the
      // integer add will not overflow.
      if (LHSConv->getOperand(0)->getType()==RHSConv->getOperand(0)->getType()&&
          (LHSConv->hasOneUse() || RHSConv->hasOneUse()) &&
          WillNotOverflowSignedAdd(LHSConv->getOperand(0),
                                   RHSConv->getOperand(0))) {
        // Insert the new integer add.
        Value *NewAdd = Builder->CreateNSWAdd(LHSConv->getOperand(0), 
                                             RHSConv->getOperand(0), "addconv");
        return new SExtInst(NewAdd, I.getType());
      }
    }
  }

  return Changed ? &I : 0;
}

Instruction *InstCombiner::visitFAdd(BinaryOperator &I) {
  bool Changed = SimplifyAssociativeOrCommutative(I);
  Value *LHS = I.getOperand(0), *RHS = I.getOperand(1);

  if (Constant *RHSC = dyn_cast<Constant>(RHS)) {
    // X + 0 --> X
    if (ConstantFP *CFP = dyn_cast<ConstantFP>(RHSC)) {
      if (CFP->isExactlyValue(ConstantFP::getNegativeZero
                              (I.getType())->getValueAPF()))
        return ReplaceInstUsesWith(I, LHS);
    }

    if (isa<PHINode>(LHS))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
  }

  // -A + B  -->  B - A
  // -A + -B  -->  -(A + B)
  if (Value *LHSV = dyn_castFNegVal(LHS))
    return BinaryOperator::CreateFSub(RHS, LHSV);

  // A + -B  -->  A - B
  if (!isa<Constant>(RHS))
    if (Value *V = dyn_castFNegVal(RHS))
      return BinaryOperator::CreateFSub(LHS, V);

  // Check for X+0.0.  Simplify it to X if we know X is not -0.0.
  if (ConstantFP *CFP = dyn_cast<ConstantFP>(RHS))
    if (CFP->getValueAPF().isPosZero() && CannotBeNegativeZero(LHS))
      return ReplaceInstUsesWith(I, LHS);

  // Check for (fadd double (sitofp x), y), see if we can merge this into an
  // integer add followed by a promotion.
  if (SIToFPInst *LHSConv = dyn_cast<SIToFPInst>(LHS)) {
    // (fadd double (sitofp x), fpcst) --> (sitofp (add int x, intcst))
    // ... if the constant fits in the integer value.  This is useful for things
    // like (double)(x & 1234) + 4.0 -> (double)((X & 1234)+4) which no longer
    // requires a constant pool load, and generally allows the add to be better
    // instcombined.
    if (ConstantFP *CFP = dyn_cast<ConstantFP>(RHS)) {
      Constant *CI = 
      ConstantExpr::getFPToSI(CFP, LHSConv->getOperand(0)->getType());
      if (LHSConv->hasOneUse() &&
          ConstantExpr::getSIToFP(CI, I.getType()) == CFP &&
          WillNotOverflowSignedAdd(LHSConv->getOperand(0), CI)) {
        // Insert the new integer add.
        Value *NewAdd = Builder->CreateNSWAdd(LHSConv->getOperand(0),
                                              CI, "addconv");
        return new SIToFPInst(NewAdd, I.getType());
      }
    }
    
    // (fadd double (sitofp x), (sitofp y)) --> (sitofp (add int x, y))
    if (SIToFPInst *RHSConv = dyn_cast<SIToFPInst>(RHS)) {
      // Only do this if x/y have the same type, if at last one of them has a
      // single use (so we don't increase the number of int->fp conversions),
      // and if the integer add will not overflow.
      if (LHSConv->getOperand(0)->getType()==RHSConv->getOperand(0)->getType()&&
          (LHSConv->hasOneUse() || RHSConv->hasOneUse()) &&
          WillNotOverflowSignedAdd(LHSConv->getOperand(0),
                                   RHSConv->getOperand(0))) {
        // Insert the new integer add.
        Value *NewAdd = Builder->CreateNSWAdd(LHSConv->getOperand(0), 
                                              RHSConv->getOperand(0),"addconv");
        return new SIToFPInst(NewAdd, I.getType());
      }
    }
  }
  
  return Changed ? &I : 0;
}


/// EmitGEPOffset - Given a getelementptr instruction/constantexpr, emit the
/// code necessary to compute the offset from the base pointer (without adding
/// in the base pointer).  Return the result as a signed integer of intptr size.
Value *InstCombiner::EmitGEPOffset(User *GEP) {
  TargetData &TD = *getTargetData();
  gep_type_iterator GTI = gep_type_begin(GEP);
  const Type *IntPtrTy = TD.getIntPtrType(GEP->getContext());
  Value *Result = Constant::getNullValue(IntPtrTy);

  // Build a mask for high order bits.
  unsigned IntPtrWidth = TD.getPointerSizeInBits();
  uint64_t PtrSizeMask = ~0ULL >> (64-IntPtrWidth);

  for (User::op_iterator i = GEP->op_begin() + 1, e = GEP->op_end(); i != e;
       ++i, ++GTI) {
    Value *Op = *i;
    uint64_t Size = TD.getTypeAllocSize(GTI.getIndexedType()) & PtrSizeMask;
    if (ConstantInt *OpC = dyn_cast<ConstantInt>(Op)) {
      if (OpC->isZero()) continue;
      
      // Handle a struct index, which adds its field offset to the pointer.
      if (const StructType *STy = dyn_cast<StructType>(*GTI)) {
        Size = TD.getStructLayout(STy)->getElementOffset(OpC->getZExtValue());
        
        Result = Builder->CreateAdd(Result,
                                    ConstantInt::get(IntPtrTy, Size),
                                    GEP->getName()+".offs");
        continue;
      }
      
      Constant *Scale = ConstantInt::get(IntPtrTy, Size);
      Constant *OC =
              ConstantExpr::getIntegerCast(OpC, IntPtrTy, true /*SExt*/);
      Scale = ConstantExpr::getMul(OC, Scale);
      // Emit an add instruction.
      Result = Builder->CreateAdd(Result, Scale, GEP->getName()+".offs");
      continue;
    }
    // Convert to correct type.
    if (Op->getType() != IntPtrTy)
      Op = Builder->CreateIntCast(Op, IntPtrTy, true, Op->getName()+".c");
    if (Size != 1) {
      Constant *Scale = ConstantInt::get(IntPtrTy, Size);
      // We'll let instcombine(mul) convert this to a shl if possible.
      Op = Builder->CreateMul(Op, Scale, GEP->getName()+".idx");
    }

    // Emit an add instruction.
    Result = Builder->CreateAdd(Op, Result, GEP->getName()+".offs");
  }
  return Result;
}




/// Optimize pointer differences into the same array into a size.  Consider:
///  &A[10] - &A[0]: we should compile this to "10".  LHS/RHS are the pointer
/// operands to the ptrtoint instructions for the LHS/RHS of the subtract.
///
Value *InstCombiner::OptimizePointerDifference(Value *LHS, Value *RHS,
                                               const Type *Ty) {
  assert(TD && "Must have target data info for this");
  
  // If LHS is a gep based on RHS or RHS is a gep based on LHS, we can optimize
  // this.
  bool Swapped = false;
  GetElementPtrInst *GEP = 0;
  ConstantExpr *CstGEP = 0;
  
  // TODO: Could also optimize &A[i] - &A[j] -> "i-j", and "&A.foo[i] - &A.foo".
  // For now we require one side to be the base pointer "A" or a constant
  // expression derived from it.
  if (GetElementPtrInst *LHSGEP = dyn_cast<GetElementPtrInst>(LHS)) {
    // (gep X, ...) - X
    if (LHSGEP->getOperand(0) == RHS) {
      GEP = LHSGEP;
      Swapped = false;
    } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(RHS)) {
      // (gep X, ...) - (ce_gep X, ...)
      if (CE->getOpcode() == Instruction::GetElementPtr &&
          LHSGEP->getOperand(0) == CE->getOperand(0)) {
        CstGEP = CE;
        GEP = LHSGEP;
        Swapped = false;
      }
    }
  }
  
  if (GetElementPtrInst *RHSGEP = dyn_cast<GetElementPtrInst>(RHS)) {
    // X - (gep X, ...)
    if (RHSGEP->getOperand(0) == LHS) {
      GEP = RHSGEP;
      Swapped = true;
    } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(LHS)) {
      // (ce_gep X, ...) - (gep X, ...)
      if (CE->getOpcode() == Instruction::GetElementPtr &&
          RHSGEP->getOperand(0) == CE->getOperand(0)) {
        CstGEP = CE;
        GEP = RHSGEP;
        Swapped = true;
      }
    }
  }
  
  if (GEP == 0)
    return 0;
  
  // Emit the offset of the GEP and an intptr_t.
  Value *Result = EmitGEPOffset(GEP);
  
  // If we had a constant expression GEP on the other side offsetting the
  // pointer, subtract it from the offset we have.
  if (CstGEP) {
    Value *CstOffset = EmitGEPOffset(CstGEP);
    Result = Builder->CreateSub(Result, CstOffset);
  }
  

  // If we have p - gep(p, ...)  then we have to negate the result.
  if (Swapped)
    Result = Builder->CreateNeg(Result, "diff.neg");

  return Builder->CreateIntCast(Result, Ty, true);
}


Instruction *InstCombiner::visitSub(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Op0 == Op1)                        // sub X, X  -> 0
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  // If this is a 'B = x-(-A)', change to B = x+A.  This preserves NSW/NUW.
  if (Value *V = dyn_castNegVal(Op1)) {
    BinaryOperator *Res = BinaryOperator::CreateAdd(Op0, V);
    Res->setHasNoSignedWrap(I.hasNoSignedWrap());
    Res->setHasNoUnsignedWrap(I.hasNoUnsignedWrap());
    return Res;
  }

  if (isa<UndefValue>(Op0))
    return ReplaceInstUsesWith(I, Op0);    // undef - X -> undef
  if (isa<UndefValue>(Op1))
    return ReplaceInstUsesWith(I, Op1);    // X - undef -> undef
  if (I.getType()->isIntegerTy(1))
    return BinaryOperator::CreateXor(Op0, Op1);
  
  if (ConstantInt *C = dyn_cast<ConstantInt>(Op0)) {
    // Replace (-1 - A) with (~A).
    if (C->isAllOnesValue())
      return BinaryOperator::CreateNot(Op1);

    // C - ~X == X + (1+C)
    Value *X = 0;
    if (match(Op1, m_Not(m_Value(X))))
      return BinaryOperator::CreateAdd(X, AddOne(C));

    // -(X >>u 31) -> (X >>s 31)
    // -(X >>s 31) -> (X >>u 31)
    if (C->isZero()) {
      if (BinaryOperator *SI = dyn_cast<BinaryOperator>(Op1)) {
        if (SI->getOpcode() == Instruction::LShr) {
          if (ConstantInt *CU = dyn_cast<ConstantInt>(SI->getOperand(1))) {
            // Check to see if we are shifting out everything but the sign bit.
            if (CU->getLimitedValue(SI->getType()->getPrimitiveSizeInBits()) ==
                SI->getType()->getPrimitiveSizeInBits()-1) {
              // Ok, the transformation is safe.  Insert AShr.
              return BinaryOperator::Create(Instruction::AShr, 
                                          SI->getOperand(0), CU, SI->getName());
            }
          }
        } else if (SI->getOpcode() == Instruction::AShr) {
          if (ConstantInt *CU = dyn_cast<ConstantInt>(SI->getOperand(1))) {
            // Check to see if we are shifting out everything but the sign bit.
            if (CU->getLimitedValue(SI->getType()->getPrimitiveSizeInBits()) ==
                SI->getType()->getPrimitiveSizeInBits()-1) {
              // Ok, the transformation is safe.  Insert LShr. 
              return BinaryOperator::CreateLShr(
                                          SI->getOperand(0), CU, SI->getName());
            }
          }
        }
      }
    }

    // Try to fold constant sub into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op1))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;

    // C - zext(bool) -> bool ? C - 1 : C
    if (ZExtInst *ZI = dyn_cast<ZExtInst>(Op1))
      if (ZI->getSrcTy() == Type::getInt1Ty(I.getContext()))
        return SelectInst::Create(ZI->getOperand(0), SubOne(C), C);
  }

  if (BinaryOperator *Op1I = dyn_cast<BinaryOperator>(Op1)) {
    if (Op1I->getOpcode() == Instruction::Add) {
      if (Op1I->getOperand(0) == Op0)              // X-(X+Y) == -Y
        return BinaryOperator::CreateNeg(Op1I->getOperand(1),
                                         I.getName());
      else if (Op1I->getOperand(1) == Op0)         // X-(Y+X) == -Y
        return BinaryOperator::CreateNeg(Op1I->getOperand(0),
                                         I.getName());
      else if (ConstantInt *CI1 = dyn_cast<ConstantInt>(I.getOperand(0))) {
        if (ConstantInt *CI2 = dyn_cast<ConstantInt>(Op1I->getOperand(1)))
          // C1-(X+C2) --> (C1-C2)-X
          return BinaryOperator::CreateSub(
            ConstantExpr::getSub(CI1, CI2), Op1I->getOperand(0));
      }
    }

    if (Op1I->hasOneUse()) {
      // Replace (x - (y - z)) with (x + (z - y)) if the (y - z) subexpression
      // is not used by anyone else...
      //
      if (Op1I->getOpcode() == Instruction::Sub) {
        // Swap the two operands of the subexpr...
        Value *IIOp0 = Op1I->getOperand(0), *IIOp1 = Op1I->getOperand(1);
        Op1I->setOperand(0, IIOp1);
        Op1I->setOperand(1, IIOp0);

        // Create the new top level add instruction...
        return BinaryOperator::CreateAdd(Op0, Op1);
      }

      // Replace (A - (A & B)) with (A & ~B) if this is the only use of (A&B)...
      //
      if (Op1I->getOpcode() == Instruction::And &&
          (Op1I->getOperand(0) == Op0 || Op1I->getOperand(1) == Op0)) {
        Value *OtherOp = Op1I->getOperand(Op1I->getOperand(0) == Op0);

        Value *NewNot = Builder->CreateNot(OtherOp, "B.not");
        return BinaryOperator::CreateAnd(Op0, NewNot);
      }

      // 0 - (X sdiv C)  -> (X sdiv -C)
      if (Op1I->getOpcode() == Instruction::SDiv)
        if (ConstantInt *CSI = dyn_cast<ConstantInt>(Op0))
          if (CSI->isZero())
            if (Constant *DivRHS = dyn_cast<Constant>(Op1I->getOperand(1)))
              return BinaryOperator::CreateSDiv(Op1I->getOperand(0),
                                          ConstantExpr::getNeg(DivRHS));

      // 0 - (C << X)  -> (-C << X)
      if (Op1I->getOpcode() == Instruction::Shl)
        if (ConstantInt *CSI = dyn_cast<ConstantInt>(Op0))
          if (CSI->isZero())
            if (Value *ShlLHSNeg = dyn_castNegVal(Op1I->getOperand(0)))
              return BinaryOperator::CreateShl(ShlLHSNeg, Op1I->getOperand(1));

      // X - X*C --> X * (1-C)
      ConstantInt *C2 = 0;
      if (dyn_castFoldableMul(Op1I, C2) == Op0) {
        Constant *CP1 = 
          ConstantExpr::getSub(ConstantInt::get(I.getType(), 1),
                                             C2);
        return BinaryOperator::CreateMul(Op0, CP1);
      }
    }
  }

  if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0)) {
    if (Op0I->getOpcode() == Instruction::Add) {
      if (Op0I->getOperand(0) == Op1)             // (Y+X)-Y == X
        return ReplaceInstUsesWith(I, Op0I->getOperand(1));
      else if (Op0I->getOperand(1) == Op1)        // (X+Y)-Y == X
        return ReplaceInstUsesWith(I, Op0I->getOperand(0));
    } else if (Op0I->getOpcode() == Instruction::Sub) {
      if (Op0I->getOperand(0) == Op1)             // (X-Y)-X == -Y
        return BinaryOperator::CreateNeg(Op0I->getOperand(1),
                                         I.getName());
    }
  }

  ConstantInt *C1;
  if (Value *X = dyn_castFoldableMul(Op0, C1)) {
    if (X == Op1)  // X*C - X --> X * (C-1)
      return BinaryOperator::CreateMul(Op1, SubOne(C1));

    ConstantInt *C2;   // X*C1 - X*C2 -> X * (C1-C2)
    if (X == dyn_castFoldableMul(Op1, C2))
      return BinaryOperator::CreateMul(X, ConstantExpr::getSub(C1, C2));
  }
  
  // Optimize pointer differences into the same array into a size.  Consider:
  //  &A[10] - &A[0]: we should compile this to "10".
  if (TD) {
    Value *LHSOp, *RHSOp;
    if (match(Op0, m_PtrToInt(m_Value(LHSOp))) &&
        match(Op1, m_PtrToInt(m_Value(RHSOp))))
      if (Value *Res = OptimizePointerDifference(LHSOp, RHSOp, I.getType()))
        return ReplaceInstUsesWith(I, Res);
    
    // trunc(p)-trunc(q) -> trunc(p-q)
    if (match(Op0, m_Trunc(m_PtrToInt(m_Value(LHSOp)))) &&
        match(Op1, m_Trunc(m_PtrToInt(m_Value(RHSOp)))))
      if (Value *Res = OptimizePointerDifference(LHSOp, RHSOp, I.getType()))
        return ReplaceInstUsesWith(I, Res);
  }
  
  return 0;
}

Instruction *InstCombiner::visitFSub(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // If this is a 'B = x-(-A)', change to B = x+A...
  if (Value *V = dyn_castFNegVal(Op1))
    return BinaryOperator::CreateFAdd(Op0, V);

  return 0;
}
