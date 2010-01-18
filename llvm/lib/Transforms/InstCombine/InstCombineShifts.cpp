//===- InstCombineShifts.cpp ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the visitShl, visitLShr, and visitAShr functions.
//
//===----------------------------------------------------------------------===//

#include "InstCombine.h"
#include "llvm/Support/PatternMatch.h"
using namespace llvm;
using namespace PatternMatch;

Instruction *InstCombiner::commonShiftTransforms(BinaryOperator &I) {
  assert(I.getOperand(1)->getType() == I.getOperand(0)->getType());
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // shl X, 0 == X and shr X, 0 == X
  // shl 0, X == 0 and shr 0, X == 0
  if (Op1 == Constant::getNullValue(Op1->getType()) ||
      Op0 == Constant::getNullValue(Op0->getType()))
    return ReplaceInstUsesWith(I, Op0);
  
  if (isa<UndefValue>(Op0)) {            
    if (I.getOpcode() == Instruction::AShr) // undef >>s X -> undef
      return ReplaceInstUsesWith(I, Op0);
    else                                    // undef << X -> 0, undef >>u X -> 0
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
  }
  if (isa<UndefValue>(Op1)) {
    if (I.getOpcode() == Instruction::AShr)  // X >>s undef -> X
      return ReplaceInstUsesWith(I, Op0);          
    else                                     // X << undef, X >>u undef -> 0
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
  }

  // See if we can fold away this shift.
  if (SimplifyDemandedInstructionBits(I))
    return &I;

  // Try to fold constant and into select arguments.
  if (isa<Constant>(Op0))
    if (SelectInst *SI = dyn_cast<SelectInst>(Op1))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;

  if (ConstantInt *CUI = dyn_cast<ConstantInt>(Op1))
    if (Instruction *Res = FoldShiftByConstant(Op0, CUI, I))
      return Res;
  return 0;
}

Instruction *InstCombiner::FoldShiftByConstant(Value *Op0, ConstantInt *Op1,
                                               BinaryOperator &I) {
  bool isLeftShift = I.getOpcode() == Instruction::Shl;

  // See if we can simplify any instructions used by the instruction whose sole 
  // purpose is to compute bits we don't care about.
  uint32_t TypeBits = Op0->getType()->getScalarSizeInBits();
  
  // shl i32 X, 32 = 0 and srl i8 Y, 9 = 0, ... just don't eliminate
  // a signed shift.
  //
  if (Op1->uge(TypeBits)) {
    if (I.getOpcode() != Instruction::AShr)
      return ReplaceInstUsesWith(I, Constant::getNullValue(Op0->getType()));
    else {
      I.setOperand(1, ConstantInt::get(I.getType(), TypeBits-1));
      return &I;
    }
  }
  
  // ((X*C1) << C2) == (X * (C1 << C2))
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(Op0))
    if (BO->getOpcode() == Instruction::Mul && isLeftShift)
      if (Constant *BOOp = dyn_cast<Constant>(BO->getOperand(1)))
        return BinaryOperator::CreateMul(BO->getOperand(0),
                                        ConstantExpr::getShl(BOOp, Op1));
  
  // Try to fold constant and into select arguments.
  if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
    if (Instruction *R = FoldOpIntoSelect(I, SI))
      return R;
  if (isa<PHINode>(Op0))
    if (Instruction *NV = FoldOpIntoPhi(I))
      return NV;
  
  // Fold shift2(trunc(shift1(x,c1)), c2) -> trunc(shift2(shift1(x,c1),c2))
  if (TruncInst *TI = dyn_cast<TruncInst>(Op0)) {
    Instruction *TrOp = dyn_cast<Instruction>(TI->getOperand(0));
    // If 'shift2' is an ashr, we would have to get the sign bit into a funny
    // place.  Don't try to do this transformation in this case.  Also, we
    // require that the input operand is a shift-by-constant so that we have
    // confidence that the shifts will get folded together.  We could do this
    // xform in more cases, but it is unlikely to be profitable.
    if (TrOp && I.isLogicalShift() && TrOp->isShift() && 
        isa<ConstantInt>(TrOp->getOperand(1))) {
      // Okay, we'll do this xform.  Make the shift of shift.
      Constant *ShAmt = ConstantExpr::getZExt(Op1, TrOp->getType());
      // (shift2 (shift1 & 0x00FF), c2)
      Value *NSh = Builder->CreateBinOp(I.getOpcode(), TrOp, ShAmt,I.getName());

      // For logical shifts, the truncation has the effect of making the high
      // part of the register be zeros.  Emulate this by inserting an AND to
      // clear the top bits as needed.  This 'and' will usually be zapped by
      // other xforms later if dead.
      unsigned SrcSize = TrOp->getType()->getScalarSizeInBits();
      unsigned DstSize = TI->getType()->getScalarSizeInBits();
      APInt MaskV(APInt::getLowBitsSet(SrcSize, DstSize));
      
      // The mask we constructed says what the trunc would do if occurring
      // between the shifts.  We want to know the effect *after* the second
      // shift.  We know that it is a logical shift by a constant, so adjust the
      // mask as appropriate.
      if (I.getOpcode() == Instruction::Shl)
        MaskV <<= Op1->getZExtValue();
      else {
        assert(I.getOpcode() == Instruction::LShr && "Unknown logical shift");
        MaskV = MaskV.lshr(Op1->getZExtValue());
      }

      // shift1 & 0x00FF
      Value *And = Builder->CreateAnd(NSh,
                                      ConstantInt::get(I.getContext(), MaskV),
                                      TI->getName());

      // Return the value truncated to the interesting size.
      return new TruncInst(And, I.getType());
    }
  }
  
  if (Op0->hasOneUse()) {
    if (BinaryOperator *Op0BO = dyn_cast<BinaryOperator>(Op0)) {
      // Turn ((X >> C) + Y) << C  ->  (X + (Y << C)) & (~0 << C)
      Value *V1, *V2;
      ConstantInt *CC;
      switch (Op0BO->getOpcode()) {
      default: break;
      case Instruction::Add:
      case Instruction::And:
      case Instruction::Or:
      case Instruction::Xor: {
        // These operators commute.
        // Turn (Y + (X >> C)) << C  ->  (X + (Y << C)) & (~0 << C)
        if (isLeftShift && Op0BO->getOperand(1)->hasOneUse() &&
            match(Op0BO->getOperand(1), m_Shr(m_Value(V1),
                  m_Specific(Op1)))) {
          Value *YS =         // (Y << C)
            Builder->CreateShl(Op0BO->getOperand(0), Op1, Op0BO->getName());
          // (X + (Y << C))
          Value *X = Builder->CreateBinOp(Op0BO->getOpcode(), YS, V1,
                                          Op0BO->getOperand(1)->getName());
          uint32_t Op1Val = Op1->getLimitedValue(TypeBits);
          return BinaryOperator::CreateAnd(X, ConstantInt::get(I.getContext(),
                     APInt::getHighBitsSet(TypeBits, TypeBits-Op1Val)));
        }
        
        // Turn (Y + ((X >> C) & CC)) << C  ->  ((X & (CC << C)) + (Y << C))
        Value *Op0BOOp1 = Op0BO->getOperand(1);
        if (isLeftShift && Op0BOOp1->hasOneUse() &&
            match(Op0BOOp1, 
                  m_And(m_Shr(m_Value(V1), m_Specific(Op1)),
                        m_ConstantInt(CC))) &&
            cast<BinaryOperator>(Op0BOOp1)->getOperand(0)->hasOneUse()) {
          Value *YS =   // (Y << C)
            Builder->CreateShl(Op0BO->getOperand(0), Op1,
                                         Op0BO->getName());
          // X & (CC << C)
          Value *XM = Builder->CreateAnd(V1, ConstantExpr::getShl(CC, Op1),
                                         V1->getName()+".mask");
          return BinaryOperator::Create(Op0BO->getOpcode(), YS, XM);
        }
      }
        
      // FALL THROUGH.
      case Instruction::Sub: {
        // Turn ((X >> C) + Y) << C  ->  (X + (Y << C)) & (~0 << C)
        if (isLeftShift && Op0BO->getOperand(0)->hasOneUse() &&
            match(Op0BO->getOperand(0), m_Shr(m_Value(V1),
                  m_Specific(Op1)))) {
          Value *YS =  // (Y << C)
            Builder->CreateShl(Op0BO->getOperand(1), Op1, Op0BO->getName());
          // (X + (Y << C))
          Value *X = Builder->CreateBinOp(Op0BO->getOpcode(), V1, YS,
                                          Op0BO->getOperand(0)->getName());
          uint32_t Op1Val = Op1->getLimitedValue(TypeBits);
          return BinaryOperator::CreateAnd(X, ConstantInt::get(I.getContext(),
                     APInt::getHighBitsSet(TypeBits, TypeBits-Op1Val)));
        }
        
        // Turn (((X >> C)&CC) + Y) << C  ->  (X + (Y << C)) & (CC << C)
        if (isLeftShift && Op0BO->getOperand(0)->hasOneUse() &&
            match(Op0BO->getOperand(0),
                  m_And(m_Shr(m_Value(V1), m_Value(V2)),
                        m_ConstantInt(CC))) && V2 == Op1 &&
            cast<BinaryOperator>(Op0BO->getOperand(0))
                ->getOperand(0)->hasOneUse()) {
          Value *YS = // (Y << C)
            Builder->CreateShl(Op0BO->getOperand(1), Op1, Op0BO->getName());
          // X & (CC << C)
          Value *XM = Builder->CreateAnd(V1, ConstantExpr::getShl(CC, Op1),
                                         V1->getName()+".mask");
          
          return BinaryOperator::Create(Op0BO->getOpcode(), XM, YS);
        }
        
        break;
      }
      }
      
      
      // If the operand is an bitwise operator with a constant RHS, and the
      // shift is the only use, we can pull it out of the shift.
      if (ConstantInt *Op0C = dyn_cast<ConstantInt>(Op0BO->getOperand(1))) {
        bool isValid = true;     // Valid only for And, Or, Xor
        bool highBitSet = false; // Transform if high bit of constant set?
        
        switch (Op0BO->getOpcode()) {
        default: isValid = false; break;   // Do not perform transform!
        case Instruction::Add:
          isValid = isLeftShift;
          break;
        case Instruction::Or:
        case Instruction::Xor:
          highBitSet = false;
          break;
        case Instruction::And:
          highBitSet = true;
          break;
        }
        
        // If this is a signed shift right, and the high bit is modified
        // by the logical operation, do not perform the transformation.
        // The highBitSet boolean indicates the value of the high bit of
        // the constant which would cause it to be modified for this
        // operation.
        //
        if (isValid && I.getOpcode() == Instruction::AShr)
          isValid = Op0C->getValue()[TypeBits-1] == highBitSet;
        
        if (isValid) {
          Constant *NewRHS = ConstantExpr::get(I.getOpcode(), Op0C, Op1);
          
          Value *NewShift =
            Builder->CreateBinOp(I.getOpcode(), Op0BO->getOperand(0), Op1);
          NewShift->takeName(Op0BO);
          
          return BinaryOperator::Create(Op0BO->getOpcode(), NewShift,
                                        NewRHS);
        }
      }
    }
  }
  
  // Find out if this is a shift of a shift by a constant.
  BinaryOperator *ShiftOp = dyn_cast<BinaryOperator>(Op0);
  if (ShiftOp && !ShiftOp->isShift())
    ShiftOp = 0;
  
  if (ShiftOp && isa<ConstantInt>(ShiftOp->getOperand(1))) {
    ConstantInt *ShiftAmt1C = cast<ConstantInt>(ShiftOp->getOperand(1));
    uint32_t ShiftAmt1 = ShiftAmt1C->getLimitedValue(TypeBits);
    uint32_t ShiftAmt2 = Op1->getLimitedValue(TypeBits);
    assert(ShiftAmt2 != 0 && "Should have been simplified earlier");
    if (ShiftAmt1 == 0) return 0;  // Will be simplified in the future.
    Value *X = ShiftOp->getOperand(0);
    
    uint32_t AmtSum = ShiftAmt1+ShiftAmt2;   // Fold into one big shift.
    
    const IntegerType *Ty = cast<IntegerType>(I.getType());
    
    // Check for (X << c1) << c2  and  (X >> c1) >> c2
    if (I.getOpcode() == ShiftOp->getOpcode()) {
      // If this is oversized composite shift, then unsigned shifts get 0, ashr
      // saturates.
      if (AmtSum >= TypeBits) {
        if (I.getOpcode() != Instruction::AShr)
          return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
        AmtSum = TypeBits-1;  // Saturate to 31 for i32 ashr.
      }
      
      return BinaryOperator::Create(I.getOpcode(), X,
                                    ConstantInt::get(Ty, AmtSum));
    }
    
    if (ShiftOp->getOpcode() == Instruction::LShr &&
        I.getOpcode() == Instruction::AShr) {
      if (AmtSum >= TypeBits)
        return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
      
      // ((X >>u C1) >>s C2) -> (X >>u (C1+C2))  since C1 != 0.
      return BinaryOperator::CreateLShr(X, ConstantInt::get(Ty, AmtSum));
    }
    
    if (ShiftOp->getOpcode() == Instruction::AShr &&
        I.getOpcode() == Instruction::LShr) {
      // ((X >>s C1) >>u C2) -> ((X >>s (C1+C2)) & mask) since C1 != 0.
      if (AmtSum >= TypeBits)
        AmtSum = TypeBits-1;
      
      Value *Shift = Builder->CreateAShr(X, ConstantInt::get(Ty, AmtSum));

      APInt Mask(APInt::getLowBitsSet(TypeBits, TypeBits - ShiftAmt2));
      return BinaryOperator::CreateAnd(Shift,
                                       ConstantInt::get(I.getContext(), Mask));
    }
    
    // Okay, if we get here, one shift must be left, and the other shift must be
    // right.  See if the amounts are equal.
    if (ShiftAmt1 == ShiftAmt2) {
      // If we have ((X >>? C) << C), turn this into X & (-1 << C).
      if (I.getOpcode() == Instruction::Shl) {
        APInt Mask(APInt::getHighBitsSet(TypeBits, TypeBits - ShiftAmt1));
        return BinaryOperator::CreateAnd(X,
                                         ConstantInt::get(I.getContext(),Mask));
      }
      // If we have ((X << C) >>u C), turn this into X & (-1 >>u C).
      if (I.getOpcode() == Instruction::LShr) {
        APInt Mask(APInt::getLowBitsSet(TypeBits, TypeBits - ShiftAmt1));
        return BinaryOperator::CreateAnd(X,
                                        ConstantInt::get(I.getContext(), Mask));
      }
    } else if (ShiftAmt1 < ShiftAmt2) {
      uint32_t ShiftDiff = ShiftAmt2-ShiftAmt1;
      
      // (X >>? C1) << C2 --> X << (C2-C1) & (-1 << C2)
      if (I.getOpcode() == Instruction::Shl) {
        assert(ShiftOp->getOpcode() == Instruction::LShr ||
               ShiftOp->getOpcode() == Instruction::AShr);
        Value *Shift = Builder->CreateShl(X, ConstantInt::get(Ty, ShiftDiff));
        
        APInt Mask(APInt::getHighBitsSet(TypeBits, TypeBits - ShiftAmt2));
        return BinaryOperator::CreateAnd(Shift,
                                         ConstantInt::get(I.getContext(),Mask));
      }
      
      // (X << C1) >>u C2  --> X >>u (C2-C1) & (-1 >> C2)
      if (I.getOpcode() == Instruction::LShr) {
        assert(ShiftOp->getOpcode() == Instruction::Shl);
        Value *Shift = Builder->CreateLShr(X, ConstantInt::get(Ty, ShiftDiff));
        
        APInt Mask(APInt::getLowBitsSet(TypeBits, TypeBits - ShiftAmt2));
        return BinaryOperator::CreateAnd(Shift,
                                         ConstantInt::get(I.getContext(),Mask));
      }
      
      // We can't handle (X << C1) >>s C2, it shifts arbitrary bits in.
    } else {
      assert(ShiftAmt2 < ShiftAmt1);
      uint32_t ShiftDiff = ShiftAmt1-ShiftAmt2;

      // (X >>? C1) << C2 --> X >>? (C1-C2) & (-1 << C2)
      if (I.getOpcode() == Instruction::Shl) {
        assert(ShiftOp->getOpcode() == Instruction::LShr ||
               ShiftOp->getOpcode() == Instruction::AShr);
        Value *Shift = Builder->CreateBinOp(ShiftOp->getOpcode(), X,
                                            ConstantInt::get(Ty, ShiftDiff));
        
        APInt Mask(APInt::getHighBitsSet(TypeBits, TypeBits - ShiftAmt2));
        return BinaryOperator::CreateAnd(Shift,
                                         ConstantInt::get(I.getContext(),Mask));
      }
      
      // (X << C1) >>u C2  --> X << (C1-C2) & (-1 >> C2)
      if (I.getOpcode() == Instruction::LShr) {
        assert(ShiftOp->getOpcode() == Instruction::Shl);
        Value *Shift = Builder->CreateShl(X, ConstantInt::get(Ty, ShiftDiff));
        
        APInt Mask(APInt::getLowBitsSet(TypeBits, TypeBits - ShiftAmt2));
        return BinaryOperator::CreateAnd(Shift,
                                         ConstantInt::get(I.getContext(),Mask));
      }
      
      // We can't handle (X << C1) >>a C2, it shifts arbitrary bits in.
    }
  }
  return 0;
}

Instruction *InstCombiner::visitShl(BinaryOperator &I) {
  return commonShiftTransforms(I);
}

Instruction *InstCombiner::visitLShr(BinaryOperator &I) {
  return commonShiftTransforms(I);
}

Instruction *InstCombiner::visitAShr(BinaryOperator &I) {
  if (Instruction *R = commonShiftTransforms(I))
    return R;
  
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  
  if (ConstantInt *CSI = dyn_cast<ConstantInt>(Op0)) {
    // ashr int -1, X = -1   (for any arithmetic shift rights of ~0)
    if (CSI->isAllOnesValue())
      return ReplaceInstUsesWith(I, CSI);
  }
  
  if (ConstantInt *Op1C = dyn_cast<ConstantInt>(Op1)) {
    // If the input is a SHL by the same constant (ashr (shl X, C), C), then we
    // have a sign-extend idiom.
    Value *X;
    if (match(Op0, m_Shl(m_Value(X), m_Specific(Op1)))) {
      // If the input value is known to already be sign extended enough, delete
      // the extension.
      if (ComputeNumSignBits(X) > Op1C->getZExtValue())
        return ReplaceInstUsesWith(I, X);

      // If the input is an extension from the shifted amount value, e.g.
      //   %x = zext i8 %A to i32
      //   %y = shl i32 %x, 24
      //   %z = ashr %y, 24
      // then turn this into "z = sext i8 A to i32".
      if (ZExtInst *ZI = dyn_cast<ZExtInst>(X)) {
        uint32_t SrcBits = ZI->getOperand(0)->getType()->getScalarSizeInBits();
        uint32_t DestBits = ZI->getType()->getScalarSizeInBits();
        if (Op1C->getZExtValue() == DestBits-SrcBits)
          return new SExtInst(ZI->getOperand(0), ZI->getType());
      }
    }
  }            
  
  // See if we can turn a signed shr into an unsigned shr.
  if (MaskedValueIsZero(Op0,
                        APInt::getSignBit(I.getType()->getScalarSizeInBits())))
    return BinaryOperator::CreateLShr(Op0, Op1);
  
  // Arithmetic shifting an all-sign-bit value is a no-op.
  unsigned NumSignBits = ComputeNumSignBits(Op0);
  if (NumSignBits == Op0->getType()->getScalarSizeInBits())
    return ReplaceInstUsesWith(I, Op0);
  
  return 0;
}

