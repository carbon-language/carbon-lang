//===-- TargetLowering.cpp - Implement the TargetLowering class -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the TargetLowering class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

TargetLowering::TargetLowering(TargetMachine &tm)
  : TM(tm), TD(TM.getTargetData()) {
  assert(ISD::BUILTIN_OP_END <= 128 &&
         "Fixed size array in TargetLowering is not large enough!");
  // All operations default to being supported.
  memset(OpActions, 0, sizeof(OpActions));

  IsLittleEndian = TD.isLittleEndian();
  ShiftAmountTy = SetCCResultTy = PointerTy = getValueType(TD.getIntPtrType());
  ShiftAmtHandling = Undefined;
  memset(RegClassForVT, 0,MVT::LAST_VALUETYPE*sizeof(TargetRegisterClass*));
  maxStoresPerMemset = maxStoresPerMemcpy = maxStoresPerMemmove = 8;
  allowUnalignedMemoryAccesses = false;
  UseUnderscoreSetJmpLongJmp = false;
  IntDivIsCheap = false;
  Pow2DivIsCheap = false;
  StackPointerRegisterToSaveRestore = 0;
  SchedPreferenceInfo = SchedulingForLatency;
}

TargetLowering::~TargetLowering() {}

/// setValueTypeAction - Set the action for a particular value type.  This
/// assumes an action has not already been set for this value type.
static void SetValueTypeAction(MVT::ValueType VT,
                               TargetLowering::LegalizeAction Action,
                               TargetLowering &TLI,
                               MVT::ValueType *TransformToType,
                        TargetLowering::ValueTypeActionImpl &ValueTypeActions) {
  ValueTypeActions.setTypeAction(VT, Action);
  if (Action == TargetLowering::Promote) {
    MVT::ValueType PromoteTo;
    if (VT == MVT::f32)
      PromoteTo = MVT::f64;
    else {
      unsigned LargerReg = VT+1;
      while (!TLI.isTypeLegal((MVT::ValueType)LargerReg)) {
        ++LargerReg;
        assert(MVT::isInteger((MVT::ValueType)LargerReg) &&
               "Nothing to promote to??");
      }
      PromoteTo = (MVT::ValueType)LargerReg;
    }

    assert(MVT::isInteger(VT) == MVT::isInteger(PromoteTo) &&
           MVT::isFloatingPoint(VT) == MVT::isFloatingPoint(PromoteTo) &&
           "Can only promote from int->int or fp->fp!");
    assert(VT < PromoteTo && "Must promote to a larger type!");
    TransformToType[VT] = PromoteTo;
  } else if (Action == TargetLowering::Expand) {
    assert((VT == MVT::Vector || MVT::isInteger(VT)) && VT > MVT::i8 &&
           "Cannot expand this type: target must support SOME integer reg!");
    // Expand to the next smaller integer type!
    TransformToType[VT] = (MVT::ValueType)(VT-1);
  }
}


/// computeRegisterProperties - Once all of the register classes are added,
/// this allows us to compute derived properties we expose.
void TargetLowering::computeRegisterProperties() {
  assert(MVT::LAST_VALUETYPE <= 32 &&
         "Too many value types for ValueTypeActions to hold!");

  // Everything defaults to one.
  for (unsigned i = 0; i != MVT::LAST_VALUETYPE; ++i)
    NumElementsForVT[i] = 1;

  // Find the largest integer register class.
  unsigned LargestIntReg = MVT::i128;
  for (; RegClassForVT[LargestIntReg] == 0; --LargestIntReg)
    assert(LargestIntReg != MVT::i1 && "No integer registers defined!");

  // Every integer value type larger than this largest register takes twice as
  // many registers to represent as the previous ValueType.
  unsigned ExpandedReg = LargestIntReg; ++LargestIntReg;
  for (++ExpandedReg; MVT::isInteger((MVT::ValueType)ExpandedReg);++ExpandedReg)
    NumElementsForVT[ExpandedReg] = 2*NumElementsForVT[ExpandedReg-1];

  // Inspect all of the ValueType's possible, deciding how to process them.
  for (unsigned IntReg = MVT::i1; IntReg <= MVT::i128; ++IntReg)
    // If we are expanding this type, expand it!
    if (getNumElements((MVT::ValueType)IntReg) != 1)
      SetValueTypeAction((MVT::ValueType)IntReg, Expand, *this, TransformToType,
                         ValueTypeActions);
    else if (!isTypeLegal((MVT::ValueType)IntReg))
      // Otherwise, if we don't have native support, we must promote to a
      // larger type.
      SetValueTypeAction((MVT::ValueType)IntReg, Promote, *this,
                         TransformToType, ValueTypeActions);
    else
      TransformToType[(MVT::ValueType)IntReg] = (MVT::ValueType)IntReg;

  // If the target does not have native support for F32, promote it to F64.
  if (!isTypeLegal(MVT::f32))
    SetValueTypeAction(MVT::f32, Promote, *this,
                       TransformToType, ValueTypeActions);
  else
    TransformToType[MVT::f32] = MVT::f32;
  
  // Set MVT::Vector to always be Expanded
  SetValueTypeAction(MVT::Vector, Expand, *this, TransformToType, 
                     ValueTypeActions);

  assert(isTypeLegal(MVT::f64) && "Target does not support FP?");
  TransformToType[MVT::f64] = MVT::f64;
}

const char *TargetLowering::getTargetNodeName(unsigned Opcode) const {
  return NULL;
}

//===----------------------------------------------------------------------===//
//  Optimization Methods
//===----------------------------------------------------------------------===//

/// ShrinkDemandedConstant - Check to see if the specified operand of the 
/// specified instruction is a constant integer.  If so, check to see if there
/// are any bits set in the constant that are not demanded.  If so, shrink the
/// constant and return true.
bool TargetLowering::TargetLoweringOpt::ShrinkDemandedConstant(SDOperand Op, 
                                                            uint64_t Demanded) {
  // FIXME: ISD::SELECT, ISD::SELECT_CC
  switch(Op.getOpcode()) {
  default: break;
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1)))
      if ((~Demanded & C->getValue()) != 0) {
        MVT::ValueType VT = Op.getValueType();
        SDOperand New = DAG.getNode(Op.getOpcode(), VT, Op.getOperand(0),
                                    DAG.getConstant(Demanded & C->getValue(), 
                                                    VT));
        return CombineTo(Op, New);
      }
    break;
  }
  return false;
}

/// SimplifyDemandedBits - Look at Op.  At this point, we know that only the
/// DemandedMask bits of the result of Op are ever used downstream.  If we can
/// use this information to simplify Op, create a new simplified DAG node and
/// return true, returning the original and new nodes in Old and New. Otherwise,
/// analyze the expression and return a mask of KnownOne and KnownZero bits for
/// the expression (used to simplify the caller).  The KnownZero/One bits may
/// only be accurate for those bits in the DemandedMask.
bool TargetLowering::SimplifyDemandedBits(SDOperand Op, uint64_t DemandedMask, 
                                          uint64_t &KnownZero,
                                          uint64_t &KnownOne,
                                          TargetLoweringOpt &TLO,
                                          unsigned Depth) const {
  KnownZero = KnownOne = 0;   // Don't know anything.
  // Other users may use these bits.
  if (!Op.Val->hasOneUse()) { 
    if (Depth != 0) {
      // If not at the root, Just compute the KnownZero/KnownOne bits to 
      // simplify things downstream.
      ComputeMaskedBits(Op, DemandedMask, KnownZero, KnownOne, Depth);
      return false;
    }
    // If this is the root being simplified, allow it to have multiple uses,
    // just set the DemandedMask to all bits.
    DemandedMask = MVT::getIntVTBitMask(Op.getValueType());
  } else if (DemandedMask == 0) {   
    // Not demanding any bits from Op.
    if (Op.getOpcode() != ISD::UNDEF)
      return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::UNDEF, Op.getValueType()));
    return false;
  } else if (Depth == 6) {        // Limit search depth.
    return false;
  }

  uint64_t KnownZero2, KnownOne2, KnownZeroOut, KnownOneOut;
  switch (Op.getOpcode()) {
  case ISD::Constant:
    // We know all of the bits for a constant!
    KnownOne = cast<ConstantSDNode>(Op)->getValue() & DemandedMask;
    KnownZero = ~KnownOne & DemandedMask;
    return false;   // Don't fall through, will infinitely loop.
  case ISD::AND:
    // If the RHS is a constant, check to see if the LHS would be zero without
    // using the bits from the RHS.  Below, we use knowledge about the RHS to
    // simplify the LHS, here we're using information from the LHS to simplify
    // the RHS.
    if (ConstantSDNode *RHSC = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      uint64_t LHSZero, LHSOne;
      ComputeMaskedBits(Op.getOperand(0), DemandedMask,
                        LHSZero, LHSOne, Depth+1);
      // If the LHS already has zeros where RHSC does, this and is dead.
      if ((LHSZero & DemandedMask) == (~RHSC->getValue() & DemandedMask))
        return TLO.CombineTo(Op, Op.getOperand(0));
      // If any of the set bits in the RHS are known zero on the LHS, shrink
      // the constant.
      if (TLO.ShrinkDemandedConstant(Op, ~LHSZero & DemandedMask))
        return true;
    }
    
    if (SimplifyDemandedBits(Op.getOperand(1), DemandedMask, KnownZero,
                             KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    if (SimplifyDemandedBits(Op.getOperand(0), DemandedMask & ~KnownZero,
                             KnownZero2, KnownOne2, TLO, Depth+1))
      return true;
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
      
    // If all of the demanded bits are known one on one side, return the other.
    // These bits cannot contribute to the result of the 'and'.
    if ((DemandedMask & ~KnownZero2 & KnownOne)==(DemandedMask & ~KnownZero2))
      return TLO.CombineTo(Op, Op.getOperand(0));
    if ((DemandedMask & ~KnownZero & KnownOne2)==(DemandedMask & ~KnownZero))
      return TLO.CombineTo(Op, Op.getOperand(1));
    // If all of the demanded bits in the inputs are known zeros, return zero.
    if ((DemandedMask & (KnownZero|KnownZero2)) == DemandedMask)
      return TLO.CombineTo(Op, TLO.DAG.getConstant(0, Op.getValueType()));
    // If the RHS is a constant, see if we can simplify it.
    if (TLO.ShrinkDemandedConstant(Op, DemandedMask & ~KnownZero2))
      return true;
      
    // Output known-1 bits are only known if set in both the LHS & RHS.
    KnownOne &= KnownOne2;
    // Output known-0 are known to be clear if zero in either the LHS | RHS.
    KnownZero |= KnownZero2;
    break;
  case ISD::OR:
    if (SimplifyDemandedBits(Op.getOperand(1), DemandedMask, KnownZero, 
                             KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    if (SimplifyDemandedBits(Op.getOperand(0), DemandedMask & ~KnownOne, 
                             KnownZero2, KnownOne2, TLO, Depth+1))
      return true;
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // If all of the demanded bits are known zero on one side, return the other.
    // These bits cannot contribute to the result of the 'or'.
    if ((DemandedMask & ~KnownOne2 & KnownZero) == (DemandedMask & ~KnownOne2))
      return TLO.CombineTo(Op, Op.getOperand(0));
    if ((DemandedMask & ~KnownOne & KnownZero2) == (DemandedMask & ~KnownOne))
      return TLO.CombineTo(Op, Op.getOperand(1));
    // If all of the potentially set bits on one side are known to be set on
    // the other side, just use the 'other' side.
    if ((DemandedMask & (~KnownZero) & KnownOne2) == 
        (DemandedMask & (~KnownZero)))
      return TLO.CombineTo(Op, Op.getOperand(0));
    if ((DemandedMask & (~KnownZero2) & KnownOne) == 
        (DemandedMask & (~KnownZero2)))
      return TLO.CombineTo(Op, Op.getOperand(1));
    // If the RHS is a constant, see if we can simplify it.
    if (TLO.ShrinkDemandedConstant(Op, DemandedMask))
      return true;
          
    // Output known-0 bits are only known if clear in both the LHS & RHS.
    KnownZero &= KnownZero2;
    // Output known-1 are known to be set if set in either the LHS | RHS.
    KnownOne |= KnownOne2;
    break;
  case ISD::XOR:
    if (SimplifyDemandedBits(Op.getOperand(1), DemandedMask, KnownZero, 
                             KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    if (SimplifyDemandedBits(Op.getOperand(0), DemandedMask, KnownZero2,
                             KnownOne2, TLO, Depth+1))
      return true;
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // If all of the demanded bits are known zero on one side, return the other.
    // These bits cannot contribute to the result of the 'xor'.
    if ((DemandedMask & KnownZero) == DemandedMask)
      return TLO.CombineTo(Op, Op.getOperand(0));
    if ((DemandedMask & KnownZero2) == DemandedMask)
      return TLO.CombineTo(Op, Op.getOperand(1));
    
    // Output known-0 bits are known if clear or set in both the LHS & RHS.
    KnownZeroOut = (KnownZero & KnownZero2) | (KnownOne & KnownOne2);
    // Output known-1 are known to be set if set in only one of the LHS, RHS.
    KnownOneOut = (KnownZero & KnownOne2) | (KnownOne & KnownZero2);
    
    // If all of the unknown bits are known to be zero on one side or the other
    // (but not both) turn this into an *inclusive* or.
    //    e.g. (A & C1)^(B & C2) -> (A & C1)|(B & C2) iff C1&C2 == 0
    if (uint64_t UnknownBits = DemandedMask & ~(KnownZeroOut|KnownOneOut))
      if ((UnknownBits & (KnownZero|KnownZero2)) == UnknownBits)
        return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::OR, Op.getValueType(),
                                                 Op.getOperand(0),
                                                 Op.getOperand(1)));
    // If all of the demanded bits on one side are known, and all of the set
    // bits on that side are also known to be set on the other side, turn this
    // into an AND, as we know the bits will be cleared.
    //    e.g. (X | C1) ^ C2 --> (X | C1) & ~C2 iff (C1&C2) == C2
    if ((DemandedMask & (KnownZero|KnownOne)) == DemandedMask) { // all known
      if ((KnownOne & KnownOne2) == KnownOne) {
        MVT::ValueType VT = Op.getValueType();
        SDOperand ANDC = TLO.DAG.getConstant(~KnownOne & DemandedMask, VT);
        return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::AND, VT, Op.getOperand(0),
                                                 ANDC));
      }
    }
    
    // If the RHS is a constant, see if we can simplify it.
    // FIXME: for XOR, we prefer to force bits to 1 if they will make a -1.
    if (TLO.ShrinkDemandedConstant(Op, DemandedMask))
      return true;
    
    KnownZero = KnownZeroOut;
    KnownOne  = KnownOneOut;
    break;
  case ISD::SETCC:
    // If we know the result of a setcc has the top bits zero, use this info.
    if (getSetCCResultContents() == TargetLowering::ZeroOrOneSetCCResult)
      KnownZero |= (MVT::getIntVTBitMask(Op.getValueType()) ^ 1ULL);
    break;
  case ISD::SELECT:
    if (SimplifyDemandedBits(Op.getOperand(2), DemandedMask, KnownZero, 
                             KnownOne, TLO, Depth+1))
      return true;
    if (SimplifyDemandedBits(Op.getOperand(1), DemandedMask, KnownZero2,
                             KnownOne2, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // If the operands are constants, see if we can simplify them.
    if (TLO.ShrinkDemandedConstant(Op, DemandedMask))
      return true;
    
    // Only known if known in both the LHS and RHS.
    KnownOne &= KnownOne2;
    KnownZero &= KnownZero2;
    break;
  case ISD::SELECT_CC:
    if (SimplifyDemandedBits(Op.getOperand(3), DemandedMask, KnownZero, 
                             KnownOne, TLO, Depth+1))
      return true;
    if (SimplifyDemandedBits(Op.getOperand(2), DemandedMask, KnownZero2,
                             KnownOne2, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // If the operands are constants, see if we can simplify them.
    if (TLO.ShrinkDemandedConstant(Op, DemandedMask))
      return true;
      
    // Only known if known in both the LHS and RHS.
    KnownOne &= KnownOne2;
    KnownZero &= KnownZero2;
    break;
  case ISD::SHL:
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      if (SimplifyDemandedBits(Op.getOperand(0), DemandedMask >> SA->getValue(),
                               KnownZero, KnownOne, TLO, Depth+1))
        return true;
      KnownZero <<= SA->getValue();
      KnownOne  <<= SA->getValue();
      KnownZero |= (1ULL << SA->getValue())-1;  // low bits known zero.
    }
    break;
  case ISD::SRL:
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      MVT::ValueType VT = Op.getValueType();
      unsigned ShAmt = SA->getValue();
      
      // Compute the new bits that are at the top now.
      uint64_t HighBits = (1ULL << ShAmt)-1;
      HighBits <<= MVT::getSizeInBits(VT) - ShAmt;
      uint64_t TypeMask = MVT::getIntVTBitMask(VT);
      
      if (SimplifyDemandedBits(Op.getOperand(0), 
                               (DemandedMask << ShAmt) & TypeMask,
                               KnownZero, KnownOne, TLO, Depth+1))
        return true;
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
      KnownZero &= TypeMask;
      KnownOne  &= TypeMask;
      KnownZero >>= ShAmt;
      KnownOne  >>= ShAmt;
      KnownZero |= HighBits;  // high bits known zero.
    }
    break;
  case ISD::SRA:
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      MVT::ValueType VT = Op.getValueType();
      unsigned ShAmt = SA->getValue();
      
      // Compute the new bits that are at the top now.
      uint64_t HighBits = (1ULL << ShAmt)-1;
      HighBits <<= MVT::getSizeInBits(VT) - ShAmt;
      uint64_t TypeMask = MVT::getIntVTBitMask(VT);
      
      if (SimplifyDemandedBits(Op.getOperand(0),
                               (DemandedMask << ShAmt) & TypeMask,
                               KnownZero, KnownOne, TLO, Depth+1))
        return true;
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
      KnownZero &= TypeMask;
      KnownOne  &= TypeMask;
      KnownZero >>= SA->getValue();
      KnownOne  >>= SA->getValue();
      
      // Handle the sign bits.
      uint64_t SignBit = MVT::getIntVTSignBit(VT);
      SignBit >>= SA->getValue();  // Adjust to where it is now in the mask.
      
      // If the input sign bit is known to be zero, or if none of the top bits
      // are demanded, turn this into an unsigned shift right.
      if ((KnownZero & SignBit) || (HighBits & ~DemandedMask) == HighBits) {
        return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::SRL, VT, Op.getOperand(0),
                                                 Op.getOperand(1)));
      } else if (KnownOne & SignBit) { // New bits are known one.
        KnownOne |= HighBits;
      }
    }
    break;
  case ISD::SIGN_EXTEND_INREG: {
    MVT::ValueType  VT = Op.getValueType();
    MVT::ValueType EVT = cast<VTSDNode>(Op.getOperand(1))->getVT();

    // Sign extension.  Compute the demanded bits in the result that are not 
    // present in the input.
    uint64_t NewBits = ~MVT::getIntVTBitMask(EVT) & DemandedMask;
    
    // If none of the extended bits are demanded, eliminate the sextinreg.
    if (NewBits == 0)
      return TLO.CombineTo(Op, Op.getOperand(0));

    uint64_t InSignBit = MVT::getIntVTSignBit(EVT);
    int64_t InputDemandedBits = DemandedMask & MVT::getIntVTBitMask(EVT);
    
    // Since the sign extended bits are demanded, we know that the sign
    // bit is demanded.
    InputDemandedBits |= InSignBit;

    if (SimplifyDemandedBits(Op.getOperand(0), InputDemandedBits,
                             KnownZero, KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 

    // If the sign bit of the input is known set or clear, then we know the
    // top bits of the result.
    
    // If the input sign bit is known zero, convert this into a zero extension.
    if (KnownZero & InSignBit)
      return TLO.CombineTo(Op, 
                           TLO.DAG.getZeroExtendInReg(Op.getOperand(0), EVT));
    
    if (KnownOne & InSignBit) {    // Input sign bit known set
      KnownOne |= NewBits;
      KnownZero &= ~NewBits;
    } else {                       // Input sign bit unknown
      KnownZero &= ~NewBits;
      KnownOne &= ~NewBits;
    }
    break;
  }
  case ISD::CTTZ:
  case ISD::CTLZ:
  case ISD::CTPOP: {
    MVT::ValueType VT = Op.getValueType();
    unsigned LowBits = Log2_32(MVT::getSizeInBits(VT))+1;
    KnownZero = ~((1ULL << LowBits)-1) & MVT::getIntVTBitMask(VT);
    KnownOne  = 0;
    break;
  }
  case ISD::ZEXTLOAD: {
    MVT::ValueType VT = cast<VTSDNode>(Op.getOperand(3))->getVT();
    KnownZero |= ~MVT::getIntVTBitMask(VT) & DemandedMask;
    break;
  }
  case ISD::ZERO_EXTEND: {
    uint64_t InMask = MVT::getIntVTBitMask(Op.getOperand(0).getValueType());
    
    // If none of the top bits are demanded, convert this into an any_extend.
    uint64_t NewBits = (~InMask) & DemandedMask;
    if (NewBits == 0)
      return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::ANY_EXTEND, 
                                               Op.getValueType(), 
                                               Op.getOperand(0)));
    
    if (SimplifyDemandedBits(Op.getOperand(0), DemandedMask & InMask,
                             KnownZero, KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    KnownZero |= NewBits;
    break;
  }
  case ISD::SIGN_EXTEND: {
    MVT::ValueType InVT = Op.getOperand(0).getValueType();
    uint64_t InMask    = MVT::getIntVTBitMask(InVT);
    uint64_t InSignBit = MVT::getIntVTSignBit(InVT);
    uint64_t NewBits   = (~InMask) & DemandedMask;
    
    // If none of the top bits are demanded, convert this into an any_extend.
    if (NewBits == 0)
      return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::ANY_EXTEND,Op.getValueType(),
                                           Op.getOperand(0)));
    
    // Since some of the sign extended bits are demanded, we know that the sign
    // bit is demanded.
    uint64_t InDemandedBits = DemandedMask & InMask;
    InDemandedBits |= InSignBit;
    
    if (SimplifyDemandedBits(Op.getOperand(0), InDemandedBits, KnownZero, 
                             KnownOne, TLO, Depth+1))
      return true;
    
    // If the sign bit is known zero, convert this to a zero extend.
    if (KnownZero & InSignBit)
      return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::ZERO_EXTEND, 
                                               Op.getValueType(), 
                                               Op.getOperand(0)));
    
    // If the sign bit is known one, the top bits match.
    if (KnownOne & InSignBit) {
      KnownOne  |= NewBits;
      KnownZero &= ~NewBits;
    } else {   // Otherwise, top bits aren't known.
      KnownOne  &= ~NewBits;
      KnownZero &= ~NewBits;
    }
    break;
  }
  case ISD::ANY_EXTEND: {
    uint64_t InMask = MVT::getIntVTBitMask(Op.getOperand(0).getValueType());
    if (SimplifyDemandedBits(Op.getOperand(0), DemandedMask & InMask,
                             KnownZero, KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    break;
  }
  case ISD::AssertZext: {
    MVT::ValueType VT = cast<VTSDNode>(Op.getOperand(1))->getVT();
    uint64_t InMask = MVT::getIntVTBitMask(VT);
    if (SimplifyDemandedBits(Op.getOperand(0), DemandedMask & InMask,
                             KnownZero, KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    KnownZero |= ~InMask & DemandedMask;
    break;
  }
  case ISD::ADD:
    if (ConstantSDNode *AA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      if (SimplifyDemandedBits(Op.getOperand(0), DemandedMask, KnownZero, 
                               KnownOne, TLO, Depth+1))
        return true;
      // Compute the KnownOne/KnownZero masks for the constant, so we can set
      // KnownZero appropriately if we're adding a constant that has all low
      // bits cleared.
      ComputeMaskedBits(Op.getOperand(1), 
                        MVT::getIntVTBitMask(Op.getValueType()), 
                        KnownZero2, KnownOne2, Depth+1);
      
      uint64_t KnownZeroOut = std::min(CountTrailingZeros_64(~KnownZero), 
                                       CountTrailingZeros_64(~KnownZero2));
      KnownZero = (1ULL << KnownZeroOut) - 1;
      KnownOne = 0;
      
      SDOperand SH = Op.getOperand(0);
      // fold (add (shl x, c1), (shl c2, c1)) -> (shl (add x, c2), c1)
      if (KnownZero && SH.getOpcode() == ISD::SHL && SH.Val->hasOneUse() &&
          Op.Val->hasOneUse()) {
        if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(SH.getOperand(1))) {
          MVT::ValueType VT = Op.getValueType();
          unsigned ShiftAmt = SA->getValue();
          uint64_t AddAmt = AA->getValue();
          uint64_t AddShr = AddAmt >> ShiftAmt;
          if (AddAmt == (AddShr << ShiftAmt)) {
            SDOperand ADD = TLO.DAG.getNode(ISD::ADD, VT, SH.getOperand(0),
                                            TLO.DAG.getConstant(AddShr, VT));
            SDOperand SHL = TLO.DAG.getNode(ISD::SHL, VT, ADD,SH.getOperand(1));
            return TLO.CombineTo(Op, SHL);
          }
        }
      }
    }
    break;
  case ISD::SUB:
    // Just use ComputeMaskedBits to compute output bits, there are no
    // simplifications that can be done here, and sub always demands all input
    // bits.
    ComputeMaskedBits(Op, DemandedMask, KnownZero, KnownOne, Depth);
    break;
  }
  
  // If we know the value of all of the demanded bits, return this as a
  // constant.
  if ((DemandedMask & (KnownZero|KnownOne)) == DemandedMask)
    return TLO.CombineTo(Op, TLO.DAG.getConstant(KnownOne, Op.getValueType()));
  
  return false;
}

/// MaskedValueIsZero - Return true if 'V & Mask' is known to be zero.  We use
/// this predicate to simplify operations downstream.  Mask is known to be zero
/// for bits that V cannot have.
bool TargetLowering::MaskedValueIsZero(SDOperand Op, uint64_t Mask, 
                                       unsigned Depth) const {
  uint64_t KnownZero, KnownOne;
  ComputeMaskedBits(Op, Mask, KnownZero, KnownOne, Depth);
  assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
  return (KnownZero & Mask) == Mask;
}

/// ComputeMaskedBits - Determine which of the bits specified in Mask are
/// known to be either zero or one and return them in the KnownZero/KnownOne
/// bitsets.  This code only analyzes bits in Mask, in order to short-circuit
/// processing.
void TargetLowering::ComputeMaskedBits(SDOperand Op, uint64_t Mask, 
                                       uint64_t &KnownZero, uint64_t &KnownOne,
                                       unsigned Depth) const {
  KnownZero = KnownOne = 0;   // Don't know anything.
  if (Depth == 6 || Mask == 0)
    return;  // Limit search depth.
  
  uint64_t KnownZero2, KnownOne2;

  switch (Op.getOpcode()) {
  case ISD::Constant:
    // We know all of the bits for a constant!
    KnownOne = cast<ConstantSDNode>(Op)->getValue() & Mask;
    KnownZero = ~KnownOne & Mask;
    return;
  case ISD::AND:
    // If either the LHS or the RHS are Zero, the result is zero.
    ComputeMaskedBits(Op.getOperand(1), Mask, KnownZero, KnownOne, Depth+1);
    Mask &= ~KnownZero;
    ComputeMaskedBits(Op.getOperand(0), Mask, KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 

    // Output known-1 bits are only known if set in both the LHS & RHS.
    KnownOne &= KnownOne2;
    // Output known-0 are known to be clear if zero in either the LHS | RHS.
    KnownZero |= KnownZero2;
    return;
  case ISD::OR:
    ComputeMaskedBits(Op.getOperand(1), Mask, KnownZero, KnownOne, Depth+1);
    Mask &= ~KnownOne;
    ComputeMaskedBits(Op.getOperand(0), Mask, KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // Output known-0 bits are only known if clear in both the LHS & RHS.
    KnownZero &= KnownZero2;
    // Output known-1 are known to be set if set in either the LHS | RHS.
    KnownOne |= KnownOne2;
    return;
  case ISD::XOR: {
    ComputeMaskedBits(Op.getOperand(1), Mask, KnownZero, KnownOne, Depth+1);
    ComputeMaskedBits(Op.getOperand(0), Mask, KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // Output known-0 bits are known if clear or set in both the LHS & RHS.
    uint64_t KnownZeroOut = (KnownZero & KnownZero2) | (KnownOne & KnownOne2);
    // Output known-1 are known to be set if set in only one of the LHS, RHS.
    KnownOne = (KnownZero & KnownOne2) | (KnownOne & KnownZero2);
    KnownZero = KnownZeroOut;
    return;
  }
  case ISD::SELECT:
    ComputeMaskedBits(Op.getOperand(2), Mask, KnownZero, KnownOne, Depth+1);
    ComputeMaskedBits(Op.getOperand(1), Mask, KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // Only known if known in both the LHS and RHS.
    KnownOne &= KnownOne2;
    KnownZero &= KnownZero2;
    return;
  case ISD::SELECT_CC:
    ComputeMaskedBits(Op.getOperand(3), Mask, KnownZero, KnownOne, Depth+1);
    ComputeMaskedBits(Op.getOperand(2), Mask, KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // Only known if known in both the LHS and RHS.
    KnownOne &= KnownOne2;
    KnownZero &= KnownZero2;
    return;
  case ISD::SETCC:
    // If we know the result of a setcc has the top bits zero, use this info.
    if (getSetCCResultContents() == TargetLowering::ZeroOrOneSetCCResult)
      KnownZero |= (MVT::getIntVTBitMask(Op.getValueType()) ^ 1ULL);
    return;
  case ISD::SHL:
    // (shl X, C1) & C2 == 0   iff   (X & C2 >>u C1) == 0
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      Mask >>= SA->getValue();
      ComputeMaskedBits(Op.getOperand(0), Mask, KnownZero, KnownOne, Depth+1);
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
      KnownZero <<= SA->getValue();
      KnownOne  <<= SA->getValue();
      KnownZero |= (1ULL << SA->getValue())-1;  // low bits known zero.
    }
    return;
  case ISD::SRL:
    // (ushr X, C1) & C2 == 0   iff  (-1 >> C1) & C2 == 0
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      uint64_t HighBits = (1ULL << SA->getValue())-1;
      HighBits <<= MVT::getSizeInBits(Op.getValueType())-SA->getValue();
      Mask <<= SA->getValue();
      ComputeMaskedBits(Op.getOperand(0), Mask, KnownZero, KnownOne, Depth+1);
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
      KnownZero >>= SA->getValue();
      KnownOne  >>= SA->getValue();
      KnownZero |= HighBits;  // high bits known zero.
    }
    return;
  case ISD::SRA:
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      uint64_t HighBits = (1ULL << SA->getValue())-1;
      HighBits <<= MVT::getSizeInBits(Op.getValueType())-SA->getValue();
      Mask <<= SA->getValue();
      ComputeMaskedBits(Op.getOperand(0), Mask, KnownZero, KnownOne, Depth+1);
      assert((KnownZero & KnownOne) == 0&&"Bits known to be one AND zero?"); 
      KnownZero >>= SA->getValue();
      KnownOne  >>= SA->getValue();
      
      // Handle the sign bits.
      uint64_t SignBit = 1ULL << (MVT::getSizeInBits(Op.getValueType())-1);
      SignBit >>= SA->getValue();  // Adjust to where it is now in the mask.
      
      if (KnownZero & SignBit) {       // New bits are known zero.
        KnownZero |= HighBits;
      } else if (KnownOne & SignBit) { // New bits are known one.
        KnownOne |= HighBits;
      }
    }
    return;
  case ISD::SIGN_EXTEND_INREG: {
    MVT::ValueType  VT = Op.getValueType();
    MVT::ValueType EVT = cast<VTSDNode>(Op.getOperand(1))->getVT();
    
    // Sign extension.  Compute the demanded bits in the result that are not 
    // present in the input.
    uint64_t NewBits = ~MVT::getIntVTBitMask(EVT) & Mask;

    uint64_t InSignBit = MVT::getIntVTSignBit(EVT);
    int64_t InputDemandedBits = Mask & MVT::getIntVTBitMask(EVT);
    
    // If the sign extended bits are demanded, we know that the sign
    // bit is demanded.
    if (NewBits)
      InputDemandedBits |= InSignBit;
    
    ComputeMaskedBits(Op.getOperand(0), InputDemandedBits,
                      KnownZero, KnownOne, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    
    // If the sign bit of the input is known set or clear, then we know the
    // top bits of the result.
    if (KnownZero & InSignBit) {          // Input sign bit known clear
      KnownZero |= NewBits;
      KnownOne  &= ~NewBits;
    } else if (KnownOne & InSignBit) {    // Input sign bit known set
      KnownOne  |= NewBits;
      KnownZero &= ~NewBits;
    } else {                              // Input sign bit unknown
      KnownZero &= ~NewBits;
      KnownOne  &= ~NewBits;
    }
    return;
  }
  case ISD::CTTZ:
  case ISD::CTLZ:
  case ISD::CTPOP: {
    MVT::ValueType VT = Op.getValueType();
    unsigned LowBits = Log2_32(MVT::getSizeInBits(VT))+1;
    KnownZero = ~((1ULL << LowBits)-1) & MVT::getIntVTBitMask(VT);
    KnownOne  = 0;
    return;
  }
  case ISD::ZEXTLOAD: {
    MVT::ValueType VT = cast<VTSDNode>(Op.getOperand(3))->getVT();
    KnownZero |= ~MVT::getIntVTBitMask(VT) & Mask;
    return;
  }
  case ISD::ZERO_EXTEND: {
    uint64_t InMask  = MVT::getIntVTBitMask(Op.getOperand(0).getValueType());
    uint64_t NewBits = (~InMask) & Mask;
    ComputeMaskedBits(Op.getOperand(0), Mask & InMask, KnownZero, 
                      KnownOne, Depth+1);
    KnownZero |= NewBits & Mask;
    KnownOne  &= ~NewBits;
    return;
  }
  case ISD::SIGN_EXTEND: {
    MVT::ValueType InVT = Op.getOperand(0).getValueType();
    unsigned InBits    = MVT::getSizeInBits(InVT);
    uint64_t InMask    = MVT::getIntVTBitMask(InVT);
    uint64_t InSignBit = 1ULL << (InBits-1);
    uint64_t NewBits   = (~InMask) & Mask;
    uint64_t InDemandedBits = Mask & InMask;

    // If any of the sign extended bits are demanded, we know that the sign
    // bit is demanded.
    if (NewBits & Mask)
      InDemandedBits |= InSignBit;
    
    ComputeMaskedBits(Op.getOperand(0), InDemandedBits, KnownZero, 
                      KnownOne, Depth+1);
    // If the sign bit is known zero or one, the  top bits match.
    if (KnownZero & InSignBit) {
      KnownZero |= NewBits;
      KnownOne  &= ~NewBits;
    } else if (KnownOne & InSignBit) {
      KnownOne  |= NewBits;
      KnownZero &= ~NewBits;
    } else {   // Otherwise, top bits aren't known.
      KnownOne  &= ~NewBits;
      KnownZero &= ~NewBits;
    }
    return;
  }
  case ISD::ANY_EXTEND: {
    MVT::ValueType VT = Op.getOperand(0).getValueType();
    ComputeMaskedBits(Op.getOperand(0), Mask & MVT::getIntVTBitMask(VT),
                      KnownZero, KnownOne, Depth+1);
    return;
  }
  case ISD::AssertZext: {
    MVT::ValueType VT = cast<VTSDNode>(Op.getOperand(1))->getVT();
    uint64_t InMask = MVT::getIntVTBitMask(VT);
    ComputeMaskedBits(Op.getOperand(0), Mask & InMask, KnownZero, 
                      KnownOne, Depth+1);
    KnownZero |= (~InMask) & Mask;
    return;
  }
  case ISD::ADD: {
    // If either the LHS or the RHS are Zero, the result is zero.
    ComputeMaskedBits(Op.getOperand(1), Mask, KnownZero, KnownOne, Depth+1);
    ComputeMaskedBits(Op.getOperand(0), Mask, KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // Output known-0 bits are known if clear or set in both the low clear bits
    // common to both LHS & RHS;
    uint64_t KnownZeroOut = std::min(CountTrailingZeros_64(~KnownZero), 
                                     CountTrailingZeros_64(~KnownZero2));
    
    KnownZero = (1ULL << KnownZeroOut) - 1;
    KnownOne = 0;
    return;
  }
  case ISD::SUB: {
    ConstantSDNode *CLHS = dyn_cast<ConstantSDNode>(Op.getOperand(0));
    if (!CLHS) return;

    // We know that the top bits of C-X are clear if X contains less bits
    // than C (i.e. no wrap-around can happen).  For example, 20-X is
    // positive if we can prove that X is >= 0 and < 16.
    MVT::ValueType VT = CLHS->getValueType(0);
    if ((CLHS->getValue() & MVT::getIntVTSignBit(VT)) == 0) {  // sign bit clear
      unsigned NLZ = CountLeadingZeros_64(CLHS->getValue()+1);
      uint64_t MaskV = (1ULL << (63-NLZ))-1; // NLZ can't be 64 with no sign bit
      MaskV = ~MaskV & MVT::getIntVTBitMask(VT);
      ComputeMaskedBits(Op.getOperand(1), MaskV, KnownZero, KnownOne, Depth+1);

      // If all of the MaskV bits are known to be zero, then we know the output
      // top bits are zero, because we now know that the output is from [0-C].
      if ((KnownZero & MaskV) == MaskV) {
        unsigned NLZ2 = CountLeadingZeros_64(CLHS->getValue());
        KnownZero = ~((1ULL << (64-NLZ2))-1) & Mask;  // Top bits known zero.
        KnownOne = 0;   // No one bits known.
      } else {
        KnownOne = KnownOne = 0;  // Otherwise, nothing known.
      }
    }
    return;
  }
  default:
    // Allow the target to implement this method for its nodes.
    if (Op.getOpcode() >= ISD::BUILTIN_OP_END)
      computeMaskedBitsForTargetNode(Op, Mask, KnownZero, KnownOne);
    return;
  }
}

/// computeMaskedBitsForTargetNode - Determine which of the bits specified 
/// in Mask are known to be either zero or one and return them in the 
/// KnownZero/KnownOne bitsets.
void TargetLowering::computeMaskedBitsForTargetNode(const SDOperand Op, 
                                                    uint64_t Mask,
                                                    uint64_t &KnownZero, 
                                                    uint64_t &KnownOne,
                                                    unsigned Depth) const {
  assert(Op.getOpcode() >= ISD::BUILTIN_OP_END &&
         "Should use MaskedValueIsZero if you don't know whether Op"
         " is a target node!");
  KnownZero = 0;
  KnownOne = 0;
}

//===----------------------------------------------------------------------===//
//  Inline Assembler Implementation Methods
//===----------------------------------------------------------------------===//

TargetLowering::ConstraintType
TargetLowering::getConstraintType(char ConstraintLetter) const {
  // FIXME: lots more standard ones to handle.
  switch (ConstraintLetter) {
  default: return C_Unknown;
  case 'r': return C_RegisterClass;
  case 'm':    // memory
  case 'o':    // offsetable
  case 'V':    // not offsetable
    return C_Memory;
  case 'i':    // Simple Integer or Relocatable Constant
  case 'n':    // Simple Integer
  case 's':    // Relocatable Constant
  case 'I':    // Target registers.
  case 'J':
  case 'K':
  case 'L':
  case 'M':
  case 'N':
  case 'O':
  case 'P':
    return C_Other;
  }
}

bool TargetLowering::isOperandValidForConstraint(SDOperand Op, 
                                                 char ConstraintLetter) {
  switch (ConstraintLetter) {
  default: return false;
  case 'i':    // Simple Integer or Relocatable Constant
  case 'n':    // Simple Integer
  case 's':    // Relocatable Constant
    return true;   // FIXME: not right.
  }
}


std::vector<unsigned> TargetLowering::
getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                  MVT::ValueType VT) const {
  return std::vector<unsigned>();
}


std::pair<unsigned, const TargetRegisterClass*> TargetLowering::
getRegForInlineAsmConstraint(const std::string &Constraint,
                             MVT::ValueType VT) const {
  if (Constraint[0] != '{')
    return std::pair<unsigned, const TargetRegisterClass*>(0, 0);
  assert(*(Constraint.end()-1) == '}' && "Not a brace enclosed constraint?");

  // Remove the braces from around the name.
  std::string RegName(Constraint.begin()+1, Constraint.end()-1);

  // Figure out which register class contains this reg.
  const MRegisterInfo *RI = TM.getRegisterInfo();
  for (MRegisterInfo::regclass_iterator RCI = RI->regclass_begin(),
       E = RI->regclass_end(); RCI != E; ++RCI) {
    const TargetRegisterClass *RC = *RCI;
    
    // If none of the the value types for this register class are valid, we 
    // can't use it.  For example, 64-bit reg classes on 32-bit targets.
    bool isLegal = false;
    for (TargetRegisterClass::vt_iterator I = RC->vt_begin(), E = RC->vt_end();
         I != E; ++I) {
      if (isTypeLegal(*I)) {
        isLegal = true;
        break;
      }
    }
    
    if (!isLegal) continue;
    
    for (TargetRegisterClass::iterator I = RC->begin(), E = RC->end(); 
         I != E; ++I) {
      if (StringsEqualNoCase(RegName, RI->get(*I).Name))
        return std::make_pair(*I, RC);
    }
  }
  
  return std::pair<unsigned, const TargetRegisterClass*>(0, 0);
}
