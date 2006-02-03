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
  maxStoresPerMemSet = maxStoresPerMemCpy = maxStoresPerMemMove = 8;
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

/// DemandedBitsAreZero - Return true if 'Op & Mask' demands no bits from a bit
/// set operation such as a sign extend or or/xor with constant whose only
/// use is Op.  If it returns true, the old node that sets bits which are
/// not demanded is returned in Old, and its replacement node is returned in
/// New, such that callers of DemandedBitsAreZero may call CombineTo on them if
/// desired.
bool TargetLowering::DemandedBitsAreZero(const SDOperand &Op, uint64_t Mask, 
                                         SDOperand &Old, SDOperand &New,
                                         SelectionDAG &DAG) {
  // If the operation has more than one use, we're not interested in it.
  // Tracking down and checking all uses would be problematic and slow.
  if (!Op.Val->hasOneUse())
    return false;
  
  switch (Op.getOpcode()) {
  case ISD::AND:
    // (X & C1) & C2 == 0   iff   C1 & C2 == 0.
    if (ConstantSDNode *AndRHS = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      uint64_t NewVal = Mask & AndRHS->getValue();
      return DemandedBitsAreZero(Op.getOperand(0), NewVal, Old, New, DAG);
    }
    break;
  case ISD::SHL:
    // (ushl X, C1) & C2 == 0   iff  X & (C2 >> C1) == 0
    if (ConstantSDNode *ShAmt = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      uint64_t NewVal = Mask >> ShAmt->getValue();
      return DemandedBitsAreZero(Op.getOperand(0), NewVal, Old, New, DAG);
    }
    break;
  case ISD::SIGN_EXTEND_INREG: {
    MVT::ValueType EVT = cast<VTSDNode>(Op.getOperand(1))->getVT();
    unsigned ExtendBits = MVT::getSizeInBits(EVT);
    // If we're extending from something smaller than MVT::i64 and all of the
    // sign extension bits are masked, return true and set New to be the
    // first operand, since we no longer care what the high bits are.
    if (ExtendBits < 64 && ((Mask & (~0ULL << ExtendBits)) == 0)) {
      Old = Op;
      New = Op.getOperand(0);
      return true;
    }
    break;
  }
  case ISD::SRA:
    if (ConstantSDNode *ShAmt = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      unsigned OpBits = MVT::getSizeInBits(Op.getValueType());
      unsigned SH = ShAmt->getValue();
      if (SH && ((Mask & (~0ULL << (OpBits-SH))) == 0)) {
        Old = Op;
        New = DAG.getNode(ISD::SRL, Op.getValueType(), Op.getOperand(0), 
                          Op.getOperand(1));
        return true;
      }
    }
    break;
  }
  return false;
}

/// MaskedValueIsZero - Return true if 'Op & Mask' is known to be zero.  We use
/// this predicate to simplify operations downstream.  Op and Mask are known to
/// be the same type.
bool TargetLowering::MaskedValueIsZero(const SDOperand &Op,
                                       uint64_t Mask) const {
  unsigned SrcBits;
  if (Mask == 0) return true;
  
  // If we know the result of a setcc has the top bits zero, use this info.
  switch (Op.getOpcode()) {
  case ISD::Constant:
    return (cast<ConstantSDNode>(Op)->getValue() & Mask) == 0;
  case ISD::SETCC:
    return ((Mask & 1) == 0) &&
      getSetCCResultContents() == TargetLowering::ZeroOrOneSetCCResult;
  case ISD::ZEXTLOAD:
    SrcBits = MVT::getSizeInBits(cast<VTSDNode>(Op.getOperand(3))->getVT());
    return (Mask & ((1ULL << SrcBits)-1)) == 0; // Returning only the zext bits.
  case ISD::ZERO_EXTEND:
    SrcBits = MVT::getSizeInBits(Op.getOperand(0).getValueType());
    return MaskedValueIsZero(Op.getOperand(0),Mask & (~0ULL >> (64-SrcBits)));
  case ISD::ANY_EXTEND:
    // If the mask only includes bits in the low part, recurse.
    SrcBits = MVT::getSizeInBits(Op.getOperand(0).getValueType());
    if (Mask >> SrcBits) return false;  // Use of unknown top bits.
    return MaskedValueIsZero(Op.getOperand(0), Mask);
  case ISD::AssertZext:
    SrcBits = MVT::getSizeInBits(cast<VTSDNode>(Op.getOperand(1))->getVT());
    return (Mask & ((1ULL << SrcBits)-1)) == 0; // Returning only the zext bits.
  case ISD::AND:
    // If either of the operands has zero bits, the result will too.
    if (MaskedValueIsZero(Op.getOperand(1), Mask) ||
        MaskedValueIsZero(Op.getOperand(0), Mask))
      return true;
    // (X & C1) & C2 == 0   iff   C1 & C2 == 0.
    if (ConstantSDNode *AndRHS = dyn_cast<ConstantSDNode>(Op.getOperand(1)))
      return MaskedValueIsZero(Op.getOperand(0),AndRHS->getValue() & Mask);
    return false;
  case ISD::OR:
  case ISD::XOR:
    return MaskedValueIsZero(Op.getOperand(0), Mask) &&
           MaskedValueIsZero(Op.getOperand(1), Mask);
  case ISD::SELECT:
    return MaskedValueIsZero(Op.getOperand(1), Mask) &&
           MaskedValueIsZero(Op.getOperand(2), Mask);
  case ISD::SELECT_CC:
    return MaskedValueIsZero(Op.getOperand(2), Mask) &&
           MaskedValueIsZero(Op.getOperand(3), Mask);
  case ISD::SRL:
    // (ushr X, C1) & C2 == 0   iff  X & (C2 << C1) == 0
    if (ConstantSDNode *ShAmt = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      uint64_t NewVal = Mask << ShAmt->getValue();
      SrcBits = MVT::getSizeInBits(Op.getValueType());
      if (SrcBits != 64) NewVal &= (1ULL << SrcBits)-1;
      return MaskedValueIsZero(Op.getOperand(0), NewVal);
    }
    return false;
  case ISD::SHL:
    // (ushl X, C1) & C2 == 0   iff  X & (C2 >> C1) == 0
    if (ConstantSDNode *ShAmt = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      uint64_t NewVal = Mask >> ShAmt->getValue();
      return MaskedValueIsZero(Op.getOperand(0), NewVal);
    }
    return false;
  case ISD::ADD:
    // (add X, Y) & C == 0 iff (X&C)|(Y&C) == 0 and all bits are low bits.
    if ((Mask&(Mask+1)) == 0) {  // All low bits
      if (MaskedValueIsZero(Op.getOperand(0), Mask) &&
          MaskedValueIsZero(Op.getOperand(1), Mask))
        return true;
    }
    break;
  case ISD::SUB:
    if (ConstantSDNode *CLHS = dyn_cast<ConstantSDNode>(Op.getOperand(0))) {
      // We know that the top bits of C-X are clear if X contains less bits
      // than C (i.e. no wrap-around can happen).  For example, 20-X is
      // positive if we can prove that X is >= 0 and < 16.
      unsigned Bits = MVT::getSizeInBits(CLHS->getValueType(0));
      if ((CLHS->getValue() & (1 << (Bits-1))) == 0) {  // sign bit clear
        unsigned NLZ = CountLeadingZeros_64(CLHS->getValue()+1);
        uint64_t MaskV = (1ULL << (63-NLZ))-1;
        if (MaskedValueIsZero(Op.getOperand(1), ~MaskV)) {
          // High bits are clear this value is known to be >= C.
          unsigned NLZ2 = CountLeadingZeros_64(CLHS->getValue());
          if ((Mask & ((1ULL << (64-NLZ2))-1)) == 0)
            return true;
        }
      }
    }
    break;
  case ISD::CTTZ:
  case ISD::CTLZ:
  case ISD::CTPOP:
    // Bit counting instructions can not set the high bits of the result
    // register.  The max number of bits sets depends on the input.
    return (Mask & (MVT::getSizeInBits(Op.getValueType())*2-1)) == 0;
  default:
    // Allow the target to implement this method for its nodes.
    if (Op.getOpcode() >= ISD::BUILTIN_OP_END)
      return isMaskedValueZeroForTargetNode(Op, Mask);
    break;
  }
  return false;
}

bool TargetLowering::isMaskedValueZeroForTargetNode(const SDOperand &Op,
                                                    uint64_t Mask) const {
  assert(Op.getOpcode() >= ISD::BUILTIN_OP_END &&
         "Should use MaskedValueIsZero if you don't know whether Op"
         " is a target node!");
  return false;
}

std::vector<unsigned> TargetLowering::
getRegForInlineAsmConstraint(const std::string &Constraint) const {
  // Not a physreg, must not be a register reference or something.
  if (Constraint[0] != '{') return std::vector<unsigned>();
  assert(*(Constraint.end()-1) == '}' && "Not a brace enclosed constraint?");

  // Remove the braces from around the name.
  std::string RegName(Constraint.begin()+1, Constraint.end()-1);
  
  // Scan to see if this constraint is a register name.
  const MRegisterInfo *RI = TM.getRegisterInfo();
  for (unsigned i = 1, e = RI->getNumRegs(); i != e; ++i) {
    if (const char *Name = RI->get(i).Name)
      if (StringsEqualNoCase(RegName, Name))
        return std::vector<unsigned>(1, i);
  }
  
  // Unknown physreg.
  return std::vector<unsigned>();
}

