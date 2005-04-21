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
#include "llvm/CodeGen/SelectionDAG.h"
using namespace llvm;

TargetLowering::TargetLowering(TargetMachine &tm)
  : TM(tm), TD(TM.getTargetData()), ValueTypeActions(0) {
  assert(ISD::BUILTIN_OP_END <= 128 &&
         "Fixed size array in TargetLowering is not large enough!");
  // All operations default to being supported.
  memset(OpActions, 0, sizeof(OpActions));

  IsLittleEndian = TD.isLittleEndian();
  ShiftAmountTy = SetCCResultTy = PointerTy = getValueType(TD.getIntPtrType());
  ShiftAmtHandling = Undefined;
  memset(RegClassForVT, 0,MVT::LAST_VALUETYPE*sizeof(TargetRegisterClass*));
}

TargetLowering::~TargetLowering() {}

/// setValueTypeAction - Set the action for a particular value type.  This
/// assumes an action has not already been set for this value type.
static void SetValueTypeAction(MVT::ValueType VT,
                               TargetLowering::LegalizeAction Action,
                               TargetLowering &TLI,
                               MVT::ValueType *TransformToType,
                               unsigned &ValueTypeActions) {
  ValueTypeActions |= Action << (VT*2);
  if (Action == TargetLowering::Promote) {
    MVT::ValueType PromoteTo;
    if (VT == MVT::f32)
      PromoteTo = MVT::f64;
    else {
      unsigned LargerReg = VT+1;
      while (!TLI.hasNativeSupportFor((MVT::ValueType)LargerReg)) {
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
    assert(MVT::isInteger(VT) && VT > MVT::i8 &&
           "Cannot expand this type: target must support SOME integer reg!");
    // Expand to the next smaller integer type!
    TransformToType[VT] = (MVT::ValueType)(VT-1);
  }
}


/// computeRegisterProperties - Once all of the register classes are added,
/// this allows us to compute derived properties we expose.
void TargetLowering::computeRegisterProperties() {
  assert(MVT::LAST_VALUETYPE <= 16 &&
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
    else if (!hasNativeSupportFor((MVT::ValueType)IntReg))
      // Otherwise, if we don't have native support, we must promote to a
      // larger type.
      SetValueTypeAction((MVT::ValueType)IntReg, Promote, *this,
                         TransformToType, ValueTypeActions);
    else
      TransformToType[(MVT::ValueType)IntReg] = (MVT::ValueType)IntReg;

  // If the target does not have native support for F32, promote it to F64.
  if (!hasNativeSupportFor(MVT::f32))
    SetValueTypeAction(MVT::f32, Promote, *this,
                       TransformToType, ValueTypeActions);
  else
    TransformToType[MVT::f32] = MVT::f32;

  assert(hasNativeSupportFor(MVT::f64) && "Target does not support FP?");
  TransformToType[MVT::f64] = MVT::f64;
}

