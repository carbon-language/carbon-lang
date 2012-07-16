//===-- AMDGPUISelLowering.cpp - AMDGPU Common DAG lowering functions -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the parent TargetLowering class for hardware code gen targets.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUISelLowering.h"
#include "AMDILIntrinsicInfo.h"
#include "AMDGPUUtil.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

AMDGPUTargetLowering::AMDGPUTargetLowering(TargetMachine &TM) :
  AMDILTargetLowering(TM)
{
  // We need to custom lower some of the intrinsics
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);

  setOperationAction(ISD::SELECT_CC, MVT::f32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::i32, Custom);

  // Library functions.  These default to Expand, but we have instructions
  // for them.
  setOperationAction(ISD::FCEIL,  MVT::f32, Legal);
  setOperationAction(ISD::FEXP2,  MVT::f32, Legal);
  setOperationAction(ISD::FRINT,  MVT::f32, Legal);

  setOperationAction(ISD::UDIV, MVT::i32, Expand);
  setOperationAction(ISD::UDIVREM, MVT::i32, Custom);
  setOperationAction(ISD::UREM, MVT::i32, Expand);
}

SDValue AMDGPUTargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG)
    const
{
  switch (Op.getOpcode()) {
  default: return AMDILTargetLowering::LowerOperation(Op, DAG);
  case ISD::INTRINSIC_WO_CHAIN: return LowerINTRINSIC_WO_CHAIN(Op, DAG);
  case ISD::SELECT_CC: return LowerSELECT_CC(Op, DAG);
  case ISD::UDIVREM: return LowerUDIVREM(Op, DAG);
  }
}

SDValue AMDGPUTargetLowering::LowerINTRINSIC_WO_CHAIN(SDValue Op,
    SelectionDAG &DAG) const
{
  unsigned IntrinsicID = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  DebugLoc DL = Op.getDebugLoc();
  EVT VT = Op.getValueType();

  switch (IntrinsicID) {
    default: return Op;
    case AMDGPUIntrinsic::AMDIL_abs:
      return LowerIntrinsicIABS(Op, DAG);
    case AMDGPUIntrinsic::AMDIL_exp:
      return DAG.getNode(ISD::FEXP2, DL, VT, Op.getOperand(1));
    case AMDGPUIntrinsic::AMDIL_fabs:
      return DAG.getNode(ISD::FABS, DL, VT, Op.getOperand(1));
    case AMDGPUIntrinsic::AMDGPU_lrp:
      return LowerIntrinsicLRP(Op, DAG);
    case AMDGPUIntrinsic::AMDIL_fraction:
      return DAG.getNode(AMDGPUISD::FRACT, DL, VT, Op.getOperand(1));
    case AMDGPUIntrinsic::AMDIL_mad:
      return DAG.getNode(AMDILISD::MAD, DL, VT, Op.getOperand(1),
                              Op.getOperand(2), Op.getOperand(3));
    case AMDGPUIntrinsic::AMDIL_max:
      return DAG.getNode(AMDGPUISD::FMAX, DL, VT, Op.getOperand(1),
                                                  Op.getOperand(2));
    case AMDGPUIntrinsic::AMDGPU_imax:
      return DAG.getNode(AMDGPUISD::SMAX, DL, VT, Op.getOperand(1),
                                                  Op.getOperand(2));
    case AMDGPUIntrinsic::AMDGPU_umax:
      return DAG.getNode(AMDGPUISD::UMAX, DL, VT, Op.getOperand(1),
                                                  Op.getOperand(2));
    case AMDGPUIntrinsic::AMDIL_min:
      return DAG.getNode(AMDGPUISD::FMIN, DL, VT, Op.getOperand(1),
                                                  Op.getOperand(2));
    case AMDGPUIntrinsic::AMDGPU_imin:
      return DAG.getNode(AMDGPUISD::SMIN, DL, VT, Op.getOperand(1),
                                                  Op.getOperand(2));
    case AMDGPUIntrinsic::AMDGPU_umin:
      return DAG.getNode(AMDGPUISD::UMIN, DL, VT, Op.getOperand(1),
                                                  Op.getOperand(2));
    case AMDGPUIntrinsic::AMDIL_round_nearest:
      return DAG.getNode(ISD::FRINT, DL, VT, Op.getOperand(1));
    case AMDGPUIntrinsic::AMDIL_round_posinf:
      return DAG.getNode(ISD::FCEIL, DL, VT, Op.getOperand(1));
  }
}

///IABS(a) = SMAX(sub(0, a), a)
SDValue AMDGPUTargetLowering::LowerIntrinsicIABS(SDValue Op,
    SelectionDAG &DAG) const
{

  DebugLoc DL = Op.getDebugLoc();
  EVT VT = Op.getValueType();
  SDValue Neg = DAG.getNode(ISD::SUB, DL, VT, DAG.getConstant(0, VT),
                                              Op.getOperand(1));

  return DAG.getNode(AMDGPUISD::SMAX, DL, VT, Neg, Op.getOperand(1));
}

/// Linear Interpolation
/// LRP(a, b, c) = muladd(a,  b, (1 - a) * c)
SDValue AMDGPUTargetLowering::LowerIntrinsicLRP(SDValue Op,
    SelectionDAG &DAG) const
{
  DebugLoc DL = Op.getDebugLoc();
  EVT VT = Op.getValueType();
  SDValue OneSubA = DAG.getNode(ISD::FSUB, DL, VT,
                                DAG.getConstantFP(1.0f, MVT::f32),
                                Op.getOperand(1));
  SDValue OneSubAC = DAG.getNode(ISD::FMUL, DL, VT, OneSubA,
                                                    Op.getOperand(3));
  return DAG.getNode(AMDILISD::MAD, DL, VT, Op.getOperand(1),
                                               Op.getOperand(2),
                                               OneSubAC);
}

SDValue AMDGPUTargetLowering::LowerSELECT_CC(SDValue Op,
    SelectionDAG &DAG) const
{
  DebugLoc DL = Op.getDebugLoc();
  EVT VT = Op.getValueType();

  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  SDValue True = Op.getOperand(2);
  SDValue False = Op.getOperand(3);
  SDValue CC = Op.getOperand(4);
  ISD::CondCode CCOpcode = cast<CondCodeSDNode>(CC)->get();
  SDValue Temp;

  // LHS and RHS are guaranteed to be the same value type
  EVT CompareVT = LHS.getValueType();

  // We need all the operands of SELECT_CC to have the same value type, so if
  // necessary we need to convert LHS and RHS to be the same type True and
  // False.  True and False are guaranteed to have the same type as this
  // SELECT_CC node.

  if (CompareVT !=  VT) {
    ISD::NodeType ConversionOp = ISD::DELETED_NODE;
    if (VT == MVT::f32 && CompareVT == MVT::i32) {
      if (isUnsignedIntSetCC(CCOpcode)) {
        ConversionOp = ISD::UINT_TO_FP;
      } else {
        ConversionOp = ISD::SINT_TO_FP;
      }
    } else if (VT == MVT::i32 && CompareVT == MVT::f32) {
      ConversionOp = ISD::FP_TO_SINT;
    } else {
      // I don't think there will be any other type pairings.
      assert(!"Unhandled operand type parings in SELECT_CC");
    }
    // XXX Check the value of LHS and RHS and avoid creating sequences like
    // (FTOI (ITOF))
    LHS = DAG.getNode(ConversionOp, DL, VT, LHS);
    RHS = DAG.getNode(ConversionOp, DL, VT, RHS);
  }

  // If True is a hardware TRUE value and False is a hardware FALSE value or
  // vice-versa we can handle this with a native instruction (SET* instructions).
  if ((isHWTrueValue(True) && isHWFalseValue(False))) {
    return DAG.getNode(ISD::SELECT_CC, DL, VT, LHS, RHS, True, False, CC);
  }

  // XXX If True is a hardware TRUE value and False is a hardware FALSE value,
  // we can handle this with a native instruction, but we need to swap true
  // and false and change the conditional.
  if (isHWTrueValue(False) && isHWFalseValue(True)) {
  }

  // XXX Check if we can lower this to a SELECT or if it is supported by a native
  // operation. (The code below does this but we don't have the Instruction
  // selection patterns to do this yet.
#if 0
  if (isZero(LHS) || isZero(RHS)) {
    SDValue Cond = (isZero(LHS) ? RHS : LHS);
    bool SwapTF = false;
    switch (CCOpcode) {
    case ISD::SETOEQ:
    case ISD::SETUEQ:
    case ISD::SETEQ:
      SwapTF = true;
      // Fall through
    case ISD::SETONE:
    case ISD::SETUNE:
    case ISD::SETNE:
      // We can lower to select
      if (SwapTF) {
        Temp = True;
        True = False;
        False = Temp;
      }
      // CNDE
      return DAG.getNode(ISD::SELECT, DL, VT, Cond, True, False);
    default:
      // Supported by a native operation (CNDGE, CNDGT)
      return DAG.getNode(ISD::SELECT_CC, DL, VT, LHS, RHS, True, False, CC);
    }
  }
#endif

  // If we make it this for it means we have no native instructions to handle
  // this SELECT_CC, so we must lower it.
  SDValue HWTrue, HWFalse;

  if (VT == MVT::f32) {
    HWTrue = DAG.getConstantFP(1.0f, VT);
    HWFalse = DAG.getConstantFP(0.0f, VT);
  } else if (VT == MVT::i32) {
    HWTrue = DAG.getConstant(-1, VT);
    HWFalse = DAG.getConstant(0, VT);
  }
  else {
    assert(!"Unhandled value type in LowerSELECT_CC");
  }

  // Lower this unsupported SELECT_CC into a combination of two supported
  // SELECT_CC operations.
  SDValue Cond = DAG.getNode(ISD::SELECT_CC, DL, VT, LHS, RHS, HWTrue, HWFalse, CC);

  return DAG.getNode(ISD::SELECT, DL, VT, Cond, True, False);
}


SDValue AMDGPUTargetLowering::LowerUDIVREM(SDValue Op,
    SelectionDAG &DAG) const
{
  DebugLoc DL = Op.getDebugLoc();
  EVT VT = Op.getValueType();

  SDValue Num = Op.getOperand(0);
  SDValue Den = Op.getOperand(1);

  SmallVector<SDValue, 8> Results;

  // RCP =  URECIP(Den) = 2^32 / Den + e
  // e is rounding error.
  SDValue RCP = DAG.getNode(AMDGPUISD::URECIP, DL, VT, Den);

  // RCP_LO = umulo(RCP, Den) */
  SDValue RCP_LO = DAG.getNode(ISD::UMULO, DL, VT, RCP, Den);

  // RCP_HI = mulhu (RCP, Den) */
  SDValue RCP_HI = DAG.getNode(ISD::MULHU, DL, VT, RCP, Den);

  // NEG_RCP_LO = -RCP_LO
  SDValue NEG_RCP_LO = DAG.getNode(ISD::SUB, DL, VT, DAG.getConstant(0, VT),
                                                     RCP_LO);

  // ABS_RCP_LO = (RCP_HI == 0 ? NEG_RCP_LO : RCP_LO)
  SDValue ABS_RCP_LO = DAG.getSelectCC(DL, RCP_HI, DAG.getConstant(0, VT),
                                           NEG_RCP_LO, RCP_LO,
                                           ISD::SETEQ);
  // Calculate the rounding error from the URECIP instruction
  // E = mulhu(ABS_RCP_LO, RCP)
  SDValue E = DAG.getNode(ISD::MULHU, DL, VT, ABS_RCP_LO, RCP);

  // RCP_A_E = RCP + E
  SDValue RCP_A_E = DAG.getNode(ISD::ADD, DL, VT, RCP, E);

  // RCP_S_E = RCP - E
  SDValue RCP_S_E = DAG.getNode(ISD::SUB, DL, VT, RCP, E);

  // Tmp0 = (RCP_HI == 0 ? RCP_A_E : RCP_SUB_E)
  SDValue Tmp0 = DAG.getSelectCC(DL, RCP_HI, DAG.getConstant(0, VT),
                                     RCP_A_E, RCP_S_E,
                                     ISD::SETEQ);
  // Quotient = mulhu(Tmp0, Num)
  SDValue Quotient = DAG.getNode(ISD::MULHU, DL, VT, Tmp0, Num);

  // Num_S_Remainder = Quotient * Den
  SDValue Num_S_Remainder = DAG.getNode(ISD::UMULO, DL, VT, Quotient, Den);

  // Remainder = Num - Num_S_Remainder
  SDValue Remainder = DAG.getNode(ISD::SUB, DL, VT, Num, Num_S_Remainder);

  // Remainder_GE_Den = (Remainder >= Den ? -1 : 0)
  SDValue Remainder_GE_Den = DAG.getSelectCC(DL, Remainder, Den,
                                                 DAG.getConstant(-1, VT),
                                                 DAG.getConstant(0, VT),
                                                 ISD::SETGE);
  // Remainder_GE_Zero = (Remainder >= 0 ? -1 : 0)
  SDValue Remainder_GE_Zero = DAG.getSelectCC(DL, Remainder,
                                                  DAG.getConstant(0, VT),
                                                  DAG.getConstant(-1, VT),
                                                  DAG.getConstant(0, VT),
                                                  ISD::SETGE);
  // Tmp1 = Remainder_GE_Den & Remainder_GE_Zero
  SDValue Tmp1 = DAG.getNode(ISD::AND, DL, VT, Remainder_GE_Den,
                                               Remainder_GE_Zero);

  // Calculate Division result:

  // Quotient_A_One = Quotient + 1
  SDValue Quotient_A_One = DAG.getNode(ISD::ADD, DL, VT, Quotient,
                                                         DAG.getConstant(1, VT));

  // Quotient_S_One = Quotient - 1
  SDValue Quotient_S_One = DAG.getNode(ISD::SUB, DL, VT, Quotient,
                                                         DAG.getConstant(1, VT));

  // Div = (Tmp1 == 0 ? Quotient : Quotient_A_One)
  SDValue Div = DAG.getSelectCC(DL, Tmp1, DAG.getConstant(0, VT),
                                     Quotient, Quotient_A_One, ISD::SETEQ);

  // Div = (Remainder_GE_Zero == 0 ? Quotient_S_One : Div)
  Div = DAG.getSelectCC(DL, Remainder_GE_Zero, DAG.getConstant(0, VT),
                            Quotient_S_One, Div, ISD::SETEQ);

  // Calculate Rem result:

  // Remainder_S_Den = Remainder - Den
  SDValue Remainder_S_Den = DAG.getNode(ISD::SUB, DL, VT, Remainder, Den);

  // Remainder_A_Den = Remainder + Den
  SDValue Remainder_A_Den = DAG.getNode(ISD::ADD, DL, VT, Remainder, Den);

  // Rem = (Tmp1 == 0 ? Remainder : Remainder_S_Den)
  SDValue Rem = DAG.getSelectCC(DL, Tmp1, DAG.getConstant(0, VT),
                                    Remainder, Remainder_S_Den, ISD::SETEQ);

  // Rem = (Remainder_GE_Zero == 0 ? Remainder_A_Den : Rem)
  Rem = DAG.getSelectCC(DL, Remainder_GE_Zero, DAG.getConstant(0, VT),
                            Remainder_A_Den, Rem, ISD::SETEQ);

  DAG.ReplaceAllUsesWith(Op.getValue(0).getNode(), &Div);
  DAG.ReplaceAllUsesWith(Op.getValue(1).getNode(), &Rem);

  return Op;
}

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

bool AMDGPUTargetLowering::isHWTrueValue(SDValue Op) const
{
  if (ConstantFPSDNode * CFP = dyn_cast<ConstantFPSDNode>(Op)) {
    return CFP->isExactlyValue(1.0);
  }
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op)) {
    return C->isAllOnesValue();
  }
  return false;
}

bool AMDGPUTargetLowering::isHWFalseValue(SDValue Op) const
{
  if (ConstantFPSDNode * CFP = dyn_cast<ConstantFPSDNode>(Op)) {
    return CFP->getValueAPF().isZero();
  }
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op)) {
    return C->isNullValue();
  }
  return false;
}

void AMDGPUTargetLowering::addLiveIn(MachineInstr * MI,
    MachineFunction * MF, MachineRegisterInfo & MRI,
    const TargetInstrInfo * TII, unsigned reg) const
{
  AMDGPU::utilAddLiveIn(MF, MRI, TII, reg, MI->getOperand(0).getReg());
}

#define NODE_NAME_CASE(node) case AMDGPUISD::node: return #node;

const char* AMDGPUTargetLowering::getTargetNodeName(unsigned Opcode) const
{
  switch (Opcode) {
  default: return AMDILTargetLowering::getTargetNodeName(Opcode);

  NODE_NAME_CASE(FRACT)
  NODE_NAME_CASE(FMAX)
  NODE_NAME_CASE(SMAX)
  NODE_NAME_CASE(UMAX)
  NODE_NAME_CASE(FMIN)
  NODE_NAME_CASE(SMIN)
  NODE_NAME_CASE(UMIN)
  NODE_NAME_CASE(URECIP)
  }
}
