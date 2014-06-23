//===-- AMDILISelLowering.cpp - AMDIL DAG Lowering Implementation ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
/// \file
/// \brief TargetLowering functions borrowed from AMDIL.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUISelLowering.h"
#include "AMDGPUSubtarget.h"
#include "llvm/CodeGen/SelectionDAG.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// TargetLowering Class Implementation Begins
//===----------------------------------------------------------------------===//
void AMDGPUTargetLowering::InitAMDILLowering() {
  static const MVT::SimpleValueType types[] = {
    MVT::i32,
    MVT::f32,
    MVT::f64,
    MVT::i64,
    MVT::v4f32,
    MVT::v4i32,
    MVT::v2f32,
    MVT::v2i32
  };

  static const MVT::SimpleValueType VectorTypes[] = {
    MVT::v4f32,
    MVT::v4i32,
    MVT::v2f32,
    MVT::v2i32
  };

  const AMDGPUSubtarget &STM = getTargetMachine().getSubtarget<AMDGPUSubtarget>();

  for (MVT VT : types) {
    setOperationAction(ISD::SUBE, VT, Expand);
    setOperationAction(ISD::SUBC, VT, Expand);
    setOperationAction(ISD::ADDE, VT, Expand);
    setOperationAction(ISD::ADDC, VT, Expand);
    setOperationAction(ISD::BRCOND, VT, Custom);
    setOperationAction(ISD::BR_JT, VT, Expand);
    setOperationAction(ISD::BRIND, VT, Expand);
  }

  for (MVT VT : VectorTypes) {
    setOperationAction(ISD::VECTOR_SHUFFLE, VT, Expand);
    setOperationAction(ISD::SELECT_CC, VT, Expand);
  }

  if (STM.hasHWFP64()) {
    setOperationAction(ISD::ConstantFP, MVT::f64, Legal);
    setOperationAction(ISD::FABS, MVT::f64, Expand);
  }

  setOperationAction(ISD::SUBC, MVT::Other, Expand);
  setOperationAction(ISD::ADDE, MVT::Other, Expand);
  setOperationAction(ISD::ADDC, MVT::Other, Expand);
  setOperationAction(ISD::BRCOND, MVT::Other, Custom);
  setOperationAction(ISD::BR_JT, MVT::Other, Expand);
  setOperationAction(ISD::BRIND, MVT::Other, Expand);

  setOperationAction(ISD::Constant, MVT::i32, Legal);
  setOperationAction(ISD::Constant, MVT::i64, Legal);
  setOperationAction(ISD::ConstantFP, MVT::f32, Legal);

  setPow2DivIsCheap(false);
  setSelectIsExpensive(true); // FIXME: This makes no sense at all
}

SDValue AMDGPUTargetLowering::LowerBRCOND(SDValue Op, SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0);
  SDValue Cond  = Op.getOperand(1);
  SDValue Jump  = Op.getOperand(2);

  return DAG.getNode(AMDGPUISD::BRANCH_COND, SDLoc(Op), Op.getValueType(),
                     Chain, Jump, Cond);
}
