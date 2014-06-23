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
#include "llvm/CodeGen/SelectionDAG.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// TargetLowering Class Implementation Begins
//===----------------------------------------------------------------------===//
void AMDGPUTargetLowering::InitAMDILLowering() {
  setOperationAction(ISD::BRCOND, MVT::Other, Custom);
  setSelectIsExpensive(true); // FIXME: This makes no sense at all
}

SDValue AMDGPUTargetLowering::LowerBRCOND(SDValue Op, SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0);
  SDValue Cond  = Op.getOperand(1);
  SDValue Jump  = Op.getOperand(2);

  return DAG.getNode(AMDGPUISD::BRANCH_COND, SDLoc(Op), Op.getValueType(),
                     Chain, Jump, Cond);
}
