//===-- PTXSelectionDAGInfo.cpp - PTX SelectionDAG Info -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PTXSelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ptx-selectiondag-info"
#include "PTXTargetMachine.h"
#include "llvm/DerivedTypes.h"
#include "llvm/CodeGen/SelectionDAG.h"
using namespace llvm;

PTXSelectionDAGInfo::PTXSelectionDAGInfo(const TargetMachine &TM)
  : TargetSelectionDAGInfo(TM),
    Subtarget(&TM.getSubtarget<PTXSubtarget>()) {
}

PTXSelectionDAGInfo::~PTXSelectionDAGInfo() {
}

SDValue
PTXSelectionDAGInfo::EmitTargetCodeForMemcpy(SelectionDAG &DAG, DebugLoc dl,
                                             SDValue Chain,
                                             SDValue Dst, SDValue Src,
                                             SDValue Size, unsigned Align,
                                             bool isVolatile, bool AlwaysInline,
                                             MachinePointerInfo DstPtrInfo,
                                          MachinePointerInfo SrcPtrInfo) const {
  // Do repeated 4-byte loads and stores. To be improved.
  // This requires 4-byte alignment.
  if ((Align & 3) != 0)
    return SDValue();
  // This requires the copy size to be a constant, preferably
  // within a subtarget-specific limit.
  ConstantSDNode *ConstantSize = dyn_cast<ConstantSDNode>(Size);
  if (!ConstantSize)
    return SDValue();
  uint64_t SizeVal = ConstantSize->getZExtValue();
  // Always inline memcpys. In PTX, we do not have a C library that provides
  // a memcpy function.
  //if (!AlwaysInline)
  //  return SDValue();

  unsigned BytesLeft = SizeVal & 3;
  unsigned NumMemOps = SizeVal >> 2;
  unsigned EmittedNumMemOps = 0;
  EVT VT = MVT::i32;
  unsigned VTSize = 4;
  unsigned i = 0;
  const unsigned MAX_LOADS_IN_LDM = 6;
  SDValue TFOps[MAX_LOADS_IN_LDM];
  SDValue Loads[MAX_LOADS_IN_LDM];
  uint64_t SrcOff = 0, DstOff = 0;
  EVT PointerType = Subtarget->is64Bit() ? MVT::i64 : MVT::i32;

  // Emit up to MAX_LOADS_IN_LDM loads, then a TokenFactor barrier, then the
  // same number of stores.  The loads and stores will get combined into
  // ldm/stm later on.
  while (EmittedNumMemOps < NumMemOps) {
    for (i = 0;
         i < MAX_LOADS_IN_LDM && EmittedNumMemOps + i < NumMemOps; ++i) {
      Loads[i] = DAG.getLoad(VT, dl, Chain,
                             DAG.getNode(ISD::ADD, dl, PointerType, Src,
                                         DAG.getConstant(SrcOff, PointerType)),
                             SrcPtrInfo.getWithOffset(SrcOff), isVolatile,
                             false, 0);
      TFOps[i] = Loads[i].getValue(1);
      SrcOff += VTSize;
    }
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, &TFOps[0], i);

    for (i = 0;
         i < MAX_LOADS_IN_LDM && EmittedNumMemOps + i < NumMemOps; ++i) {
      TFOps[i] = DAG.getStore(Chain, dl, Loads[i],
                              DAG.getNode(ISD::ADD, dl, PointerType, Dst,
                                          DAG.getConstant(DstOff, PointerType)),
                              DstPtrInfo.getWithOffset(DstOff),
                              isVolatile, false, 0);
      DstOff += VTSize;
    }
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, &TFOps[0], i);

    EmittedNumMemOps += i;
  }

  if (BytesLeft == 0)
    return Chain;

  // Issue loads / stores for the trailing (1 - 3) bytes.
  unsigned BytesLeftSave = BytesLeft;
  i = 0;
  while (BytesLeft) {
    if (BytesLeft >= 2) {
      VT = MVT::i16;
      VTSize = 2;
    } else {
      VT = MVT::i8;
      VTSize = 1;
    }

    Loads[i] = DAG.getLoad(VT, dl, Chain,
                           DAG.getNode(ISD::ADD, dl, PointerType, Src,
                                       DAG.getConstant(SrcOff, PointerType)),
                           SrcPtrInfo.getWithOffset(SrcOff), false, false, 0);
    TFOps[i] = Loads[i].getValue(1);
    ++i;
    SrcOff += VTSize;
    BytesLeft -= VTSize;
  }
  Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, &TFOps[0], i);

  i = 0;
  BytesLeft = BytesLeftSave;
  while (BytesLeft) {
    if (BytesLeft >= 2) {
      VT = MVT::i16;
      VTSize = 2;
    } else {
      VT = MVT::i8;
      VTSize = 1;
    }

    TFOps[i] = DAG.getStore(Chain, dl, Loads[i],
                            DAG.getNode(ISD::ADD, dl, PointerType, Dst,
                                        DAG.getConstant(DstOff, PointerType)),
                            DstPtrInfo.getWithOffset(DstOff), false, false, 0);
    ++i;
    DstOff += VTSize;
    BytesLeft -= VTSize;
  }
  return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, &TFOps[0], i);
}

SDValue PTXSelectionDAGInfo::
EmitTargetCodeForMemset(SelectionDAG &DAG, DebugLoc dl,
                        SDValue Chain, SDValue Dst,
                        SDValue Src, SDValue Size,
                        unsigned Align, bool isVolatile,
                        MachinePointerInfo DstPtrInfo) const {
  llvm_unreachable("memset lowering not implemented for PTX yet");
}

