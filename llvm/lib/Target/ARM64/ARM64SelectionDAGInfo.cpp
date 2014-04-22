//===-- ARM64SelectionDAGInfo.cpp - ARM64 SelectionDAG Info ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ARM64SelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARM64TargetMachine.h"
using namespace llvm;

#define DEBUG_TYPE "arm64-selectiondag-info"

ARM64SelectionDAGInfo::ARM64SelectionDAGInfo(const TargetMachine &TM)
    : TargetSelectionDAGInfo(TM),
      Subtarget(&TM.getSubtarget<ARM64Subtarget>()) {}

ARM64SelectionDAGInfo::~ARM64SelectionDAGInfo() {}

SDValue ARM64SelectionDAGInfo::EmitTargetCodeForMemset(
    SelectionDAG &DAG, SDLoc dl, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, unsigned Align, bool isVolatile,
    MachinePointerInfo DstPtrInfo) const {
  // Check to see if there is a specialized entry-point for memory zeroing.
  ConstantSDNode *V = dyn_cast<ConstantSDNode>(Src);
  ConstantSDNode *SizeValue = dyn_cast<ConstantSDNode>(Size);
  const char *bzeroEntry =
      (V && V->isNullValue()) ? Subtarget->getBZeroEntry() : 0;
  // For small size (< 256), it is not beneficial to use bzero
  // instead of memset.
  if (bzeroEntry && (!SizeValue || SizeValue->getZExtValue() > 256)) {
    const ARM64TargetLowering &TLI = *static_cast<const ARM64TargetLowering *>(
                                          DAG.getTarget().getTargetLowering());

    EVT IntPtr = TLI.getPointerTy();
    Type *IntPtrTy = getDataLayout()->getIntPtrType(*DAG.getContext());
    TargetLowering::ArgListTy Args;
    TargetLowering::ArgListEntry Entry;
    Entry.Node = Dst;
    Entry.Ty = IntPtrTy;
    Args.push_back(Entry);
    Entry.Node = Size;
    Args.push_back(Entry);
    TargetLowering::CallLoweringInfo CLI(
        Chain, Type::getVoidTy(*DAG.getContext()), false, false, false, false,
        0, CallingConv::C, /*isTailCall=*/false,
        /*doesNotRet=*/false, /*isReturnValueUsed=*/false,
        DAG.getExternalSymbol(bzeroEntry, IntPtr), Args, DAG, dl);
    std::pair<SDValue, SDValue> CallResult = TLI.LowerCallTo(CLI);
    return CallResult.second;
  }
  return SDValue();
}
