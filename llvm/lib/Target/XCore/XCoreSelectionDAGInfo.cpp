//===-- XCoreSelectionDAGInfo.cpp - XCore SelectionDAG Info ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the XCoreSelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#include "XCoreTargetMachine.h"
using namespace llvm;

#define DEBUG_TYPE "xcore-selectiondag-info"

XCoreSelectionDAGInfo::XCoreSelectionDAGInfo(const XCoreTargetMachine &TM)
  : TargetSelectionDAGInfo(TM) {
}

XCoreSelectionDAGInfo::~XCoreSelectionDAGInfo() {
}

SDValue XCoreSelectionDAGInfo::
EmitTargetCodeForMemcpy(SelectionDAG &DAG, SDLoc dl, SDValue Chain,
                        SDValue Dst, SDValue Src, SDValue Size, unsigned Align,
                        bool isVolatile, bool AlwaysInline,
                        MachinePointerInfo DstPtrInfo,
                        MachinePointerInfo SrcPtrInfo) const
{
  unsigned SizeBitWidth = Size.getValueType().getSizeInBits();
  // Call __memcpy_4 if the src, dst and size are all 4 byte aligned.
  if (!AlwaysInline && (Align & 3) == 0 &&
      DAG.MaskedValueIsZero(Size, APInt(SizeBitWidth, 3))) {
    const TargetLowering &TLI = *DAG.getTarget().getTargetLowering();
    TargetLowering::ArgListTy Args;
    TargetLowering::ArgListEntry Entry;
    Entry.Ty = TLI.getDataLayout()->getIntPtrType(*DAG.getContext());
    Entry.Node = Dst; Args.push_back(Entry);
    Entry.Node = Src; Args.push_back(Entry);
    Entry.Node = Size; Args.push_back(Entry);

    TargetLowering::CallLoweringInfo
    CLI(Chain, Type::getVoidTy(*DAG.getContext()), false, false, false, false,
        0, TLI.getLibcallCallingConv(RTLIB::MEMCPY), /*isTailCall=*/false,
        /*doesNotRet=*/false, /*isReturnValueUsed=*/false,
        DAG.getExternalSymbol("__memcpy_4", TLI.getPointerTy()), Args, DAG, dl);
    std::pair<SDValue,SDValue> CallResult =
      TLI.LowerCallTo(CLI);
    return CallResult.second;
  }

  // Otherwise have the target-independent code call memcpy.
  return SDValue();
}
