//===-- SystemZSelectionDAGInfo.cpp - SystemZ SelectionDAG Info -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SystemZSelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "systemz-selectiondag-info"
#include "SystemZTargetMachine.h"
#include "llvm/CodeGen/SelectionDAG.h"

using namespace llvm;

SystemZSelectionDAGInfo::
SystemZSelectionDAGInfo(const SystemZTargetMachine &TM)
  : TargetSelectionDAGInfo(TM) {
}

SystemZSelectionDAGInfo::~SystemZSelectionDAGInfo() {
}

SDValue SystemZSelectionDAGInfo::
EmitTargetCodeForMemcpy(SelectionDAG &DAG, SDLoc DL, SDValue Chain,
                        SDValue Dst, SDValue Src, SDValue Size, unsigned Align,
                        bool IsVolatile, bool AlwaysInline,
                        MachinePointerInfo DstPtrInfo,
                        MachinePointerInfo SrcPtrInfo) const {
  if (IsVolatile)
    return SDValue();

  if (ConstantSDNode *CSize = dyn_cast<ConstantSDNode>(Size)) {
    uint64_t Bytes = CSize->getZExtValue();
    if (Bytes >= 1 && Bytes <= 0x100) {
      // A single MVC.
      return DAG.getNode(SystemZISD::MVC, DL, MVT::Other,
                         Chain, Dst, Src, Size);
    }
  }
  return SDValue();
}
