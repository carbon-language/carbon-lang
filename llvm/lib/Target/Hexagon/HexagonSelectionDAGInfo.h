//===-- HexagonSelectionDAGInfo.h - Hexagon SelectionDAG Info ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Hexagon subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_HEXAGONSELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_HEXAGON_HEXAGONSELECTIONDAGINFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

class HexagonSelectionDAGInfo : public TargetSelectionDAGInfo {
public:
  explicit HexagonSelectionDAGInfo(const DataLayout &DL);
  ~HexagonSelectionDAGInfo();

  SDValue EmitTargetCodeForMemcpy(SelectionDAG &DAG, SDLoc dl,
                                  SDValue Chain,
                                  SDValue Dst, SDValue Src,
                                  SDValue Size, unsigned Align,
                                  bool isVolatile, bool AlwaysInline,
                                  MachinePointerInfo DstPtrInfo,
                                  MachinePointerInfo SrcPtrInfo) const override;
};

}

#endif
