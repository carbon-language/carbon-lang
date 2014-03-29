//===-- ARM64SelectionDAGInfo.h - ARM64 SelectionDAG Info -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ARM64 subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef ARM64SELECTIONDAGINFO_H
#define ARM64SELECTIONDAGINFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

class ARM64SelectionDAGInfo : public TargetSelectionDAGInfo {
  /// Subtarget - Keep a pointer to the ARMSubtarget around so that we can
  /// make the right decision when generating code for different targets.
  const ARM64Subtarget *Subtarget;

public:
  explicit ARM64SelectionDAGInfo(const TargetMachine &TM);
  ~ARM64SelectionDAGInfo();

  virtual SDValue EmitTargetCodeForMemset(SelectionDAG &DAG, SDLoc dl,
                                          SDValue Chain, SDValue Dst,
                                          SDValue Src, SDValue Size,
                                          unsigned Align, bool isVolatile,
                                          MachinePointerInfo DstPtrInfo) const;
};
}

#endif
