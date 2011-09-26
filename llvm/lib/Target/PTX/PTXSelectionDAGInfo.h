//===-- PTXSelectionDAGInfo.h - PTX SelectionDAG Info -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the PTX subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef PTXSELECTIONDAGINFO_H
#define PTXSELECTIONDAGINFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

/// PTXSelectionDAGInfo - TargetSelectionDAGInfo sub-class for the PTX target.
/// At the moment, this is mostly just a copy of ARMSelectionDAGInfo.
class PTXSelectionDAGInfo : public TargetSelectionDAGInfo {
  /// Subtarget - Keep a pointer to the PTXSubtarget around so that we can
  /// make the right decision when generating code for different targets.
  const PTXSubtarget *Subtarget;

public:
  explicit PTXSelectionDAGInfo(const TargetMachine &TM);
  ~PTXSelectionDAGInfo();

  virtual
  SDValue EmitTargetCodeForMemcpy(SelectionDAG &DAG, DebugLoc dl,
                                  SDValue Chain,
                                  SDValue Dst, SDValue Src,
                                  SDValue Size, unsigned Align,
                                  bool isVolatile, bool AlwaysInline,
                                  MachinePointerInfo DstPtrInfo,
                                  MachinePointerInfo SrcPtrInfo) const;

  virtual
  SDValue EmitTargetCodeForMemset(SelectionDAG &DAG, DebugLoc dl,
                                  SDValue Chain,
                                  SDValue Op1, SDValue Op2,
                                  SDValue Op3, unsigned Align,
                                  bool isVolatile,
                                  MachinePointerInfo DstPtrInfo) const;
};

}

#endif

