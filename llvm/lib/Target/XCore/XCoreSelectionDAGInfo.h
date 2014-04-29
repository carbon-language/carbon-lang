//===-- XCoreSelectionDAGInfo.h - XCore SelectionDAG Info -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the XCore subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef XCORESELECTIONDAGINFO_H
#define XCORESELECTIONDAGINFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

class XCoreTargetMachine;

class XCoreSelectionDAGInfo : public TargetSelectionDAGInfo {
public:
  explicit XCoreSelectionDAGInfo(const XCoreTargetMachine &TM);
  ~XCoreSelectionDAGInfo();

  SDValue
  EmitTargetCodeForMemcpy(SelectionDAG &DAG, SDLoc dl,
                          SDValue Chain,
                          SDValue Op1, SDValue Op2,
                          SDValue Op3, unsigned Align, bool isVolatile,
                          bool AlwaysInline,
                          MachinePointerInfo DstPtrInfo,
                          MachinePointerInfo SrcPtrInfo) const override;
};

}

#endif
