//===-- AArch64SelectionDAGInfo.h - AArch64 SelectionDAG Info ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the AArch64 subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_AARCH64SELECTIONDAGINFO_H
#define LLVM_AARCH64SELECTIONDAGINFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

class AArch64TargetMachine;

class AArch64SelectionDAGInfo : public TargetSelectionDAGInfo {
  const AArch64Subtarget *Subtarget;
public:
  explicit AArch64SelectionDAGInfo(const AArch64TargetMachine &TM);
  ~AArch64SelectionDAGInfo();
};

}

#endif
