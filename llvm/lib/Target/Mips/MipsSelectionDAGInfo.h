//===-- MipsSelectionDAGInfo.h - Mips SelectionDAG Info ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Mips subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MIPS_MIPSSELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_MIPS_MIPSSELECTIONDAGINFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

class MipsTargetMachine;

class MipsSelectionDAGInfo : public TargetSelectionDAGInfo {
public:
  explicit MipsSelectionDAGInfo(const DataLayout &DL);
  ~MipsSelectionDAGInfo();
};

}

#endif
