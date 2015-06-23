//===-- PPCSelectionDAGInfo.h - PowerPC SelectionDAG Info -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the PowerPC subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_POWERPC_PPCSELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_POWERPC_PPCSELECTIONDAGINFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

class PPCTargetMachine;

class PPCSelectionDAGInfo : public TargetSelectionDAGInfo {
public:
  explicit PPCSelectionDAGInfo(const DataLayout *DL);
  ~PPCSelectionDAGInfo();
};

}

#endif
