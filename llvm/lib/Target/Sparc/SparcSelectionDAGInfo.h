//===-- SparcSelectionDAGInfo.h - Sparc SelectionDAG Info -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Sparc subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPARC_SPARCSELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_SPARC_SPARCSELECTIONDAGINFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

class SparcTargetMachine;

class SparcSelectionDAGInfo : public TargetSelectionDAGInfo {
public:
  explicit SparcSelectionDAGInfo(const DataLayout &DL);
  ~SparcSelectionDAGInfo();
};

}

#endif
