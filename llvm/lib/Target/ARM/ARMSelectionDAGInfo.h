//===-- ARMSelectionDAGInfo.h - ARM SelectionDAG Info -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ARM subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef ARMSELECTIONDAGINFO_H
#define ARMSELECTIONDAGINFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

class ARMSelectionDAGInfo : public TargetSelectionDAGInfo {
public:
  ARMSelectionDAGInfo();
  ~ARMSelectionDAGInfo();
};

}

#endif
