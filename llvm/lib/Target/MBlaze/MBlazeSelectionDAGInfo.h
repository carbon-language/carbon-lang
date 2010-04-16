//===-- MBlazeSelectionDAGInfo.h - MBlaze SelectionDAG Info -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MBlaze subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef MBLAZESELECTIONDAGINFO_H
#define MBLAZESELECTIONDAGINFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

class MBlazeSelectionDAGInfo : public TargetSelectionDAGInfo {
public:
  MBlazeSelectionDAGInfo();
  ~MBlazeSelectionDAGInfo();
};

}

#endif
