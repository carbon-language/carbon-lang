//===-- AlphaSelectionDAGInfo.h - Alpha SelectionDAG Info -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Alpha subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef ALPHASELECTIONDAGINFO_H
#define ALPHASELECTIONDAGINFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

class AlphaSelectionDAGInfo : public TargetSelectionDAGInfo {
public:
  AlphaSelectionDAGInfo();
  ~AlphaSelectionDAGInfo();
};

}

#endif
