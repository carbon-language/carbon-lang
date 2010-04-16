//===-- MSP430SelectionDAGInfo.h - MSP430 SelectionDAG Info -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MSP430 subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef MSP430SELECTIONDAGINFO_H
#define MSP430SELECTIONDAGINFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

class MSP430SelectionDAGInfo : public TargetSelectionDAGInfo {
public:
  MSP430SelectionDAGInfo();
  ~MSP430SelectionDAGInfo();
};

}

#endif
