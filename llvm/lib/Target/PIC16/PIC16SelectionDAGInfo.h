//===-- PIC16SelectionDAGInfo.h - PIC16 SelectionDAG Info -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the PIC16 subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef PIC16SELECTIONDAGINFO_H
#define PIC16SELECTIONDAGINFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

class PIC16SelectionDAGInfo : public TargetSelectionDAGInfo {
public:
  PIC16SelectionDAGInfo();
  ~PIC16SelectionDAGInfo();
};

}

#endif
