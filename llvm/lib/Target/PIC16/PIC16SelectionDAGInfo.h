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

class PIC16TargetMachine;

class PIC16SelectionDAGInfo : public TargetSelectionDAGInfo {
public:
  explicit PIC16SelectionDAGInfo(const PIC16TargetMachine &TM);
  ~PIC16SelectionDAGInfo();
};

}

#endif
