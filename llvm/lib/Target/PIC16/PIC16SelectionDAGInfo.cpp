//===-- PIC16SelectionDAGInfo.cpp - PIC16 SelectionDAG Info ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PIC16SelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "pic16-selectiondag-info"
#include "PIC16TargetMachine.h"
using namespace llvm;

PIC16SelectionDAGInfo::PIC16SelectionDAGInfo(const PIC16TargetMachine &TM)
  : TargetSelectionDAGInfo(TM) {
}

PIC16SelectionDAGInfo::~PIC16SelectionDAGInfo() {
}
