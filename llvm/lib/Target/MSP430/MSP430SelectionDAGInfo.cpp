//===-- MSP430SelectionDAGInfo.cpp - MSP430 SelectionDAG Info -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MSP430SelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#include "MSP430TargetMachine.h"
using namespace llvm;

#define DEBUG_TYPE "msp430-selectiondag-info"

MSP430SelectionDAGInfo::MSP430SelectionDAGInfo(const DataLayout &DL)
    : TargetSelectionDAGInfo(&DL) {}

MSP430SelectionDAGInfo::~MSP430SelectionDAGInfo() {
}
