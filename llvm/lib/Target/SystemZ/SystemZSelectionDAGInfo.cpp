//===-- SystemZSelectionDAGInfo.cpp - SystemZ SelectionDAG Info -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SystemZSelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "systemz-selectiondag-info"
#include "SystemZTargetMachine.h"
using namespace llvm;

SystemZSelectionDAGInfo::SystemZSelectionDAGInfo(const SystemZTargetMachine &TM)
  : TargetSelectionDAGInfo(TM) {
}

SystemZSelectionDAGInfo::~SystemZSelectionDAGInfo() {
}
