//===-- AArch64SelectionDAGInfo.cpp - AArch64 SelectionDAG Info -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AArch64SelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "arm-selectiondag-info"
#include "AArch64TargetMachine.h"
#include "llvm/CodeGen/SelectionDAG.h"
using namespace llvm;

AArch64SelectionDAGInfo::AArch64SelectionDAGInfo(const AArch64TargetMachine &TM)
  : TargetSelectionDAGInfo(TM),
    Subtarget(&TM.getSubtarget<AArch64Subtarget>()) {
}

AArch64SelectionDAGInfo::~AArch64SelectionDAGInfo() {
}
