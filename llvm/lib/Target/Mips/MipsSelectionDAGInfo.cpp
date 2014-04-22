//===-- MipsSelectionDAGInfo.cpp - Mips SelectionDAG Info -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MipsSelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#include "MipsTargetMachine.h"
using namespace llvm;

#define DEBUG_TYPE "mips-selectiondag-info"

MipsSelectionDAGInfo::MipsSelectionDAGInfo(const MipsTargetMachine &TM)
  : TargetSelectionDAGInfo(TM) {
}

MipsSelectionDAGInfo::~MipsSelectionDAGInfo() {
}
