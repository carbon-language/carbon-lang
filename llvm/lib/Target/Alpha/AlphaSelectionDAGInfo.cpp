//===-- AlphaSelectionDAGInfo.cpp - Alpha SelectionDAG Info ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AlphaSelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "alpha-selectiondag-info"
#include "AlphaTargetMachine.h"
using namespace llvm;

AlphaSelectionDAGInfo::AlphaSelectionDAGInfo(const AlphaTargetMachine &TM)
  : TargetSelectionDAGInfo(TM) {
}

AlphaSelectionDAGInfo::~AlphaSelectionDAGInfo() {
}
