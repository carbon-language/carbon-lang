//===-- PPCSelectionDAGInfo.cpp - PowerPC SelectionDAG Info ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PPCSelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#include "PPCTargetMachine.h"
using namespace llvm;

#define DEBUG_TYPE "powerpc-selectiondag-info"

PPCSelectionDAGInfo::PPCSelectionDAGInfo(const DataLayout *DL)
    : TargetSelectionDAGInfo(DL) {}

PPCSelectionDAGInfo::~PPCSelectionDAGInfo() {}
