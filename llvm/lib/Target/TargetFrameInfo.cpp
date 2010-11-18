//===-- TargetFrameInfo.cpp - Implement machine frame interface -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the layout of a stack frame on the target machine.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetFrameInfo.h"
#include <cstdlib>
using namespace llvm;

TargetFrameInfo::~TargetFrameInfo() {
}

/// getInitialFrameState - Returns a list of machine moves that are assumed
/// on entry to a function.
void
TargetFrameInfo::getInitialFrameState(std::vector<MachineMove> &Moves) const {
  // Default is to do nothing.
}
