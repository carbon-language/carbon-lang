//=======- PTXFrameInfo.cpp - PTX Frame Information -----------*- C++ -*-=====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PTX implementation of TargetFrameInfo class.
//
//===----------------------------------------------------------------------===//

#include "PTXFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"

using namespace llvm;

void PTXFrameInfo::emitPrologue(MachineFunction &MF) const {
}

void PTXFrameInfo::emitEpilogue(MachineFunction &MF,
                                MachineBasicBlock &MBB) const {
}
