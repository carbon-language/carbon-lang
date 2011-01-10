//=======- PTXFrameLowering.cpp - PTX Frame Information -------*- C++ -*-=====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PTX implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "PTXFrameLowering.h"
#include "llvm/CodeGen/MachineFunction.h"

using namespace llvm;

void PTXFrameLowering::emitPrologue(MachineFunction &MF) const {
}

void PTXFrameLowering::emitEpilogue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {
}
