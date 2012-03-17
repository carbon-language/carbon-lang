//===-- PTXRegisterInfo.cpp - PTX Register Information --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PTX implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "PTXRegisterInfo.h"
#include "PTX.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define GET_REGINFO_TARGET_DESC
#include "PTXGenRegisterInfo.inc"

using namespace llvm;

PTXRegisterInfo::PTXRegisterInfo(PTXTargetMachine &TM,
                                 const TargetInstrInfo &tii)
  // PTX does not have a return address register.
  : PTXGenRegisterInfo(0), TII(tii) {
}

void PTXRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator /*II*/,
                                          int /*SPAdj*/,
                                          RegScavenger * /*RS*/) const {
  llvm_unreachable("FrameIndex should have been previously eliminated!");
}
