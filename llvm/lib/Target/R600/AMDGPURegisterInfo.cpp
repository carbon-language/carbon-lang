//===-- AMDGPURegisterInfo.cpp - AMDGPU Register Information -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Parent TargetRegisterInfo class common to all hw codegen targets.
//
//===----------------------------------------------------------------------===//

#include "AMDGPURegisterInfo.h"
#include "AMDGPUTargetMachine.h"

using namespace llvm;

AMDGPURegisterInfo::AMDGPURegisterInfo(TargetMachine &tm,
    const TargetInstrInfo &tii)
: AMDGPUGenRegisterInfo(0),
  TM(tm),
  TII(tii)
  { }

//===----------------------------------------------------------------------===//
// Function handling callbacks - Functions are a seldom used feature of GPUS, so
// they are not supported at this time.
//===----------------------------------------------------------------------===//

const uint16_t AMDGPURegisterInfo::CalleeSavedReg = AMDGPU::NoRegister;

const uint16_t* AMDGPURegisterInfo::getCalleeSavedRegs(const MachineFunction *MF)
                                                                         const {
  return &CalleeSavedReg;
}

void AMDGPURegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator MI,
                                             int SPAdj,
                                             RegScavenger *RS) const {
  assert(!"Subroutines not supported yet");
}

unsigned AMDGPURegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  assert(!"Subroutines not supported yet");
  return 0;
}

#define GET_REGINFO_TARGET_DESC
#include "AMDGPUGenRegisterInfo.inc"
