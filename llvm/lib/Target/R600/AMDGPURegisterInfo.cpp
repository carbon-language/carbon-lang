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
                                             unsigned FIOperandNum,
                                             RegScavenger *RS) const {
  assert(!"Subroutines not supported yet");
}

unsigned AMDGPURegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  assert(!"Subroutines not supported yet");
  return 0;
}

unsigned AMDGPURegisterInfo::getIndirectSubReg(unsigned IndirectIndex) const {

  switch(IndirectIndex) {
  case 0: return AMDGPU::indirect_0;
  case 1: return AMDGPU::indirect_1;
  case 2: return AMDGPU::indirect_2;
  case 3: return AMDGPU::indirect_3;
  case 4: return AMDGPU::indirect_4;
  case 5: return AMDGPU::indirect_5;
  case 6: return AMDGPU::indirect_6;
  case 7: return AMDGPU::indirect_7;
  case 8: return AMDGPU::indirect_8;
  case 9: return AMDGPU::indirect_9;
  case 10: return AMDGPU::indirect_10;
  case 11: return AMDGPU::indirect_11;
  case 12: return AMDGPU::indirect_12;
  case 13: return AMDGPU::indirect_13;
  case 14: return AMDGPU::indirect_14;
  case 15: return AMDGPU::indirect_15;
  default: llvm_unreachable("indirect index out of range");
  }
}

#define GET_REGINFO_TARGET_DESC
#include "AMDGPUGenRegisterInfo.inc"
