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

AMDGPURegisterInfo::AMDGPURegisterInfo(const AMDGPUSubtarget &st)
: AMDGPUGenRegisterInfo(0),
  ST(st)
  { }

//===----------------------------------------------------------------------===//
// Function handling callbacks - Functions are a seldom used feature of GPUS, so
// they are not supported at this time.
//===----------------------------------------------------------------------===//

const MCPhysReg AMDGPURegisterInfo::CalleeSavedReg = AMDGPU::NoRegister;

const MCPhysReg*
AMDGPURegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  return &CalleeSavedReg;
}

void AMDGPURegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator MI,
                                             int SPAdj,
                                             unsigned FIOperandNum,
                                             RegScavenger *RS) const {
  llvm_unreachable("Subroutines not supported yet");
}

unsigned AMDGPURegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  assert(!"Subroutines not supported yet");
  return 0;
}

unsigned AMDGPURegisterInfo::getSubRegFromChannel(unsigned Channel) const {
  static const unsigned SubRegs[] = {
    AMDGPU::sub0, AMDGPU::sub1, AMDGPU::sub2, AMDGPU::sub3, AMDGPU::sub4,
    AMDGPU::sub5, AMDGPU::sub6, AMDGPU::sub7, AMDGPU::sub8, AMDGPU::sub9,
    AMDGPU::sub10, AMDGPU::sub11, AMDGPU::sub12, AMDGPU::sub13, AMDGPU::sub14,
    AMDGPU::sub15
  };

  assert(Channel < array_lengthof(SubRegs));
  return SubRegs[Channel];
}

unsigned AMDGPURegisterInfo::getIndirectSubReg(unsigned IndirectIndex) const {

  return getSubRegFromChannel(IndirectIndex);
}

#define GET_REGINFO_TARGET_DESC
#include "AMDGPUGenRegisterInfo.inc"
