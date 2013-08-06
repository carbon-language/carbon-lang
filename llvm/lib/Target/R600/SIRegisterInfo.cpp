//===-- SIRegisterInfo.cpp - SI Register Information ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief SI implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//


#include "SIRegisterInfo.h"
#include "AMDGPUTargetMachine.h"

using namespace llvm;

SIRegisterInfo::SIRegisterInfo(AMDGPUTargetMachine &tm)
: AMDGPURegisterInfo(tm),
  TM(tm)
  { }

BitVector SIRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  return Reserved;
}

unsigned SIRegisterInfo::getRegPressureLimit(const TargetRegisterClass *RC,
                                             MachineFunction &MF) const {
  return RC->getNumRegs();
}

const TargetRegisterClass *
SIRegisterInfo::getISARegClass(const TargetRegisterClass * rc) const {
  switch (rc->getID()) {
  case AMDGPU::GPRF32RegClassID:
    return &AMDGPU::VReg_32RegClass;
  default: return rc;
  }
}

const TargetRegisterClass * SIRegisterInfo::getCFGStructurizerRegClass(
                                                                   MVT VT) const {
  switch(VT.SimpleTy) {
    default:
    case MVT::i32: return &AMDGPU::VReg_32RegClass;
  }
}

const TargetRegisterClass *SIRegisterInfo::getPhysRegClass(unsigned Reg) const {
  assert(!TargetRegisterInfo::isVirtualRegister(Reg));

  const TargetRegisterClass *BaseClasses[] = {
    &AMDGPU::VReg_32RegClass,
    &AMDGPU::SReg_32RegClass,
    &AMDGPU::VReg_64RegClass,
    &AMDGPU::SReg_64RegClass,
    &AMDGPU::SReg_128RegClass,
    &AMDGPU::SReg_256RegClass
  };

  for (unsigned i = 0, e = sizeof(BaseClasses) /
                           sizeof(const TargetRegisterClass*); i != e; ++i) {
    if (BaseClasses[i]->contains(Reg)) {
      return BaseClasses[i];
    }
  }
  return NULL;
}
