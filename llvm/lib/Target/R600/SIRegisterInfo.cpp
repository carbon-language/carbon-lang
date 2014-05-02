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
#include "SIInstrInfo.h"

using namespace llvm;

SIRegisterInfo::SIRegisterInfo(AMDGPUTargetMachine &tm)
: AMDGPURegisterInfo(tm),
  TM(tm)
  { }

BitVector SIRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  Reserved.set(AMDGPU::EXEC);
  Reserved.set(AMDGPU::INDIRECT_BASE_ADDR);
  const SIInstrInfo *TII = static_cast<const SIInstrInfo*>(TM.getInstrInfo());
  TII->reserveIndirectRegisters(Reserved, MF);
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

unsigned SIRegisterInfo::getHWRegIndex(unsigned Reg) const {
  return getEncodingValue(Reg) & 0xff;
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
  return nullptr;
}

bool SIRegisterInfo::isSGPRClass(const TargetRegisterClass *RC) const {
  if (!RC) {
    return false;
  }
  return !hasVGPRs(RC);
}

bool SIRegisterInfo::hasVGPRs(const TargetRegisterClass *RC) const {
  return getCommonSubClass(&AMDGPU::VReg_32RegClass, RC) ||
         getCommonSubClass(&AMDGPU::VReg_64RegClass, RC) ||
         getCommonSubClass(&AMDGPU::VReg_96RegClass, RC) ||
         getCommonSubClass(&AMDGPU::VReg_128RegClass, RC) ||
         getCommonSubClass(&AMDGPU::VReg_256RegClass, RC) ||
         getCommonSubClass(&AMDGPU::VReg_512RegClass, RC);
}

const TargetRegisterClass *SIRegisterInfo::getEquivalentVGPRClass(
                                         const TargetRegisterClass *SRC) const {
    if (hasVGPRs(SRC)) {
      return SRC;
    } else if (SRC == &AMDGPU::SCCRegRegClass) {
      return &AMDGPU::VCCRegRegClass;
    } else if (getCommonSubClass(SRC, &AMDGPU::SGPR_32RegClass)) {
      return &AMDGPU::VReg_32RegClass;
    } else if (getCommonSubClass(SRC, &AMDGPU::SGPR_64RegClass)) {
      return &AMDGPU::VReg_64RegClass;
    } else if (getCommonSubClass(SRC, &AMDGPU::SReg_128RegClass)) {
      return &AMDGPU::VReg_128RegClass;
    } else if (getCommonSubClass(SRC, &AMDGPU::SReg_256RegClass)) {
      return &AMDGPU::VReg_256RegClass;
    } else if (getCommonSubClass(SRC, &AMDGPU::SReg_512RegClass)) {
      return &AMDGPU::VReg_512RegClass;
    }
    return nullptr;
}

const TargetRegisterClass *SIRegisterInfo::getSubRegClass(
                         const TargetRegisterClass *RC, unsigned SubIdx) const {
  if (SubIdx == AMDGPU::NoSubRegister)
    return RC;

  // If this register has a sub-register, we can safely assume it is a 32-bit
  // register, because all of SI's sub-registers are 32-bit.
  if (isSGPRClass(RC)) {
    return &AMDGPU::SGPR_32RegClass;
  } else {
    return &AMDGPU::VGPR_32RegClass;
  }
}

unsigned SIRegisterInfo::getPhysRegSubReg(unsigned Reg,
                                          const TargetRegisterClass *SubRC,
                                          unsigned Channel) const {
  unsigned Index = getHWRegIndex(Reg);
  return SubRC->getRegister(Index + Channel);
}
