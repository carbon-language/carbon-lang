//===-- SIRegisterInfo.cpp - SI Register Information ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SI implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//


#include "SIRegisterInfo.h"
#include "AMDGPUTargetMachine.h"
#include "AMDGPUUtil.h"

using namespace llvm;

SIRegisterInfo::SIRegisterInfo(AMDGPUTargetMachine &tm,
    const TargetInstrInfo &tii)
: AMDGPURegisterInfo(tm, tii),
  TM(tm),
  TII(tii)
  { }

BitVector SIRegisterInfo::getReservedRegs(const MachineFunction &MF) const
{
  BitVector Reserved(getNumRegs());
  return Reserved;
}

const TargetRegisterClass *
SIRegisterInfo::getISARegClass(const TargetRegisterClass * rc) const
{
  switch (rc->getID()) {
  case AMDGPU::GPRF32RegClassID:
    return &AMDGPU::VReg_32RegClass;
  default: return rc;
  }
}

const TargetRegisterClass * SIRegisterInfo::getCFGStructurizerRegClass(
                                                                   MVT VT) const
{
  switch(VT.SimpleTy) {
    default:
    case MVT::i32: return &AMDGPU::VReg_32RegClass;
  }
}
