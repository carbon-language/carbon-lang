//===-- SIMachineFunctionInfo.cpp - SI Machine Function Info -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
/// \file
//===----------------------------------------------------------------------===//


#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

#define MAX_LANES 64

using namespace llvm;


// Pin the vtable to this file.
void SIMachineFunctionInfo::anchor() {}

SIMachineFunctionInfo::SIMachineFunctionInfo(const MachineFunction &MF)
  : AMDGPUMachineFunction(MF),
    PSInputAddr(0),
    SpillTracker() { }

static unsigned createLaneVGPR(MachineRegisterInfo &MRI) {
  return MRI.createVirtualRegister(&AMDGPU::VReg_32RegClass);
}

unsigned SIMachineFunctionInfo::RegSpillTracker::getNextLane(MachineRegisterInfo &MRI) {
  if (!LaneVGPR) {
    LaneVGPR = createLaneVGPR(MRI);
  } else {
    CurrentLane++;
    if (CurrentLane == MAX_LANES) {
      CurrentLane = 0;
      LaneVGPR = createLaneVGPR(MRI);
    }
  }
  return CurrentLane;
}

void SIMachineFunctionInfo::RegSpillTracker::addSpilledReg(unsigned FrameIndex,
                                                           unsigned Reg,
                                                           int Lane) {
  SpilledRegisters[FrameIndex] = SpilledReg(Reg, Lane);
}

const SIMachineFunctionInfo::SpilledReg&
SIMachineFunctionInfo::RegSpillTracker::getSpilledReg(unsigned FrameIndex) {
  return SpilledRegisters[FrameIndex];
}
