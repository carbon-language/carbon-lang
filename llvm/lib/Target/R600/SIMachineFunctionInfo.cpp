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
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"

#define MAX_LANES 64

using namespace llvm;


// Pin the vtable to this file.
void SIMachineFunctionInfo::anchor() {}

SIMachineFunctionInfo::SIMachineFunctionInfo(const MachineFunction &MF)
  : AMDGPUMachineFunction(MF),
    PSInputAddr(0),
    NumUserSGPRs(0) { }

/// \brief Returns a register that is not used at any point in the function.
///        If all registers are used, then this function will return
//         AMDGPU::NoRegister.
static unsigned findUnusedVGPR(const MachineRegisterInfo &MRI) {

  const TargetRegisterClass *RC = &AMDGPU::VGPR_32RegClass;

  for (TargetRegisterClass::iterator I = RC->begin(), E = RC->end();
       I != E; ++I) {
    if (!MRI.isPhysRegUsed(*I))
      return *I;
  }
  return AMDGPU::NoRegister;
}

SIMachineFunctionInfo::SpilledReg SIMachineFunctionInfo::getSpilledReg(
                                                       MachineFunction *MF,
                                                       unsigned FrameIndex,
                                                       unsigned SubIdx) {
  const MachineFrameInfo *FrameInfo = MF->getFrameInfo();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  int64_t Offset = FrameInfo->getObjectOffset(FrameIndex);
  Offset += SubIdx * 4;

  unsigned LaneVGPRIdx = Offset / (64 * 4);
  unsigned Lane = (Offset / 4) % 64;

  struct SpilledReg Spill;

  if (!LaneVGPRs.count(LaneVGPRIdx)) {
    unsigned LaneVGPR = findUnusedVGPR(MRI);
    LaneVGPRs[LaneVGPRIdx] = LaneVGPR;
    MRI.setPhysRegUsed(LaneVGPR);

    // Add this register as live-in to all blocks to avoid machine verifer
    // complaining about use of an undefined physical register.
    for (MachineFunction::iterator BI = MF->begin(), BE = MF->end();
         BI != BE; ++BI) {
      BI->addLiveIn(LaneVGPR);
    }
  }

  Spill.VGPR = LaneVGPRs[LaneVGPRIdx];
  Spill.Lane = Lane;
  return Spill;
}
