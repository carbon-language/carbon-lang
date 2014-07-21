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
    SpillTracker(),
    NumUserSGPRs(0) { }

static unsigned createLaneVGPR(MachineRegisterInfo &MRI, MachineFunction *MF) {
  unsigned VGPR = MRI.createVirtualRegister(&AMDGPU::VReg_32RegClass);

  // We need to add this register as live out for the function, in order to
  // have the live range calculated directly.
  //
  // When register spilling begins, we have already calculated the live
  // live intervals for all the registers.  Since we are spilling SGPRs to
  // VGPRs, we need to update the Lane VGPR's live interval every time we
  // spill or restore a register.
  //
  // Unfortunately, there is no good way to update the live interval as
  // the TargetInstrInfo callbacks for spilling and restoring don't give
  // us access to the live interval information.
  //
  // We are lucky, though, because the InlineSpiller calls
  // LiveRangeEdit::calculateRegClassAndHint() which iterates through
  // all the new register that have been created when restoring a register
  // and calls LiveIntervals::getInterval(), which creates and computes
  // the live interval for the newly created register.  However, once this
  // live intervals is created, it doesn't change and since we usually reuse
  // the Lane VGPR multiple times, this means any uses after the first aren't
  // added to the live interval.
  //
  // To work around this, we add Lane VGPRs to the functions live out list,
  // so that we can guarantee its live range will cover all of its uses.

  for (MachineBasicBlock &MBB : *MF) {
    if (MBB.back().getOpcode() == AMDGPU::S_ENDPGM) {
      MBB.back().addOperand(*MF, MachineOperand::CreateReg(VGPR, false, true));
      return VGPR;
    }
  }

  LLVMContext &Ctx = MF->getFunction()->getContext();
  Ctx.emitError("Could not find S_ENDPGM instruction.");

  return VGPR;
}

unsigned SIMachineFunctionInfo::RegSpillTracker::reserveLanes(
    MachineRegisterInfo &MRI, MachineFunction *MF, unsigned NumRegs) {
  unsigned StartLane = CurrentLane;
  CurrentLane += NumRegs;
  if (!LaneVGPR) {
    LaneVGPR = createLaneVGPR(MRI, MF);
  } else {
    if (CurrentLane >= MAX_LANES) {
      StartLane = CurrentLane = 0;
      LaneVGPR = createLaneVGPR(MRI, MF);
    }
  }
  return StartLane;
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
