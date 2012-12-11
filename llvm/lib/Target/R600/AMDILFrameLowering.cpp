//===----------------------- AMDILFrameLowering.cpp -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
/// \file
/// \brief Interface to describe a layout of a stack frame on a AMDGPU target
/// machine.
//
//===----------------------------------------------------------------------===//
#include "AMDILFrameLowering.h"
#include "llvm/CodeGen/MachineFrameInfo.h"

using namespace llvm;
AMDGPUFrameLowering::AMDGPUFrameLowering(StackDirection D, unsigned StackAl,
    int LAO, unsigned TransAl)
  : TargetFrameLowering(D, StackAl, LAO, TransAl) {
}

AMDGPUFrameLowering::~AMDGPUFrameLowering() {
}

int AMDGPUFrameLowering::getFrameIndexOffset(const MachineFunction &MF,
                                         int FI) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  return MFI->getObjectOffset(FI);
}

const TargetFrameLowering::SpillSlot *
AMDGPUFrameLowering::getCalleeSavedSpillSlots(unsigned &NumEntries) const {
  NumEntries = 0;
  return 0;
}
void
AMDGPUFrameLowering::emitPrologue(MachineFunction &MF) const {
}
void
AMDGPUFrameLowering::emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const {
}
bool
AMDGPUFrameLowering::hasFP(const MachineFunction &MF) const {
  return false;
}
