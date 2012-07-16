//===----------------------- AMDILFrameLowering.cpp -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
// Interface to describe a layout of a stack frame on a AMDIL target machine
//
//===----------------------------------------------------------------------===//
#include "AMDILFrameLowering.h"
#include "llvm/CodeGen/MachineFrameInfo.h"

using namespace llvm;
AMDILFrameLowering::AMDILFrameLowering(StackDirection D, unsigned StackAl,
    int LAO, unsigned TransAl)
  : TargetFrameLowering(D, StackAl, LAO, TransAl)
{
}

AMDILFrameLowering::~AMDILFrameLowering()
{
}

/// getFrameIndexOffset - Returns the displacement from the frame register to
/// the stack frame of the specified index.
int AMDILFrameLowering::getFrameIndexOffset(const MachineFunction &MF,
                                         int FI) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  return MFI->getObjectOffset(FI);
}

const TargetFrameLowering::SpillSlot *
AMDILFrameLowering::getCalleeSavedSpillSlots(unsigned &NumEntries) const
{
  NumEntries = 0;
  return 0;
}
void
AMDILFrameLowering::emitPrologue(MachineFunction &MF) const
{
}
void
AMDILFrameLowering::emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const
{
}
bool
AMDILFrameLowering::hasFP(const MachineFunction &MF) const
{
  return false;
}
