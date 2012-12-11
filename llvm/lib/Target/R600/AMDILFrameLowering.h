//===--------------------- AMDILFrameLowering.h -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Interface to describe a layout of a stack frame on a AMDIL target
/// machine.
//
//===----------------------------------------------------------------------===//
#ifndef AMDILFRAME_LOWERING_H
#define AMDILFRAME_LOWERING_H

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {

/// \brief Information about the stack frame layout on the AMDGPU targets.
///
/// It holds the direction of the stack growth, the known stack alignment on
/// entry to each function, and the offset to the locals area.
/// See TargetFrameInfo for more comments.
class AMDGPUFrameLowering : public TargetFrameLowering {
public:
  AMDGPUFrameLowering(StackDirection D, unsigned StackAl, int LAO,
                      unsigned TransAl = 1);
  virtual ~AMDGPUFrameLowering();
  virtual int getFrameIndexOffset(const MachineFunction &MF, int FI) const;
  virtual const SpillSlot *getCalleeSavedSpillSlots(unsigned &NumEntries) const;
  virtual void emitPrologue(MachineFunction &MF) const;
  virtual void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;
  virtual bool hasFP(const MachineFunction &MF) const;
};
} // namespace llvm
#endif // AMDILFRAME_LOWERING_H
