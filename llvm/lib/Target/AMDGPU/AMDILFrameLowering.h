//===--------------------- AMDILFrameLowering.h -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interface to describe a layout of a stack frame on a AMDIL target machine
//
//===----------------------------------------------------------------------===//
#ifndef _AMDILFRAME_LOWERING_H_
#define _AMDILFRAME_LOWERING_H_

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Target/TargetFrameLowering.h"

/// Information about the stack frame layout on the AMDIL targets. It holds
/// the direction of the stack growth, the known stack alignment on entry to
/// each function, and the offset to the locals area.
/// See TargetFrameInfo for more comments.

namespace llvm {
  class AMDILFrameLowering : public TargetFrameLowering {
    public:
      AMDILFrameLowering(StackDirection D, unsigned StackAl, int LAO, unsigned
          TransAl = 1);
      virtual ~AMDILFrameLowering();
      virtual int getFrameIndexOffset(const MachineFunction &MF,
                                         int FI) const;
      virtual const SpillSlot *
        getCalleeSavedSpillSlots(unsigned &NumEntries) const;
      virtual void emitPrologue(MachineFunction &MF) const;
      virtual void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;
      virtual bool hasFP(const MachineFunction &MF) const;
  }; // class AMDILFrameLowering
} // namespace llvm
#endif // _AMDILFRAME_LOWERING_H_
