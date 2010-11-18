//===-- ARMTargetFrameInfo.h - Define TargetFrameInfo for ARM ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//

#ifndef ARM_FRAMEINFO_H
#define ARM_FRAMEINFO_H

#include "ARM.h"
#include "ARMSubtarget.h"
#include "llvm/Target/TargetFrameInfo.h"

namespace llvm {
  class ARMSubtarget;

class ARMFrameInfo : public TargetFrameInfo {
protected:
  const ARMSubtarget &STI;

public:
  explicit ARMFrameInfo(const ARMSubtarget &sti)
    : TargetFrameInfo(StackGrowsDown, sti.getStackAlignment(), 0, 4), STI(sti) {
  }

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

  bool hasFP(const MachineFunction &MF) const;
  bool hasReservedCallFrame(const MachineFunction &MF) const;
  bool canSimplifyCallFramePseudos(const MachineFunction &MF) const;
};

} // End llvm namespace

#endif
