//==- SystemZFrameInfo.h - Define TargetFrameInfo for z/System --*- C++ -*--==//
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

#ifndef SYSTEMZ_FRAMEINFO_H
#define SYSTEMZ_FRAMEINFO_H

#include "SystemZ.h"
#include "SystemZSubtarget.h"
#include "llvm/Target/TargetFrameInfo.h"

namespace llvm {
  class SystemZSubtarget;

class SystemZFrameInfo : public TargetFrameInfo {
protected:
  const SystemZSubtarget &STI;

public:
  explicit SystemZFrameInfo(const SystemZSubtarget &sti)
    : TargetFrameInfo(TargetFrameInfo::StackGrowsDown, 8, -160), STI(sti) {
  }

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

  bool hasReservedCallFrame(const MachineFunction &MF) const { return true; }
  bool hasFP(const MachineFunction &MF) const;
};

} // End llvm namespace

#endif
