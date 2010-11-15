//===-- MSP430FrameInfo.h - Define TargetFrameInfo for MSP430 --*- C++ -*--===//
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

#ifndef MSP430_FRAMEINFO_H
#define MSP430_FRAMEINFO_H

#include "MSP430.h"
#include "MSP430Subtarget.h"
#include "llvm/Target/TargetFrameInfo.h"

namespace llvm {
  class MSP430Subtarget;

class MSP430FrameInfo : public TargetFrameInfo {
protected:
  const MSP430Subtarget &STI;

public:
  explicit MSP430FrameInfo(const MSP430Subtarget &sti)
    : TargetFrameInfo(TargetFrameInfo::StackGrowsDown, 2, -2), STI(sti) {
  }

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;
};

} // End llvm namespace

#endif
