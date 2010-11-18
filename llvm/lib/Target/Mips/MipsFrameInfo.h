//====--- MipsFrameInfo.h - Define TargetFrameInfo for Mips --*- C++ -*---====//
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

#ifndef ALPHA_FRAMEINFO_H
#define ALPHA_FRAMEINFO_H

#include "Mips.h"
#include "MipsSubtarget.h"
#include "llvm/Target/TargetFrameInfo.h"

namespace llvm {
  class MipsSubtarget;

class MipsFrameInfo : public TargetFrameInfo {
protected:
  const MipsSubtarget &STI;

public:
  explicit MipsFrameInfo(const MipsSubtarget &sti)
    // FIXME: Is this correct at all?
    : TargetFrameInfo(StackGrowsUp, 8, 0), STI(sti) {
  }

  void adjustMipsStackFrame(MachineFunction &MF) const;

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

  bool hasFP(const MachineFunction &MF) const;
};

} // End llvm namespace

#endif
