//===-- X86TargetFrameInfo.h - Define TargetFrameInfo for X86 ---*- C++ -*-===//
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

#ifndef X86_FRAMEINFO_H
#define X86_FRAMEINFO_H

#include "X86Subtarget.h"
#include "llvm/Target/TargetFrameInfo.h"

namespace llvm {
  class MCSymbol;

class X86FrameInfo : public TargetFrameInfo {
protected:
  const X86Subtarget &STI;

public:
  explicit X86FrameInfo(const X86Subtarget &sti)
    : TargetFrameInfo(StackGrowsDown,
                      sti.getStackAlignment(),
                      (sti.isTargetWin64() ? -40 : (sti.is64Bit() ? -8 : -4))),
      STI(sti) {
  }

  void emitCalleeSavedFrameMoves(MachineFunction &MF, MCSymbol *Label,
                                 unsigned FramePtr) const;

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;
};

} // End llvm namespace

#endif
