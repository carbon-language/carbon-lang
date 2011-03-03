//===--- PTXFrameLowering.h - Define frame lowering for PTX --*- C++ -*----===//
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

#ifndef PTX_FRAMEINFO_H
#define PTX_FRAMEINFO_H

#include "PTX.h"
#include "PTXSubtarget.h"
#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {
  class PTXSubtarget;

class PTXFrameLowering : public TargetFrameLowering {
protected:
  const PTXSubtarget &STI;

public:
  explicit PTXFrameLowering(const PTXSubtarget &sti)
    : TargetFrameLowering(TargetFrameLowering::StackGrowsDown, 2, -2),
      STI(sti) {
  }

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

  bool hasFP(const MachineFunction &MF) const { return false; }
};

} // End llvm namespace

#endif
