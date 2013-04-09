//===-- SparcFrameLowering.h - Define frame lowering for Sparc --*- C++ -*-===//
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

#ifndef SPARC_FRAMEINFO_H
#define SPARC_FRAMEINFO_H

#include "Sparc.h"
#include "SparcSubtarget.h"
#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {
  class SparcSubtarget;

class SparcFrameLowering : public TargetFrameLowering {
  const SparcSubtarget &SubTarget;
public:
  explicit SparcFrameLowering(const SparcSubtarget &ST)
    : TargetFrameLowering(TargetFrameLowering::StackGrowsDown,
                          ST.is64Bit() ? 16 : 8, 0, ST.is64Bit() ? 16 : 8),
      SubTarget(ST) {}

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  bool hasFP(const MachineFunction &MF) const { return false; }
};

} // End llvm namespace

#endif
