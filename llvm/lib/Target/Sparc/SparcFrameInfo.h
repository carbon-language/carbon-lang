//===--- SparcFrameInfo.h - Define TargetFrameInfo for Sparc --*- C++ -*---===//
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
#include "llvm/Target/TargetFrameInfo.h"

namespace llvm {
  class SparcSubtarget;

class SparcFrameInfo : public TargetFrameInfo {
  const SparcSubtarget &STI;
public:
  explicit SparcFrameInfo(const SparcSubtarget &sti)
    : TargetFrameInfo(TargetFrameInfo::StackGrowsDown, 8, 0), STI(sti) {
  }

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

  bool hasFP(const MachineFunction &MF) const { return false; }
};

} // End llvm namespace

#endif
