//===-- MipsFrameLowering.h - Define frame lowering for Mips ----*- C++ -*-===//
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

#ifndef MIPS_FRAMEINFO_H
#define MIPS_FRAMEINFO_H

#include "Mips.h"
#include "MipsSubtarget.h"
#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {
  class MipsSubtarget;

class MipsFrameLowering : public TargetFrameLowering {
protected:
  const MipsSubtarget &STI;

public:
  explicit MipsFrameLowering(const MipsSubtarget &sti)
    : TargetFrameLowering(StackGrowsDown, sti.hasMips64() ? 16 : 8, 0,
                          sti.hasMips64() ? 16 : 8), STI(sti) {}

  bool hasFP(const MachineFunction &MF) const;
};

} // End llvm namespace

#endif
