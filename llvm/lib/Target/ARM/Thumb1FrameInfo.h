//===-- Thumb1FrameInfo.h - Thumb1-specific frame info stuff ----*- C++ -*-===//
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

#ifndef __THUMB_FRAMEINFO_H_
#define __THUMM_FRAMEINFO_H_

#include "ARM.h"
#include "ARMFrameInfo.h"
#include "ARMSubtarget.h"
#include "Thumb1InstrInfo.h"
#include "Thumb1RegisterInfo.h"
#include "llvm/Target/TargetFrameInfo.h"

namespace llvm {
  class ARMSubtarget;

class Thumb1FrameInfo : public ARMFrameInfo {
public:
  explicit Thumb1FrameInfo(const ARMSubtarget &sti)
    : ARMFrameInfo(sti) {
  }

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;
};

} // End llvm namespace

#endif
