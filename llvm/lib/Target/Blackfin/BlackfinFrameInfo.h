//=- BlackfinFrameInfo.h - Define TargetFrameInfo for Blackfin --*- C++ -*--==//
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

#include "Blackfin.h"
#include "BlackfinSubtarget.h"
#include "llvm/Target/TargetFrameInfo.h"

namespace llvm {
  class BlackfinSubtarget;

class BlackfinFrameInfo : public TargetFrameInfo {
protected:
  const BlackfinSubtarget &STI;

public:
  explicit BlackfinFrameInfo(const BlackfinSubtarget &sti)
    : TargetFrameInfo(TargetFrameInfo::StackGrowsDown, 4, 0), STI(sti) {
  }

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

  bool hasFP(const MachineFunction &MF) const;

  void processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                            RegScavenger *RS) const;
};

} // End llvm namespace

#endif
