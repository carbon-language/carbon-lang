//===- Thumb2RegisterInfo.h - Thumb-2 Register Information Impl ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Thumb-2 implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef THUMB2REGISTERINFO_H
#define THUMB2REGISTERINFO_H

#include "ARM.h"
#include "ARMRegisterInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"

namespace llvm {
  class ARMSubtarget;
  class ARMBaseInstrInfo;
  class Type;

struct Thumb2RegisterInfo : public ARMBaseRegisterInfo {
public:
  Thumb2RegisterInfo(const ARMBaseInstrInfo &tii, const ARMSubtarget &STI);

  /// emitLoadConstPool - Emits a load from constpool to materialize the
  /// specified immediate.
  void emitLoadConstPool(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator &MBBI,
                         DebugLoc dl,
                         unsigned DestReg, unsigned SubIdx, int Val,
                         ARMCC::CondCodes Pred = ARMCC::AL,
                         unsigned PredReg = 0) const;

  bool requiresRegisterScavenging(const MachineFunction &MF) const;

  // rewrite MI to access 'Offset' bytes from the FP. Return the offset that
  // could not be handled directly in MI.
  virtual int
  rewriteFrameIndex(MachineInstr &MI, unsigned FrameRegIdx,
                    unsigned MOVOpc, unsigned ADDriOpc, unsigned SUBriOpc,
                    unsigned FrameReg, int Offset) const;

  void eliminateFrameIndex(MachineBasicBlock::iterator II,
                           int SPAdj, RegScavenger *RS = NULL) const {
    ARMBaseRegisterInfo::eliminateFrameIndexImpl(II, ARM::t2MOVr, ARM::t2ADDri,
                                                 ARM::t2SUBri, SPAdj, RS);
  }
};
}

#endif // THUMB2REGISTERINFO_H
