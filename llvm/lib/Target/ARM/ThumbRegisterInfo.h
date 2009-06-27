//===- ThumbRegisterInfo.h - Thumb Register Information Impl ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef THUMBREGISTERINFO_H
#define THUMBREGISTERINFO_H

#include "ARM.h"
#include "ARMRegisterInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"

namespace llvm {
  class ARMSubtarget;
  class TargetInstrInfo;
  class Type;

struct ThumbRegisterInfo : public ARMBaseRegisterInfo {
public:
  ThumbRegisterInfo(const TargetInstrInfo &tii, const ARMSubtarget &STI);

  /// emitLoadConstPool - Emits a load from constpool to materialize the
  /// specified immediate.
  void emitLoadConstPool(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator &MBBI,
                         unsigned DestReg, int Val,
                         unsigned Pred, unsigned PredReg,
                         const TargetInstrInfo *TII,
                         DebugLoc dl) const;

  /// Code Generation virtual methods...
  const TargetRegisterClass *
    getPhysicalRegisterRegClass(unsigned Reg, MVT VT = MVT::Other) const;

  bool isReservedReg(const MachineFunction &MF, unsigned Reg) const;

  bool requiresRegisterScavenging(const MachineFunction &MF) const;

  bool hasReservedCallFrame(MachineFunction &MF) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  void eliminateFrameIndex(MachineBasicBlock::iterator II,
                           int SPAdj, RegScavenger *RS = NULL) const;

  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

  void emitSPUpdate(MachineBasicBlock &MBB, MachineBasicBlock::iterator &MBBI,
                    int NumBytes, ARMCC::CondCodes Pred, unsigned PredReg,
                    const TargetInstrInfo &TII, DebugLoc dl) const;
};
}

#endif // THUMBREGISTERINFO_H
