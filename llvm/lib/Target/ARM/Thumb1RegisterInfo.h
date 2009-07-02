//===- Thumb1RegisterInfo.h - Thumb-1 Register Information Impl ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Thumb-1 implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef THUMB1REGISTERINFO_H
#define THUMB1REGISTERINFO_H

#include "ARM.h"
#include "ARMRegisterInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"

namespace llvm {
  class ARMSubtarget;
  class TargetInstrInfo;
  class Type;

struct Thumb1RegisterInfo : public ARMBaseRegisterInfo {
public:
  Thumb1RegisterInfo(const TargetInstrInfo &tii, const ARMSubtarget &STI);

  /// emitLoadConstPool - Emits a load from constpool to materialize the
  /// specified immediate.
  void emitLoadConstPool(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator &MBBI,
                         unsigned DestReg, int Val,
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
};
}

#endif // THUMB1REGISTERINFO_H
