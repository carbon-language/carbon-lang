//===- Thumb1RegisterInfo.h - Thumb-1 Register Information Impl -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Thumb-1 implementation of the TargetRegisterInfo
// class.
//
//===----------------------------------------------------------------------===//

#ifndef THUMB1REGISTERINFO_H
#define THUMB1REGISTERINFO_H

#include "ARM.h"
#include "ARMRegisterInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"

namespace llvm {
  class ARMSubtarget;
  class ARMBaseInstrInfo;
  class Type;

struct Thumb1RegisterInfo : public ARMBaseRegisterInfo {
public:
  Thumb1RegisterInfo(const ARMBaseInstrInfo &tii, const ARMSubtarget &STI);

  /// emitLoadConstPool - Emits a load from constpool to materialize the
  /// specified immediate.
 void emitLoadConstPool(MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator &MBBI,
                        DebugLoc dl,
                        unsigned DestReg, unsigned SubIdx, int Val,
                        ARMCC::CondCodes Pred = ARMCC::AL,
                        unsigned PredReg = 0) const;

  /// Code Generation virtual methods...
  const TargetRegisterClass *
    getPhysicalRegisterRegClass(unsigned Reg, EVT VT = MVT::Other) const;

  bool hasReservedCallFrame(MachineFunction &MF) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  // rewrite MI to access 'Offset' bytes from the FP. Return the offset that
  // could not be handled directly in MI.
  int rewriteFrameIndex(MachineInstr &MI, unsigned FrameRegIdx,
                        unsigned FrameReg, int Offset,
                        unsigned MOVOpc, unsigned ADDriOpc, unsigned SUBriOpc) const;

  bool saveScavengerRegister(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator I,
                             MachineBasicBlock::iterator &UseMI,
                             const TargetRegisterClass *RC,
                             unsigned Reg) const;
  unsigned eliminateFrameIndex(MachineBasicBlock::iterator II,
                               int SPAdj, FrameIndexValue *Value = NULL,
                               RegScavenger *RS = NULL) const;

  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;
};
}

#endif // THUMB1REGISTERINFO_H
