//===-- SparcRegisterInfo.h - Sparc Register Information Impl ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Sparc implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef SPARCREGISTERINFO_H
#define SPARCREGISTERINFO_H

#include "llvm/Target/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "SparcGenRegisterInfo.inc"

namespace llvm {

class SparcSubtarget;
class TargetInstrInfo;
class Type;

struct SparcRegisterInfo : public SparcGenRegisterInfo {
  SparcSubtarget &Subtarget;

  SparcRegisterInfo(SparcSubtarget &st);

  /// Code Generation virtual methods...
  const MCPhysReg *getCalleeSavedRegs(const MachineFunction *MF = 0) const;
  const uint32_t* getCallPreservedMask(CallingConv::ID CC) const;

  const uint32_t* getRTCallPreservedMask(CallingConv::ID CC) const;

  BitVector getReservedRegs(const MachineFunction &MF) const;

  const TargetRegisterClass *getPointerRegClass(const MachineFunction &MF,
                                                unsigned Kind) const;

  void eliminateFrameIndex(MachineBasicBlock::iterator II,
                           int SPAdj, unsigned FIOperandNum,
                           RegScavenger *RS = NULL) const;

  void processFunctionBeforeFrameFinalized(MachineFunction &MF,
                                       RegScavenger *RS = NULL) const;

  // Debug information queries.
  unsigned getFrameRegister(const MachineFunction &MF) const;
};

} // end namespace llvm

#endif
