//==- AArch64RegisterInfo.h - AArch64 Register Information Impl -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the AArch64 implementation of the MCRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_AARCH64REGISTERINFO_H
#define LLVM_TARGET_AARCH64REGISTERINFO_H

#include "llvm/Target/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "AArch64GenRegisterInfo.inc"

namespace llvm {

class AArch64InstrInfo;
class AArch64Subtarget;

struct AArch64RegisterInfo : public AArch64GenRegisterInfo {
  AArch64RegisterInfo();

  const MCPhysReg *getCalleeSavedRegs(const MachineFunction *MF = 0) const;
  const uint32_t *getCallPreservedMask(CallingConv::ID) const;

  const uint32_t *getTLSDescCallPreservedMask() const;

  BitVector getReservedRegs(const MachineFunction &MF) const;
  unsigned getFrameRegister(const MachineFunction &MF) const;

  void eliminateFrameIndex(MachineBasicBlock::iterator II, int SPAdj,
                           unsigned FIOperandNum,
                           RegScavenger *Rs = NULL) const;

  /// getCrossCopyRegClass - Returns a legal register class to copy a register
  /// in the specified class to or from. Returns original class if it is
  /// possible to copy between a two registers of the specified class.
  const TargetRegisterClass *
  getCrossCopyRegClass(const TargetRegisterClass *RC) const;

  /// getLargestLegalSuperClass - Returns the largest super class of RC that is
  /// legal to use in the current sub-target and has the same spill size.
  const TargetRegisterClass*
  getLargestLegalSuperClass(const TargetRegisterClass *RC) const {
    if (RC == &AArch64::tcGPR64RegClass)
      return &AArch64::GPR64RegClass;

    return RC;
  }

  bool requiresRegisterScavenging(const MachineFunction &MF) const {
    return true;
  }

  bool requiresFrameIndexScavenging(const MachineFunction &MF) const {
    return true;
  }

  bool useFPForScavengingIndex(const MachineFunction &MF) const;
};

} // end namespace llvm

#endif // LLVM_TARGET_AARCH64REGISTERINFO_H
