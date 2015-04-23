//==- HexagonRegisterInfo.h - Hexagon Register Information Impl --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Hexagon implementation of the TargetRegisterInfo
// class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_HEXAGONREGISTERINFO_H
#define LLVM_LIB_TARGET_HEXAGON_HEXAGONREGISTERINFO_H

#include "llvm/MC/MachineLocation.h"
#include "llvm/Target/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "HexagonGenRegisterInfo.inc"

//
//  We try not to hard code the reserved registers in our code,
//  so the following two macros were defined. However, there
//  are still a few places that R11 and R10 are hard wired.
//  See below. If, in the future, we decided to change the reserved
//  register. Don't forget changing the following places.
//
//  1. the "Defs" set of STriw_pred in HexagonInstrInfo.td
//  2. the "Defs" set of LDri_pred in HexagonInstrInfo.td
//  3. the definition of "IntRegs" in HexagonRegisterInfo.td
//  4. the definition of "DoubleRegs" in HexagonRegisterInfo.td
//
#define HEXAGON_RESERVED_REG_1 Hexagon::R10
#define HEXAGON_RESERVED_REG_2 Hexagon::R11

namespace llvm {
class HexagonRegisterInfo : public HexagonGenRegisterInfo {
public:
  HexagonRegisterInfo();

  /// Code Generation virtual methods...
  const MCPhysReg *getCalleeSavedRegs(const MachineFunction *MF)
        const override;


  BitVector getReservedRegs(const MachineFunction &MF) const override;

  void eliminateFrameIndex(MachineBasicBlock::iterator II, int SPAdj,
        unsigned FIOperandNum, RegScavenger *RS = nullptr) const override;

  /// Returns true since we may need scavenging for a temporary register
  /// when generating hardware loop instructions.
  bool requiresRegisterScavenging(const MachineFunction &MF) const override {
    return true;
  }

  /// Returns true. Spill code for predicate registers might need an extra
  /// register.
  bool requiresFrameIndexScavenging(const MachineFunction &MF) const override {
    return true;
  }

  bool needsStackRealignment(const MachineFunction &MF) const override;

  /// Returns true if the frame pointer is valid.
  bool useFPForScavengingIndex(const MachineFunction &MF) const override;

  bool trackLivenessAfterRegAlloc(const MachineFunction &MF) const override {
    return true;
  }

  // Debug information queries.
  unsigned getRARegister() const;
  unsigned getFrameRegister(const MachineFunction &MF) const override;
  unsigned getFrameRegister() const;
  unsigned getStackRegister() const;

  const MCPhysReg *getCallerSavedRegs(const MachineFunction *MF) const;

  unsigned getFirstCallerSavedNonParamReg() const;

  bool isEHReturnCalleeSaveReg(unsigned Reg) const;
  bool isCalleeSaveReg(unsigned Reg) const;
};

} // end namespace llvm

#endif
