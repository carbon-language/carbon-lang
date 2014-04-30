//===-- R600RegisterInfo.h - R600 Register Info Interface ------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Interface definition for R600RegisterInfo
//
//===----------------------------------------------------------------------===//

#ifndef R600REGISTERINFO_H_
#define R600REGISTERINFO_H_

#include "AMDGPURegisterInfo.h"
#include "AMDGPUTargetMachine.h"

namespace llvm {

class R600TargetMachine;

struct R600RegisterInfo : public AMDGPURegisterInfo {
  AMDGPUTargetMachine &TM;
  RegClassWeight RCW;

  R600RegisterInfo(AMDGPUTargetMachine &tm);

  BitVector getReservedRegs(const MachineFunction &MF) const override;

  /// \param RC is an AMDIL reg class.
  ///
  /// \returns the R600 reg class that is equivalent to \p RC.
  const TargetRegisterClass *getISARegClass(
    const TargetRegisterClass *RC) const override;

  /// \brief get the HW encoding for a register's channel.
  unsigned getHWRegChan(unsigned reg) const;

  unsigned getHWRegIndex(unsigned Reg) const override;

  /// \brief get the register class of the specified type to use in the
  /// CFGStructurizer
  const TargetRegisterClass * getCFGStructurizerRegClass(MVT VT) const override;

  const RegClassWeight &
    getRegClassWeight(const TargetRegisterClass *RC) const override;

  // \returns true if \p Reg can be defined in one ALU caluse and used in another.
  bool isPhysRegLiveAcrossClauses(unsigned Reg) const;
};

} // End namespace llvm

#endif // AMDIDSAREGISTERINFO_H_
