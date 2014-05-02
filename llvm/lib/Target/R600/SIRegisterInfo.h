//===-- SIRegisterInfo.h - SI Register Info Interface ----------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Interface definition for SIRegisterInfo
//
//===----------------------------------------------------------------------===//


#ifndef SIREGISTERINFO_H_
#define SIREGISTERINFO_H_

#include "AMDGPURegisterInfo.h"

namespace llvm {

class AMDGPUTargetMachine;

struct SIRegisterInfo : public AMDGPURegisterInfo {
  AMDGPUTargetMachine &TM;

  SIRegisterInfo(AMDGPUTargetMachine &tm);

  BitVector getReservedRegs(const MachineFunction &MF) const override;

  unsigned getRegPressureLimit(const TargetRegisterClass *RC,
                               MachineFunction &MF) const override;

  /// \param RC is an AMDIL reg class.
  ///
  /// \returns the SI register class that is equivalent to \p RC.
  const TargetRegisterClass *
    getISARegClass(const TargetRegisterClass *RC) const override;

  /// \brief get the register class of the specified type to use in the
  /// CFGStructurizer
  const TargetRegisterClass * getCFGStructurizerRegClass(MVT VT) const override;

  unsigned getHWRegIndex(unsigned Reg) const override;

  /// \brief Return the 'base' register class for this register.
  /// e.g. SGPR0 => SReg_32, VGPR => VReg_32 SGPR0_SGPR1 -> SReg_32, etc.
  const TargetRegisterClass *getPhysRegClass(unsigned Reg) const;

  /// \returns true if this class contains only SGPR registers
  bool isSGPRClass(const TargetRegisterClass *RC) const;

  /// \returns true if this class contains VGPR registers.
  bool hasVGPRs(const TargetRegisterClass *RC) const;

  /// \returns A VGPR reg class with the same width as \p SRC
  const TargetRegisterClass *getEquivalentVGPRClass(
                                          const TargetRegisterClass *SRC) const;

  /// \returns The register class that is used for a sub-register of \p RC for
  /// the given \p SubIdx.  If \p SubIdx equals NoSubRegister, \p RC will
  /// be returned.
  const TargetRegisterClass *getSubRegClass(const TargetRegisterClass *RC,
                                            unsigned SubIdx) const;

  /// \p Channel This is the register channel (e.g. a value from 0-16), not the
  ///            SubReg index.
  /// \returns The sub-register of Reg that is in Channel.
  unsigned getPhysRegSubReg(unsigned Reg, const TargetRegisterClass *SubRC,
                            unsigned Channel) const;
};

} // End namespace llvm

#endif // SIREGISTERINFO_H_
