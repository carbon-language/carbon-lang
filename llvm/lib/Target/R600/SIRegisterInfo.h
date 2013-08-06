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

  virtual BitVector getReservedRegs(const MachineFunction &MF) const;

  virtual unsigned getRegPressureLimit(const TargetRegisterClass *RC,
                                       MachineFunction &MF) const;

  /// \param RC is an AMDIL reg class.
  ///
  /// \returns the SI register class that is equivalent to \p RC.
  virtual const TargetRegisterClass *
    getISARegClass(const TargetRegisterClass *RC) const;

  /// \brief get the register class of the specified type to use in the
  /// CFGStructurizer
  virtual const TargetRegisterClass * getCFGStructurizerRegClass(MVT VT) const;

  /// \brief Return the 'base' register class for this register.
  /// e.g. SGPR0 => SReg_32, VGPR => VReg_32 SGPR0_SGPR1 -> SReg_32, etc.
  const TargetRegisterClass *getPhysRegClass(unsigned Reg) const;
};

} // End namespace llvm

#endif // SIREGISTERINFO_H_
