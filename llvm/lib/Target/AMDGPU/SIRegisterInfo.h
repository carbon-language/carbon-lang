//===-- SIRegisterInfo.h - SI Register Info Interface ----------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interface definition for SIRegisterInfo
//
//===----------------------------------------------------------------------===//


#ifndef SIREGISTERINFO_H_
#define SIREGISTERINFO_H_

#include "AMDGPURegisterInfo.h"

namespace llvm {

class AMDGPUTargetMachine;
class TargetInstrInfo;

struct SIRegisterInfo : public AMDGPURegisterInfo
{
  AMDGPUTargetMachine &TM;
  const TargetInstrInfo &TII;

  SIRegisterInfo(AMDGPUTargetMachine &tm, const TargetInstrInfo &tii);

  virtual BitVector getReservedRegs(const MachineFunction &MF) const;

  /// getISARegClass - rc is an AMDIL reg class.  This function returns the
  /// SI register class that is equivalent to the given AMDIL register class.
  virtual const TargetRegisterClass *
    getISARegClass(const TargetRegisterClass * rc) const;

  /// getCFGStructurizerRegClass - get the register class of the specified
  /// type to use in the CFGStructurizer
  virtual const TargetRegisterClass * getCFGStructurizerRegClass(MVT VT) const;

};

} // End namespace llvm

#endif // SIREGISTERINFO_H_
