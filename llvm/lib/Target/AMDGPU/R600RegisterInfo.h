//===-- R600RegisterInfo.h - R600 Register Info Interface ------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interface definition for R600RegisterInfo
//
//===----------------------------------------------------------------------===//

#ifndef R600REGISTERINFO_H_
#define R600REGISTERINFO_H_

#include "AMDGPUTargetMachine.h"
#include "AMDILRegisterInfo.h"

namespace llvm {

class R600TargetMachine;
class TargetInstrInfo;

struct R600RegisterInfo : public AMDGPURegisterInfo
{
  AMDGPUTargetMachine &TM;
  const TargetInstrInfo &TII;

  R600RegisterInfo(AMDGPUTargetMachine &tm, const TargetInstrInfo &tii);

  virtual BitVector getReservedRegs(const MachineFunction &MF) const;

  /// getISARegClass - rc is an AMDIL reg class.  This function returns the
  /// R600 reg class that is equivalent to the given AMDIL reg class.
  virtual const TargetRegisterClass * getISARegClass(
    const TargetRegisterClass * rc) const;

  /// getHWRegChan - get the HW encoding for a register's channel.
  unsigned getHWRegChan(unsigned reg) const;

  /// getCFGStructurizerRegClass - get the register class of the specified
  /// type to use in the CFGStructurizer
  virtual const TargetRegisterClass * getCFGStructurizerRegClass(MVT VT) const;

private:
  /// getHWRegChanGen - Generated function returns a register's channel
  /// encoding.
  unsigned getHWRegChanGen(unsigned reg) const;
};

} // End namespace llvm

#endif // AMDIDSAREGISTERINFO_H_
