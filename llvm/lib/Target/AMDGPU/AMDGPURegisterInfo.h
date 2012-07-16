//===-- AMDGPURegisterInfo.h - AMDGPURegisterInfo Interface -*- C++ -*-----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the TargetRegisterInfo interface that is implemented
// by all hw codegen targets.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGPUREGISTERINFO_H_
#define AMDGPUREGISTERINFO_H_

#include "AMDILRegisterInfo.h"

namespace llvm {

class AMDGPUTargetMachine;
class TargetInstrInfo;

struct AMDGPURegisterInfo : public AMDILRegisterInfo
{
  AMDGPUTargetMachine &TM;
  const TargetInstrInfo &TII;

  AMDGPURegisterInfo(AMDGPUTargetMachine &tm, const TargetInstrInfo &tii);

  virtual BitVector getReservedRegs(const MachineFunction &MF) const = 0;

  /// getISARegClass - rc is an AMDIL reg class.  This function returns the
  /// ISA reg class that is equivalent to the given AMDIL reg class.
  virtual const TargetRegisterClass *
    getISARegClass(const TargetRegisterClass * rc) const = 0;
};

} // End namespace llvm

#endif // AMDIDSAREGISTERINFO_H_
