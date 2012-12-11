//===-- AMDGPURegisterInfo.h - AMDGPURegisterInfo Interface -*- C++ -*-----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief TargetRegisterInfo interface that is implemented by all hw codegen
/// targets.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGPUREGISTERINFO_H
#define AMDGPUREGISTERINFO_H

#include "llvm/ADT/BitVector.h"
#include "llvm/Target/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#define GET_REGINFO_ENUM
#include "AMDGPUGenRegisterInfo.inc"

namespace llvm {

class AMDGPUTargetMachine;
class TargetInstrInfo;

struct AMDGPURegisterInfo : public AMDGPUGenRegisterInfo {
  TargetMachine &TM;
  const TargetInstrInfo &TII;
  static const uint16_t CalleeSavedReg;

  AMDGPURegisterInfo(TargetMachine &tm, const TargetInstrInfo &tii);

  virtual BitVector getReservedRegs(const MachineFunction &MF) const {
    assert(!"Unimplemented");  return BitVector();
  }

  /// \param RC is an AMDIL reg class.
  ///
  /// \returns The ISA reg class that is equivalent to \p RC.
  virtual const TargetRegisterClass * getISARegClass(
                                         const TargetRegisterClass * RC) const {
    assert(!"Unimplemented"); return NULL;
  }

  virtual const TargetRegisterClass* getCFGStructurizerRegClass(MVT VT) const {
    assert(!"Unimplemented"); return NULL;
  }

  const uint16_t* getCalleeSavedRegs(const MachineFunction *MF) const;
  void eliminateFrameIndex(MachineBasicBlock::iterator MI, int SPAdj,
                           RegScavenger *RS) const;
  unsigned getFrameRegister(const MachineFunction &MF) const;

};

} // End namespace llvm

#endif // AMDIDSAREGISTERINFO_H
