//===-- SystemZRegisterInfo.h - SystemZ register information ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SystemZREGISTERINFO_H
#define SystemZREGISTERINFO_H

#include "SystemZ.h"
#include "llvm/Target/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "SystemZGenRegisterInfo.inc"

namespace llvm {

namespace SystemZ {
  // Return the subreg to use for referring to the even and odd registers
  // in a GR128 pair.  Is32Bit says whether we want a GR32 or GR64.
  inline unsigned even128(bool Is32bit) {
    return Is32bit ? subreg_hl32 : subreg_h64;
  }
  inline unsigned odd128(bool Is32bit) {
    return Is32bit ? subreg_l32 : subreg_l64;
  }
}

class SystemZSubtarget;
class SystemZInstrInfo;

struct SystemZRegisterInfo : public SystemZGenRegisterInfo {
private:
  SystemZTargetMachine &TM;

public:
  SystemZRegisterInfo(SystemZTargetMachine &tm);

  // Override TargetRegisterInfo.h.
  virtual bool requiresRegisterScavenging(const MachineFunction &MF) const
    LLVM_OVERRIDE {
    return true;
  }
  virtual bool requiresFrameIndexScavenging(const MachineFunction &MF) const
    LLVM_OVERRIDE {
    return true;
  }
  virtual bool trackLivenessAfterRegAlloc(const MachineFunction &MF) const
    LLVM_OVERRIDE {
    return true;
  }
  virtual const uint16_t *getCalleeSavedRegs(const MachineFunction *MF = 0)
    const LLVM_OVERRIDE;
  virtual BitVector getReservedRegs(const MachineFunction &MF)
    const LLVM_OVERRIDE;
  virtual void eliminateFrameIndex(MachineBasicBlock::iterator MI,
                                   int SPAdj, unsigned FIOperandNum,
                                   RegScavenger *RS) const LLVM_OVERRIDE;
  virtual unsigned getFrameRegister(const MachineFunction &MF) const
    LLVM_OVERRIDE;
};

} // end namespace llvm

#endif
