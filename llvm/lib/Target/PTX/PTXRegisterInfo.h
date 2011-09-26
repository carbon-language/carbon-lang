//===- PTXRegisterInfo.h - PTX Register Information Impl --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PTX implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef PTX_REGISTER_INFO_H
#define PTX_REGISTER_INFO_H

#include "llvm/Support/ErrorHandling.h"
#include "llvm/ADT/BitVector.h"

#define GET_REGINFO_HEADER
#include "PTXGenRegisterInfo.inc"

namespace llvm {
class PTXTargetMachine;
class MachineFunction;

struct PTXRegisterInfo : public PTXGenRegisterInfo {
private:
  const TargetInstrInfo &TII;

public:
  PTXRegisterInfo(PTXTargetMachine &TM,
                  const TargetInstrInfo &tii);

  virtual const unsigned
    *getCalleeSavedRegs(const MachineFunction *MF = 0) const {
    static const unsigned CalleeSavedRegs[] = { 0 };
    return CalleeSavedRegs; // save nothing
  }

  virtual BitVector getReservedRegs(const MachineFunction &MF) const {
    BitVector Reserved(getNumRegs());
    return Reserved; // reserve no regs
  }

  virtual void eliminateFrameIndex(MachineBasicBlock::iterator II,
                                   int SPAdj,
                                   RegScavenger *RS = NULL) const;

  virtual unsigned getFrameRegister(const MachineFunction &MF) const {
    llvm_unreachable("PTX does not have a frame register");
    return 0;
  }
}; // struct PTXRegisterInfo
} // namespace llvm

#endif // PTX_REGISTER_INFO_H
