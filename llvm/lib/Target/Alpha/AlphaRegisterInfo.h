//===- AlphaRegisterInfo.h - Alpha Register Information Impl ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Alpha implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef ALPHAREGISTERINFO_H
#define ALPHAREGISTERINFO_H

#include "llvm/Target/TargetRegisterInfo.h"
#include "AlphaGenRegisterInfo.h.inc"

namespace llvm {

class TargetInstrInfo;
class Type;

struct AlphaRegisterInfo : public AlphaGenRegisterInfo {
  const TargetInstrInfo &TII;

  AlphaRegisterInfo(const TargetInstrInfo &tii);

  /// Code Generation virtual methods...
  const unsigned *getCalleeSavedRegs(const MachineFunction *MF = 0) const;

  BitVector getReservedRegs(const MachineFunction &MF) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  void eliminateFrameIndex(MachineBasicBlock::iterator II,
                           int SPAdj, RegScavenger *RS = NULL) const;

  // Debug information queries.
  unsigned getRARegister() const;
  unsigned getFrameRegister(const MachineFunction &MF) const;

  // Exception handling queries.
  unsigned getEHExceptionRegister() const;
  unsigned getEHHandlerRegister() const;

  int getDwarfRegNum(unsigned RegNum, bool isEH) const;
  int getLLVMRegNum(unsigned RegNum, bool isEH) const;

  static std::string getPrettyName(unsigned reg);
};

} // end namespace llvm

#endif
