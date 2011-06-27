//===- MBlazeRegisterInfo.h - MBlaze Register Information Impl --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MBlaze implementation of the TargetRegisterInfo
// class.
//
//===----------------------------------------------------------------------===//

#ifndef MBLAZEREGISTERINFO_H
#define MBLAZEREGISTERINFO_H

#include "MBlaze.h"
#include "llvm/Target/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "MBlazeGenRegisterInfo.inc"

namespace llvm {
class MBlazeSubtarget;
class TargetInstrInfo;
class Type;

namespace MBlaze {
  /// SubregIndex - The index of various sized subregister classes. Note that
  /// these indices must be kept in sync with the class indices in the
  /// MBlazeRegisterInfo.td file.
  enum SubregIndex {
    SUBREG_FPEVEN = 1, SUBREG_FPODD = 2
  };
}

struct MBlazeRegisterInfo : public MBlazeGenRegisterInfo {
  const MBlazeSubtarget &Subtarget;
  const TargetInstrInfo &TII;

  MBlazeRegisterInfo(const MBlazeSubtarget &Subtarget,
                     const TargetInstrInfo &tii);

  /// getRegisterNumbering - Given the enum value for some register, e.g.
  /// MBlaze::RA, return the number that it corresponds to (e.g. 31).
  static unsigned getRegisterNumbering(unsigned RegEnum);
  static unsigned getRegisterFromNumbering(unsigned RegEnum);
  static unsigned getSpecialRegisterFromNumbering(unsigned RegEnum);
  static bool isRegister(unsigned RegEnum);
  static bool isSpecialRegister(unsigned RegEnum);

  /// Get PIC indirect call register
  static unsigned getPICCallReg();

  /// Code Generation virtual methods...
  const unsigned *getCalleeSavedRegs(const MachineFunction* MF = 0) const;

  BitVector getReservedRegs(const MachineFunction &MF) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  /// Stack Frame Processing Methods
  void eliminateFrameIndex(MachineBasicBlock::iterator II,
                           int SPAdj, RegScavenger *RS = NULL) const;

  void processFunctionBeforeFrameFinalized(MachineFunction &MF) const;

  /// Debug information queries.
  unsigned getRARegister() const;
  unsigned getFrameRegister(const MachineFunction &MF) const;

  /// Exception handling queries.
  unsigned getEHExceptionRegister() const;
  unsigned getEHHandlerRegister() const;

  int getDwarfRegNum(unsigned RegNum, bool isEH) const;
  int getLLVMRegNum(unsigned RegNum, bool isEH) const;
};

} // end namespace llvm

#endif
