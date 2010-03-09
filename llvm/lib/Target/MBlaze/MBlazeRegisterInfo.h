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
#include "MBlazeGenRegisterInfo.h.inc"

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

  /// Get PIC indirect call register
  static unsigned getPICCallReg();

  /// Adjust the MBlaze stack frame.
  void adjustMBlazeStackFrame(MachineFunction &MF) const;

  /// Code Generation virtual methods...
  const unsigned *getCalleeSavedRegs(const MachineFunction* MF = 0) const;

  const TargetRegisterClass* const*
  getCalleeSavedRegClasses(const MachineFunction* MF = 0) const;

  BitVector getReservedRegs(const MachineFunction &MF) const;

  bool hasFP(const MachineFunction &MF) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  /// Stack Frame Processing Methods
  unsigned eliminateFrameIndex(MachineBasicBlock::iterator II,
                               int SPAdj, FrameIndexValue *Value = NULL,
                               RegScavenger *RS = NULL) const;

  void processFunctionBeforeFrameFinalized(MachineFunction &MF) const;

  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;
  
  /// Debug information queries.
  unsigned getRARegister() const;
  unsigned getFrameRegister(const MachineFunction &MF) const;

  /// Exception handling queries.
  unsigned getEHExceptionRegister() const;
  unsigned getEHHandlerRegister() const;

  /// targetHandlesStackFrameRounding - Returns true if the target is
  /// responsible for rounding up the stack frame (probably at emitPrologue
  /// time).
  bool targetHandlesStackFrameRounding() const { return true; }

  int getDwarfRegNum(unsigned RegNum, bool isEH) const;
};

} // end namespace llvm

#endif
