//===- PPCRegisterInfo.h - PowerPC Register Information Impl -----*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PowerPC implementation of the TargetRegisterInfo
// class.
//
//===----------------------------------------------------------------------===//

#ifndef POWERPC32_REGISTERINFO_H
#define POWERPC32_REGISTERINFO_H

#include "PPC.h"
#include "PPCGenRegisterInfo.h.inc"
#include <map>

namespace llvm {
class PPCSubtarget;
class TargetInstrInfo;
class Type;

class PPCRegisterInfo : public PPCGenRegisterInfo {
  std::map<unsigned, unsigned> ImmToIdxMap;
  const PPCSubtarget &Subtarget;
  const TargetInstrInfo &TII;
public:
  PPCRegisterInfo(const PPCSubtarget &SubTarget, const TargetInstrInfo &tii);
  
  /// getRegisterNumbering - Given the enum value for some register, e.g.
  /// PPC::F14, return the number that it corresponds to (e.g. 14).
  static unsigned getRegisterNumbering(unsigned RegEnum);

  /// getPointerRegClass - Return the register class to use to hold pointers.
  /// This is used for addressing modes.
  virtual const TargetRegisterClass *getPointerRegClass(unsigned Kind=0) const;  

  /// Code Generation virtual methods...
  const unsigned *getCalleeSavedRegs(const MachineFunction* MF = 0) const;

  BitVector getReservedRegs(const MachineFunction &MF) const;

  /// requiresRegisterScavenging - We require a register scavenger.
  /// FIXME (64-bit): Should be inlined.
  bool requiresRegisterScavenging(const MachineFunction &MF) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  void lowerDynamicAlloc(MachineBasicBlock::iterator II,
                         int SPAdj, RegScavenger *RS) const;
  void lowerCRSpilling(MachineBasicBlock::iterator II, unsigned FrameIndex,
                       int SPAdj, RegScavenger *RS) const;
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
};

} // end namespace llvm

#endif
