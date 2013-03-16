//===-- PPCRegisterInfo.h - PowerPC Register Information Impl ---*- C++ -*-===//
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
#include <map>

#define GET_REGINFO_HEADER
#include "PPCGenRegisterInfo.inc"

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
  
  /// getPointerRegClass - Return the register class to use to hold pointers.
  /// This is used for addressing modes.
  virtual const TargetRegisterClass *
  getPointerRegClass(const MachineFunction &MF, unsigned Kind=0) const;

  unsigned getRegPressureLimit(const TargetRegisterClass *RC,
                               MachineFunction &MF) const;

  /// Code Generation virtual methods...
  const uint16_t *getCalleeSavedRegs(const MachineFunction* MF = 0) const;
  const uint32_t *getCallPreservedMask(CallingConv::ID CC) const;

  BitVector getReservedRegs(const MachineFunction &MF) const;

  /// We require the register scavenger.
  bool requiresRegisterScavenging(const MachineFunction &MF) const {
    return true;
  }

  bool requiresFrameIndexScavenging(const MachineFunction &MF) const {
    return true;
  }

  bool trackLivenessAfterRegAlloc(const MachineFunction &MF) const {
    return true;
  }

  void lowerDynamicAlloc(MachineBasicBlock::iterator II,
                         int SPAdj, RegScavenger *RS) const;
  void lowerCRSpilling(MachineBasicBlock::iterator II, unsigned FrameIndex,
                       int SPAdj, RegScavenger *RS) const;
  void lowerCRRestore(MachineBasicBlock::iterator II, unsigned FrameIndex,
                       int SPAdj, RegScavenger *RS) const;
  bool hasReservedSpillSlot(const MachineFunction &MF, unsigned Reg,
			    int &FrameIdx) const;
  void eliminateFrameIndex(MachineBasicBlock::iterator II,
                           int SPAdj, unsigned FIOperandNum,
                           RegScavenger *RS = NULL) const;

  // Debug information queries.
  unsigned getFrameRegister(const MachineFunction &MF) const;

  // Exception handling queries.
  unsigned getEHExceptionRegister() const;
  unsigned getEHHandlerRegister() const;
};

} // end namespace llvm

#endif
