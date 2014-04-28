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
#include "llvm/ADT/DenseMap.h"

#define GET_REGINFO_HEADER
#include "PPCGenRegisterInfo.inc"

namespace llvm {
class PPCSubtarget;
class TargetInstrInfo;
class Type;

class PPCRegisterInfo : public PPCGenRegisterInfo {
  DenseMap<unsigned, unsigned> ImmToIdxMap;
  const PPCSubtarget &Subtarget;
public:
  PPCRegisterInfo(const PPCSubtarget &SubTarget);
  
  /// getPointerRegClass - Return the register class to use to hold pointers.
  /// This is used for addressing modes.
  virtual const TargetRegisterClass *
  getPointerRegClass(const MachineFunction &MF, unsigned Kind=0) const;

  unsigned getRegPressureLimit(const TargetRegisterClass *RC,
                               MachineFunction &MF) const;

  const TargetRegisterClass*
  getLargestLegalSuperClass(const TargetRegisterClass *RC) const;

  /// Code Generation virtual methods...
  const MCPhysReg *getCalleeSavedRegs(const MachineFunction* MF =nullptr) const;
  const uint32_t *getCallPreservedMask(CallingConv::ID CC) const;
  const uint32_t *getNoPreservedMask() const;

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

  virtual bool requiresVirtualBaseRegisters(const MachineFunction &MF) const {
    return true;
  }

  void lowerDynamicAlloc(MachineBasicBlock::iterator II) const;
  void lowerCRSpilling(MachineBasicBlock::iterator II,
                       unsigned FrameIndex) const;
  void lowerCRRestore(MachineBasicBlock::iterator II,
                      unsigned FrameIndex) const;
  void lowerCRBitSpilling(MachineBasicBlock::iterator II,
                          unsigned FrameIndex) const;
  void lowerCRBitRestore(MachineBasicBlock::iterator II,
                         unsigned FrameIndex) const;
  void lowerVRSAVESpilling(MachineBasicBlock::iterator II,
                           unsigned FrameIndex) const;
  void lowerVRSAVERestore(MachineBasicBlock::iterator II,
                          unsigned FrameIndex) const;

  bool hasReservedSpillSlot(const MachineFunction &MF, unsigned Reg,
			    int &FrameIdx) const;
  void eliminateFrameIndex(MachineBasicBlock::iterator II,
                           int SPAdj, unsigned FIOperandNum,
                           RegScavenger *RS = nullptr) const;

  // Support for virtual base registers.
  bool needsFrameBaseReg(MachineInstr *MI, int64_t Offset) const;
  void materializeFrameBaseRegister(MachineBasicBlock *MBB,
                                    unsigned BaseReg, int FrameIdx,
                                    int64_t Offset) const;
  void resolveFrameIndex(MachineInstr &MI, unsigned BaseReg,
                         int64_t Offset) const;
  bool isFrameOffsetLegal(const MachineInstr *MI, int64_t Offset) const;

  // Debug information queries.
  unsigned getFrameRegister(const MachineFunction &MF) const;

  // Base pointer (stack realignment) support.
  unsigned getBaseRegister(const MachineFunction &MF) const;
  bool hasBasePointer(const MachineFunction &MF) const;
  bool canRealignStack(const MachineFunction &MF) const;
  bool needsStackRealignment(const MachineFunction &MF) const;
};

} // end namespace llvm

#endif
