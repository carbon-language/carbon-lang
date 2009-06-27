//===- ARMRegisterInfo.h - ARM Register Information Impl --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef ARMREGISTERINFO_H
#define ARMREGISTERINFO_H

#include "ARM.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "ARMGenRegisterInfo.h.inc"

namespace llvm {
  class ARMSubtarget;
  class TargetInstrInfo;
  class Type;

/// Register allocation hints.
namespace ARMRI {
  enum {
    RegPairOdd  = 1,
    RegPairEven = 2
  };
}

/// isARMLowRegister - Returns true if the register is low register r0-r7.
///
static inline bool isARMLowRegister(unsigned Reg) {
  using namespace ARM;
  switch (Reg) {
  case R0:  case R1:  case R2:  case R3:
  case R4:  case R5:  case R6:  case R7:
    return true;
  default:
    return false;
  }
}

struct ARMBaseRegisterInfo : public ARMGenRegisterInfo {
protected:
  const TargetInstrInfo &TII;
  const ARMSubtarget &STI;

  /// FramePtr - ARM physical register used as frame ptr.
  unsigned FramePtr;
public:
  ARMBaseRegisterInfo(const TargetInstrInfo &tii, const ARMSubtarget &STI);

  /// getRegisterNumbering - Given the enum value for some register, e.g.
  /// ARM::LR, return the number that it corresponds to (e.g. 14).
  static unsigned getRegisterNumbering(unsigned RegEnum);

  /// Same as previous getRegisterNumbering except it returns true in isSPVFP
  /// if the register is a single precision VFP register.
  static unsigned getRegisterNumbering(unsigned RegEnum, bool &isSPVFP);

  /// Code Generation virtual methods...
  const unsigned *getCalleeSavedRegs(const MachineFunction *MF = 0) const;

  const TargetRegisterClass* const*
  getCalleeSavedRegClasses(const MachineFunction *MF = 0) const;

  BitVector getReservedRegs(const MachineFunction &MF) const;

  bool isReservedReg(const MachineFunction &MF, unsigned Reg) const;

  const TargetRegisterClass *getPointerRegClass() const;

  std::pair<TargetRegisterClass::iterator,TargetRegisterClass::iterator>
  getAllocationOrder(const TargetRegisterClass *RC,
                     unsigned HintType, unsigned HintReg,
                     const MachineFunction &MF) const;

  unsigned ResolveRegAllocHint(unsigned Type, unsigned Reg,
                               const MachineFunction &MF) const;

  void UpdateRegAllocHint(unsigned Reg, unsigned NewReg,
                          MachineFunction &MF) const;

  bool hasFP(const MachineFunction &MF) const;

  void processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                            RegScavenger *RS = NULL) const;

  // Debug information queries.
  unsigned getRARegister() const;
  unsigned getFrameRegister(MachineFunction &MF) const;

  // Exception handling queries.
  unsigned getEHExceptionRegister() const;
  unsigned getEHHandlerRegister() const;

  int getDwarfRegNum(unsigned RegNum, bool isEH) const;

  bool isLowRegister(unsigned Reg) const;

private:
  unsigned getRegisterPairEven(unsigned Reg, const MachineFunction &MF) const;

  unsigned getRegisterPairOdd(unsigned Reg, const MachineFunction &MF) const;
};

struct ARMRegisterInfo : public ARMBaseRegisterInfo {
public:
  ARMRegisterInfo(const TargetInstrInfo &tii, const ARMSubtarget &STI);

  /// emitLoadConstPool - Emits a load from constpool to materialize the
  /// specified immediate.
  void emitLoadConstPool(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator &MBBI,
                         const TargetInstrInfo *TII, DebugLoc dl,
                         unsigned DestReg, int Val,
                         ARMCC::CondCodes Pred = ARMCC::AL,
                         unsigned PredReg = 0) const;

  /// Code Generation virtual methods...
  bool isReservedReg(const MachineFunction &MF, unsigned Reg) const;

  bool requiresRegisterScavenging(const MachineFunction &MF) const;

  bool hasReservedCallFrame(MachineFunction &MF) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  void eliminateFrameIndex(MachineBasicBlock::iterator II,
                           int SPAdj, RegScavenger *RS = NULL) const;

  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;
};

} // end namespace llvm

#endif
