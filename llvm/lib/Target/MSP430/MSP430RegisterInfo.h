//===- MSP430RegisterInfo.h - MSP430 Register Information Impl --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MSP430 implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_MSP430REGISTERINFO_H
#define LLVM_TARGET_MSP430REGISTERINFO_H

#include "llvm/Target/TargetRegisterInfo.h"
#include "MSP430GenRegisterInfo.h.inc"

namespace llvm {

class TargetInstrInfo;
class MSP430TargetMachine;

struct MSP430RegisterInfo : public MSP430GenRegisterInfo {
private:
  MSP430TargetMachine &TM;
  const TargetInstrInfo &TII;

  /// StackAlign - Default stack alignment.
  ///
  unsigned StackAlign;
public:
  MSP430RegisterInfo(MSP430TargetMachine &tm, const TargetInstrInfo &tii);

  /// Code Generation virtual methods...
  const unsigned *getCalleeSavedRegs(const MachineFunction *MF = 0) const;

  const TargetRegisterClass* const*
    getCalleeSavedRegClasses(const MachineFunction *MF = 0) const;

  BitVector getReservedRegs(const MachineFunction &MF) const;
  const TargetRegisterClass* getPointerRegClass(unsigned Kind = 0) const;

  bool hasFP(const MachineFunction &MF) const;
  bool hasReservedCallFrame(MachineFunction &MF) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  unsigned eliminateFrameIndex(MachineBasicBlock::iterator II,
                               int SPAdj, int *Value = NULL,
                               RegScavenger *RS = NULL) const;

  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

  void processFunctionBeforeFrameFinalized(MachineFunction &MF) const;

  // Debug information queries.
  unsigned getRARegister() const;
  unsigned getFrameRegister(const MachineFunction &MF) const;

  //! Get DWARF debugging register number
  int getDwarfRegNum(unsigned RegNum, bool isEH) const;
};

} // end namespace llvm

#endif // LLVM_TARGET_MSP430REGISTERINFO_H
