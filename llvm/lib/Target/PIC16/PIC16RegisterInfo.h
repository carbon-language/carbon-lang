//===- PIC16RegisterInfo.h - PIC16 Register Information Impl ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PIC16 implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef PIC16REGISTERINFO_H
#define PIC16REGISTERINFO_H

#include "PIC16GenRegisterInfo.h.inc"
#include "llvm/Target/TargetRegisterInfo.h"

namespace llvm {

// Forward Declarations.
class TargetInstrInfo;
class Type;

struct PIC16RegisterInfo : public PIC16GenRegisterInfo {
  const TargetInstrInfo &TII;
  
  explicit PIC16RegisterInfo(const TargetInstrInfo &tii);

  /// getRegisterNumbering - Given the enum value for some register, e.g.
  /// PIC16::RA, return the number that it corresponds to (e.g. 31).
  static unsigned getRegisterNumbering(unsigned RegEnum);

  void reMaterialize(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                     unsigned DestReg, const MachineInstr *Orig) const;

  MachineInstr* foldMemoryOperand(MachineInstr* MI, unsigned OpNum,
                                  int FrameIndex) const;

  MachineInstr* foldMemoryOperand(MachineInstr* MI, unsigned OpNum,
                                  MachineInstr* LoadMI) const {
    return 0;
  }

  void copyRegToReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
          	    unsigned DestReg, unsigned SrcReg,
          	    const TargetRegisterClass *RC) const;
  

  const unsigned *getCalleeSavedRegs(const MachineFunction* MF = 0) const;

  const TargetRegisterClass* const*
  getCalleeSavedRegClasses(const MachineFunction* MF = 0) const;

  BitVector getReservedRegs(const MachineFunction &MF) const;

  bool hasFP(const MachineFunction &MF) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  /// Stack Frame Processing Methods.
  void eliminateFrameIndex(MachineBasicBlock::iterator II,
                           int SPAdj, RegScavenger *RS = NULL) const;

  void processFunctionBeforeFrameFinalized(MachineFunction &MF) const;

  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;
  
  /// Debug information queries.
  unsigned getRARegister() const;
  unsigned getFrameRegister(MachineFunction &MF) const;

  /// Exception handling queries.
  unsigned getEHExceptionRegister() const;
  unsigned getEHHandlerRegister() const;

  int getDwarfRegNum(unsigned RegNum, bool isEH) const;
};

} // end namespace llvm

#endif
