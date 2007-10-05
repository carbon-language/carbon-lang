//===- MipsRegisterInfo.h - Mips Register Information Impl ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bruno Cardoso Lopes and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Mips implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSREGISTERINFO_H
#define MIPSREGISTERINFO_H

#include "llvm/Target/MRegisterInfo.h"
#include "MipsGenRegisterInfo.h.inc"

namespace llvm {

class TargetInstrInfo;
class Type;

struct MipsRegisterInfo : public MipsGenRegisterInfo {
  const TargetInstrInfo &TII;
  
  MipsRegisterInfo(const TargetInstrInfo &tii);

  /// getRegisterNumbering - Given the enum value for some register, e.g.
  /// Mips::RA, return the number that it corresponds to (e.g. 31).
  static unsigned getRegisterNumbering(unsigned RegEnum);

  /// Code Generation virtual methods...
  void storeRegToStackSlot(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI,
                           unsigned SrcReg, int FrameIndex,
                           const TargetRegisterClass *RC) const;

  void storeRegToAddr(MachineFunction &MF, unsigned SrcReg,
                      SmallVector<MachineOperand,4> Addr,
                      const TargetRegisterClass *RC,
                      SmallVector<MachineInstr*, 4> &NewMIs) const;

  void loadRegFromStackSlot(MachineBasicBlock &MBB,
              MachineBasicBlock::iterator MBBI,
              unsigned DestReg, int FrameIndex,
              const TargetRegisterClass *RC) const;

  void loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                       SmallVector<MachineOperand,4> Addr,
                       const TargetRegisterClass *RC,
                       SmallVector<MachineInstr*, 4> &NewMIs) const;

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
          const TargetRegisterClass *DestRC,
          const TargetRegisterClass *SrcRC) const;
  

  const unsigned *getCalleeSavedRegs(const MachineFunction* MF = 0) const;

  const TargetRegisterClass* const*
  getCalleeSavedRegClasses(const MachineFunction* MF = 0) const;

  BitVector getReservedRegs(const MachineFunction &MF) const;

  bool hasFP(const MachineFunction &MF) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  /// Stack Frame Processing Methods
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
};

} // end namespace llvm

#endif
