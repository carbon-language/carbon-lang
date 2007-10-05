//===- AlphaRegisterInfo.h - Alpha Register Information Impl ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Alpha implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef ALPHAREGISTERINFO_H
#define ALPHAREGISTERINFO_H

#include "llvm/Target/MRegisterInfo.h"
#include "AlphaGenRegisterInfo.h.inc"

namespace llvm {

class TargetInstrInfo;
class Type;

struct AlphaRegisterInfo : public AlphaGenRegisterInfo {
  const TargetInstrInfo &TII;

  AlphaRegisterInfo(const TargetInstrInfo &tii);

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

  MachineInstr* foldMemoryOperand(MachineInstr *MI, unsigned OpNum, 
                                  int FrameIndex) const;

  MachineInstr* foldMemoryOperand(MachineInstr* MI, unsigned OpNum,
                                  MachineInstr* LoadMI) const {
    return 0;
  }

  void copyRegToReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                    unsigned DestReg, unsigned SrcReg,
                    const TargetRegisterClass *DestRC,
                    const TargetRegisterClass *SrcRC) const;

  void reMaterialize(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                     unsigned DestReg, const MachineInstr *Orig) const;

  const unsigned *getCalleeSavedRegs(const MachineFunction *MF = 0) const;

  const TargetRegisterClass* const* getCalleeSavedRegClasses(
                                     const MachineFunction *MF = 0) const;

  BitVector getReservedRegs(const MachineFunction &MF) const;

  bool hasFP(const MachineFunction &MF) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  void eliminateFrameIndex(MachineBasicBlock::iterator II,
                           int SPAdj, RegScavenger *RS = NULL) const;

  //void processFunctionBeforeFrameFinalized(MachineFunction &MF) const;

  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

  // Debug information queries.
  unsigned getRARegister() const;
  unsigned getFrameRegister(MachineFunction &MF) const;

  // Exception handling queries.
  unsigned getEHExceptionRegister() const;
  unsigned getEHHandlerRegister() const;

  static std::string getPrettyName(unsigned reg);
};

} // end namespace llvm

#endif
