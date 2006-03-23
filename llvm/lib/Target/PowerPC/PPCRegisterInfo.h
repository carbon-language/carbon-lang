//===- PPCRegisterInfo.h - PowerPC Register Information Impl -----*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PowerPC implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef POWERPC32_REGISTERINFO_H
#define POWERPC32_REGISTERINFO_H

#include "PPC.h"
#include "PPCGenRegisterInfo.h.inc"
#include <map>

namespace llvm {

class Type;

class PPCRegisterInfo : public PPCGenRegisterInfo {
  std::map<unsigned, unsigned> ImmToIdxMap;
public:
  PPCRegisterInfo();

  /// Code Generation virtual methods...
  void storeRegToStackSlot(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI,
                           unsigned SrcReg, int FrameIndex,
                           const TargetRegisterClass *RC) const;

  void loadRegFromStackSlot(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MBBI,
                            unsigned DestReg, int FrameIndex,
                            const TargetRegisterClass *RC) const;

  void copyRegToReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                    unsigned DestReg, unsigned SrcReg,
                    const TargetRegisterClass *RC) const;

  /// foldMemoryOperand - PowerPC (like most RISC's) can only fold spills into
  /// copy instructions, turning them into load/store instructions.
  virtual MachineInstr* foldMemoryOperand(MachineInstr* MI, unsigned OpNum,
                                          int FrameIndex) const;
  
  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  void eliminateFrameIndex(MachineBasicBlock::iterator II) const;

  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

  void getLocation(MachineFunction &MF, unsigned Index,
                   MachineLocation &ML) const;
};

} // end namespace llvm

#endif
