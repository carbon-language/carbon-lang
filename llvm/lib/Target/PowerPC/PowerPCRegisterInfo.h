//===- PowerPCRegisterInfo.h - PowerPC Register Information Impl -*- C++ -*-==//
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

#ifndef POWERPCREGISTERINFO_H
#define POWERPCREGISTERINFO_H

#include "llvm/Target/MRegisterInfo.h"
#include "PowerPCGenRegisterInfo.h.inc"

namespace llvm {

class Type;

struct PowerPCRegisterInfo : public PowerPCGenRegisterInfo {
  PowerPCRegisterInfo();
  const TargetRegisterClass* getRegClassForType(const Type* Ty) const;

  /// Code Generation virtual methods...
  int storeRegToStackSlot(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator MBBI,
                          unsigned SrcReg, int FrameIndex,
                          const TargetRegisterClass *RC) const;

  int loadRegFromStackSlot(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI,
                           unsigned DestReg, int FrameIndex,
                           const TargetRegisterClass *RC) const;
  
  int copyRegToReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
		   unsigned DestReg, unsigned SrcReg,
		   const TargetRegisterClass *RC) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  void eliminateFrameIndex(MachineFunction &MF,
                           MachineBasicBlock::iterator II) const;

  void processFunctionBeforeFrameFinalized(MachineFunction &MF) const;

  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;
};

} // end namespace llvm

#endif
