//===- SystemZInstrInfo.h - SystemZ Instruction Information -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SystemZ implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_SYSTEMZINSTRINFO_H
#define LLVM_TARGET_SYSTEMZINSTRINFO_H

#include "llvm/Target/TargetInstrInfo.h"
#include "SystemZRegisterInfo.h"

namespace llvm {

class SystemZTargetMachine;

class SystemZInstrInfo : public TargetInstrInfoImpl {
  const SystemZRegisterInfo RI;
  SystemZTargetMachine &TM;
public:
  explicit SystemZInstrInfo(SystemZTargetMachine &TM);

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  virtual const TargetRegisterInfo &getRegisterInfo() const { return RI; }

  bool copyRegToReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                    unsigned DestReg, unsigned SrcReg,
                    const TargetRegisterClass *DestRC,
                    const TargetRegisterClass *SrcRC) const;

  bool isMoveInstr(const MachineInstr& MI,
                   unsigned &SrcReg, unsigned &DstReg,
                   unsigned &SrcSubIdx, unsigned &DstSubIdx) const;

  virtual void storeRegToStackSlot(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   unsigned SrcReg, bool isKill,
                                   int FrameIndex,
                                   const TargetRegisterClass *RC) const;
  virtual void loadRegFromStackSlot(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MI,
                                    unsigned DestReg, int FrameIdx,
                                    const TargetRegisterClass *RC) const;

  virtual bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MI,
                                 const std::vector<CalleeSavedInfo> &CSI) const;
  virtual bool restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator MI,
                                 const std::vector<CalleeSavedInfo> &CSI) const;

  virtual unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                                MachineBasicBlock *FBB,
                             const SmallVectorImpl<MachineOperand> &Cond) const;

};

}

#endif
