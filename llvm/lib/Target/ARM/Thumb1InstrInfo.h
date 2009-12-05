//===- Thumb1InstrInfo.h - Thumb-1 Instruction Information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Thumb-1 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef THUMB1INSTRUCTIONINFO_H
#define THUMB1INSTRUCTIONINFO_H

#include "llvm/Target/TargetInstrInfo.h"
#include "ARM.h"
#include "ARMInstrInfo.h"
#include "Thumb1RegisterInfo.h"

namespace llvm {
  class ARMSubtarget;

class Thumb1InstrInfo : public ARMBaseInstrInfo {
  Thumb1RegisterInfo RI;
public:
  explicit Thumb1InstrInfo(const ARMSubtarget &STI);

  // Return the non-pre/post incrementing version of 'Opc'. Return 0
  // if there is not such an opcode.
  unsigned getUnindexedOpcode(unsigned Opc) const;

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  const Thumb1RegisterInfo &getRegisterInfo() const { return RI; }

  bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MI,
                                 const std::vector<CalleeSavedInfo> &CSI) const;
  bool restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   const std::vector<CalleeSavedInfo> &CSI) const;

  bool copyRegToReg(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator I,
                            unsigned DestReg, unsigned SrcReg,
                            const TargetRegisterClass *DestRC,
                            const TargetRegisterClass *SrcRC) const;
  void storeRegToStackSlot(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MBBI,
                                   unsigned SrcReg, bool isKill, int FrameIndex,
                                   const TargetRegisterClass *RC) const;

  void loadRegFromStackSlot(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MBBI,
                                    unsigned DestReg, int FrameIndex,
                                    const TargetRegisterClass *RC) const;

  bool canFoldMemoryOperand(const MachineInstr *MI,
                                    const SmallVectorImpl<unsigned> &Ops) const;

  MachineInstr* foldMemoryOperandImpl(MachineFunction &MF,
                                      MachineInstr* MI,
                                      const SmallVectorImpl<unsigned> &Ops,
                                      int FrameIndex) const;

  MachineInstr* foldMemoryOperandImpl(MachineFunction &MF,
                                      MachineInstr* MI,
                                      const SmallVectorImpl<unsigned> &Ops,
                                      MachineInstr* LoadMI) const {
    return 0;
  }
};
}

#endif // THUMB1INSTRUCTIONINFO_H
