//===-- Mips16InstrInfo.h - Mips16 Instruction Information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Mips16 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef MIPS16INSTRUCTIONINFO_H
#define MIPS16INSTRUCTIONINFO_H

#include "Mips16RegisterInfo.h"
#include "MipsInstrInfo.h"

namespace llvm {

class Mips16InstrInfo : public MipsInstrInfo {
  const Mips16RegisterInfo RI;

public:
  explicit Mips16InstrInfo(MipsTargetMachine &TM);

  virtual const MipsRegisterInfo &getRegisterInfo() const;

  /// isLoadFromStackSlot - If the specified machine instruction is a direct
  /// load from a stack slot, return the virtual or physical register number of
  /// the destination along with the FrameIndex of the loaded stack slot.  If
  /// not, return 0.  This predicate must return 0 if the instruction has
  /// any side effects other than loading from the stack slot.
  virtual unsigned isLoadFromStackSlot(const MachineInstr *MI,
                                       int &FrameIndex) const;

  /// isStoreToStackSlot - If the specified machine instruction is a direct
  /// store to a stack slot, return the virtual or physical register number of
  /// the source reg along with the FrameIndex of the loaded stack slot.  If
  /// not, return 0.  This predicate must return 0 if the instruction has
  /// any side effects other than storing to the stack slot.
  virtual unsigned isStoreToStackSlot(const MachineInstr *MI,
                                      int &FrameIndex) const;

  virtual void copyPhysReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MI, DebugLoc DL,
                           unsigned DestReg, unsigned SrcReg,
                           bool KillSrc) const;

  virtual void storeRegToStack(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator MBBI,
                               unsigned SrcReg, bool isKill, int FrameIndex,
                               const TargetRegisterClass *RC,
                               const TargetRegisterInfo *TRI,
                               int64_t Offset) const;

  virtual void loadRegFromStack(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MBBI,
                                unsigned DestReg, int FrameIndex,
                                const TargetRegisterClass *RC,
                                const TargetRegisterInfo *TRI,
                                int64_t Offset) const;

  virtual bool expandPostRAPseudo(MachineBasicBlock::iterator MI) const;

  virtual unsigned getOppositeBranchOpc(unsigned Opc) const;

  // Adjust SP by FrameSize bytes. Save RA, S0, S1
  void makeFrame(unsigned SP, int64_t FrameSize, MachineBasicBlock &MBB,
                 MachineBasicBlock::iterator I) const;

  // Adjust SP by FrameSize bytes. Restore RA, S0, S1
  void restoreFrame(unsigned SP, int64_t FrameSize, MachineBasicBlock &MBB,
                      MachineBasicBlock::iterator I) const;


  /// Adjust SP by Amount bytes.
  void adjustStackPtr(unsigned SP, int64_t Amount, MachineBasicBlock &MBB,
                      MachineBasicBlock::iterator I) const;

  /// Emit a series of instructions to load an immediate.
  // This is to adjust some FrameReg. We return the new register to be used
  // in place of FrameReg and the adjusted immediate field (&NewImm)
  //
  unsigned loadImmediate(unsigned FrameReg,
                         int64_t Imm, MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator II, DebugLoc DL,
                         unsigned &NewImm) const;

  unsigned basicLoadImmediate(unsigned FrameReg,
                              int64_t Imm, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator II, DebugLoc DL,
                              unsigned &NewImm) const;

  static bool validImmediate(unsigned Opcode, unsigned Reg, int64_t Amount);

  static bool validSpImm8(int offset) {
    return ((offset & 7) == 0) && isInt<11>(offset);
  }

  //
  // build the proper one based on the Imm field
  //

  const MCInstrDesc& AddiuSpImm(int64_t Imm) const;

  void BuildAddiuSpImm
    (MachineBasicBlock &MBB, MachineBasicBlock::iterator I, int64_t Imm) const;

  unsigned getInlineAsmLength(const char *Str,
                              const MCAsmInfo &MAI) const;
private:
  virtual unsigned getAnalyzableBrOpc(unsigned Opc) const;

  void ExpandRetRA16(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                   unsigned Opc) const;

  // Adjust SP by Amount bytes where bytes can be up to 32bit number.
  void adjustStackPtrBig(unsigned SP, int64_t Amount, MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator I,
                         unsigned Reg1, unsigned Reg2) const;

  // Adjust SP by Amount bytes where bytes can be up to 32bit number.
  void adjustStackPtrBigUnrestricted(unsigned SP, int64_t Amount,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

};

}

#endif
