//===-- SIInstrInfo.h - SI Instruction Info Interface -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Interface definition for SIInstrInfo.
//
//===----------------------------------------------------------------------===//


#ifndef SIINSTRINFO_H
#define SIINSTRINFO_H

#include "AMDGPUInstrInfo.h"
#include "SIRegisterInfo.h"

namespace llvm {

class SIInstrInfo : public AMDGPUInstrInfo {
private:
  const SIRegisterInfo RI;

  MachineInstrBuilder buildIndirectIndexLoop(MachineBasicBlock &MBB,
                                             MachineBasicBlock::iterator I,
                                             unsigned OffsetVGPR,
                                             unsigned MovRelOp,
                                             unsigned Dst,
                                             unsigned Src0) const;
  // If you add or remove instructions from this function, you will

public:
  explicit SIInstrInfo(AMDGPUTargetMachine &tm);

  const SIRegisterInfo &getRegisterInfo() const;

  virtual void copyPhysReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MI, DebugLoc DL,
                           unsigned DestReg, unsigned SrcReg,
                           bool KillSrc) const;

  unsigned commuteOpcode(unsigned Opcode) const;

  virtual MachineInstr *commuteInstruction(MachineInstr *MI,
                                           bool NewMI=false) const;

  virtual unsigned getIEQOpcode() const { assert(!"Implement"); return 0;}
  MachineInstr *buildMovInstr(MachineBasicBlock *MBB,
                              MachineBasicBlock::iterator I,
                              unsigned DstReg, unsigned SrcReg) const;
  virtual bool isMov(unsigned Opcode) const;

  virtual bool isSafeToMoveRegClassDefs(const TargetRegisterClass *RC) const;
  int isMIMG(uint16_t Opcode) const;
  int isSMRD(uint16_t Opcode) const;
  bool isVOP1(uint16_t Opcode) const;
  bool isVOP2(uint16_t Opcode) const;
  bool isVOP3(uint16_t Opcode) const;
  bool isVOPC(uint16_t Opcode) const;
  bool isInlineConstant(const MachineOperand &MO) const;
  bool isLiteralConstant(const MachineOperand &MO) const;

  virtual bool verifyInstruction(const MachineInstr *MI,
                                 StringRef &ErrInfo) const;

  bool isSALUInstr(const MachineInstr &MI) const;
  static unsigned getVALUOp(const MachineInstr &MI);
  bool isSALUOpSupportedOnVALU(const MachineInstr &MI) const;

  /// \brief Return the correct register class for \p OpNo.  For target-specific
  /// instructions, this will return the register class that has been defined
  /// in tablegen.  For generic instructions, like REG_SEQUENCE it will return
  /// the register class of its machine operand.
  /// to infer the correct register class base on the other operands.
  const TargetRegisterClass *getOpRegClass(const MachineInstr &MI,
                                           unsigned OpNo) const;\

  /// \returns true if it is legal for the operand at index \p OpNo
  /// to read a VGPR.
  bool canReadVGPR(const MachineInstr &MI, unsigned OpNo) const;

  /// \brief Legalize the \p OpIndex operand of this instruction by inserting
  /// a MOV.  For example:
  /// ADD_I32_e32 VGPR0, 15
  /// to
  /// MOV VGPR1, 15
  /// ADD_I32_e32 VGPR0, VGPR1
  ///
  /// If the operand being legalized is a register, then a COPY will be used
  /// instead of MOV.
  void legalizeOpWithMove(MachineInstr *MI, unsigned OpIdx) const;

  /// \brief Legalize all operands in this instruction.  This function may
  /// create new instruction and insert them before \p MI.
  void legalizeOperands(MachineInstr *MI) const;

  /// \brief Replace this instruction's opcode with the equivalent VALU
  /// opcode.  This function will also move the users of \p MI to the
  /// VALU if necessary.
  void moveToVALU(MachineInstr &MI) const;

  virtual unsigned calculateIndirectAddress(unsigned RegIndex,
                                            unsigned Channel) const;

  virtual const TargetRegisterClass *getIndirectAddrRegClass() const;

  virtual MachineInstrBuilder buildIndirectWrite(MachineBasicBlock *MBB,
                                                 MachineBasicBlock::iterator I,
                                                 unsigned ValueReg,
                                                 unsigned Address,
                                                 unsigned OffsetReg) const;

  virtual MachineInstrBuilder buildIndirectRead(MachineBasicBlock *MBB,
                                                MachineBasicBlock::iterator I,
                                                unsigned ValueReg,
                                                unsigned Address,
                                                unsigned OffsetReg) const;
  void reserveIndirectRegisters(BitVector &Reserved,
                                const MachineFunction &MF) const;

  void LoadM0(MachineInstr *MoveRel, MachineBasicBlock::iterator I,
              unsigned SavReg, unsigned IndexReg) const;
};

namespace AMDGPU {

  int getVOPe64(uint16_t Opcode);
  int getCommuteRev(uint16_t Opcode);
  int getCommuteOrig(uint16_t Opcode);

} // End namespace AMDGPU

} // End namespace llvm

namespace SIInstrFlags {
  enum Flags {
    // First 4 bits are the instruction encoding
    VM_CNT = 1 << 0,
    EXP_CNT = 1 << 1,
    LGKM_CNT = 1 << 2
  };
}

#endif //SIINSTRINFO_H
