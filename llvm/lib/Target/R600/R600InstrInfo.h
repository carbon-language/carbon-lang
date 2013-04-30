//===-- R600InstrInfo.h - R600 Instruction Info Interface -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Interface definition for R600InstrInfo
//
//===----------------------------------------------------------------------===//

#ifndef R600INSTRUCTIONINFO_H_
#define R600INSTRUCTIONINFO_H_

#include "AMDGPUInstrInfo.h"
#include "AMDIL.h"
#include "R600Defines.h"
#include "R600RegisterInfo.h"
#include <map>

namespace llvm {

  class AMDGPUTargetMachine;
  class DFAPacketizer;
  class ScheduleDAG;
  class MachineFunction;
  class MachineInstr;
  class MachineInstrBuilder;

  class R600InstrInfo : public AMDGPUInstrInfo {
  private:
  const R600RegisterInfo RI;
  const AMDGPUSubtarget &ST;

  int getBranchInstr(const MachineOperand &op) const;

  public:
  explicit R600InstrInfo(AMDGPUTargetMachine &tm);

  const R600RegisterInfo &getRegisterInfo() const;
  virtual void copyPhysReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MI, DebugLoc DL,
                           unsigned DestReg, unsigned SrcReg,
                           bool KillSrc) const;

  bool isTrig(const MachineInstr &MI) const;
  bool isPlaceHolderOpcode(unsigned opcode) const;
  bool isReductionOp(unsigned opcode) const;
  bool isCubeOp(unsigned opcode) const;

  /// \returns true if this \p Opcode represents an ALU instruction.
  bool isALUInstr(unsigned Opcode) const;

  bool isTransOnly(unsigned Opcode) const;
  bool isTransOnly(const MachineInstr *MI) const;

  bool usesVertexCache(unsigned Opcode) const;
  bool usesVertexCache(const MachineInstr *MI) const;
  bool usesTextureCache(unsigned Opcode) const;
  bool usesTextureCache(const MachineInstr *MI) const;

  bool fitsConstReadLimitations(const std::vector<unsigned>&) const;
  bool canBundle(const std::vector<MachineInstr *> &) const;

  /// \breif Vector instructions are instructions that must fill all
  /// instruction slots within an instruction group.
  bool isVector(const MachineInstr &MI) const;

  virtual MachineInstr * getMovImmInstr(MachineFunction *MF, unsigned DstReg,
                                        int64_t Imm) const;

  virtual unsigned getIEQOpcode() const;
  virtual bool isMov(unsigned Opcode) const;

  DFAPacketizer *CreateTargetScheduleState(const TargetMachine *TM,
                                           const ScheduleDAG *DAG) const;

  bool ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const;

  bool AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB, MachineBasicBlock *&FBB,
                     SmallVectorImpl<MachineOperand> &Cond, bool AllowModify) const;

  unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB, MachineBasicBlock *FBB, const SmallVectorImpl<MachineOperand> &Cond, DebugLoc DL) const;

  unsigned RemoveBranch(MachineBasicBlock &MBB) const;

  bool isPredicated(const MachineInstr *MI) const;

  bool isPredicable(MachineInstr *MI) const;

  bool
   isProfitableToDupForIfCvt(MachineBasicBlock &MBB, unsigned NumCyles,
                             const BranchProbability &Probability) const;

  bool isProfitableToIfCvt(MachineBasicBlock &MBB, unsigned NumCyles,
                           unsigned ExtraPredCycles,
                           const BranchProbability &Probability) const ;

  bool
   isProfitableToIfCvt(MachineBasicBlock &TMBB,
                       unsigned NumTCycles, unsigned ExtraTCycles,
                       MachineBasicBlock &FMBB,
                       unsigned NumFCycles, unsigned ExtraFCycles,
                       const BranchProbability &Probability) const;

  bool DefinesPredicate(MachineInstr *MI,
                                  std::vector<MachineOperand> &Pred) const;

  bool SubsumesPredicate(const SmallVectorImpl<MachineOperand> &Pred1,
                         const SmallVectorImpl<MachineOperand> &Pred2) const;

  bool isProfitableToUnpredicate(MachineBasicBlock &TMBB,
                                          MachineBasicBlock &FMBB) const;

  bool PredicateInstruction(MachineInstr *MI,
                        const SmallVectorImpl<MachineOperand> &Pred) const;

  unsigned int getInstrLatency(const InstrItineraryData *ItinData,
                               const MachineInstr *MI,
                               unsigned *PredCost = 0) const;

  virtual int getInstrLatency(const InstrItineraryData *ItinData,
                              SDNode *Node) const { return 1;}

  /// \returns a list of all the registers that may be accesed using indirect
  /// addressing.
  std::vector<unsigned> getIndirectReservedRegs(const MachineFunction &MF) const;

  virtual int getIndirectIndexBegin(const MachineFunction &MF) const;

  virtual int getIndirectIndexEnd(const MachineFunction &MF) const;


  virtual unsigned calculateIndirectAddress(unsigned RegIndex,
                                            unsigned Channel) const;

  virtual const TargetRegisterClass *getIndirectAddrStoreRegClass(
                                                      unsigned SourceReg) const;

  virtual const TargetRegisterClass *getIndirectAddrLoadRegClass() const;

  virtual MachineInstrBuilder buildIndirectWrite(MachineBasicBlock *MBB,
                                  MachineBasicBlock::iterator I,
                                  unsigned ValueReg, unsigned Address,
                                  unsigned OffsetReg) const;

  virtual MachineInstrBuilder buildIndirectRead(MachineBasicBlock *MBB,
                                  MachineBasicBlock::iterator I,
                                  unsigned ValueReg, unsigned Address,
                                  unsigned OffsetReg) const;

  virtual const TargetRegisterClass *getSuperIndirectRegClass() const;

  unsigned getMaxAlusPerClause() const;

  ///buildDefaultInstruction - This function returns a MachineInstr with
  /// all the instruction modifiers initialized to their default values.
  /// You can use this function to avoid manually specifying each instruction
  /// modifier operand when building a new instruction.
  ///
  /// \returns a MachineInstr with all the instruction modifiers initialized
  /// to their default values.
  MachineInstrBuilder buildDefaultInstruction(MachineBasicBlock &MBB,
                                              MachineBasicBlock::iterator I,
                                              unsigned Opcode,
                                              unsigned DstReg,
                                              unsigned Src0Reg,
                                              unsigned Src1Reg = 0) const;

  MachineInstr *buildMovImm(MachineBasicBlock &BB,
                                  MachineBasicBlock::iterator I,
                                  unsigned DstReg,
                                  uint64_t Imm) const;

  /// \brief Get the index of Op in the MachineInstr.
  ///
  /// \returns -1 if the Instruction does not contain the specified \p Op.
  int getOperandIdx(const MachineInstr &MI, R600Operands::Ops Op) const;

  /// \brief Get the index of \p Op for the given Opcode.
  ///
  /// \returns -1 if the Instruction does not contain the specified \p Op.
  int getOperandIdx(unsigned Opcode, R600Operands::Ops Op) const;

  /// \brief Helper function for setting instruction flag values.
  void setImmOperand(MachineInstr *MI, R600Operands::Ops Op, int64_t Imm) const;

  /// \returns true if this instruction has an operand for storing target flags.
  bool hasFlagOperand(const MachineInstr &MI) const;

  ///\brief Add one of the MO_FLAG* flags to the specified \p Operand.
  void addFlag(MachineInstr *MI, unsigned Operand, unsigned Flag) const;

  ///\brief Determine if the specified \p Flag is set on this \p Operand.
  bool isFlagSet(const MachineInstr &MI, unsigned Operand, unsigned Flag) const;

  /// \param SrcIdx The register source to set the flag on (e.g src0, src1, src2)
  /// \param Flag The flag being set.
  ///
  /// \returns the operand containing the flags for this instruction.
  MachineOperand &getFlagOp(MachineInstr *MI, unsigned SrcIdx = 0,
                            unsigned Flag = 0) const;

  /// \brief Clear the specified flag on the instruction.
  void clearFlag(MachineInstr *MI, unsigned Operand, unsigned Flag) const;
};

} // End llvm namespace

#endif // R600INSTRINFO_H_
