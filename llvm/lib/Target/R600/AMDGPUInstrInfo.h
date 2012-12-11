//===-- AMDGPUInstrInfo.h - AMDGPU Instruction Information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Contains the definition of a TargetInstrInfo class that is common
/// to all AMD GPUs.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGPUINSTRUCTIONINFO_H
#define AMDGPUINSTRUCTIONINFO_H

#include "AMDGPURegisterInfo.h"
#include "AMDGPUInstrInfo.h"
#include "llvm/Target/TargetInstrInfo.h"

#include <map>

#define GET_INSTRINFO_HEADER
#define GET_INSTRINFO_ENUM
#include "AMDGPUGenInstrInfo.inc"

#define OPCODE_IS_ZERO_INT AMDGPU::PRED_SETE_INT
#define OPCODE_IS_NOT_ZERO_INT AMDGPU::PRED_SETNE_INT
#define OPCODE_IS_ZERO AMDGPU::PRED_SETE
#define OPCODE_IS_NOT_ZERO AMDGPU::PRED_SETNE

namespace llvm {

class AMDGPUTargetMachine;
class MachineFunction;
class MachineInstr;
class MachineInstrBuilder;

class AMDGPUInstrInfo : public AMDGPUGenInstrInfo {
private:
  const AMDGPURegisterInfo RI;
  TargetMachine &TM;
  bool getNextBranchInstr(MachineBasicBlock::iterator &iter,
                          MachineBasicBlock &MBB) const;
public:
  explicit AMDGPUInstrInfo(TargetMachine &tm);

  virtual const AMDGPURegisterInfo &getRegisterInfo() const = 0;

  bool isCoalescableExtInstr(const MachineInstr &MI, unsigned &SrcReg,
                             unsigned &DstReg, unsigned &SubIdx) const;

  unsigned isLoadFromStackSlot(const MachineInstr *MI, int &FrameIndex) const;
  unsigned isLoadFromStackSlotPostFE(const MachineInstr *MI,
                                     int &FrameIndex) const;
  bool hasLoadFromStackSlot(const MachineInstr *MI,
                            const MachineMemOperand *&MMO,
                            int &FrameIndex) const;
  unsigned isStoreFromStackSlot(const MachineInstr *MI, int &FrameIndex) const;
  unsigned isStoreFromStackSlotPostFE(const MachineInstr *MI,
                                      int &FrameIndex) const;
  bool hasStoreFromStackSlot(const MachineInstr *MI,
                             const MachineMemOperand *&MMO,
                             int &FrameIndex) const;

  MachineInstr *
  convertToThreeAddress(MachineFunction::iterator &MFI,
                        MachineBasicBlock::iterator &MBBI,
                        LiveVariables *LV) const;


  virtual void copyPhysReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MI, DebugLoc DL,
                           unsigned DestReg, unsigned SrcReg,
                           bool KillSrc) const = 0;

  void storeRegToStackSlot(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MI,
                           unsigned SrcReg, bool isKill, int FrameIndex,
                           const TargetRegisterClass *RC,
                           const TargetRegisterInfo *TRI) const;
  void loadRegFromStackSlot(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI,
                            unsigned DestReg, int FrameIndex,
                            const TargetRegisterClass *RC,
                            const TargetRegisterInfo *TRI) const;

protected:
  MachineInstr *foldMemoryOperandImpl(MachineFunction &MF,
                                      MachineInstr *MI,
                                      const SmallVectorImpl<unsigned> &Ops,
                                      int FrameIndex) const;
  MachineInstr *foldMemoryOperandImpl(MachineFunction &MF,
                                      MachineInstr *MI,
                                      const SmallVectorImpl<unsigned> &Ops,
                                      MachineInstr *LoadMI) const;
public:
  bool canFoldMemoryOperand(const MachineInstr *MI,
                            const SmallVectorImpl<unsigned> &Ops) const;
  bool unfoldMemoryOperand(MachineFunction &MF, MachineInstr *MI,
                           unsigned Reg, bool UnfoldLoad, bool UnfoldStore,
                           SmallVectorImpl<MachineInstr *> &NewMIs) const;
  bool unfoldMemoryOperand(SelectionDAG &DAG, SDNode *N,
                           SmallVectorImpl<SDNode *> &NewNodes) const;
  unsigned getOpcodeAfterMemoryUnfold(unsigned Opc,
                                      bool UnfoldLoad, bool UnfoldStore,
                                      unsigned *LoadRegIndex = 0) const;
  bool shouldScheduleLoadsNear(SDNode *Load1, SDNode *Load2,
                               int64_t Offset1, int64_t Offset2,
                               unsigned NumLoads) const;

  bool ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const;
  void insertNoop(MachineBasicBlock &MBB,
                  MachineBasicBlock::iterator MI) const;
  bool isPredicated(const MachineInstr *MI) const;
  bool SubsumesPredicate(const SmallVectorImpl<MachineOperand> &Pred1,
                         const SmallVectorImpl<MachineOperand> &Pred2) const;
  bool DefinesPredicate(MachineInstr *MI,
                        std::vector<MachineOperand> &Pred) const;
  bool isPredicable(MachineInstr *MI) const;
  bool isSafeToMoveRegClassDefs(const TargetRegisterClass *RC) const;

  // Helper functions that check the opcode for status information
  bool isLoadInst(llvm::MachineInstr *MI) const;
  bool isExtLoadInst(llvm::MachineInstr *MI) const;
  bool isSWSExtLoadInst(llvm::MachineInstr *MI) const;
  bool isSExtLoadInst(llvm::MachineInstr *MI) const;
  bool isZExtLoadInst(llvm::MachineInstr *MI) const;
  bool isAExtLoadInst(llvm::MachineInstr *MI) const;
  bool isStoreInst(llvm::MachineInstr *MI) const;
  bool isTruncStoreInst(llvm::MachineInstr *MI) const;

  virtual MachineInstr* getMovImmInstr(MachineFunction *MF, unsigned DstReg,
                                       int64_t Imm) const = 0;
  virtual unsigned getIEQOpcode() const = 0;
  virtual bool isMov(unsigned opcode) const = 0;

  /// \brief Convert the AMDIL MachineInstr to a supported ISA
  /// MachineInstr
  virtual void convertToISA(MachineInstr & MI, MachineFunction &MF,
    DebugLoc DL) const;

};

} // End llvm namespace

#endif // AMDGPUINSTRINFO_H
