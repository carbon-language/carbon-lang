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

#include "AMDGPUInstrInfo.h"
#include "AMDGPURegisterInfo.h"
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
  bool getNextBranchInstr(MachineBasicBlock::iterator &iter,
                          MachineBasicBlock &MBB) const;
protected:
  TargetMachine &TM;
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
  bool isRegisterStore(const MachineInstr &MI) const;
  bool isRegisterLoad(const MachineInstr &MI) const;

//===---------------------------------------------------------------------===//
// Pure virtual funtions to be implemented by sub-classes.
//===---------------------------------------------------------------------===//

  virtual MachineInstr* getMovImmInstr(MachineFunction *MF, unsigned DstReg,
                                       int64_t Imm) const = 0;
  virtual unsigned getIEQOpcode() const = 0;
  virtual bool isMov(unsigned opcode) const = 0;

  /// \returns the smallest register index that will be accessed by an indirect
  /// read or write or -1 if indirect addressing is not used by this program.
  virtual int getIndirectIndexBegin(const MachineFunction &MF) const = 0;

  /// \returns the largest register index that will be accessed by an indirect
  /// read or write or -1 if indirect addressing is not used by this program.
  virtual int getIndirectIndexEnd(const MachineFunction &MF) const = 0;

  /// \brief Calculate the "Indirect Address" for the given \p RegIndex and
  ///        \p Channel
  ///
  /// We model indirect addressing using a virtual address space that can be
  /// accesed with loads and stores.  The "Indirect Address" is the memory
  /// address in this virtual address space that maps to the given \p RegIndex
  /// and \p Channel.
  virtual unsigned calculateIndirectAddress(unsigned RegIndex,
                                            unsigned Channel) const = 0;

  /// \returns The register class to be used for storing values to an
  /// "Indirect Address" .
  virtual const TargetRegisterClass *getIndirectAddrStoreRegClass(
                                                  unsigned SourceReg) const = 0;

  /// \returns The register class to be used for loading values from
  /// an "Indirect Address" .
  virtual const TargetRegisterClass *getIndirectAddrLoadRegClass() const = 0;

  /// \brief Build instruction(s) for an indirect register write.
  ///
  /// \returns The instruction that performs the indirect register write
  virtual MachineInstrBuilder buildIndirectWrite(MachineBasicBlock *MBB,
                                    MachineBasicBlock::iterator I,
                                    unsigned ValueReg, unsigned Address,
                                    unsigned OffsetReg) const = 0;

  /// \brief Build instruction(s) for an indirect register read.
  ///
  /// \returns The instruction that performs the indirect register read
  virtual MachineInstrBuilder buildIndirectRead(MachineBasicBlock *MBB,
                                    MachineBasicBlock::iterator I,
                                    unsigned ValueReg, unsigned Address,
                                    unsigned OffsetReg) const = 0;

  /// \returns the register class whose sub registers are the set of all
  /// possible registers that can be used for indirect addressing.
  virtual const TargetRegisterClass *getSuperIndirectRegClass() const = 0;


  /// \brief Convert the AMDIL MachineInstr to a supported ISA
  /// MachineInstr
  virtual void convertToISA(MachineInstr & MI, MachineFunction &MF,
    DebugLoc DL) const;

};

} // End llvm namespace

#define AMDGPU_FLAG_REGISTER_LOAD  (UINT64_C(1) << 63)
#define AMDGPU_FLAG_REGISTER_STORE (UINT64_C(1) << 62)

#endif // AMDGPUINSTRINFO_H
