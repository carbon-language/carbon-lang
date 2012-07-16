//===- AMDILInstrInfo.h - AMDIL Instruction Information ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
// This file contains the AMDIL implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef AMDILINSTRUCTIONINFO_H_
#define AMDILINSTRUCTIONINFO_H_

#include "AMDILRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "AMDGPUGenInstrInfo.inc"

namespace llvm {
  // AMDIL - This namespace holds all of the target specific flags that
  // instruction info tracks.
  //
  //class AMDILTargetMachine;
class AMDILInstrInfo : public AMDILGenInstrInfo {
private:
  const AMDILRegisterInfo RI;
  bool getNextBranchInstr(MachineBasicBlock::iterator &iter,
                          MachineBasicBlock &MBB) const;
  unsigned int getBranchInstr(const MachineOperand &op) const;
public:
  explicit AMDILInstrInfo(TargetMachine &tm);

  // getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  // such, whenever a client has an instance of instruction info, it should
  // always be able to get register info as well (through this method).
  const AMDILRegisterInfo &getRegisterInfo() const;

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

  bool AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                     MachineBasicBlock *&FBB,
                     SmallVectorImpl<MachineOperand> &Cond,
                     bool AllowModify) const;

  unsigned RemoveBranch(MachineBasicBlock &MBB) const;

  unsigned
  InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
               MachineBasicBlock *FBB,
               const SmallVectorImpl<MachineOperand> &Cond,
               DebugLoc DL) const;

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
  bool isAtomicInst(llvm::MachineInstr *MI) const;
  bool isVolatileInst(llvm::MachineInstr *MI) const;
  bool isGlobalInst(llvm::MachineInstr *MI) const;
  bool isPrivateInst(llvm::MachineInstr *MI) const;
  bool isConstantInst(llvm::MachineInstr *MI) const;
  bool isRegionInst(llvm::MachineInstr *MI) const;
  bool isLocalInst(llvm::MachineInstr *MI) const;
  bool isImageInst(llvm::MachineInstr *MI) const;
  bool isAppendInst(llvm::MachineInstr *MI) const;
  bool isRegionAtomic(llvm::MachineInstr *MI) const;
  bool isLocalAtomic(llvm::MachineInstr *MI) const;
  bool isGlobalAtomic(llvm::MachineInstr *MI) const;
  bool isArenaAtomic(llvm::MachineInstr *MI) const;

  virtual MachineInstr * getMovImmInstr(MachineFunction *MF, unsigned DstReg,
                                        int64_t Imm) const = 0;

  virtual unsigned getIEQOpcode() const = 0;

  virtual bool isMov(unsigned Opcode) const = 0;
};

}

#endif // AMDILINSTRINFO_H_
