//===- ARM64InstrInfo.h - ARM64 Instruction Information ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM64 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_ARM64INSTRINFO_H
#define LLVM_TARGET_ARM64INSTRINFO_H

#include "ARM64.h"
#include "ARM64RegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "ARM64GenInstrInfo.inc"

namespace llvm {

class ARM64Subtarget;
class ARM64TargetMachine;

class ARM64InstrInfo : public ARM64GenInstrInfo {
  // Reserve bits in the MachineMemOperand target hint flags, starting at 1.
  // They will be shifted into MOTargetHintStart when accessed.
  enum TargetMemOperandFlags {
    MOSuppressPair = 1
  };

  const ARM64RegisterInfo RI;
  const ARM64Subtarget &Subtarget;

public:
  explicit ARM64InstrInfo(const ARM64Subtarget &STI);

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  const ARM64RegisterInfo &getRegisterInfo() const { return RI; }

  const ARM64Subtarget &getSubTarget() const { return Subtarget; }

  unsigned GetInstSizeInBytes(const MachineInstr *MI) const;

  bool isCoalescableExtInstr(const MachineInstr &MI, unsigned &SrcReg,
                             unsigned &DstReg, unsigned &SubIdx) const override;

  unsigned isLoadFromStackSlot(const MachineInstr *MI,
                               int &FrameIndex) const override;
  unsigned isStoreToStackSlot(const MachineInstr *MI,
                              int &FrameIndex) const override;

  /// \brief Is there a non-zero immediate?
  bool hasNonZeroImm(const MachineInstr *MI) const;

  /// \brief Does this instruction set its full destination register to zero?
  bool isGPRZero(const MachineInstr *MI) const;

  /// \brief Does this instruction rename a GPR without modifying bits?
  bool isGPRCopy(const MachineInstr *MI) const;

  /// \brief Does this instruction rename an FPR without modifying bits?
  bool isFPRCopy(const MachineInstr *MI) const;

  /// Return true if this is load/store scales or extends its register offset.
  /// This refers to scaling a dynamic index as opposed to scaled immediates.
  /// MI should be a memory op that allows scaled addressing.
  bool isScaledAddr(const MachineInstr *MI) const;

  /// Return true if pairing the given load or store is hinted to be
  /// unprofitable.
  bool isLdStPairSuppressed(const MachineInstr *MI) const;

  /// Hint that pairing the given load or store is unprofitable.
  void suppressLdStPair(MachineInstr *MI) const;

  bool getLdStBaseRegImmOfs(MachineInstr *LdSt, unsigned &BaseReg,
                            unsigned &Offset,
                            const TargetRegisterInfo *TRI) const override;

  bool enableClusterLoads() const override { return true; }

  bool shouldClusterLoads(MachineInstr *FirstLdSt, MachineInstr *SecondLdSt,
                          unsigned NumLoads) const override;

  bool shouldScheduleAdjacent(MachineInstr *First,
                              MachineInstr *Second) const override;

  MachineInstr *emitFrameIndexDebugValue(MachineFunction &MF, int FrameIx,
                                         uint64_t Offset, const MDNode *MDPtr,
                                         DebugLoc DL) const;
  void copyPhysRegTuple(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                        DebugLoc DL, unsigned DestReg, unsigned SrcReg,
                        bool KillSrc, unsigned Opcode,
                        llvm::ArrayRef<unsigned> Indices) const;
  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                   DebugLoc DL, unsigned DestReg, unsigned SrcReg,
                   bool KillSrc) const override;

  void storeRegToStackSlot(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI, unsigned SrcReg,
                           bool isKill, int FrameIndex,
                           const TargetRegisterClass *RC,
                           const TargetRegisterInfo *TRI) const override;

  void loadRegFromStackSlot(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MBBI, unsigned DestReg,
                            int FrameIndex, const TargetRegisterClass *RC,
                            const TargetRegisterInfo *TRI) const override;

  MachineInstr *
  foldMemoryOperandImpl(MachineFunction &MF, MachineInstr *MI,
                        const SmallVectorImpl<unsigned> &Ops,
                        int FrameIndex) const override;

  bool AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                     MachineBasicBlock *&FBB,
                     SmallVectorImpl<MachineOperand> &Cond,
                     bool AllowModify = false) const override;
  unsigned RemoveBranch(MachineBasicBlock &MBB) const override;
  unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                        MachineBasicBlock *FBB,
                        const SmallVectorImpl<MachineOperand> &Cond,
                        DebugLoc DL) const override;
  bool
  ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const override;
  bool canInsertSelect(const MachineBasicBlock &,
                       const SmallVectorImpl<MachineOperand> &Cond, unsigned,
                       unsigned, int &, int &, int &) const override;
  void insertSelect(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                    DebugLoc DL, unsigned DstReg,
                    const SmallVectorImpl<MachineOperand> &Cond,
                    unsigned TrueReg, unsigned FalseReg) const override;
  void getNoopForMachoTarget(MCInst &NopInst) const override;

  /// analyzeCompare - For a comparison instruction, return the source registers
  /// in SrcReg and SrcReg2, and the value it compares against in CmpValue.
  /// Return true if the comparison instruction can be analyzed.
  bool analyzeCompare(const MachineInstr *MI, unsigned &SrcReg,
                      unsigned &SrcReg2, int &CmpMask,
                      int &CmpValue) const override;
  /// optimizeCompareInstr - Convert the instruction supplying the argument to
  /// the comparison into one that sets the zero bit in the flags register.
  bool optimizeCompareInstr(MachineInstr *CmpInstr, unsigned SrcReg,
                            unsigned SrcReg2, int CmpMask, int CmpValue,
                            const MachineRegisterInfo *MRI) const override;

private:
  void instantiateCondBranch(MachineBasicBlock &MBB, DebugLoc DL,
                             MachineBasicBlock *TBB,
                             const SmallVectorImpl<MachineOperand> &Cond) const;
};

/// emitFrameOffset - Emit instructions as needed to set DestReg to SrcReg
/// plus Offset.  This is intended to be used from within the prolog/epilog
/// insertion (PEI) pass, where a virtual scratch register may be allocated
/// if necessary, to be replaced by the scavenger at the end of PEI.
void emitFrameOffset(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                     DebugLoc DL, unsigned DestReg, unsigned SrcReg, int Offset,
                     const ARM64InstrInfo *TII,
                     MachineInstr::MIFlag = MachineInstr::NoFlags,
                     bool SetNZCV = false);

/// rewriteARM64FrameIndex - Rewrite MI to access 'Offset' bytes from the
/// FP. Return false if the offset could not be handled directly in MI, and
/// return the left-over portion by reference.
bool rewriteARM64FrameIndex(MachineInstr &MI, unsigned FrameRegIdx,
                            unsigned FrameReg, int &Offset,
                            const ARM64InstrInfo *TII);

/// \brief Use to report the frame offset status in isARM64FrameOffsetLegal.
enum ARM64FrameOffsetStatus {
  ARM64FrameOffsetCannotUpdate = 0x0, ///< Offset cannot apply.
  ARM64FrameOffsetIsLegal = 0x1,      ///< Offset is legal.
  ARM64FrameOffsetCanUpdate = 0x2     ///< Offset can apply, at least partly.
};

/// \brief Check if the @p Offset is a valid frame offset for @p MI.
/// The returned value reports the validity of the frame offset for @p MI.
/// It uses the values defined by ARM64FrameOffsetStatus for that.
/// If result == ARM64FrameOffsetCannotUpdate, @p MI cannot be updated to
/// use an offset.eq
/// If result & ARM64FrameOffsetIsLegal, @p Offset can completely be
/// rewriten in @p MI.
/// If result & ARM64FrameOffsetCanUpdate, @p Offset contains the
/// amount that is off the limit of the legal offset.
/// If set, @p OutUseUnscaledOp will contain the whether @p MI should be
/// turned into an unscaled operator, which opcode is in @p OutUnscaledOp.
/// If set, @p EmittableOffset contains the amount that can be set in @p MI
/// (possibly with @p OutUnscaledOp if OutUseUnscaledOp is true) and that
/// is a legal offset.
int isARM64FrameOffsetLegal(const MachineInstr &MI, int &Offset,
                            bool *OutUseUnscaledOp = nullptr,
                            unsigned *OutUnscaledOp = nullptr,
                            int *EmittableOffset = nullptr);

static inline bool isUncondBranchOpcode(int Opc) { return Opc == ARM64::B; }

static inline bool isCondBranchOpcode(int Opc) {
  switch (Opc) {
  case ARM64::Bcc:
  case ARM64::CBZW:
  case ARM64::CBZX:
  case ARM64::CBNZW:
  case ARM64::CBNZX:
  case ARM64::TBZ:
  case ARM64::TBNZ:
    return true;
  default:
    return false;
  }
}

static inline bool isIndirectBranchOpcode(int Opc) { return Opc == ARM64::BR; }

} // end namespace llvm

#endif
