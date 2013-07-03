//===-- SystemZInstrInfo.h - SystemZ instruction information ----*- C++ -*-===//
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

#include "SystemZ.h"
#include "SystemZRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "SystemZGenInstrInfo.inc"

namespace llvm {

class SystemZTargetMachine;

namespace SystemZII {
  enum {
    // See comments in SystemZInstrFormats.td.
    SimpleBDXLoad  = (1 << 0),
    SimpleBDXStore = (1 << 1),
    Has20BitOffset = (1 << 2),
    HasIndex       = (1 << 3),
    Is128Bit       = (1 << 4),
    AccessSizeMask = (31 << 5),
    AccessSizeShift = 5
  };
  static inline unsigned getAccessSize(unsigned int Flags) {
    return (Flags & AccessSizeMask) >> AccessSizeShift;
  }

  // SystemZ MachineOperand target flags.
  enum {
    // Masks out the bits for the access model.
    MO_SYMBOL_MODIFIER = (1 << 0),

    // @GOT (aka @GOTENT)
    MO_GOT = (1 << 0)
  };
  // Classifies a branch.
  enum BranchType {
    // An instruction that branches on the current value of CC.
    BranchNormal,

    // An instruction that peforms a 32-bit signed comparison and branches
    // on the result.
    BranchC,

    // An instruction that peforms a 64-bit signed comparison and branches
    // on the result.
    BranchCG
  };
  // Information about a branch instruction.
  struct Branch {
    // The type of the branch.
    BranchType Type;

    // CCMASK_<N> is set if the branch should be taken when CC == N.
    unsigned CCMask;

    // The target of the branch.
    const MachineOperand *Target;

    Branch(BranchType type, unsigned ccMask, const MachineOperand *target)
      : Type(type), CCMask(ccMask), Target(target) {}
  };
}

class SystemZInstrInfo : public SystemZGenInstrInfo {
  const SystemZRegisterInfo RI;

  void splitMove(MachineBasicBlock::iterator MI, unsigned NewOpcode) const;
  void splitAdjDynAlloc(MachineBasicBlock::iterator MI) const;

public:
  explicit SystemZInstrInfo(SystemZTargetMachine &TM);

  // Override TargetInstrInfo.
  virtual unsigned isLoadFromStackSlot(const MachineInstr *MI,
                                       int &FrameIndex) const LLVM_OVERRIDE;
  virtual unsigned isStoreToStackSlot(const MachineInstr *MI,
                                      int &FrameIndex) const LLVM_OVERRIDE;
  virtual bool AnalyzeBranch(MachineBasicBlock &MBB,
                             MachineBasicBlock *&TBB,
                             MachineBasicBlock *&FBB,
                             SmallVectorImpl<MachineOperand> &Cond,
                             bool AllowModify) const LLVM_OVERRIDE;
  virtual unsigned RemoveBranch(MachineBasicBlock &MBB) const LLVM_OVERRIDE;
  virtual unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                                MachineBasicBlock *FBB,
                                const SmallVectorImpl<MachineOperand> &Cond,
                                DebugLoc DL) const LLVM_OVERRIDE;
  virtual void copyPhysReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI, DebugLoc DL,
                           unsigned DestReg, unsigned SrcReg,
                           bool KillSrc) const LLVM_OVERRIDE;
  virtual void
    storeRegToStackSlot(MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator MBBI,
                        unsigned SrcReg, bool isKill, int FrameIndex,
                        const TargetRegisterClass *RC,
                        const TargetRegisterInfo *TRI) const LLVM_OVERRIDE;
  virtual void
    loadRegFromStackSlot(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI,
                         unsigned DestReg, int FrameIdx,
                         const TargetRegisterClass *RC,
                         const TargetRegisterInfo *TRI) const LLVM_OVERRIDE;
  virtual MachineInstr *
    foldMemoryOperandImpl(MachineFunction &MF, MachineInstr *MI,
                          const SmallVectorImpl<unsigned> &Ops,
                          int FrameIndex) const;
  virtual MachineInstr *
    foldMemoryOperandImpl(MachineFunction &MF, MachineInstr* MI,
                          const SmallVectorImpl<unsigned> &Ops,
                          MachineInstr* LoadMI) const;
  virtual bool
    expandPostRAPseudo(MachineBasicBlock::iterator MBBI) const LLVM_OVERRIDE;
  virtual bool
    ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const
    LLVM_OVERRIDE;

  // Return the SystemZRegisterInfo, which this class owns.
  const SystemZRegisterInfo &getRegisterInfo() const { return RI; }

  // Return the size in bytes of MI.
  uint64_t getInstSizeInBytes(const MachineInstr *MI) const;

  // Return true if MI is a conditional or unconditional branch.
  // When returning true, set Cond to the mask of condition-code
  // values on which the instruction will branch, and set Target
  // to the operand that contains the branch target.  This target
  // can be a register or a basic block.
  SystemZII::Branch getBranchInfo(const MachineInstr *MI) const;

  // Get the load and store opcodes for a given register class.
  void getLoadStoreOpcodes(const TargetRegisterClass *RC,
                           unsigned &LoadOpcode, unsigned &StoreOpcode) const;

  // Opcode is the opcode of an instruction that has an address operand,
  // and the caller wants to perform that instruction's operation on an
  // address that has displacement Offset.  Return the opcode of a suitable
  // instruction (which might be Opcode itself) or 0 if no such instruction
  // exists.
  unsigned getOpcodeForOffset(unsigned Opcode, int64_t Offset) const;

  // If Opcode is a COMPARE opcode for which an associated COMPARE AND
  // BRANCH exists, return the opcode for the latter, otherwise return 0.
  // MI, if nonnull, is the compare instruction.
  unsigned getCompareAndBranch(unsigned Opcode,
                               const MachineInstr *MI = 0) const;

  // Emit code before MBBI in MI to move immediate value Value into
  // physical register Reg.
  void loadImmediate(MachineBasicBlock &MBB,
                     MachineBasicBlock::iterator MBBI,
                     unsigned Reg, uint64_t Value) const;
};
} // end namespace llvm

#endif
