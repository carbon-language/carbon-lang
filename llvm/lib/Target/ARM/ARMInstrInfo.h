//===- ARMInstrInfo.h - ARM Instruction Information -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef ARMINSTRUCTIONINFO_H
#define ARMINSTRUCTIONINFO_H

#include "llvm/Target/TargetInstrInfo.h"
#include "ARMRegisterInfo.h"

namespace llvm {
  class ARMSubtarget;

/// ARMII - This namespace holds all of the target specific flags that
/// instruction info tracks.
///
namespace ARMII {
  enum {
    //===------------------------------------------------------------------===//
    // Instruction Flags.

    //===------------------------------------------------------------------===//
    // This three-bit field describes the addressing mode used.  Zero is unused
    // so that we can tell if we forgot to set a value.

    AddrModeMask  = 0xf,
    AddrModeNone  = 0,
    AddrMode1     = 1,
    AddrMode2     = 2,
    AddrMode3     = 3,
    AddrMode4     = 4,
    AddrMode5     = 5,
    AddrModeT1    = 6,
    AddrModeT2    = 7,
    AddrModeT4    = 8,
    AddrModeTs    = 9,   // i8 * 4 for pc and sp relative data

    // Size* - Flags to keep track of the size of an instruction.
    SizeShift     = 4,
    SizeMask      = 7 << SizeShift,
    SizeSpecial   = 1,   // 0 byte pseudo or special case.
    Size8Bytes    = 2,
    Size4Bytes    = 3,
    Size2Bytes    = 4,
    
    // IndexMode - Unindex, pre-indexed, or post-indexed. Only valid for load
    // and store ops 
    IndexModeShift = 7,
    IndexModeMask  = 3 << IndexModeShift,
    IndexModePre   = 1,
    IndexModePost  = 2,
    
    // Opcode
    OpcodeShift   = 9,
    OpcodeMask    = 0xf << OpcodeShift,
    
    // Format
    FormShift   = 13,
    FormMask    = 31 << FormShift,

    // Pseudo instructions
    Pseudo      = 1 << FormShift,

    // Multiply instructions
    MulFrm      = 2 << FormShift,
    MulSMLAW    = 3 << FormShift,
    MulSMULW    = 4 << FormShift,
    MulSMLA     = 5 << FormShift,
    MulSMUL     = 6 << FormShift,

    // Branch instructions
    Branch      = 7 << FormShift,
    BranchMisc  = 8 << FormShift,

    // Data Processing instructions
    DPRdIm      = 9 << FormShift,
    DPRdReg     = 10 << FormShift,
    DPRdSoReg   = 11 << FormShift,
    DPRdMisc    = 12 << FormShift,

    DPRnIm      = 13 << FormShift,
    DPRnReg     = 14 << FormShift,
    DPRnSoReg   = 15 << FormShift,

    DPRIm       = 16 << FormShift,
    DPRReg      = 17 << FormShift,
    DPRSoReg    = 18 << FormShift,

    DPRImS      = 19 << FormShift,
    DPRRegS     = 20 << FormShift,
    DPRSoRegS   = 21 << FormShift,

    // Load and Store
    LdFrm       = 22 << FormShift,
    StFrm       = 23 << FormShift,

    // Miscellaneous arithmetic instructions
    ArithMisc   = 24 << FormShift,

    // Thumb format
    ThumbFrm    = 25 << FormShift,

    // VFP format
    VPFFrm      = 26 << FormShift,

    // Field shifts - such shifts are used to set field while generating
    // machine instructions.
    RegRsShift  = 8,
    RegRdShift  = 12,
    RegRnShift  = 16,
    L_BitShift  = 20,
    S_BitShift  = 20,
    U_BitShift  = 23,
    IndexShift  = 24,
    I_BitShift  = 25
  };
}

class ARMInstrInfo : public TargetInstrInfoImpl {
  const ARMRegisterInfo RI;
public:
  explicit ARMInstrInfo(const ARMSubtarget &STI);

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  virtual const ARMRegisterInfo &getRegisterInfo() const { return RI; }

  /// getPointerRegClass - Return the register class to use to hold pointers.
  /// This is used for addressing modes.
  virtual const TargetRegisterClass *getPointerRegClass() const;

  /// Return true if the instruction is a register to register move and
  /// leave the source and dest operands in the passed parameters.
  ///
  virtual bool isMoveInstr(const MachineInstr &MI,
                           unsigned &SrcReg, unsigned &DstReg) const;
  virtual unsigned isLoadFromStackSlot(MachineInstr *MI, int &FrameIndex) const;
  virtual unsigned isStoreToStackSlot(MachineInstr *MI, int &FrameIndex) const;
  
  void reMaterialize(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                     unsigned DestReg, const MachineInstr *Orig) const;

  virtual MachineInstr *convertToThreeAddress(MachineFunction::iterator &MFI,
                                              MachineBasicBlock::iterator &MBBI,
                                              LiveVariables *LV) const;

  // Branch analysis.
  virtual bool AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                             MachineBasicBlock *&FBB,
                             std::vector<MachineOperand> &Cond) const;
  virtual unsigned RemoveBranch(MachineBasicBlock &MBB) const;
  virtual unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                                MachineBasicBlock *FBB,
                                const std::vector<MachineOperand> &Cond) const;
  virtual void copyRegToReg(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator I,
                            unsigned DestReg, unsigned SrcReg,
                            const TargetRegisterClass *DestRC,
                            const TargetRegisterClass *SrcRC) const;
  virtual void storeRegToStackSlot(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MBBI,
                                   unsigned SrcReg, bool isKill, int FrameIndex,
                                   const TargetRegisterClass *RC) const;

  virtual void storeRegToAddr(MachineFunction &MF, unsigned SrcReg, bool isKill,
                              SmallVectorImpl<MachineOperand> &Addr,
                              const TargetRegisterClass *RC,
                              SmallVectorImpl<MachineInstr*> &NewMIs) const;

  virtual void loadRegFromStackSlot(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MBBI,
                                    unsigned DestReg, int FrameIndex,
                                    const TargetRegisterClass *RC) const;

  virtual void loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                               SmallVectorImpl<MachineOperand> &Addr,
                               const TargetRegisterClass *RC,
                               SmallVectorImpl<MachineInstr*> &NewMIs) const;
  virtual bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MI,
                                 const std::vector<CalleeSavedInfo> &CSI) const;
  virtual bool restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator MI,
                                 const std::vector<CalleeSavedInfo> &CSI) const;
  
  virtual MachineInstr* foldMemoryOperand(MachineFunction &MF,
                                          MachineInstr* MI,
                                          SmallVectorImpl<unsigned> &Ops,
                                          int FrameIndex) const;

  virtual MachineInstr* foldMemoryOperand(MachineFunction &MF,
                                          MachineInstr* MI,
                                          SmallVectorImpl<unsigned> &Ops,
                                          MachineInstr* LoadMI) const {
    return 0;
  }

  virtual bool canFoldMemoryOperand(MachineInstr *MI,
                                    SmallVectorImpl<unsigned> &Ops) const;
  
  virtual bool BlockHasNoFallThrough(MachineBasicBlock &MBB) const;
  virtual bool ReverseBranchCondition(std::vector<MachineOperand> &Cond) const;

  // Predication support.
  virtual bool isPredicated(const MachineInstr *MI) const;

  virtual
  bool PredicateInstruction(MachineInstr *MI,
                            const std::vector<MachineOperand> &Pred) const;

  virtual
  bool SubsumesPredicate(const std::vector<MachineOperand> &Pred1,
                         const std::vector<MachineOperand> &Pred2) const;

  virtual bool DefinesPredicate(MachineInstr *MI,
                                std::vector<MachineOperand> &Pred) const;
    
  /// GetInstSize - Returns the size of the specified MachineInstr.
  ///
  virtual unsigned GetInstSizeInBytes(const MachineInstr* MI) const;
};

}

#endif
