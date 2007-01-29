//===- ARMInstrInfo.h - ARM Instruction Information -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the "Instituto Nokia de Tecnologia" and
// is distributed under the University of Illinois Open Source
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
    OpcodeMask    = 0xf << OpcodeShift
  };
}

class ARMInstrInfo : public TargetInstrInfo {
  const ARMRegisterInfo RI;
public:
  ARMInstrInfo(const ARMSubtarget &STI);

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  virtual const MRegisterInfo &getRegisterInfo() const { return RI; }

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
  
  virtual MachineInstr *convertToThreeAddress(MachineFunction::iterator &MFI,
                                              MachineBasicBlock::iterator &MBBI,
                                              LiveVariables &LV) const;

  // Branch analysis.
  virtual bool AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                             MachineBasicBlock *&FBB,
                             std::vector<MachineOperand> &Cond) const;
  virtual void RemoveBranch(MachineBasicBlock &MBB) const;
  virtual void InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                            MachineBasicBlock *FBB,
                            const std::vector<MachineOperand> &Cond) const;
  virtual bool BlockHasNoFallThrough(MachineBasicBlock &MBB) const;
  virtual bool ReverseBranchCondition(std::vector<MachineOperand> &Cond) const;
};

  // Utility routines
  namespace ARM {
    /// GetInstSize - Returns the size of the specified MachineInstr.
    ///
    unsigned GetInstSize(MachineInstr *MI);

    /// GetFunctionSize - Returns the size of the specified MachineFunction.
    ///
    unsigned GetFunctionSize(MachineFunction &MF);
  }
}

#endif
