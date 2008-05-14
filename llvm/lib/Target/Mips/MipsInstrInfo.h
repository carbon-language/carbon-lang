//===- MipsInstrInfo.h - Mips Instruction Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Mips implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSINSTRUCTIONINFO_H
#define MIPSINSTRUCTIONINFO_H

#include "Mips.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "MipsRegisterInfo.h"

namespace llvm {

namespace Mips {

  // Mips Condition Codes
  enum CondCode {
    COND_E,
    COND_GZ,
    COND_GEZ,
    COND_LZ,
    COND_LEZ,
    COND_NE,
    COND_INVALID
  };
  
  // Turn condition code into conditional branch opcode.
  unsigned GetCondBranchFromCond(CondCode CC);
  
  /// GetOppositeBranchCondition - Return the inverse of the specified cond,
  /// e.g. turning COND_E to COND_NE.
  CondCode GetOppositeBranchCondition(Mips::CondCode CC);

}

class MipsInstrInfo : public TargetInstrInfoImpl {
  MipsTargetMachine &TM;
  const MipsRegisterInfo RI;
public:
  explicit MipsInstrInfo(MipsTargetMachine &TM);

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  virtual const MipsRegisterInfo &getRegisterInfo() const { return RI; }

  /// Return true if the instruction is a register to register move and
  /// leave the source and dest operands in the passed parameters.
  ///
  virtual bool isMoveInstr(const MachineInstr &MI,
                           unsigned &SrcReg, unsigned &DstReg) const;
  
  /// isLoadFromStackSlot - If the specified machine instruction is a direct
  /// load from a stack slot, return the virtual or physical register number of
  /// the destination along with the FrameIndex of the loaded stack slot.  If
  /// not, return 0.  This predicate must return 0 if the instruction has
  /// any side effects other than loading from the stack slot.
  virtual unsigned isLoadFromStackSlot(MachineInstr *MI, int &FrameIndex) const;
  
  /// isStoreToStackSlot - If the specified machine instruction is a direct
  /// store to a stack slot, return the virtual or physical register number of
  /// the source reg along with the FrameIndex of the loaded stack slot.  If
  /// not, return 0.  This predicate must return 0 if the instruction has
  /// any side effects other than storing to the stack slot.
  virtual unsigned isStoreToStackSlot(MachineInstr *MI, int &FrameIndex) const;
 
  /// Branch Analysis
  virtual bool AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                             MachineBasicBlock *&FBB,
                             std::vector<MachineOperand> &Cond) const;
  virtual unsigned RemoveBranch(MachineBasicBlock &MBB) const;
  virtual unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                                MachineBasicBlock *FBB,
                                const std::vector<MachineOperand> &Cond) const;
  virtual void copyRegToReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
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
  
  virtual bool BlockHasNoFallThrough(MachineBasicBlock &MBB) const;
  virtual bool ReverseBranchCondition(std::vector<MachineOperand> &Cond) const;

  /// Insert nop instruction when hazard condition is found
  virtual void insertNoop(MachineBasicBlock &MBB, 
                          MachineBasicBlock::iterator MI) const;
};

}

#endif
