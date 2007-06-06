//===- MipsInstrInfo.cpp - Mips Instruction Information ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bruno Cardoso Lopes and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Mips implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "Mips.h"
#include "MipsInstrInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "MipsGenInstrInfo.inc"

using namespace llvm;

// TODO: Add the subtarget support on this constructor
MipsInstrInfo::MipsInstrInfo(MipsTargetMachine &tm)
  : TargetInstrInfo(MipsInsts, sizeof(MipsInsts)/sizeof(MipsInsts[0])),
    TM(tm), RI(*this) {}

static bool isZeroImm(const MachineOperand &op) {
  return op.isImmediate() && op.getImmedValue() == 0;
}

/// Return true if the instruction is a register to register move and
/// leave the source and dest operands in the passed parameters.
bool MipsInstrInfo::
isMoveInstr(const MachineInstr &MI, unsigned &SrcReg, unsigned &DstReg) const 
{
  //  addu  $dst, $src, $zero || addu  $dst, $zero, $src
  //  or    $dst, $src, $zero || or    $dst, $zero, $src
  if ((MI.getOpcode() == Mips::ADDu) || (MI.getOpcode() == Mips::OR))
  {
    if (MI.getOperand(1).getReg() == Mips::ZERO) {
      DstReg = MI.getOperand(0).getReg();
      SrcReg = MI.getOperand(2).getReg();
      return true;
    } else if (MI.getOperand(2).getReg() == Mips::ZERO) {
      DstReg = MI.getOperand(0).getReg();
      SrcReg = MI.getOperand(1).getReg();
      return true;
    }
  }

  //  addiu $dst, $src, 0
  if (MI.getOpcode() == Mips::ADDiu) 
  {
    if ((MI.getOperand(1).isRegister()) && (isZeroImm(MI.getOperand(2)))) {
      DstReg = MI.getOperand(0).getReg();
      SrcReg = MI.getOperand(1).getReg();
      return true;
    }
  }
  return false;
}

/// isLoadFromStackSlot - If the specified machine instruction is a direct
/// load from a stack slot, return the virtual or physical register number of
/// the destination along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than loading from the stack slot.
unsigned MipsInstrInfo::
isLoadFromStackSlot(MachineInstr *MI, int &FrameIndex) const 
{
  // TODO: add lhu, lbu ???
  if (MI->getOpcode() == Mips::LW) 
  {
    if ((MI->getOperand(2).isFrameIndex()) && // is a stack slot
        (MI->getOperand(1).isImmediate()) &&  // the imm is zero
        (isZeroImm(MI->getOperand(1)))) 
    {
      FrameIndex = MI->getOperand(2).getFrameIndex();
      return MI->getOperand(0).getReg();
    }
  }

  return 0;
}

/// isStoreToStackSlot - If the specified machine instruction is a direct
/// store to a stack slot, return the virtual or physical register number of
/// the source reg along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than storing to the stack slot.
unsigned MipsInstrInfo::
isStoreToStackSlot(MachineInstr *MI, int &FrameIndex) const 
{
  // TODO: add sb, sh ???
  if (MI->getOpcode() == Mips::SW) {
    if ((MI->getOperand(0).isFrameIndex()) && // is a stack slot
        (MI->getOperand(1).isImmediate()) &&  // the imm is zero
        (isZeroImm(MI->getOperand(1)))) 
    {
      FrameIndex = MI->getOperand(0).getFrameIndex();
      return MI->getOperand(2).getReg();
    }
  }
  return 0;
}

unsigned MipsInstrInfo::
InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB, 
             MachineBasicBlock *FBB, const std::vector<MachineOperand> &Cond)
             const
{
  // TODO: add Mips::J here.
  assert(0 && "Cant handle any kind of branches!");
  return 1;
}
