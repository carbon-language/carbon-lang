//===- PPCInstrInfo.cpp - PowerPC32 Instruction Information -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PowerPC implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "PPCInstrInfo.h"
#include "PPCPredicates.h"
#include "PPCGenInstrInfo.inc"
#include "PPCTargetMachine.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

PPCInstrInfo::PPCInstrInfo(PPCTargetMachine &tm)
  : TargetInstrInfo(PPCInsts, sizeof(PPCInsts)/sizeof(PPCInsts[0])), TM(tm),
    RI(*TM.getSubtargetImpl(), *this) {}

/// getPointerRegClass - Return the register class to use to hold pointers.
/// This is used for addressing modes.
const TargetRegisterClass *PPCInstrInfo::getPointerRegClass() const {
  if (TM.getSubtargetImpl()->isPPC64())
    return &PPC::G8RCRegClass;
  else
    return &PPC::GPRCRegClass;
}


bool PPCInstrInfo::isMoveInstr(const MachineInstr& MI,
                               unsigned& sourceReg,
                               unsigned& destReg) const {
  MachineOpCode oc = MI.getOpcode();
  if (oc == PPC::OR || oc == PPC::OR8 || oc == PPC::VOR ||
      oc == PPC::OR4To8 || oc == PPC::OR8To4) {                // or r1, r2, r2
    assert(MI.getNumOperands() >= 3 &&
           MI.getOperand(0).isRegister() &&
           MI.getOperand(1).isRegister() &&
           MI.getOperand(2).isRegister() &&
           "invalid PPC OR instruction!");
    if (MI.getOperand(1).getReg() == MI.getOperand(2).getReg()) {
      sourceReg = MI.getOperand(1).getReg();
      destReg = MI.getOperand(0).getReg();
      return true;
    }
  } else if (oc == PPC::ADDI) {             // addi r1, r2, 0
    assert(MI.getNumOperands() >= 3 &&
           MI.getOperand(0).isRegister() &&
           MI.getOperand(2).isImmediate() &&
           "invalid PPC ADDI instruction!");
    if (MI.getOperand(1).isRegister() && MI.getOperand(2).getImmedValue()==0) {
      sourceReg = MI.getOperand(1).getReg();
      destReg = MI.getOperand(0).getReg();
      return true;
    }
  } else if (oc == PPC::ORI) {             // ori r1, r2, 0
    assert(MI.getNumOperands() >= 3 &&
           MI.getOperand(0).isRegister() &&
           MI.getOperand(1).isRegister() &&
           MI.getOperand(2).isImmediate() &&
           "invalid PPC ORI instruction!");
    if (MI.getOperand(2).getImmedValue()==0) {
      sourceReg = MI.getOperand(1).getReg();
      destReg = MI.getOperand(0).getReg();
      return true;
    }
  } else if (oc == PPC::FMRS || oc == PPC::FMRD ||
             oc == PPC::FMRSD) {      // fmr r1, r2
    assert(MI.getNumOperands() >= 2 &&
           MI.getOperand(0).isRegister() &&
           MI.getOperand(1).isRegister() &&
           "invalid PPC FMR instruction");
    sourceReg = MI.getOperand(1).getReg();
    destReg = MI.getOperand(0).getReg();
    return true;
  } else if (oc == PPC::MCRF) {             // mcrf cr1, cr2
    assert(MI.getNumOperands() >= 2 &&
           MI.getOperand(0).isRegister() &&
           MI.getOperand(1).isRegister() &&
           "invalid PPC MCRF instruction");
    sourceReg = MI.getOperand(1).getReg();
    destReg = MI.getOperand(0).getReg();
    return true;
  }
  return false;
}

unsigned PPCInstrInfo::isLoadFromStackSlot(MachineInstr *MI, 
                                           int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case PPC::LD:
  case PPC::LWZ:
  case PPC::LFS:
  case PPC::LFD:
    if (MI->getOperand(1).isImmediate() && !MI->getOperand(1).getImmedValue() &&
        MI->getOperand(2).isFrameIndex()) {
      FrameIndex = MI->getOperand(2).getFrameIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

unsigned PPCInstrInfo::isStoreToStackSlot(MachineInstr *MI, 
                                          int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case PPC::STD:
  case PPC::STW:
  case PPC::STFS:
  case PPC::STFD:
    if (MI->getOperand(1).isImmediate() && !MI->getOperand(1).getImmedValue() &&
        MI->getOperand(2).isFrameIndex()) {
      FrameIndex = MI->getOperand(2).getFrameIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

// commuteInstruction - We can commute rlwimi instructions, but only if the
// rotate amt is zero.  We also have to munge the immediates a bit.
MachineInstr *PPCInstrInfo::commuteInstruction(MachineInstr *MI) const {
  // Normal instructions can be commuted the obvious way.
  if (MI->getOpcode() != PPC::RLWIMI)
    return TargetInstrInfo::commuteInstruction(MI);
  
  // Cannot commute if it has a non-zero rotate count.
  if (MI->getOperand(3).getImmedValue() != 0)
    return 0;
  
  // If we have a zero rotate count, we have:
  //   M = mask(MB,ME)
  //   Op0 = (Op1 & ~M) | (Op2 & M)
  // Change this to:
  //   M = mask((ME+1)&31, (MB-1)&31)
  //   Op0 = (Op2 & ~M) | (Op1 & M)

  // Swap op1/op2
  unsigned Reg1 = MI->getOperand(1).getReg();
  unsigned Reg2 = MI->getOperand(2).getReg();
  bool Reg1IsKill = MI->getOperand(1).isKill();
  bool Reg2IsKill = MI->getOperand(2).isKill();
  MI->getOperand(2).setReg(Reg1);
  MI->getOperand(1).setReg(Reg2);
  if (Reg1IsKill)
    MI->getOperand(2).setIsKill();
  else
    MI->getOperand(2).unsetIsKill();
  if (Reg2IsKill)
    MI->getOperand(1).setIsKill();
  else
    MI->getOperand(1).unsetIsKill();
  
  // Swap the mask around.
  unsigned MB = MI->getOperand(4).getImmedValue();
  unsigned ME = MI->getOperand(5).getImmedValue();
  MI->getOperand(4).setImmedValue((ME+1) & 31);
  MI->getOperand(5).setImmedValue((MB-1) & 31);
  return MI;
}

void PPCInstrInfo::insertNoop(MachineBasicBlock &MBB, 
                              MachineBasicBlock::iterator MI) const {
  BuildMI(MBB, MI, get(PPC::NOP));
}


// Branch analysis.
bool PPCInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,MachineBasicBlock *&TBB,
                                 MachineBasicBlock *&FBB,
                                 std::vector<MachineOperand> &Cond) const {
  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin() || !isTerminatorInstr((--I)->getOpcode()))
    return false;

  // Get the last instruction in the block.
  MachineInstr *LastInst = I;
  
  // If there is only one terminator instruction, process it.
  if (I == MBB.begin() || !isTerminatorInstr((--I)->getOpcode())) {
    if (LastInst->getOpcode() == PPC::B) {
      TBB = LastInst->getOperand(0).getMachineBasicBlock();
      return false;
    } else if (LastInst->getOpcode() == PPC::BCC) {
      // Block ends with fall-through condbranch.
      TBB = LastInst->getOperand(2).getMachineBasicBlock();
      Cond.push_back(LastInst->getOperand(0));
      Cond.push_back(LastInst->getOperand(1));
      return false;
    }
    // Otherwise, don't know what this is.
    return true;
  }
  
  // Get the instruction before it if it's a terminator.
  MachineInstr *SecondLastInst = I;

  // If there are three terminators, we don't know what sort of block this is.
  if (SecondLastInst && I != MBB.begin() &&
      isTerminatorInstr((--I)->getOpcode()))
    return true;
  
  // If the block ends with PPC::B and PPC:BCC, handle it.
  if (SecondLastInst->getOpcode() == PPC::BCC && 
      LastInst->getOpcode() == PPC::B) {
    TBB =  SecondLastInst->getOperand(2).getMachineBasicBlock();
    Cond.push_back(SecondLastInst->getOperand(0));
    Cond.push_back(SecondLastInst->getOperand(1));
    FBB = LastInst->getOperand(0).getMachineBasicBlock();
    return false;
  }
  
  // Otherwise, can't handle this.
  return true;
}

void PPCInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin()) return;
  --I;
  if (I->getOpcode() != PPC::B && I->getOpcode() != PPC::BCC)
    return;
  
  // Remove the branch.
  I->eraseFromParent();
  
  I = MBB.end();

  if (I == MBB.begin()) return;
  --I;
  if (I->getOpcode() != PPC::BCC)
    return;
  
  // Remove the branch.
  I->eraseFromParent();
}

void PPCInstrInfo::InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                                MachineBasicBlock *FBB,
                                const std::vector<MachineOperand> &Cond) const {
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 2 || Cond.size() == 0) && 
         "PPC branch conditions have two components!");
  
  // One-way branch.
  if (FBB == 0) {
    if (Cond.empty())   // Unconditional branch
      BuildMI(&MBB, get(PPC::B)).addMBB(TBB);
    else                // Conditional branch
      BuildMI(&MBB, get(PPC::BCC))
        .addImm(Cond[0].getImm()).addReg(Cond[1].getReg()).addMBB(TBB);
    return;
  }
  
  // Two-way Conditional Branch.
  BuildMI(&MBB, get(PPC::BCC))
    .addImm(Cond[0].getImm()).addReg(Cond[1].getReg()).addMBB(TBB);
  BuildMI(&MBB, get(PPC::B)).addMBB(FBB);
}

bool PPCInstrInfo::BlockHasNoFallThrough(MachineBasicBlock &MBB) const {
  if (MBB.empty()) return false;
  
  switch (MBB.back().getOpcode()) {
  case PPC::B:     // Uncond branch.
  case PPC::BCTR:  // Indirect branch.
    return true;
  default: return false;
  }
}

bool PPCInstrInfo::
ReverseBranchCondition(std::vector<MachineOperand> &Cond) const {
  assert(Cond.size() == 2 && "Invalid PPC branch opcode!");
  // Leave the CR# the same, but invert the condition.
  Cond[0].setImm(PPC::InvertPredicate((PPC::Predicate)Cond[0].getImm()));
  return false;
}
