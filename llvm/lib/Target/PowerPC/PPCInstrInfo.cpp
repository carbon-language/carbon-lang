//===- PPCInstrInfo.cpp - PowerPC32 Instruction Information -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PowerPC implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "PPCInstrInfo.h"
#include "PPCInstrBuilder.h"
#include "PPCMachineFunctionInfo.h"
#include "PPCPredicates.h"
#include "PPCGenInstrInfo.inc"
#include "PPCTargetMachine.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/MC/MCAsmInfo.h"

namespace llvm {
extern cl::opt<bool> EnablePPC32RS;  // FIXME (64-bit): See PPCRegisterInfo.cpp.
extern cl::opt<bool> EnablePPC64RS;  // FIXME (64-bit): See PPCRegisterInfo.cpp.
}

using namespace llvm;

PPCInstrInfo::PPCInstrInfo(PPCTargetMachine &tm)
  : TargetInstrInfoImpl(PPCInsts, array_lengthof(PPCInsts)), TM(tm),
    RI(*TM.getSubtargetImpl(), *this) {}

bool PPCInstrInfo::isMoveInstr(const MachineInstr& MI,
                               unsigned& sourceReg,
                               unsigned& destReg,
                               unsigned& sourceSubIdx,
                               unsigned& destSubIdx) const {
  sourceSubIdx = destSubIdx = 0; // No sub-registers.

  unsigned oc = MI.getOpcode();
  if (oc == PPC::OR || oc == PPC::OR8 || oc == PPC::VOR ||
      oc == PPC::OR4To8 || oc == PPC::OR8To4) {                // or r1, r2, r2
    assert(MI.getNumOperands() >= 3 &&
           MI.getOperand(0).isReg() &&
           MI.getOperand(1).isReg() &&
           MI.getOperand(2).isReg() &&
           "invalid PPC OR instruction!");
    if (MI.getOperand(1).getReg() == MI.getOperand(2).getReg()) {
      sourceReg = MI.getOperand(1).getReg();
      destReg = MI.getOperand(0).getReg();
      return true;
    }
  } else if (oc == PPC::ADDI) {             // addi r1, r2, 0
    assert(MI.getNumOperands() >= 3 &&
           MI.getOperand(0).isReg() &&
           MI.getOperand(2).isImm() &&
           "invalid PPC ADDI instruction!");
    if (MI.getOperand(1).isReg() && MI.getOperand(2).getImm() == 0) {
      sourceReg = MI.getOperand(1).getReg();
      destReg = MI.getOperand(0).getReg();
      return true;
    }
  } else if (oc == PPC::ORI) {             // ori r1, r2, 0
    assert(MI.getNumOperands() >= 3 &&
           MI.getOperand(0).isReg() &&
           MI.getOperand(1).isReg() &&
           MI.getOperand(2).isImm() &&
           "invalid PPC ORI instruction!");
    if (MI.getOperand(2).getImm() == 0) {
      sourceReg = MI.getOperand(1).getReg();
      destReg = MI.getOperand(0).getReg();
      return true;
    }
  } else if (oc == PPC::FMR || oc == PPC::FMRSD) { // fmr r1, r2
    assert(MI.getNumOperands() >= 2 &&
           MI.getOperand(0).isReg() &&
           MI.getOperand(1).isReg() &&
           "invalid PPC FMR instruction");
    sourceReg = MI.getOperand(1).getReg();
    destReg = MI.getOperand(0).getReg();
    return true;
  } else if (oc == PPC::MCRF) {             // mcrf cr1, cr2
    assert(MI.getNumOperands() >= 2 &&
           MI.getOperand(0).isReg() &&
           MI.getOperand(1).isReg() &&
           "invalid PPC MCRF instruction");
    sourceReg = MI.getOperand(1).getReg();
    destReg = MI.getOperand(0).getReg();
    return true;
  }
  return false;
}

unsigned PPCInstrInfo::isLoadFromStackSlot(const MachineInstr *MI, 
                                           int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case PPC::LD:
  case PPC::LWZ:
  case PPC::LFS:
  case PPC::LFD:
    if (MI->getOperand(1).isImm() && !MI->getOperand(1).getImm() &&
        MI->getOperand(2).isFI()) {
      FrameIndex = MI->getOperand(2).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

unsigned PPCInstrInfo::isStoreToStackSlot(const MachineInstr *MI, 
                                          int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case PPC::STD:
  case PPC::STW:
  case PPC::STFS:
  case PPC::STFD:
    if (MI->getOperand(1).isImm() && !MI->getOperand(1).getImm() &&
        MI->getOperand(2).isFI()) {
      FrameIndex = MI->getOperand(2).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

// commuteInstruction - We can commute rlwimi instructions, but only if the
// rotate amt is zero.  We also have to munge the immediates a bit.
MachineInstr *
PPCInstrInfo::commuteInstruction(MachineInstr *MI, bool NewMI) const {
  MachineFunction &MF = *MI->getParent()->getParent();

  // Normal instructions can be commuted the obvious way.
  if (MI->getOpcode() != PPC::RLWIMI)
    return TargetInstrInfoImpl::commuteInstruction(MI, NewMI);
  
  // Cannot commute if it has a non-zero rotate count.
  if (MI->getOperand(3).getImm() != 0)
    return 0;
  
  // If we have a zero rotate count, we have:
  //   M = mask(MB,ME)
  //   Op0 = (Op1 & ~M) | (Op2 & M)
  // Change this to:
  //   M = mask((ME+1)&31, (MB-1)&31)
  //   Op0 = (Op2 & ~M) | (Op1 & M)

  // Swap op1/op2
  unsigned Reg0 = MI->getOperand(0).getReg();
  unsigned Reg1 = MI->getOperand(1).getReg();
  unsigned Reg2 = MI->getOperand(2).getReg();
  bool Reg1IsKill = MI->getOperand(1).isKill();
  bool Reg2IsKill = MI->getOperand(2).isKill();
  bool ChangeReg0 = false;
  // If machine instrs are no longer in two-address forms, update
  // destination register as well.
  if (Reg0 == Reg1) {
    // Must be two address instruction!
    assert(MI->getDesc().getOperandConstraint(0, TOI::TIED_TO) &&
           "Expecting a two-address instruction!");
    Reg2IsKill = false;
    ChangeReg0 = true;
  }

  // Masks.
  unsigned MB = MI->getOperand(4).getImm();
  unsigned ME = MI->getOperand(5).getImm();

  if (NewMI) {
    // Create a new instruction.
    unsigned Reg0 = ChangeReg0 ? Reg2 : MI->getOperand(0).getReg();
    bool Reg0IsDead = MI->getOperand(0).isDead();
    return BuildMI(MF, MI->getDebugLoc(), MI->getDesc())
      .addReg(Reg0, RegState::Define | getDeadRegState(Reg0IsDead))
      .addReg(Reg2, getKillRegState(Reg2IsKill))
      .addReg(Reg1, getKillRegState(Reg1IsKill))
      .addImm((ME+1) & 31)
      .addImm((MB-1) & 31);
  }

  if (ChangeReg0)
    MI->getOperand(0).setReg(Reg2);
  MI->getOperand(2).setReg(Reg1);
  MI->getOperand(1).setReg(Reg2);
  MI->getOperand(2).setIsKill(Reg1IsKill);
  MI->getOperand(1).setIsKill(Reg2IsKill);
  
  // Swap the mask around.
  MI->getOperand(4).setImm((ME+1) & 31);
  MI->getOperand(5).setImm((MB-1) & 31);
  return MI;
}

void PPCInstrInfo::insertNoop(MachineBasicBlock &MBB, 
                              MachineBasicBlock::iterator MI) const {
  DebugLoc DL;
  BuildMI(MBB, MI, DL, get(PPC::NOP));
}


// Branch analysis.
bool PPCInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,MachineBasicBlock *&TBB,
                                 MachineBasicBlock *&FBB,
                                 SmallVectorImpl<MachineOperand> &Cond,
                                 bool AllowModify) const {
  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin())
    return false;
  --I;
  while (I->isDebugValue()) {
    if (I == MBB.begin())
      return false;
    --I;
  }
  if (!isUnpredicatedTerminator(I))
    return false;

  // Get the last instruction in the block.
  MachineInstr *LastInst = I;
  
  // If there is only one terminator instruction, process it.
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I)) {
    if (LastInst->getOpcode() == PPC::B) {
      if (!LastInst->getOperand(0).isMBB())
        return true;
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    } else if (LastInst->getOpcode() == PPC::BCC) {
      if (!LastInst->getOperand(2).isMBB())
        return true;
      // Block ends with fall-through condbranch.
      TBB = LastInst->getOperand(2).getMBB();
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
      isUnpredicatedTerminator(--I))
    return true;
  
  // If the block ends with PPC::B and PPC:BCC, handle it.
  if (SecondLastInst->getOpcode() == PPC::BCC && 
      LastInst->getOpcode() == PPC::B) {
    if (!SecondLastInst->getOperand(2).isMBB() ||
        !LastInst->getOperand(0).isMBB())
      return true;
    TBB =  SecondLastInst->getOperand(2).getMBB();
    Cond.push_back(SecondLastInst->getOperand(0));
    Cond.push_back(SecondLastInst->getOperand(1));
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }
  
  // If the block ends with two PPC:Bs, handle it.  The second one is not
  // executed, so remove it.
  if (SecondLastInst->getOpcode() == PPC::B && 
      LastInst->getOpcode() == PPC::B) {
    if (!SecondLastInst->getOperand(0).isMBB())
      return true;
    TBB = SecondLastInst->getOperand(0).getMBB();
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return false;
  }

  // Otherwise, can't handle this.
  return true;
}

unsigned PPCInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin()) return 0;
  --I;
  while (I->isDebugValue()) {
    if (I == MBB.begin())
      return 0;
    --I;
  }
  if (I->getOpcode() != PPC::B && I->getOpcode() != PPC::BCC)
    return 0;
  
  // Remove the branch.
  I->eraseFromParent();
  
  I = MBB.end();

  if (I == MBB.begin()) return 1;
  --I;
  if (I->getOpcode() != PPC::BCC)
    return 1;
  
  // Remove the branch.
  I->eraseFromParent();
  return 2;
}

unsigned
PPCInstrInfo::InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                           MachineBasicBlock *FBB,
                           const SmallVectorImpl<MachineOperand> &Cond) const {
  // FIXME this should probably have a DebugLoc argument
  DebugLoc dl;
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 2 || Cond.size() == 0) && 
         "PPC branch conditions have two components!");
  
  // One-way branch.
  if (FBB == 0) {
    if (Cond.empty())   // Unconditional branch
      BuildMI(&MBB, dl, get(PPC::B)).addMBB(TBB);
    else                // Conditional branch
      BuildMI(&MBB, dl, get(PPC::BCC))
        .addImm(Cond[0].getImm()).addReg(Cond[1].getReg()).addMBB(TBB);
    return 1;
  }
  
  // Two-way Conditional Branch.
  BuildMI(&MBB, dl, get(PPC::BCC))
    .addImm(Cond[0].getImm()).addReg(Cond[1].getReg()).addMBB(TBB);
  BuildMI(&MBB, dl, get(PPC::B)).addMBB(FBB);
  return 2;
}

bool PPCInstrInfo::copyRegToReg(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   unsigned DestReg, unsigned SrcReg,
                                   const TargetRegisterClass *DestRC,
                                   const TargetRegisterClass *SrcRC,
                                   DebugLoc DL) const {
  if (DestRC != SrcRC) {
    // Not yet supported!
    return false;
  }

  if (DestRC == PPC::GPRCRegisterClass) {
    BuildMI(MBB, MI, DL, get(PPC::OR), DestReg).addReg(SrcReg).addReg(SrcReg);
  } else if (DestRC == PPC::G8RCRegisterClass) {
    BuildMI(MBB, MI, DL, get(PPC::OR8), DestReg).addReg(SrcReg).addReg(SrcReg);
  } else if (DestRC == PPC::F4RCRegisterClass ||
             DestRC == PPC::F8RCRegisterClass) {
    BuildMI(MBB, MI, DL, get(PPC::FMR), DestReg).addReg(SrcReg);
  } else if (DestRC == PPC::CRRCRegisterClass) {
    BuildMI(MBB, MI, DL, get(PPC::MCRF), DestReg).addReg(SrcReg);
  } else if (DestRC == PPC::VRRCRegisterClass) {
    BuildMI(MBB, MI, DL, get(PPC::VOR), DestReg).addReg(SrcReg).addReg(SrcReg);
  } else if (DestRC == PPC::CRBITRCRegisterClass) {
    BuildMI(MBB, MI, DL, get(PPC::CROR), DestReg).addReg(SrcReg).addReg(SrcReg);
  } else {
    // Attempt to copy register that is not GPR or FPR
    return false;
  }
  
  return true;
}

bool
PPCInstrInfo::StoreRegToStackSlot(MachineFunction &MF,
                                  unsigned SrcReg, bool isKill,
                                  int FrameIdx,
                                  const TargetRegisterClass *RC,
                                  SmallVectorImpl<MachineInstr*> &NewMIs) const{
  DebugLoc DL;
  if (RC == PPC::GPRCRegisterClass) {
    if (SrcReg != PPC::LR) {
      NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::STW))
                                         .addReg(SrcReg,
                                                 getKillRegState(isKill)),
                                         FrameIdx));
    } else {
      // FIXME: this spills LR immediately to memory in one step.  To do this,
      // we use R11, which we know cannot be used in the prolog/epilog.  This is
      // a hack.
      NewMIs.push_back(BuildMI(MF, DL, get(PPC::MFLR), PPC::R11));
      NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::STW))
                                         .addReg(PPC::R11,
                                                 getKillRegState(isKill)),
                                         FrameIdx));
    }
  } else if (RC == PPC::G8RCRegisterClass) {
    if (SrcReg != PPC::LR8) {
      NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::STD))
                                         .addReg(SrcReg,
                                                 getKillRegState(isKill)),
                                         FrameIdx));
    } else {
      // FIXME: this spills LR immediately to memory in one step.  To do this,
      // we use R11, which we know cannot be used in the prolog/epilog.  This is
      // a hack.
      NewMIs.push_back(BuildMI(MF, DL, get(PPC::MFLR8), PPC::X11));
      NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::STD))
                                         .addReg(PPC::X11,
                                                 getKillRegState(isKill)),
                                         FrameIdx));
    }
  } else if (RC == PPC::F8RCRegisterClass) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::STFD))
                                       .addReg(SrcReg,
                                               getKillRegState(isKill)),
                                       FrameIdx));
  } else if (RC == PPC::F4RCRegisterClass) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::STFS))
                                       .addReg(SrcReg,
                                               getKillRegState(isKill)),
                                       FrameIdx));
  } else if (RC == PPC::CRRCRegisterClass) {
    if ((EnablePPC32RS && !TM.getSubtargetImpl()->isPPC64()) ||
        (EnablePPC64RS && TM.getSubtargetImpl()->isPPC64())) {
      // FIXME (64-bit): Enable
      NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::SPILL_CR))
                                         .addReg(SrcReg,
                                                 getKillRegState(isKill)),
                                         FrameIdx));
      return true;
    } else {
      // FIXME: We need a scatch reg here.  The trouble with using R0 is that
      // it's possible for the stack frame to be so big the save location is
      // out of range of immediate offsets, necessitating another register.
      // We hack this on Darwin by reserving R2.  It's probably broken on Linux
      // at the moment.

      // We need to store the CR in the low 4-bits of the saved value.  First,
      // issue a MFCR to save all of the CRBits.
      unsigned ScratchReg = TM.getSubtargetImpl()->isDarwinABI() ? 
                                                           PPC::R2 : PPC::R0;
      NewMIs.push_back(BuildMI(MF, DL, get(PPC::MFCR), ScratchReg));
    
      // If the saved register wasn't CR0, shift the bits left so that they are
      // in CR0's slot.
      if (SrcReg != PPC::CR0) {
        unsigned ShiftBits = PPCRegisterInfo::getRegisterNumbering(SrcReg)*4;
        // rlwinm scratch, scratch, ShiftBits, 0, 31.
        NewMIs.push_back(BuildMI(MF, DL, get(PPC::RLWINM), ScratchReg)
                       .addReg(ScratchReg).addImm(ShiftBits)
                       .addImm(0).addImm(31));
      }
    
      NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::STW))
                                         .addReg(ScratchReg,
                                                 getKillRegState(isKill)),
                                         FrameIdx));
    }
  } else if (RC == PPC::CRBITRCRegisterClass) {
    // FIXME: We use CRi here because there is no mtcrf on a bit. Since the
    // backend currently only uses CR1EQ as an individual bit, this should
    // not cause any bug. If we need other uses of CR bits, the following
    // code may be invalid.
    unsigned Reg = 0;
    if (SrcReg == PPC::CR0LT || SrcReg == PPC::CR0GT ||
        SrcReg == PPC::CR0EQ || SrcReg == PPC::CR0UN)
      Reg = PPC::CR0;
    else if (SrcReg == PPC::CR1LT || SrcReg == PPC::CR1GT ||
             SrcReg == PPC::CR1EQ || SrcReg == PPC::CR1UN)
      Reg = PPC::CR1;
    else if (SrcReg == PPC::CR2LT || SrcReg == PPC::CR2GT ||
             SrcReg == PPC::CR2EQ || SrcReg == PPC::CR2UN)
      Reg = PPC::CR2;
    else if (SrcReg == PPC::CR3LT || SrcReg == PPC::CR3GT ||
             SrcReg == PPC::CR3EQ || SrcReg == PPC::CR3UN)
      Reg = PPC::CR3;
    else if (SrcReg == PPC::CR4LT || SrcReg == PPC::CR4GT ||
             SrcReg == PPC::CR4EQ || SrcReg == PPC::CR4UN)
      Reg = PPC::CR4;
    else if (SrcReg == PPC::CR5LT || SrcReg == PPC::CR5GT ||
             SrcReg == PPC::CR5EQ || SrcReg == PPC::CR5UN)
      Reg = PPC::CR5;
    else if (SrcReg == PPC::CR6LT || SrcReg == PPC::CR6GT ||
             SrcReg == PPC::CR6EQ || SrcReg == PPC::CR6UN)
      Reg = PPC::CR6;
    else if (SrcReg == PPC::CR7LT || SrcReg == PPC::CR7GT ||
             SrcReg == PPC::CR7EQ || SrcReg == PPC::CR7UN)
      Reg = PPC::CR7;

    return StoreRegToStackSlot(MF, Reg, isKill, FrameIdx, 
                               PPC::CRRCRegisterClass, NewMIs);

  } else if (RC == PPC::VRRCRegisterClass) {
    // We don't have indexed addressing for vector loads.  Emit:
    // R0 = ADDI FI#
    // STVX VAL, 0, R0
    // 
    // FIXME: We use R0 here, because it isn't available for RA.
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::ADDI), PPC::R0),
                                       FrameIdx, 0, 0));
    NewMIs.push_back(BuildMI(MF, DL, get(PPC::STVX))
                     .addReg(SrcReg, getKillRegState(isKill))
                     .addReg(PPC::R0)
                     .addReg(PPC::R0));
  } else {
    llvm_unreachable("Unknown regclass!");
  }

  return false;
}

void
PPCInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator MI,
                                  unsigned SrcReg, bool isKill, int FrameIdx,
                                  const TargetRegisterClass *RC,
                                  const TargetRegisterInfo *TRI) const {
  MachineFunction &MF = *MBB.getParent();
  SmallVector<MachineInstr*, 4> NewMIs;

  if (StoreRegToStackSlot(MF, SrcReg, isKill, FrameIdx, RC, NewMIs)) {
    PPCFunctionInfo *FuncInfo = MF.getInfo<PPCFunctionInfo>();
    FuncInfo->setSpillsCR();
  }

  for (unsigned i = 0, e = NewMIs.size(); i != e; ++i)
    MBB.insert(MI, NewMIs[i]);
}

void
PPCInstrInfo::LoadRegFromStackSlot(MachineFunction &MF, DebugLoc DL,
                                   unsigned DestReg, int FrameIdx,
                                   const TargetRegisterClass *RC,
                                   SmallVectorImpl<MachineInstr*> &NewMIs)const{
  if (RC == PPC::GPRCRegisterClass) {
    if (DestReg != PPC::LR) {
      NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::LWZ),
                                                 DestReg), FrameIdx));
    } else {
      NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::LWZ),
                                                 PPC::R11), FrameIdx));
      NewMIs.push_back(BuildMI(MF, DL, get(PPC::MTLR)).addReg(PPC::R11));
    }
  } else if (RC == PPC::G8RCRegisterClass) {
    if (DestReg != PPC::LR8) {
      NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::LD), DestReg),
                                         FrameIdx));
    } else {
      NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::LD),
                                                 PPC::R11), FrameIdx));
      NewMIs.push_back(BuildMI(MF, DL, get(PPC::MTLR8)).addReg(PPC::R11));
    }
  } else if (RC == PPC::F8RCRegisterClass) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::LFD), DestReg),
                                       FrameIdx));
  } else if (RC == PPC::F4RCRegisterClass) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::LFS), DestReg),
                                       FrameIdx));
  } else if (RC == PPC::CRRCRegisterClass) {
    // FIXME: We need a scatch reg here.  The trouble with using R0 is that
    // it's possible for the stack frame to be so big the save location is
    // out of range of immediate offsets, necessitating another register.
    // We hack this on Darwin by reserving R2.  It's probably broken on Linux
    // at the moment.
    unsigned ScratchReg = TM.getSubtargetImpl()->isDarwinABI() ?
                                                          PPC::R2 : PPC::R0;
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::LWZ), 
                                       ScratchReg), FrameIdx));
    
    // If the reloaded register isn't CR0, shift the bits right so that they are
    // in the right CR's slot.
    if (DestReg != PPC::CR0) {
      unsigned ShiftBits = PPCRegisterInfo::getRegisterNumbering(DestReg)*4;
      // rlwinm r11, r11, 32-ShiftBits, 0, 31.
      NewMIs.push_back(BuildMI(MF, DL, get(PPC::RLWINM), ScratchReg)
                    .addReg(ScratchReg).addImm(32-ShiftBits).addImm(0)
                    .addImm(31));
    }
    
    NewMIs.push_back(BuildMI(MF, DL, get(PPC::MTCRF), DestReg)
                     .addReg(ScratchReg));
  } else if (RC == PPC::CRBITRCRegisterClass) {
   
    unsigned Reg = 0;
    if (DestReg == PPC::CR0LT || DestReg == PPC::CR0GT ||
        DestReg == PPC::CR0EQ || DestReg == PPC::CR0UN)
      Reg = PPC::CR0;
    else if (DestReg == PPC::CR1LT || DestReg == PPC::CR1GT ||
             DestReg == PPC::CR1EQ || DestReg == PPC::CR1UN)
      Reg = PPC::CR1;
    else if (DestReg == PPC::CR2LT || DestReg == PPC::CR2GT ||
             DestReg == PPC::CR2EQ || DestReg == PPC::CR2UN)
      Reg = PPC::CR2;
    else if (DestReg == PPC::CR3LT || DestReg == PPC::CR3GT ||
             DestReg == PPC::CR3EQ || DestReg == PPC::CR3UN)
      Reg = PPC::CR3;
    else if (DestReg == PPC::CR4LT || DestReg == PPC::CR4GT ||
             DestReg == PPC::CR4EQ || DestReg == PPC::CR4UN)
      Reg = PPC::CR4;
    else if (DestReg == PPC::CR5LT || DestReg == PPC::CR5GT ||
             DestReg == PPC::CR5EQ || DestReg == PPC::CR5UN)
      Reg = PPC::CR5;
    else if (DestReg == PPC::CR6LT || DestReg == PPC::CR6GT ||
             DestReg == PPC::CR6EQ || DestReg == PPC::CR6UN)
      Reg = PPC::CR6;
    else if (DestReg == PPC::CR7LT || DestReg == PPC::CR7GT ||
             DestReg == PPC::CR7EQ || DestReg == PPC::CR7UN)
      Reg = PPC::CR7;

    return LoadRegFromStackSlot(MF, DL, Reg, FrameIdx, 
                                PPC::CRRCRegisterClass, NewMIs);

  } else if (RC == PPC::VRRCRegisterClass) {
    // We don't have indexed addressing for vector loads.  Emit:
    // R0 = ADDI FI#
    // Dest = LVX 0, R0
    // 
    // FIXME: We use R0 here, because it isn't available for RA.
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::ADDI), PPC::R0),
                                       FrameIdx, 0, 0));
    NewMIs.push_back(BuildMI(MF, DL, get(PPC::LVX),DestReg).addReg(PPC::R0)
                     .addReg(PPC::R0));
  } else {
    llvm_unreachable("Unknown regclass!");
  }
}

void
PPCInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   unsigned DestReg, int FrameIdx,
                                   const TargetRegisterClass *RC,
                                   const TargetRegisterInfo *TRI) const {
  MachineFunction &MF = *MBB.getParent();
  SmallVector<MachineInstr*, 4> NewMIs;
  DebugLoc DL;
  if (MI != MBB.end()) DL = MI->getDebugLoc();
  LoadRegFromStackSlot(MF, DL, DestReg, FrameIdx, RC, NewMIs);
  for (unsigned i = 0, e = NewMIs.size(); i != e; ++i)
    MBB.insert(MI, NewMIs[i]);
}

MachineInstr*
PPCInstrInfo::emitFrameIndexDebugValue(MachineFunction &MF,
                                       int FrameIx, uint64_t Offset,
                                       const MDNode *MDPtr,
                                       DebugLoc DL) const {
  MachineInstrBuilder MIB = BuildMI(MF, DL, get(PPC::DBG_VALUE));
  addFrameReference(MIB, FrameIx, 0, false).addImm(Offset).addMetadata(MDPtr);
  return &*MIB;
}

/// foldMemoryOperand - PowerPC (like most RISC's) can only fold spills into
/// copy instructions, turning them into load/store instructions.
MachineInstr *PPCInstrInfo::foldMemoryOperandImpl(MachineFunction &MF,
                                                  MachineInstr *MI,
                                           const SmallVectorImpl<unsigned> &Ops,
                                                  int FrameIndex) const {
  if (Ops.size() != 1) return NULL;

  // Make sure this is a reg-reg copy.  Note that we can't handle MCRF, because
  // it takes more than one instruction to store it.
  unsigned Opc = MI->getOpcode();
  unsigned OpNum = Ops[0];

  MachineInstr *NewMI = NULL;
  if ((Opc == PPC::OR &&
       MI->getOperand(1).getReg() == MI->getOperand(2).getReg())) {
    if (OpNum == 0) {  // move -> store
      unsigned InReg = MI->getOperand(1).getReg();
      bool isKill = MI->getOperand(1).isKill();
      bool isUndef = MI->getOperand(1).isUndef();
      NewMI = addFrameReference(BuildMI(MF, MI->getDebugLoc(), get(PPC::STW))
                                .addReg(InReg,
                                        getKillRegState(isKill) |
                                        getUndefRegState(isUndef)),
                                FrameIndex);
    } else {           // move -> load
      unsigned OutReg = MI->getOperand(0).getReg();
      bool isDead = MI->getOperand(0).isDead();
      bool isUndef = MI->getOperand(0).isUndef();
      NewMI = addFrameReference(BuildMI(MF, MI->getDebugLoc(), get(PPC::LWZ))
                                .addReg(OutReg,
                                        RegState::Define |
                                        getDeadRegState(isDead) |
                                        getUndefRegState(isUndef)),
                                FrameIndex);
    }
  } else if ((Opc == PPC::OR8 &&
              MI->getOperand(1).getReg() == MI->getOperand(2).getReg())) {
    if (OpNum == 0) {  // move -> store
      unsigned InReg = MI->getOperand(1).getReg();
      bool isKill = MI->getOperand(1).isKill();
      bool isUndef = MI->getOperand(1).isUndef();
      NewMI = addFrameReference(BuildMI(MF, MI->getDebugLoc(), get(PPC::STD))
                                .addReg(InReg,
                                        getKillRegState(isKill) |
                                        getUndefRegState(isUndef)),
                                FrameIndex);
    } else {           // move -> load
      unsigned OutReg = MI->getOperand(0).getReg();
      bool isDead = MI->getOperand(0).isDead();
      bool isUndef = MI->getOperand(0).isUndef();
      NewMI = addFrameReference(BuildMI(MF, MI->getDebugLoc(), get(PPC::LD))
                                .addReg(OutReg,
                                        RegState::Define |
                                        getDeadRegState(isDead) |
                                        getUndefRegState(isUndef)),
                                FrameIndex);
    }
  } else if (Opc == PPC::FMR || Opc == PPC::FMRSD) {
    // The register may be F4RC or F8RC, and that determines the memory op.
    unsigned OrigReg = MI->getOperand(OpNum).getReg();
    // We cannot tell the register class from a physreg alone.
    if (TargetRegisterInfo::isPhysicalRegister(OrigReg))
      return NULL;
    const TargetRegisterClass *RC = MF.getRegInfo().getRegClass(OrigReg);
    const bool is64 = RC == PPC::F8RCRegisterClass;

    if (OpNum == 0) {  // move -> store
      unsigned InReg = MI->getOperand(1).getReg();
      bool isKill = MI->getOperand(1).isKill();
      bool isUndef = MI->getOperand(1).isUndef();
      NewMI = addFrameReference(BuildMI(MF, MI->getDebugLoc(),
                                        get(is64 ? PPC::STFD : PPC::STFS))
                                .addReg(InReg,
                                        getKillRegState(isKill) |
                                        getUndefRegState(isUndef)),
                                FrameIndex);
    } else {           // move -> load
      unsigned OutReg = MI->getOperand(0).getReg();
      bool isDead = MI->getOperand(0).isDead();
      bool isUndef = MI->getOperand(0).isUndef();
      NewMI = addFrameReference(BuildMI(MF, MI->getDebugLoc(),
                                        get(is64 ? PPC::LFD : PPC::LFS))
                                .addReg(OutReg,
                                        RegState::Define |
                                        getDeadRegState(isDead) |
                                        getUndefRegState(isUndef)),
                                FrameIndex);
    }
  }

  return NewMI;
}

bool PPCInstrInfo::canFoldMemoryOperand(const MachineInstr *MI,
                                  const SmallVectorImpl<unsigned> &Ops) const {
  if (Ops.size() != 1) return false;

  // Make sure this is a reg-reg copy.  Note that we can't handle MCRF, because
  // it takes more than one instruction to store it.
  unsigned Opc = MI->getOpcode();

  if ((Opc == PPC::OR &&
       MI->getOperand(1).getReg() == MI->getOperand(2).getReg()))
    return true;
  else if ((Opc == PPC::OR8 &&
              MI->getOperand(1).getReg() == MI->getOperand(2).getReg()))
    return true;
  else if (Opc == PPC::FMR || Opc == PPC::FMRSD)
    return true;

  return false;
}


bool PPCInstrInfo::
ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const {
  assert(Cond.size() == 2 && "Invalid PPC branch opcode!");
  // Leave the CR# the same, but invert the condition.
  Cond[0].setImm(PPC::InvertPredicate((PPC::Predicate)Cond[0].getImm()));
  return false;
}

/// GetInstSize - Return the number of bytes of code the specified
/// instruction may be.  This returns the maximum number of bytes.
///
unsigned PPCInstrInfo::GetInstSizeInBytes(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
  case PPC::INLINEASM: {       // Inline Asm: Variable size.
    const MachineFunction *MF = MI->getParent()->getParent();
    const char *AsmStr = MI->getOperand(0).getSymbolName();
    return getInlineAsmLength(AsmStr, *MF->getTarget().getMCAsmInfo());
  }
  case PPC::DBG_LABEL:
  case PPC::EH_LABEL:
  case PPC::GC_LABEL:
  case PPC::DBG_VALUE:
    return 0;
  default:
    return 4; // PowerPC instructions are all 4 bytes
  }
}
