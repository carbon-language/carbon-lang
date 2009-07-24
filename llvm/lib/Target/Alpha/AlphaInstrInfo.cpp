//===- AlphaInstrInfo.cpp - Alpha Instruction Information -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Alpha implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "Alpha.h"
#include "AlphaInstrInfo.h"
#include "AlphaMachineFunctionInfo.h"
#include "AlphaGenInstrInfo.inc"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;

AlphaInstrInfo::AlphaInstrInfo()
  : TargetInstrInfoImpl(AlphaInsts, array_lengthof(AlphaInsts)),
    RI(*this) { }


bool AlphaInstrInfo::isMoveInstr(const MachineInstr& MI,
                                 unsigned& sourceReg, unsigned& destReg,
                                 unsigned& SrcSR, unsigned& DstSR) const {
  unsigned oc = MI.getOpcode();
  if (oc == Alpha::BISr   || 
      oc == Alpha::CPYSS  || 
      oc == Alpha::CPYST  ||
      oc == Alpha::CPYSSt || 
      oc == Alpha::CPYSTs) {
    // or r1, r2, r2 
    // cpys(s|t) r1 r2 r2
    assert(MI.getNumOperands() >= 3 &&
           MI.getOperand(0).isReg() &&
           MI.getOperand(1).isReg() &&
           MI.getOperand(2).isReg() &&
           "invalid Alpha BIS instruction!");
    if (MI.getOperand(1).getReg() == MI.getOperand(2).getReg()) {
      sourceReg = MI.getOperand(1).getReg();
      destReg = MI.getOperand(0).getReg();
      SrcSR = DstSR = 0;
      return true;
    }
  }
  return false;
}

unsigned 
AlphaInstrInfo::isLoadFromStackSlot(const MachineInstr *MI,
                                    int &FrameIndex) const {
  switch (MI->getOpcode()) {
  case Alpha::LDL:
  case Alpha::LDQ:
  case Alpha::LDBU:
  case Alpha::LDWU:
  case Alpha::LDS:
  case Alpha::LDT:
    if (MI->getOperand(1).isFI()) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

unsigned 
AlphaInstrInfo::isStoreToStackSlot(const MachineInstr *MI,
                                   int &FrameIndex) const {
  switch (MI->getOpcode()) {
  case Alpha::STL:
  case Alpha::STQ:
  case Alpha::STB:
  case Alpha::STW:
  case Alpha::STS:
  case Alpha::STT:
    if (MI->getOperand(1).isFI()) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

static bool isAlphaIntCondCode(unsigned Opcode) {
  switch (Opcode) {
  case Alpha::BEQ: 
  case Alpha::BNE: 
  case Alpha::BGE: 
  case Alpha::BGT: 
  case Alpha::BLE: 
  case Alpha::BLT: 
  case Alpha::BLBC: 
  case Alpha::BLBS:
    return true;
  default:
    return false;
  }
}

unsigned AlphaInstrInfo::InsertBranch(MachineBasicBlock &MBB,
                                      MachineBasicBlock *TBB,
                                      MachineBasicBlock *FBB,
                            const SmallVectorImpl<MachineOperand> &Cond) const {
  // FIXME this should probably have a DebugLoc argument
  DebugLoc dl = DebugLoc::getUnknownLoc();
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 2 || Cond.size() == 0) && 
         "Alpha branch conditions have two components!");

  // One-way branch.
  if (FBB == 0) {
    if (Cond.empty())   // Unconditional branch
      BuildMI(&MBB, dl, get(Alpha::BR)).addMBB(TBB);
    else                // Conditional branch
      if (isAlphaIntCondCode(Cond[0].getImm()))
        BuildMI(&MBB, dl, get(Alpha::COND_BRANCH_I))
          .addImm(Cond[0].getImm()).addReg(Cond[1].getReg()).addMBB(TBB);
      else
        BuildMI(&MBB, dl, get(Alpha::COND_BRANCH_F))
          .addImm(Cond[0].getImm()).addReg(Cond[1].getReg()).addMBB(TBB);
    return 1;
  }
  
  // Two-way Conditional Branch.
  if (isAlphaIntCondCode(Cond[0].getImm()))
    BuildMI(&MBB, dl, get(Alpha::COND_BRANCH_I))
      .addImm(Cond[0].getImm()).addReg(Cond[1].getReg()).addMBB(TBB);
  else
    BuildMI(&MBB, dl, get(Alpha::COND_BRANCH_F))
      .addImm(Cond[0].getImm()).addReg(Cond[1].getReg()).addMBB(TBB);
  BuildMI(&MBB, dl, get(Alpha::BR)).addMBB(FBB);
  return 2;
}

bool AlphaInstrInfo::copyRegToReg(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator MI,
                                  unsigned DestReg, unsigned SrcReg,
                                  const TargetRegisterClass *DestRC,
                                  const TargetRegisterClass *SrcRC) const {
  //cerr << "copyRegToReg " << DestReg << " <- " << SrcReg << "\n";
  if (DestRC != SrcRC) {
    // Not yet supported!
    return false;
  }

  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (MI != MBB.end()) DL = MI->getDebugLoc();

  if (DestRC == Alpha::GPRCRegisterClass) {
    BuildMI(MBB, MI, DL, get(Alpha::BISr), DestReg)
      .addReg(SrcReg)
      .addReg(SrcReg);
  } else if (DestRC == Alpha::F4RCRegisterClass) {
    BuildMI(MBB, MI, DL, get(Alpha::CPYSS), DestReg)
      .addReg(SrcReg)
      .addReg(SrcReg);
  } else if (DestRC == Alpha::F8RCRegisterClass) {
    BuildMI(MBB, MI, DL, get(Alpha::CPYST), DestReg)
      .addReg(SrcReg)
      .addReg(SrcReg);
  } else {
    // Attempt to copy register that is not GPR or FPR
    return false;
  }
  
  return true;
}

void
AlphaInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MI,
                                    unsigned SrcReg, bool isKill, int FrameIdx,
                                    const TargetRegisterClass *RC) const {
  //cerr << "Trying to store " << getPrettyName(SrcReg) << " to "
  //     << FrameIdx << "\n";
  //BuildMI(MBB, MI, Alpha::WTF, 0).addReg(SrcReg);

  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (MI != MBB.end()) DL = MI->getDebugLoc();

  if (RC == Alpha::F4RCRegisterClass)
    BuildMI(MBB, MI, DL, get(Alpha::STS))
      .addReg(SrcReg, getKillRegState(isKill))
      .addFrameIndex(FrameIdx).addReg(Alpha::F31);
  else if (RC == Alpha::F8RCRegisterClass)
    BuildMI(MBB, MI, DL, get(Alpha::STT))
      .addReg(SrcReg, getKillRegState(isKill))
      .addFrameIndex(FrameIdx).addReg(Alpha::F31);
  else if (RC == Alpha::GPRCRegisterClass)
    BuildMI(MBB, MI, DL, get(Alpha::STQ))
      .addReg(SrcReg, getKillRegState(isKill))
      .addFrameIndex(FrameIdx).addReg(Alpha::F31);
  else
    llvm_unreachable("Unhandled register class");
}

void
AlphaInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MI,
                                        unsigned DestReg, int FrameIdx,
                                        const TargetRegisterClass *RC) const {
  //cerr << "Trying to load " << getPrettyName(DestReg) << " to "
  //     << FrameIdx << "\n";
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (MI != MBB.end()) DL = MI->getDebugLoc();

  if (RC == Alpha::F4RCRegisterClass)
    BuildMI(MBB, MI, DL, get(Alpha::LDS), DestReg)
      .addFrameIndex(FrameIdx).addReg(Alpha::F31);
  else if (RC == Alpha::F8RCRegisterClass)
    BuildMI(MBB, MI, DL, get(Alpha::LDT), DestReg)
      .addFrameIndex(FrameIdx).addReg(Alpha::F31);
  else if (RC == Alpha::GPRCRegisterClass)
    BuildMI(MBB, MI, DL, get(Alpha::LDQ), DestReg)
      .addFrameIndex(FrameIdx).addReg(Alpha::F31);
  else
    llvm_unreachable("Unhandled register class");
}

MachineInstr *AlphaInstrInfo::foldMemoryOperandImpl(MachineFunction &MF,
                                                    MachineInstr *MI,
                                          const SmallVectorImpl<unsigned> &Ops,
                                                    int FrameIndex) const {
   if (Ops.size() != 1) return NULL;

   // Make sure this is a reg-reg copy.
   unsigned Opc = MI->getOpcode();

   MachineInstr *NewMI = NULL;
   switch(Opc) {
   default:
     break;
   case Alpha::BISr:
   case Alpha::CPYSS:
   case Alpha::CPYST:
     if (MI->getOperand(1).getReg() == MI->getOperand(2).getReg()) {
       if (Ops[0] == 0) {  // move -> store
         unsigned InReg = MI->getOperand(1).getReg();
         bool isKill = MI->getOperand(1).isKill();
         bool isUndef = MI->getOperand(1).isUndef();
         Opc = (Opc == Alpha::BISr) ? Alpha::STQ : 
           ((Opc == Alpha::CPYSS) ? Alpha::STS : Alpha::STT);
         NewMI = BuildMI(MF, MI->getDebugLoc(), get(Opc))
           .addReg(InReg, getKillRegState(isKill) | getUndefRegState(isUndef))
           .addFrameIndex(FrameIndex)
           .addReg(Alpha::F31);
       } else {           // load -> move
         unsigned OutReg = MI->getOperand(0).getReg();
         bool isDead = MI->getOperand(0).isDead();
         bool isUndef = MI->getOperand(0).isUndef();
         Opc = (Opc == Alpha::BISr) ? Alpha::LDQ : 
           ((Opc == Alpha::CPYSS) ? Alpha::LDS : Alpha::LDT);
         NewMI = BuildMI(MF, MI->getDebugLoc(), get(Opc))
           .addReg(OutReg, RegState::Define | getDeadRegState(isDead) |
                   getUndefRegState(isUndef))
           .addFrameIndex(FrameIndex)
           .addReg(Alpha::F31);
       }
     }
     break;
   }
  return NewMI;
}

static unsigned AlphaRevCondCode(unsigned Opcode) {
  switch (Opcode) {
  case Alpha::BEQ: return Alpha::BNE;
  case Alpha::BNE: return Alpha::BEQ;
  case Alpha::BGE: return Alpha::BLT;
  case Alpha::BGT: return Alpha::BLE;
  case Alpha::BLE: return Alpha::BGT;
  case Alpha::BLT: return Alpha::BGE;
  case Alpha::BLBC: return Alpha::BLBS;
  case Alpha::BLBS: return Alpha::BLBC;
  case Alpha::FBEQ: return Alpha::FBNE;
  case Alpha::FBNE: return Alpha::FBEQ;
  case Alpha::FBGE: return Alpha::FBLT;
  case Alpha::FBGT: return Alpha::FBLE;
  case Alpha::FBLE: return Alpha::FBGT;
  case Alpha::FBLT: return Alpha::FBGE;
  default:
    llvm_unreachable("Unknown opcode");
  }
  return 0; // Not reached
}

// Branch analysis.
bool AlphaInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,MachineBasicBlock *&TBB,
                                   MachineBasicBlock *&FBB,
                                   SmallVectorImpl<MachineOperand> &Cond,
                                   bool AllowModify) const {
  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I))
    return false;

  // Get the last instruction in the block.
  MachineInstr *LastInst = I;
  
  // If there is only one terminator instruction, process it.
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I)) {
    if (LastInst->getOpcode() == Alpha::BR) {
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    } else if (LastInst->getOpcode() == Alpha::COND_BRANCH_I ||
               LastInst->getOpcode() == Alpha::COND_BRANCH_F) {
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
  
  // If the block ends with Alpha::BR and Alpha::COND_BRANCH_*, handle it.
  if ((SecondLastInst->getOpcode() == Alpha::COND_BRANCH_I ||
      SecondLastInst->getOpcode() == Alpha::COND_BRANCH_F) && 
      LastInst->getOpcode() == Alpha::BR) {
    TBB =  SecondLastInst->getOperand(2).getMBB();
    Cond.push_back(SecondLastInst->getOperand(0));
    Cond.push_back(SecondLastInst->getOperand(1));
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }
  
  // If the block ends with two Alpha::BRs, handle it.  The second one is not
  // executed, so remove it.
  if (SecondLastInst->getOpcode() == Alpha::BR && 
      LastInst->getOpcode() == Alpha::BR) {
    TBB = SecondLastInst->getOperand(0).getMBB();
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return false;
  }

  // Otherwise, can't handle this.
  return true;
}

unsigned AlphaInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin()) return 0;
  --I;
  if (I->getOpcode() != Alpha::BR && 
      I->getOpcode() != Alpha::COND_BRANCH_I &&
      I->getOpcode() != Alpha::COND_BRANCH_F)
    return 0;
  
  // Remove the branch.
  I->eraseFromParent();
  
  I = MBB.end();

  if (I == MBB.begin()) return 1;
  --I;
  if (I->getOpcode() != Alpha::COND_BRANCH_I && 
      I->getOpcode() != Alpha::COND_BRANCH_F)
    return 1;
  
  // Remove the branch.
  I->eraseFromParent();
  return 2;
}

void AlphaInstrInfo::insertNoop(MachineBasicBlock &MBB, 
                                MachineBasicBlock::iterator MI) const {
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (MI != MBB.end()) DL = MI->getDebugLoc();
  BuildMI(MBB, MI, DL, get(Alpha::BISr), Alpha::R31)
    .addReg(Alpha::R31)
    .addReg(Alpha::R31);
}

bool AlphaInstrInfo::BlockHasNoFallThrough(const MachineBasicBlock &MBB) const {
  if (MBB.empty()) return false;
  
  switch (MBB.back().getOpcode()) {
  case Alpha::RETDAG: // Return.
  case Alpha::RETDAGp:
  case Alpha::BR:     // Uncond branch.
  case Alpha::JMP:  // Indirect branch.
    return true;
  default: return false;
  }
}
bool AlphaInstrInfo::
ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const {
  assert(Cond.size() == 2 && "Invalid Alpha branch opcode!");
  Cond[0].setImm(AlphaRevCondCode(Cond[0].getImm()));
  return false;
}

/// getGlobalBaseReg - Return a virtual register initialized with the
/// the global base register value. Output instructions required to
/// initialize the register in the function entry block, if necessary.
///
unsigned AlphaInstrInfo::getGlobalBaseReg(MachineFunction *MF) const {
  AlphaMachineFunctionInfo *AlphaFI = MF->getInfo<AlphaMachineFunctionInfo>();
  unsigned GlobalBaseReg = AlphaFI->getGlobalBaseReg();
  if (GlobalBaseReg != 0)
    return GlobalBaseReg;

  // Insert the set of GlobalBaseReg into the first MBB of the function
  MachineBasicBlock &FirstMBB = MF->front();
  MachineBasicBlock::iterator MBBI = FirstMBB.begin();
  MachineRegisterInfo &RegInfo = MF->getRegInfo();
  const TargetInstrInfo *TII = MF->getTarget().getInstrInfo();

  GlobalBaseReg = RegInfo.createVirtualRegister(&Alpha::GPRCRegClass);
  bool Ok = TII->copyRegToReg(FirstMBB, MBBI, GlobalBaseReg, Alpha::R29,
                              &Alpha::GPRCRegClass, &Alpha::GPRCRegClass);
  assert(Ok && "Couldn't assign to global base register!");
  Ok = Ok; // Silence warning when assertions are turned off.
  RegInfo.addLiveIn(Alpha::R29);

  AlphaFI->setGlobalBaseReg(GlobalBaseReg);
  return GlobalBaseReg;
}

/// getGlobalRetAddr - Return a virtual register initialized with the
/// the global base register value. Output instructions required to
/// initialize the register in the function entry block, if necessary.
///
unsigned AlphaInstrInfo::getGlobalRetAddr(MachineFunction *MF) const {
  AlphaMachineFunctionInfo *AlphaFI = MF->getInfo<AlphaMachineFunctionInfo>();
  unsigned GlobalRetAddr = AlphaFI->getGlobalRetAddr();
  if (GlobalRetAddr != 0)
    return GlobalRetAddr;

  // Insert the set of GlobalRetAddr into the first MBB of the function
  MachineBasicBlock &FirstMBB = MF->front();
  MachineBasicBlock::iterator MBBI = FirstMBB.begin();
  MachineRegisterInfo &RegInfo = MF->getRegInfo();
  const TargetInstrInfo *TII = MF->getTarget().getInstrInfo();

  GlobalRetAddr = RegInfo.createVirtualRegister(&Alpha::GPRCRegClass);
  bool Ok = TII->copyRegToReg(FirstMBB, MBBI, GlobalRetAddr, Alpha::R26,
                              &Alpha::GPRCRegClass, &Alpha::GPRCRegClass);
  assert(Ok && "Couldn't assign to global return address register!");
  Ok = Ok; // Silence warning when assertions are turned off.
  RegInfo.addLiveIn(Alpha::R26);

  AlphaFI->setGlobalRetAddr(GlobalRetAddr);
  return GlobalRetAddr;
}
