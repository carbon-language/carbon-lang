//===-- RISCVInstrInfo.cpp - RISCV Instruction Information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the RISCV implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "RISCVInstrInfo.h"
#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "RISCVTargetMachine.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_INSTRINFO_CTOR_DTOR
#include "RISCVGenInstrInfo.inc"

using namespace llvm;

RISCVInstrInfo::RISCVInstrInfo()
    : RISCVGenInstrInfo(RISCV::ADJCALLSTACKDOWN, RISCV::ADJCALLSTACKUP) {}

void RISCVInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MBBI,
                                 const DebugLoc &DL, unsigned DstReg,
                                 unsigned SrcReg, bool KillSrc) const {
  assert(RISCV::GPRRegClass.contains(DstReg, SrcReg) &&
         "Impossible reg-to-reg copy");

  BuildMI(MBB, MBBI, DL, get(RISCV::ADDI), DstReg)
      .addReg(SrcReg, getKillRegState(KillSrc))
      .addImm(0);
}

void RISCVInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator I,
                                         unsigned SrcReg, bool IsKill, int FI,
                                         const TargetRegisterClass *RC,
                                         const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  if (I != MBB.end())
    DL = I->getDebugLoc();

  if (RISCV::GPRRegClass.hasSubClassEq(RC))
    BuildMI(MBB, I, DL, get(RISCV::SW))
        .addReg(SrcReg, getKillRegState(IsKill))
        .addFrameIndex(FI)
        .addImm(0);
  else
    llvm_unreachable("Can't store this register to stack slot");
}

void RISCVInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator I,
                                          unsigned DstReg, int FI,
                                          const TargetRegisterClass *RC,
                                          const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  if (I != MBB.end())
    DL = I->getDebugLoc();

  if (RISCV::GPRRegClass.hasSubClassEq(RC))
    BuildMI(MBB, I, DL, get(RISCV::LW), DstReg).addFrameIndex(FI).addImm(0);
  else
    llvm_unreachable("Can't load this register from stack slot");
}

void RISCVInstrInfo::movImm32(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI,
                              const DebugLoc &DL, unsigned DstReg, uint64_t Val,
                              MachineInstr::MIFlag Flag) const {
  assert(isInt<32>(Val) && "Can only materialize 32-bit constants");

  // TODO: If the value can be materialized using only one instruction, only
  // insert a single instruction.

  uint64_t Hi20 = ((Val + 0x800) >> 12) & 0xfffff;
  uint64_t Lo12 = SignExtend64<12>(Val);
  BuildMI(MBB, MBBI, DL, get(RISCV::LUI), DstReg)
      .addImm(Hi20)
      .setMIFlag(Flag);
  BuildMI(MBB, MBBI, DL, get(RISCV::ADDI), DstReg)
      .addReg(DstReg, RegState::Kill)
      .addImm(Lo12)
      .setMIFlag(Flag);
}

// The contents of values added to Cond are not examined outside of
// RISCVInstrInfo, giving us flexibility in what to push to it. For RISCV, we
// push BranchOpcode, Reg1, Reg2.
static void parseCondBranch(MachineInstr &LastInst, MachineBasicBlock *&Target,
                            SmallVectorImpl<MachineOperand> &Cond) {
  // Block ends with fall-through condbranch.
  assert(LastInst.getDesc().isConditionalBranch() &&
         "Unknown conditional branch");
  Target = LastInst.getOperand(2).getMBB();
  Cond.push_back(MachineOperand::CreateImm(LastInst.getOpcode()));
  Cond.push_back(LastInst.getOperand(0));
  Cond.push_back(LastInst.getOperand(1));
}

static unsigned getOppositeBranchOpcode(int Opc) {
  switch (Opc) {
  default:
    llvm_unreachable("Unrecognized conditional branch");
  case RISCV::BEQ:
    return RISCV::BNE;
  case RISCV::BNE:
    return RISCV::BEQ;
  case RISCV::BLT:
    return RISCV::BGE;
  case RISCV::BGE:
    return RISCV::BLT;
  case RISCV::BLTU:
    return RISCV::BGEU;
  case RISCV::BGEU:
    return RISCV::BLTU;
  }
}

bool RISCVInstrInfo::analyzeBranch(MachineBasicBlock &MBB,
                                   MachineBasicBlock *&TBB,
                                   MachineBasicBlock *&FBB,
                                   SmallVectorImpl<MachineOperand> &Cond,
                                   bool AllowModify) const {
  TBB = FBB = nullptr;
  Cond.clear();

  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.getLastNonDebugInstr();
  if (I == MBB.end() || !isUnpredicatedTerminator(*I))
    return false;

  // Count the number of terminators and find the first unconditional or
  // indirect branch.
  MachineBasicBlock::iterator FirstUncondOrIndirectBr = MBB.end();
  int NumTerminators = 0;
  for (auto J = I.getReverse(); J != MBB.rend() && isUnpredicatedTerminator(*J);
       J++) {
    NumTerminators++;
    if (J->getDesc().isUnconditionalBranch() ||
        J->getDesc().isIndirectBranch()) {
      FirstUncondOrIndirectBr = J.getReverse();
    }
  }

  // If AllowModify is true, we can erase any terminators after
  // FirstUncondOrIndirectBR.
  if (AllowModify && FirstUncondOrIndirectBr != MBB.end()) {
    while (std::next(FirstUncondOrIndirectBr) != MBB.end()) {
      std::next(FirstUncondOrIndirectBr)->eraseFromParent();
      NumTerminators--;
    }
    I = FirstUncondOrIndirectBr;
  }

  // We can't handle blocks that end in an indirect branch.
  if (I->getDesc().isIndirectBranch())
    return true;

  // We can't handle blocks with more than 2 terminators.
  if (NumTerminators > 2)
    return true;

  // Handle a single unconditional branch.
  if (NumTerminators == 1 && I->getDesc().isUnconditionalBranch()) {
    TBB = I->getOperand(0).getMBB();
    return false;
  }

  // Handle a single conditional branch.
  if (NumTerminators == 1 && I->getDesc().isConditionalBranch()) {
    parseCondBranch(*I, TBB, Cond);
    return false;
  }

  // Handle a conditional branch followed by an unconditional branch.
  if (NumTerminators == 2 && std::prev(I)->getDesc().isConditionalBranch() &&
      I->getDesc().isUnconditionalBranch()) {
    parseCondBranch(*std::prev(I), TBB, Cond);
    FBB = I->getOperand(0).getMBB();
    return false;
  }

  // Otherwise, we can't handle this.
  return true;
}

unsigned RISCVInstrInfo::removeBranch(MachineBasicBlock &MBB,
                                      int *BytesRemoved) const {
  assert(!BytesRemoved && "Code size not handled");
  MachineBasicBlock::iterator I = MBB.getLastNonDebugInstr();
  if (I == MBB.end())
    return 0;

  if (!I->getDesc().isUnconditionalBranch() &&
      !I->getDesc().isConditionalBranch())
    return 0;

  // Remove the branch.
  I->eraseFromParent();

  I = MBB.end();

  if (I == MBB.begin())
    return 1;
  --I;
  if (!I->getDesc().isConditionalBranch())
    return 1;

  // Remove the branch.
  I->eraseFromParent();
  return 2;
}

// Inserts a branch into the end of the specific MachineBasicBlock, returning
// the number of instructions inserted.
unsigned RISCVInstrInfo::insertBranch(
    MachineBasicBlock &MBB, MachineBasicBlock *TBB, MachineBasicBlock *FBB,
    ArrayRef<MachineOperand> Cond, const DebugLoc &DL, int *BytesAdded) const {
  assert(!BytesAdded && "Code size not handled.");

  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 3 || Cond.size() == 0) &&
         "RISCV branch conditions have two components!");

  // Unconditional branch.
  if (Cond.empty()) {
    BuildMI(&MBB, DL, get(RISCV::PseudoBR)).addMBB(TBB);
    return 1;
  }

  // Either a one or two-way conditional branch.
  unsigned Opc = Cond[0].getImm();
  BuildMI(&MBB, DL, get(Opc)).add(Cond[1]).add(Cond[2]).addMBB(TBB);

  // One-way conditional branch.
  if (!FBB)
    return 1;

  // Two-way conditional branch.
  BuildMI(&MBB, DL, get(RISCV::PseudoBR)).addMBB(FBB);
  return 2;
}

bool RISCVInstrInfo::reverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  assert((Cond.size() == 3) && "Invalid branch condition!");
  Cond[0].setImm(getOppositeBranchOpcode(Cond[0].getImm()));
  return false;
}
