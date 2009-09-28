//===-- Thumb2ITBlockPass.cpp - Insert Thumb IT blocks -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "thumb2-it"
#include "ARM.h"
#include "ARMMachineFunctionInfo.h"
#include "Thumb2InstrInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumITs,     "Number of IT blocks inserted");

namespace {
  struct VISIBILITY_HIDDEN Thumb2ITBlockPass : public MachineFunctionPass {
    static char ID;
    Thumb2ITBlockPass() : MachineFunctionPass(&ID) {}

    const Thumb2InstrInfo *TII;
    ARMFunctionInfo *AFI;

    virtual bool runOnMachineFunction(MachineFunction &Fn);

    virtual const char *getPassName() const {
      return "Thumb IT blocks insertion pass";
    }

  private:
    MachineBasicBlock::iterator
      SplitT2MOV32imm(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                      MachineInstr *MI, DebugLoc dl,
                      unsigned PredReg, ARMCC::CondCodes CC);
    bool InsertITBlocks(MachineBasicBlock &MBB);
  };
  char Thumb2ITBlockPass::ID = 0;
}

static ARMCC::CondCodes getPredicate(const MachineInstr *MI, unsigned &PredReg){
  unsigned Opc = MI->getOpcode();
  if (Opc == ARM::tBcc || Opc == ARM::t2Bcc)
    return ARMCC::AL;
  return llvm::getInstrPredicate(MI, PredReg);
}

MachineBasicBlock::iterator
Thumb2ITBlockPass::SplitT2MOV32imm(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MBBI,
                                   MachineInstr *MI,
                                   DebugLoc dl, unsigned PredReg,
                                   ARMCC::CondCodes CC) {
  // Splitting t2MOVi32imm into a pair of t2MOVi16 + t2MOVTi16 here.
  // The only reason it was a single instruction was so it could be
  // re-materialized. We want to split it before this and the thumb2
  // size reduction pass to make sure the IT mask is correct and expose
  // width reduction opportunities. It doesn't make sense to do this in a 
  // separate pass so here it is.
  unsigned DstReg = MI->getOperand(0).getReg();
  bool DstDead = MI->getOperand(0).isDead(); // Is this possible?
  unsigned Imm = MI->getOperand(1).getImm();
  unsigned Lo16 = Imm & 0xffff;
  unsigned Hi16 = (Imm >> 16) & 0xffff;
  BuildMI(MBB, MBBI, dl, TII->get(ARM::t2MOVi16), DstReg)
    .addImm(Lo16).addImm(CC).addReg(PredReg);
  BuildMI(MBB, MBBI, dl, TII->get(ARM::t2MOVTi16))
    .addReg(DstReg, getDefRegState(true) | getDeadRegState(DstDead))
    .addReg(DstReg).addImm(Hi16).addImm(CC).addReg(PredReg);
  --MBBI;
  --MBBI;
  MI->eraseFromParent();
  return MBBI;
}

bool Thumb2ITBlockPass::InsertITBlocks(MachineBasicBlock &MBB) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineInstr *MI = &*MBBI;
    DebugLoc dl = MI->getDebugLoc();
    unsigned PredReg = 0;
    ARMCC::CondCodes CC = getPredicate(MI, PredReg);

    if (MI->getOpcode() == ARM::t2MOVi32imm) {
      MBBI = SplitT2MOV32imm(MBB, MBBI, MI, dl, PredReg, CC);
      continue;
    }

    if (CC == ARMCC::AL) {
      ++MBBI;
      continue;
    }

    // Insert an IT instruction.
    MachineInstrBuilder MIB = BuildMI(MBB, MBBI, dl, TII->get(ARM::t2IT))
      .addImm(CC);
    ++MBBI;

    // Finalize IT mask.
    ARMCC::CondCodes OCC = ARMCC::getOppositeCondition(CC);
    unsigned Mask = 0, Pos = 3;
    while (MBBI != E && Pos) {
      MachineInstr *NMI = &*MBBI;
      DebugLoc ndl = NMI->getDebugLoc();
      unsigned NPredReg = 0;
      ARMCC::CondCodes NCC = getPredicate(NMI, NPredReg);
      if (NMI->getOpcode() == ARM::t2MOVi32imm) {
        MBBI = SplitT2MOV32imm(MBB, MBBI, NMI, ndl, NPredReg, NCC);
        continue;
      }

      if (NCC == OCC) {
        Mask |= (1 << Pos);
      } else if (NCC != CC)
        break;
      --Pos;
      ++MBBI;
    }
    Mask |= (1 << Pos);
    MIB.addImm(Mask);
    Modified = true;
    ++NumITs;
  }

  return Modified;
}

bool Thumb2ITBlockPass::runOnMachineFunction(MachineFunction &Fn) {
  const TargetMachine &TM = Fn.getTarget();
  AFI = Fn.getInfo<ARMFunctionInfo>();
  TII = static_cast<const Thumb2InstrInfo*>(TM.getInstrInfo());

  if (!AFI->isThumbFunction())
    return false;

  bool Modified = false;
  for (MachineFunction::iterator MFI = Fn.begin(), E = Fn.end(); MFI != E;
       ++MFI) {
    MachineBasicBlock &MBB = *MFI;
    Modified |= InsertITBlocks(MBB);
  }

  return Modified;
}

/// createThumb2ITBlockPass - Returns an instance of the Thumb2 IT blocks
/// insertion pass.
FunctionPass *llvm::createThumb2ITBlockPass() {
  return new Thumb2ITBlockPass();
}
