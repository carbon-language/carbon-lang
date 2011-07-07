//===-- MipsEmitGPRestore.cpp - Emit GP restore instruction----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass emits instructions that restore $gp right
// after jalr instructions.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "emit-gp-restore"

#include "Mips.h"
#include "MipsTargetMachine.h"
#include "MipsMachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/ADT/Statistic.h"

using namespace llvm;

namespace {
  struct Inserter : public MachineFunctionPass {

    TargetMachine &TM;
    const TargetInstrInfo *TII;

    static char ID;
    Inserter(TargetMachine &tm)
      : MachineFunctionPass(ID), TM(tm), TII(tm.getInstrInfo()) { }

    virtual const char *getPassName() const {
      return "Mips Emit GP Restore";
    }

    bool runOnMachineFunction(MachineFunction &F);
  };
  char Inserter::ID = 0;
} // end of anonymous namespace

bool Inserter::runOnMachineFunction(MachineFunction &F) {
  if (TM.getRelocationModel() != Reloc::PIC_)
    return false;

  bool Changed = false;
  int FI =  F.getInfo<MipsFunctionInfo>()->getGPFI();

  for (MachineFunction::iterator MFI = F.begin(), MFE = F.end();
       MFI != MFE; ++MFI) {
    MachineBasicBlock& MBB = *MFI;
    MachineBasicBlock::iterator I = MFI->begin();

    // If MBB is a landing pad, insert instruction that restores $gp after
    // EH_LABEL.
    if (MBB.isLandingPad()) {
      // Find EH_LABEL first.
      for (; I->getOpcode() != TargetOpcode::EH_LABEL; ++I) ;
      
      // Insert lw.
      ++I;
      DebugLoc dl = I != MBB.end() ? I->getDebugLoc() : DebugLoc();
      BuildMI(MBB, I, dl, TII->get(Mips::LW), Mips::GP).addFrameIndex(FI)
                                                       .addImm(0);
      Changed = true;
    }

    while (I != MFI->end()) {
      if (I->getOpcode() != Mips::JALR) {
        ++I;
        continue;
      }

      DebugLoc dl = I->getDebugLoc();
      // emit lw $gp, ($gp save slot on stack) after jalr
      BuildMI(MBB, ++I, dl, TII->get(Mips::LW), Mips::GP).addFrameIndex(FI)
                                                         .addImm(0);
      Changed = true;
    }
  } 

  return Changed;
}

/// createMipsEmitGPRestorePass - Returns a pass that emits instructions that
/// restores $gp clobbered by jalr instructions.
FunctionPass *llvm::createMipsEmitGPRestorePass(MipsTargetMachine &tm) {
  return new Inserter(tm);
}

