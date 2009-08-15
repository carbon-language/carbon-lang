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
    bool InsertITBlocks(MachineBasicBlock &MBB);
  };
  char Thumb2ITBlockPass::ID = 0;
}

static ARMCC::CondCodes getPredicate(const MachineInstr *MI,
                                     const Thumb2InstrInfo *TII) {
  unsigned Opc = MI->getOpcode();
  if (Opc == ARM::tBcc || Opc == ARM::t2Bcc)
    return ARMCC::AL;
  return TII->getPredicate(MI);
}

bool Thumb2ITBlockPass::InsertITBlocks(MachineBasicBlock &MBB) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineInstr *MI = &*MBBI;
    ARMCC::CondCodes CC = getPredicate(MI, TII);
    if (CC == ARMCC::AL) {
      ++MBBI;
      continue;
    }

    // Insert an IT instruction.
    DebugLoc dl = MI->getDebugLoc();
    MachineInstrBuilder MIB = BuildMI(MBB, MBBI, dl, TII->get(ARM::t2IT))
      .addImm(CC);
    ++MBBI;

    // Finalize IT mask.
    ARMCC::CondCodes OCC = ARMCC::getOppositeCondition(CC);
    unsigned Mask = 0, Pos = 3;
    while (MBBI != E && Pos) {
      ARMCC::CondCodes NCC = getPredicate(&*MBBI, TII);
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
