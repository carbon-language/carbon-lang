//===-- ARMExpandPseudoInsts.cpp - Expand pseudo instructions -----*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that expand pseudo instructions into target
// instructions to allow proper scheduling, if-conversion, and other late
// optimizations. This pass should be run after register allocation but before
// post- regalloc scheduling pass.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "arm-pseudo"
#include "ARM.h"
#include "ARMBaseInstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

namespace {
  class ARMExpandPseudo : public MachineFunctionPass {
  public:
    static char ID;
    ARMExpandPseudo() : MachineFunctionPass(&ID) {}

    const TargetInstrInfo *TII;

    virtual bool runOnMachineFunction(MachineFunction &Fn);

    virtual const char *getPassName() const {
      return "ARM pseudo instruction expansion pass";
    }

  private:
    bool ExpandMBB(MachineBasicBlock &MBB);
  };
  char ARMExpandPseudo::ID = 0;
}

bool ARMExpandPseudo::ExpandMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineInstr &MI = *MBBI;
    MachineBasicBlock::iterator NMBBI = next(MBBI);

    unsigned Opcode = MI.getOpcode();
    switch (Opcode) {
    default: break;
    case ARM::tLDRpci_pic: 
    case ARM::t2LDRpci_pic: {
      unsigned NewLdOpc = (Opcode == ARM::tLDRpci_pic)
        ? ARM::tLDRpci : ARM::t2LDRpci;
      unsigned DstReg = MI.getOperand(0).getReg();
      if (!MI.getOperand(0).isDead()) {
        MachineInstr *NewMI =
          AddDefaultPred(BuildMI(MBB, MBBI, MI.getDebugLoc(),
                                 TII->get(NewLdOpc), DstReg)
                         .addOperand(MI.getOperand(1)));
        NewMI->setMemRefs(MI.memoperands_begin(), MI.memoperands_end());
        BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(ARM::tPICADD))
          .addReg(DstReg, getDefRegState(true))
          .addReg(DstReg)
          .addOperand(MI.getOperand(2));
      }
      MI.eraseFromParent();
      Modified = true;
      break;
    }
    case ARM::t2MOVi32imm: {
      unsigned DstReg = MI.getOperand(0).getReg();
      unsigned Imm = MI.getOperand(1).getImm();
      unsigned Lo16 = Imm & 0xffff;
      unsigned Hi16 = (Imm >> 16) & 0xffff;
      if (!MI.getOperand(0).isDead()) {
        AddDefaultPred(BuildMI(MBB, MBBI, MI.getDebugLoc(),
                               TII->get(ARM::t2MOVi16), DstReg)
                       .addImm(Lo16));
        AddDefaultPred(BuildMI(MBB, MBBI, MI.getDebugLoc(),
                               TII->get(ARM::t2MOVTi16))
                       .addReg(DstReg, getDefRegState(true))
                       .addReg(DstReg).addImm(Hi16));
      }
      MI.eraseFromParent();
      Modified = true;
    }
    // FIXME: expand t2MOVi32imm
    }
    MBBI = NMBBI;
  }

  return Modified;
}

bool ARMExpandPseudo::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getTarget().getInstrInfo();

  bool Modified = false;
  for (MachineFunction::iterator MFI = MF.begin(), E = MF.end(); MFI != E;
       ++MFI)
    Modified |= ExpandMBB(*MFI);
  return Modified;
}

/// createARMExpandPseudoPass - returns an instance of the pseudo instruction
/// expansion pass.
FunctionPass *llvm::createARMExpandPseudoPass() {
  return new ARMExpandPseudo();
}
