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
    const TargetRegisterInfo *TRI;

    virtual bool runOnMachineFunction(MachineFunction &Fn);

    virtual const char *getPassName() const {
      return "ARM pseudo instruction expansion pass";
    }

  private:
    void TransferImpOps(MachineInstr &OldMI,
                        MachineInstrBuilder &UseMI, MachineInstrBuilder &DefMI);
    bool ExpandMBB(MachineBasicBlock &MBB);
  };
  char ARMExpandPseudo::ID = 0;
}

/// TransferImpOps - Transfer implicit operands on the pseudo instruction to
/// the instructions created from the expansion.
void ARMExpandPseudo::TransferImpOps(MachineInstr &OldMI,
                                     MachineInstrBuilder &UseMI,
                                     MachineInstrBuilder &DefMI) {
  const TargetInstrDesc &Desc = OldMI.getDesc();
  for (unsigned i = Desc.getNumOperands(), e = OldMI.getNumOperands();
       i != e; ++i) {
    const MachineOperand &MO = OldMI.getOperand(i);
    assert(MO.isReg() && MO.getReg());
    if (MO.isUse())
      UseMI.addReg(MO.getReg(), getKillRegState(MO.isKill()));
    else
      DefMI.addReg(MO.getReg(),
                   getDefRegState(true) | getDeadRegState(MO.isDead()));
  }
}

bool ARMExpandPseudo::ExpandMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineInstr &MI = *MBBI;
    MachineBasicBlock::iterator NMBBI = llvm::next(MBBI);

    unsigned Opcode = MI.getOpcode();
    switch (Opcode) {
    default: break;
    case ARM::tLDRpci_pic: 
    case ARM::t2LDRpci_pic: {
      unsigned NewLdOpc = (Opcode == ARM::tLDRpci_pic)
        ? ARM::tLDRpci : ARM::t2LDRpci;
      unsigned DstReg = MI.getOperand(0).getReg();
      bool DstIsDead = MI.getOperand(0).isDead();
      MachineInstrBuilder MIB1 =
        AddDefaultPred(BuildMI(MBB, MBBI, MI.getDebugLoc(),
                               TII->get(NewLdOpc), DstReg)
                       .addOperand(MI.getOperand(1)));
      (*MIB1).setMemRefs(MI.memoperands_begin(), MI.memoperands_end());
      MachineInstrBuilder MIB2 = BuildMI(MBB, MBBI, MI.getDebugLoc(),
                                         TII->get(ARM::tPICADD))
        .addReg(DstReg, getDefRegState(true) | getDeadRegState(DstIsDead))
        .addReg(DstReg)
        .addOperand(MI.getOperand(2));
      TransferImpOps(MI, MIB1, MIB2);
      MI.eraseFromParent();
      Modified = true;
      break;
    }

    case ARM::t2MOVi32imm: {
      unsigned PredReg = 0;
      ARMCC::CondCodes Pred = llvm::getInstrPredicate(&MI, PredReg);
      unsigned DstReg = MI.getOperand(0).getReg();
      bool DstIsDead = MI.getOperand(0).isDead();
      const MachineOperand &MO = MI.getOperand(1);
      MachineInstrBuilder LO16, HI16;

      LO16 = BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(ARM::t2MOVi16),
                     DstReg);
      HI16 = BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(ARM::t2MOVTi16))
        .addReg(DstReg, getDefRegState(true) | getDeadRegState(DstIsDead))
        .addReg(DstReg);

      if (MO.isImm()) {
        unsigned Imm = MO.getImm();
        unsigned Lo16 = Imm & 0xffff;
        unsigned Hi16 = (Imm >> 16) & 0xffff;
        LO16 = LO16.addImm(Lo16);
        HI16 = HI16.addImm(Hi16);
      } else {
        const GlobalValue *GV = MO.getGlobal();
        unsigned TF = MO.getTargetFlags();
        LO16 = LO16.addGlobalAddress(GV, MO.getOffset(), TF | ARMII::MO_LO16);
        HI16 = HI16.addGlobalAddress(GV, MO.getOffset(), TF | ARMII::MO_HI16);
      }
      (*LO16).setMemRefs(MI.memoperands_begin(), MI.memoperands_end());
      (*HI16).setMemRefs(MI.memoperands_begin(), MI.memoperands_end());
      LO16.addImm(Pred).addReg(PredReg);
      HI16.addImm(Pred).addReg(PredReg);
      TransferImpOps(MI, LO16, HI16);
      MI.eraseFromParent();
      Modified = true;
      break;
    }

    case ARM::VMOVQQ: {
      unsigned DstReg = MI.getOperand(0).getReg();
      bool DstIsDead = MI.getOperand(0).isDead();
      unsigned EvenDst = TRI->getSubReg(DstReg, ARM::qsub_0);
      unsigned OddDst  = TRI->getSubReg(DstReg, ARM::qsub_1);
      unsigned SrcReg = MI.getOperand(1).getReg();
      bool SrcIsKill = MI.getOperand(1).isKill();
      unsigned EvenSrc = TRI->getSubReg(SrcReg, ARM::qsub_0);
      unsigned OddSrc  = TRI->getSubReg(SrcReg, ARM::qsub_1);
      MachineInstrBuilder Even =
        AddDefaultPred(BuildMI(MBB, MBBI, MI.getDebugLoc(),
                               TII->get(ARM::VMOVQ))
                       .addReg(EvenDst, getDefRegState(true) | getDeadRegState(DstIsDead))
                       .addReg(EvenSrc, getKillRegState(SrcIsKill)));
      MachineInstrBuilder Odd =
        AddDefaultPred(BuildMI(MBB, MBBI, MI.getDebugLoc(),
                               TII->get(ARM::VMOVQ))
                       .addReg(OddDst, getDefRegState(true) | getDeadRegState(DstIsDead))
                       .addReg(OddSrc, getKillRegState(SrcIsKill)));
      TransferImpOps(MI, Even, Odd);
      MI.eraseFromParent();
      Modified = true;
    }
    }
    MBBI = NMBBI;
  }

  return Modified;
}

bool ARMExpandPseudo::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getTarget().getInstrInfo();
  TRI = MF.getTarget().getRegisterInfo();

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
