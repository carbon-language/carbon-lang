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
#include "llvm/Target/TargetRegisterInfo.h"
using namespace llvm;

namespace {
  class ARMExpandPseudo : public MachineFunctionPass {
    // Constants for register spacing in NEON load/store instructions.
    enum NEONRegSpacing {
      SingleSpc,
      EvenDblSpc,
      OddDblSpc
    };

  public:
    static char ID;
    ARMExpandPseudo() : MachineFunctionPass(ID) {}

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
    void ExpandVLD(MachineBasicBlock::iterator &MBBI, unsigned Opc,
                   bool hasWriteBack, NEONRegSpacing RegSpc, unsigned NumRegs);
    void ExpandVST(MachineBasicBlock::iterator &MBBI, unsigned Opc,
                   bool hasWriteBack, NEONRegSpacing RegSpc, unsigned NumRegs);
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

/// ExpandVLD - Translate VLD pseudo instructions with Q, QQ or QQQQ register
/// operands to real VLD instructions with D register operands.
void ARMExpandPseudo::ExpandVLD(MachineBasicBlock::iterator &MBBI,
                                unsigned Opc, bool hasWriteBack,
                                NEONRegSpacing RegSpc, unsigned NumRegs) {
  MachineInstr &MI = *MBBI;
  MachineBasicBlock &MBB = *MI.getParent();

  MachineInstrBuilder MIB = BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(Opc));
  unsigned OpIdx = 0;

  bool DstIsDead = MI.getOperand(OpIdx).isDead();
  unsigned DstReg = MI.getOperand(OpIdx++).getReg();
  unsigned D0, D1, D2, D3;
  if (RegSpc == SingleSpc) {
    D0 = TRI->getSubReg(DstReg, ARM::dsub_0);
    D1 = TRI->getSubReg(DstReg, ARM::dsub_1);
    D2 = TRI->getSubReg(DstReg, ARM::dsub_2);
    D3 = TRI->getSubReg(DstReg, ARM::dsub_3);
  } else if (RegSpc == EvenDblSpc) {
    D0 = TRI->getSubReg(DstReg, ARM::dsub_0);
    D1 = TRI->getSubReg(DstReg, ARM::dsub_2);
    D2 = TRI->getSubReg(DstReg, ARM::dsub_4);
    D3 = TRI->getSubReg(DstReg, ARM::dsub_6);
  } else {
    assert(RegSpc == OddDblSpc && "unknown register spacing for VLD");
    D0 = TRI->getSubReg(DstReg, ARM::dsub_1);
    D1 = TRI->getSubReg(DstReg, ARM::dsub_3);
    D2 = TRI->getSubReg(DstReg, ARM::dsub_5);
    D3 = TRI->getSubReg(DstReg, ARM::dsub_7);
  } 
  MIB.addReg(D0, RegState::Define | getDeadRegState(DstIsDead))
    .addReg(D1, RegState::Define | getDeadRegState(DstIsDead));
  if (NumRegs > 2)
    MIB.addReg(D2, RegState::Define | getDeadRegState(DstIsDead));
  if (NumRegs > 3)
    MIB.addReg(D3, RegState::Define | getDeadRegState(DstIsDead));

  if (hasWriteBack) {
    bool WBIsDead = MI.getOperand(OpIdx).isDead();
    unsigned WBReg = MI.getOperand(OpIdx++).getReg();
    MIB.addReg(WBReg, RegState::Define | getDeadRegState(WBIsDead));
  }
  // Copy the addrmode6 operands.
  bool AddrIsKill = MI.getOperand(OpIdx).isKill();
  MIB.addReg(MI.getOperand(OpIdx++).getReg(), getKillRegState(AddrIsKill));
  MIB.addImm(MI.getOperand(OpIdx++).getImm());
  if (hasWriteBack) {
    // Copy the am6offset operand.
    bool OffsetIsKill = MI.getOperand(OpIdx).isKill();
    MIB.addReg(MI.getOperand(OpIdx++).getReg(), getKillRegState(OffsetIsKill));
  }

  MIB = AddDefaultPred(MIB);
  TransferImpOps(MI, MIB, MIB);
  // For an instruction writing the odd subregs, add an implicit use of the
  // super-register because the even subregs were loaded separately.
  if (RegSpc == OddDblSpc)
    MIB.addReg(DstReg, RegState::Implicit);
  // Add an implicit def for the super-register.
  MIB.addReg(DstReg, RegState::ImplicitDefine | getDeadRegState(DstIsDead));
  MI.eraseFromParent();
}

/// ExpandVST - Translate VST pseudo instructions with Q, QQ or QQQQ register
/// operands to real VST instructions with D register operands.
void ARMExpandPseudo::ExpandVST(MachineBasicBlock::iterator &MBBI,
                                unsigned Opc, bool hasWriteBack,
                                NEONRegSpacing RegSpc, unsigned NumRegs) {
  MachineInstr &MI = *MBBI;
  MachineBasicBlock &MBB = *MI.getParent();

  MachineInstrBuilder MIB = BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(Opc));
  unsigned OpIdx = 0;
  if (hasWriteBack) {
    bool DstIsDead = MI.getOperand(OpIdx).isDead();
    unsigned DstReg = MI.getOperand(OpIdx++).getReg();
    MIB.addReg(DstReg, RegState::Define | getDeadRegState(DstIsDead));
  }
  // Copy the addrmode6 operands.
  bool AddrIsKill = MI.getOperand(OpIdx).isKill();
  MIB.addReg(MI.getOperand(OpIdx++).getReg(), getKillRegState(AddrIsKill));
  MIB.addImm(MI.getOperand(OpIdx++).getImm());
  if (hasWriteBack) {
    // Copy the am6offset operand.
    bool OffsetIsKill = MI.getOperand(OpIdx).isKill();
    MIB.addReg(MI.getOperand(OpIdx++).getReg(), getKillRegState(OffsetIsKill));
  }

  bool SrcIsKill = MI.getOperand(OpIdx).isKill();
  unsigned SrcReg = MI.getOperand(OpIdx).getReg();
  unsigned D0, D1, D2, D3;
  if (RegSpc == SingleSpc) {
    D0 = TRI->getSubReg(SrcReg, ARM::dsub_0);
    D1 = TRI->getSubReg(SrcReg, ARM::dsub_1);
    D2 = TRI->getSubReg(SrcReg, ARM::dsub_2);
    D3 = TRI->getSubReg(SrcReg, ARM::dsub_3);
  } else if (RegSpc == EvenDblSpc) {
    D0 = TRI->getSubReg(SrcReg, ARM::dsub_0);
    D1 = TRI->getSubReg(SrcReg, ARM::dsub_2);
    D2 = TRI->getSubReg(SrcReg, ARM::dsub_4);
    D3 = TRI->getSubReg(SrcReg, ARM::dsub_6);
  } else {
    assert(RegSpc == OddDblSpc && "unknown register spacing for VST");
    D0 = TRI->getSubReg(SrcReg, ARM::dsub_1);
    D1 = TRI->getSubReg(SrcReg, ARM::dsub_3);
    D2 = TRI->getSubReg(SrcReg, ARM::dsub_5);
    D3 = TRI->getSubReg(SrcReg, ARM::dsub_7);
  } 

  MIB.addReg(D0).addReg(D1);
  if (NumRegs > 2)
    MIB.addReg(D2);
  if (NumRegs > 3)
    MIB.addReg(D3);
  MIB = AddDefaultPred(MIB);
  TransferImpOps(MI, MIB, MIB);
  if (SrcIsKill)
    // Add an implicit kill for the super-reg.
    (*MIB).addRegisterKilled(SrcReg, TRI, true);
  MI.eraseFromParent();
}

bool ARMExpandPseudo::ExpandMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineInstr &MI = *MBBI;
    MachineBasicBlock::iterator NMBBI = llvm::next(MBBI);

    bool ModifiedOp = true;
    unsigned Opcode = MI.getOpcode();
    switch (Opcode) {
    default:
      ModifiedOp = false;
      break;

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
      break;
    }

    case ARM::MOVi32imm:
    case ARM::t2MOVi32imm: {
      unsigned PredReg = 0;
      ARMCC::CondCodes Pred = llvm::getInstrPredicate(&MI, PredReg);
      unsigned DstReg = MI.getOperand(0).getReg();
      bool DstIsDead = MI.getOperand(0).isDead();
      const MachineOperand &MO = MI.getOperand(1);
      MachineInstrBuilder LO16, HI16;

      LO16 = BuildMI(MBB, MBBI, MI.getDebugLoc(),
                     TII->get(Opcode == ARM::MOVi32imm ?
                              ARM::MOVi16 : ARM::t2MOVi16),
                     DstReg);
      HI16 = BuildMI(MBB, MBBI, MI.getDebugLoc(),
                     TII->get(Opcode == ARM::MOVi32imm ?
                              ARM::MOVTi16 : ARM::t2MOVTi16))
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
                     .addReg(EvenDst,
                             getDefRegState(true) | getDeadRegState(DstIsDead))
                     .addReg(EvenSrc, getKillRegState(SrcIsKill)));
      MachineInstrBuilder Odd =
        AddDefaultPred(BuildMI(MBB, MBBI, MI.getDebugLoc(),
                               TII->get(ARM::VMOVQ))
                     .addReg(OddDst,
                             getDefRegState(true) | getDeadRegState(DstIsDead))
                     .addReg(OddSrc, getKillRegState(SrcIsKill)));
      TransferImpOps(MI, Even, Odd);
      MI.eraseFromParent();
    }

    case ARM::VLD1q8Pseudo:
      ExpandVLD(MBBI, ARM::VLD1q8, false, SingleSpc, 2); break;
    case ARM::VLD1q16Pseudo:
      ExpandVLD(MBBI, ARM::VLD1q16, false, SingleSpc, 2); break;
    case ARM::VLD1q32Pseudo:
      ExpandVLD(MBBI, ARM::VLD1q32, false, SingleSpc, 2); break;
    case ARM::VLD1q64Pseudo:
      ExpandVLD(MBBI, ARM::VLD1q64, false, SingleSpc, 2); break;
    case ARM::VLD1q8Pseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD1q8, true, SingleSpc, 2); break;
    case ARM::VLD1q16Pseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD1q16, true, SingleSpc, 2); break;
    case ARM::VLD1q32Pseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD1q32, true, SingleSpc, 2); break;
    case ARM::VLD1q64Pseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD1q64, true, SingleSpc, 2); break;

    case ARM::VLD2d8Pseudo:
      ExpandVLD(MBBI, ARM::VLD2d8, false, SingleSpc, 2); break;
    case ARM::VLD2d16Pseudo:
      ExpandVLD(MBBI, ARM::VLD2d16, false, SingleSpc, 2); break;
    case ARM::VLD2d32Pseudo:
      ExpandVLD(MBBI, ARM::VLD2d32, false, SingleSpc, 2); break;
    case ARM::VLD2q8Pseudo:
      ExpandVLD(MBBI, ARM::VLD2q8, false, SingleSpc, 4); break;
    case ARM::VLD2q16Pseudo:
      ExpandVLD(MBBI, ARM::VLD2q16, false, SingleSpc, 4); break;
    case ARM::VLD2q32Pseudo:
      ExpandVLD(MBBI, ARM::VLD2q32, false, SingleSpc, 4); break;
    case ARM::VLD2d8Pseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD2d8, true, SingleSpc, 2); break;
    case ARM::VLD2d16Pseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD2d16, true, SingleSpc, 2); break;
    case ARM::VLD2d32Pseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD2d32, true, SingleSpc, 2); break;
    case ARM::VLD2q8Pseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD2q8, true, SingleSpc, 4); break;
    case ARM::VLD2q16Pseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD2q16, true, SingleSpc, 4); break;
    case ARM::VLD2q32Pseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD2q32, true, SingleSpc, 4); break;

    case ARM::VLD3d8Pseudo:
      ExpandVLD(MBBI, ARM::VLD3d8, false, SingleSpc, 3); break;
    case ARM::VLD3d16Pseudo:
      ExpandVLD(MBBI, ARM::VLD3d16, false, SingleSpc, 3); break;
    case ARM::VLD3d32Pseudo:
      ExpandVLD(MBBI, ARM::VLD3d32, false, SingleSpc, 3); break;
    case ARM::VLD1d64TPseudo:
      ExpandVLD(MBBI, ARM::VLD1d64T, false, SingleSpc, 3); break;
    case ARM::VLD3d8Pseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD3d8_UPD, true, SingleSpc, 3); break;
    case ARM::VLD3d16Pseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD3d16_UPD, true, SingleSpc, 3); break;
    case ARM::VLD3d32Pseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD3d32_UPD, true, SingleSpc, 3); break;
    case ARM::VLD1d64TPseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD1d64T_UPD, true, SingleSpc, 3); break;
    case ARM::VLD3q8Pseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD3q8_UPD, true, EvenDblSpc, 3); break;
    case ARM::VLD3q16Pseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD3q16_UPD, true, EvenDblSpc, 3); break;
    case ARM::VLD3q32Pseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD3q32_UPD, true, EvenDblSpc, 3); break;
    case ARM::VLD3q8oddPseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD3q8_UPD, true, OddDblSpc, 3); break;
    case ARM::VLD3q16oddPseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD3q16_UPD, true, OddDblSpc, 3); break;
    case ARM::VLD3q32oddPseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD3q32_UPD, true, OddDblSpc, 3); break;

    case ARM::VLD4d8Pseudo:
      ExpandVLD(MBBI, ARM::VLD4d8, false, SingleSpc, 4); break;
    case ARM::VLD4d16Pseudo:
      ExpandVLD(MBBI, ARM::VLD4d16, false, SingleSpc, 4); break;
    case ARM::VLD4d32Pseudo:
      ExpandVLD(MBBI, ARM::VLD4d32, false, SingleSpc, 4); break;
    case ARM::VLD1d64QPseudo:
      ExpandVLD(MBBI, ARM::VLD1d64Q, false, SingleSpc, 4); break;
    case ARM::VLD4d8Pseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD4d8_UPD, true, SingleSpc, 4); break;
    case ARM::VLD4d16Pseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD4d16_UPD, true, SingleSpc, 4); break;
    case ARM::VLD4d32Pseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD4d32_UPD, true, SingleSpc, 4); break;
    case ARM::VLD1d64QPseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD1d64Q_UPD, true, SingleSpc, 4); break;
    case ARM::VLD4q8Pseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD4q8_UPD, true, EvenDblSpc, 4); break;
    case ARM::VLD4q16Pseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD4q16_UPD, true, EvenDblSpc, 4); break;
    case ARM::VLD4q32Pseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD4q32_UPD, true, EvenDblSpc, 4); break;
    case ARM::VLD4q8oddPseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD4q8_UPD, true, OddDblSpc, 4); break;
    case ARM::VLD4q16oddPseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD4q16_UPD, true, OddDblSpc, 4); break;
    case ARM::VLD4q32oddPseudo_UPD:
      ExpandVLD(MBBI, ARM::VLD4q32_UPD, true, OddDblSpc, 4); break;

    case ARM::VST1q8Pseudo:
      ExpandVST(MBBI, ARM::VST1q8, false, SingleSpc, 2); break;
    case ARM::VST1q16Pseudo:
      ExpandVST(MBBI, ARM::VST1q16, false, SingleSpc, 2); break;
    case ARM::VST1q32Pseudo:
      ExpandVST(MBBI, ARM::VST1q32, false, SingleSpc, 2); break;
    case ARM::VST1q64Pseudo:
      ExpandVST(MBBI, ARM::VST1q64, false, SingleSpc, 2); break;
    case ARM::VST1q8Pseudo_UPD:
      ExpandVST(MBBI, ARM::VST1q8_UPD, true, SingleSpc, 2); break;
    case ARM::VST1q16Pseudo_UPD:
      ExpandVST(MBBI, ARM::VST1q16_UPD, true, SingleSpc, 2); break;
    case ARM::VST1q32Pseudo_UPD:
      ExpandVST(MBBI, ARM::VST1q32_UPD, true, SingleSpc, 2); break;
    case ARM::VST1q64Pseudo_UPD:
      ExpandVST(MBBI, ARM::VST1q64_UPD, true, SingleSpc, 2); break;

    case ARM::VST2d8Pseudo:
      ExpandVST(MBBI, ARM::VST2d8, false, SingleSpc, 2); break;
    case ARM::VST2d16Pseudo:
      ExpandVST(MBBI, ARM::VST2d16, false, SingleSpc, 2); break;
    case ARM::VST2d32Pseudo:
      ExpandVST(MBBI, ARM::VST2d32, false, SingleSpc, 2); break;
    case ARM::VST2q8Pseudo:
      ExpandVST(MBBI, ARM::VST2q8, false, SingleSpc, 4); break;
    case ARM::VST2q16Pseudo:
      ExpandVST(MBBI, ARM::VST2q16, false, SingleSpc, 4); break;
    case ARM::VST2q32Pseudo:
      ExpandVST(MBBI, ARM::VST2q32, false, SingleSpc, 4); break;
    case ARM::VST2d8Pseudo_UPD:
      ExpandVST(MBBI, ARM::VST2d8_UPD, true, SingleSpc, 2); break;
    case ARM::VST2d16Pseudo_UPD:
      ExpandVST(MBBI, ARM::VST2d16_UPD, true, SingleSpc, 2); break;
    case ARM::VST2d32Pseudo_UPD:
      ExpandVST(MBBI, ARM::VST2d32_UPD, true, SingleSpc, 2); break;
    case ARM::VST2q8Pseudo_UPD:
      ExpandVST(MBBI, ARM::VST2q8_UPD, true, SingleSpc, 4); break;
    case ARM::VST2q16Pseudo_UPD:
      ExpandVST(MBBI, ARM::VST2q16_UPD, true, SingleSpc, 4); break;
    case ARM::VST2q32Pseudo_UPD:
      ExpandVST(MBBI, ARM::VST2q32_UPD, true, SingleSpc, 4); break;

    case ARM::VST3d8Pseudo:
      ExpandVST(MBBI, ARM::VST3d8, false, SingleSpc, 3); break;
    case ARM::VST3d16Pseudo:
      ExpandVST(MBBI, ARM::VST3d16, false, SingleSpc, 3); break;
    case ARM::VST3d32Pseudo:
      ExpandVST(MBBI, ARM::VST3d32, false, SingleSpc, 3); break;
    case ARM::VST1d64TPseudo:
      ExpandVST(MBBI, ARM::VST1d64T, false, SingleSpc, 3); break;
    case ARM::VST3d8Pseudo_UPD:
      ExpandVST(MBBI, ARM::VST3d8_UPD, true, SingleSpc, 3); break;
    case ARM::VST3d16Pseudo_UPD:
      ExpandVST(MBBI, ARM::VST3d16_UPD, true, SingleSpc, 3); break;
    case ARM::VST3d32Pseudo_UPD:
      ExpandVST(MBBI, ARM::VST3d32_UPD, true, SingleSpc, 3); break;
    case ARM::VST1d64TPseudo_UPD:
      ExpandVST(MBBI, ARM::VST1d64T_UPD, true, SingleSpc, 3); break;
    case ARM::VST3q8Pseudo_UPD:
      ExpandVST(MBBI, ARM::VST3q8_UPD, true, EvenDblSpc, 3); break;
    case ARM::VST3q16Pseudo_UPD:
      ExpandVST(MBBI, ARM::VST3q16_UPD, true, EvenDblSpc, 3); break;
    case ARM::VST3q32Pseudo_UPD:
      ExpandVST(MBBI, ARM::VST3q32_UPD, true, EvenDblSpc, 3); break;
    case ARM::VST3q8oddPseudo_UPD:
      ExpandVST(MBBI, ARM::VST3q8_UPD, true, OddDblSpc, 3); break;
    case ARM::VST3q16oddPseudo_UPD:
      ExpandVST(MBBI, ARM::VST3q16_UPD, true, OddDblSpc, 3); break;
    case ARM::VST3q32oddPseudo_UPD:
      ExpandVST(MBBI, ARM::VST3q32_UPD, true, OddDblSpc, 3); break;

    case ARM::VST4d8Pseudo:
      ExpandVST(MBBI, ARM::VST4d8, false, SingleSpc, 4); break;
    case ARM::VST4d16Pseudo:
      ExpandVST(MBBI, ARM::VST4d16, false, SingleSpc, 4); break;
    case ARM::VST4d32Pseudo:
      ExpandVST(MBBI, ARM::VST4d32, false, SingleSpc, 4); break;
    case ARM::VST1d64QPseudo:
      ExpandVST(MBBI, ARM::VST1d64Q, false, SingleSpc, 4); break;
    case ARM::VST4d8Pseudo_UPD:
      ExpandVST(MBBI, ARM::VST4d8_UPD, true, SingleSpc, 4); break;
    case ARM::VST4d16Pseudo_UPD:
      ExpandVST(MBBI, ARM::VST4d16_UPD, true, SingleSpc, 4); break;
    case ARM::VST4d32Pseudo_UPD:
      ExpandVST(MBBI, ARM::VST4d32_UPD, true, SingleSpc, 4); break;
    case ARM::VST1d64QPseudo_UPD:
      ExpandVST(MBBI, ARM::VST1d64Q_UPD, true, SingleSpc, 4); break;
    case ARM::VST4q8Pseudo_UPD:
      ExpandVST(MBBI, ARM::VST4q8_UPD, true, EvenDblSpc, 4); break;
    case ARM::VST4q16Pseudo_UPD:
      ExpandVST(MBBI, ARM::VST4q16_UPD, true, EvenDblSpc, 4); break;
    case ARM::VST4q32Pseudo_UPD:
      ExpandVST(MBBI, ARM::VST4q32_UPD, true, EvenDblSpc, 4); break;
    case ARM::VST4q8oddPseudo_UPD:
      ExpandVST(MBBI, ARM::VST4q8_UPD, true, OddDblSpc, 4); break;
    case ARM::VST4q16oddPseudo_UPD:
      ExpandVST(MBBI, ARM::VST4q16_UPD, true, OddDblSpc, 4); break;
    case ARM::VST4q32oddPseudo_UPD:
      ExpandVST(MBBI, ARM::VST4q32_UPD, true, OddDblSpc, 4); break;
    }

    if (ModifiedOp)
      Modified = true;
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
