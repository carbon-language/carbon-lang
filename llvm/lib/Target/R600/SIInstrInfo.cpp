//===-- SIInstrInfo.cpp - SI Instruction Information  ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief SI Implementation of TargetInstrInfo.
//
//===----------------------------------------------------------------------===//


#include "SIInstrInfo.h"
#include "AMDGPUTargetMachine.h"
#include "SIDefines.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/MC/MCInstrDesc.h"

using namespace llvm;

SIInstrInfo::SIInstrInfo(AMDGPUTargetMachine &tm)
  : AMDGPUInstrInfo(tm),
    RI(tm) { }

//===----------------------------------------------------------------------===//
// TargetInstrInfo callbacks
//===----------------------------------------------------------------------===//

void
SIInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MI, DebugLoc DL,
                         unsigned DestReg, unsigned SrcReg,
                         bool KillSrc) const {

  // If we are trying to copy to or from SCC, there is a bug somewhere else in
  // the backend.  While it may be theoretically possible to do this, it should
  // never be necessary.
  assert(DestReg != AMDGPU::SCC && SrcReg != AMDGPU::SCC);

  static const int16_t Sub0_15[] = {
    AMDGPU::sub0, AMDGPU::sub1, AMDGPU::sub2, AMDGPU::sub3,
    AMDGPU::sub4, AMDGPU::sub5, AMDGPU::sub6, AMDGPU::sub7,
    AMDGPU::sub8, AMDGPU::sub9, AMDGPU::sub10, AMDGPU::sub11,
    AMDGPU::sub12, AMDGPU::sub13, AMDGPU::sub14, AMDGPU::sub15, 0
  };

  static const int16_t Sub0_7[] = {
    AMDGPU::sub0, AMDGPU::sub1, AMDGPU::sub2, AMDGPU::sub3,
    AMDGPU::sub4, AMDGPU::sub5, AMDGPU::sub6, AMDGPU::sub7, 0
  };

  static const int16_t Sub0_3[] = {
    AMDGPU::sub0, AMDGPU::sub1, AMDGPU::sub2, AMDGPU::sub3, 0
  };

  static const int16_t Sub0_2[] = {
    AMDGPU::sub0, AMDGPU::sub1, AMDGPU::sub2, 0
  };

  static const int16_t Sub0_1[] = {
    AMDGPU::sub0, AMDGPU::sub1, 0
  };

  unsigned Opcode;
  const int16_t *SubIndices;

  if (AMDGPU::M0 == DestReg) {
    // Check if M0 isn't already set to this value
    for (MachineBasicBlock::reverse_iterator E = MBB.rend(),
      I = MachineBasicBlock::reverse_iterator(MI); I != E; ++I) {

      if (!I->definesRegister(AMDGPU::M0))
        continue;

      unsigned Opc = I->getOpcode();
      if (Opc != TargetOpcode::COPY && Opc != AMDGPU::S_MOV_B32)
        break;

      if (!I->readsRegister(SrcReg))
        break;

      // The copy isn't necessary
      return;
    }
  }

  if (AMDGPU::SReg_32RegClass.contains(DestReg)) {
    assert(AMDGPU::SReg_32RegClass.contains(SrcReg));
    BuildMI(MBB, MI, DL, get(AMDGPU::S_MOV_B32), DestReg)
            .addReg(SrcReg, getKillRegState(KillSrc));
    return;

  } else if (AMDGPU::SReg_64RegClass.contains(DestReg)) {
    assert(AMDGPU::SReg_64RegClass.contains(SrcReg));
    BuildMI(MBB, MI, DL, get(AMDGPU::S_MOV_B64), DestReg)
            .addReg(SrcReg, getKillRegState(KillSrc));
    return;

  } else if (AMDGPU::SReg_128RegClass.contains(DestReg)) {
    assert(AMDGPU::SReg_128RegClass.contains(SrcReg));
    Opcode = AMDGPU::S_MOV_B32;
    SubIndices = Sub0_3;

  } else if (AMDGPU::SReg_256RegClass.contains(DestReg)) {
    assert(AMDGPU::SReg_256RegClass.contains(SrcReg));
    Opcode = AMDGPU::S_MOV_B32;
    SubIndices = Sub0_7;

  } else if (AMDGPU::SReg_512RegClass.contains(DestReg)) {
    assert(AMDGPU::SReg_512RegClass.contains(SrcReg));
    Opcode = AMDGPU::S_MOV_B32;
    SubIndices = Sub0_15;

  } else if (AMDGPU::VReg_32RegClass.contains(DestReg)) {
    assert(AMDGPU::VReg_32RegClass.contains(SrcReg) ||
           AMDGPU::SReg_32RegClass.contains(SrcReg));
    BuildMI(MBB, MI, DL, get(AMDGPU::V_MOV_B32_e32), DestReg)
            .addReg(SrcReg, getKillRegState(KillSrc));
    return;

  } else if (AMDGPU::VReg_64RegClass.contains(DestReg)) {
    assert(AMDGPU::VReg_64RegClass.contains(SrcReg) ||
           AMDGPU::SReg_64RegClass.contains(SrcReg));
    Opcode = AMDGPU::V_MOV_B32_e32;
    SubIndices = Sub0_1;

  } else if (AMDGPU::VReg_96RegClass.contains(DestReg)) {
    assert(AMDGPU::VReg_96RegClass.contains(SrcReg));
    Opcode = AMDGPU::V_MOV_B32_e32;
    SubIndices = Sub0_2;

  } else if (AMDGPU::VReg_128RegClass.contains(DestReg)) {
    assert(AMDGPU::VReg_128RegClass.contains(SrcReg) ||
           AMDGPU::SReg_128RegClass.contains(SrcReg));
    Opcode = AMDGPU::V_MOV_B32_e32;
    SubIndices = Sub0_3;

  } else if (AMDGPU::VReg_256RegClass.contains(DestReg)) {
    assert(AMDGPU::VReg_256RegClass.contains(SrcReg) ||
           AMDGPU::SReg_256RegClass.contains(SrcReg));
    Opcode = AMDGPU::V_MOV_B32_e32;
    SubIndices = Sub0_7;

  } else if (AMDGPU::VReg_512RegClass.contains(DestReg)) {
    assert(AMDGPU::VReg_512RegClass.contains(SrcReg) ||
           AMDGPU::SReg_512RegClass.contains(SrcReg));
    Opcode = AMDGPU::V_MOV_B32_e32;
    SubIndices = Sub0_15;

  } else {
    llvm_unreachable("Can't copy register!");
  }

  while (unsigned SubIdx = *SubIndices++) {
    MachineInstrBuilder Builder = BuildMI(MBB, MI, DL,
      get(Opcode), RI.getSubReg(DestReg, SubIdx));

    Builder.addReg(RI.getSubReg(SrcReg, SubIdx), getKillRegState(KillSrc));

    if (*SubIndices)
      Builder.addReg(DestReg, RegState::Define | RegState::Implicit);
  }
}

unsigned SIInstrInfo::commuteOpcode(unsigned Opcode) const {
  int NewOpc;

  // Try to map original to commuted opcode
  if ((NewOpc = AMDGPU::getCommuteRev(Opcode)) != -1)
    return NewOpc;

  // Try to map commuted to original opcode
  if ((NewOpc = AMDGPU::getCommuteOrig(Opcode)) != -1)
    return NewOpc;

  return Opcode;
}

void SIInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator MI,
                                      unsigned SrcReg, bool isKill,
                                      int FrameIndex,
                                      const TargetRegisterClass *RC,
                                      const TargetRegisterInfo *TRI) const {
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  SIMachineFunctionInfo *MFI = MBB.getParent()->getInfo<SIMachineFunctionInfo>();
  DebugLoc DL = MBB.findDebugLoc(MI);
  unsigned KillFlag = isKill ? RegState::Kill : 0;

  if (TRI->getCommonSubClass(RC, &AMDGPU::SGPR_32RegClass)) {
    unsigned Lane = MFI->SpillTracker.getNextLane(MRI);
    BuildMI(MBB, MI, DL, get(AMDGPU::V_WRITELANE_B32),
            MFI->SpillTracker.LaneVGPR)
            .addReg(SrcReg, KillFlag)
            .addImm(Lane);
    MFI->SpillTracker.addSpilledReg(FrameIndex, MFI->SpillTracker.LaneVGPR,
                                    Lane);
  } else {
    for (unsigned i = 0, e = RC->getSize() / 4; i != e; ++i) {
      unsigned SubReg = MRI.createVirtualRegister(&AMDGPU::SReg_32RegClass);
      BuildMI(MBB, MI, MBB.findDebugLoc(MI), get(AMDGPU::COPY), SubReg)
              .addReg(SrcReg, 0, RI.getSubRegFromChannel(i));
      storeRegToStackSlot(MBB, MI, SubReg, isKill, FrameIndex + i,
                          &AMDGPU::SReg_32RegClass, TRI);
    }
  }
}

void SIInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator MI,
                                       unsigned DestReg, int FrameIndex,
                                       const TargetRegisterClass *RC,
                                       const TargetRegisterInfo *TRI) const {
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  SIMachineFunctionInfo *MFI = MBB.getParent()->getInfo<SIMachineFunctionInfo>();
  DebugLoc DL = MBB.findDebugLoc(MI);
  if (TRI->getCommonSubClass(RC, &AMDGPU::SReg_32RegClass)) {
     SIMachineFunctionInfo::SpilledReg Spill =
        MFI->SpillTracker.getSpilledReg(FrameIndex);
    assert(Spill.VGPR);
    BuildMI(MBB, MI, DL, get(AMDGPU::V_READLANE_B32), DestReg)
            .addReg(Spill.VGPR)
            .addImm(Spill.Lane);
  } else {
    for (unsigned i = 0, e = RC->getSize() / 4; i != e; ++i) {
      unsigned Flags = RegState::Define;
      if (i == 0) {
        Flags |= RegState::Undef;
      }
      unsigned SubReg = MRI.createVirtualRegister(&AMDGPU::SReg_32RegClass);
      loadRegFromStackSlot(MBB, MI, SubReg, FrameIndex + i,
                           &AMDGPU::SReg_32RegClass, TRI);
      BuildMI(MBB, MI, DL, get(AMDGPU::COPY))
              .addReg(DestReg, Flags, RI.getSubRegFromChannel(i))
              .addReg(SubReg);
    }
  }
}

MachineInstr *SIInstrInfo::commuteInstruction(MachineInstr *MI,
                                              bool NewMI) const {

  MachineRegisterInfo &MRI = MI->getParent()->getParent()->getRegInfo();
  if (MI->getNumOperands() < 3 || !MI->getOperand(1).isReg())
    return 0;

  // Cannot commute VOP2 if src0 is SGPR.
  if (isVOP2(MI->getOpcode()) && MI->getOperand(1).isReg() &&
      RI.isSGPRClass(MRI.getRegClass(MI->getOperand(1).getReg())))
   return 0;

  if (!MI->getOperand(2).isReg()) {
    // XXX: Commute instructions with FPImm operands
    if (NewMI || MI->getOperand(2).isFPImm() ||
       (!isVOP2(MI->getOpcode()) && !isVOP3(MI->getOpcode()))) {
      return 0;
    }

    // XXX: Commute VOP3 instructions with abs and neg set.
    if (isVOP3(MI->getOpcode()) &&
        (MI->getOperand(AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                        AMDGPU::OpName::abs)).getImm() ||
         MI->getOperand(AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                        AMDGPU::OpName::neg)).getImm()))
      return 0;

    unsigned Reg = MI->getOperand(1).getReg();
    unsigned SubReg = MI->getOperand(1).getSubReg();
    MI->getOperand(1).ChangeToImmediate(MI->getOperand(2).getImm());
    MI->getOperand(2).ChangeToRegister(Reg, false);
    MI->getOperand(2).setSubReg(SubReg);
  } else {
    MI = TargetInstrInfo::commuteInstruction(MI, NewMI);
  }

  if (MI)
    MI->setDesc(get(commuteOpcode(MI->getOpcode())));

  return MI;
}

MachineInstr *SIInstrInfo::buildMovInstr(MachineBasicBlock *MBB,
                                         MachineBasicBlock::iterator I,
                                         unsigned DstReg,
                                         unsigned SrcReg) const {
  return BuildMI(*MBB, I, MBB->findDebugLoc(I), get(AMDGPU::V_MOV_B32_e32),
                 DstReg) .addReg(SrcReg);
}

bool SIInstrInfo::isMov(unsigned Opcode) const {
  switch(Opcode) {
  default: return false;
  case AMDGPU::S_MOV_B32:
  case AMDGPU::S_MOV_B64:
  case AMDGPU::V_MOV_B32_e32:
  case AMDGPU::V_MOV_B32_e64:
    return true;
  }
}

bool
SIInstrInfo::isSafeToMoveRegClassDefs(const TargetRegisterClass *RC) const {
  return RC != &AMDGPU::EXECRegRegClass;
}

namespace llvm {
namespace AMDGPU {
// Helper function generated by tablegen.  We are wrapping this with
// an SIInstrInfo function that reutrns bool rather than int.
int isDS(uint16_t Opcode);
}
}

bool SIInstrInfo::isDS(uint16_t Opcode) const {
  return ::AMDGPU::isDS(Opcode) != -1;
}

int SIInstrInfo::isMIMG(uint16_t Opcode) const {
  return get(Opcode).TSFlags & SIInstrFlags::MIMG;
}

int SIInstrInfo::isSMRD(uint16_t Opcode) const {
  return get(Opcode).TSFlags & SIInstrFlags::SMRD;
}

bool SIInstrInfo::isVOP1(uint16_t Opcode) const {
  return get(Opcode).TSFlags & SIInstrFlags::VOP1;
}

bool SIInstrInfo::isVOP2(uint16_t Opcode) const {
  return get(Opcode).TSFlags & SIInstrFlags::VOP2;
}

bool SIInstrInfo::isVOP3(uint16_t Opcode) const {
  return get(Opcode).TSFlags & SIInstrFlags::VOP3;
}

bool SIInstrInfo::isVOPC(uint16_t Opcode) const {
  return get(Opcode).TSFlags & SIInstrFlags::VOPC;
}

bool SIInstrInfo::isSALUInstr(const MachineInstr &MI) const {
  return get(MI.getOpcode()).TSFlags & SIInstrFlags::SALU;
}

bool SIInstrInfo::isInlineConstant(const MachineOperand &MO) const {
  if(MO.isImm()) {
    return MO.getImm() >= -16 && MO.getImm() <= 64;
  }
  if (MO.isFPImm()) {
    return MO.getFPImm()->isExactlyValue(0.0)  ||
           MO.getFPImm()->isExactlyValue(0.5)  ||
           MO.getFPImm()->isExactlyValue(-0.5) ||
           MO.getFPImm()->isExactlyValue(1.0)  ||
           MO.getFPImm()->isExactlyValue(-1.0) ||
           MO.getFPImm()->isExactlyValue(2.0)  ||
           MO.getFPImm()->isExactlyValue(-2.0) ||
           MO.getFPImm()->isExactlyValue(4.0)  ||
           MO.getFPImm()->isExactlyValue(-4.0);
  }
  return false;
}

bool SIInstrInfo::isLiteralConstant(const MachineOperand &MO) const {
  return (MO.isImm() || MO.isFPImm()) && !isInlineConstant(MO);
}

bool SIInstrInfo::verifyInstruction(const MachineInstr *MI,
                                    StringRef &ErrInfo) const {
  uint16_t Opcode = MI->getOpcode();
  int Src0Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src0);
  int Src1Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src1);
  int Src2Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src2);

  // Verify VOP*
  if (isVOP1(Opcode) || isVOP2(Opcode) || isVOP3(Opcode) || isVOPC(Opcode)) {
    unsigned ConstantBusCount = 0;
    unsigned SGPRUsed = AMDGPU::NoRegister;
    for (int i = 0, e = MI->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg() && MO.isUse() &&
          !TargetRegisterInfo::isVirtualRegister(MO.getReg())) {

        // EXEC register uses the constant bus.
        if (!MO.isImplicit() && MO.getReg() == AMDGPU::EXEC)
          ++ConstantBusCount;

        // SGPRs use the constant bus
        if (MO.getReg() == AMDGPU::M0 || MO.getReg() == AMDGPU::VCC ||
            (!MO.isImplicit() &&
            (AMDGPU::SGPR_32RegClass.contains(MO.getReg()) ||
            AMDGPU::SGPR_64RegClass.contains(MO.getReg())))) {
          if (SGPRUsed != MO.getReg()) {
            ++ConstantBusCount;
            SGPRUsed = MO.getReg();
          }
        }
      }
      // Literal constants use the constant bus.
      if (isLiteralConstant(MO))
        ++ConstantBusCount;
    }
    if (ConstantBusCount > 1) {
      ErrInfo = "VOP* instruction uses the constant bus more than once";
      return false;
    }
  }

  // Verify SRC1 for VOP2 and VOPC
  if (Src1Idx != -1 && (isVOP2(Opcode) || isVOPC(Opcode))) {
    const MachineOperand &Src1 = MI->getOperand(Src1Idx);
    if (Src1.isImm() || Src1.isFPImm()) {
      ErrInfo = "VOP[2C] src1 cannot be an immediate.";
      return false;
    }
  }

  // Verify VOP3
  if (isVOP3(Opcode)) {
    if (Src0Idx != -1 && isLiteralConstant(MI->getOperand(Src0Idx))) {
      ErrInfo = "VOP3 src0 cannot be a literal constant.";
      return false;
    }
    if (Src1Idx != -1 && isLiteralConstant(MI->getOperand(Src1Idx))) {
      ErrInfo = "VOP3 src1 cannot be a literal constant.";
      return false;
    }
    if (Src2Idx != -1 && isLiteralConstant(MI->getOperand(Src2Idx))) {
      ErrInfo = "VOP3 src2 cannot be a literal constant.";
      return false;
    }
  }
  return true;
}

unsigned SIInstrInfo::getVALUOp(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  default: return AMDGPU::INSTRUCTION_LIST_END;
  case AMDGPU::REG_SEQUENCE: return AMDGPU::REG_SEQUENCE;
  case AMDGPU::COPY: return AMDGPU::COPY;
  case AMDGPU::PHI: return AMDGPU::PHI;
  case AMDGPU::S_ADD_I32: return AMDGPU::V_ADD_I32_e32;
  case AMDGPU::S_ADDC_U32: return AMDGPU::V_ADDC_U32_e32;
  case AMDGPU::S_SUB_I32: return AMDGPU::V_SUB_I32_e32;
  case AMDGPU::S_SUBB_U32: return AMDGPU::V_SUBB_U32_e32;
  case AMDGPU::S_ASHR_I32: return AMDGPU::V_ASHR_I32_e32;
  case AMDGPU::S_ASHR_I64: return AMDGPU::V_ASHR_I64;
  case AMDGPU::S_LSHL_B32: return AMDGPU::V_LSHL_B32_e32;
  case AMDGPU::S_LSHL_B64: return AMDGPU::V_LSHL_B64;
  case AMDGPU::S_LSHR_B32: return AMDGPU::V_LSHR_B32_e32;
  case AMDGPU::S_LSHR_B64: return AMDGPU::V_LSHR_B64;
  }
}

bool SIInstrInfo::isSALUOpSupportedOnVALU(const MachineInstr &MI) const {
  return getVALUOp(MI) != AMDGPU::INSTRUCTION_LIST_END;
}

const TargetRegisterClass *SIInstrInfo::getOpRegClass(const MachineInstr &MI,
                                                      unsigned OpNo) const {
  const MachineRegisterInfo &MRI = MI.getParent()->getParent()->getRegInfo();
  const MCInstrDesc &Desc = get(MI.getOpcode());
  if (MI.isVariadic() || OpNo >= Desc.getNumOperands() ||
      Desc.OpInfo[OpNo].RegClass == -1)
    return MRI.getRegClass(MI.getOperand(OpNo).getReg());

  unsigned RCID = Desc.OpInfo[OpNo].RegClass;
  return RI.getRegClass(RCID);
}

bool SIInstrInfo::canReadVGPR(const MachineInstr &MI, unsigned OpNo) const {
  switch (MI.getOpcode()) {
  case AMDGPU::COPY:
  case AMDGPU::REG_SEQUENCE:
    return RI.hasVGPRs(getOpRegClass(MI, 0));
  default:
    return RI.hasVGPRs(getOpRegClass(MI, OpNo));
  }
}

void SIInstrInfo::legalizeOpWithMove(MachineInstr *MI, unsigned OpIdx) const {
  MachineBasicBlock::iterator I = MI;
  MachineOperand &MO = MI->getOperand(OpIdx);
  MachineRegisterInfo &MRI = MI->getParent()->getParent()->getRegInfo();
  unsigned RCID = get(MI->getOpcode()).OpInfo[OpIdx].RegClass;
  const TargetRegisterClass *RC = RI.getRegClass(RCID);
  unsigned Opcode = AMDGPU::V_MOV_B32_e32;
  if (MO.isReg()) {
    Opcode = AMDGPU::COPY;
  } else if (RI.isSGPRClass(RC)) {
    Opcode = AMDGPU::S_MOV_B32;
  }

  const TargetRegisterClass *VRC = RI.getEquivalentVGPRClass(RC);
  unsigned Reg = MRI.createVirtualRegister(VRC);
  BuildMI(*MI->getParent(), I, MI->getParent()->findDebugLoc(I), get(Opcode),
          Reg).addOperand(MO);
  MO.ChangeToRegister(Reg, false);
}

void SIInstrInfo::legalizeOperands(MachineInstr *MI) const {
  MachineRegisterInfo &MRI = MI->getParent()->getParent()->getRegInfo();
  int Src0Idx = AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                           AMDGPU::OpName::src0);
  int Src1Idx = AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                           AMDGPU::OpName::src1);
  int Src2Idx = AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                           AMDGPU::OpName::src2);

  // Legalize VOP2
  if (isVOP2(MI->getOpcode()) && Src1Idx != -1) {
    MachineOperand &Src0 = MI->getOperand(Src0Idx);
    MachineOperand &Src1 = MI->getOperand(Src1Idx);

    // If the instruction implicitly reads VCC, we can't have any SGPR operands,
    // so move any.
    bool ReadsVCC = MI->readsRegister(AMDGPU::VCC, &RI);
    if (ReadsVCC && Src0.isReg() &&
        RI.isSGPRClass(MRI.getRegClass(Src0.getReg()))) {
      legalizeOpWithMove(MI, Src0Idx);
      return;
    }

    if (ReadsVCC && Src1.isReg() &&
        RI.isSGPRClass(MRI.getRegClass(Src1.getReg()))) {
      legalizeOpWithMove(MI, Src1Idx);
      return;
    }

    // Legalize VOP2 instructions where src1 is not a VGPR. An SGPR input must
    // be the first operand, and there can only be one.
    if (Src1.isImm() || Src1.isFPImm() ||
        (Src1.isReg() && RI.isSGPRClass(MRI.getRegClass(Src1.getReg())))) {
      if (MI->isCommutable()) {
        if (commuteInstruction(MI))
          return;
      }
      legalizeOpWithMove(MI, Src1Idx);
    }
  }

  // XXX - Do any VOP3 instructions read VCC?
  // Legalize VOP3
  if (isVOP3(MI->getOpcode())) {
    int VOP3Idx[3] = {Src0Idx, Src1Idx, Src2Idx};
    unsigned SGPRReg = AMDGPU::NoRegister;
    for (unsigned i = 0; i < 3; ++i) {
      int Idx = VOP3Idx[i];
      if (Idx == -1)
        continue;
      MachineOperand &MO = MI->getOperand(Idx);

      if (MO.isReg()) {
        if (!RI.isSGPRClass(MRI.getRegClass(MO.getReg())))
          continue; // VGPRs are legal

        assert(MO.getReg() != AMDGPU::SCC && "SCC operand to VOP3 instruction");

        if (SGPRReg == AMDGPU::NoRegister || SGPRReg == MO.getReg()) {
          SGPRReg = MO.getReg();
          // We can use one SGPR in each VOP3 instruction.
          continue;
        }
      } else if (!isLiteralConstant(MO)) {
        // If it is not a register and not a literal constant, then it must be
        // an inline constant which is always legal.
        continue;
      }
      // If we make it this far, then the operand is not legal and we must
      // legalize it.
      legalizeOpWithMove(MI, Idx);
    }
  }

  // Legalize REG_SEQUENCE
  // The register class of the operands much be the same type as the register
  // class of the output.
  if (MI->getOpcode() == AMDGPU::REG_SEQUENCE) {
    const TargetRegisterClass *RC = NULL, *SRC = NULL, *VRC = NULL;
    for (unsigned i = 1, e = MI->getNumOperands(); i != e; i+=2) {
      if (!MI->getOperand(i).isReg() ||
          !TargetRegisterInfo::isVirtualRegister(MI->getOperand(i).getReg()))
        continue;
      const TargetRegisterClass *OpRC =
              MRI.getRegClass(MI->getOperand(i).getReg());
      if (RI.hasVGPRs(OpRC)) {
        VRC = OpRC;
      } else {
        SRC = OpRC;
      }
    }

    // If any of the operands are VGPR registers, then they all most be
    // otherwise we will create illegal VGPR->SGPR copies when legalizing
    // them.
    if (VRC || !RI.isSGPRClass(getOpRegClass(*MI, 0))) {
      if (!VRC) {
        assert(SRC);
        VRC = RI.getEquivalentVGPRClass(SRC);
      }
      RC = VRC;
    } else {
      RC = SRC;
    }

    // Update all the operands so they have the same type.
    for (unsigned i = 1, e = MI->getNumOperands(); i != e; i+=2) {
      if (!MI->getOperand(i).isReg() ||
          !TargetRegisterInfo::isVirtualRegister(MI->getOperand(i).getReg()))
        continue;
      unsigned DstReg = MRI.createVirtualRegister(RC);
      BuildMI(*MI->getParent(), MI, MI->getDebugLoc(),
              get(AMDGPU::COPY), DstReg)
              .addOperand(MI->getOperand(i));
      MI->getOperand(i).setReg(DstReg);
    }
  }
}

void SIInstrInfo::moveToVALU(MachineInstr &TopInst) const {
  SmallVector<MachineInstr *, 128> Worklist;
  Worklist.push_back(&TopInst);

  while (!Worklist.empty()) {
    MachineInstr *Inst = Worklist.pop_back_val();
    unsigned NewOpcode = getVALUOp(*Inst);
    if (NewOpcode == AMDGPU::INSTRUCTION_LIST_END)
      continue;

    MachineRegisterInfo &MRI = Inst->getParent()->getParent()->getRegInfo();

    // Use the new VALU Opcode.
    const MCInstrDesc &NewDesc = get(NewOpcode);
    Inst->setDesc(NewDesc);

    // Remove any references to SCC. Vector instructions can't read from it, and
    // We're just about to add the implicit use / defs of VCC, and we don't want
    // both.
    for (unsigned i = Inst->getNumOperands() - 1; i > 0; --i) {
      MachineOperand &Op = Inst->getOperand(i);
      if (Op.isReg() && Op.getReg() == AMDGPU::SCC)
        Inst->RemoveOperand(i);
    }

    // Add the implict and explicit register definitions.
    if (NewDesc.ImplicitUses) {
      for (unsigned i = 0; NewDesc.ImplicitUses[i]; ++i) {
        unsigned Reg = NewDesc.ImplicitUses[i];
        Inst->addOperand(MachineOperand::CreateReg(Reg, false, true));
      }
    }

    if (NewDesc.ImplicitDefs) {
      for (unsigned i = 0; NewDesc.ImplicitDefs[i]; ++i) {
        unsigned Reg = NewDesc.ImplicitDefs[i];
        Inst->addOperand(MachineOperand::CreateReg(Reg, true, true));
      }
    }

    legalizeOperands(Inst);

    // Update the destination register class.
    const TargetRegisterClass *NewDstRC = getOpRegClass(*Inst, 0);

    switch (Inst->getOpcode()) {
      // For target instructions, getOpRegClass just returns the virtual
      // register class associated with the operand, so we need to find an
      // equivalent VGPR register class in order to move the instruction to the
      // VALU.
    case AMDGPU::COPY:
    case AMDGPU::PHI:
    case AMDGPU::REG_SEQUENCE:
      if (RI.hasVGPRs(NewDstRC))
        continue;
      NewDstRC = RI.getEquivalentVGPRClass(NewDstRC);
      if (!NewDstRC)
        continue;
      break;
    default:
      break;
    }

    unsigned DstReg = Inst->getOperand(0).getReg();
    unsigned NewDstReg = MRI.createVirtualRegister(NewDstRC);
    MRI.replaceRegWith(DstReg, NewDstReg);

    for (MachineRegisterInfo::use_iterator I = MRI.use_begin(NewDstReg),
           E = MRI.use_end(); I != E; ++I) {
      MachineInstr &UseMI = *I;
      if (!canReadVGPR(UseMI, I.getOperandNo())) {
        Worklist.push_back(&UseMI);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Indirect addressing callbacks
//===----------------------------------------------------------------------===//

unsigned SIInstrInfo::calculateIndirectAddress(unsigned RegIndex,
                                                 unsigned Channel) const {
  assert(Channel == 0);
  return RegIndex;
}

const TargetRegisterClass *SIInstrInfo::getIndirectAddrRegClass() const {
  return &AMDGPU::VReg_32RegClass;
}

MachineInstrBuilder SIInstrInfo::buildIndirectWrite(
                                   MachineBasicBlock *MBB,
                                   MachineBasicBlock::iterator I,
                                   unsigned ValueReg,
                                   unsigned Address, unsigned OffsetReg) const {
  const DebugLoc &DL = MBB->findDebugLoc(I);
  unsigned IndirectBaseReg = AMDGPU::VReg_32RegClass.getRegister(
                                      getIndirectIndexBegin(*MBB->getParent()));

  return BuildMI(*MBB, I, DL, get(AMDGPU::SI_INDIRECT_DST_V1))
          .addReg(IndirectBaseReg, RegState::Define)
          .addOperand(I->getOperand(0))
          .addReg(IndirectBaseReg)
          .addReg(OffsetReg)
          .addImm(0)
          .addReg(ValueReg);
}

MachineInstrBuilder SIInstrInfo::buildIndirectRead(
                                   MachineBasicBlock *MBB,
                                   MachineBasicBlock::iterator I,
                                   unsigned ValueReg,
                                   unsigned Address, unsigned OffsetReg) const {
  const DebugLoc &DL = MBB->findDebugLoc(I);
  unsigned IndirectBaseReg = AMDGPU::VReg_32RegClass.getRegister(
                                      getIndirectIndexBegin(*MBB->getParent()));

  return BuildMI(*MBB, I, DL, get(AMDGPU::SI_INDIRECT_SRC))
          .addOperand(I->getOperand(0))
          .addOperand(I->getOperand(1))
          .addReg(IndirectBaseReg)
          .addReg(OffsetReg)
          .addImm(0);

}

void SIInstrInfo::reserveIndirectRegisters(BitVector &Reserved,
                                            const MachineFunction &MF) const {
  int End = getIndirectIndexEnd(MF);
  int Begin = getIndirectIndexBegin(MF);

  if (End == -1)
    return;


  for (int Index = Begin; Index <= End; ++Index)
    Reserved.set(AMDGPU::VReg_32RegClass.getRegister(Index));

  for (int Index = std::max(0, Begin - 1); Index <= End; ++Index)
    Reserved.set(AMDGPU::VReg_64RegClass.getRegister(Index));

  for (int Index = std::max(0, Begin - 2); Index <= End; ++Index)
    Reserved.set(AMDGPU::VReg_96RegClass.getRegister(Index));

  for (int Index = std::max(0, Begin - 3); Index <= End; ++Index)
    Reserved.set(AMDGPU::VReg_128RegClass.getRegister(Index));

  for (int Index = std::max(0, Begin - 7); Index <= End; ++Index)
    Reserved.set(AMDGPU::VReg_256RegClass.getRegister(Index));

  for (int Index = std::max(0, Begin - 15); Index <= End; ++Index)
    Reserved.set(AMDGPU::VReg_512RegClass.getRegister(Index));
}
