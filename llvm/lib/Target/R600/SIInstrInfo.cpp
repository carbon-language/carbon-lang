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
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/MC/MCInstrDesc.h"

using namespace llvm;

SIInstrInfo::SIInstrInfo(AMDGPUTargetMachine &tm)
  : AMDGPUInstrInfo(tm),
    RI(tm)
    { }

const SIRegisterInfo &SIInstrInfo::getRegisterInfo() const {
  return RI;
}

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

MachineInstr *SIInstrInfo::commuteInstruction(MachineInstr *MI,
                                              bool NewMI) const {

  if (MI->getNumOperands() < 3 || !MI->getOperand(1).isReg() ||
      !MI->getOperand(2).isReg())
    return 0;

  MI = TargetInstrInfo::commuteInstruction(MI, NewMI);

  if (MI)
    MI->setDesc(get(commuteOpcode(MI->getOpcode())));

  return MI;
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
    if (Src1.isImm()) {
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

//===----------------------------------------------------------------------===//
// Indirect addressing callbacks
//===----------------------------------------------------------------------===//

unsigned SIInstrInfo::calculateIndirectAddress(unsigned RegIndex,
                                                 unsigned Channel) const {
  assert(Channel == 0);
  return RegIndex;
}


int SIInstrInfo::getIndirectIndexBegin(const MachineFunction &MF) const {
  llvm_unreachable("Unimplemented");
}

int SIInstrInfo::getIndirectIndexEnd(const MachineFunction &MF) const {
  llvm_unreachable("Unimplemented");
}

const TargetRegisterClass *SIInstrInfo::getIndirectAddrStoreRegClass(
                                                     unsigned SourceReg) const {
  llvm_unreachable("Unimplemented");
}

const TargetRegisterClass *SIInstrInfo::getIndirectAddrLoadRegClass() const {
  llvm_unreachable("Unimplemented");
}

MachineInstrBuilder SIInstrInfo::buildIndirectWrite(
                                   MachineBasicBlock *MBB,
                                   MachineBasicBlock::iterator I,
                                   unsigned ValueReg,
                                   unsigned Address, unsigned OffsetReg) const {
  llvm_unreachable("Unimplemented");
}

MachineInstrBuilder SIInstrInfo::buildIndirectRead(
                                   MachineBasicBlock *MBB,
                                   MachineBasicBlock::iterator I,
                                   unsigned ValueReg,
                                   unsigned Address, unsigned OffsetReg) const {
  llvm_unreachable("Unimplemented");
}

const TargetRegisterClass *SIInstrInfo::getSuperIndirectRegClass() const {
  llvm_unreachable("Unimplemented");
}
