//===- Thumb2InstrInfo.cpp - Thumb-2 Instruction Information --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Thumb-2 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARMInstrInfo.h"
#include "ARM.h"
#include "ARMGenInstrInfo.inc"
#include "ARMMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/ADT/SmallVector.h"
#include "Thumb2InstrInfo.h"

using namespace llvm;

Thumb2InstrInfo::Thumb2InstrInfo(const ARMSubtarget &STI)
  : ARMBaseInstrInfo(STI), RI(*this, STI) {
}

unsigned Thumb2InstrInfo::
getUnindexedOpcode(unsigned Opc) const {
  // FIXME
  return 0;
}

unsigned Thumb2InstrInfo::
getOpcode(ARMII::Op Op) const {
  switch (Op) {
  case ARMII::ADDri: return ARM::t2ADDri;
  case ARMII::ADDrs: return ARM::t2ADDrs;
  case ARMII::ADDrr: return ARM::t2ADDrr;
  case ARMII::B: return ARM::t2B;
  case ARMII::Bcc: return ARM::t2Bcc;
  case ARMII::BR_JTr: return ARM::t2BR_JTr;
  case ARMII::BR_JTm: return ARM::t2BR_JTm;
  case ARMII::BR_JTadd: return ARM::t2BR_JTadd;
  case ARMII::FCPYS: return ARM::FCPYS;
  case ARMII::FCPYD: return ARM::FCPYD;
  case ARMII::FLDD: return ARM::FLDD;
  case ARMII::FLDS: return ARM::FLDS;
  case ARMII::FSTD: return ARM::FSTD;
  case ARMII::FSTS: return ARM::FSTS;
  case ARMII::LDR: return ARM::LDR;   // FIXME
  case ARMII::MOVr: return ARM::t2MOVr;
  case ARMII::STR: return ARM::STR;   // FIXME
  case ARMII::SUBri: return ARM::t2SUBri;
  case ARMII::SUBrs: return ARM::t2SUBrs;
  case ARMII::SUBrr: return ARM::t2SUBrr;
  case ARMII::VMOVD: return ARM::VMOVD;
  case ARMII::VMOVQ: return ARM::VMOVQ;
  default:
    break;
  }

  return 0;
}

bool
Thumb2InstrInfo::BlockHasNoFallThrough(const MachineBasicBlock &MBB) const {
  if (MBB.empty()) return false;

  // FIXME
  switch (MBB.back().getOpcode()) {
    //case ARM::t2BX_RET:
    //  case ARM::LDM_RET:
  case ARM::t2B:        // Uncond branch.
  case ARM::t2BR_JTr:   // Jumptable branch.
  case ARM::t2BR_JTm:   // Jumptable branch through mem.
  case ARM::t2BR_JTadd: // Jumptable branch add to pc.
    return true;
  case ARM::tBX_RET:
  case ARM::tBX_RET_vararg:
  case ARM::tPOP_RET:
  case ARM::tB:
  case ARM::tBR_JTr:
    return true;
  default:
    break;
  }

  return false;
}


bool Thumb2InstrInfo::copyRegToReg(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator I,
                                   unsigned DestReg, unsigned SrcReg,
                                   const TargetRegisterClass *DestRC,
                                   const TargetRegisterClass *SrcRC) const {
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (I != MBB.end()) DL = I->getDebugLoc();

  if (DestRC == ARM::GPRRegisterClass) {
    if (SrcRC == ARM::GPRRegisterClass) {
      return ARMBaseInstrInfo::copyRegToReg(MBB, I, DestReg, SrcReg, DestRC, SrcRC);
    } else if (SrcRC == ARM::tGPRRegisterClass) {
      BuildMI(MBB, I, DL, get(ARM::tMOVlor2hir), DestReg).addReg(SrcReg);
      return true;
    }
  } else if (DestRC == ARM::tGPRRegisterClass) {
    if (SrcRC == ARM::GPRRegisterClass) {
      BuildMI(MBB, I, DL, get(ARM::tMOVhir2lor), DestReg).addReg(SrcReg);
      return true;
    } else if (SrcRC == ARM::tGPRRegisterClass) {
      BuildMI(MBB, I, DL, get(ARM::tMOVr), DestReg).addReg(SrcReg);
      return true;
    }
  }

  return false;
}






bool Thumb2InstrInfo::isMoveInstr(const MachineInstr &MI,
                                  unsigned &SrcReg, unsigned &DstReg,
                                  unsigned& SrcSubIdx, unsigned& DstSubIdx) const {
  SrcSubIdx = DstSubIdx = 0; // No sub-registers.

  unsigned oc = MI.getOpcode();
  switch (oc) {
  default:
    return false;
  case ARM::tMOVr:
  case ARM::tMOVhir2lor:
  case ARM::tMOVlor2hir:
  case ARM::tMOVhir2hir:
    assert(MI.getDesc().getNumOperands() >= 2 &&
           MI.getOperand(0).isReg() &&
           MI.getOperand(1).isReg() &&
           "Invalid Thumb MOV instruction");
    SrcReg = MI.getOperand(1).getReg();
    DstReg = MI.getOperand(0).getReg();
    return true;
  }
}

unsigned Thumb2InstrInfo::isLoadFromStackSlot(const MachineInstr *MI,
                                              int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case ARM::tRestore:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isImm() &&
        MI->getOperand(2).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

unsigned Thumb2InstrInfo::isStoreToStackSlot(const MachineInstr *MI,
                                             int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case ARM::tSpill:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isImm() &&
        MI->getOperand(2).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

bool Thumb2InstrInfo::
canFoldMemoryOperand(const MachineInstr *MI,
                     const SmallVectorImpl<unsigned> &Ops) const {
  if (Ops.size() != 1) return false;

  unsigned OpNum = Ops[0];
  unsigned Opc = MI->getOpcode();
  switch (Opc) {
  default: break;
  case ARM::tMOVr:
  case ARM::tMOVlor2hir:
  case ARM::tMOVhir2lor:
  case ARM::tMOVhir2hir: {
    if (OpNum == 0) { // move -> store
      unsigned SrcReg = MI->getOperand(1).getReg();
      if (RI.isPhysicalRegister(SrcReg) && !isARMLowRegister(SrcReg))
        // tSpill cannot take a high register operand.
        return false;
    } else {          // move -> load
      unsigned DstReg = MI->getOperand(0).getReg();
      if (RI.isPhysicalRegister(DstReg) && !isARMLowRegister(DstReg))
        // tRestore cannot target a high register operand.
        return false;
    }
    return true;
  }
  }

  return false;
}

void Thumb2InstrInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                    unsigned SrcReg, bool isKill, int FI,
                    const TargetRegisterClass *RC) const {
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (I != MBB.end()) DL = I->getDebugLoc();

  assert(RC == ARM::tGPRRegisterClass && "Unknown regclass!");

  if (RC == ARM::tGPRRegisterClass) {
    BuildMI(MBB, I, DL, get(ARM::tSpill))
      .addReg(SrcReg, getKillRegState(isKill))
      .addFrameIndex(FI).addImm(0);
  }
}

void Thumb2InstrInfo::storeRegToAddr(MachineFunction &MF, unsigned SrcReg,
                                     bool isKill,
                                     SmallVectorImpl<MachineOperand> &Addr,
                                     const TargetRegisterClass *RC,
                                     SmallVectorImpl<MachineInstr*> &NewMIs) const{
  DebugLoc DL = DebugLoc::getUnknownLoc();
  unsigned Opc = 0;

  assert(RC == ARM::GPRRegisterClass && "Unknown regclass!");
  if (RC == ARM::GPRRegisterClass) {
    Opc = Addr[0].isFI() ? ARM::tSpill : ARM::tSTR;
  }

  MachineInstrBuilder MIB =
    BuildMI(MF, DL,  get(Opc)).addReg(SrcReg, getKillRegState(isKill));
  for (unsigned i = 0, e = Addr.size(); i != e; ++i)
    MIB.addOperand(Addr[i]);
  NewMIs.push_back(MIB);
  return;
}

void Thumb2InstrInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     unsigned DestReg, int FI,
                     const TargetRegisterClass *RC) const {
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (I != MBB.end()) DL = I->getDebugLoc();

  assert(RC == ARM::tGPRRegisterClass && "Unknown regclass!");

  if (RC == ARM::tGPRRegisterClass) {
    BuildMI(MBB, I, DL, get(ARM::tRestore), DestReg)
      .addFrameIndex(FI).addImm(0);
  }
}

void Thumb2InstrInfo::
loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                SmallVectorImpl<MachineOperand> &Addr,
                const TargetRegisterClass *RC,
                SmallVectorImpl<MachineInstr*> &NewMIs) const {
  DebugLoc DL = DebugLoc::getUnknownLoc();
  unsigned Opc = 0;

  if (RC == ARM::GPRRegisterClass) {
    Opc = Addr[0].isFI() ? ARM::tRestore : ARM::tLDR;
  }

  MachineInstrBuilder MIB = BuildMI(MF, DL, get(Opc), DestReg);
  for (unsigned i = 0, e = Addr.size(); i != e; ++i)
    MIB.addOperand(Addr[i]);
  NewMIs.push_back(MIB);
  return;
}

bool Thumb2InstrInfo::
spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator MI,
                          const std::vector<CalleeSavedInfo> &CSI) const {
  if (CSI.empty())
    return false;

  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (MI != MBB.end()) DL = MI->getDebugLoc();

  MachineInstrBuilder MIB = BuildMI(MBB, MI, DL, get(ARM::tPUSH));
  for (unsigned i = CSI.size(); i != 0; --i) {
    unsigned Reg = CSI[i-1].getReg();
    // Add the callee-saved register as live-in. It's killed at the spill.
    MBB.addLiveIn(Reg);
    MIB.addReg(Reg, RegState::Kill);
  }
  return true;
}

bool Thumb2InstrInfo::
restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI,
                            const std::vector<CalleeSavedInfo> &CSI) const {
  MachineFunction &MF = *MBB.getParent();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  if (CSI.empty())
    return false;

  bool isVarArg = AFI->getVarArgsRegSaveSize() > 0;
  MachineInstr *PopMI = MF.CreateMachineInstr(get(ARM::tPOP),MI->getDebugLoc());
  for (unsigned i = CSI.size(); i != 0; --i) {
    unsigned Reg = CSI[i-1].getReg();
    if (Reg == ARM::LR) {
      // Special epilogue for vararg functions. See emitEpilogue
      if (isVarArg)
        continue;
      Reg = ARM::PC;
      PopMI->setDesc(get(ARM::tPOP_RET));
      MI = MBB.erase(MI);
    }
    PopMI->addOperand(MachineOperand::CreateReg(Reg, true));
  }

  // It's illegal to emit pop instruction without operands.
  if (PopMI->getNumOperands() > 0)
    MBB.insert(MI, PopMI);

  return true;
}

MachineInstr *Thumb2InstrInfo::
foldMemoryOperandImpl(MachineFunction &MF, MachineInstr *MI,
                      const SmallVectorImpl<unsigned> &Ops, int FI) const {
  if (Ops.size() != 1) return NULL;

  unsigned OpNum = Ops[0];
  unsigned Opc = MI->getOpcode();
  MachineInstr *NewMI = NULL;
  switch (Opc) {
  default: break;
  case ARM::tMOVr:
  case ARM::tMOVlor2hir:
  case ARM::tMOVhir2lor:
  case ARM::tMOVhir2hir: {
    if (OpNum == 0) { // move -> store
      unsigned SrcReg = MI->getOperand(1).getReg();
      bool isKill = MI->getOperand(1).isKill();
      if (RI.isPhysicalRegister(SrcReg) && !isARMLowRegister(SrcReg))
        // tSpill cannot take a high register operand.
        break;
      NewMI = BuildMI(MF, MI->getDebugLoc(), get(ARM::tSpill))
        .addReg(SrcReg, getKillRegState(isKill))
        .addFrameIndex(FI).addImm(0);
    } else {          // move -> load
      unsigned DstReg = MI->getOperand(0).getReg();
      if (RI.isPhysicalRegister(DstReg) && !isARMLowRegister(DstReg))
        // tRestore cannot target a high register operand.
        break;
      bool isDead = MI->getOperand(0).isDead();
      NewMI = BuildMI(MF, MI->getDebugLoc(), get(ARM::tRestore))
        .addReg(DstReg, RegState::Define | getDeadRegState(isDead))
        .addFrameIndex(FI).addImm(0);
    }
    break;
  }
  }

  return NewMI;
}
