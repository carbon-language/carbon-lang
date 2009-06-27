//===- ThumbInstrInfo.cpp - Thumb Instruction Information --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Thumb implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARMInstrInfo.h"
#include "ARM.h"
#include "ARMGenInstrInfo.inc"
#include "ARMMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/ADT/SmallVector.h"
#include "ThumbInstrInfo.h"

using namespace llvm;

ThumbInstrInfo::ThumbInstrInfo(const ARMSubtarget &STI)
  : ARMBaseInstrInfo(STI), RI(*this, STI) {
}

bool ThumbInstrInfo::isMoveInstr(const MachineInstr &MI,
                                 unsigned &SrcReg, unsigned &DstReg,
                                 unsigned& SrcSubIdx, unsigned& DstSubIdx) const {
  SrcSubIdx = DstSubIdx = 0; // No sub-registers.

  unsigned oc = MI.getOpcode();
  switch (oc) {
  default:
    return false;
  // FIXME: Thumb2
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

unsigned ThumbInstrInfo::isLoadFromStackSlot(const MachineInstr *MI,
                                             int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  // FIXME: Thumb2
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

unsigned ThumbInstrInfo::isStoreToStackSlot(const MachineInstr *MI,
                                            int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  // FIXME: Thumb2
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

bool ThumbInstrInfo::copyRegToReg(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator I,
                                  unsigned DestReg, unsigned SrcReg,
                                  const TargetRegisterClass *DestRC,
                                  const TargetRegisterClass *SrcRC) const {
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (I != MBB.end()) DL = I->getDebugLoc();

  // FIXME: Thumb2
  if (DestRC == ARM::GPRRegisterClass) {
    if (SrcRC == ARM::GPRRegisterClass) {
      BuildMI(MBB, I, DL, get(ARM::tMOVhir2hir), DestReg).addReg(SrcReg);
      return true;
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

bool ThumbInstrInfo::
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

void ThumbInstrInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                    unsigned SrcReg, bool isKill, int FI,
                    const TargetRegisterClass *RC) const {
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (I != MBB.end()) DL = I->getDebugLoc();

  assert(RC == ARM::tGPRRegisterClass && "Unknown regclass!");

  // FIXME: Thumb2
  if (RC == ARM::tGPRRegisterClass) {
    BuildMI(MBB, I, DL, get(ARM::tSpill))
      .addReg(SrcReg, getKillRegState(isKill))
      .addFrameIndex(FI).addImm(0);
  }
}

void ThumbInstrInfo::storeRegToAddr(MachineFunction &MF, unsigned SrcReg,
                                    bool isKill,
                                    SmallVectorImpl<MachineOperand> &Addr,
                                    const TargetRegisterClass *RC,
                                   SmallVectorImpl<MachineInstr*> &NewMIs) const{
  DebugLoc DL = DebugLoc::getUnknownLoc();
  unsigned Opc = 0;

  // FIXME: Thumb2. Is GPRRegClass here correct?
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

void ThumbInstrInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     unsigned DestReg, int FI,
                     const TargetRegisterClass *RC) const {
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (I != MBB.end()) DL = I->getDebugLoc();

  // FIXME: Thumb2
  assert(RC == ARM::tGPRRegisterClass && "Unknown regclass!");

  if (RC == ARM::tGPRRegisterClass) {
    BuildMI(MBB, I, DL, get(ARM::tRestore), DestReg)
      .addFrameIndex(FI).addImm(0);
  }
}

void ThumbInstrInfo::
loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                SmallVectorImpl<MachineOperand> &Addr,
                const TargetRegisterClass *RC,
                SmallVectorImpl<MachineInstr*> &NewMIs) const {
  DebugLoc DL = DebugLoc::getUnknownLoc();
  unsigned Opc = 0;

  // FIXME: Thumb2. Is GPRRegClass ok here?
  if (RC == ARM::GPRRegisterClass) {
    Opc = Addr[0].isFI() ? ARM::tRestore : ARM::tLDR;
  }

  MachineInstrBuilder MIB = BuildMI(MF, DL, get(Opc), DestReg);
  for (unsigned i = 0, e = Addr.size(); i != e; ++i)
    MIB.addOperand(Addr[i]);
  NewMIs.push_back(MIB);
  return;
}

bool ThumbInstrInfo::
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

bool ThumbInstrInfo::
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

MachineInstr *ThumbInstrInfo::
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
