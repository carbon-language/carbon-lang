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

unsigned Thumb2InstrInfo::getUnindexedOpcode(unsigned Opc) const {
  // FIXME
  return 0;
}

unsigned Thumb2InstrInfo::getOpcode(ARMII::Op Op) const {
  switch (Op) {
  case ARMII::ADDri: return ARM::t2ADDri;
  case ARMII::ADDrs: return ARM::t2ADDrs;
  case ARMII::ADDrr: return ARM::t2ADDrr;
  case ARMII::MOVr: return ARM::t2MOVr;
  case ARMII::SUBri: return ARM::t2SUBri;
  case ARMII::SUBrs: return ARM::t2SUBrs;
  case ARMII::SUBrr: return ARM::t2SUBrr;
  default:
    break;
  }

  return 0;
}

bool
Thumb2InstrInfo::BlockHasNoFallThrough(const MachineBasicBlock &MBB) const {
  if (MBB.empty()) return false;

  switch (MBB.back().getOpcode()) {
  case ARM::t2LDM_RET:
  case ARM::t2B:        // Uncond branch.
  case ARM::t2BR_JT:    // Jumptable branch.
  case ARM::tBR_JTr:    // Jumptable branch (16-bit version).
  case ARM::tBX_RET:
  case ARM::tBX_RET_vararg:
  case ARM::tPOP_RET:
  case ARM::tB:
    return true;
  default:
    break;
  }

  return false;
}

bool
Thumb2InstrInfo::copyRegToReg(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I,
                              unsigned DestReg, unsigned SrcReg,
                              const TargetRegisterClass *DestRC,
                              const TargetRegisterClass *SrcRC) const {
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (I != MBB.end()) DL = I->getDebugLoc();

  if (DestRC == ARM::GPRRegisterClass &&
      SrcRC == ARM::GPRRegisterClass) {
    AddDefaultCC(AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::t2MOVr),
                                        DestReg).addReg(SrcReg)));
    return true;
  } else if (DestRC == ARM::GPRRegisterClass &&
      SrcRC == ARM::tGPRRegisterClass) {
    BuildMI(MBB, I, DL, get(ARM::tMOVtgpr2gpr), DestReg).addReg(SrcReg);
    return true;
  } else if (DestRC == ARM::tGPRRegisterClass &&
             SrcRC == ARM::GPRRegisterClass) {
    BuildMI(MBB, I, DL, get(ARM::tMOVgpr2tgpr), DestReg).addReg(SrcReg);
    return true;
  }

  // Handle SPR, DPR, and QPR copies.
  return ARMBaseInstrInfo::copyRegToReg(MBB, I, DestReg, SrcReg, DestRC, SrcRC);
}

void Thumb2InstrInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                    unsigned SrcReg, bool isKill, int FI,
                    const TargetRegisterClass *RC) const {
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (I != MBB.end()) DL = I->getDebugLoc();

  if (RC == ARM::GPRRegisterClass) {
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::t2STRi12))
                   .addReg(SrcReg, getKillRegState(isKill))
                   .addFrameIndex(FI).addImm(0));
    return;
  }

  ARMBaseInstrInfo::storeRegToStackSlot(MBB, I, SrcReg, isKill, FI, RC);
}

void Thumb2InstrInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     unsigned DestReg, int FI,
                     const TargetRegisterClass *RC) const {
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (I != MBB.end()) DL = I->getDebugLoc();

  if (RC == ARM::GPRRegisterClass) {
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::t2LDRi12), DestReg)
                   .addFrameIndex(FI).addImm(0));
    return;
  }

  ARMBaseInstrInfo::loadRegFromStackSlot(MBB, I, DestReg, FI, RC);
}
