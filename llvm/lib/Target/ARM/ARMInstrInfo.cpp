//===- ARMInstrInfo.cpp - ARM Instruction Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARMInstrInfo.h"
#include "ARM.h"
#include "ARMAddressingModes.h"
#include "ARMGenInstrInfo.inc"
#include "ARMMachineFunctionInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

ARMInstrInfo::ARMInstrInfo(const ARMSubtarget &STI)
  : ARMBaseInstrInfo(STI), RI(*this, STI) {
}

unsigned ARMInstrInfo::
getUnindexedOpcode(unsigned Opc) const {
  switch (Opc) {
  default: break;
  case ARM::LDR_PRE:
  case ARM::LDR_POST:
    return ARM::LDR;
  case ARM::LDRH_PRE:
  case ARM::LDRH_POST:
    return ARM::LDRH;
  case ARM::LDRB_PRE:
  case ARM::LDRB_POST:
    return ARM::LDRB;
  case ARM::LDRSH_PRE:
  case ARM::LDRSH_POST:
    return ARM::LDRSH;
  case ARM::LDRSB_PRE:
  case ARM::LDRSB_POST:
    return ARM::LDRSB;
  case ARM::STR_PRE:
  case ARM::STR_POST:
    return ARM::STR;
  case ARM::STRH_PRE:
  case ARM::STRH_POST:
    return ARM::STRH;
  case ARM::STRB_PRE:
  case ARM::STRB_POST:
    return ARM::STRB;
  }

  return 0;
}

unsigned ARMInstrInfo::
getOpcode(ARMII::Op Op) const {
  switch (Op) {
  case ARMII::ADDri: return ARM::ADDri;
  case ARMII::ADDrs: return ARM::ADDrs;
  case ARMII::ADDrr: return ARM::ADDrr;
  case ARMII::B: return ARM::B;
  case ARMII::Bcc: return ARM::Bcc;
  case ARMII::BR_JTr: return ARM::BR_JTr;
  case ARMII::BR_JTm: return ARM::BR_JTm;
  case ARMII::BR_JTadd: return ARM::BR_JTadd;
  case ARMII::BX_RET: return ARM::BX_RET;
  case ARMII::FCPYS: return ARM::FCPYS;
  case ARMII::FCPYD: return ARM::FCPYD;
  case ARMII::FLDD: return ARM::FLDD;
  case ARMII::FLDS: return ARM::FLDS;
  case ARMII::FSTD: return ARM::FSTD;
  case ARMII::FSTS: return ARM::FSTS;
  case ARMII::LDR: return ARM::LDR;
  case ARMII::MOVr: return ARM::MOVr;
  case ARMII::STR: return ARM::STR;
  case ARMII::SUBri: return ARM::SUBri;
  case ARMII::SUBrs: return ARM::SUBrs;
  case ARMII::SUBrr: return ARM::SUBrr;
  case ARMII::VMOVD: return ARM::VMOVD;
  case ARMII::VMOVQ: return ARM::VMOVQ;
  default:
    break;
  }

  return 0;
}

bool ARMInstrInfo::
BlockHasNoFallThrough(const MachineBasicBlock &MBB) const {
  if (MBB.empty()) return false;

  switch (MBB.back().getOpcode()) {
  case ARM::BX_RET:   // Return.
  case ARM::LDM_RET:
  case ARM::B:
  case ARM::BR_JTr:   // Jumptable branch.
  case ARM::BR_JTm:   // Jumptable branch through mem.
  case ARM::BR_JTadd: // Jumptable branch add to pc.
    return true;
  default:
    break;
  }

  return false;
}

void ARMInstrInfo::
reMaterialize(MachineBasicBlock &MBB,
              MachineBasicBlock::iterator I,
              unsigned DestReg, unsigned SubIdx,
              const MachineInstr *Orig) const {
  DebugLoc dl = Orig->getDebugLoc();
  if (Orig->getOpcode() == ARM::MOVi2pieces) {
    RI.emitLoadConstPool(MBB, I, dl,
                         DestReg, SubIdx,
                         Orig->getOperand(1).getImm(),
                         (ARMCC::CondCodes)Orig->getOperand(2).getImm(),
                         Orig->getOperand(3).getReg());
    return;
  }

  MachineInstr *MI = MBB.getParent()->CloneMachineInstr(Orig);
  MI->getOperand(0).setReg(DestReg);
  MBB.insert(I, MI);
}
