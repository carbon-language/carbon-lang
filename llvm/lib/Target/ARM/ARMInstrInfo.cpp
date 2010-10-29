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
#include "llvm/MC/MCAsmInfo.h"
using namespace llvm;

ARMInstrInfo::ARMInstrInfo(const ARMSubtarget &STI)
  : ARMBaseInstrInfo(STI), RI(*this, STI) {
}

unsigned ARMInstrInfo::getUnindexedOpcode(unsigned Opc) const {
  switch (Opc) {
  default: break;
  case ARM::LDR_PRE:
  case ARM::LDR_POST:
    return ARM::LDRi12;
  case ARM::LDRH_PRE:
  case ARM::LDRH_POST:
    return ARM::LDRH;
  case ARM::LDRB_PRE:
  case ARM::LDRB_POST:
    return ARM::LDRBi12;
  case ARM::LDRSH_PRE:
  case ARM::LDRSH_POST:
    return ARM::LDRSH;
  case ARM::LDRSB_PRE:
  case ARM::LDRSB_POST:
    return ARM::LDRSB;
  case ARM::STR_PRE:
  case ARM::STR_POST:
    return ARM::STRi12;
  case ARM::STRH_PRE:
  case ARM::STRH_POST:
    return ARM::STRH;
  case ARM::STRB_PRE:
  case ARM::STRB_POST:
    return ARM::STRBi12;
  }

  return 0;
}

void ARMInstrInfo::
reMaterialize(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
              unsigned DestReg, unsigned SubIdx, const MachineInstr *Orig,
              const TargetRegisterInfo &TRI) const {
  DebugLoc dl = Orig->getDebugLoc();
  unsigned Opcode = Orig->getOpcode();
  switch (Opcode) {
  default:
    break;
  case ARM::MOVi2pieces: {
    RI.emitLoadConstPool(MBB, I, dl,
                         DestReg, SubIdx,
                         Orig->getOperand(1).getImm(),
                         ARMCC::AL, 0); // Pre-if-conversion, so default pred.
    MachineInstr *NewMI = prior(I);
    NewMI->getOperand(0).setSubReg(SubIdx);
    return;
  }
  }

  return ARMBaseInstrInfo::reMaterialize(MBB, I, DestReg, SubIdx, Orig, TRI);
}

