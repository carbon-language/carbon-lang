//===- MSP430InstrInfo.cpp - MSP430 Instruction Information ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MSP430 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "MSP430.h"
#include "MSP430InstrInfo.h"
#include "MSP430TargetMachine.h"
#include "MSP430GenInstrInfo.inc"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"


using namespace llvm;

MSP430InstrInfo::MSP430InstrInfo(MSP430TargetMachine &tm)
  : TargetInstrInfoImpl(MSP430Insts, array_lengthof(MSP430Insts)),
    RI(*this), TM(tm) {}

bool MSP430InstrInfo::copyRegToReg(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator I,
                                   unsigned DestReg, unsigned SrcReg,
                                   const TargetRegisterClass *DestRC,
                                   const TargetRegisterClass *SrcRC) const {
  if (DestRC != SrcRC) {
    // Not yet supported!
    return false;
  }

  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (I != MBB.end()) DL = I->getDebugLoc();

  BuildMI(MBB, I, DL, get(MSP430::MOV16rr), DestReg).addReg(SrcReg);
  return true;
}

bool
MSP430InstrInfo::isMoveInstr(const MachineInstr& MI,
                             unsigned &SrcReg, unsigned &DstReg,
                             unsigned &SrcSubIdx, unsigned &DstSubIdx) const {
  SrcSubIdx = DstSubIdx = 0; // No sub-registers yet.

  switch (MI.getOpcode()) {
  default:
    return false;
  case MSP430::MOV16rr:
    assert(MI.getNumOperands() >= 2 &&
           MI.getOperand(0).isReg() &&
           MI.getOperand(1).isReg() &&
           "invalid register-register move instruction");
    SrcReg = MI.getOperand(1).getReg();
    DstReg = MI.getOperand(0).getReg();
    return true;
  }
}
