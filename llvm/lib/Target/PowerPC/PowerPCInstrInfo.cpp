//===- PowerPCInstrInfo.cpp - PowerPC Instruction Information ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains the PowerPC implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "PowerPCInstrInfo.h"
#include "PowerPC.h"
#include "PowerPCGenInstrInfo.inc"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include <iostream>
using namespace llvm;

PowerPCInstrInfo::PowerPCInstrInfo(bool is64b)
  : TargetInstrInfo(PowerPCInsts, sizeof(PowerPCInsts)/sizeof(PowerPCInsts[0])),
    RI(is64b),
    is64bit(is64b)
{ }

bool PowerPCInstrInfo::isMoveInstr(const MachineInstr& MI,
                                   unsigned& sourceReg,
                                   unsigned& destReg) const {
  MachineOpCode oc = MI.getOpcode();
  if (oc == PPC::OR) {                      // or r1, r2, r2
    assert(MI.getNumOperands() == 3 &&
           MI.getOperand(0).isRegister() &&
           MI.getOperand(1).isRegister() &&
           MI.getOperand(2).isRegister() &&
           "invalid PPC OR instruction!");
    if (MI.getOperand(1).getReg() == MI.getOperand(2).getReg()) {
      sourceReg = MI.getOperand(1).getReg();
      destReg = MI.getOperand(0).getReg();
      return true;
    }
  } else if (oc == PPC::ADDI) {             // addi r1, r2, 0
    assert(MI.getNumOperands() == 3 &&
           MI.getOperand(0).isRegister() &&
           MI.getOperand(2).isImmediate() &&
           "invalid PPC ADDI instruction!");
    if (MI.getOperand(1).isRegister() && MI.getOperand(2).getImmedValue()==0) {
      sourceReg = MI.getOperand(1).getReg();
      destReg = MI.getOperand(0).getReg();
      return true;
    }
  } else if (oc == PPC::FMR) {              // fmr r1, r2
    assert(MI.getNumOperands() == 2 &&
           MI.getOperand(0).isRegister() &&
           MI.getOperand(1).isRegister() &&
           "invalid PPC FMR instruction");
    sourceReg = MI.getOperand(1).getReg();
    destReg = MI.getOperand(0).getReg();
    return true;
  }
  return false;
}
