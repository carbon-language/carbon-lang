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

PowerPCInstrInfo::PowerPCInstrInfo()
  : TargetInstrInfo(PowerPCInsts, sizeof(PowerPCInsts)/sizeof(PowerPCInsts[0]))
{ }

bool PowerPCInstrInfo::isMoveInstr(const MachineInstr& MI,
                                   unsigned& sourceReg,
                                   unsigned& destReg) const {
  MachineOpCode oc = MI.getOpcode();
  if (oc == PPC32::OR) {
    assert(MI.getNumOperands() == 3 &&
           MI.getOperand(0).isRegister() &&
           MI.getOperand(1).isRegister() &&
           MI.getOperand(2).isRegister() &&
           "invalid register-register int move instruction");
    if (MI.getOperand(1).getReg() == MI.getOperand(2).getReg()) {
      sourceReg = MI.getOperand(1).getReg();
      destReg = MI.getOperand(0).getReg();
      return true;
    }
  } else if (oc == PPC32::FMR) {
    assert(MI.getNumOperands() == 2 &&
           MI.getOperand(0).isRegister() &&
           MI.getOperand(1).isRegister() &&
           "invalid register-register fp move instruction");
    sourceReg = MI.getOperand(1).getReg();
    destReg = MI.getOperand(0).getReg();
    return true;
  }
  return false;
}
