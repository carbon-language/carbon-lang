//===- AlphaInstrInfo.cpp - Alpha Instruction Information -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Alpha implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "Alpha.h"
#include "AlphaInstrInfo.h"
#include "AlphaGenInstrInfo.inc"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include <iostream>
using namespace llvm;

AlphaInstrInfo::AlphaInstrInfo()
  : TargetInstrInfo(AlphaInsts, sizeof(AlphaInsts)/sizeof(AlphaInsts[0])) { }


bool AlphaInstrInfo::isMoveInstr(const MachineInstr& MI,
                                 unsigned& sourceReg,
                                 unsigned& destReg) const {
  MachineOpCode oc = MI.getOpcode();
  if (oc == Alpha::BIS || oc == Alpha::CPYSS || oc == Alpha::CPYST) {
    // or r1, r2, r2 
    // cpys(s|t) r1 r2 r2
    assert(MI.getNumOperands() == 3 &&
           MI.getOperand(0).isRegister() &&
           MI.getOperand(1).isRegister() &&
           MI.getOperand(2).isRegister() &&
           "invalid Alpha BIS instruction!");
    if (MI.getOperand(1).getReg() == MI.getOperand(2).getReg()) {
      sourceReg = MI.getOperand(1).getReg();
      destReg = MI.getOperand(0).getReg();
      return true;
    }
  }
  return false;
}
