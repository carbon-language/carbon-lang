//===- SparcV8InstrInfo.cpp - SparcV8 Instruction Information ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains the SparcV8 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "SparcV8InstrInfo.h"
#include "SparcV8.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "SparcV8GenInstrInfo.inc"
using namespace llvm;

SparcV8InstrInfo::SparcV8InstrInfo()
  : TargetInstrInfo(SparcV8Insts, sizeof(SparcV8Insts)/sizeof(SparcV8Insts[0])){
}

/// Return true if the instruction is a register to register move and
/// leave the source and dest operands in the passed parameters.
///
bool SparcV8InstrInfo::isMoveInstr(const MachineInstr &MI,
                                   unsigned &SrcReg, unsigned &DstReg) const {
  if (MI.getOpcode() == V8::ORrr) {
    if (MI.getOperand(1).getReg() == V8::G0) {  // X = or G0, Y -> X = Y
      DstReg = MI.getOperand(0).getReg();
      SrcReg = MI.getOperand(2).getReg();
      return true;
    }
  } else if (MI.getOpcode() == V8::FMOVS || MI.getOpcode() == V8::FpMOVD) {
    SrcReg = MI.getOperand(1).getReg();
    DstReg = MI.getOperand(0).getReg();
    return true;
  }
  return false;
}
