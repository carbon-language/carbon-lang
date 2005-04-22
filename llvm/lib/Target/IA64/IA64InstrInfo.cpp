//===- IA64InstrInfo.cpp - IA64 Instruction Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the IA64 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "IA64InstrInfo.h"
#include "IA64.h"
#include "IA64InstrBuilder.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "IA64GenInstrInfo.inc"
using namespace llvm;

IA64InstrInfo::IA64InstrInfo()
  : TargetInstrInfo(IA64Insts, sizeof(IA64Insts)/sizeof(IA64Insts[0])) {
}


bool IA64InstrInfo::isMoveInstr(const MachineInstr& MI,
                               unsigned& sourceReg,
                               unsigned& destReg) const {
  MachineOpCode oc = MI.getOpcode();
  if (oc == IA64::MOV || oc == IA64::FMOV) {
     assert(MI.getNumOperands() == 2 &&
             /* MI.getOperand(0).isRegister() &&
             MI.getOperand(1).isRegister() && */
             "invalid register-register move instruction");
     if( MI.getOperand(0).isRegister() &&
         MI.getOperand(1).isRegister() ) {
       // if both operands of the MOV/FMOV are registers, then
       // yes, this is a move instruction
       sourceReg = MI.getOperand(1).getReg();
       destReg = MI.getOperand(0).getReg();
       return true;
     }
  }
  return false; // we don't consider e.g. %regN = MOV <FrameIndex #x> a
                // move instruction
}

