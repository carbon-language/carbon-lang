//===- X86InstrInfo.cpp - X86 Instruction Information -----------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains the X86 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "X86InstrInfo.h"
#include "X86.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "X86GenInstrInfo.inc"
using namespace llvm;

X86InstrInfo::X86InstrInfo()
  : TargetInstrInfo(X86Insts, sizeof(X86Insts)/sizeof(X86Insts[0])) {
}


bool X86InstrInfo::isMoveInstr(const MachineInstr& MI,
                               unsigned& sourceReg,
                               unsigned& destReg) const {
  MachineOpCode oc = MI.getOpcode();
  if (oc == X86::MOV8rr || oc == X86::MOV16rr || oc == X86::MOV32rr ||
      oc == X86::FpMOV) {
      assert(MI.getNumOperands() == 2 &&
             MI.getOperand(0).isRegister() &&
             MI.getOperand(1).isRegister() &&
             "invalid register-register move instruction");
      sourceReg = MI.getOperand(1).getReg();
      destReg = MI.getOperand(0).getReg();
      return true;
  }
  return false;
}

void X86InstrInfo::insertGoto(MachineBasicBlock& MBB,
                              MachineBasicBlock& TMBB) const {
  BuildMI(MBB, MBB.end(), X86::JMP, 1).addMBB(&TMBB);
}

MachineBasicBlock::iterator
X86InstrInfo::reverseBranchCondition(MachineBasicBlock::iterator MI) const {
  unsigned Opcode = MI->getOpcode();
  assert(isBranch(Opcode) && "MachineInstr must be a branch");
  unsigned ROpcode;
  switch (Opcode) {
  default: assert(0 && "Cannot reverse unconditional branches!");
  case X86::JB:  ROpcode = X86::JAE; break;
  case X86::JAE: ROpcode = X86::JB;  break;
  case X86::JE:  ROpcode = X86::JNE; break;
  case X86::JNE: ROpcode = X86::JE;  break;
  case X86::JBE: ROpcode = X86::JA;  break;
  case X86::JA:  ROpcode = X86::JBE; break;
  case X86::JS:  ROpcode = X86::JNS; break;
  case X86::JNS: ROpcode = X86::JS;  break;
  case X86::JL:  ROpcode = X86::JGE; break;
  case X86::JGE: ROpcode = X86::JL;  break;
  case X86::JLE: ROpcode = X86::JG;  break;
  case X86::JG:  ROpcode = X86::JLE; break;
  }
  MachineBasicBlock* MBB = MI->getParent();
  MachineBasicBlock* TMBB = MI->getOperand(0).getMachineBasicBlock();
  return BuildMI(*MBB, MBB->erase(MI), ROpcode, 1).addMBB(TMBB);
}
