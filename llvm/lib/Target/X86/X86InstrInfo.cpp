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
#include "X86InstrBuilder.h"
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

/// convertToThreeAddress - This method must be implemented by targets that
/// set the M_CONVERTIBLE_TO_3_ADDR flag.  When this flag is set, the target
/// may be able to convert a two-address instruction into a true
/// three-address instruction on demand.  This allows the X86 target (for
/// example) to convert ADD and SHL instructions into LEA instructions if they
/// would require register copies due to two-addressness.
///
/// This method returns a null pointer if the transformation cannot be
/// performed, otherwise it returns the new instruction.
///
MachineInstr *X86InstrInfo::convertToThreeAddress(MachineInstr *MI) const {
  // All instructions input are two-addr instructions.  Get the known operands.
  unsigned Dest = MI->getOperand(0).getReg();
  unsigned Src = MI->getOperand(1).getReg();

  // FIXME: 16-bit LEA's are really slow on Athlons, but not bad on P4's.  When
  // we have subtarget support, enable the 16-bit LEA generation here.
  bool DisableLEA16 = true;

  switch (MI->getOpcode()) {
  case X86::INC32r:
    assert(MI->getNumOperands() == 2 && "Unknown inc instruction!");
    return addRegOffset(BuildMI(X86::LEA32r, 5, Dest), Src, 1);
  case X86::INC16r:
    if (DisableLEA16) return 0;
    assert(MI->getNumOperands() == 2 && "Unknown inc instruction!");
    return addRegOffset(BuildMI(X86::LEA16r, 5, Dest), Src, 1);
  case X86::DEC32r:
    assert(MI->getNumOperands() == 2 && "Unknown dec instruction!");
    return addRegOffset(BuildMI(X86::LEA32r, 5, Dest), Src, -1);
  case X86::DEC16r:
    if (DisableLEA16) return 0;
    assert(MI->getNumOperands() == 2 && "Unknown dec instruction!");
    return addRegOffset(BuildMI(X86::LEA16r, 5, Dest), Src, -1);
  case X86::ADD32rr:
    assert(MI->getNumOperands() == 3 && "Unknown add instruction!");
    return addRegReg(BuildMI(X86::LEA32r, 5, Dest), Src,
                     MI->getOperand(2).getReg());
  case X86::ADD16rr:
    if (DisableLEA16) return 0;
    assert(MI->getNumOperands() == 3 && "Unknown add instruction!");
    return addRegReg(BuildMI(X86::LEA16r, 5, Dest), Src,
                     MI->getOperand(2).getReg());
  case X86::ADD32ri:
    assert(MI->getNumOperands() == 3 && "Unknown add instruction!");
    if (MI->getOperand(2).isImmediate())
      return addRegOffset(BuildMI(X86::LEA32r, 5, Dest), Src,
                          MI->getOperand(2).getImmedValue());
    return 0;
  case X86::ADD16ri:
    if (DisableLEA16) return 0;
    assert(MI->getNumOperands() == 3 && "Unknown add instruction!");
    if (MI->getOperand(2).isImmediate())
      return addRegOffset(BuildMI(X86::LEA16r, 5, Dest), Src,
                          MI->getOperand(2).getImmedValue());
    break;

  case X86::SHL16ri:
    if (DisableLEA16) return 0;
  case X86::SHL32ri:
    assert(MI->getNumOperands() == 3 && MI->getOperand(2).isImmediate() &&
           "Unknown shl instruction!");
    unsigned ShAmt = MI->getOperand(2).getImmedValue();
    if (ShAmt == 1 || ShAmt == 2 || ShAmt == 3) {
      X86AddressMode AM;
      AM.Scale = 1 << ShAmt;
      AM.IndexReg = Src;
      unsigned Opc = MI->getOpcode() == X86::SHL32ri ? X86::LEA32r :X86::LEA16r;
      return addFullAddress(BuildMI(Opc, 5, Dest), AM);
    }
    break;
  }

  return 0;
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
  case X86::JP:  ROpcode = X86::JNP; break;
  case X86::JNP: ROpcode = X86::JP;  break;
  case X86::JL:  ROpcode = X86::JGE; break;
  case X86::JGE: ROpcode = X86::JL;  break;
  case X86::JLE: ROpcode = X86::JG;  break;
  case X86::JG:  ROpcode = X86::JLE; break;
  }
  MachineBasicBlock* MBB = MI->getParent();
  MachineBasicBlock* TMBB = MI->getOperand(0).getMachineBasicBlock();
  return BuildMI(*MBB, MBB->erase(MI), ROpcode, 1).addMBB(TMBB);
}

