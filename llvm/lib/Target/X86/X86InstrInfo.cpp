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
#include "X86GenInstrInfo.inc"
#include "X86InstrBuilder.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

X86InstrInfo::X86InstrInfo(X86TargetMachine &tm)
  : TargetInstrInfo(X86Insts, sizeof(X86Insts)/sizeof(X86Insts[0])),
    TM(tm) {
}


bool X86InstrInfo::isMoveInstr(const MachineInstr& MI,
                               unsigned& sourceReg,
                               unsigned& destReg) const {
  MachineOpCode oc = MI.getOpcode();
  if (oc == X86::MOV8rr || oc == X86::MOV16rr || oc == X86::MOV32rr ||
      oc == X86::MOV16to16_ || oc == X86::MOV32to32_ ||
      oc == X86::FpMOV  || oc == X86::MOVSSrr || oc == X86::MOVSDrr ||
      oc == X86::FsMOVAPSrr || oc == X86::FsMOVAPDrr ||
      oc == X86::MOVAPSrr || oc == X86::MOVAPDrr ||
      oc == X86::MOVSS2PSrr || oc == X86::MOVSD2PDrr ||
      oc == X86::MOVPS2SSrr || oc == X86::MOVPD2SDrr ||
      oc == X86::MOVDI2PDIrr || oc == X86::MOVQI2PQIrr ||
      oc == X86::MOVPDI2DIrr) {
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

unsigned X86InstrInfo::isLoadFromStackSlot(MachineInstr *MI, 
                                           int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case X86::MOV8rm:
  case X86::MOV16rm:
  case X86::MOV16_rm:
  case X86::MOV32rm:
  case X86::MOV32_rm:
  case X86::FpLD64m:
  case X86::MOVSSrm:
  case X86::MOVSDrm:
  case X86::MOVAPSrm:
  case X86::MOVAPDrm:
    if (MI->getOperand(1).isFrameIndex() && MI->getOperand(2).isImmediate() &&
        MI->getOperand(3).isRegister() && MI->getOperand(4).isImmediate() &&
        MI->getOperand(2).getImmedValue() == 1 &&
        MI->getOperand(3).getReg() == 0 &&
        MI->getOperand(4).getImmedValue() == 0) {
      FrameIndex = MI->getOperand(1).getFrameIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

unsigned X86InstrInfo::isStoreToStackSlot(MachineInstr *MI,
                                          int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case X86::MOV8mr:
  case X86::MOV16mr:
  case X86::MOV16_mr:
  case X86::MOV32mr:
  case X86::MOV32_mr:
  case X86::FpSTP64m:
  case X86::MOVSSmr:
  case X86::MOVSDmr:
  case X86::MOVAPSmr:
  case X86::MOVAPDmr:
    if (MI->getOperand(0).isFrameIndex() && MI->getOperand(1).isImmediate() &&
        MI->getOperand(2).isRegister() && MI->getOperand(3).isImmediate() &&
        MI->getOperand(1).getImmedValue() == 1 &&
        MI->getOperand(2).getReg() == 0 &&
        MI->getOperand(3).getImmedValue() == 0) {
      FrameIndex = MI->getOperand(0).getFrameIndex();
      return MI->getOperand(4).getReg();
    }
    break;
  }
  return 0;
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

  switch (MI->getOpcode()) {
  default: break;
  case X86::SHUFPSrri: {
    assert(MI->getNumOperands() == 4 && "Unknown shufps instruction!");
    const X86Subtarget *Subtarget = &TM.getSubtarget<X86Subtarget>();
    unsigned A = MI->getOperand(0).getReg();
    unsigned B = MI->getOperand(1).getReg();
    unsigned C = MI->getOperand(2).getReg();
    unsigned M = MI->getOperand(3).getImmedValue();
    if (!Subtarget->hasSSE2() || B != C) return 0;
    return BuildMI(X86::PSHUFDri, 2, A).addReg(B).addImm(M);
  }
  }

  // FIXME: None of these instructions are promotable to LEAs without
  // additional information.  In particular, LEA doesn't set the flags that
  // add and inc do.  :(
  return 0;

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
  case X86::ADD32ri8:
    assert(MI->getNumOperands() == 3 && "Unknown add instruction!");
    if (MI->getOperand(2).isImmediate())
      return addRegOffset(BuildMI(X86::LEA32r, 5, Dest), Src,
                          MI->getOperand(2).getImmedValue());
    return 0;
  case X86::ADD16ri:
  case X86::ADD16ri8:
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

/// commuteInstruction - We have a few instructions that must be hacked on to
/// commute them.
///
MachineInstr *X86InstrInfo::commuteInstruction(MachineInstr *MI) const {
  switch (MI->getOpcode()) {
  case X86::SHRD16rri8: // A = SHRD16rri8 B, C, I -> A = SHLD16rri8 C, B, (16-I)
  case X86::SHLD16rri8: // A = SHLD16rri8 B, C, I -> A = SHRD16rri8 C, B, (16-I)
  case X86::SHRD32rri8: // A = SHRD32rri8 B, C, I -> A = SHLD32rri8 C, B, (32-I)
  case X86::SHLD32rri8:{// A = SHLD32rri8 B, C, I -> A = SHRD32rri8 C, B, (32-I)
    unsigned Opc;
    unsigned Size;
    switch (MI->getOpcode()) {
    default: assert(0 && "Unreachable!");
    case X86::SHRD16rri8: Size = 16; Opc = X86::SHLD16rri8; break;
    case X86::SHLD16rri8: Size = 16; Opc = X86::SHRD16rri8; break;
    case X86::SHRD32rri8: Size = 32; Opc = X86::SHLD32rri8; break;
    case X86::SHLD32rri8: Size = 32; Opc = X86::SHRD32rri8; break;
    }
    unsigned Amt = MI->getOperand(3).getImmedValue();
    unsigned A = MI->getOperand(0).getReg();
    unsigned B = MI->getOperand(1).getReg();
    unsigned C = MI->getOperand(2).getReg();
    return BuildMI(Opc, 3, A).addReg(C).addReg(B).addImm(Size-Amt);
  }
  default:
    return TargetInstrInfo::commuteInstruction(MI);
  }
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

