//===- X86InstrInfo.cpp - X86 Instruction Information -----------*- C++ -*-===//
//
// This file contains the X86 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "X86InstrInfo.h"
#include "X86.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

#include "X86GenInstrInfo.inc"

X86InstrInfo::X86InstrInfo()
  : TargetInstrInfo(X86Insts, sizeof(X86Insts)/sizeof(X86Insts[0]), 0) {
}


// createNOPinstr - returns the target's implementation of NOP, which is
// usually a pseudo-instruction, implemented by a degenerate version of
// another instruction, e.g. X86: `xchg ax, ax'; SparcV9: `sethi r0, r0, r0'
//
MachineInstr* X86InstrInfo::createNOPinstr() const {
  return BuildMI(X86::XCHGrr16, 2).addReg(X86::AX, MOTy::UseAndDef)
                                  .addReg(X86::AX, MOTy::UseAndDef);
}


/// isNOPinstr - not having a special NOP opcode, we need to know if a given
/// instruction is interpreted as an `official' NOP instr, i.e., there may be
/// more than one way to `do nothing' but only one canonical way to slack off.
//
bool X86InstrInfo::isNOPinstr(const MachineInstr &MI) const {
  // Make sure the instruction is EXACTLY `xchg ax, ax'
  if (MI.getOpcode() == X86::XCHGrr16) {
    const MachineOperand &op0 = MI.getOperand(0), &op1 = MI.getOperand(1);
    if (op0.isMachineRegister() && op0.getMachineRegNum() == X86::AX &&
        op1.isMachineRegister() && op1.getMachineRegNum() == X86::AX) {
      return true;
    }
  }
  // FIXME: there are several NOOP instructions, we should check for them here.
  return false;
}

