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
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "SparcV8GenInstrInfo.inc"

namespace llvm {

SparcV8InstrInfo::SparcV8InstrInfo()
  : TargetInstrInfo(SparcV8Insts,
                    sizeof(SparcV8Insts)/sizeof(SparcV8Insts[0]), 0) {
}

// createNOPinstr - returns the target's implementation of NOP, which is
// usually a pseudo-instruction, implemented by a degenerate version of
// another instruction.
//
MachineInstr* SparcV8InstrInfo::createNOPinstr() const {
  return 0;
}

/// isNOPinstr - not having a special NOP opcode, we need to know if a given
/// instruction is interpreted as an `official' NOP instr, i.e., there may be
/// more than one way to `do nothing' but only one canonical way to slack off.
//
bool SparcV8InstrInfo::isNOPinstr(const MachineInstr &MI) const {
  return false;
}

} // end namespace llvm

