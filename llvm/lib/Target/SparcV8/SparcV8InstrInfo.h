//===- SparcV8InstrInfo.h - SparcV8 Instruction Information -----*- C++ -*-===//
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

#ifndef SPARCV8INSTRUCTIONINFO_H
#define SPARCV8INSTRUCTIONINFO_H

#include "llvm/Target/TargetInstrInfo.h"
#include "SparcV8RegisterInfo.h"

namespace llvm {

class SparcV8InstrInfo : public TargetInstrInfo {
  const SparcV8RegisterInfo RI;
public:
  SparcV8InstrInfo();

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  virtual const MRegisterInfo &getRegisterInfo() const { return RI; }

  /// createNOPinstr - returns the target's implementation of NOP, which is
  /// usually a pseudo-instruction, implemented by a degenerate version of
  /// another instruction.
  ///
  MachineInstr* createNOPinstr() const;

  /// isNOPinstr - not having a special NOP opcode, we need to know if a given
  /// instruction is interpreted as an `official' NOP instr, i.e., there may be
  /// more than one way to `do nothing' but only one canonical way to slack off.
  ///
  bool isNOPinstr(const MachineInstr &MI) const;
};

}

#endif
