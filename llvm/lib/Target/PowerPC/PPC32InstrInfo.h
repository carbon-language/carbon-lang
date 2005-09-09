//===- PPC32InstrInfo.h - PowerPC32 Instruction Information -----*- C++ -*-===//
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

#ifndef POWERPC32_INSTRUCTIONINFO_H
#define POWERPC32_INSTRUCTIONINFO_H

#include "PowerPCInstrInfo.h"
#include "PPC32RegisterInfo.h"

namespace llvm {

class PPC32InstrInfo : public TargetInstrInfo {
  const PPC32RegisterInfo RI;
public:
  PPC32InstrInfo();

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  virtual const MRegisterInfo &getRegisterInfo() const { return RI; }

  //
  // Return true if the instruction is a register to register move and
  // leave the source and dest operands in the passed parameters.
  //
  virtual bool isMoveInstr(const MachineInstr& MI,
                           unsigned& sourceReg,
                           unsigned& destReg) const;

  // commuteInstruction - We can commute rlwimi instructions, but only if the
  // rotate amt is zero.  We also have to munge the immediates a bit.
  virtual MachineInstr *commuteInstruction(MachineInstr *MI) const;
  
  static unsigned invertPPCBranchOpcode(unsigned Opcode) {
    switch (Opcode) {
    default: assert(0 && "Unknown PPC branch opcode!");
    case PPC::BEQ: return PPC::BNE;
    case PPC::BNE: return PPC::BEQ;
    case PPC::BLT: return PPC::BGE;
    case PPC::BGE: return PPC::BLT;
    case PPC::BGT: return PPC::BLE;
    case PPC::BLE: return PPC::BGT;
    }
  }
};

}

#endif
