//===-- CodeGen/MachineInstBuilder.h - Simplify creation of MIs -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes a function named BuildMI, which is useful for dramatically
// simplifying how MachineInstr's are created.  It allows use of code like this:
//
//   M = BuildMI(X86::ADDrr8, 2).addReg(argVal1).addReg(argVal2);
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEINSTRBUILDER_H
#define LLVM_CODEGEN_MACHINEINSTRBUILDER_H

#include "llvm/CodeGen/MachineBasicBlock.h"

namespace llvm {

class MachineInstrBuilder {
  MachineInstr *MI;
public:
  MachineInstrBuilder(MachineInstr *mi) : MI(mi) {}

  /// Allow automatic conversion to the machine instruction we are working on.
  ///
  operator MachineInstr*() const { return MI; }
  operator MachineBasicBlock::iterator() const { return MI; }

  /// addReg - Add a new virtual register operand...
  ///
  const MachineInstrBuilder &addReg(
    int RegNo,
    MachineOperand::UseType Ty = MachineOperand::Use) const {
    MI->addRegOperand(RegNo, Ty);
    return *this;
  }

  /// addImm - Add a new immediate operand.
  ///
  const MachineInstrBuilder &addImm(int Val) const {
    MI->addZeroExtImmOperand(Val);
    return *this;
  }

  /// addSImm - Add a new sign extended immediate operand...
  ///
  const MachineInstrBuilder &addSImm(int val) const {
    MI->addSignExtImmOperand(val);
    return *this;
  }

  /// addZImm - Add a new zero extended immediate operand...
  ///
  const MachineInstrBuilder &addZImm(unsigned Val) const {
    MI->addZeroExtImmOperand(Val);
    return *this;
  }

  /// addImm64 - Add a new 64-bit immediate operand...
  ///
  const MachineInstrBuilder &addImm64(uint64_t Val) const {
    MI->addZeroExtImm64Operand(Val);
    return *this;
  }

  const MachineInstrBuilder &addMBB(MachineBasicBlock *MBB) const {
    MI->addMachineBasicBlockOperand(MBB);
    return *this;
  }

  const MachineInstrBuilder &addFrameIndex(unsigned Idx) const {
    MI->addFrameIndexOperand(Idx);
    return *this;
  }

  const MachineInstrBuilder &addConstantPoolIndex(unsigned Idx,
                                                  int Offset = 0) const {
    MI->addConstantPoolIndexOperand(Idx, Offset);
    return *this;
  }

  const MachineInstrBuilder &addJumpTableIndex(unsigned Idx) const {
    MI->addJumpTableIndexOperand(Idx);
    return *this;
  }

  const MachineInstrBuilder &addGlobalAddress(GlobalValue *GV,
                                              int Offset = 0) const {
    MI->addGlobalAddressOperand(GV, Offset);
    return *this;
  }

  const MachineInstrBuilder &addExternalSymbol(const char *FnName) const{
    MI->addExternalSymbolOperand(FnName);
    return *this;
  }
};

/// BuildMI - Builder interface.  Specify how to create the initial instruction
/// itself.  NumOperands is the number of operands to the machine instruction to
/// allow for memory efficient representation of machine instructions.
///
inline MachineInstrBuilder BuildMI(int Opcode, unsigned NumOperands) {
  return MachineInstrBuilder(new MachineInstr(Opcode, NumOperands, true, true));
}

/// BuildMI - This version of the builder sets up the first operand as a
/// destination virtual register.  NumOperands is the number of additional add*
/// calls that are expected, not including the destination register.
///
inline MachineInstrBuilder BuildMI(
  int Opcode, unsigned NumOperands,
  unsigned DestReg,
  MachineOperand::UseType useType = MachineOperand::Def) {
  return MachineInstrBuilder(new MachineInstr(Opcode, NumOperands+1,
                                   true, true)).addReg(DestReg, useType);
}

/// BuildMI - This version of the builder inserts the newly-built
/// instruction before the given position in the given MachineBasicBlock, and
/// sets up the first operand as a destination virtual register.
/// NumOperands is the number of additional add* calls that are expected,
/// not including the destination register.
///
inline MachineInstrBuilder BuildMI(MachineBasicBlock &BB,
                                   MachineBasicBlock::iterator I,
                                   int Opcode, unsigned NumOperands,
                                   unsigned DestReg) {
  MachineInstr *MI = new MachineInstr(Opcode, NumOperands+1, true, true);
  BB.insert(I, MI);
  return MachineInstrBuilder(MI).addReg(DestReg, MachineOperand::Def);
}

/// BuildMI - This version of the builder inserts the newly-built
/// instruction before the given position in the given MachineBasicBlock, and
/// does NOT take a destination register.
///
inline MachineInstrBuilder BuildMI(MachineBasicBlock &BB,
                                   MachineBasicBlock::iterator I,
                                   int Opcode, unsigned NumOperands) {
  MachineInstr *MI = new MachineInstr(Opcode, NumOperands, true, true);
  BB.insert(I, MI);
  return MachineInstrBuilder(MI);
}

/// BuildMI - This version of the builder inserts the newly-built
/// instruction at the end of the given MachineBasicBlock, and does NOT take a
/// destination register.
///
inline MachineInstrBuilder BuildMI(MachineBasicBlock *BB, int Opcode,
                                   unsigned NumOperands) {
  return BuildMI(*BB, BB->end(), Opcode, NumOperands);
}

/// BuildMI - This version of the builder inserts the newly-built
/// instruction at the end of the given MachineBasicBlock, and sets up the first
/// operand as a destination virtual register. NumOperands is the number of
/// additional add* calls that are expected, not including the destination
/// register.
///
inline MachineInstrBuilder BuildMI(MachineBasicBlock *BB, int Opcode,
                                   unsigned NumOperands, unsigned DestReg) {
  return BuildMI(*BB, BB->end(), Opcode, NumOperands, DestReg);
}

} // End llvm namespace

#endif
