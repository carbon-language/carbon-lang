//===-- CodeGen/MInstBuilder.h - Simplify creation of MInstcn's -*- C++ -*-===//
//
// This file exposes a function named BuildMInst that is useful for dramatically
// simplifying how MInstruction's are created.  Instead of using code like this:
//
//   M = new MInstruction(BB, X86::ADDrr32, DestReg);
//   M->addOperand(Arg0Reg, MOperand::Register);
//   M->addOperand(Arg1Reg, MOperand::Register);
//
// we can now use code like this:
//
//   M = BuildMInst(BB, X86::ADDrr8, DestReg).addReg(Arg0Reg).addReg(Arg1Reg);
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MINSTBUILDER_H
#define LLVM_CODEGEN_MINSTBUILDER_H

#include "llvm/CodeGen/MInstruction.h"

struct MInstructionBuilder { 
  MInstruction *MI;

  MInstructionBuilder(MInstruction *mi) : MI(mi) {}

  /// Allow automatic conversion to the machine instruction we are working on.
  ///
  operator MInstruction*() const { return MI; }

  /// addReg - Add a new register operand...
  ///
  MInstructionBuilder &addReg(unsigned RegNo) {
    MI->addOperand(RegNo, MOperand::Register);
    return *this;
  }

  /// addSImm - Add a new sign extended immediate operand...
  ///
  MInstructionBuilder &addSImm(int Val) {
    MI->addOperand(Val, MOperand::SignExtImmediate);
    return *this;
  }

  /// addZImm - Add a new zero extended immediate operand...
  ///
  MInstructionBuilder &addZImm(unsigned Val) {
    MI->addOperand(Val, MOperand::ZeroExtImmediate);
    return *this;
  }

  /// addPCDisp - Add a PC Relative Displacement operand...
  ///
  MInstructionBuilder &addPCDisp(int Disp) {
    MI->addOperand(Disp, MOperand::PCRelativeDisp);
    return *this;
  }
};

/// BuildMInst - Builder interface.  Specify how to create the initial
/// instruction itself.
///
inline MInstructionBuilder BuildMInst(unsigned Opcode, unsigned DestReg = 0) {
  return MInstructionBuilder(new MInstruction(Opcode, DestReg));
}

inline MInstructionBuilder BuildMInst(MBasicBlock *BB, unsigned Opcode,
                               unsigned DestReg = 0) {
  return MInstructionBuilder(new MInstruction(BB, Opcode, DestReg));
}
                                
#endif
