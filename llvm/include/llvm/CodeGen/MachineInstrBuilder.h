//===-- CodeGen/MachineInstBuilder.h - Simplify creation of MIs -*- C++ -*-===//
//
// This file exposes a function named BuildMI, which is useful for dramatically
// simplifying how MachineInstr's are created.  Instead of using code like this:
//
//   M = new MachineInstr(X86::ADDrr32);
//   M->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister, argVal1);
//   M->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister, argVal2);
//
// we can now use code like this:
//
//   M = BuildMI(X86::ADDrr8, 2).addReg(argVal1).addReg(argVal2);
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEINSTRBUILDER_H
#define LLVM_CODEGEN_MACHINEINSTRBUILDER_H

#include "llvm/CodeGen/MachineInstr.h"

struct MachineInstrBuilder { 
  MachineInstr *MI;

  MachineInstrBuilder(MachineInstr *mi) : MI(mi) {}

  /// Allow automatic conversion to the machine instruction we are working on.
  ///
  operator MachineInstr*() const { return MI; }

  /// addReg - Add a new virtual register operand...
  ///
  MachineInstrBuilder &addReg(int RegNo, bool isDef = false) {
    MI->addRegOperand(RegNo, isDef);
    return *this;
  }

  /// addReg - Add an LLVM value that is to be used as a register...x
  ///
  MachineInstrBuilder &addReg(Value *V, bool isDef = false, bool isDNU = false){
    MI->addRegOperand(V, isDef, isDNU);
    return *this;
  }

  /// addPCDisp - Add an LLVM value to be treated as a PC relative
  /// displacement...
  ///
  MachineInstrBuilder &addPCDisp(Value *V) {
    MI->addPCDispOperand(V);
    return *this;
  }

  /// addMReg - Add a machine register operand...
  ///
  MachineInstrBuilder &addMReg(int Reg, bool isDef=false) {
    MI->addMachineRegOperand(Reg, isDef);
    return *this;
  }

  /// addSImm - Add a new sign extended immediate operand...
  ///
  MachineInstrBuilder &addSImm(int64_t val) {
    MI->addSignExtImmOperand(val);
    return *this;
  }

  /// addZImm - Add a new zero extended immediate operand...
  ///
  MachineInstrBuilder &addZImm(int64_t Val) {
    MI->addZeroExtImmOperand(Val);
    return *this;
  }
};

/// BuildMI - Builder interface.  Specify how to create the initial instruction
/// itself.  NumOperands is the number of operands to the machine instruction to
/// allow for memory efficient representation of machine instructions.
///
inline MachineInstrBuilder BuildMI(MachineOpCode Opcode, unsigned NumOperands) {
  return MachineInstrBuilder(new MachineInstr(Opcode, NumOperands, true, true));
}

/// BuildMI - This version of the builder inserts the built MachineInstr into
/// the specified MachineBasicBlock.
///
inline MachineInstrBuilder BuildMI(MachineBasicBlock *BB, MachineOpCode Opcode,
                                   unsigned NumOperands) {
  return MachineInstrBuilder(new MachineInstr(BB, Opcode, NumOperands));
}

/// BuildMI - This version of the builder inserts the built MachineInstr into
/// the specified MachineBasicBlock, and also sets up the first "operand" as a
/// destination virtual register.  NumOperands is the number of additional add*
/// calls that are expected, it does not include the destination register.
///
inline MachineInstrBuilder BuildMI(MachineBasicBlock *BB, MachineOpCode Opcode,
                                   unsigned NumOperands, unsigned DestReg) {
  return MachineInstrBuilder(new MachineInstr(BB, Opcode,
                                              NumOperands+1)).addReg(DestReg,
                                                                     true);
}

#endif
