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
//   M = BuildMI(X86::ADDrr8).addReg(argVal1).addReg(argVal2);
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

  /// addReg - Add a new register operand...
  ///
  MachineInstrBuilder &addReg(int RegNo) {
    MI->addRegOperand(RegNo);
    return *this;
  }

  MachineInstrBuilder &addReg(Value *V, bool isDef = false, bool isDNU = false){
    MI->addRegOperand(V, isDef, isDNU);
    return *this;
  }

  MachineInstrBuilder &addPCDisp(Value *V) {
    MI->addPCDispOperand(V);
    return *this;
  }

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
/// itself.
///
inline MachineInstrBuilder BuildMI(MachineOpCode Opcode, unsigned NumOperands) {
  return MachineInstrBuilder(new MachineInstr(Opcode, NumOperands, true, true));
}

#if 0
inline MachineInstrBuilder BuildMI(MBasicBlock *BB, MachineOpCode Opcode,
                                   unsigned DestReg = 0) {
  return MachineInstrBuilder(new MachineInstr(BB, Opcode, DestReg));
}
#endif
                                
#endif
