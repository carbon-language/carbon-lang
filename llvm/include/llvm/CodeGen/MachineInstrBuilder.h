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

namespace llvm {

class MachineInstrBuilder {
  MachineInstr *MI;
public:
  MachineInstrBuilder(MachineInstr *mi) : MI(mi) {}

  /// Allow automatic conversion to the machine instruction we are working on.
  ///
  operator MachineInstr*() const { return MI; }

  /// addReg - Add a new virtual register operand...
  ///
  const MachineInstrBuilder &addReg(int RegNo,
                                    MOTy::UseType Ty = MOTy::Use) const {
    MI->addRegOperand(RegNo, Ty);
    return *this;
  }

  /// addReg - Add an LLVM value that is to be used as a register...
  ///
  const MachineInstrBuilder &addReg(Value *V,
                                    MOTy::UseType Ty = MOTy::Use) const {
    MI->addRegOperand(V, Ty);
    return *this;
  }

  /// addReg - Add an LLVM value that is to be used as a register...
  ///
  const MachineInstrBuilder &addCCReg(Value *V,
                                      MOTy::UseType Ty = MOTy::Use) const {
    MI->addCCRegOperand(V, Ty);
    return *this;
  }

  /// addRegDef - Add an LLVM value that is to be defined as a register... this
  /// is the same as addReg(V, MOTy::Def).
  ///
  const MachineInstrBuilder &addRegDef(Value *V) const {
    return addReg(V, MOTy::Def);
  }

  /// addPCDisp - Add an LLVM value to be treated as a PC relative
  /// displacement...
  ///
  const MachineInstrBuilder &addPCDisp(Value *V) const {
    MI->addPCDispOperand(V);
    return *this;
  }

  /// addMReg - Add a machine register operand...
  ///
  const MachineInstrBuilder &addMReg(int Reg,
                                     MOTy::UseType Ty = MOTy::Use) const {
    MI->addMachineRegOperand(Reg, Ty);
    return *this;
  }

  /// addSImm - Add a new sign extended immediate operand...
  ///
  const MachineInstrBuilder &addSImm(int64_t val) const {
    MI->addSignExtImmOperand(val);
    return *this;
  }

  /// addZImm - Add a new zero extended immediate operand...
  ///
  const MachineInstrBuilder &addZImm(int64_t Val) const {
    MI->addZeroExtImmOperand(Val);
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

  const MachineInstrBuilder &addConstantPoolIndex(unsigned Idx) const {
    MI->addConstantPoolIndexOperand(Idx);
    return *this;
  }

  const MachineInstrBuilder &addGlobalAddress(GlobalValue *GV,
					      bool isPCRelative = false) const {
    MI->addGlobalAddressOperand(GV, isPCRelative);
    return *this;
  }

  const MachineInstrBuilder &addExternalSymbol(const std::string &Name,
					       bool isPCRelative = false) const{
    MI->addExternalSymbolOperand(Name, isPCRelative);
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

/// BuildMI - This version of the builder also sets up the first "operand" as a
/// destination virtual register.  NumOperands is the number of additional add*
/// calls that are expected, it does not include the destination register.
///
inline MachineInstrBuilder BuildMI(int Opcode, unsigned NumOperands,
                                   unsigned DestReg,
                                   MOTy::UseType useType = MOTy::Def) {
  return MachineInstrBuilder(new MachineInstr(Opcode, NumOperands+1,
                                   true, true)).addReg(DestReg, useType);
}


/// BuildMI - This version of the builder inserts the built MachineInstr into
/// the specified MachineBasicBlock.
///
inline MachineInstrBuilder BuildMI(MachineBasicBlock *BB, int Opcode,
                                   unsigned NumOperands) {
  return MachineInstrBuilder(new MachineInstr(BB, Opcode, NumOperands));
}

/// BuildMI - This version of the builder inserts the built MachineInstr into
/// the specified MachineBasicBlock, and also sets up the first "operand" as a
/// destination virtual register.  NumOperands is the number of additional add*
/// calls that are expected, it does not include the destination register.
///
inline MachineInstrBuilder BuildMI(MachineBasicBlock *BB, int Opcode,
                                   unsigned NumOperands, unsigned DestReg) {
  return MachineInstrBuilder(new MachineInstr(BB, Opcode,
                                              NumOperands+1)).addReg(DestReg,
                                                                     MOTy::Def);
}

} // End llvm namespace

#endif
