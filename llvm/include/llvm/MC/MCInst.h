//===-- llvm/MC/MCInst.h - MCInst class -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MCInst and MCOperand classes, which
// is the basic representation used to represent low-level machine code
// instructions.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_MC_MCINST_H
#define LLVM_MC_MCINST_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

/// MCOperand - Instances of this class represent operands of the MCInst class.
/// This is a simple discriminated union.
class MCOperand {
  enum MachineOperandType {
    kInvalid,                 ///< Uninitialized.
    kRegister,                ///< Register operand.
    kImmediate                ///< Immediate operand.
  };
  unsigned char Kind;
  
  union {
    unsigned RegVal;
    uint64_t ImmVal;
  };
public:
  
  MCOperand() : Kind(kInvalid) {}
  MCOperand(const MCOperand &RHS) { *this = RHS; }

  bool isReg() const { return Kind == kRegister; }
  bool isImm() const { return Kind == kImmediate; }
  
  /// getReg - Returns the register number.
  unsigned getReg() const {
    assert(isReg() && "This is not a register operand!");
    return RegVal;
  }

  /// setReg - Set the register number.
  void setReg(unsigned Reg) {
    assert(isReg() && "This is not a register operand!");
    RegVal = Reg;
  }
  
  uint64_t getImm() const {
    assert(isImm() && "This is not an immediate");
    return ImmVal;
  }
  void setImm(uint64_t Val) {
    assert(isImm() && "This is not an immediate");
    ImmVal = Val;
  }
  
  void MakeReg(unsigned Reg) {
    Kind = kRegister;
    RegVal = Reg;
  }
  void MakeImm(uint64_t Val) {
    Kind = kImmediate;
    ImmVal = Val;
  }
};

  
/// MCInst - Instances of this class represent a single low-level machine
/// instruction. 
class MCInst {
  unsigned Opcode;
  SmallVector<MCOperand, 8> Operands;
public:
  MCInst() : Opcode(~0U) {}
  
  
  
};


} // end namespace llvm

#endif
