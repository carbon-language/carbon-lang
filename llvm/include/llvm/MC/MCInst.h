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
#include "llvm/ADT/StringRef.h"
#include "llvm/System/DataTypes.h"

namespace llvm {
class raw_ostream;
class MCAsmInfo;
class MCInstPrinter;
class MCExpr;

/// MCOperand - Instances of this class represent operands of the MCInst class.
/// This is a simple discriminated union.
class MCOperand {
  enum MachineOperandType {
    kInvalid,                 ///< Uninitialized.
    kRegister,                ///< Register operand.
    kImmediate,               ///< Immediate operand.
    kExpr                     ///< Relocatable immediate operand.
  };
  unsigned char Kind;
  
  union {
    unsigned RegVal;
    int64_t ImmVal;
    const MCExpr *ExprVal;
  };
public:
  
  MCOperand() : Kind(kInvalid) {}

  bool isValid() const { return Kind != kInvalid; }
  bool isReg() const { return Kind == kRegister; }
  bool isImm() const { return Kind == kImmediate; }
  bool isExpr() const { return Kind == kExpr; }
  
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
  
  int64_t getImm() const {
    assert(isImm() && "This is not an immediate");
    return ImmVal;
  }
  void setImm(int64_t Val) {
    assert(isImm() && "This is not an immediate");
    ImmVal = Val;
  }
  
  const MCExpr *getExpr() const {
    assert(isExpr() && "This is not an expression");
    return ExprVal;
  }
  void setExpr(const MCExpr *Val) {
    assert(isExpr() && "This is not an expression");
    ExprVal = Val;
  }
  
  static MCOperand CreateReg(unsigned Reg) {
    MCOperand Op;
    Op.Kind = kRegister;
    Op.RegVal = Reg;
    return Op;
  }
  static MCOperand CreateImm(int64_t Val) {
    MCOperand Op;
    Op.Kind = kImmediate;
    Op.ImmVal = Val;
    return Op;
  }
  static MCOperand CreateExpr(const MCExpr *Val) {
    MCOperand Op;
    Op.Kind = kExpr;
    Op.ExprVal = Val;
    return Op;
  }

  void print(raw_ostream &OS, const MCAsmInfo *MAI) const;
  void dump() const;
};

  
/// MCInst - Instances of this class represent a single low-level machine
/// instruction. 
class MCInst {
  unsigned Opcode;
  SmallVector<MCOperand, 8> Operands;
public:
  MCInst() : Opcode(0) {}
  
  void setOpcode(unsigned Op) { Opcode = Op; }
  
  unsigned getOpcode() const { return Opcode; }

  const MCOperand &getOperand(unsigned i) const { return Operands[i]; }
  MCOperand &getOperand(unsigned i) { return Operands[i]; }
  unsigned getNumOperands() const { return Operands.size(); }
  
  void addOperand(const MCOperand &Op) {
    Operands.push_back(Op);
  }

  void print(raw_ostream &OS, const MCAsmInfo *MAI) const;
  void dump() const;

  /// \brief Dump the MCInst as prettily as possible using the additional MC
  /// structures, if given. Operators are separated by the \arg Separator
  /// string.
  void dump_pretty(raw_ostream &OS, const MCAsmInfo *MAI = 0,
                   const MCInstPrinter *Printer = 0,
                   StringRef Separator = " ") const;
};


} // end namespace llvm

#endif
