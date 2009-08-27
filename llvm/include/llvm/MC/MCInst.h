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

#include "llvm/MC/MCValue.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/DebugLoc.h"

namespace llvm {
class raw_ostream;

/// MCOperand - Instances of this class represent operands of the MCInst class.
/// This is a simple discriminated union.
class MCOperand {
  enum MachineOperandType {
    kInvalid,                 ///< Uninitialized.
    kRegister,                ///< Register operand.
    kImmediate,               ///< Immediate operand.
    kMBBLabel,                ///< Basic block label.
    kMCValue                  ///< Relocatable immediate operand.
  };
  unsigned char Kind;
  
  union {
    unsigned RegVal;
    int64_t ImmVal;
    MCValue MCValueVal;
    struct {
      unsigned FunctionNo;
      unsigned BlockNo;
    } MBBLabel;
  };
public:
  
  MCOperand() : Kind(kInvalid) {}
  MCOperand(const MCOperand &RHS) { *this = RHS; }

  bool isValid() const { return Kind != kInvalid; }
  bool isReg() const { return Kind == kRegister; }
  bool isImm() const { return Kind == kImmediate; }
  bool isMBBLabel() const { return Kind == kMBBLabel; }
  bool isMCValue() const { return Kind == kMCValue; }
  
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
  
  unsigned getMBBLabelFunction() const {
    assert(isMBBLabel() && "Wrong accessor");
    return MBBLabel.FunctionNo; 
  }
  unsigned getMBBLabelBlock() const {
    assert(isMBBLabel() && "Wrong accessor");
    return MBBLabel.BlockNo; 
  }

  const MCValue &getMCValue() const {
    assert(isMCValue() && "This is not an MCValue");
    return MCValueVal;
  }
  void setMCValue(const MCValue &Val) {
    assert(isMCValue() && "This is not an MCValue");
    MCValueVal = Val;
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
  static MCOperand CreateMBBLabel(unsigned Fn, unsigned MBB) {
    MCOperand Op;
    Op.Kind = kMBBLabel;
    Op.MBBLabel.FunctionNo = Fn;
    Op.MBBLabel.BlockNo = MBB;
    return Op;
  }
  static MCOperand CreateMCValue(const MCValue &Val) {
    MCOperand Op;
    Op.Kind = kMCValue;
    Op.MCValueVal = Val;
    return Op;
  }

  void print(raw_ostream &OS) const;
  void dump() const;
};

  
/// MCInst - Instances of this class represent a single low-level machine
/// instruction. 
class MCInst {
  unsigned Opcode;
  SmallVector<MCOperand, 8> Operands;
public:
  MCInst() : Opcode(~0U) {}
  
  void setOpcode(unsigned Op) { Opcode = Op; }
  
  unsigned getOpcode() const { return Opcode; }
  DebugLoc getDebugLoc() const { return DebugLoc(); }
  
  const MCOperand &getOperand(unsigned i) const { return Operands[i]; }
  MCOperand &getOperand(unsigned i) { return Operands[i]; }
  unsigned getNumOperands() const { return Operands.size(); }
  
  void addOperand(const MCOperand &Op) {
    Operands.push_back(Op);
  }

  void print(raw_ostream &OS) const;
  void dump() const;
};


} // end namespace llvm

#endif
