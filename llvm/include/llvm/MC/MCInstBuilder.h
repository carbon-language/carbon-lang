//===-- llvm/MC/MCInstBuilder.h - Simplify creation of MCInsts --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MCInstBuilder class for convenient creation of
// MCInsts.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCINSTBUILDER_H
#define LLVM_MC_MCINSTBUILDER_H

#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"

namespace llvm {

class MCInstBuilder {
  MCInst Inst;

public:
  /// \brief Create a new MCInstBuilder for an MCInst with a specific opcode.
  MCInstBuilder(unsigned Opcode) {
    Inst.setOpcode(Opcode);
  }

  /// \brief Add a new register operand.
  MCInstBuilder &addReg(unsigned Reg) {
    Inst.addOperand(MCOperand::CreateReg(Reg));
    return *this;
  }

  /// \brief Add a new integer immediate operand.
  MCInstBuilder &addImm(int64_t Val) {
    Inst.addOperand(MCOperand::CreateImm(Val));
    return *this;
  }

  /// \brief Add a new floating point immediate operand.
  MCInstBuilder &addFPImm(double Val) {
    Inst.addOperand(MCOperand::CreateFPImm(Val));
    return *this;
  }

  /// \brief Add a new MCExpr operand.
  MCInstBuilder &addExpr(const MCExpr *Val) {
    Inst.addOperand(MCOperand::CreateExpr(Val));
    return *this;
  }

  /// \brief Add a new MCInst operand.
  MCInstBuilder &addInst(const MCInst *Val) {
    Inst.addOperand(MCOperand::CreateInst(Val));
    return *this;
  }

  /// \brief Emit the built instruction to an MCStreamer.
  void emit(MCStreamer &OutStreamer) {
    OutStreamer.EmitInstruction(Inst);
  }
};

} // end namespace llvm

#endif
