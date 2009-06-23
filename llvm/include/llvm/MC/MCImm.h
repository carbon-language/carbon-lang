//===-- llvm/MC/MCImm.h - MCImm class ---------------------------*- C++ -*-===//
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

#ifndef LLVM_MC_MCIMM_H
#define LLVM_MC_MCIMM_H

#include "llvm/Support/DataTypes.h"

namespace llvm {
class MCSymbol;

/// MCImm - This represents an "assembler immediate".  In its most general form,
/// this can hold "SymbolA - SymbolB + imm64".  Not all targets supports
/// relocations of this general form, but we need to represent this anyway.
class MCImm {
  MCSymbol *SymA, *SymB;
  int64_t Cst;
public:

  int64_t getCst() const { return Cst; }
  MCSymbol *getSymA() const { return SymA; }
  MCSymbol *getSymB() const { return SymB; }
  
  
  static MCImm get(MCSymbol *SymA, MCSymbol *SymB = 0, int64_t Val = 0) {
    MCImm R;
    R.Cst = Val;
    R.SymA = SymA;
    R.SymB = SymB;
    return R;
  }
  
  static MCImm get(int64_t Val) {
    MCImm R;
    R.Cst = Val;
    R.SymA = 0;
    R.SymB = 0;
    return R;
  }
  
};

} // end namespace llvm

#endif
