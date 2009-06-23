//===-- llvm/MC/MCValue.h - MCValue class -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MCValue class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCVALUE_H
#define LLVM_MC_MCVALUE_H

#include "llvm/Support/DataTypes.h"

namespace llvm {
class MCSymbol;

/// MCValue - This represents an "assembler immediate".  In its most general
/// form, this can hold "SymbolA - SymbolB + imm64".  Not all targets supports
/// relocations of this general form, but we need to represent this anyway.
///
/// Note that this class must remain a simple POD value class, because we need
/// it to live in unions etc.
class MCValue {
  MCSymbol *SymA, *SymB;
  int64_t Cst;
public:

  int64_t getCst() const { return Cst; }
  MCSymbol *getSymA() const { return SymA; }
  MCSymbol *getSymB() const { return SymB; }
  
  
  static MCValue get(MCSymbol *SymA, MCSymbol *SymB = 0, int64_t Val = 0) {
    MCValue R;
    R.Cst = Val;
    R.SymA = SymA;
    R.SymB = SymB;
    return R;
  }
  
  static MCValue get(int64_t Val) {
    MCValue R;
    R.Cst = Val;
    R.SymA = 0;
    R.SymB = 0;
    return R;
  }
  
};

} // end namespace llvm

#endif
