//===- X86MCTargetExpr.h - X86 Target Specific MCExpr -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef X86_MCTARGETEXPR_H
#define X86_MCTARGETEXPR_H

#include "llvm/MC/MCExpr.h"

namespace llvm {

/// X86MCTargetExpr - This class represents symbol variants, like foo@GOT.
class X86MCTargetExpr : public MCTargetExpr {
public:
  enum VariantKind {
    Invalid,
    GOT,
    GOTOFF,
    GOTPCREL,
    GOTTPOFF,
    INDNTPOFF,
    NTPOFF,
    PLT,
    TLSGD,
    TPOFF
  };
private:
  /// Sym - The symbol being referenced.
  const MCSymbol * const Sym;
  /// Kind - The modifier.
  const VariantKind Kind;
  
  X86MCTargetExpr(const MCSymbol *S, VariantKind K) : Sym(S), Kind(K) {}
public:
  static X86MCTargetExpr *Create(const MCSymbol *Sym, VariantKind K,
                                 MCContext &Ctx);
  
  void PrintImpl(raw_ostream &OS) const;
  bool EvaluateAsRelocatableImpl(MCValue &Res) const;
};
  
} // end namespace llvm

#endif
