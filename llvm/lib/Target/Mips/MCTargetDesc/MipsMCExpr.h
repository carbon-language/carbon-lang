//===-- MipsMCExpr.h - Mips specific MC expression classes ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSMCEXPR_H
#define MIPSMCEXPR_H

#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCValue.h"

namespace llvm {

class MipsMCExpr : public MCTargetExpr {
public:
  enum VariantKind {
    VK_Mips_None,
    VK_Mips_ABS_LO,
    VK_Mips_ABS_HI
  };

private:
  const VariantKind Kind;
  const MCExpr *Expr;

  explicit MipsMCExpr(VariantKind Kind, const MCExpr *Expr)
    : Kind(Kind), Expr(Expr) {}

public:
  static const MipsMCExpr *Create(VariantKind Kind, const MCExpr *Expr,
                                  MCContext &Ctx);

  static const MipsMCExpr *CreateLo(const MCExpr *Expr, MCContext &Ctx) {
    return Create(VK_Mips_ABS_LO, Expr, Ctx);
  }

  static const MipsMCExpr *CreateHi(const MCExpr *Expr, MCContext &Ctx) {
    return Create(VK_Mips_ABS_HI, Expr, Ctx);
  }

  /// getOpcode - Get the kind of this expression.
  VariantKind getKind() const { return Kind; }

  /// getSubExpr - Get the child of this expression.
  const MCExpr *getSubExpr() const { return Expr; }

  void PrintImpl(raw_ostream &OS) const;
  bool EvaluateAsRelocatableImpl(MCValue &Res,
                                 const MCAsmLayout *Layout) const;
  void AddValueSymbols(MCAssembler *) const;
  const MCSection *FindAssociatedSection() const {
    return getSubExpr()->FindAssociatedSection();
  }

  // There are no TLS MipsMCExprs at the moment.
  void fixELFSymbolsInTLSFixups(MCAssembler &Asm) const {}

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Target;
  }
};
} // end namespace llvm

#endif
