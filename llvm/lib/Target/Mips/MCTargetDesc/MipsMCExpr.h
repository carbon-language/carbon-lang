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
    VK_Mips_LO,
    VK_Mips_HI,
    VK_Mips_HIGHER,
    VK_Mips_HIGHEST
  };

private:
  const VariantKind Kind;
  const MCExpr *Expr;

  explicit MipsMCExpr(VariantKind Kind, const MCExpr *Expr)
    : Kind(Kind), Expr(Expr) {}

public:
  static bool isSupportedBinaryExpr(MCSymbolRefExpr::VariantKind VK,
                                    const MCBinaryExpr *BE);

  static const MipsMCExpr *Create(MCSymbolRefExpr::VariantKind VK,
                                  const MCExpr *Expr, MCContext &Ctx);

  /// getOpcode - Get the kind of this expression.
  VariantKind getKind() const { return Kind; }

  /// getSubExpr - Get the child of this expression.
  const MCExpr *getSubExpr() const { return Expr; }

  void PrintImpl(raw_ostream &OS) const override;
  bool EvaluateAsRelocatableImpl(MCValue &Res,
                                 const MCAsmLayout *Layout,
                                 const MCFixup *Fixup) const override;
  void visitUsedExpr(MCStreamer &Streamer) const override;
  const MCSection *FindAssociatedSection() const override {
    return getSubExpr()->FindAssociatedSection();
  }

  // There are no TLS MipsMCExprs at the moment.
  void fixELFSymbolsInTLSFixups(MCAssembler &Asm) const override {}

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Target;
  }
};
} // end namespace llvm

#endif
