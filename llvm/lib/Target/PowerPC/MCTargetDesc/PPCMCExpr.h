//===-- PPCMCExpr.h - PPC specific MC expression classes --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PPCMCEXPR_H
#define PPCMCEXPR_H

#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCValue.h"

namespace llvm {

class PPCMCExpr : public MCTargetExpr {
public:
  enum VariantKind {
    VK_PPC_None,
    VK_PPC_LO,
    VK_PPC_HI,
    VK_PPC_HA,
    VK_PPC_HIGHER,
    VK_PPC_HIGHERA,
    VK_PPC_HIGHEST,
    VK_PPC_HIGHESTA
  };

private:
  const VariantKind Kind;
  const MCExpr *Expr;
  bool IsDarwin;

  explicit PPCMCExpr(VariantKind _Kind, const MCExpr *_Expr,
                     bool _IsDarwin)
    : Kind(_Kind), Expr(_Expr), IsDarwin(_IsDarwin) {}

public:
  /// @name Construction
  /// @{

  static const PPCMCExpr *Create(VariantKind Kind, const MCExpr *Expr,
                                 bool isDarwin, MCContext &Ctx);

  static const PPCMCExpr *CreateLo(const MCExpr *Expr,
                                   bool isDarwin, MCContext &Ctx) {
    return Create(VK_PPC_LO, Expr, isDarwin, Ctx);
  }

  static const PPCMCExpr *CreateHi(const MCExpr *Expr,
                                   bool isDarwin, MCContext &Ctx) {
    return Create(VK_PPC_HI, Expr, isDarwin, Ctx);
  }

  static const PPCMCExpr *CreateHa(const MCExpr *Expr,
                                   bool isDarwin, MCContext &Ctx) {
    return Create(VK_PPC_HA, Expr, isDarwin, Ctx);
  }

  /// @}
  /// @name Accessors
  /// @{

  /// getOpcode - Get the kind of this expression.
  VariantKind getKind() const { return Kind; }

  /// getSubExpr - Get the child of this expression.
  const MCExpr *getSubExpr() const { return Expr; }

  /// isDarwinSyntax - True if expression is to be printed using Darwin syntax.
  bool isDarwinSyntax() const { return IsDarwin; }


  /// @}

  void PrintImpl(raw_ostream &OS) const override;
  bool EvaluateAsRelocatableImpl(MCValue &Res,
                                 const MCAsmLayout *Layout) const override;
  void AddValueSymbols(MCAssembler *) const override;
  const MCSection *FindAssociatedSection() const override {
    return getSubExpr()->FindAssociatedSection();
  }

  // There are no TLS PPCMCExprs at the moment.
  void fixELFSymbolsInTLSFixups(MCAssembler &Asm) const override {}

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Target;
  }
};
} // end namespace llvm

#endif
