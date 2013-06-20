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

#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/MCAsmLayout.h"

namespace llvm {

class PPCMCExpr : public MCTargetExpr {
public:
  enum VariantKind {
    VK_PPC_None,
    VK_PPC_HA16,
    VK_PPC_LO16
  };

private:
  const VariantKind Kind;
  const MCExpr *Expr;
  const int AssemblerDialect;

  explicit PPCMCExpr(VariantKind _Kind, const MCExpr *_Expr,
                     int _AssemblerDialect)
    : Kind(_Kind), Expr(_Expr), AssemblerDialect(_AssemblerDialect) {}

public:
  /// @name Construction
  /// @{

  static const PPCMCExpr *Create(VariantKind Kind, const MCExpr *Expr,
                                      MCContext &Ctx);

  static const PPCMCExpr *CreateHa16(const MCExpr *Expr, MCContext &Ctx) {
    return Create(VK_PPC_HA16, Expr, Ctx);
  }

  static const PPCMCExpr *CreateLo16(const MCExpr *Expr, MCContext &Ctx) {
    return Create(VK_PPC_LO16, Expr, Ctx);
  }

  /// @}
  /// @name Accessors
  /// @{

  /// getOpcode - Get the kind of this expression.
  VariantKind getKind() const { return Kind; }

  /// getSubExpr - Get the child of this expression.
  const MCExpr *getSubExpr() const { return Expr; }

  /// isDarwinSyntax - True if expression is to be printed using Darwin syntax.
  bool isDarwinSyntax() const { return AssemblerDialect == 1; }


  /// @}

  void PrintImpl(raw_ostream &OS) const;
  bool EvaluateAsRelocatableImpl(MCValue &Res,
                                 const MCAsmLayout *Layout) const;
  void AddValueSymbols(MCAssembler *) const;
  const MCSection *FindAssociatedSection() const {
    return getSubExpr()->FindAssociatedSection();
  }

  // There are no TLS PPCMCExprs at the moment.
  void fixELFSymbolsInTLSFixups(MCAssembler &Asm) const {}

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Target;
  }
};
} // end namespace llvm

#endif
