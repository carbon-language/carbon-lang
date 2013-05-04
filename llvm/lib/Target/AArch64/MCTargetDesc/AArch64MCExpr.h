//==- AArch64MCExpr.h - AArch64 specific MC expression classes --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes AArch64-specific MCExprs, used for modifiers like
// ":lo12:" or ":gottprel_g1:".
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_AARCH64MCEXPR_H
#define LLVM_AARCH64MCEXPR_H

#include "llvm/MC/MCExpr.h"

namespace llvm {

class AArch64MCExpr : public MCTargetExpr {
public:
  enum VariantKind {
    VK_AARCH64_None,
    VK_AARCH64_GOT,      // :got: modifier in assembly
    VK_AARCH64_GOT_LO12, // :got_lo12:
    VK_AARCH64_LO12,     // :lo12:

    VK_AARCH64_ABS_G0, // :abs_g0:
    VK_AARCH64_ABS_G0_NC, // :abs_g0_nc:
    VK_AARCH64_ABS_G1,
    VK_AARCH64_ABS_G1_NC,
    VK_AARCH64_ABS_G2,
    VK_AARCH64_ABS_G2_NC,
    VK_AARCH64_ABS_G3,

    VK_AARCH64_SABS_G0, // :abs_g0_s:
    VK_AARCH64_SABS_G1,
    VK_AARCH64_SABS_G2,

    VK_AARCH64_DTPREL_G2, // :dtprel_g2:
    VK_AARCH64_DTPREL_G1,
    VK_AARCH64_DTPREL_G1_NC,
    VK_AARCH64_DTPREL_G0,
    VK_AARCH64_DTPREL_G0_NC,
    VK_AARCH64_DTPREL_HI12,
    VK_AARCH64_DTPREL_LO12,
    VK_AARCH64_DTPREL_LO12_NC,

    VK_AARCH64_GOTTPREL_G1, // :gottprel:
    VK_AARCH64_GOTTPREL_G0_NC,
    VK_AARCH64_GOTTPREL,
    VK_AARCH64_GOTTPREL_LO12,

    VK_AARCH64_TPREL_G2, // :tprel:
    VK_AARCH64_TPREL_G1,
    VK_AARCH64_TPREL_G1_NC,
    VK_AARCH64_TPREL_G0,
    VK_AARCH64_TPREL_G0_NC,
    VK_AARCH64_TPREL_HI12,
    VK_AARCH64_TPREL_LO12,
    VK_AARCH64_TPREL_LO12_NC,

    VK_AARCH64_TLSDESC, // :tlsdesc:
    VK_AARCH64_TLSDESC_LO12
  };

private:
  const VariantKind Kind;
  const MCExpr *Expr;

  explicit AArch64MCExpr(VariantKind _Kind, const MCExpr *_Expr)
    : Kind(_Kind), Expr(_Expr) {}

public:
  /// @name Construction
  /// @{

  static const AArch64MCExpr *Create(VariantKind Kind, const MCExpr *Expr,
                                     MCContext &Ctx);

  static const AArch64MCExpr *CreateLo12(const MCExpr *Expr, MCContext &Ctx) {
    return Create(VK_AARCH64_LO12, Expr, Ctx);
  }

  static const AArch64MCExpr *CreateGOT(const MCExpr *Expr, MCContext &Ctx) {
    return Create(VK_AARCH64_GOT, Expr, Ctx);
  }

  static const AArch64MCExpr *CreateGOTLo12(const MCExpr *Expr,
                                            MCContext &Ctx) {
    return Create(VK_AARCH64_GOT_LO12, Expr, Ctx);
  }

  static const AArch64MCExpr *CreateDTPREL_G1(const MCExpr *Expr,
                                             MCContext &Ctx) {
    return Create(VK_AARCH64_DTPREL_G1, Expr, Ctx);
  }

  static const AArch64MCExpr *CreateDTPREL_G0_NC(const MCExpr *Expr,
                                                MCContext &Ctx) {
    return Create(VK_AARCH64_DTPREL_G0_NC, Expr, Ctx);
  }

  static const AArch64MCExpr *CreateGOTTPREL(const MCExpr *Expr,
                                             MCContext &Ctx) {
    return Create(VK_AARCH64_GOTTPREL, Expr, Ctx);
  }

  static const AArch64MCExpr *CreateGOTTPRELLo12(const MCExpr *Expr,
                                                 MCContext &Ctx) {
    return Create(VK_AARCH64_GOTTPREL_LO12, Expr, Ctx);
  }

  static const AArch64MCExpr *CreateTLSDesc(const MCExpr *Expr,
                                            MCContext &Ctx) {
    return Create(VK_AARCH64_TLSDESC, Expr, Ctx);
  }

  static const AArch64MCExpr *CreateTLSDescLo12(const MCExpr *Expr,
                                                MCContext &Ctx) {
    return Create(VK_AARCH64_TLSDESC_LO12, Expr, Ctx);
  }

  static const AArch64MCExpr *CreateTPREL_G1(const MCExpr *Expr,
                                             MCContext &Ctx) {
    return Create(VK_AARCH64_TPREL_G1, Expr, Ctx);
  }

  static const AArch64MCExpr *CreateTPREL_G0_NC(const MCExpr *Expr,
                                                MCContext &Ctx) {
    return Create(VK_AARCH64_TPREL_G0_NC, Expr, Ctx);
  }

  static const AArch64MCExpr *CreateABS_G3(const MCExpr *Expr,
                                           MCContext &Ctx) {
    return Create(VK_AARCH64_ABS_G3, Expr, Ctx);
  }

  static const AArch64MCExpr *CreateABS_G2_NC(const MCExpr *Expr,
                                           MCContext &Ctx) {
    return Create(VK_AARCH64_ABS_G2_NC, Expr, Ctx);
  }

  static const AArch64MCExpr *CreateABS_G1_NC(const MCExpr *Expr,
                                           MCContext &Ctx) {
    return Create(VK_AARCH64_ABS_G1_NC, Expr, Ctx);
  }

  static const AArch64MCExpr *CreateABS_G0_NC(const MCExpr *Expr,
                                           MCContext &Ctx) {
    return Create(VK_AARCH64_ABS_G0_NC, Expr, Ctx);
  }

  /// @}
  /// @name Accessors
  /// @{

  /// getOpcode - Get the kind of this expression.
  VariantKind getKind() const { return Kind; }

  /// getSubExpr - Get the child of this expression.
  const MCExpr *getSubExpr() const { return Expr; }

  /// @}

  void PrintImpl(raw_ostream &OS) const;
  bool EvaluateAsRelocatableImpl(MCValue &Res,
                                 const MCAsmLayout *Layout) const;
  void AddValueSymbols(MCAssembler *) const;
  const MCSection *FindAssociatedSection() const {
    return getSubExpr()->FindAssociatedSection();
  }

  void fixELFSymbolsInTLSFixups(MCAssembler &Asm) const;

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Target;
  }

  static bool classof(const AArch64MCExpr *) { return true; }

};
} // end namespace llvm

#endif
