//===-- MipsMCSymbolRefExpr.h - Mips specific MCSymbolRefExpr class -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSMCSYMBOLREFEXPR_H
#define MIPSMCSYMBOLREFEXPR_H
#include "llvm/MC/MCExpr.h"

namespace llvm {

class MipsMCSymbolRefExpr : public MCTargetExpr {
public:
  enum VariantKind {
    VK_Mips_None,
    VK_Mips_GPREL,
    VK_Mips_GOT_CALL,
    VK_Mips_GOT,
    VK_Mips_ABS_HI,
    VK_Mips_ABS_LO,
    VK_Mips_TLSGD,
    VK_Mips_GOTTPREL,
    VK_Mips_TPREL_HI,
    VK_Mips_TPREL_LO,
    VK_Mips_GPOFF_HI,
    VK_Mips_GPOFF_LO,
    VK_Mips_GOT_DISP,
    VK_Mips_GOT_PAGE,
    VK_Mips_GOT_OFST
  };

private:
  const VariantKind Kind;
  const MCSymbol *Symbol;
  int Offset;

  explicit MipsMCSymbolRefExpr(VariantKind _Kind, const MCSymbol *_Symbol,
                               int _Offset)
    : Kind(_Kind), Symbol(_Symbol), Offset(_Offset) {}
  
public:
  static const MipsMCSymbolRefExpr *Create(VariantKind Kind,
                                           const MCSymbol *Symbol, int Offset,
                                           MCContext &Ctx);

  void PrintImpl(raw_ostream &OS) const;
  bool EvaluateAsRelocatableImpl(MCValue &Res,
                                 const MCAsmLayout *Layout) const;
  void AddValueSymbols(MCAssembler *) const;
  const MCSection *FindAssociatedSection() const;

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Target;
  }

  static bool classof(const MipsMCSymbolRefExpr *) { return true; }

  int getOffset() const { return Offset; }
  void setOffset(int O) { Offset = O; }
};
} // end namespace llvm

#endif
