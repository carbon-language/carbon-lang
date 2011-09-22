//===-- MipsMCSymbolRefExpr.cpp - Mips specific MC expression classes -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mipsmcsymbolrefexpr"
#include "MipsMCSymbolRefExpr.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbol.h"
using namespace llvm;

const MipsMCSymbolRefExpr*
MipsMCSymbolRefExpr::Create(VariantKind Kind, const MCSymbol *Symbol,
                            int Offset, MCContext &Ctx) {
  return new (Ctx) MipsMCSymbolRefExpr(Kind, Symbol, Offset);
}

void MipsMCSymbolRefExpr::PrintImpl(raw_ostream &OS) const {
  switch (Kind) {
  default: assert(0 && "Invalid kind!");
  case VK_Mips_None:     break;
  case VK_Mips_GPREL:    OS << "%gp_rel("; break;
  case VK_Mips_GOT_CALL: OS << "%call16("; break;
  case VK_Mips_GOT:      OS << "%got(";    break;
  case VK_Mips_ABS_HI:   OS << "%hi(";     break;
  case VK_Mips_ABS_LO:   OS << "%lo(";     break;
  case VK_Mips_TLSGD:    OS << "%tlsgd(";  break;
  case VK_Mips_GOTTPREL: OS << "%gottprel("; break;
  case VK_Mips_TPREL_HI: OS << "%tprel_hi("; break;
  case VK_Mips_TPREL_LO: OS << "%tprel_lo("; break;
  case VK_Mips_GPOFF_HI: OS << "%hi(%neg(%gp_rel("; break;
  case VK_Mips_GPOFF_LO: OS << "%lo(%neg(%gp_rel("; break;
  case VK_Mips_GOT_DISP: OS << "%got_disp("; break;
  case VK_Mips_GOT_PAGE: OS << "%got_page("; break;
  case VK_Mips_GOT_OFST: OS << "%got_ofst("; break;
  }

  OS << *Symbol;

  if (Offset) {
    if (Offset > 0)
      OS << '+';
    OS << Offset;
  }

  if (Kind != VK_Mips_None)
    OS << ')';
}

bool
MipsMCSymbolRefExpr::EvaluateAsRelocatableImpl(MCValue &Res,
                                              const MCAsmLayout *Layout) const {
  return false;
}

void MipsMCSymbolRefExpr::AddValueSymbols(MCAssembler *Asm) const {
  Asm->getOrCreateSymbolData(*Symbol);
}

const MCSection *MipsMCSymbolRefExpr::FindAssociatedSection() const {
  return Symbol->isDefined() ? &Symbol->getSection() : NULL;
}
  
