//===-- MipsMCExpr.cpp - Mips specific MC expression classes --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MipsMCExpr.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectStreamer.h"

using namespace llvm;

#define DEBUG_TYPE "mipsmcexpr"

bool MipsMCExpr::isSupportedBinaryExpr(MCSymbolRefExpr::VariantKind VK,
                                       const MCBinaryExpr *BE) {
  switch (VK) {
  case MCSymbolRefExpr::VK_Mips_ABS_LO:
  case MCSymbolRefExpr::VK_Mips_ABS_HI:
  case MCSymbolRefExpr::VK_Mips_HIGHER:
  case MCSymbolRefExpr::VK_Mips_HIGHEST:
    break;
  default:
    return false;
  }

  // We support expressions of the form "(sym1 binop1 sym2) binop2 const",
  // where "binop2 const" is optional.
  if (isa<MCBinaryExpr>(BE->getLHS())) {
    if (!isa<MCConstantExpr>(BE->getRHS()))
      return false;
    BE = cast<MCBinaryExpr>(BE->getLHS());
  }
  return (isa<MCSymbolRefExpr>(BE->getLHS())
          && isa<MCSymbolRefExpr>(BE->getRHS()));
}

const MipsMCExpr*
MipsMCExpr::Create(MCSymbolRefExpr::VariantKind VK, const MCExpr *Expr,
                   MCContext &Ctx) {
  VariantKind Kind;
  switch (VK) {
  case MCSymbolRefExpr::VK_Mips_ABS_LO:
    Kind = VK_Mips_LO;
    break;
  case MCSymbolRefExpr::VK_Mips_ABS_HI:
    Kind = VK_Mips_HI;
    break;
  case MCSymbolRefExpr::VK_Mips_HIGHER:
    Kind = VK_Mips_HIGHER;
    break;
  case MCSymbolRefExpr::VK_Mips_HIGHEST:
    Kind = VK_Mips_HIGHEST;
    break;
  default:
    llvm_unreachable("Invalid kind!");
  }

  return new (Ctx) MipsMCExpr(Kind, Expr);
}

void MipsMCExpr::PrintImpl(raw_ostream &OS) const {
  switch (Kind) {
  default: llvm_unreachable("Invalid kind!");
  case VK_Mips_LO: OS << "%lo"; break;
  case VK_Mips_HI: OS << "%hi"; break;
  case VK_Mips_HIGHER: OS << "%higher"; break;
  case VK_Mips_HIGHEST: OS << "%highest"; break;
  }

  OS << '(';
  Expr->print(OS);
  OS << ')';
}

bool
MipsMCExpr::EvaluateAsRelocatableImpl(MCValue &Res,
                                      const MCAsmLayout *Layout) const {
  return getSubExpr()->EvaluateAsRelocatable(Res, Layout);
}

void MipsMCExpr::visitUsedExpr(MCStreamer &Streamer) const {
  Streamer.visitUsedExpr(*getSubExpr());
}
