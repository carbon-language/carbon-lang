//===-- MipsMCExpr.cpp - Mips specific MC expression classes --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mipsmcexpr"
#include "MipsMCExpr.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"

using namespace llvm;

const MipsMCExpr*
MipsMCExpr::Create(VariantKind Kind, const MCExpr *Expr,
                   MCContext &Ctx) {
  return new (Ctx) MipsMCExpr(Kind, Expr);
}

void MipsMCExpr::PrintImpl(raw_ostream &OS) const {
  switch (Kind) {
  default: llvm_unreachable("Invalid kind!");
  case VK_Mips_ABS_LO: OS << "%lo"; break;
  case VK_Mips_ABS_HI: OS << "%hi"; break;
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

// FIXME: This basically copies MCObjectStreamer::AddValueSymbols. Perhaps
// that method should be made public?
static void AddValueSymbolsImpl(const MCExpr *Value, MCAssembler *Asm) {
  switch (Value->getKind()) {
  case MCExpr::Target:
    llvm_unreachable("Can't handle nested target expr!");

  case MCExpr::Constant:
    break;

  case MCExpr::Binary: {
    const MCBinaryExpr *BE = cast<MCBinaryExpr>(Value);
    AddValueSymbolsImpl(BE->getLHS(), Asm);
    AddValueSymbolsImpl(BE->getRHS(), Asm);
    break;
  }

  case MCExpr::SymbolRef:
    Asm->getOrCreateSymbolData(cast<MCSymbolRefExpr>(Value)->getSymbol());
    break;

  case MCExpr::Unary:
    AddValueSymbolsImpl(cast<MCUnaryExpr>(Value)->getSubExpr(), Asm);
    break;
  }
}

void MipsMCExpr::AddValueSymbols(MCAssembler *Asm) const {
  AddValueSymbolsImpl(getSubExpr(), Asm);
}
