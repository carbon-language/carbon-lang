//===-- ARM64MCExpr.cpp - ARM64 specific MC expression classes --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the assembly expression modifiers
// accepted by the AArch64 architecture (e.g. ":lo12:", ":gottprel_g1:", ...).
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "aarch64symbolrefexpr"
#include "ARM64MCExpr.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELF.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

const ARM64MCExpr *ARM64MCExpr::Create(const MCExpr *Expr, VariantKind Kind,
                                       MCContext &Ctx) {
  return new (Ctx) ARM64MCExpr(Expr, Kind);
}

StringRef ARM64MCExpr::getVariantKindName() const {
  switch (static_cast<uint32_t>(getKind())) {
  case VK_CALL:                return "";
  case VK_LO12:                return ":lo12:";
  case VK_ABS_G3:              return ":abs_g3:";
  case VK_ABS_G2:              return ":abs_g2:";
  case VK_ABS_G2_NC:           return ":abs_g2_nc:";
  case VK_ABS_G1:              return ":abs_g1:";
  case VK_ABS_G1_NC:           return ":abs_g1_nc:";
  case VK_ABS_G0:              return ":abs_g0:";
  case VK_ABS_G0_NC:           return ":abs_g0_nc:";
  case VK_DTPREL_G2:           return ":dtprel_g2:";
  case VK_DTPREL_G1:           return ":dtprel_g1:";
  case VK_DTPREL_G1_NC:        return ":dtprel_g1_nc:";
  case VK_DTPREL_G0:           return ":dtprel_g0:";
  case VK_DTPREL_G0_NC:        return ":dtprel_g0_nc:";
  case VK_DTPREL_LO12:         return ":dtprel_lo12:";
  case VK_DTPREL_LO12_NC:      return ":dtprel_lo12_nc:";
  case VK_TPREL_G2:            return ":tprel_g2:";
  case VK_TPREL_G1:            return ":tprel_g1:";
  case VK_TPREL_G1_NC:         return ":tprel_g1_nc:";
  case VK_TPREL_G0:            return ":tprel_g0:";
  case VK_TPREL_G0_NC:         return ":tprel_g0_nc:";
  case VK_TPREL_LO12:          return ":tprel_lo12:";
  case VK_TPREL_LO12_NC:       return ":tprel_lo12_nc:";
  case VK_TLSDESC_LO12:        return ":tlsdesc_lo12:";
  case VK_ABS_PAGE:            return "";
  case VK_GOT_PAGE:            return ":got:";
  case VK_GOT_LO12:            return ":got_lo12:";
  case VK_GOTTPREL_PAGE:       return ":gottprel:";
  case VK_GOTTPREL_LO12_NC:    return ":gottprel_lo12:";
  case VK_GOTTPREL_G1:         return ":gottprel_g1:";
  case VK_GOTTPREL_G0_NC:      return ":gottprel_g0_nc:";
  case VK_TLSDESC:             return "";
  case VK_TLSDESC_PAGE:        return ":tlsdesc:";
  default:
    llvm_unreachable("Invalid ELF symbol kind");
  }
}

void ARM64MCExpr::PrintImpl(raw_ostream &OS) const {
  if (getKind() != VK_NONE)
    OS << getVariantKindName();
  OS << *Expr;
}

// FIXME: This basically copies MCObjectStreamer::AddValueSymbols. Perhaps
// that method should be made public?
// FIXME: really do above: now that two backends are using it.
static void AddValueSymbolsImpl(const MCExpr *Value, MCAssembler *Asm) {
  switch (Value->getKind()) {
  case MCExpr::Target:
    llvm_unreachable("Can't handle nested target expr!");
    break;

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

void ARM64MCExpr::AddValueSymbols(MCAssembler *Asm) const {
  AddValueSymbolsImpl(getSubExpr(), Asm);
}

const MCSection *ARM64MCExpr::FindAssociatedSection() const {
  llvm_unreachable("FIXME: what goes here?");
}

bool ARM64MCExpr::EvaluateAsRelocatableImpl(MCValue &Res,
                                            const MCAsmLayout *Layout) const {
  if (!getSubExpr()->EvaluateAsRelocatable(Res, Layout))
    return false;

  Res =
      MCValue::get(Res.getSymA(), Res.getSymB(), Res.getConstant(), getKind());

  return true;
}

static void fixELFSymbolsInTLSFixupsImpl(const MCExpr *Expr, MCAssembler &Asm) {
  switch (Expr->getKind()) {
  case MCExpr::Target:
    llvm_unreachable("Can't handle nested target expression");
    break;
  case MCExpr::Constant:
    break;

  case MCExpr::Binary: {
    const MCBinaryExpr *BE = cast<MCBinaryExpr>(Expr);
    fixELFSymbolsInTLSFixupsImpl(BE->getLHS(), Asm);
    fixELFSymbolsInTLSFixupsImpl(BE->getRHS(), Asm);
    break;
  }

  case MCExpr::SymbolRef: {
    // We're known to be under a TLS fixup, so any symbol should be
    // modified. There should be only one.
    const MCSymbolRefExpr &SymRef = *cast<MCSymbolRefExpr>(Expr);
    MCSymbolData &SD = Asm.getOrCreateSymbolData(SymRef.getSymbol());
    MCELF::SetType(SD, ELF::STT_TLS);
    break;
  }

  case MCExpr::Unary:
    fixELFSymbolsInTLSFixupsImpl(cast<MCUnaryExpr>(Expr)->getSubExpr(), Asm);
    break;
  }
}

void ARM64MCExpr::fixELFSymbolsInTLSFixups(MCAssembler &Asm) const {
  switch (getSymbolLoc(Kind)) {
  default:
    return;
  case VK_DTPREL:
  case VK_GOTTPREL:
  case VK_TPREL:
  case VK_TLSDESC:
    break;
  }

  fixELFSymbolsInTLSFixupsImpl(getSubExpr(), Asm);
}
