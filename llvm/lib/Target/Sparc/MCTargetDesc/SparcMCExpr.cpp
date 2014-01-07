//===-- SparcMCExpr.cpp - Sparc specific MC expression classes --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the assembly expression modifiers
// accepted by the Sparc architecture (e.g. "%hi", "%lo", ...).
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sparcmcexpr"
#include "SparcMCExpr.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCELF.h"
#include "llvm/Object/ELF.h"


using namespace llvm;

const SparcMCExpr*
SparcMCExpr::Create(VariantKind Kind, const MCExpr *Expr,
                      MCContext &Ctx) {
    return new (Ctx) SparcMCExpr(Kind, Expr);
}


void SparcMCExpr::PrintImpl(raw_ostream &OS) const
{
  bool closeParen = true;
  switch (Kind) {
  case VK_Sparc_None:     closeParen = false; break;
  case VK_Sparc_LO:       OS << "%lo(";  break;
  case VK_Sparc_HI:       OS << "%hi(";  break;
  case VK_Sparc_H44:      OS << "%h44("; break;
  case VK_Sparc_M44:      OS << "%m44("; break;
  case VK_Sparc_L44:      OS << "%l44("; break;
  case VK_Sparc_HH:       OS << "%hh(";  break;
  case VK_Sparc_HM:       OS << "%hm(";  break;
  case VK_Sparc_TLS_GD_HI22:   OS << "%tgd_hi22(";   break;
  case VK_Sparc_TLS_GD_LO10:   OS << "%tgd_lo10(";   break;
  case VK_Sparc_TLS_GD_ADD:    OS << "%tgd_add(";    break;
  case VK_Sparc_TLS_GD_CALL:   OS << "%tgd_call(";   break;
  case VK_Sparc_TLS_LDM_HI22:  OS << "%tldm_hi22(";  break;
  case VK_Sparc_TLS_LDM_LO10:  OS << "%tldm_lo10(";  break;
  case VK_Sparc_TLS_LDM_ADD:   OS << "%tldm_add(";   break;
  case VK_Sparc_TLS_LDM_CALL:  OS << "%tldm_call(";  break;
  case VK_Sparc_TLS_LDO_HIX22: OS << "%tldo_hix22("; break;
  case VK_Sparc_TLS_LDO_LOX10: OS << "%tldo_lox10("; break;
  case VK_Sparc_TLS_LDO_ADD:   OS << "%tldo_add(";   break;
  case VK_Sparc_TLS_IE_HI22:   OS << "%tie_hi22(";   break;
  case VK_Sparc_TLS_IE_LO10:   OS << "%tie_lo10(";   break;
  case VK_Sparc_TLS_IE_LD:     OS << "%tie_ld(";     break;
  case VK_Sparc_TLS_IE_LDX:    OS << "%tie_ldx(";    break;
  case VK_Sparc_TLS_IE_ADD:    OS << "%tie_add(";    break;
  case VK_Sparc_TLS_LE_HIX22:  OS << "%tle_hix22(";  break;
  case VK_Sparc_TLS_LE_LOX10:  OS << "%tle_lox10(";  break;
  }

  const MCExpr *Expr = getSubExpr();
  Expr->print(OS);
  if (closeParen)
    OS << ')';
}

SparcMCExpr::VariantKind SparcMCExpr::parseVariantKind(StringRef name)
{
  return StringSwitch<SparcMCExpr::VariantKind>(name)
    .Case("lo",  VK_Sparc_LO)
    .Case("hi",  VK_Sparc_HI)
    .Case("h44", VK_Sparc_H44)
    .Case("m44", VK_Sparc_M44)
    .Case("l44", VK_Sparc_L44)
    .Case("hh",  VK_Sparc_HH)
    .Case("hm",  VK_Sparc_HM)
    .Case("tgd_hi22",   VK_Sparc_TLS_GD_HI22)
    .Case("tgd_lo10",   VK_Sparc_TLS_GD_LO10)
    .Case("tgd_add",    VK_Sparc_TLS_GD_ADD)
    .Case("tgd_call",   VK_Sparc_TLS_GD_CALL)
    .Case("tldm_hi22",  VK_Sparc_TLS_LDM_HI22)
    .Case("tldm_lo10",  VK_Sparc_TLS_LDM_LO10)
    .Case("tldm_add",   VK_Sparc_TLS_LDM_ADD)
    .Case("tldm_call",  VK_Sparc_TLS_LDM_CALL)
    .Case("tldo_hix22", VK_Sparc_TLS_LDO_HIX22)
    .Case("tldo_lox10", VK_Sparc_TLS_LDO_LOX10)
    .Case("tldo_add",   VK_Sparc_TLS_LDO_ADD)
    .Case("tie_hi22",   VK_Sparc_TLS_IE_HI22)
    .Case("tie_lo10",   VK_Sparc_TLS_IE_LO10)
    .Case("tie_ld",     VK_Sparc_TLS_IE_LD)
    .Case("tie_ldx",    VK_Sparc_TLS_IE_LDX)
    .Case("tie_add",    VK_Sparc_TLS_IE_ADD)
    .Case("tle_hix22",  VK_Sparc_TLS_LE_HIX22)
    .Case("tle_lox10",  VK_Sparc_TLS_LE_LOX10)
    .Default(VK_Sparc_None);
}

bool
SparcMCExpr::EvaluateAsRelocatableImpl(MCValue &Res,
                                         const MCAsmLayout *Layout) const {
  return getSubExpr()->EvaluateAsRelocatable(Res, *Layout);
}

static void fixELFSymbolsInTLSFixupsImpl(const MCExpr *Expr, MCAssembler &Asm) {
  assert(0 && "Implement fixELFSymbolsInTLSFixupsImpl!");
}

void SparcMCExpr::fixELFSymbolsInTLSFixups(MCAssembler &Asm) const {
  switch(getKind()) {
  default: return;
  case VK_Sparc_TLS_GD_HI22:
  case VK_Sparc_TLS_GD_LO10:
  case VK_Sparc_TLS_GD_ADD:
  case VK_Sparc_TLS_GD_CALL:
  case VK_Sparc_TLS_LDM_HI22:
  case VK_Sparc_TLS_LDM_LO10:
  case VK_Sparc_TLS_LDM_ADD:
  case VK_Sparc_TLS_LDM_CALL:
  case VK_Sparc_TLS_LDO_HIX22:
  case VK_Sparc_TLS_LDO_LOX10:
  case VK_Sparc_TLS_LDO_ADD:
  case VK_Sparc_TLS_IE_HI22:
  case VK_Sparc_TLS_IE_LO10:
  case VK_Sparc_TLS_IE_LD:
  case VK_Sparc_TLS_IE_LDX:
  case VK_Sparc_TLS_IE_ADD:
  case VK_Sparc_TLS_LE_HIX22:
  case VK_Sparc_TLS_LE_LOX10: break;
  }
  fixELFSymbolsInTLSFixupsImpl(getSubExpr(), Asm);
}

// FIXME: This basically copies MCObjectStreamer::AddValueSymbols. Perhaps
// that method should be made public?
// FIXME: really do above: now that at least three other backends are using it.
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

void SparcMCExpr::AddValueSymbols(MCAssembler *Asm) const {
  AddValueSymbolsImpl(getSubExpr(), Asm);
}
