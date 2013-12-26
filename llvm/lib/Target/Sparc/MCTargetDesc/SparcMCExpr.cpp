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

bool
SparcMCExpr::EvaluateAsRelocatableImpl(MCValue &Res,
                                         const MCAsmLayout *Layout) const {
  assert(0 && "FIXME: Implement SparcMCExpr::EvaluateAsRelocatableImpl");
  return getSubExpr()->EvaluateAsRelocatable(Res, *Layout);
}


void SparcMCExpr::fixELFSymbolsInTLSFixups(MCAssembler &Asm) const {
  assert(0 && "FIXME: Implement SparcMCExpr::fixELFSymbolsInTLSFixups");
}

void SparcMCExpr::AddValueSymbols(MCAssembler *Asm) const {
  assert(0 && "FIXME: Implement SparcMCExpr::AddValueSymbols");
}
