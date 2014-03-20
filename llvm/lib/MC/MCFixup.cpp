//===- MCFixup.cpp - Assembly Fixup Implementation ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCFixup.h"
using namespace llvm;

static MCSymbolRefExpr::VariantKind getAccessVariant(const MCExpr *Expr) {
  switch (Expr->getKind()) {
  case MCExpr::Unary:
  case MCExpr::Target:
    llvm_unreachable("unsupported");

  case MCExpr::Constant:
    return MCSymbolRefExpr::VK_None;

  case MCExpr::SymbolRef: {
    const MCSymbolRefExpr *SRE = cast<MCSymbolRefExpr>(Expr);
    return SRE->getKind();
  }
  case MCExpr::Binary: {
    const MCBinaryExpr *ABE = cast<MCBinaryExpr>(Expr);
    assert(getAccessVariant(ABE->getRHS()) == MCSymbolRefExpr::VK_None);
    return getAccessVariant(ABE->getLHS());
  }
  }
  llvm_unreachable("unknown MCExpr kind");
}

MCSymbolRefExpr::VariantKind MCFixup::getAccessVariant() const {
  return ::getAccessVariant(getValue());
}
