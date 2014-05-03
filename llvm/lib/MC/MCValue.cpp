//===- lib/MC/MCValue.cpp - MCValue implementation ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCValue.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

void MCValue::print(raw_ostream &OS, const MCAsmInfo *MAI) const {
  if (isAbsolute()) {
    OS << getConstant();
    return;
  }

  // FIXME: prints as a number, which isn't ideal. But the meaning will be
  // target-specific anyway.
  if (getRefKind())
    OS << ':' << getRefKind() <<  ':';

  getSymA()->print(OS);

  if (getSymB()) {
    OS << " - ";
    getSymB()->print(OS);
  }

  if (getConstant())
    OS << " + " << getConstant();
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void MCValue::dump() const {
  print(dbgs(), nullptr);
}
#endif

MCSymbolRefExpr::VariantKind MCValue::getAccessVariant() const {
  const MCSymbolRefExpr *B = getSymB();
  if (B) {
    if (B->getKind() != MCSymbolRefExpr::VK_None)
      llvm_unreachable("unsupported");
  }

  const MCSymbolRefExpr *A = getSymA();
  if (!A)
    return MCSymbolRefExpr::VK_None;

  MCSymbolRefExpr::VariantKind Kind = A->getKind();
  if (Kind == MCSymbolRefExpr::VK_WEAKREF)
    return MCSymbolRefExpr::VK_None;
  return Kind;
}
