//===- X86MCTargetExpr.cpp - X86 Target Specific MCExpr Implementation ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86MCTargetExpr.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

X86MCTargetExpr *X86MCTargetExpr::Create(const MCSymbol *Sym, VariantKind K,
                                         MCContext &Ctx) {
  return new (Ctx) X86MCTargetExpr(Sym, K);
}

void X86MCTargetExpr::PrintImpl(raw_ostream &OS) const {
  OS << *Sym;
  
  switch (Kind) {
  case GOT: OS << "@GOT"; break;
  case PLT: OS << "@PLT"; break;
  case GOTPCREL: OS << "@GOTPCREL"; break;
  }
}

bool X86MCTargetExpr::EvaluateAsRelocatableImpl(MCValue &Res) const {
  // FIXME: I don't know if this is right, it followed MCSymbolRefExpr.
  
  // Evaluate recursively if this is a variable.
  if (Sym->isVariable())
    return Sym->getValue()->EvaluateAsRelocatable(Res);
  
  Res = MCValue::get(Sym, 0, 0);
  return true;
}

X86MCTargetExpr *foo(MCExpr *A) { return (X86MCTargetExpr*)A; }