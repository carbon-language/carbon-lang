//===- AsmExpr.cpp - Assembly file expressions ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AsmExpr.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCValue.h"
using namespace llvm;

AsmExpr::~AsmExpr() {
}

bool AsmExpr::EvaluateAsAbsolute(MCContext &Ctx, int64_t &Res) const {
  switch (getKind()) {
  default:
    assert(0 && "Invalid assembly expression kind!");

  case Constant:
    Res = cast<AsmConstantExpr>(this)->getValue();
    return true;

  case SymbolRef: {
    MCSymbol *Sym = cast<AsmSymbolRefExpr>(this)->getSymbol();
    const MCValue *Value = Ctx.GetSymbolValue(Sym);

    // FIXME: Return more information about the failure.
    if (!Value || !Value->isConstant())
      return false;

    Res = Value->getConstant();
    return true;
  }

  case Unary: {
    const AsmUnaryExpr *AUE = cast<AsmUnaryExpr>(this);
    int64_t Value;

    if (!AUE->getSubExpr()->EvaluateAsAbsolute(Ctx, Value))
      return false;

    switch (AUE->getOpcode()) {
    case AsmUnaryExpr::LNot:  Res = !Value; break;
    case AsmUnaryExpr::Minus: Res = -Value; break;
    case AsmUnaryExpr::Not:   Res = ~Value; break;
    case AsmUnaryExpr::Plus:  Res = +Value; break;
    }

    return true;
  }

  case Binary: {
    const AsmBinaryExpr *ABE = cast<AsmBinaryExpr>(this);
    int64_t LHS, RHS;
    
    if (!ABE->getLHS()->EvaluateAsAbsolute(Ctx, LHS) ||
        !ABE->getRHS()->EvaluateAsAbsolute(Ctx, RHS))
      return false;

    // FIXME: We need target hooks for the evaluation. It may be limited in
    // width, and gas defines the result of comparisons differently from Apple
    // as (the result is sign extended).
    switch (ABE->getOpcode()) {
    case AsmBinaryExpr::Add:  Res = LHS + RHS; break;
    case AsmBinaryExpr::And:  Res = LHS & RHS; break;
    case AsmBinaryExpr::Div:  Res = LHS / RHS; break;
    case AsmBinaryExpr::EQ:   Res = LHS == RHS; break;
    case AsmBinaryExpr::GT:   Res = LHS > RHS; break;
    case AsmBinaryExpr::GTE:  Res = LHS >= RHS; break;
    case AsmBinaryExpr::LAnd: Res = LHS && RHS; break;
    case AsmBinaryExpr::LOr:  Res = LHS || RHS; break;
    case AsmBinaryExpr::LT:   Res = LHS < RHS; break;
    case AsmBinaryExpr::LTE:  Res = LHS <= RHS; break;
    case AsmBinaryExpr::Mod:  Res = LHS % RHS; break;
    case AsmBinaryExpr::Mul:  Res = LHS * RHS; break;
    case AsmBinaryExpr::NE:   Res = LHS != RHS; break;
    case AsmBinaryExpr::Or:   Res = LHS | RHS; break;
    case AsmBinaryExpr::Shl:  Res = LHS << RHS; break;
    case AsmBinaryExpr::Shr:  Res = LHS >> RHS; break;
    case AsmBinaryExpr::Sub:  Res = LHS - RHS; break;
    case AsmBinaryExpr::Xor:  Res = LHS ^ RHS; break;
    }

    return true;
  }
  }
}

