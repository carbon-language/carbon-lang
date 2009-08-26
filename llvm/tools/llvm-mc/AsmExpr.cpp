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
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
using namespace llvm;

AsmExpr::~AsmExpr() {
}

bool AsmExpr::EvaluateAsAbsolute(MCContext &Ctx, int64_t &Res) const {
  MCValue Value;
  
  if (!EvaluateAsRelocatable(Ctx, Value) || !Value.isAbsolute())
    return false;

  Res = Value.getConstant();
  return true;
}

static bool EvaluateSymbolicAdd(const MCValue &LHS, const MCSymbol *RHS_A, 
                                const MCSymbol *RHS_B, int64_t RHS_Cst,
                                MCValue &Res) {
  // We can't add or subtract two symbols.
  if ((LHS.getSymA() && RHS_A) ||
      (LHS.getSymB() && RHS_B))
    return false;

  const MCSymbol *A = LHS.getSymA() ? LHS.getSymA() : RHS_A;
  const MCSymbol *B = LHS.getSymB() ? LHS.getSymB() : RHS_B;
  if (B) {
    // If we have a negated symbol, then we must have also have a non-negated
    // symbol in order to encode the expression. We can do this check later to
    // permit expressions which eventually fold to a representable form -- such
    // as (a + (0 - b)) -- if necessary.
    if (!A)
      return false;
  }
  Res = MCValue::get(A, B, LHS.getConstant() + RHS_Cst);
  return true;
}

bool AsmExpr::EvaluateAsRelocatable(MCContext &Ctx, MCValue &Res) const {
  switch (getKind()) {
  default:
    assert(0 && "Invalid assembly expression kind!");

  case Constant:
    Res = MCValue::get(cast<AsmConstantExpr>(this)->getValue());
    return true;

  case SymbolRef: {
    MCSymbol *Sym = cast<AsmSymbolRefExpr>(this)->getSymbol();
    if (const MCValue *Value = Ctx.GetSymbolValue(Sym))
      Res = *Value;
    else
      Res = MCValue::get(Sym, 0, 0);
    return true;
  }

  case Unary: {
    const AsmUnaryExpr *AUE = cast<AsmUnaryExpr>(this);
    MCValue Value;

    if (!AUE->getSubExpr()->EvaluateAsRelocatable(Ctx, Value))
      return false;

    switch (AUE->getOpcode()) {
    case AsmUnaryExpr::LNot:
      if (!Value.isAbsolute())
        return false;
      Res = MCValue::get(!Value.getConstant());
      break;
    case AsmUnaryExpr::Minus:
      /// -(a - b + const) ==> (b - a - const)
      if (Value.getSymA() && !Value.getSymB())
        return false;
      Res = MCValue::get(Value.getSymB(), Value.getSymA(), 
                         -Value.getConstant()); 
      break;
    case AsmUnaryExpr::Not:
      if (!Value.isAbsolute())
        return false;
      Res = MCValue::get(~Value.getConstant()); 
      break;
    case AsmUnaryExpr::Plus:
      Res = Value;
      break;
    }

    return true;
  }

  case Binary: {
    const AsmBinaryExpr *ABE = cast<AsmBinaryExpr>(this);
    MCValue LHSValue, RHSValue;
    
    if (!ABE->getLHS()->EvaluateAsRelocatable(Ctx, LHSValue) ||
        !ABE->getRHS()->EvaluateAsRelocatable(Ctx, RHSValue))
      return false;

    // We only support a few operations on non-constant expressions, handle
    // those first.
    if (!LHSValue.isAbsolute() || !RHSValue.isAbsolute()) {
      switch (ABE->getOpcode()) {
      default:
        return false;
      case AsmBinaryExpr::Sub:
        // Negate RHS and add.
        return EvaluateSymbolicAdd(LHSValue,
                                   RHSValue.getSymB(), RHSValue.getSymA(),
                                   -RHSValue.getConstant(),
                                   Res);

      case AsmBinaryExpr::Add:
        return EvaluateSymbolicAdd(LHSValue,
                                   RHSValue.getSymA(), RHSValue.getSymB(),
                                   RHSValue.getConstant(),
                                   Res);
      }
    }

    // FIXME: We need target hooks for the evaluation. It may be limited in
    // width, and gas defines the result of comparisons differently from Apple
    // as (the result is sign extended).
    int64_t LHS = LHSValue.getConstant(), RHS = RHSValue.getConstant();
    int64_t Result = 0;
    switch (ABE->getOpcode()) {
    case AsmBinaryExpr::Add:  Result = LHS + RHS; break;
    case AsmBinaryExpr::And:  Result = LHS & RHS; break;
    case AsmBinaryExpr::Div:  Result = LHS / RHS; break;
    case AsmBinaryExpr::EQ:   Result = LHS == RHS; break;
    case AsmBinaryExpr::GT:   Result = LHS > RHS; break;
    case AsmBinaryExpr::GTE:  Result = LHS >= RHS; break;
    case AsmBinaryExpr::LAnd: Result = LHS && RHS; break;
    case AsmBinaryExpr::LOr:  Result = LHS || RHS; break;
    case AsmBinaryExpr::LT:   Result = LHS < RHS; break;
    case AsmBinaryExpr::LTE:  Result = LHS <= RHS; break;
    case AsmBinaryExpr::Mod:  Result = LHS % RHS; break;
    case AsmBinaryExpr::Mul:  Result = LHS * RHS; break;
    case AsmBinaryExpr::NE:   Result = LHS != RHS; break;
    case AsmBinaryExpr::Or:   Result = LHS | RHS; break;
    case AsmBinaryExpr::Shl:  Result = LHS << RHS; break;
    case AsmBinaryExpr::Shr:  Result = LHS >> RHS; break;
    case AsmBinaryExpr::Sub:  Result = LHS - RHS; break;
    case AsmBinaryExpr::Xor:  Result = LHS ^ RHS; break;
    }

    Res = MCValue::get(Result);
    return true;
  }
  }
}

