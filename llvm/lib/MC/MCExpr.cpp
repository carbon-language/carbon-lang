//===- MCExpr.cpp - Assembly Level Expression Implementation --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

void MCExpr::print(raw_ostream &OS, const MCAsmInfo *MAI) const {
  switch (getKind()) {
  case MCExpr::Constant:
    OS << cast<MCConstantExpr>(*this).getValue();
    return;

  case MCExpr::SymbolRef: {
    const MCSymbol &Sym = cast<MCSymbolRefExpr>(*this).getSymbol();
    
    // Parenthesize names that start with $ so that they don't look like
    // absolute names.
    if (Sym.getName()[0] == '$') {
      OS << '(';
      Sym.print(OS, MAI);
      OS << ')';
    } else {
      Sym.print(OS, MAI);
    }
    return;
  }

  case MCExpr::Unary: {
    const MCUnaryExpr &UE = cast<MCUnaryExpr>(*this);
    switch (UE.getOpcode()) {
    default: assert(0 && "Invalid opcode!");
    case MCUnaryExpr::LNot:  OS << '!'; break;
    case MCUnaryExpr::Minus: OS << '-'; break;
    case MCUnaryExpr::Not:   OS << '~'; break;
    case MCUnaryExpr::Plus:  OS << '+'; break;
    }
    UE.getSubExpr()->print(OS, MAI);
    return;
  }

  case MCExpr::Binary: {
    const MCBinaryExpr &BE = cast<MCBinaryExpr>(*this);
    
    // Only print parens around the LHS if it is non-trivial.
    if (isa<MCConstantExpr>(BE.getLHS()) || isa<MCSymbolRefExpr>(BE.getLHS())) {
      BE.getLHS()->print(OS, MAI);
    } else {
      OS << '(';
      BE.getLHS()->print(OS, MAI);
      OS << ')';
    }
    
    switch (BE.getOpcode()) {
    default: assert(0 && "Invalid opcode!");
    case MCBinaryExpr::Add:
      // Print "X-42" instead of "X+-42".
      if (const MCConstantExpr *RHSC = dyn_cast<MCConstantExpr>(BE.getRHS())) {
        if (RHSC->getValue() < 0) {
          OS << RHSC->getValue();
          return;
        }
      }
        
      OS <<  '+';
      break;
    case MCBinaryExpr::And:  OS <<  '&'; break;
    case MCBinaryExpr::Div:  OS <<  '/'; break;
    case MCBinaryExpr::EQ:   OS << "=="; break;
    case MCBinaryExpr::GT:   OS <<  '>'; break;
    case MCBinaryExpr::GTE:  OS << ">="; break;
    case MCBinaryExpr::LAnd: OS << "&&"; break;
    case MCBinaryExpr::LOr:  OS << "||"; break;
    case MCBinaryExpr::LT:   OS <<  '<'; break;
    case MCBinaryExpr::LTE:  OS << "<="; break;
    case MCBinaryExpr::Mod:  OS <<  '%'; break;
    case MCBinaryExpr::Mul:  OS <<  '*'; break;
    case MCBinaryExpr::NE:   OS << "!="; break;
    case MCBinaryExpr::Or:   OS <<  '|'; break;
    case MCBinaryExpr::Shl:  OS << "<<"; break;
    case MCBinaryExpr::Shr:  OS << ">>"; break;
    case MCBinaryExpr::Sub:  OS <<  '-'; break;
    case MCBinaryExpr::Xor:  OS <<  '^'; break;
    }
    
    // Only print parens around the LHS if it is non-trivial.
    if (isa<MCConstantExpr>(BE.getRHS()) || isa<MCSymbolRefExpr>(BE.getRHS())) {
      BE.getRHS()->print(OS, MAI);
    } else {
      OS << '(';
      BE.getRHS()->print(OS, MAI);
      OS << ')';
    }
    return;
  }
  }

  assert(0 && "Invalid expression kind!");
}

void MCExpr::dump() const {
  print(errs(), 0);
  errs() << '\n';
}

/* *** */

const MCBinaryExpr *MCBinaryExpr::Create(Opcode Opc, const MCExpr *LHS,
                                         const MCExpr *RHS, MCContext &Ctx) {
  return new (Ctx) MCBinaryExpr(Opc, LHS, RHS);
}

const MCUnaryExpr *MCUnaryExpr::Create(Opcode Opc, const MCExpr *Expr,
                                       MCContext &Ctx) {
  return new (Ctx) MCUnaryExpr(Opc, Expr);
}

const MCConstantExpr *MCConstantExpr::Create(int64_t Value, MCContext &Ctx) {
  return new (Ctx) MCConstantExpr(Value);
}

const MCSymbolRefExpr *MCSymbolRefExpr::Create(const MCSymbol *Sym,
                                               MCContext &Ctx) {
  return new (Ctx) MCSymbolRefExpr(Sym);
}

const MCSymbolRefExpr *MCSymbolRefExpr::Create(const StringRef &Name,
                                               MCContext &Ctx) {
  return Create(Ctx.GetOrCreateSymbol(Name), Ctx);
}


/* *** */

bool MCExpr::EvaluateAsAbsolute(MCContext &Ctx, int64_t &Res) const {
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

bool MCExpr::EvaluateAsRelocatable(MCContext &Ctx, MCValue &Res) const {
  switch (getKind()) {
  case Constant:
    Res = MCValue::get(cast<MCConstantExpr>(this)->getValue());
    return true;

  case SymbolRef: {
    const MCSymbol &Sym = cast<MCSymbolRefExpr>(this)->getSymbol();

    // Evaluate recursively if this is a variable.
    if (Sym.isVariable())
      return Sym.getValue()->EvaluateAsRelocatable(Ctx, Res);

    Res = MCValue::get(&Sym, 0, 0);
    return true;
  }

  case Unary: {
    const MCUnaryExpr *AUE = cast<MCUnaryExpr>(this);
    MCValue Value;

    if (!AUE->getSubExpr()->EvaluateAsRelocatable(Ctx, Value))
      return false;

    switch (AUE->getOpcode()) {
    case MCUnaryExpr::LNot:
      if (!Value.isAbsolute())
        return false;
      Res = MCValue::get(!Value.getConstant());
      break;
    case MCUnaryExpr::Minus:
      /// -(a - b + const) ==> (b - a - const)
      if (Value.getSymA() && !Value.getSymB())
        return false;
      Res = MCValue::get(Value.getSymB(), Value.getSymA(), 
                         -Value.getConstant()); 
      break;
    case MCUnaryExpr::Not:
      if (!Value.isAbsolute())
        return false;
      Res = MCValue::get(~Value.getConstant()); 
      break;
    case MCUnaryExpr::Plus:
      Res = Value;
      break;
    }

    return true;
  }

  case Binary: {
    const MCBinaryExpr *ABE = cast<MCBinaryExpr>(this);
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
      case MCBinaryExpr::Sub:
        // Negate RHS and add.
        return EvaluateSymbolicAdd(LHSValue,
                                   RHSValue.getSymB(), RHSValue.getSymA(),
                                   -RHSValue.getConstant(),
                                   Res);

      case MCBinaryExpr::Add:
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
    case MCBinaryExpr::Add:  Result = LHS + RHS; break;
    case MCBinaryExpr::And:  Result = LHS & RHS; break;
    case MCBinaryExpr::Div:  Result = LHS / RHS; break;
    case MCBinaryExpr::EQ:   Result = LHS == RHS; break;
    case MCBinaryExpr::GT:   Result = LHS > RHS; break;
    case MCBinaryExpr::GTE:  Result = LHS >= RHS; break;
    case MCBinaryExpr::LAnd: Result = LHS && RHS; break;
    case MCBinaryExpr::LOr:  Result = LHS || RHS; break;
    case MCBinaryExpr::LT:   Result = LHS < RHS; break;
    case MCBinaryExpr::LTE:  Result = LHS <= RHS; break;
    case MCBinaryExpr::Mod:  Result = LHS % RHS; break;
    case MCBinaryExpr::Mul:  Result = LHS * RHS; break;
    case MCBinaryExpr::NE:   Result = LHS != RHS; break;
    case MCBinaryExpr::Or:   Result = LHS | RHS; break;
    case MCBinaryExpr::Shl:  Result = LHS << RHS; break;
    case MCBinaryExpr::Shr:  Result = LHS >> RHS; break;
    case MCBinaryExpr::Sub:  Result = LHS - RHS; break;
    case MCBinaryExpr::Xor:  Result = LHS ^ RHS; break;
    }

    Res = MCValue::get(Result);
    return true;
  }
  }

  assert(0 && "Invalid assembly expression kind!");
  return false;
}
