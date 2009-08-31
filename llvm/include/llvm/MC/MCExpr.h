//===- MCExpr.h - Assembly Level Expressions --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCEXPR_H
#define LLVM_MC_MCEXPR_H

#include "llvm/Support/Casting.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {
class MCContext;
class MCSymbol;
class MCValue;

/// MCExpr - Base class for the full range of assembler expressions which are
/// needed for parsing.  
class MCExpr {
public:
  enum ExprKind {
    Binary,    ///< Binary expressions.
    Constant,  ///< Constant expressions.
    SymbolRef, ///< References to labels and assigned expressions.
    Unary      ///< Unary expressions.
  };
  
private:
  ExprKind Kind;
  
protected:
  MCExpr(ExprKind _Kind) : Kind(_Kind) {}
  
public:
  virtual ~MCExpr();

  ExprKind getKind() const { return Kind; }

  /// EvaluateAsAbsolute - Try to evaluate the expression to an absolute value.
  ///
  /// @param Res - The absolute value, if evaluation succeeds.
  /// @result - True on success.
  bool EvaluateAsAbsolute(MCContext &Ctx, int64_t &Res) const;

  /// EvaluateAsRelocatable - Try to evaluate the expression to a relocatable
  /// value, i.e. an expression of the fixed form (a - b + constant).
  ///
  /// @param Res - The relocatable value, if evaluation succeeds.
  /// @result - True on success.
  bool EvaluateAsRelocatable(MCContext &Ctx, MCValue &Res) const;

  static bool classof(const MCExpr *) { return true; }
};

//// MCConstantExpr - Represent a constant integer expression.
class MCConstantExpr : public MCExpr {
  int64_t Value;

public:
  MCConstantExpr(int64_t _Value) 
    : MCExpr(MCExpr::Constant), Value(_Value) {}
  
  int64_t getValue() const { return Value; }

  static bool classof(const MCExpr *E) { 
    return E->getKind() == MCExpr::Constant; 
  }
  static bool classof(const MCConstantExpr *) { return true; }
};

/// MCSymbolRefExpr - Represent a reference to a symbol from inside an
/// expression.
///
/// A symbol reference in an expression may be a use of a label, a use of an
/// assembler variable (defined constant), or constitute an implicit definition
/// of the symbol as external.
class MCSymbolRefExpr : public MCExpr {
  MCSymbol *Symbol;

public:
  MCSymbolRefExpr(MCSymbol *_Symbol) 
    : MCExpr(MCExpr::SymbolRef), Symbol(_Symbol) {}
  
  MCSymbol *getSymbol() const { return Symbol; }

  static bool classof(const MCExpr *E) { 
    return E->getKind() == MCExpr::SymbolRef; 
  }
  static bool classof(const MCSymbolRefExpr *) { return true; }
};

/// MCUnaryExpr - Unary assembler expressions.
class MCUnaryExpr : public MCExpr {
public:
  enum Opcode {
    LNot,  ///< Logical negation.
    Minus, ///< Unary minus.
    Not,   ///< Bitwise negation.
    Plus   ///< Unary plus.
  };

private:
  Opcode Op;
  MCExpr *Expr;

public:
  MCUnaryExpr(Opcode _Op, MCExpr *_Expr)
    : MCExpr(MCExpr::Unary), Op(_Op), Expr(_Expr) {}
  ~MCUnaryExpr() {
    delete Expr;
  }

  Opcode getOpcode() const { return Op; }

  MCExpr *getSubExpr() const { return Expr; }

  static bool classof(const MCExpr *E) { 
    return E->getKind() == MCExpr::Unary; 
  }
  static bool classof(const MCUnaryExpr *) { return true; }
};

/// MCBinaryExpr - Binary assembler expressions.
class MCBinaryExpr : public MCExpr {
public:
  enum Opcode {
    Add,  ///< Addition.
    And,  ///< Bitwise and.
    Div,  ///< Division.
    EQ,   ///< Equality comparison.
    GT,   ///< Greater than comparison.
    GTE,  ///< Greater than or equal comparison.
    LAnd, ///< Logical and.
    LOr,  ///< Logical or.
    LT,   ///< Less than comparison.
    LTE,  ///< Less than or equal comparison.
    Mod,  ///< Modulus.
    Mul,  ///< Multiplication.
    NE,   ///< Inequality comparison.
    Or,   ///< Bitwise or.
    Shl,  ///< Bitwise shift left.
    Shr,  ///< Bitwise shift right.
    Sub,  ///< Subtraction.
    Xor   ///< Bitwise exclusive or.
  };

private:
  Opcode Op;
  MCExpr *LHS, *RHS;

public:
  MCBinaryExpr(Opcode _Op, MCExpr *_LHS, MCExpr *_RHS)
    : MCExpr(MCExpr::Binary), Op(_Op), LHS(_LHS), RHS(_RHS) {}
  ~MCBinaryExpr() {
    delete LHS;
    delete RHS;
  }

  Opcode getOpcode() const { return Op; }

  /// getLHS - Get the left-hand side expression of the binary operator.
  MCExpr *getLHS() const { return LHS; }

  /// getRHS - Get the right-hand side expression of the binary operator.
  MCExpr *getRHS() const { return RHS; }

  static bool classof(const MCExpr *E) { 
    return E->getKind() == MCExpr::Binary; 
  }
  static bool classof(const MCBinaryExpr *) { return true; }
};

} // end namespace llvm

#endif
