//===--- Expr.h - Classes for representing expressions ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Expr interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_EXPR_H
#define LLVM_CLANG_AST_EXPR_H

#include "clang/Basic/SourceLocation.h"

namespace llvm {
namespace clang {
  
/// Expr - This represents one expression etc.  
///
class Expr {
  /// Type.
public:
  Expr() {}
  virtual ~Expr() {}
};

class IntegerConstant : public Expr {
public:
  IntegerConstant() {}
};

class FloatingConstant : public Expr {
public:
  FloatingConstant() {}
};

class BinaryOperator : public Expr {
public:
  enum Opcode {
    // Operators listed in order of precedence.
    Mul, Div, Rem,    // [C99 6.5.5] Multiplicative operators.
    Add, Sub,         // [C99 6.5.6] Additive operators.
    Shl, Shr,         // [C99 6.5.7] Bitwise shift operators.
    LT, GT, LE, GE,   // [C99 6.5.8] Relational operators.
    EQ, NE,           // [C99 6.5.9] Equality operators.
    And,              // [C99 6.5.10] Bitwise AND operator.
    Xor,              // [C99 6.5.11] Bitwise XOR operator.
    Or,               // [C99 6.5.12] Bitwise OR operator.
    LAnd,             // [C99 6.5.13] Logical AND operator.
    LOr,              // [C99 6.5.14] Logical OR operator.
    Assign, MulAssign,// [C99 6.5.16] Assignment operators.
    DivAssign, RemAssign,
    AddAssign, SubAssign,
    ShlAssign, ShrAssign,
    AndAssign, XorAssign,
    OrAssign,
    Comma             // [C99 6.5.17] Comma operator.
  };
  
  BinaryOperator(Expr *lhs, Expr *rhs, Opcode opc)
    : LHS(lhs), RHS(rhs), Opc(opc) {}

private:
  Expr *LHS, *RHS;
  Opcode Opc;
};

class BinaryOperatorLOC : public BinaryOperator {
  SourceLocation OperatorLoc;
public:
  BinaryOperatorLOC(Expr *LHS, SourceLocation OpLoc, Expr *RHS, Opcode Opc)
    : BinaryOperator(LHS, RHS, Opc), OperatorLoc(OpLoc) {
  }
};

/// ConditionalOperator - The ?: operator.  Note that LHS may be null when the
/// GNU "missing LHS" extension is in use.
///
class ConditionalOperator : public Expr {
  Expr *Cond, *LHS, *RHS;  // Left/Middle/Right hand sides.
public:
  ConditionalOperator(Expr *cond, Expr *lhs, Expr *rhs)
    : Cond(cond), LHS(lhs), RHS(rhs) {}
};

/// ConditionalOperatorLOC - ConditionalOperator with full location info.
///
class ConditionalOperatorLOC : public ConditionalOperator {
  SourceLocation QuestionLoc, ColonLoc;
public:
  ConditionalOperatorLOC(Expr *Cond, SourceLocation QLoc, Expr *LHS,
                         SourceLocation CLoc, Expr *RHS)
    : ConditionalOperator(Cond, LHS, RHS), QuestionLoc(QLoc), ColonLoc(CLoc) {}
};

  
}  // end namespace clang
}  // end namespace llvm

#endif
