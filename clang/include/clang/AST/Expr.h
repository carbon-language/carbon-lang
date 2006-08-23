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
  
  // FIXME: Change to non-virtual method that uses visitor pattern to do this.
  void dump() const;
  
private:
  virtual void dump_impl() const = 0;
};

//===----------------------------------------------------------------------===//
// Primary Expressions.
//===----------------------------------------------------------------------===//

class IntegerConstant : public Expr {
public:
  IntegerConstant() {}
  virtual void dump_impl() const;
};

class FloatingConstant : public Expr {
public:
  FloatingConstant() {}
  virtual void dump_impl() const;
};

/// ParenExpr - This represents a parethesized expression, e.g. "(1)".  This
/// AST node is only formed if full location information is requested.
class ParenExpr : public Expr {
  SourceLocation L, R;
  Expr *Val;
public:
  ParenExpr(SourceLocation l, SourceLocation r, Expr *val)
    : L(l), R(r), Val(val) {}
  virtual void dump_impl() const;
};


/// UnaryOperator - This represents the unary-expression's (except sizeof), the
/// postinc/postdec operators from postfix-expression, and various extensions.
class UnaryOperator : public Expr {
public:
  enum Opcode {
    PostInc, PostDec, // [C99 6.5.2.4] Postfix increment and decrement operators
    PreInc, PreDec,   // [C99 6.5.3.1] Prefix increment and decrement operators.
    AddrOf, Deref,    // [C99 6.5.3.2] Address and indirection operators.
    Plus, Minus,      // [C99 6.5.3.3] Unary arithmetic operators.
    Not, LNot,        // [C99 6.5.3.3] Unary arithmetic operators.
    Real, Imag        // "__real expr"/"__imag expr" Extension.
  };

  UnaryOperator(Expr *input, Opcode opc)
    : Input(input), Opc(opc) {}
  
  /// getOpcodeStr - Turn an Opcode enum value into the punctuation char it
  /// corresponds to, e.g. "sizeof" or "[pre]++"
  static const char *getOpcodeStr(Opcode Op);
  
  virtual void dump_impl() const;
  
private:
  Expr *Input;
  Opcode Opc;
};

class UnaryOperatorLOC : public UnaryOperator {
  SourceLocation Loc;
public:
  UnaryOperatorLOC(SourceLocation loc, Expr *Input, Opcode Opc)
   : UnaryOperator(Input, Opc), Loc(loc) {}

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

  /// getOpcodeStr - Turn an Opcode enum value into the punctuation char it
  /// corresponds to, e.g. "<<=".
  static const char *getOpcodeStr(Opcode Op);
  
  virtual void dump_impl() const;

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
  virtual void dump_impl() const;
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
