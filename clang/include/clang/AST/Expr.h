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

#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/AST/Decl.h"

namespace llvm {
namespace clang {
  class IdentifierInfo;
  class Decl;
  
/// Expr - This represents one expression.  Note that Expr's are subclasses of
/// Stmt.  This allows an expression to be transparently used any place a Stmt
/// is required.
///
class Expr : public Stmt {
  TypeRef TR;
protected:
  Expr(StmtClass SC, TypeRef T=0) : Stmt(SC), TR(T) {}
  ~Expr() {}
public:  
  TypeRef getType() const { return TR; }
  
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() >= firstExprConstant &&
           T->getStmtClass() <= lastExprConstant; 
  }
  static bool classof(const Expr *) { return true; }
};

//===----------------------------------------------------------------------===//
// Primary Expressions.
//===----------------------------------------------------------------------===//

/// DeclRefExpr - [C99 6.5.1p2] - A reference to a declared variable, function,
/// enum, etc.
class DeclRefExpr : public Expr {
  Decl *D; // a ValueDecl or EnumConstantDecl
public:
  DeclRefExpr(Decl *d, TypeRef t) : Expr(DeclRefExprClass, t), D(d) {}
  
  Decl *getDecl() const { return D; }
  
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == DeclRefExprClass; 
  }
  static bool classof(const DeclRefExpr *) { return true; }
};

class IntegerLiteral : public Expr {
  intmax_t Value;
public:
  // type should be IntTy, LongTy, LongLongTy, UnsignedIntTy, UnsignedLongTy, 
  // or UnsignedLongLongTy
  IntegerLiteral(intmax_t value, TypeRef type)
    : Expr(IntegerLiteralClass, type), Value(value) {
    assert(type->isIntegralType() && "Illegal type in IntegerLiteral");
  }
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == IntegerLiteralClass; 
  }
  static bool classof(const IntegerLiteral *) { return true; }
};

class FloatingLiteral : public Expr {
  float Value; // FIXME
public:
  FloatingLiteral(float value, TypeRef type) : 
    Expr(FloatingLiteralClass, type), Value(value) {} 
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == FloatingLiteralClass; 
  }
  static bool classof(const FloatingLiteral *) { return true; }
};

class StringLiteral : public Expr {
  const char *StrData;
  unsigned ByteLength;
  bool IsWide;
public:
  StringLiteral(const char *strData, unsigned byteLength, bool Wide, TypeRef t);
  virtual ~StringLiteral();
  
  const char *getStrData() const { return StrData; }
  unsigned getByteLength() const { return ByteLength; }
  bool isWide() const { return IsWide; }
  
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == StringLiteralClass; 
  }
  static bool classof(const StringLiteral *) { return true; }
};

/// ParenExpr - This represents a parethesized expression, e.g. "(1)".  This
/// AST node is only formed if full location information is requested.
class ParenExpr : public Expr {
  SourceLocation L, R;
  Expr *Val;
public:
  ParenExpr(SourceLocation l, SourceLocation r, Expr *val)
    : Expr(ParenExprClass), L(l), R(r), Val(val) {}
  
  Expr *getSubExpr() { return Val; }
  
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ParenExprClass; 
  }
  static bool classof(const ParenExpr *) { return true; }
};


/// UnaryOperator - This represents the unary-expression's (except sizeof of
/// types), the postinc/postdec operators from postfix-expression, and various
/// extensions.
class UnaryOperator : public Expr {
public:
  enum Opcode {
    PostInc, PostDec, // [C99 6.5.2.4] Postfix increment and decrement operators
    PreInc, PreDec,   // [C99 6.5.3.1] Prefix increment and decrement operators.
    AddrOf, Deref,    // [C99 6.5.3.2] Address and indirection operators.
    Plus, Minus,      // [C99 6.5.3.3] Unary arithmetic operators.
    Not, LNot,        // [C99 6.5.3.3] Unary arithmetic operators.
    SizeOf, AlignOf,  // [C99 6.5.3.4] Sizeof (expr, not type) operator.
    Real, Imag,       // "__real expr"/"__imag expr" Extension.
    AddrLabel,        // && label Extension.
    Extension         // __extension__ marker.
  };

  UnaryOperator(Expr *input, Opcode opc, TypeRef type)
    : Expr(UnaryOperatorClass, type), Val(input), Opc(opc) {}
  
  /// getOpcodeStr - Turn an Opcode enum value into the punctuation char it
  /// corresponds to, e.g. "sizeof" or "[pre]++"
  static const char *getOpcodeStr(Opcode Op);

  /// isPostfix - Return true if this is a postfix operation, like x++.
  static bool isPostfix(Opcode Op);

  
  Opcode getOpcode() const { return Opc; }
  Expr *getSubExpr() { return Val; }
  
  bool isPostfix() const { return isPostfix(Opc); }
  
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == UnaryOperatorClass; 
  }
  static bool classof(const UnaryOperator *) { return true; }
  
private:
  Expr *Val;
  Opcode Opc;
};

/// SizeOfAlignOfTypeExpr - [C99 6.5.3.4] - This is only for sizeof/alignof of
/// *types*.  sizeof(expr) is handled by UnaryOperator.
class SizeOfAlignOfTypeExpr : public Expr {
  bool isSizeof;  // true if sizeof, false if alignof.
  TypeRef Ty;
public:
  SizeOfAlignOfTypeExpr(bool issizeof, TypeRef argType, TypeRef resultType) : 
    Expr(SizeOfAlignOfTypeExprClass, resultType),
    isSizeof(issizeof), Ty(argType) {}
  
  bool isSizeOf() const { return isSizeof; }
  TypeRef getArgumentType() const { return Ty; }

  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == SizeOfAlignOfTypeExprClass; 
  }
  static bool classof(const SizeOfAlignOfTypeExpr *) { return true; }
};

//===----------------------------------------------------------------------===//
// Postfix Operators.
//===----------------------------------------------------------------------===//

/// ArraySubscriptExpr - [C99 6.5.2.1] Array Subscripting.
class ArraySubscriptExpr : public Expr {
  Expr *Base, *Idx;
public:
  ArraySubscriptExpr(Expr *base, Expr *idx, TypeRef t) : 
    Expr(ArraySubscriptExprClass, t),
    Base(base), Idx(idx) {}
  
  Expr *getBase() { return Base; }
  Expr *getIdx() { return Idx; }
  
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ArraySubscriptExprClass; 
  }
  static bool classof(const ArraySubscriptExpr *) { return true; }
};


/// CallExpr - [C99 6.5.2.2] Function Calls.
///
class CallExpr : public Expr {
  Expr *Fn;
  Expr **Args;
  unsigned NumArgs;
public:
  CallExpr(Expr *fn, Expr **args, unsigned numargs);
  ~CallExpr() {
    delete [] Args;
  }
  
  Expr *getCallee() { return Fn; }
  
  /// getNumArgs - Return the number of actual arguments to this call.
  ///
  unsigned getNumArgs() const { return NumArgs; }
  
  /// getArg - Return the specified argument.
  Expr *getArg(unsigned Arg) const {
    assert(Arg < NumArgs && "Arg access out of range!");
    return Args[Arg];
  }
  
  /// getNumCommas - Return the number of commas that must have been present in
  /// this function call.
  unsigned getNumCommas() const { return NumArgs ? NumArgs - 1 : 0; }
  
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == CallExprClass; 
  }
  static bool classof(const CallExpr *) { return true; }
};

/// MemberExpr - [C99 6.5.2.3] Structure and Union Members.
///
class MemberExpr : public Expr {
  Expr *Base;
  FieldDecl *MemberDecl;
  bool IsArrow;      // True if this is "X->F", false if this is "X.F".
public:
  MemberExpr(Expr *base, bool isarrow, FieldDecl *memberdecl) 
    : Expr(MemberExprClass, memberdecl->getType()),
      Base(base), MemberDecl(memberdecl), IsArrow(isarrow) {}
  
  Expr *getBase() { return Base; }
  FieldDecl *getMemberDecl() { return MemberDecl; }
  bool isArrow() const { return IsArrow; }
  
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == MemberExprClass; 
  }
  static bool classof(const MemberExpr *) { return true; }
};

/// CastExpr - [C99 6.5.4] Cast Operators.
///
class CastExpr : public Expr {
  TypeRef Ty;
  Expr *Op;
public:
  CastExpr(TypeRef ty, Expr *op) : Expr(CastExprClass), Ty(ty), Op(op) {}
  CastExpr(StmtClass SC, TypeRef ty, Expr *op) : Expr(SC), Ty(ty), Op(op) {}
  
  TypeRef getDestType() const { return Ty; }
  
  Expr *getSubExpr() { return Op; }
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == CastExprClass; 
  }
  static bool classof(const CastExpr *) { return true; }
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
    : Expr(BinaryOperatorClass), LHS(lhs), RHS(rhs), Opc(opc) {}

  /// getOpcodeStr - Turn an Opcode enum value into the punctuation char it
  /// corresponds to, e.g. "<<=".
  static const char *getOpcodeStr(Opcode Op);

  /// predicates to categorize the respective opcodes.
  static bool isMultiplicativeOp(Opcode Op) { return Op >= Mul && Op <= Rem; }
  static bool isAdditiveOp(Opcode Op) { return Op == Add || Op == Sub; }
  static bool isShiftOp(Opcode Op) { return Op == Shl || Op == Shr; }
  static bool isRelationalOp(Opcode Op) { return Op >= LT && Op <= GE; }
  static bool isEqualityOp(Opcode Op) { return Op == EQ || Op == NE; }
  static bool isBitwiseOp(Opcode Op) { return Op >= And && Op <= Or; }
  static bool isLogicalOp(Opcode Op) { return Op == LAnd || Op == LOr; }
  
  Opcode getOpcode() const { return Opc; }
  Expr *getLHS() { return LHS; }
  Expr *getRHS() { return RHS; }
  
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == BinaryOperatorClass; 
  }
  static bool classof(const BinaryOperator *) { return true; }
private:
  Expr *LHS, *RHS;
  Opcode Opc;
};

/// ConditionalOperator - The ?: operator.  Note that LHS may be null when the
/// GNU "missing LHS" extension is in use.
///
class ConditionalOperator : public Expr {
  Expr *Cond, *LHS, *RHS;  // Left/Middle/Right hand sides.
public:
  ConditionalOperator(Expr *cond, Expr *lhs, Expr *rhs)
    : Expr(ConditionalOperatorClass), Cond(cond), LHS(lhs), RHS(rhs) {}

  Expr *getCond() { return Cond; }
  Expr *getLHS() { return LHS; }
  Expr *getRHS() { return RHS; }
  
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ConditionalOperatorClass; 
  }
  static bool classof(const ConditionalOperator *) { return true; }
};

  
}  // end namespace clang
}  // end namespace llvm

#endif
