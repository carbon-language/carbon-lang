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
#include "llvm/ADT/APInt.h"

namespace llvm {
namespace clang {
  class IdentifierInfo;
  class Decl;
  
/// Expr - This represents one expression.  Note that Expr's are subclasses of
/// Stmt.  This allows an expression to be transparently used any place a Stmt
/// is required.
///
class Expr : public Stmt {
  QualType TR;
protected:
  Expr(StmtClass SC, QualType T) : Stmt(SC), TR(T) {}
  ~Expr() {}
public:  
  QualType getType() const { return TR; }
  
  /// SourceLocation tokens are not useful in isolation - they are low level
  /// value objects created/interpreted by SourceManager. We assume AST
  /// clients will have a pointer to the respective SourceManager.
  virtual SourceRange getSourceRange() const = 0;
  SourceLocation getLocStart() const { return getSourceRange().Begin(); }
  SourceLocation getLocEnd() const { return getSourceRange().End(); }

  /// isLvalue - C99 6.3.2.1: an lvalue is an expression with an object type or
  /// incomplete type other than void. Nonarray expressions that can be lvalues:
  ///  - name, where name must be a variable
  ///  - e[i]
  ///  - (e), where e must be an lvalue
  ///  - e.name, where e must be an lvalue
  ///  - e->name
  ///  - *e, the type of e cannot be a function type
  ///  - string-constant
  ///
  enum isLvalueResult {
    LV_Valid,
    LV_NotObjectType,
    LV_IncompleteVoidType,
    LV_InvalidExpression
  };
  isLvalueResult isLvalue();
  
  /// isModifiableLvalue - C99 6.3.2.1: an lvalue that does not have array type,
  /// does not have an incomplete type, does not have a const-qualified type,
  /// and if it is a structure or union, does not have any member (including, 
  /// recursively, any member or element of all contained aggregates or unions)
  /// with a const-qualified type.
  enum isModifiableLvalueResult {
    MLV_Valid,
    MLV_NotObjectType,
    MLV_IncompleteVoidType,
    MLV_InvalidExpression,
    MLV_IncompleteType,
    MLV_ConstQualified,
    MLV_ArrayType
  };
  isModifiableLvalueResult isModifiableLvalue();
  
  bool isNullPointerConstant() const;

  bool isConstantExpr(SourceLocation &loc) const 
    { return isConstantExpr(false, loc); }
  bool isIntegerConstantExpr(SourceLocation &loc) const 
    { return isConstantExpr(true, loc); }
  
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() >= firstExprConstant &&
           T->getStmtClass() <= lastExprConstant; 
  }
  static bool classof(const Expr *) { return true; }
private:
  bool isConstantExpr(bool isIntegerConstant, SourceLocation &loc) const;
};

//===----------------------------------------------------------------------===//
// Primary Expressions.
//===----------------------------------------------------------------------===//

/// DeclRefExpr - [C99 6.5.1p2] - A reference to a declared variable, function,
/// enum, etc.
class DeclRefExpr : public Expr {
  Decl *D; // a ValueDecl or EnumConstantDecl
  SourceLocation Loc;
public:
  DeclRefExpr(Decl *d, QualType t, SourceLocation l) : 
    Expr(DeclRefExprClass, t), D(d), Loc(l) {}
  
  Decl *getDecl() const { return D; }
  virtual SourceRange getSourceRange() const { return SourceRange(Loc); }
  
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == DeclRefExprClass; 
  }
  static bool classof(const DeclRefExpr *) { return true; }
};

class IntegerLiteral : public Expr {
  APInt Value;
  SourceLocation Loc;
public:
  // type should be IntTy, LongTy, LongLongTy, UnsignedIntTy, UnsignedLongTy, 
  // or UnsignedLongLongTy
  IntegerLiteral(const APInt &V, QualType type, SourceLocation l)
    : Expr(IntegerLiteralClass, type), Value(V), Loc(l) {
    assert(type->isIntegerType() && "Illegal type in IntegerLiteral");
  }
  const APInt &getValue() const { return Value; }
  virtual SourceRange getSourceRange() const { return SourceRange(Loc); }

  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == IntegerLiteralClass; 
  }
  static bool classof(const IntegerLiteral *) { return true; }
};

class CharacterLiteral : public Expr {
  unsigned Value;
  SourceLocation Loc;
public:
  // type should be IntTy
  CharacterLiteral(unsigned value, QualType type, SourceLocation l)
    : Expr(CharacterLiteralClass, type), Value(value), Loc(l) {
  }
  virtual SourceRange getSourceRange() const { return SourceRange(Loc); }

  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == CharacterLiteralClass; 
  }
  static bool classof(const CharacterLiteral *) { return true; }
};

class FloatingLiteral : public Expr {
  float Value; // FIXME
  SourceLocation Loc;
public:
  FloatingLiteral(float value, QualType type, SourceLocation l)
    : Expr(FloatingLiteralClass, type), Value(value), Loc(l) {} 

  virtual SourceRange getSourceRange() const { return SourceRange(Loc); }

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
  // if the StringLiteral was composed using token pasting, both locations
  // are needed. If not (the common case), firstTokLoc == lastTokLoc.
  // FIXME: if space becomes an issue, we should create a sub-class.
  SourceLocation firstTokLoc, lastTokLoc;
public:
  StringLiteral(const char *strData, unsigned byteLength, bool Wide, 
                QualType t, SourceLocation b, SourceLocation e);
  virtual ~StringLiteral();
  
  const char *getStrData() const { return StrData; }
  unsigned getByteLength() const { return ByteLength; }
  bool isWide() const { return IsWide; }

  virtual SourceRange getSourceRange() const { 
    return SourceRange(firstTokLoc,lastTokLoc); 
  }
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
    : Expr(ParenExprClass, val->getType()), L(l), R(r), Val(val) {}
  
  const Expr *getSubExpr() const { return Val; }
  Expr *getSubExpr() { return Val; }
  SourceRange getSourceRange() const { return SourceRange(L, R); }

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
    Extension         // __extension__ marker.
  };

  UnaryOperator(Expr *input, Opcode opc, QualType type, SourceLocation l)
    : Expr(UnaryOperatorClass, type), Val(input), Opc(opc), Loc(l) {}
  
  /// getOpcodeStr - Turn an Opcode enum value into the punctuation char it
  /// corresponds to, e.g. "sizeof" or "[pre]++"
  static const char *getOpcodeStr(Opcode Op);

  /// isPostfix - Return true if this is a postfix operation, like x++.
  static bool isPostfix(Opcode Op);

  static bool isArithmeticOp(Opcode Op) { return Op >= Plus && Op <= LNot; }

  Opcode getOpcode() const { return Opc; }
  Expr *getSubExpr() const { return Val; }
  virtual SourceRange getSourceRange() const {
    if (isPostfix())
      return SourceRange(Val->getLocStart(), Loc);
    else
      return SourceRange(Loc, Val->getLocEnd());
  }
  bool isPostfix() const { return isPostfix(Opc); }
  bool isIncrementDecrementOp() const { return Opc>=PostInc && Opc<=PreDec; }
  bool isSizeOfAlignOfOp() const { return Opc == SizeOf || Opc == AlignOf; }
  
  /// getDecl - a recursive routine that derives the base decl for an
  /// expression. For example, it will return the declaration for "s" from
  /// the following complex expression "s.zz[2].bb.vv".
  static bool isAddressable(Expr *e);

  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == UnaryOperatorClass; 
  }
  static bool classof(const UnaryOperator *) { return true; }
  
private:
  Expr *Val;
  Opcode Opc;
  SourceLocation Loc;
};

/// SizeOfAlignOfTypeExpr - [C99 6.5.3.4] - This is only for sizeof/alignof of
/// *types*.  sizeof(expr) is handled by UnaryOperator.
class SizeOfAlignOfTypeExpr : public Expr {
  bool isSizeof;  // true if sizeof, false if alignof.
  QualType Ty;
  SourceLocation OpLoc, RParenLoc;
public:
  SizeOfAlignOfTypeExpr(bool issizeof, QualType argType, QualType resultType,
                        SourceLocation op, SourceLocation rp) : 
    Expr(SizeOfAlignOfTypeExprClass, resultType),
    isSizeof(issizeof), Ty(argType), OpLoc(op), RParenLoc(rp) {}
  
  bool isSizeOf() const { return isSizeof; }
  QualType getArgumentType() const { return Ty; }
  SourceRange getSourceRange() const { return SourceRange(OpLoc, RParenLoc); }

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
  SourceLocation Loc; // the location of the right bracket
public:
  ArraySubscriptExpr(Expr *base, Expr *idx, QualType t, SourceLocation l) : 
    Expr(ArraySubscriptExprClass, t),
    Base(base), Idx(idx), Loc(l) {}
  
  Expr *getBase() const { return Base; }
  Expr *getIdx() { return Idx; }
  SourceRange getSourceRange() const { 
    return SourceRange(Base->getLocStart(), Loc);
  }
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
  SourceLocation Loc; // the location of the right paren
public:
  CallExpr(Expr *fn, Expr **args, unsigned numargs, QualType t, 
           SourceLocation l);
  ~CallExpr() {
    delete [] Args;
  }
  
  Expr *getCallee() const { return Fn; }
  SourceRange getSourceRange() const { 
    return SourceRange(Fn->getLocStart(), Loc);
  }
  
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
  SourceLocation MemberLoc;
  bool IsArrow;      // True if this is "X->F", false if this is "X.F".
public:
  MemberExpr(Expr *base, bool isarrow, FieldDecl *memberdecl, SourceLocation l) 
    : Expr(MemberExprClass, memberdecl->getType()),
      Base(base), MemberDecl(memberdecl), MemberLoc(l), IsArrow(isarrow) {}
  
  Expr *getBase() const { return Base; }
  FieldDecl *getMemberDecl() const { return MemberDecl; }
  bool isArrow() const { return IsArrow; }
  virtual SourceRange getSourceRange() const {
    return SourceRange(getBase()->getLocStart(), MemberLoc);
  }
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == MemberExprClass; 
  }
  static bool classof(const MemberExpr *) { return true; }
};

/// CastExpr - [C99 6.5.4] Cast Operators.
///
class CastExpr : public Expr {
  QualType Ty;
  Expr *Op;
  SourceLocation Loc; // the location of the left paren
public:
  CastExpr(QualType ty, Expr *op, SourceLocation l) : 
    Expr(CastExprClass, ty), Ty(ty), Op(op), Loc(l) {}
  CastExpr(StmtClass SC, QualType ty, Expr *op) : 
    Expr(SC, QualType()), Ty(ty), Op(op), Loc(SourceLocation()) {}
  
  QualType getDestType() const { return Ty; }
  Expr *getSubExpr() const { return Op; }
  virtual SourceRange getSourceRange() const {
    return SourceRange(Loc, getSubExpr()->getSourceRange().End());
  }
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
  
  BinaryOperator(Expr *lhs, Expr *rhs, Opcode opc, QualType t=QualType())
    : Expr(BinaryOperatorClass, t), LHS(lhs), RHS(rhs), Opc(opc) {}

  /// getOpcodeStr - Turn an Opcode enum value into the punctuation char it
  /// corresponds to, e.g. "<<=".
  static const char *getOpcodeStr(Opcode Op);

  /// predicates to categorize the respective opcodes.
  bool isMultiplicativeOp() const { return Opc >= Mul && Opc <= Rem; }
  bool isAdditiveOp() const { return Opc == Add || Opc == Sub; }
  bool isShiftOp() const { return Opc == Shl || Opc == Shr; }
  bool isBitwiseOp() const { return Opc >= And && Opc <= Or; }
  bool isRelationalOp() const { return Opc >= LT && Opc <= GE; }
  bool isEqualityOp() const { return Opc == EQ || Opc == NE; }
  bool isLogicalOp() const { return Opc == LAnd || Opc == LOr; }
  bool isAssignmentOp() const { return Opc >= Assign && Opc <= OrAssign; }
  
  Opcode getOpcode() const { return Opc; }
  Expr *getLHS() const { return LHS; }
  Expr *getRHS() const { return RHS; }
  virtual SourceRange getSourceRange() const {
    return SourceRange(getLHS()->getLocStart(), getRHS()->getLocEnd());
  }

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
  ConditionalOperator(Expr *cond, Expr *lhs, Expr *rhs, QualType t)
    : Expr(ConditionalOperatorClass, t), Cond(cond), LHS(lhs), RHS(rhs) {}

  Expr *getCond() const { return Cond; }
  Expr *getLHS() const { return LHS; }
  Expr *getRHS() const { return RHS; }

  virtual SourceRange getSourceRange() const {
    return SourceRange(getCond()->getLocStart(), getRHS()->getLocEnd());
  }
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ConditionalOperatorClass; 
  }
  static bool classof(const ConditionalOperator *) { return true; }
};

/// AddrLabel - The GNU address of label extension, representing &&label.
class AddrLabel : public Expr {
  SourceLocation AmpAmpLoc, LabelLoc;
  LabelStmt *Label;
public:
  AddrLabel(SourceLocation AALoc, SourceLocation LLoc, LabelStmt *L, QualType t)
    : Expr(AddrLabelClass, t), AmpAmpLoc(AALoc), LabelLoc(LLoc), Label(L) {}
  
  virtual SourceRange getSourceRange() const {
    return SourceRange(AmpAmpLoc, LabelLoc);
  }
  
  LabelStmt *getLabel() const { return Label; }
  
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == AddrLabelClass; 
  }
  static bool classof(const AddrLabel *) { return true; }
};
  
}  // end namespace clang
}  // end namespace llvm

#endif
