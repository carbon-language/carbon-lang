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
#include "llvm/ADT/APSInt.h"

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

  /// getExprLoc - Return the preferred location for the arrow when diagnosing
  /// a problem with a generic expression.
  virtual SourceLocation getExprLoc() const { return getLocStart(); }
  
  /// hasLocalSideEffect - Return true if this immediate expression has side
  /// effects, not counting any sub-expressions.
  bool hasLocalSideEffect() const;
  
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

  /// isIntegerConstantExpr - Return true if this expression is a valid integer
  /// constant expression, and, if so, return its value in Result.  If not a
  /// valid i-c-e, return false and fill in Loc (if specified) with the location
  /// of the invalid expression.
  bool isIntegerConstantExpr(llvm::APSInt &Result, SourceLocation *Loc = 0,
                             bool isEvaluated = true) const;
  bool isIntegerConstantExpr(SourceLocation *Loc = 0) const {
    llvm::APSInt X(32);
    return isIntegerConstantExpr(X, Loc);
  }
  
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
  SourceLocation Loc;
public:
  DeclRefExpr(Decl *d, QualType t, SourceLocation l) : 
    Expr(DeclRefExprClass, t), D(d), Loc(l) {}
  
  Decl *getDecl() { return D; }
  const Decl *getDecl() const { return D; }
  virtual SourceRange getSourceRange() const { return SourceRange(Loc); }
  
  
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == DeclRefExprClass; 
  }
  static bool classof(const DeclRefExpr *) { return true; }
};

class IntegerLiteral : public Expr {
  llvm::APInt Value;
  SourceLocation Loc;
public:
  // type should be IntTy, LongTy, LongLongTy, UnsignedIntTy, UnsignedLongTy, 
  // or UnsignedLongLongTy
  IntegerLiteral(const llvm::APInt &V, QualType type, SourceLocation l)
    : Expr(IntegerLiteralClass, type), Value(V), Loc(l) {
    assert(type->isIntegerType() && "Illegal type in IntegerLiteral");
  }
  const llvm::APInt &getValue() const { return Value; }
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
  
  unsigned getValue() const { return Value; }

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

  float getValue() const { return Value; }
  
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
private:
  Expr *Val;
  Opcode Opc;
  SourceLocation Loc;
public:  

  UnaryOperator(Expr *input, Opcode opc, QualType type, SourceLocation l)
    : Expr(UnaryOperatorClass, type), Val(input), Opc(opc), Loc(l) {}

  Opcode getOpcode() const { return Opc; }
  Expr *getSubExpr() const { return Val; }
  
  /// getOperatorLoc - Return the location of the operator.
  SourceLocation getOperatorLoc() const { return Loc; }
  
  /// isPostfix - Return true if this is a postfix operation, like x++.
  static bool isPostfix(Opcode Op);

  bool isPostfix() const { return isPostfix(Opc); }
  bool isIncrementDecrementOp() const { return Opc>=PostInc && Opc<=PreDec; }
  bool isSizeOfAlignOfOp() const { return Opc == SizeOf || Opc == AlignOf; }
  static bool isArithmeticOp(Opcode Op) { return Op >= Plus && Op <= LNot; }
  
  /// getDecl - a recursive routine that derives the base decl for an
  /// expression. For example, it will return the declaration for "s" from
  /// the following complex expression "s.zz[2].bb.vv".
  static bool isAddressable(Expr *e);
  
  /// getOpcodeStr - Turn an Opcode enum value into the punctuation char it
  /// corresponds to, e.g. "sizeof" or "[pre]++"
  static const char *getOpcodeStr(Opcode Op);

  virtual SourceRange getSourceRange() const {
    if (isPostfix())
      return SourceRange(Val->getLocStart(), Loc);
    else
      return SourceRange(Loc, Val->getLocEnd());
  }
  virtual SourceLocation getExprLoc() const { return Loc; }
  
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == UnaryOperatorClass; 
  }
  static bool classof(const UnaryOperator *) { return true; }
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
  SourceLocation RBracketLoc;
public:
  ArraySubscriptExpr(Expr *base, Expr *idx, QualType t,
                     SourceLocation rbracketloc) : 
    Expr(ArraySubscriptExprClass, t),
    Base(base), Idx(idx), RBracketLoc(rbracketloc) {}
  
  Expr *getBase() { return Base; }
  const Expr *getBase() const { return Base; }
  Expr *getIdx() { return Idx; }
  const Expr *getIdx() const { return Idx; }
  
  SourceRange getSourceRange() const { 
    return SourceRange(Base->getLocStart(), RBracketLoc);
  }
  virtual SourceLocation getExprLoc() const { return RBracketLoc; }

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
  SourceLocation RParenLoc;
public:
  CallExpr(Expr *fn, Expr **args, unsigned numargs, QualType t, 
           SourceLocation rparenloc);
  ~CallExpr() {
    delete [] Args;
  }
  
  const Expr *getCallee() const { return Fn; }
  Expr *getCallee() { return Fn; }
  
  /// getNumArgs - Return the number of actual arguments to this call.
  ///
  unsigned getNumArgs() const { return NumArgs; }
  
  /// getArg - Return the specified argument.
  Expr *getArg(unsigned Arg) {
    assert(Arg < NumArgs && "Arg access out of range!");
    return Args[Arg];
  }
  const Expr *getArg(unsigned Arg) const {
    assert(Arg < NumArgs && "Arg access out of range!");
    return Args[Arg];
  }
  
  /// getNumCommas - Return the number of commas that must have been present in
  /// this function call.
  unsigned getNumCommas() const { return NumArgs ? NumArgs - 1 : 0; }

  SourceRange getSourceRange() const { 
    return SourceRange(Fn->getLocStart(), RParenLoc);
  }
  
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
  virtual SourceLocation getExprLoc() const { return MemberLoc; }

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
  
  SourceLocation getLParenLoc() const { return Loc; }
  
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
  
  BinaryOperator(Expr *lhs, Expr *rhs, Opcode opc, QualType ResTy)
    : Expr(BinaryOperatorClass, ResTy), LHS(lhs), RHS(rhs), Opc(opc) {
    assert(!isCompoundAssignmentOp() && 
           "Use ArithAssignBinaryOperator for compound assignments");
  }

  Opcode getOpcode() const { return Opc; }
  Expr *getLHS() const { return LHS; }
  Expr *getRHS() const { return RHS; }
  virtual SourceRange getSourceRange() const {
    return SourceRange(getLHS()->getLocStart(), getRHS()->getLocEnd());
  }
  
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
  bool isCompoundAssignmentOp() const { return Opc > Assign && Opc <= OrAssign;}
  
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == BinaryOperatorClass; 
  }
  static bool classof(const BinaryOperator *) { return true; }
private:
  Expr *LHS, *RHS;
  Opcode Opc;
protected:
  BinaryOperator(Expr *lhs, Expr *rhs, Opcode opc, QualType ResTy, bool dead)
    : Expr(BinaryOperatorClass, ResTy), LHS(lhs), RHS(rhs), Opc(opc) {
  }
};

/// CompoundAssignOperator - For compound assignments (e.g. +=), we keep
/// track of the type the operation is performed in.  Due to the semantics of
/// these operators, the operands are promoted, the aritmetic performed, an
/// implicit conversion back to the result type done, then the assignment takes
/// place.  This captures the intermediate type which the computation is done
/// in.
class CompoundAssignOperator : public BinaryOperator {
  QualType ComputationType;
public:
  CompoundAssignOperator(Expr *lhs, Expr *rhs, Opcode opc,
                         QualType ResType, QualType CompType)
    : BinaryOperator(lhs, rhs, opc, ResType, true), ComputationType(CompType) {
    assert(isCompoundAssignmentOp() && 
           "Only should be used for compound assignments");
  }

  QualType getComputationType() const { return ComputationType; }
  
  static bool classof(const CompoundAssignOperator *) { return true; }
  static bool classof(const BinaryOperator *B) { 
    return B->isCompoundAssignmentOp(); 
  }
  static bool classof(const Stmt *S) { 
    return isa<BinaryOperator>(S) && classof(cast<BinaryOperator>(S));
  }
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

#endif
