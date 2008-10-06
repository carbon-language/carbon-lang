//===--- Expr.h - Classes for representing expressions ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include <vector>

namespace clang {
  class ASTContext;
  class APValue;
  class Decl;
  class IdentifierInfo;
  class ParmVarDecl;
  class ValueDecl;
    
/// Expr - This represents one expression.  Note that Expr's are subclasses of
/// Stmt.  This allows an expression to be transparently used any place a Stmt
/// is required.
///
class Expr : public Stmt {
  QualType TR;
protected:
  Expr(StmtClass SC, QualType T) : Stmt(SC), TR(T) {}
public:  
  QualType getType() const { return TR; }
  void setType(QualType t) { TR = t; }

  /// SourceLocation tokens are not useful in isolation - they are low level
  /// value objects created/interpreted by SourceManager. We assume AST
  /// clients will have a pointer to the respective SourceManager.
  virtual SourceRange getSourceRange() const = 0;

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
  ///  - reference type [C++ [expr]]
  ///
  enum isLvalueResult {
    LV_Valid,
    LV_NotObjectType,
    LV_IncompleteVoidType,
    LV_DuplicateVectorComponents,
    LV_InvalidExpression
  };
  isLvalueResult isLvalue(ASTContext &Ctx) const;
  
  /// isModifiableLvalue - C99 6.3.2.1: an lvalue that does not have array type,
  /// does not have an incomplete type, does not have a const-qualified type,
  /// and if it is a structure or union, does not have any member (including, 
  /// recursively, any member or element of all contained aggregates or unions)
  /// with a const-qualified type.
  enum isModifiableLvalueResult {
    MLV_Valid,
    MLV_NotObjectType,
    MLV_IncompleteVoidType,
    MLV_DuplicateVectorComponents,
    MLV_InvalidExpression,
    MLV_IncompleteType,
    MLV_ConstQualified,
    MLV_ArrayType,
    MLV_NotBlockQualified
  };
  isModifiableLvalueResult isModifiableLvalue(ASTContext &Ctx) const;
  
  bool isNullPointerConstant(ASTContext &Ctx) const;

  /// getIntegerConstantExprValue() - Return the value of an integer
  /// constant expression. The expression must be a valid integer
  /// constant expression as determined by isIntegerConstantExpr.
  llvm::APSInt getIntegerConstantExprValue(ASTContext &Ctx) const {
    llvm::APSInt X;
    bool success = isIntegerConstantExpr(X, Ctx);
    success = success;
    assert(success && "Illegal argument to getIntegerConstantExpr");
    return X;
  }

  /// isIntegerConstantExpr - Return true if this expression is a valid integer
  /// constant expression, and, if so, return its value in Result.  If not a
  /// valid i-c-e, return false and fill in Loc (if specified) with the location
  /// of the invalid expression.
  bool isIntegerConstantExpr(llvm::APSInt &Result, ASTContext &Ctx,
                             SourceLocation *Loc = 0,
                             bool isEvaluated = true) const;
  bool isIntegerConstantExpr(ASTContext &Ctx, SourceLocation *Loc = 0) const {
    llvm::APSInt X;
    return isIntegerConstantExpr(X, Ctx, Loc);
  }
  /// isConstantExpr - Return true if this expression is a valid constant expr.
  bool isConstantExpr(ASTContext &Ctx, SourceLocation *Loc) const;
  
  /// tryEvaluate - Return true if this is a constant which we can fold using
  /// any crazy technique (that has nothing to do with language standards) that
  /// we want to.  If this function returns true, it returns the folded constant
  /// in Result.
  bool tryEvaluate(APValue& Result, ASTContext &Ctx) const;

  /// hasGlobalStorage - Return true if this expression has static storage
  /// duration.  This means that the address of this expression is a link-time
  /// constant.
  bool hasGlobalStorage() const;  
  
  /// IgnoreParens - Ignore parentheses.  If this Expr is a ParenExpr, return
  ///  its subexpression.  If that subexpression is also a ParenExpr, 
  ///  then this method recursively returns its subexpression, and so forth.
  ///  Otherwise, the method returns the current Expr.
  Expr* IgnoreParens();

  /// IgnoreParenCasts - Ignore parentheses and casts.  Strip off any ParenExpr
  /// or CastExprs or ImplicitCastExprs, returning their operand.
  Expr *IgnoreParenCasts();
  
  const Expr* IgnoreParens() const {
    return const_cast<Expr*>(this)->IgnoreParens();
  }
  const Expr *IgnoreParenCasts() const {
    return const_cast<Expr*>(this)->IgnoreParenCasts();
  }
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() >= firstExprConstant &&
           T->getStmtClass() <= lastExprConstant; 
  }
  static bool classof(const Expr *) { return true; }
  
  static inline Expr* Create(llvm::Deserializer& D, ASTContext& C) {
    return cast<Expr>(Stmt::Create(D, C));
  }
};

//===----------------------------------------------------------------------===//
// ExprIterator - Iterators for iterating over Stmt* arrays that contain
//  only Expr*.  This is needed because AST nodes use Stmt* arrays to store
//  references to children (to be compatible with StmtIterator).
//===----------------------------------------------------------------------===//
  
class ExprIterator {
  Stmt** I;
public:
  ExprIterator(Stmt** i) : I(i) {}
  ExprIterator() : I(0) {}    
  ExprIterator& operator++() { ++I; return *this; }
  ExprIterator operator-(size_t i) { return I-i; }
  ExprIterator operator+(size_t i) { return I+i; }
  Expr* operator[](size_t idx) { return cast<Expr>(I[idx]); }
  // FIXME: Verify that this will correctly return a signed distance.
  signed operator-(const ExprIterator& R) const { return I - R.I; }
  Expr* operator*() const { return cast<Expr>(*I); }
  Expr* operator->() const { return cast<Expr>(*I); }
  bool operator==(const ExprIterator& R) const { return I == R.I; }
  bool operator!=(const ExprIterator& R) const { return I != R.I; }
  bool operator>(const ExprIterator& R) const { return I > R.I; }
  bool operator>=(const ExprIterator& R) const { return I >= R.I; }
};

class ConstExprIterator {
  Stmt* const * I;
public:
  ConstExprIterator(Stmt* const* i) : I(i) {}
  ConstExprIterator() : I(0) {}    
  ConstExprIterator& operator++() { ++I; return *this; }
  ConstExprIterator operator+(size_t i) { return I+i; }
  ConstExprIterator operator-(size_t i) { return I-i; }
  Expr * operator[](size_t idx) const { return cast<Expr>(I[idx]); }
  signed operator-(const ConstExprIterator& R) const { return I - R.I; }
  Expr * operator*() const { return cast<Expr>(*I); }
  Expr * operator->() const { return cast<Expr>(*I); }
  bool operator==(const ConstExprIterator& R) const { return I == R.I; }
  bool operator!=(const ConstExprIterator& R) const { return I != R.I; }
  bool operator>(const ConstExprIterator& R) const { return I > R.I; }
  bool operator>=(const ConstExprIterator& R) const { return I >= R.I; }
}; 
  
  
//===----------------------------------------------------------------------===//
// Primary Expressions.
//===----------------------------------------------------------------------===//

/// DeclRefExpr - [C99 6.5.1p2] - A reference to a declared variable, function,
/// enum, etc.
class DeclRefExpr : public Expr {
  ValueDecl *D; 
  SourceLocation Loc;

protected:
  DeclRefExpr(StmtClass SC, ValueDecl *d, QualType t, SourceLocation l) :
    Expr(SC, t), D(d), Loc(l) {}

public:
  DeclRefExpr(ValueDecl *d, QualType t, SourceLocation l) : 
    Expr(DeclRefExprClass, t), D(d), Loc(l) {}
  
  ValueDecl *getDecl() { return D; }
  const ValueDecl *getDecl() const { return D; }
  SourceLocation getLocation() const { return Loc; }
  virtual SourceRange getSourceRange() const { return SourceRange(Loc); }
  
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == DeclRefExprClass ||
           T->getStmtClass() == CXXConditionDeclExprClass; 
  }
  static bool classof(const DeclRefExpr *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static DeclRefExpr* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

/// PredefinedExpr - [C99 6.4.2.2] - A predefined identifier such as __func__.
class PredefinedExpr : public Expr {
public:
  enum IdentType {
    Func,
    Function,
    PrettyFunction,
    CXXThis,
    ObjCSuper // super
  };
  
private:
  SourceLocation Loc;
  IdentType Type;
public:
  PredefinedExpr(SourceLocation l, QualType type, IdentType IT) 
    : Expr(PredefinedExprClass, type), Loc(l), Type(IT) {}
  
  IdentType getIdentType() const { return Type; }
  
  virtual SourceRange getSourceRange() const { return SourceRange(Loc); }

  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == PredefinedExprClass; 
  }
  static bool classof(const PredefinedExpr *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static PredefinedExpr* CreateImpl(llvm::Deserializer& D, ASTContext& C);
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

  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == IntegerLiteralClass; 
  }
  static bool classof(const IntegerLiteral *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static IntegerLiteral* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

class CharacterLiteral : public Expr {
  unsigned Value;
  SourceLocation Loc;
  bool IsWide;
public:
  // type should be IntTy
  CharacterLiteral(unsigned value, bool iswide, QualType type, SourceLocation l)
    : Expr(CharacterLiteralClass, type), Value(value), Loc(l), IsWide(iswide) {
  }
  SourceLocation getLoc() const { return Loc; }
  bool isWide() const { return IsWide; }
  
  virtual SourceRange getSourceRange() const { return SourceRange(Loc); }
  
  unsigned getValue() const { return Value; }

  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == CharacterLiteralClass; 
  }
  static bool classof(const CharacterLiteral *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static CharacterLiteral* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

class FloatingLiteral : public Expr {
  llvm::APFloat Value;
  bool IsExact : 1;
  SourceLocation Loc;
public:
  FloatingLiteral(const llvm::APFloat &V, bool* isexact, 
                  QualType Type, SourceLocation L)
    : Expr(FloatingLiteralClass, Type), Value(V), IsExact(*isexact), Loc(L) {} 

  const llvm::APFloat &getValue() const { return Value; }
  
  bool isExact() const { return IsExact; }

  /// getValueAsApproximateDouble - This returns the value as an inaccurate
  /// double.  Note that this may cause loss of precision, but is useful for
  /// debugging dumps, etc.
  double getValueAsApproximateDouble() const;
 
  virtual SourceRange getSourceRange() const { return SourceRange(Loc); }

  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == FloatingLiteralClass; 
  }
  static bool classof(const FloatingLiteral *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static FloatingLiteral* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

/// ImaginaryLiteral - We support imaginary integer and floating point literals,
/// like "1.0i".  We represent these as a wrapper around FloatingLiteral and
/// IntegerLiteral classes.  Instances of this class always have a Complex type
/// whose element type matches the subexpression.
///
class ImaginaryLiteral : public Expr {
  Stmt *Val;
public:
  ImaginaryLiteral(Expr *val, QualType Ty)
    : Expr(ImaginaryLiteralClass, Ty), Val(val) {}
  
  const Expr *getSubExpr() const { return cast<Expr>(Val); }
  Expr *getSubExpr() { return cast<Expr>(Val); }
  
  virtual SourceRange getSourceRange() const { return Val->getSourceRange(); }
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ImaginaryLiteralClass; 
  }
  static bool classof(const ImaginaryLiteral *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static ImaginaryLiteral* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

/// StringLiteral - This represents a string literal expression, e.g. "foo"
/// or L"bar" (wide strings).  The actual string is returned by getStrData()
/// is NOT null-terminated, and the length of the string is determined by
/// calling getByteLength().  The C type for a string is always a
/// ConstantArrayType.
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
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == StringLiteralClass; 
  }
  static bool classof(const StringLiteral *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static StringLiteral* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

/// ParenExpr - This represents a parethesized expression, e.g. "(1)".  This
/// AST node is only formed if full location information is requested.
class ParenExpr : public Expr {
  SourceLocation L, R;
  Stmt *Val;
public:
  ParenExpr(SourceLocation l, SourceLocation r, Expr *val)
    : Expr(ParenExprClass, val->getType()), L(l), R(r), Val(val) {}
  
  const Expr *getSubExpr() const { return cast<Expr>(Val); }
  Expr *getSubExpr() { return cast<Expr>(Val); }
  virtual SourceRange getSourceRange() const { return SourceRange(L, R); }

  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ParenExprClass; 
  }
  static bool classof(const ParenExpr *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static ParenExpr* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};


/// UnaryOperator - This represents the unary-expression's (except sizeof of
/// types), the postinc/postdec operators from postfix-expression, and various
/// extensions.
///
/// Notes on various nodes:
///
/// Real/Imag - These return the real/imag part of a complex operand.  If
///   applied to a non-complex value, the former returns its operand and the
///   later returns zero in the type of the operand.
///
/// __builtin_offsetof(type, a.b[10]) is represented as a unary operator whose
///   subexpression is a compound literal with the various MemberExpr and 
///   ArraySubscriptExpr's applied to it.
///
class UnaryOperator : public Expr {
public:
  // Note that additions to this should also update the StmtVisitor class.
  enum Opcode {
    PostInc, PostDec, // [C99 6.5.2.4] Postfix increment and decrement operators
    PreInc, PreDec,   // [C99 6.5.3.1] Prefix increment and decrement operators.
    AddrOf, Deref,    // [C99 6.5.3.2] Address and indirection operators.
    Plus, Minus,      // [C99 6.5.3.3] Unary arithmetic operators.
    Not, LNot,        // [C99 6.5.3.3] Unary arithmetic operators.
    SizeOf, AlignOf,  // [C99 6.5.3.4] Sizeof (expr, not type) operator.
    Real, Imag,       // "__real expr"/"__imag expr" Extension.
    Extension,        // __extension__ marker.
    OffsetOf          // __builtin_offsetof
  };
private:
  Stmt *Val;
  Opcode Opc;
  SourceLocation Loc;
public:  

  UnaryOperator(Expr *input, Opcode opc, QualType type, SourceLocation l)
    : Expr(UnaryOperatorClass, type), Val(input), Opc(opc), Loc(l) {}

  Opcode getOpcode() const { return Opc; }
  Expr *getSubExpr() const { return cast<Expr>(Val); }
  
  /// getOperatorLoc - Return the location of the operator.
  SourceLocation getOperatorLoc() const { return Loc; }
  
  /// isPostfix - Return true if this is a postfix operation, like x++.
  static bool isPostfix(Opcode Op);

  /// isPostfix - Return true if this is a prefix operation, like --x.
  static bool isPrefix(Opcode Op);

  bool isPrefix() const { return isPrefix(Opc); }
  bool isPostfix() const { return isPostfix(Opc); }
  bool isIncrementOp() const {return Opc==PreInc || Opc==PostInc; }
  bool isIncrementDecrementOp() const { return Opc>=PostInc && Opc<=PreDec; }
  bool isSizeOfAlignOfOp() const { return Opc == SizeOf || Opc == AlignOf; }
  bool isOffsetOfOp() const { return Opc == OffsetOf; }
  static bool isArithmeticOp(Opcode Op) { return Op >= Plus && Op <= LNot; }
  
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
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == UnaryOperatorClass; 
  }
  static bool classof(const UnaryOperator *) { return true; }
  
  int64_t evaluateOffsetOf(ASTContext& C) const;
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static UnaryOperator* CreateImpl(llvm::Deserializer& D, ASTContext& C);
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
  
  virtual void Destroy(ASTContext& C);

  bool isSizeOf() const { return isSizeof; }
  QualType getArgumentType() const { return Ty; }
  
  SourceLocation getOperatorLoc() const { return OpLoc; }

  virtual SourceRange getSourceRange() const {
    return SourceRange(OpLoc, RParenLoc);
  }

  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == SizeOfAlignOfTypeExprClass; 
  }
  static bool classof(const SizeOfAlignOfTypeExpr *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static SizeOfAlignOfTypeExpr* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

//===----------------------------------------------------------------------===//
// Postfix Operators.
//===----------------------------------------------------------------------===//

/// ArraySubscriptExpr - [C99 6.5.2.1] Array Subscripting.
class ArraySubscriptExpr : public Expr {
  enum { LHS, RHS, END_EXPR=2 };
  Stmt* SubExprs[END_EXPR]; 
  SourceLocation RBracketLoc;
public:
  ArraySubscriptExpr(Expr *lhs, Expr *rhs, QualType t,
                     SourceLocation rbracketloc)
  : Expr(ArraySubscriptExprClass, t), RBracketLoc(rbracketloc) {
    SubExprs[LHS] = lhs;
    SubExprs[RHS] = rhs;
  }
  
  /// An array access can be written A[4] or 4[A] (both are equivalent).
  /// - getBase() and getIdx() always present the normalized view: A[4].
  ///    In this case getBase() returns "A" and getIdx() returns "4".
  /// - getLHS() and getRHS() present the syntactic view. e.g. for
  ///    4[A] getLHS() returns "4".
  /// Note: Because vector element access is also written A[4] we must
  /// predicate the format conversion in getBase and getIdx only on the
  /// the type of the RHS, as it is possible for the LHS to be a vector of
  /// integer type
  Expr *getLHS() { return cast<Expr>(SubExprs[LHS]); }
  const Expr *getLHS() const { return cast<Expr>(SubExprs[LHS]); }
  
  Expr *getRHS() { return cast<Expr>(SubExprs[RHS]); }
  const Expr *getRHS() const { return cast<Expr>(SubExprs[RHS]); }
  
  Expr *getBase() { 
    return cast<Expr>(getRHS()->getType()->isIntegerType() ? getLHS():getRHS());
  }
    
  const Expr *getBase() const { 
    return cast<Expr>(getRHS()->getType()->isIntegerType() ? getLHS():getRHS());
  }
  
  Expr *getIdx() { 
    return cast<Expr>(getRHS()->getType()->isIntegerType() ? getRHS():getLHS());
  }
  
  const Expr *getIdx() const {
    return cast<Expr>(getRHS()->getType()->isIntegerType() ? getRHS():getLHS());
  }  
  
  virtual SourceRange getSourceRange() const { 
    return SourceRange(getLHS()->getLocStart(), RBracketLoc);
  }
  
  virtual SourceLocation getExprLoc() const { return RBracketLoc; }

  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ArraySubscriptExprClass; 
  }
  static bool classof(const ArraySubscriptExpr *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static ArraySubscriptExpr* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};


/// CallExpr - [C99 6.5.2.2] Function Calls.
///
class CallExpr : public Expr {
  enum { FN=0, ARGS_START=1 };
  Stmt **SubExprs;
  unsigned NumArgs;
  SourceLocation RParenLoc;
  
  // This version of the ctor is for deserialization.
  CallExpr(Stmt** subexprs, unsigned numargs, QualType t, 
           SourceLocation rparenloc)
  : Expr(CallExprClass,t), SubExprs(subexprs), 
    NumArgs(numargs), RParenLoc(rparenloc) {}
  
public:
  CallExpr(Expr *fn, Expr **args, unsigned numargs, QualType t, 
           SourceLocation rparenloc);
  ~CallExpr() {
    delete [] SubExprs;
  }
  
  const Expr *getCallee() const { return cast<Expr>(SubExprs[FN]); }
  Expr *getCallee() { return cast<Expr>(SubExprs[FN]); }
  void setCallee(Expr *F) { SubExprs[FN] = F; }
  
  /// getNumArgs - Return the number of actual arguments to this call.
  ///
  unsigned getNumArgs() const { return NumArgs; }
  
  /// getArg - Return the specified argument.
  Expr *getArg(unsigned Arg) {
    assert(Arg < NumArgs && "Arg access out of range!");
    return cast<Expr>(SubExprs[Arg+ARGS_START]);
  }
  const Expr *getArg(unsigned Arg) const {
    assert(Arg < NumArgs && "Arg access out of range!");
    return cast<Expr>(SubExprs[Arg+ARGS_START]);
  }
  /// setArg - Set the specified argument.
  void setArg(unsigned Arg, Expr *ArgExpr) {
    assert(Arg < NumArgs && "Arg access out of range!");
    SubExprs[Arg+ARGS_START] = ArgExpr;
  }
  
  /// setNumArgs - This changes the number of arguments present in this call.
  /// Any orphaned expressions are deleted by this, and any new operands are set
  /// to null.
  void setNumArgs(unsigned NumArgs);
  
  typedef ExprIterator arg_iterator;
  typedef ConstExprIterator const_arg_iterator;
    
  arg_iterator arg_begin() { return SubExprs+ARGS_START; }
  arg_iterator arg_end() { return SubExprs+ARGS_START+getNumArgs(); }
  const_arg_iterator arg_begin() const { return SubExprs+ARGS_START; }
  const_arg_iterator arg_end() const { return SubExprs+ARGS_START+getNumArgs();}
  
  /// getNumCommas - Return the number of commas that must have been present in
  /// this function call.
  unsigned getNumCommas() const { return NumArgs ? NumArgs - 1 : 0; }

  /// isBuiltinCall - If this is a call to a builtin, return the builtin ID.  If
  /// not, return 0.
  unsigned isBuiltinCall() const;
  
  
  /// isBuiltinConstantExpr - Return true if this built-in call is constant.
  bool isBuiltinConstantExpr(ASTContext &Ctx) const;
  
  SourceLocation getRParenLoc() const { return RParenLoc; }

  virtual SourceRange getSourceRange() const { 
    return SourceRange(getCallee()->getLocStart(), RParenLoc);
  }
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == CallExprClass; 
  }
  static bool classof(const CallExpr *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static CallExpr* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

/// MemberExpr - [C99 6.5.2.3] Structure and Union Members.
///
class MemberExpr : public Expr {
  Stmt *Base;
  FieldDecl *MemberDecl;
  SourceLocation MemberLoc;
  bool IsArrow;      // True if this is "X->F", false if this is "X.F".
public:
  MemberExpr(Expr *base, bool isarrow, FieldDecl *memberdecl, SourceLocation l,
             QualType ty) 
    : Expr(MemberExprClass, ty),
      Base(base), MemberDecl(memberdecl), MemberLoc(l), IsArrow(isarrow) {}

  Expr *getBase() const { return cast<Expr>(Base); }
  FieldDecl *getMemberDecl() const { return MemberDecl; }
  bool isArrow() const { return IsArrow; }

  virtual SourceRange getSourceRange() const {
    return SourceRange(getBase()->getLocStart(), MemberLoc);
  }
  
  virtual SourceLocation getExprLoc() const { return MemberLoc; }

  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == MemberExprClass; 
  }
  static bool classof(const MemberExpr *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static MemberExpr* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

/// CompoundLiteralExpr - [C99 6.5.2.5] 
///
class CompoundLiteralExpr : public Expr {
  /// LParenLoc - If non-null, this is the location of the left paren in a
  /// compound literal like "(int){4}".  This can be null if this is a
  /// synthesized compound expression.
  SourceLocation LParenLoc;
  Stmt *Init;
  bool FileScope;
public:
  CompoundLiteralExpr(SourceLocation lparenloc, QualType ty, Expr *init,
                      bool fileScope)
    : Expr(CompoundLiteralExprClass, ty), LParenLoc(lparenloc), Init(init),
      FileScope(fileScope) {}
  
  const Expr *getInitializer() const { return cast<Expr>(Init); }
  Expr *getInitializer() { return cast<Expr>(Init); }

  bool isFileScope() const { return FileScope; }
  
  SourceLocation getLParenLoc() const { return LParenLoc; }
  
  virtual SourceRange getSourceRange() const {
    // FIXME: Init should never be null.
    if (!Init)
      return SourceRange();
    if (LParenLoc.isInvalid())
      return Init->getSourceRange();
    return SourceRange(LParenLoc, Init->getLocEnd());
  }

  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == CompoundLiteralExprClass; 
  }
  static bool classof(const CompoundLiteralExpr *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static CompoundLiteralExpr* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

/// CastExpr - Base class for Cast Operators (explicit, implicit, etc.).
/// Classes that derive from CastExpr are:
///
///   ImplicitCastExpr
///   ExplicitCastExpr
///
class CastExpr : public Expr {
  Stmt *Op;
protected:
  CastExpr(StmtClass SC, QualType ty, Expr *op) : 
    Expr(SC, ty), Op(op) {}
  
public:
  Expr *getSubExpr() { return cast<Expr>(Op); }
  const Expr *getSubExpr() const { return cast<Expr>(Op); }
  
  static bool classof(const Stmt *T) { 
    switch (T->getStmtClass()) {
    case ImplicitCastExprClass:
    case ExplicitCastExprClass:
    case CXXFunctionalCastExprClass:
      return true;
    default:
      return false;
    }
  }
  static bool classof(const CastExpr *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// ImplicitCastExpr - Allows us to explicitly represent implicit type 
/// conversions. For example: converting T[]->T*, void f()->void (*f)(), 
/// float->double, short->int, etc.
///
class ImplicitCastExpr : public CastExpr {
public:
  ImplicitCastExpr(QualType ty, Expr *op) : 
    CastExpr(ImplicitCastExprClass, ty, op) {}

  virtual SourceRange getSourceRange() const {
    return getSubExpr()->getSourceRange();
  }

  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ImplicitCastExprClass; 
  }
  static bool classof(const ImplicitCastExpr *) { return true; }
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static ImplicitCastExpr* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

/// ExplicitCastExpr - [C99 6.5.4] Cast Operators.
///
class ExplicitCastExpr : public CastExpr {
  SourceLocation Loc; // the location of the left paren
public:
  ExplicitCastExpr(QualType ty, Expr *op, SourceLocation l) : 
    CastExpr(ExplicitCastExprClass, ty, op), Loc(l) {}

  SourceLocation getLParenLoc() const { return Loc; }
  
  virtual SourceRange getSourceRange() const {
    return SourceRange(Loc, getSubExpr()->getSourceRange().getEnd());
  }
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ExplicitCastExprClass; 
  }
  static bool classof(const ExplicitCastExpr *) { return true; }
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static ExplicitCastExpr* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

class BinaryOperator : public Expr {
public:
  enum Opcode {
    // Operators listed in order of precedence.
    // Note that additions to this should also update the StmtVisitor class.
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
private:
  enum { LHS, RHS, END_EXPR };
  Stmt* SubExprs[END_EXPR];
  Opcode Opc;
  SourceLocation OpLoc;
public:  
  
  BinaryOperator(Expr *lhs, Expr *rhs, Opcode opc, QualType ResTy,
                 SourceLocation opLoc)
    : Expr(BinaryOperatorClass, ResTy), Opc(opc), OpLoc(opLoc) {
    SubExprs[LHS] = lhs;
    SubExprs[RHS] = rhs;
    assert(!isCompoundAssignmentOp() && 
           "Use ArithAssignBinaryOperator for compound assignments");
  }

  SourceLocation getOperatorLoc() const { return OpLoc; }
  Opcode getOpcode() const { return Opc; }
  Expr *getLHS() const { return cast<Expr>(SubExprs[LHS]); }
  Expr *getRHS() const { return cast<Expr>(SubExprs[RHS]); }
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

  static bool isRelationalOp(Opcode Opc) { return Opc >= LT && Opc <= GE; }
  bool isRelationalOp() const { return isRelationalOp(Opc); }

  static bool isEqualityOp(Opcode Opc) { return Opc == EQ || Opc == NE; }  
  bool isEqualityOp() const { return isEqualityOp(Opc); }
  
  static bool isLogicalOp(Opcode Opc) { return Opc == LAnd || Opc == LOr; }
  bool isLogicalOp() const { return isLogicalOp(Opc); }

  bool isAssignmentOp() const { return Opc >= Assign && Opc <= OrAssign; }
  bool isCompoundAssignmentOp() const { return Opc > Assign && Opc <= OrAssign;}
  bool isShiftAssignOp() const { return Opc == ShlAssign || Opc == ShrAssign; }
  
  static bool classof(const Stmt *S) { 
    return S->getStmtClass() == BinaryOperatorClass ||
           S->getStmtClass() == CompoundAssignOperatorClass; 
  }
  static bool classof(const BinaryOperator *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static BinaryOperator* CreateImpl(llvm::Deserializer& D, ASTContext& C);

protected:
  BinaryOperator(Expr *lhs, Expr *rhs, Opcode opc, QualType ResTy,
                 SourceLocation oploc, bool dead)
    : Expr(CompoundAssignOperatorClass, ResTy), Opc(opc), OpLoc(oploc) {
    SubExprs[LHS] = lhs;
    SubExprs[RHS] = rhs;
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
                         QualType ResType, QualType CompType,
                         SourceLocation OpLoc)
    : BinaryOperator(lhs, rhs, opc, ResType, OpLoc, true),
      ComputationType(CompType) {
    assert(isCompoundAssignmentOp() && 
           "Only should be used for compound assignments");
  }

  QualType getComputationType() const { return ComputationType; }
  
  static bool classof(const CompoundAssignOperator *) { return true; }
  static bool classof(const Stmt *S) { 
    return S->getStmtClass() == CompoundAssignOperatorClass; 
  }
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static CompoundAssignOperator* CreateImpl(llvm::Deserializer& D,
                                            ASTContext& C);
};

/// ConditionalOperator - The ?: operator.  Note that LHS may be null when the
/// GNU "missing LHS" extension is in use.
///
class ConditionalOperator : public Expr {
  enum { COND, LHS, RHS, END_EXPR };
  Stmt* SubExprs[END_EXPR]; // Left/Middle/Right hand sides.
public:
  ConditionalOperator(Expr *cond, Expr *lhs, Expr *rhs, QualType t)
    : Expr(ConditionalOperatorClass, t) {
    SubExprs[COND] = cond;
    SubExprs[LHS] = lhs;
    SubExprs[RHS] = rhs;
  }

  // getCond - Return the expression representing the condition for
  //  the ?: operator.
  Expr *getCond() const { return cast<Expr>(SubExprs[COND]); }

  // getTrueExpr - Return the subexpression representing the value of the ?:
  //  expression if the condition evaluates to true.  In most cases this value
  //  will be the same as getLHS() except a GCC extension allows the left
  //  subexpression to be omitted, and instead of the condition be returned.
  //  e.g: x ?: y is shorthand for x ? x : y, except that the expression "x"
  //  is only evaluated once.  
  Expr *getTrueExpr() const {
    return cast<Expr>(SubExprs[LHS] ? SubExprs[LHS] : SubExprs[COND]);
  }
  
  // getTrueExpr - Return the subexpression representing the value of the ?:
  // expression if the condition evaluates to false. This is the same as getRHS.
  Expr *getFalseExpr() const { return cast<Expr>(SubExprs[RHS]); }
  
  Expr *getLHS() const { return cast_or_null<Expr>(SubExprs[LHS]); }
  Expr *getRHS() const { return cast<Expr>(SubExprs[RHS]); }

  virtual SourceRange getSourceRange() const {
    return SourceRange(getCond()->getLocStart(), getRHS()->getLocEnd());
  }
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ConditionalOperatorClass; 
  }
  static bool classof(const ConditionalOperator *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static ConditionalOperator* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

/// AddrLabelExpr - The GNU address of label extension, representing &&label.
class AddrLabelExpr : public Expr {
  SourceLocation AmpAmpLoc, LabelLoc;
  LabelStmt *Label;
public:
  AddrLabelExpr(SourceLocation AALoc, SourceLocation LLoc, LabelStmt *L,
                QualType t)
    : Expr(AddrLabelExprClass, t), AmpAmpLoc(AALoc), LabelLoc(LLoc), Label(L) {}
  
  virtual SourceRange getSourceRange() const {
    return SourceRange(AmpAmpLoc, LabelLoc);
  }
  
  LabelStmt *getLabel() const { return Label; }
  
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == AddrLabelExprClass; 
  }
  static bool classof(const AddrLabelExpr *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static AddrLabelExpr* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

/// StmtExpr - This is the GNU Statement Expression extension: ({int X=4; X;}).
/// The StmtExpr contains a single CompoundStmt node, which it evaluates and
/// takes the value of the last subexpression.
class StmtExpr : public Expr {
  Stmt *SubStmt;
  SourceLocation LParenLoc, RParenLoc;
public:
  StmtExpr(CompoundStmt *substmt, QualType T,
           SourceLocation lp, SourceLocation rp) :
    Expr(StmtExprClass, T), SubStmt(substmt),  LParenLoc(lp), RParenLoc(rp) { }
  
  CompoundStmt *getSubStmt() { return cast<CompoundStmt>(SubStmt); }
  const CompoundStmt *getSubStmt() const { return cast<CompoundStmt>(SubStmt); }
  
  virtual SourceRange getSourceRange() const {
    return SourceRange(LParenLoc, RParenLoc);
  }
  
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == StmtExprClass; 
  }
  static bool classof(const StmtExpr *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static StmtExpr* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

/// TypesCompatibleExpr - GNU builtin-in function __builtin_type_compatible_p.
/// This AST node represents a function that returns 1 if two *types* (not
/// expressions) are compatible. The result of this built-in function can be
/// used in integer constant expressions.
class TypesCompatibleExpr : public Expr {
  QualType Type1;
  QualType Type2;
  SourceLocation BuiltinLoc, RParenLoc;
public:
  TypesCompatibleExpr(QualType ReturnType, SourceLocation BLoc, 
                      QualType t1, QualType t2, SourceLocation RP) : 
    Expr(TypesCompatibleExprClass, ReturnType), Type1(t1), Type2(t2),
    BuiltinLoc(BLoc), RParenLoc(RP) {}

  QualType getArgType1() const { return Type1; }
  QualType getArgType2() const { return Type2; }
    
  virtual SourceRange getSourceRange() const {
    return SourceRange(BuiltinLoc, RParenLoc);
  }
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == TypesCompatibleExprClass; 
  }
  static bool classof(const TypesCompatibleExpr *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// ShuffleVectorExpr - clang-specific builtin-in function
/// __builtin_shufflevector.
/// This AST node represents a operator that does a constant
/// shuffle, similar to LLVM's shufflevector instruction. It takes
/// two vectors and a variable number of constant indices,
/// and returns the appropriately shuffled vector.
class ShuffleVectorExpr : public Expr {
  SourceLocation BuiltinLoc, RParenLoc;

  // SubExprs - the list of values passed to the __builtin_shufflevector
  // function. The first two are vectors, and the rest are constant
  // indices.  The number of values in this list is always
  // 2+the number of indices in the vector type.
  Stmt **SubExprs;
  unsigned NumExprs;

public:
  ShuffleVectorExpr(Expr **args, unsigned nexpr,
                    QualType Type, SourceLocation BLoc, 
                    SourceLocation RP) : 
    Expr(ShuffleVectorExprClass, Type), BuiltinLoc(BLoc),
    RParenLoc(RP), NumExprs(nexpr) {
      
    SubExprs = new Stmt*[nexpr];
    for (unsigned i = 0; i < nexpr; i++)
      SubExprs[i] = args[i];
  }
    
  virtual SourceRange getSourceRange() const {
    return SourceRange(BuiltinLoc, RParenLoc);
  }
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ShuffleVectorExprClass; 
  }
  static bool classof(const ShuffleVectorExpr *) { return true; }
  
  ~ShuffleVectorExpr() {
    delete [] SubExprs;
  }
  
  /// getNumSubExprs - Return the size of the SubExprs array.  This includes the
  /// constant expression, the actual arguments passed in, and the function
  /// pointers.
  unsigned getNumSubExprs() const { return NumExprs; }
  
  /// getExpr - Return the Expr at the specified index.
  Expr *getExpr(unsigned Index) {
    assert((Index < NumExprs) && "Arg access out of range!");
    return cast<Expr>(SubExprs[Index]);
  }
  const Expr *getExpr(unsigned Index) const {
    assert((Index < NumExprs) && "Arg access out of range!");
    return cast<Expr>(SubExprs[Index]);
  }

  unsigned getShuffleMaskIdx(ASTContext &Ctx, unsigned N) {
    assert((N < NumExprs - 2) && "Shuffle idx out of range!");
    return getExpr(N+2)->getIntegerConstantExprValue(Ctx).getZExtValue();
  }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// ChooseExpr - GNU builtin-in function __builtin_choose_expr.
/// This AST node is similar to the conditional operator (?:) in C, with 
/// the following exceptions:
/// - the test expression much be a constant expression.
/// - the expression returned has it's type unaltered by promotion rules.
/// - does not evaluate the expression that was not chosen.
class ChooseExpr : public Expr {
  enum { COND, LHS, RHS, END_EXPR };
  Stmt* SubExprs[END_EXPR]; // Left/Middle/Right hand sides.
  SourceLocation BuiltinLoc, RParenLoc;
public:
  ChooseExpr(SourceLocation BLoc, Expr *cond, Expr *lhs, Expr *rhs, QualType t,
             SourceLocation RP)
    : Expr(ChooseExprClass, t),  
      BuiltinLoc(BLoc), RParenLoc(RP) {
      SubExprs[COND] = cond;
      SubExprs[LHS] = lhs;
      SubExprs[RHS] = rhs;
    }        
  
  /// isConditionTrue - Return true if the condition is true.  This is always
  /// statically knowable for a well-formed choosexpr.
  bool isConditionTrue(ASTContext &C) const;
  
  Expr *getCond() const { return cast<Expr>(SubExprs[COND]); }
  Expr *getLHS() const { return cast<Expr>(SubExprs[LHS]); }
  Expr *getRHS() const { return cast<Expr>(SubExprs[RHS]); }

  virtual SourceRange getSourceRange() const {
    return SourceRange(BuiltinLoc, RParenLoc);
  }
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ChooseExprClass; 
  }
  static bool classof(const ChooseExpr *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// OverloadExpr - Clang builtin function __builtin_overload.
/// This AST node provides a way to overload functions in C.
///
/// The first argument is required to be a constant expression, for the number
/// of arguments passed to each candidate function.
///
/// The next N arguments, where N is the value of the constant expression,
/// are the values to be passed as arguments.
///
/// The rest of the arguments are values of pointer to function type, which 
/// are the candidate functions for overloading.
///
/// The result is a equivalent to a CallExpr taking N arguments to the 
/// candidate function whose parameter types match the types of the N arguments.
///
/// example: float Z = __builtin_overload(2, X, Y, modf, mod, modl);
/// If X and Y are long doubles, Z will assigned the result of modl(X, Y);
/// If X and Y are floats, Z will be assigned the result of modf(X, Y);
class OverloadExpr : public Expr {
  // SubExprs - the list of values passed to the __builtin_overload function.
  // SubExpr[0] is a constant expression
  // SubExpr[1-N] are the parameters to pass to the matching function call
  // SubExpr[N-...] are the candidate functions, of type pointer to function.
  Stmt **SubExprs;

  // NumExprs - the size of the SubExprs array
  unsigned NumExprs;

  // The index of the matching candidate function
  unsigned FnIndex;

  SourceLocation BuiltinLoc;
  SourceLocation RParenLoc;
public:
  OverloadExpr(Expr **args, unsigned nexprs, unsigned idx, QualType t, 
               SourceLocation bloc, SourceLocation rploc)
    : Expr(OverloadExprClass, t), NumExprs(nexprs), FnIndex(idx),
      BuiltinLoc(bloc), RParenLoc(rploc) {
    SubExprs = new Stmt*[nexprs];
    for (unsigned i = 0; i != nexprs; ++i)
      SubExprs[i] = args[i];
  }
  ~OverloadExpr() {
    delete [] SubExprs;
  }

  /// arg_begin - Return a pointer to the list of arguments that will be passed
  /// to the matching candidate function, skipping over the initial constant
  /// expression.
  typedef ConstExprIterator const_arg_iterator;
  const_arg_iterator arg_begin() const { return &SubExprs[0]+1; }
  const_arg_iterator arg_end(ASTContext& Ctx) const {
    return &SubExprs[0]+1+getNumArgs(Ctx);
  }
  
  /// getNumArgs - Return the number of arguments to pass to the candidate
  /// functions.
  unsigned getNumArgs(ASTContext &Ctx) const {
    return getExpr(0)->getIntegerConstantExprValue(Ctx).getZExtValue();
  }

  /// getNumSubExprs - Return the size of the SubExprs array.  This includes the
  /// constant expression, the actual arguments passed in, and the function
  /// pointers.
  unsigned getNumSubExprs() const { return NumExprs; }
  
  /// getExpr - Return the Expr at the specified index.
  Expr *getExpr(unsigned Index) const {
    assert((Index < NumExprs) && "Arg access out of range!");
    return cast<Expr>(SubExprs[Index]);
  }
  
  /// getFn - Return the matching candidate function for this OverloadExpr.
  Expr *getFn() const { return cast<Expr>(SubExprs[FnIndex]); }
  
  virtual SourceRange getSourceRange() const {
    return SourceRange(BuiltinLoc, RParenLoc);
  }
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OverloadExprClass; 
  }
  static bool classof(const OverloadExpr *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// VAArgExpr, used for the builtin function __builtin_va_start.
class VAArgExpr : public Expr {
  Stmt *Val;
  SourceLocation BuiltinLoc, RParenLoc;
public:
  VAArgExpr(SourceLocation BLoc, Expr* e, QualType t, SourceLocation RPLoc)
    : Expr(VAArgExprClass, t),
      Val(e),
      BuiltinLoc(BLoc),
      RParenLoc(RPLoc) { }
  
  const Expr *getSubExpr() const { return cast<Expr>(Val); }
  Expr *getSubExpr() { return cast<Expr>(Val); }
  virtual SourceRange getSourceRange() const {
    return SourceRange(BuiltinLoc, RParenLoc);
  }  
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == VAArgExprClass;
  }
  static bool classof(const VAArgExpr *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();  
};
  
/// InitListExpr - used for struct and array initializers, such as:
///    struct foo x = { 1, { 2, 3 } };
///
/// Because C is somewhat loose with braces, the AST does not necessarily
/// directly model the C source.  Instead, the semantic analyzer aims to make
/// the InitListExprs match up with the type of the decl being initialized.  We
/// have the following exceptions:
///
///  1. Elements at the end of the list may be dropped from the initializer.  
///     These elements are defined to be initialized to zero.  For example:
///         int x[20] = { 1 };
///  2. Initializers may have excess initializers which are to be ignored by the
///     compiler.  For example:
///         int x[1] = { 1, 2 };
///  3. Redundant InitListExprs may be present around scalar elements.  These
///     always have a single element whose type is the same as the InitListExpr.
///     this can only happen for Type::isScalarType() types.
///         int x = { 1 };  int y[2] = { {1}, {2} };
///
class InitListExpr : public Expr {
  std::vector<Stmt *> InitExprs;
  SourceLocation LBraceLoc, RBraceLoc;
public:
  InitListExpr(SourceLocation lbraceloc, Expr **initexprs, unsigned numinits,
               SourceLocation rbraceloc);
  
  unsigned getNumInits() const { return InitExprs.size(); }
  
  const Expr* getInit(unsigned Init) const { 
    assert(Init < getNumInits() && "Initializer access out of range!");
    return cast<Expr>(InitExprs[Init]);
  }
  
  Expr* getInit(unsigned Init) { 
    assert(Init < getNumInits() && "Initializer access out of range!");
    return cast<Expr>(InitExprs[Init]);
  }
  
  void setInit(unsigned Init, Expr *expr) { 
    assert(Init < getNumInits() && "Initializer access out of range!");
    InitExprs[Init] = expr;
  }

  // Dynamic removal/addition (for constructing implicit InitExpr's).
  void removeInit(unsigned Init) {
    InitExprs.erase(InitExprs.begin()+Init);
  }
  void addInit(unsigned Init, Expr *expr) {
    InitExprs.insert(InitExprs.begin()+Init, expr);
  }

  // Explicit InitListExpr's originate from source code (and have valid source
  // locations). Implicit InitListExpr's are created by the semantic analyzer.
  bool isExplicit() {
    return LBraceLoc.isValid() && RBraceLoc.isValid();
  }
  
  virtual SourceRange getSourceRange() const {
    return SourceRange(LBraceLoc, RBraceLoc);
  } 
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == InitListExprClass; 
  }
  static bool classof(const InitListExpr *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static InitListExpr* CreateImpl(llvm::Deserializer& D, ASTContext& C);

private:
  // Used by serializer.
  InitListExpr() : Expr(InitListExprClass, QualType()) {}
};

//===----------------------------------------------------------------------===//
// Clang Extensions
//===----------------------------------------------------------------------===//


/// ExtVectorElementExpr - This represents access to specific elements of a
/// vector, and may occur on the left hand side or right hand side.  For example
/// the following is legal:  "V.xy = V.zw" if V is a 4 element extended vector.
///
class ExtVectorElementExpr : public Expr {
  Stmt *Base;
  IdentifierInfo &Accessor;
  SourceLocation AccessorLoc;
public:
  ExtVectorElementExpr(QualType ty, Expr *base, IdentifierInfo &accessor,
                       SourceLocation loc)
    : Expr(ExtVectorElementExprClass, ty), 
      Base(base), Accessor(accessor), AccessorLoc(loc) {}
                     
  const Expr *getBase() const { return cast<Expr>(Base); }
  Expr *getBase() { return cast<Expr>(Base); }
  
  IdentifierInfo &getAccessor() const { return Accessor; }
  
  /// getNumElements - Get the number of components being selected.
  unsigned getNumElements() const;
  
  /// containsDuplicateElements - Return true if any element access is
  /// repeated.
  bool containsDuplicateElements() const;
  
  /// getEncodedElementAccess - Encode the elements accessed into an llvm
  /// aggregate Constant of ConstantInt(s).
  void getEncodedElementAccess(llvm::SmallVectorImpl<unsigned> &Elts) const;
  
  virtual SourceRange getSourceRange() const {
    return SourceRange(getBase()->getLocStart(), AccessorLoc);
  }
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ExtVectorElementExprClass; 
  }
  static bool classof(const ExtVectorElementExpr *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};


/// BlockExpr - Represent a block literal with a syntax:
/// ^{ statement-body }   or   ^(int arg1, float arg2){ statement-body }
class BlockExpr : public Expr {
  SourceLocation CaretLocation;
  llvm::SmallVector<ParmVarDecl*, 8> Args;
  Stmt *Body;
public:
  BlockExpr(SourceLocation caretloc, QualType ty, ParmVarDecl **args, 
            unsigned numargs, CompoundStmt *body) : Expr(BlockExprClass, ty), 
            CaretLocation(caretloc), Args(args, args+numargs), Body(body) {}

  SourceLocation getCaretLocation() const { return CaretLocation; }

  /// getFunctionType - Return the underlying function type for this block.
  const FunctionType *getFunctionType() const;

  const CompoundStmt *getBody() const { return cast<CompoundStmt>(Body); }
  CompoundStmt *getBody() { return cast<CompoundStmt>(Body); }

  virtual SourceRange getSourceRange() const {
    return SourceRange(getCaretLocation(), Body->getLocEnd());
  }

  /// arg_iterator - Iterate over the ParmVarDecl's for the arguments to this
  /// block.
  typedef llvm::SmallVector<ParmVarDecl*, 8>::const_iterator arg_iterator;
  bool arg_empty() const { return Args.empty(); }
  arg_iterator arg_begin() const { return Args.begin(); }
  arg_iterator arg_end() const { return Args.end(); }
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == BlockExprClass;
  }
  static bool classof(const BlockExpr *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
    
  virtual void EmitImpl(llvm::Serializer& S) const;
  static BlockExpr* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};
    
/// BlockDeclRefExpr - A reference to a declared variable, function,
/// enum, etc.
class BlockDeclRefExpr : public Expr {
  ValueDecl *D; 
  SourceLocation Loc;
  bool IsByRef;
public:
  BlockDeclRefExpr(ValueDecl *d, QualType t, SourceLocation l, bool ByRef) : 
       Expr(BlockDeclRefExprClass, t), D(d), Loc(l), IsByRef(ByRef) {}
  
  ValueDecl *getDecl() { return D; }
  const ValueDecl *getDecl() const { return D; }
  virtual SourceRange getSourceRange() const { return SourceRange(Loc); }
  
  bool isByRef() const { return IsByRef; }
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == BlockDeclRefExprClass; 
  }
  static bool classof(const BlockDeclRefExpr *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static BlockDeclRefExpr* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

}  // end namespace clang

#endif
