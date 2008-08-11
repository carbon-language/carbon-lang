//===--- ExprCXX.h - Classes for representing expressions -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Expr interface and subclasses for C++ expressions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_EXPRCXX_H
#define LLVM_CLANG_AST_EXPRCXX_H

#include "clang/AST/Expr.h"
#include "clang/AST/Decl.h"

namespace clang {

//===--------------------------------------------------------------------===//
// C++ Expressions.
//===--------------------------------------------------------------------===//

/// CXXCastExpr - [C++ 5.2.7, 5.2.9, 5.2.10, 5.2.11] C++ Cast Operators.
/// 
class CXXCastExpr : public Expr {
public:
  enum Opcode {
    DynamicCast,
    StaticCast,
    ReinterpretCast,
    ConstCast
  };
private:
  QualType Ty;
  Opcode Opc;
  Stmt *Op;
  SourceLocation Loc; // the location of the casting op
public:
  CXXCastExpr(Opcode op, QualType ty, Expr *expr, SourceLocation l)
    : Expr(CXXCastExprClass, ty), Ty(ty), Opc(op), Op(expr), Loc(l) {}

  QualType getDestType() const { return Ty; }
  Expr *getSubExpr() const { return cast<Expr>(Op); }

  Opcode getOpcode() const { return Opc; }

  /// getOpcodeStr - Turn an Opcode enum value into the string it represents,
  /// e.g. "reinterpret_cast".
  static const char *getOpcodeStr(Opcode Op) {
    // FIXME: move out of line.
    switch (Op) {
    default: assert(0 && "Not a C++ cast expression");
    case CXXCastExpr::ConstCast:       return "const_cast";
    case CXXCastExpr::DynamicCast:     return "dynamic_cast";
    case CXXCastExpr::ReinterpretCast: return "reinterpret_cast";
    case CXXCastExpr::StaticCast:      return "static_cast";
    }
  }
  
  virtual SourceRange getSourceRange() const {
    return SourceRange(Loc, getSubExpr()->getSourceRange().getEnd());
  }
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == CXXCastExprClass;
  }
  static bool classof(const CXXCastExpr *) { return true; }
      
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// CXXBoolLiteralExpr - [C++ 2.13.5] C++ Boolean Literal.
/// 
class CXXBoolLiteralExpr : public Expr {
  bool Value;
  SourceLocation Loc;
public:
  CXXBoolLiteralExpr(bool val, QualType Ty, SourceLocation l) : 
    Expr(CXXBoolLiteralExprClass, Ty), Value(val), Loc(l) {}
  
  bool getValue() const { return Value; }

  virtual SourceRange getSourceRange() const { return SourceRange(Loc); }
    
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == CXXBoolLiteralExprClass;
  }
  static bool classof(const CXXBoolLiteralExpr *) { return true; }
      
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

///  CXXThrowExpr - [C++ 15] C++ Throw Expression.  This handles
///  'throw' and 'throw' assignment-expression.  When
///  assignment-expression isn't present, Op will be null.
///
class CXXThrowExpr : public Expr {
  Stmt *Op;
  SourceLocation ThrowLoc;
public:
  // Ty is the void type which is used as the result type of the
  // exepression.  The l is the location of the throw keyword.  expr
  // can by null, if the optional expression to throw isn't present.
  CXXThrowExpr(Expr *expr, QualType Ty, SourceLocation l) :
    Expr(CXXThrowExprClass, Ty), Op(expr), ThrowLoc(l) {}
  const Expr *getSubExpr() const { return cast_or_null<Expr>(Op); }
  Expr *getSubExpr() { return cast_or_null<Expr>(Op); }

  virtual SourceRange getSourceRange() const {
    if (getSubExpr() == 0)
      return SourceRange(ThrowLoc, ThrowLoc);
    return SourceRange(ThrowLoc, getSubExpr()->getSourceRange().getEnd());
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXThrowExprClass;
  }
  static bool classof(const CXXThrowExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// CXXDefaultArgExpr - C++ [dcl.fct.default]. This wraps up a
/// function call argument that was created from the corresponding
/// parameter's default argument, when the call did not explicitly
/// supply arguments for all of the parameters.
class CXXDefaultArgExpr : public Expr {
  ParmVarDecl *Param;
public:
  // Param is the parameter whose default argument is used by this
  // expression.
  explicit CXXDefaultArgExpr(ParmVarDecl *param) 
    : Expr(CXXDefaultArgExprClass, param->getDefaultArg()->getType()),
      Param(param) { }

  // Retrieve the parameter that the argument was created from.
  const ParmVarDecl *getParam() const { return Param; }
  ParmVarDecl *getParam() { return Param; }

  // Retrieve the actual argument to the function call.
  const Expr *getExpr() const { return Param->getDefaultArg(); }
  Expr *getExpr() { return Param->getDefaultArg(); }

  virtual SourceRange getSourceRange() const {
    // Default argument expressions have no representation in the
    // source, so they have an empty source range.
    return SourceRange();
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXDefaultArgExprClass;
  }
  static bool classof(const CXXDefaultArgExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();

  // Serialization
  virtual void EmitImpl(llvm::Serializer& S) const;
  static CXXDefaultArgExpr* CreateImpl(llvm::Deserializer& D,
                                       ASTContext& C);
};
}  // end namespace clang

#endif
