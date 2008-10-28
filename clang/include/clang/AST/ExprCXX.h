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

/// CXXNamedCastExpr - Abstract class common to all of the C++ "named"
/// casts, @c static_cast, @c dynamic_cast, @c reinterpret_cast, or @c
/// const_cast.
///
/// This abstract class is inherited by all of the classes
/// representing "named" casts, e.g., CXXStaticCastExpr,
/// CXXDynamicCastExpr, CXXReinterpretCastExpr, and CXXConstCastExpr.
class CXXNamedCastExpr : public ExplicitCastExpr {
private:
  SourceLocation Loc; // the location of the casting op

protected:
  CXXNamedCastExpr(StmtClass SC, QualType ty, Expr *op, QualType writtenTy, 
                   SourceLocation l)
    : ExplicitCastExpr(SC, ty, op, writtenTy), Loc(l) {}

public:
  const char *getCastName() const;

  virtual SourceRange getSourceRange() const {
    return SourceRange(Loc, getSubExpr()->getSourceRange().getEnd());
  }
  static bool classof(const Stmt *T) { 
    switch (T->getStmtClass()) {
    case CXXNamedCastExprClass:
    case CXXStaticCastExprClass:
    case CXXDynamicCastExprClass:
    case CXXReinterpretCastExprClass:
    case CXXConstCastExprClass:
      return true;
    default:
      return false;
    }
  }
  static bool classof(const CXXNamedCastExpr *) { return true; }
      
  virtual void EmitImpl(llvm::Serializer& S) const;
  static CXXNamedCastExpr *CreateImpl(llvm::Deserializer& D, ASTContext& C,
                                      StmtClass SC);
};

/// CXXStaticCastExpr - A C++ @c static_cast expression (C++ [expr.static.cast]).
/// 
/// This expression node represents a C++ static cast, e.g.,
/// @c static_cast<int>(1.0).
class CXXStaticCastExpr : public CXXNamedCastExpr {
public:
  CXXStaticCastExpr(QualType ty, Expr *op, QualType writtenTy, SourceLocation l)
    : CXXNamedCastExpr(CXXStaticCastExprClass, ty, op, writtenTy, l) {}

  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == CXXStaticCastExprClass;
  }
  static bool classof(const CXXStaticCastExpr *) { return true; }
};

/// CXXDynamicCastExpr - A C++ @c dynamic_cast expression
/// (C++ [expr.dynamic.cast]), which may perform a run-time check to 
/// determine how to perform the type cast.
/// 
/// This expression node represents a dynamic cast, e.g.,
/// @c dynamic_cast<Derived*>(BasePtr).
class CXXDynamicCastExpr : public CXXNamedCastExpr {
public:
  CXXDynamicCastExpr(QualType ty, Expr *op, QualType writtenTy, SourceLocation l)
    : CXXNamedCastExpr(CXXDynamicCastExprClass, ty, op, writtenTy, l) {}

  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == CXXDynamicCastExprClass;
  }
  static bool classof(const CXXDynamicCastExpr *) { return true; }
};

/// CXXReinterpretCastExpr - A C++ @c reinterpret_cast expression (C++
/// [expr.reinterpret.cast]), which provides a differently-typed view
/// of a value but performs no actual work at run time.
/// 
/// This expression node represents a reinterpret cast, e.g.,
/// @c reinterpret_cast<int>(VoidPtr).
class CXXReinterpretCastExpr : public CXXNamedCastExpr {
public:
  CXXReinterpretCastExpr(QualType ty, Expr *op, QualType writtenTy, 
                         SourceLocation l)
    : CXXNamedCastExpr(CXXReinterpretCastExprClass, ty, op, writtenTy, l) {}

  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == CXXReinterpretCastExprClass;
  }
  static bool classof(const CXXReinterpretCastExpr *) { return true; }
};

/// CXXConstCastExpr - A C++ @c const_cast expression (C++ [expr.const.cast]),
/// which can remove type qualifiers but does not change the underlying value.
/// 
/// This expression node represents a const cast, e.g.,
/// @c const_cast<char*>(PtrToConstChar).
class CXXConstCastExpr : public CXXNamedCastExpr {
public:
  CXXConstCastExpr(QualType ty, Expr *op, QualType writtenTy, 
                   SourceLocation l)
    : CXXNamedCastExpr(CXXConstCastExprClass, ty, op, writtenTy, l) {}

  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == CXXConstCastExprClass;
  }
  static bool classof(const CXXConstCastExpr *) { return true; }
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

/// CXXFunctionalCastExpr - Represents an explicit C++ type conversion
/// that uses "functional" notion (C++ [expr.type.conv]). Example: @c
/// x = int(0.5);
class CXXFunctionalCastExpr : public ExplicitCastExpr {
  SourceLocation TyBeginLoc;
  SourceLocation RParenLoc;
public:
  CXXFunctionalCastExpr(QualType ty, QualType writtenTy, 
                        SourceLocation tyBeginLoc, Expr *castExpr,
                        SourceLocation rParenLoc) : 
    ExplicitCastExpr(CXXFunctionalCastExprClass, ty, castExpr, writtenTy),
    TyBeginLoc(tyBeginLoc), RParenLoc(rParenLoc) {}

  SourceLocation getTypeBeginLoc() const { return TyBeginLoc; }
  SourceLocation getRParenLoc() const { return RParenLoc; }
  
  virtual SourceRange getSourceRange() const {
    return SourceRange(TyBeginLoc, RParenLoc);
  }
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == CXXFunctionalCastExprClass; 
  }
  static bool classof(const CXXFunctionalCastExpr *) { return true; }
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static CXXFunctionalCastExpr *
      CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

/// CXXZeroInitValueExpr - [C++ 5.2.3p2]
/// Expression "T()" which creates a value-initialized Rvalue of non-class
/// type T.
///
class CXXZeroInitValueExpr : public Expr {
  SourceLocation TyBeginLoc;
  SourceLocation RParenLoc;

public:
  CXXZeroInitValueExpr(QualType ty, SourceLocation tyBeginLoc,
                       SourceLocation rParenLoc ) : 
    Expr(CXXZeroInitValueExprClass, ty),
    TyBeginLoc(tyBeginLoc), RParenLoc(rParenLoc) {}
  
  SourceLocation getTypeBeginLoc() const { return TyBeginLoc; }
  SourceLocation getRParenLoc() const { return RParenLoc; }

  virtual SourceRange getSourceRange() const {
    return SourceRange(TyBeginLoc, RParenLoc);
  }
    
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == CXXZeroInitValueExprClass;
  }
  static bool classof(const CXXZeroInitValueExpr *) { return true; }
      
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();

  virtual void EmitImpl(llvm::Serializer& S) const;
  static CXXZeroInitValueExpr *
      CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

/// CXXConditionDeclExpr - Condition declaration of a if/switch/while/for
/// statement, e.g: "if (int x = f()) {...}".
/// The main difference with DeclRefExpr is that CXXConditionDeclExpr owns the
/// decl that it references.
///
class CXXConditionDeclExpr : public DeclRefExpr {
public:
  CXXConditionDeclExpr(SourceLocation startLoc,
                       SourceLocation eqLoc, VarDecl *var)
    : DeclRefExpr(CXXConditionDeclExprClass, var, 
                  var->getType().getNonReferenceType(), startLoc) {}

  virtual void Destroy(ASTContext& Ctx);

  SourceLocation getStartLoc() const { return getLocation(); }
  
  VarDecl *getVarDecl() { return cast<VarDecl>(getDecl()); }
  const VarDecl *getVarDecl() const { return cast<VarDecl>(getDecl()); }

  virtual SourceRange getSourceRange() const {
    return SourceRange(getStartLoc(), getVarDecl()->getInit()->getLocEnd());
  }
    
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == CXXConditionDeclExprClass;
  }
  static bool classof(const CXXConditionDeclExpr *) { return true; }
      
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();

  // FIXME: Implement these.
  //virtual void EmitImpl(llvm::Serializer& S) const;
  //static CXXConditionDeclExpr *
  //    CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

}  // end namespace clang

#endif
