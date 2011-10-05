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

#include "clang/Basic/TypeTraits.h"
#include "clang/Basic/ExpressionTraits.h"
#include "clang/AST/Expr.h"
#include "clang/AST/UnresolvedSet.h"
#include "clang/AST/TemplateBase.h"

namespace clang {

class CXXConstructorDecl;
class CXXDestructorDecl;
class CXXMethodDecl;
class CXXTemporary;
class TemplateArgumentListInfo;

//===--------------------------------------------------------------------===//
// C++ Expressions.
//===--------------------------------------------------------------------===//

/// \brief A call to an overloaded operator written using operator
/// syntax.
///
/// Represents a call to an overloaded operator written using operator
/// syntax, e.g., "x + y" or "*p". While semantically equivalent to a
/// normal call, this AST node provides better information about the
/// syntactic representation of the call.
///
/// In a C++ template, this expression node kind will be used whenever
/// any of the arguments are type-dependent. In this case, the
/// function itself will be a (possibly empty) set of functions and
/// function templates that were found by name lookup at template
/// definition time.
class CXXOperatorCallExpr : public CallExpr {
  /// \brief The overloaded operator.
  OverloadedOperatorKind Operator;

public:
  CXXOperatorCallExpr(ASTContext& C, OverloadedOperatorKind Op, Expr *fn,
                      Expr **args, unsigned numargs, QualType t,
                      ExprValueKind VK, SourceLocation operatorloc)
    : CallExpr(C, CXXOperatorCallExprClass, fn, 0, args, numargs, t, VK,
               operatorloc),
      Operator(Op) {}
  explicit CXXOperatorCallExpr(ASTContext& C, EmptyShell Empty) :
    CallExpr(C, CXXOperatorCallExprClass, Empty) { }


  /// getOperator - Returns the kind of overloaded operator that this
  /// expression refers to.
  OverloadedOperatorKind getOperator() const { return Operator; }
  void setOperator(OverloadedOperatorKind Kind) { Operator = Kind; }

  /// getOperatorLoc - Returns the location of the operator symbol in
  /// the expression. When @c getOperator()==OO_Call, this is the
  /// location of the right parentheses; when @c
  /// getOperator()==OO_Subscript, this is the location of the right
  /// bracket.
  SourceLocation getOperatorLoc() const { return getRParenLoc(); }

  SourceRange getSourceRange() const;

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXOperatorCallExprClass;
  }
  static bool classof(const CXXOperatorCallExpr *) { return true; }
};

/// CXXMemberCallExpr - Represents a call to a member function that
/// may be written either with member call syntax (e.g., "obj.func()"
/// or "objptr->func()") or with normal function-call syntax
/// ("func()") within a member function that ends up calling a member
/// function. The callee in either case is a MemberExpr that contains
/// both the object argument and the member function, while the
/// arguments are the arguments within the parentheses (not including
/// the object argument).
class CXXMemberCallExpr : public CallExpr {
public:
  CXXMemberCallExpr(ASTContext &C, Expr *fn, Expr **args, unsigned numargs,
                    QualType t, ExprValueKind VK, SourceLocation RP)
    : CallExpr(C, CXXMemberCallExprClass, fn, 0, args, numargs, t, VK, RP) {}

  CXXMemberCallExpr(ASTContext &C, EmptyShell Empty)
    : CallExpr(C, CXXMemberCallExprClass, Empty) { }

  /// getImplicitObjectArgument - Retrieves the implicit object
  /// argument for the member call. For example, in "x.f(5)", this
  /// operation would return "x".
  Expr *getImplicitObjectArgument() const;
  
  /// Retrieves the declaration of the called method.
  CXXMethodDecl *getMethodDecl() const;

  /// getRecordDecl - Retrieves the CXXRecordDecl for the underlying type of
  /// the implicit object argument. Note that this is may not be the same
  /// declaration as that of the class context of the CXXMethodDecl which this
  /// function is calling.
  /// FIXME: Returns 0 for member pointer call exprs.
  CXXRecordDecl *getRecordDecl();

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXMemberCallExprClass;
  }
  static bool classof(const CXXMemberCallExpr *) { return true; }
};

/// CUDAKernelCallExpr - Represents a call to a CUDA kernel function.
class CUDAKernelCallExpr : public CallExpr {
private:
  enum { CONFIG, END_PREARG };

public:
  CUDAKernelCallExpr(ASTContext &C, Expr *fn, CallExpr *Config,
                     Expr **args, unsigned numargs, QualType t,
                     ExprValueKind VK, SourceLocation RP)
    : CallExpr(C, CUDAKernelCallExprClass, fn, END_PREARG, args, numargs, t, VK,
               RP) {
    setConfig(Config);
  }

  CUDAKernelCallExpr(ASTContext &C, EmptyShell Empty)
    : CallExpr(C, CUDAKernelCallExprClass, END_PREARG, Empty) { }

  const CallExpr *getConfig() const {
    return cast_or_null<CallExpr>(getPreArg(CONFIG));
  }
  CallExpr *getConfig() { return cast_or_null<CallExpr>(getPreArg(CONFIG)); }
  void setConfig(CallExpr *E) { setPreArg(CONFIG, E); }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CUDAKernelCallExprClass;
  }
  static bool classof(const CUDAKernelCallExpr *) { return true; }
};

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
  SourceLocation RParenLoc; // the location of the right parenthesis
  
protected:
  CXXNamedCastExpr(StmtClass SC, QualType ty, ExprValueKind VK,
                   CastKind kind, Expr *op, unsigned PathSize,
                   TypeSourceInfo *writtenTy, SourceLocation l,
                   SourceLocation RParenLoc)
    : ExplicitCastExpr(SC, ty, VK, kind, op, PathSize, writtenTy), Loc(l),
      RParenLoc(RParenLoc) {}

  explicit CXXNamedCastExpr(StmtClass SC, EmptyShell Shell, unsigned PathSize)
    : ExplicitCastExpr(SC, Shell, PathSize) { }

  friend class ASTStmtReader;
  
public:
  const char *getCastName() const;

  /// \brief Retrieve the location of the cast operator keyword, e.g.,
  /// "static_cast".
  SourceLocation getOperatorLoc() const { return Loc; }

  /// \brief Retrieve the location of the closing parenthesis.
  SourceLocation getRParenLoc() const { return RParenLoc; }
  
  SourceRange getSourceRange() const {
    return SourceRange(Loc, RParenLoc);
  }
  static bool classof(const Stmt *T) {
    switch (T->getStmtClass()) {
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
};

/// CXXStaticCastExpr - A C++ @c static_cast expression (C++ [expr.static.cast]).
///
/// This expression node represents a C++ static cast, e.g.,
/// @c static_cast<int>(1.0).
class CXXStaticCastExpr : public CXXNamedCastExpr {
  CXXStaticCastExpr(QualType ty, ExprValueKind vk, CastKind kind, Expr *op,
                    unsigned pathSize, TypeSourceInfo *writtenTy,
                    SourceLocation l, SourceLocation RParenLoc)
    : CXXNamedCastExpr(CXXStaticCastExprClass, ty, vk, kind, op, pathSize,
                       writtenTy, l, RParenLoc) {}

  explicit CXXStaticCastExpr(EmptyShell Empty, unsigned PathSize)
    : CXXNamedCastExpr(CXXStaticCastExprClass, Empty, PathSize) { }

public:
  static CXXStaticCastExpr *Create(ASTContext &Context, QualType T,
                                   ExprValueKind VK, CastKind K, Expr *Op,
                                   const CXXCastPath *Path,
                                   TypeSourceInfo *Written, SourceLocation L, 
                                   SourceLocation RParenLoc);
  static CXXStaticCastExpr *CreateEmpty(ASTContext &Context,
                                        unsigned PathSize);

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
  CXXDynamicCastExpr(QualType ty, ExprValueKind VK, CastKind kind,
                     Expr *op, unsigned pathSize, TypeSourceInfo *writtenTy,
                     SourceLocation l, SourceLocation RParenLoc)
    : CXXNamedCastExpr(CXXDynamicCastExprClass, ty, VK, kind, op, pathSize,
                       writtenTy, l, RParenLoc) {}

  explicit CXXDynamicCastExpr(EmptyShell Empty, unsigned pathSize)
    : CXXNamedCastExpr(CXXDynamicCastExprClass, Empty, pathSize) { }

public:
  static CXXDynamicCastExpr *Create(ASTContext &Context, QualType T,
                                    ExprValueKind VK, CastKind Kind, Expr *Op,
                                    const CXXCastPath *Path,
                                    TypeSourceInfo *Written, SourceLocation L, 
                                    SourceLocation RParenLoc);
  
  static CXXDynamicCastExpr *CreateEmpty(ASTContext &Context,
                                         unsigned pathSize);

  bool isAlwaysNull() const;

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
  CXXReinterpretCastExpr(QualType ty, ExprValueKind vk, CastKind kind,
                         Expr *op, unsigned pathSize,
                         TypeSourceInfo *writtenTy, SourceLocation l, 
                         SourceLocation RParenLoc)
    : CXXNamedCastExpr(CXXReinterpretCastExprClass, ty, vk, kind, op,
                       pathSize, writtenTy, l, RParenLoc) {}

  CXXReinterpretCastExpr(EmptyShell Empty, unsigned pathSize)
    : CXXNamedCastExpr(CXXReinterpretCastExprClass, Empty, pathSize) { }

public:
  static CXXReinterpretCastExpr *Create(ASTContext &Context, QualType T,
                                        ExprValueKind VK, CastKind Kind,
                                        Expr *Op, const CXXCastPath *Path,
                                 TypeSourceInfo *WrittenTy, SourceLocation L, 
                                        SourceLocation RParenLoc);
  static CXXReinterpretCastExpr *CreateEmpty(ASTContext &Context,
                                             unsigned pathSize);

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
  CXXConstCastExpr(QualType ty, ExprValueKind VK, Expr *op,
                   TypeSourceInfo *writtenTy, SourceLocation l, 
                   SourceLocation RParenLoc)
    : CXXNamedCastExpr(CXXConstCastExprClass, ty, VK, CK_NoOp, op, 
                       0, writtenTy, l, RParenLoc) {}

  explicit CXXConstCastExpr(EmptyShell Empty)
    : CXXNamedCastExpr(CXXConstCastExprClass, Empty, 0) { }

public:
  static CXXConstCastExpr *Create(ASTContext &Context, QualType T,
                                  ExprValueKind VK, Expr *Op,
                                  TypeSourceInfo *WrittenTy, SourceLocation L, 
                                  SourceLocation RParenLoc);
  static CXXConstCastExpr *CreateEmpty(ASTContext &Context);

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
    Expr(CXXBoolLiteralExprClass, Ty, VK_RValue, OK_Ordinary, false, false,
         false, false),
    Value(val), Loc(l) {}

  explicit CXXBoolLiteralExpr(EmptyShell Empty)
    : Expr(CXXBoolLiteralExprClass, Empty) { }

  bool getValue() const { return Value; }
  void setValue(bool V) { Value = V; }

  SourceRange getSourceRange() const { return SourceRange(Loc); }

  SourceLocation getLocation() const { return Loc; }
  void setLocation(SourceLocation L) { Loc = L; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXBoolLiteralExprClass;
  }
  static bool classof(const CXXBoolLiteralExpr *) { return true; }

  // Iterators
  child_range children() { return child_range(); }
};

/// CXXNullPtrLiteralExpr - [C++0x 2.14.7] C++ Pointer Literal
class CXXNullPtrLiteralExpr : public Expr {
  SourceLocation Loc;
public:
  CXXNullPtrLiteralExpr(QualType Ty, SourceLocation l) :
    Expr(CXXNullPtrLiteralExprClass, Ty, VK_RValue, OK_Ordinary, false, false,
         false, false),
    Loc(l) {}

  explicit CXXNullPtrLiteralExpr(EmptyShell Empty)
    : Expr(CXXNullPtrLiteralExprClass, Empty) { }

  SourceRange getSourceRange() const { return SourceRange(Loc); }

  SourceLocation getLocation() const { return Loc; }
  void setLocation(SourceLocation L) { Loc = L; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXNullPtrLiteralExprClass;
  }
  static bool classof(const CXXNullPtrLiteralExpr *) { return true; }

  child_range children() { return child_range(); }
};

/// CXXTypeidExpr - A C++ @c typeid expression (C++ [expr.typeid]), which gets
/// the type_info that corresponds to the supplied type, or the (possibly
/// dynamic) type of the supplied expression.
///
/// This represents code like @c typeid(int) or @c typeid(*objPtr)
class CXXTypeidExpr : public Expr {
private:
  llvm::PointerUnion<Stmt *, TypeSourceInfo *> Operand;
  SourceRange Range;

public:
  CXXTypeidExpr(QualType Ty, TypeSourceInfo *Operand, SourceRange R)
    : Expr(CXXTypeidExprClass, Ty, VK_LValue, OK_Ordinary,
           // typeid is never type-dependent (C++ [temp.dep.expr]p4)
           false,
           // typeid is value-dependent if the type or expression are dependent
           Operand->getType()->isDependentType(),
           Operand->getType()->isInstantiationDependentType(),
           Operand->getType()->containsUnexpandedParameterPack()),
      Operand(Operand), Range(R) { }
  
  CXXTypeidExpr(QualType Ty, Expr *Operand, SourceRange R)
    : Expr(CXXTypeidExprClass, Ty, VK_LValue, OK_Ordinary,
        // typeid is never type-dependent (C++ [temp.dep.expr]p4)
           false,
        // typeid is value-dependent if the type or expression are dependent
           Operand->isTypeDependent() || Operand->isValueDependent(),
           Operand->isInstantiationDependent(),
           Operand->containsUnexpandedParameterPack()),
      Operand(Operand), Range(R) { }

  CXXTypeidExpr(EmptyShell Empty, bool isExpr)
    : Expr(CXXTypeidExprClass, Empty) {
    if (isExpr)
      Operand = (Expr*)0;
    else
      Operand = (TypeSourceInfo*)0;
  }
  
  bool isTypeOperand() const { return Operand.is<TypeSourceInfo *>(); }
  
  /// \brief Retrieves the type operand of this typeid() expression after
  /// various required adjustments (removing reference types, cv-qualifiers).
  QualType getTypeOperand() const;

  /// \brief Retrieve source information for the type operand.
  TypeSourceInfo *getTypeOperandSourceInfo() const {
    assert(isTypeOperand() && "Cannot call getTypeOperand for typeid(expr)");
    return Operand.get<TypeSourceInfo *>();
  }

  void setTypeOperandSourceInfo(TypeSourceInfo *TSI) {
    assert(isTypeOperand() && "Cannot call getTypeOperand for typeid(expr)");
    Operand = TSI;
  }
  
  Expr *getExprOperand() const {
    assert(!isTypeOperand() && "Cannot call getExprOperand for typeid(type)");
    return static_cast<Expr*>(Operand.get<Stmt *>());
  }
  
  void setExprOperand(Expr *E) {
    assert(!isTypeOperand() && "Cannot call getExprOperand for typeid(type)");
    Operand = E;
  }
  
  SourceRange getSourceRange() const { return Range; }
  void setSourceRange(SourceRange R) { Range = R; }
  
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXTypeidExprClass;
  }
  static bool classof(const CXXTypeidExpr *) { return true; }

  // Iterators
  child_range children() {
    if (isTypeOperand()) return child_range();
    Stmt **begin = reinterpret_cast<Stmt**>(&Operand);
    return child_range(begin, begin + 1);
  }
};

/// CXXUuidofExpr - A microsoft C++ @c __uuidof expression, which gets
/// the _GUID that corresponds to the supplied type or expression.
///
/// This represents code like @c __uuidof(COMTYPE) or @c __uuidof(*comPtr)
class CXXUuidofExpr : public Expr {
private:
  llvm::PointerUnion<Stmt *, TypeSourceInfo *> Operand;
  SourceRange Range;

public:
  CXXUuidofExpr(QualType Ty, TypeSourceInfo *Operand, SourceRange R)
    : Expr(CXXUuidofExprClass, Ty, VK_LValue, OK_Ordinary,
           false, Operand->getType()->isDependentType(),
           Operand->getType()->isInstantiationDependentType(),
           Operand->getType()->containsUnexpandedParameterPack()),
      Operand(Operand), Range(R) { }
  
  CXXUuidofExpr(QualType Ty, Expr *Operand, SourceRange R)
    : Expr(CXXUuidofExprClass, Ty, VK_LValue, OK_Ordinary,
           false, Operand->isTypeDependent(),
           Operand->isInstantiationDependent(),
           Operand->containsUnexpandedParameterPack()),
      Operand(Operand), Range(R) { }

  CXXUuidofExpr(EmptyShell Empty, bool isExpr)
    : Expr(CXXUuidofExprClass, Empty) {
    if (isExpr)
      Operand = (Expr*)0;
    else
      Operand = (TypeSourceInfo*)0;
  }
  
  bool isTypeOperand() const { return Operand.is<TypeSourceInfo *>(); }
  
  /// \brief Retrieves the type operand of this __uuidof() expression after
  /// various required adjustments (removing reference types, cv-qualifiers).
  QualType getTypeOperand() const;

  /// \brief Retrieve source information for the type operand.
  TypeSourceInfo *getTypeOperandSourceInfo() const {
    assert(isTypeOperand() && "Cannot call getTypeOperand for __uuidof(expr)");
    return Operand.get<TypeSourceInfo *>();
  }

  void setTypeOperandSourceInfo(TypeSourceInfo *TSI) {
    assert(isTypeOperand() && "Cannot call getTypeOperand for __uuidof(expr)");
    Operand = TSI;
  }
  
  Expr *getExprOperand() const {
    assert(!isTypeOperand() && "Cannot call getExprOperand for __uuidof(type)");
    return static_cast<Expr*>(Operand.get<Stmt *>());
  }
  
  void setExprOperand(Expr *E) {
    assert(!isTypeOperand() && "Cannot call getExprOperand for __uuidof(type)");
    Operand = E;
  }

  SourceRange getSourceRange() const { return Range; }
  void setSourceRange(SourceRange R) { Range = R; }
  
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXUuidofExprClass;
  }
  static bool classof(const CXXUuidofExpr *) { return true; }

  // Iterators
  child_range children() {
    if (isTypeOperand()) return child_range();
    Stmt **begin = reinterpret_cast<Stmt**>(&Operand);
    return child_range(begin, begin + 1);
  }
};

/// CXXThisExpr - Represents the "this" expression in C++, which is a
/// pointer to the object on which the current member function is
/// executing (C++ [expr.prim]p3). Example:
///
/// @code
/// class Foo {
/// public:
///   void bar();
///   void test() { this->bar(); }
/// };
/// @endcode
class CXXThisExpr : public Expr {
  SourceLocation Loc;
  bool Implicit : 1;
  
public:
  CXXThisExpr(SourceLocation L, QualType Type, bool isImplicit)
    : Expr(CXXThisExprClass, Type, VK_RValue, OK_Ordinary,
           // 'this' is type-dependent if the class type of the enclosing
           // member function is dependent (C++ [temp.dep.expr]p2)
           Type->isDependentType(), Type->isDependentType(),
           Type->isInstantiationDependentType(),
           /*ContainsUnexpandedParameterPack=*/false),
      Loc(L), Implicit(isImplicit) { }

  CXXThisExpr(EmptyShell Empty) : Expr(CXXThisExprClass, Empty) {}

  SourceLocation getLocation() const { return Loc; }
  void setLocation(SourceLocation L) { Loc = L; }

  SourceRange getSourceRange() const { return SourceRange(Loc); }

  bool isImplicit() const { return Implicit; }
  void setImplicit(bool I) { Implicit = I; }
  
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXThisExprClass;
  }
  static bool classof(const CXXThisExpr *) { return true; }

  // Iterators
  child_range children() { return child_range(); }
};

///  CXXThrowExpr - [C++ 15] C++ Throw Expression.  This handles
///  'throw' and 'throw' assignment-expression.  When
///  assignment-expression isn't present, Op will be null.
///
class CXXThrowExpr : public Expr {
  Stmt *Op;
  SourceLocation ThrowLoc;
  /// \brief Whether the thrown variable (if any) is in scope.
  unsigned IsThrownVariableInScope : 1;
  
  friend class ASTStmtReader;
  
public:
  // Ty is the void type which is used as the result type of the
  // exepression.  The l is the location of the throw keyword.  expr
  // can by null, if the optional expression to throw isn't present.
  CXXThrowExpr(Expr *expr, QualType Ty, SourceLocation l,
               bool IsThrownVariableInScope) :
    Expr(CXXThrowExprClass, Ty, VK_RValue, OK_Ordinary, false, false,
         expr && expr->isInstantiationDependent(),
         expr && expr->containsUnexpandedParameterPack()),
    Op(expr), ThrowLoc(l), IsThrownVariableInScope(IsThrownVariableInScope) {}
  CXXThrowExpr(EmptyShell Empty) : Expr(CXXThrowExprClass, Empty) {}

  const Expr *getSubExpr() const { return cast_or_null<Expr>(Op); }
  Expr *getSubExpr() { return cast_or_null<Expr>(Op); }

  SourceLocation getThrowLoc() const { return ThrowLoc; }

  /// \brief Determines whether the variable thrown by this expression (if any!)
  /// is within the innermost try block.
  ///
  /// This information is required to determine whether the NRVO can apply to
  /// this variable.
  bool isThrownVariableInScope() const { return IsThrownVariableInScope; }
  
  SourceRange getSourceRange() const {
    if (getSubExpr() == 0)
      return SourceRange(ThrowLoc, ThrowLoc);
    return SourceRange(ThrowLoc, getSubExpr()->getSourceRange().getEnd());
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXThrowExprClass;
  }
  static bool classof(const CXXThrowExpr *) { return true; }

  // Iterators
  child_range children() {
    return child_range(&Op, Op ? &Op+1 : &Op);
  }
};

/// CXXDefaultArgExpr - C++ [dcl.fct.default]. This wraps up a
/// function call argument that was created from the corresponding
/// parameter's default argument, when the call did not explicitly
/// supply arguments for all of the parameters.
class CXXDefaultArgExpr : public Expr {
  /// \brief The parameter whose default is being used.
  ///
  /// When the bit is set, the subexpression is stored after the 
  /// CXXDefaultArgExpr itself. When the bit is clear, the parameter's
  /// actual default expression is the subexpression.
  llvm::PointerIntPair<ParmVarDecl *, 1, bool> Param;

  /// \brief The location where the default argument expression was used.
  SourceLocation Loc;
  
  CXXDefaultArgExpr(StmtClass SC, SourceLocation Loc, ParmVarDecl *param)
    : Expr(SC, 
           param->hasUnparsedDefaultArg()
             ? param->getType().getNonReferenceType()
             : param->getDefaultArg()->getType(),
           param->getDefaultArg()->getValueKind(),
           param->getDefaultArg()->getObjectKind(), false, false, false, false),
      Param(param, false), Loc(Loc) { }

  CXXDefaultArgExpr(StmtClass SC, SourceLocation Loc, ParmVarDecl *param, 
                    Expr *SubExpr)
    : Expr(SC, SubExpr->getType(),
           SubExpr->getValueKind(), SubExpr->getObjectKind(),
           false, false, false, false), 
      Param(param, true), Loc(Loc) {
    *reinterpret_cast<Expr **>(this + 1) = SubExpr;
  }
  
public:
  CXXDefaultArgExpr(EmptyShell Empty) : Expr(CXXDefaultArgExprClass, Empty) {}

  
  // Param is the parameter whose default argument is used by this
  // expression.
  static CXXDefaultArgExpr *Create(ASTContext &C, SourceLocation Loc,
                                   ParmVarDecl *Param) {
    return new (C) CXXDefaultArgExpr(CXXDefaultArgExprClass, Loc, Param);
  }

  // Param is the parameter whose default argument is used by this
  // expression, and SubExpr is the expression that will actually be used.
  static CXXDefaultArgExpr *Create(ASTContext &C, 
                                   SourceLocation Loc,
                                   ParmVarDecl *Param, 
                                   Expr *SubExpr);
  
  // Retrieve the parameter that the argument was created from.
  const ParmVarDecl *getParam() const { return Param.getPointer(); }
  ParmVarDecl *getParam() { return Param.getPointer(); }
  
  // Retrieve the actual argument to the function call.
  const Expr *getExpr() const { 
    if (Param.getInt())
      return *reinterpret_cast<Expr const * const*> (this + 1);
    return getParam()->getDefaultArg(); 
  }
  Expr *getExpr() { 
    if (Param.getInt())
      return *reinterpret_cast<Expr **> (this + 1);
    return getParam()->getDefaultArg(); 
  }

  /// \brief Retrieve the location where this default argument was actually 
  /// used.
  SourceLocation getUsedLocation() const { return Loc; }
  
  SourceRange getSourceRange() const {
    // Default argument expressions have no representation in the
    // source, so they have an empty source range.
    return SourceRange();
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXDefaultArgExprClass;
  }
  static bool classof(const CXXDefaultArgExpr *) { return true; }

  // Iterators
  child_range children() { return child_range(); }

  friend class ASTStmtReader;
  friend class ASTStmtWriter;
};

/// CXXTemporary - Represents a C++ temporary.
class CXXTemporary {
  /// Destructor - The destructor that needs to be called.
  const CXXDestructorDecl *Destructor;

  CXXTemporary(const CXXDestructorDecl *destructor)
    : Destructor(destructor) { }

public:
  static CXXTemporary *Create(ASTContext &C,
                              const CXXDestructorDecl *Destructor);

  const CXXDestructorDecl *getDestructor() const { return Destructor; }
};

/// \brief Represents binding an expression to a temporary.
///
/// This ensures the destructor is called for the temporary. It should only be
/// needed for non-POD, non-trivially destructable class types. For example:
///
/// \code
///   struct S {
///     S() { }  // User defined constructor makes S non-POD.
///     ~S() { } // User defined destructor makes it non-trivial.
///   };
///   void test() {
///     const S &s_ref = S(); // Requires a CXXBindTemporaryExpr.
///   }
/// \endcode
class CXXBindTemporaryExpr : public Expr {
  CXXTemporary *Temp;

  Stmt *SubExpr;

  CXXBindTemporaryExpr(CXXTemporary *temp, Expr* SubExpr)
   : Expr(CXXBindTemporaryExprClass, SubExpr->getType(),
          VK_RValue, OK_Ordinary, SubExpr->isTypeDependent(), 
          SubExpr->isValueDependent(),
          SubExpr->isInstantiationDependent(),
          SubExpr->containsUnexpandedParameterPack()),
     Temp(temp), SubExpr(SubExpr) { }

public:
  CXXBindTemporaryExpr(EmptyShell Empty)
    : Expr(CXXBindTemporaryExprClass, Empty), Temp(0), SubExpr(0) {}
  
  static CXXBindTemporaryExpr *Create(ASTContext &C, CXXTemporary *Temp,
                                      Expr* SubExpr);

  CXXTemporary *getTemporary() { return Temp; }
  const CXXTemporary *getTemporary() const { return Temp; }
  void setTemporary(CXXTemporary *T) { Temp = T; }

  const Expr *getSubExpr() const { return cast<Expr>(SubExpr); }
  Expr *getSubExpr() { return cast<Expr>(SubExpr); }
  void setSubExpr(Expr *E) { SubExpr = E; }

  SourceRange getSourceRange() const { 
    return SubExpr->getSourceRange();
  }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXBindTemporaryExprClass;
  }
  static bool classof(const CXXBindTemporaryExpr *) { return true; }

  // Iterators
  child_range children() { return child_range(&SubExpr, &SubExpr + 1); }
};

/// CXXConstructExpr - Represents a call to a C++ constructor.
class CXXConstructExpr : public Expr {
public:
  enum ConstructionKind {
    CK_Complete,
    CK_NonVirtualBase,
    CK_VirtualBase,
    CK_Delegating
  };
    
private:
  CXXConstructorDecl *Constructor;

  SourceLocation Loc;
  SourceRange ParenRange;
  unsigned NumArgs : 16;
  bool Elidable : 1;
  bool HadMultipleCandidates : 1;
  bool ZeroInitialization : 1;
  unsigned ConstructKind : 2;
  Stmt **Args;

protected:
  CXXConstructExpr(ASTContext &C, StmtClass SC, QualType T,
                   SourceLocation Loc,
                   CXXConstructorDecl *d, bool elidable,
                   Expr **args, unsigned numargs,
                   bool HadMultipleCandidates,
                   bool ZeroInitialization = false,
                   ConstructionKind ConstructKind = CK_Complete,
                   SourceRange ParenRange = SourceRange());

  /// \brief Construct an empty C++ construction expression.
  CXXConstructExpr(StmtClass SC, EmptyShell Empty)
    : Expr(SC, Empty), Constructor(0), NumArgs(0), Elidable(0),
      HadMultipleCandidates(false), ZeroInitialization(0),
      ConstructKind(0), Args(0) { }

public:
  /// \brief Construct an empty C++ construction expression.
  explicit CXXConstructExpr(EmptyShell Empty)
    : Expr(CXXConstructExprClass, Empty), Constructor(0),
      NumArgs(0), Elidable(0), HadMultipleCandidates(false),
      ZeroInitialization(0), ConstructKind(0), Args(0) { }

  static CXXConstructExpr *Create(ASTContext &C, QualType T,
                                  SourceLocation Loc,
                                  CXXConstructorDecl *D, bool Elidable,
                                  Expr **Args, unsigned NumArgs,
                                  bool HadMultipleCandidates,
                                  bool ZeroInitialization = false,
                                  ConstructionKind ConstructKind = CK_Complete,
                                  SourceRange ParenRange = SourceRange());


  CXXConstructorDecl* getConstructor() const { return Constructor; }
  void setConstructor(CXXConstructorDecl *C) { Constructor = C; }
  
  SourceLocation getLocation() const { return Loc; }
  void setLocation(SourceLocation Loc) { this->Loc = Loc; }
  
  /// \brief Whether this construction is elidable.
  bool isElidable() const { return Elidable; }
  void setElidable(bool E) { Elidable = E; }

  /// \brief Whether the referred constructor was resolved from
  /// an overloaded set having size greater than 1.
  bool hadMultipleCandidates() const { return HadMultipleCandidates; }
  void setHadMultipleCandidates(bool V) { HadMultipleCandidates = V; }

  /// \brief Whether this construction first requires
  /// zero-initialization before the initializer is called.
  bool requiresZeroInitialization() const { return ZeroInitialization; }
  void setRequiresZeroInitialization(bool ZeroInit) {
    ZeroInitialization = ZeroInit;
  }
  
  /// \brief Determines whether this constructor is actually constructing
  /// a base class (rather than a complete object).
  ConstructionKind getConstructionKind() const {
    return (ConstructionKind)ConstructKind;
  }
  void setConstructionKind(ConstructionKind CK) { 
    ConstructKind = CK;
  }
  
  typedef ExprIterator arg_iterator;
  typedef ConstExprIterator const_arg_iterator;

  arg_iterator arg_begin() { return Args; }
  arg_iterator arg_end() { return Args + NumArgs; }
  const_arg_iterator arg_begin() const { return Args; }
  const_arg_iterator arg_end() const { return Args + NumArgs; }

  Expr **getArgs() const { return reinterpret_cast<Expr **>(Args); }
  unsigned getNumArgs() const { return NumArgs; }

  /// getArg - Return the specified argument.
  Expr *getArg(unsigned Arg) {
    assert(Arg < NumArgs && "Arg access out of range!");
    return cast<Expr>(Args[Arg]);
  }
  const Expr *getArg(unsigned Arg) const {
    assert(Arg < NumArgs && "Arg access out of range!");
    return cast<Expr>(Args[Arg]);
  }

  /// setArg - Set the specified argument.
  void setArg(unsigned Arg, Expr *ArgExpr) {
    assert(Arg < NumArgs && "Arg access out of range!");
    Args[Arg] = ArgExpr;
  }

  SourceRange getSourceRange() const;
  SourceRange getParenRange() const { return ParenRange; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXConstructExprClass ||
      T->getStmtClass() == CXXTemporaryObjectExprClass;
  }
  static bool classof(const CXXConstructExpr *) { return true; }

  // Iterators
  child_range children() {
    return child_range(&Args[0], &Args[0]+NumArgs);
  }

  friend class ASTStmtReader;
};

/// CXXFunctionalCastExpr - Represents an explicit C++ type conversion
/// that uses "functional" notion (C++ [expr.type.conv]). Example: @c
/// x = int(0.5);
class CXXFunctionalCastExpr : public ExplicitCastExpr {
  SourceLocation TyBeginLoc;
  SourceLocation RParenLoc;

  CXXFunctionalCastExpr(QualType ty, ExprValueKind VK,
                        TypeSourceInfo *writtenTy,
                        SourceLocation tyBeginLoc, CastKind kind,
                        Expr *castExpr, unsigned pathSize,
                        SourceLocation rParenLoc) 
    : ExplicitCastExpr(CXXFunctionalCastExprClass, ty, VK, kind,
                       castExpr, pathSize, writtenTy),
      TyBeginLoc(tyBeginLoc), RParenLoc(rParenLoc) {}

  explicit CXXFunctionalCastExpr(EmptyShell Shell, unsigned PathSize)
    : ExplicitCastExpr(CXXFunctionalCastExprClass, Shell, PathSize) { }

public:
  static CXXFunctionalCastExpr *Create(ASTContext &Context, QualType T,
                                       ExprValueKind VK,
                                       TypeSourceInfo *Written,
                                       SourceLocation TyBeginLoc,
                                       CastKind Kind, Expr *Op,
                                       const CXXCastPath *Path,
                                       SourceLocation RPLoc);
  static CXXFunctionalCastExpr *CreateEmpty(ASTContext &Context,
                                            unsigned PathSize);

  SourceLocation getTypeBeginLoc() const { return TyBeginLoc; }
  void setTypeBeginLoc(SourceLocation L) { TyBeginLoc = L; }
  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation L) { RParenLoc = L; }

  SourceRange getSourceRange() const {
    return SourceRange(TyBeginLoc, RParenLoc);
  }
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXFunctionalCastExprClass;
  }
  static bool classof(const CXXFunctionalCastExpr *) { return true; }
};

/// @brief Represents a C++ functional cast expression that builds a
/// temporary object.
///
/// This expression type represents a C++ "functional" cast
/// (C++[expr.type.conv]) with N != 1 arguments that invokes a
/// constructor to build a temporary object. With N == 1 arguments the 
/// functional cast expression will be represented by CXXFunctionalCastExpr.
/// Example:
/// @code
/// struct X { X(int, float); }
///
/// X create_X() {
///   return X(1, 3.14f); // creates a CXXTemporaryObjectExpr
/// };
/// @endcode
class CXXTemporaryObjectExpr : public CXXConstructExpr {
  TypeSourceInfo *Type;

public:
  CXXTemporaryObjectExpr(ASTContext &C, CXXConstructorDecl *Cons,
                         TypeSourceInfo *Type,
                         Expr **Args,unsigned NumArgs,
                         SourceRange parenRange,
                         bool HadMultipleCandidates,
                         bool ZeroInitialization = false);
  explicit CXXTemporaryObjectExpr(EmptyShell Empty)
    : CXXConstructExpr(CXXTemporaryObjectExprClass, Empty), Type() { }

  TypeSourceInfo *getTypeSourceInfo() const { return Type; }

  SourceRange getSourceRange() const;
  
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXTemporaryObjectExprClass;
  }
  static bool classof(const CXXTemporaryObjectExpr *) { return true; }

  friend class ASTStmtReader;
};

/// CXXScalarValueInitExpr - [C++ 5.2.3p2]
/// Expression "T()" which creates a value-initialized rvalue of type
/// T, which is a non-class type.
///
class CXXScalarValueInitExpr : public Expr {
  SourceLocation RParenLoc;
  TypeSourceInfo *TypeInfo;

  friend class ASTStmtReader;
  
public:
  /// \brief Create an explicitly-written scalar-value initialization 
  /// expression.
  CXXScalarValueInitExpr(QualType Type,
                         TypeSourceInfo *TypeInfo,
                         SourceLocation rParenLoc ) :
    Expr(CXXScalarValueInitExprClass, Type, VK_RValue, OK_Ordinary,
         false, false, Type->isInstantiationDependentType(), false),
    RParenLoc(rParenLoc), TypeInfo(TypeInfo) {}

  explicit CXXScalarValueInitExpr(EmptyShell Shell)
    : Expr(CXXScalarValueInitExprClass, Shell) { }

  TypeSourceInfo *getTypeSourceInfo() const {
    return TypeInfo;
  }
  
  SourceLocation getRParenLoc() const { return RParenLoc; }

  SourceRange getSourceRange() const;

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXScalarValueInitExprClass;
  }
  static bool classof(const CXXScalarValueInitExpr *) { return true; }

  // Iterators
  child_range children() { return child_range(); }
};

/// CXXNewExpr - A new expression for memory allocation and constructor calls,
/// e.g: "new CXXNewExpr(foo)".
class CXXNewExpr : public Expr {
  // Was the usage ::new, i.e. is the global new to be used?
  bool GlobalNew : 1;
  // Is there an initializer? If not, built-ins are uninitialized, else they're
  // value-initialized.
  bool Initializer : 1;
  // Do we allocate an array? If so, the first SubExpr is the size expression.
  bool Array : 1;
  // If this is an array allocation, does the usual deallocation
  // function for the allocated type want to know the allocated size?
  bool UsualArrayDeleteWantsSize : 1;
  // Whether the referred constructor (if any) was resolved from an
  // overload set having size greater than 1.
  bool HadMultipleCandidates : 1;
  // The number of placement new arguments.
  unsigned NumPlacementArgs : 13;
  // The number of constructor arguments. This may be 1 even for non-class
  // types; use the pseudo copy constructor.
  unsigned NumConstructorArgs : 14;
  // Contains an optional array size expression, any number of optional
  // placement arguments, and any number of optional constructor arguments,
  // in that order.
  Stmt **SubExprs;
  // Points to the allocation function used.
  FunctionDecl *OperatorNew;
  // Points to the deallocation function used in case of error. May be null.
  FunctionDecl *OperatorDelete;
  // Points to the constructor used. Cannot be null if AllocType is a record;
  // it would still point at the default constructor (even an implicit one).
  // Must be null for all other types.
  CXXConstructorDecl *Constructor;

  /// \brief The allocated type-source information, as written in the source.
  TypeSourceInfo *AllocatedTypeInfo;
  
  /// \brief If the allocated type was expressed as a parenthesized type-id, 
  /// the source range covering the parenthesized type-id.
  SourceRange TypeIdParens;
  
  SourceLocation StartLoc;
  SourceLocation EndLoc;
  SourceLocation ConstructorLParen;
  SourceLocation ConstructorRParen;

  friend class ASTStmtReader;
public:
  CXXNewExpr(ASTContext &C, bool globalNew, FunctionDecl *operatorNew,
             Expr **placementArgs, unsigned numPlaceArgs,
             SourceRange TypeIdParens,
             Expr *arraySize, CXXConstructorDecl *constructor, bool initializer,
             Expr **constructorArgs, unsigned numConsArgs,
             bool HadMultipleCandidates,
             FunctionDecl *operatorDelete, bool usualArrayDeleteWantsSize,
             QualType ty, TypeSourceInfo *AllocatedTypeInfo,
             SourceLocation startLoc, SourceLocation endLoc,
             SourceLocation constructorLParen,
             SourceLocation constructorRParen);
  explicit CXXNewExpr(EmptyShell Shell)
    : Expr(CXXNewExprClass, Shell), SubExprs(0) { }

  void AllocateArgsArray(ASTContext &C, bool isArray, unsigned numPlaceArgs,
                         unsigned numConsArgs);
  
  QualType getAllocatedType() const {
    assert(getType()->isPointerType());
    return getType()->getAs<PointerType>()->getPointeeType();
  }

  TypeSourceInfo *getAllocatedTypeSourceInfo() const {
    return AllocatedTypeInfo;
  }

  /// \brief True if the allocation result needs to be null-checked.
  /// C++0x [expr.new]p13:
  ///   If the allocation function returns null, initialization shall
  ///   not be done, the deallocation function shall not be called,
  ///   and the value of the new-expression shall be null.
  /// An allocation function is not allowed to return null unless it
  /// has a non-throwing exception-specification.  The '03 rule is
  /// identical except that the definition of a non-throwing
  /// exception specification is just "is it throw()?".
  bool shouldNullCheckAllocation(ASTContext &Ctx) const;
  
  FunctionDecl *getOperatorNew() const { return OperatorNew; }
  void setOperatorNew(FunctionDecl *D) { OperatorNew = D; }
  FunctionDecl *getOperatorDelete() const { return OperatorDelete; }
  void setOperatorDelete(FunctionDecl *D) { OperatorDelete = D; }
  CXXConstructorDecl *getConstructor() const { return Constructor; }
  void setConstructor(CXXConstructorDecl *D) { Constructor = D; }

  bool isArray() const { return Array; }
  Expr *getArraySize() {
    return Array ? cast<Expr>(SubExprs[0]) : 0;
  }
  const Expr *getArraySize() const {
    return Array ? cast<Expr>(SubExprs[0]) : 0;
  }

  unsigned getNumPlacementArgs() const { return NumPlacementArgs; }
  Expr **getPlacementArgs() { 
    return reinterpret_cast<Expr **>(SubExprs + Array); 
  }
  
  Expr *getPlacementArg(unsigned i) {
    assert(i < NumPlacementArgs && "Index out of range");
    return cast<Expr>(SubExprs[Array + i]);
  }
  const Expr *getPlacementArg(unsigned i) const {
    assert(i < NumPlacementArgs && "Index out of range");
    return cast<Expr>(SubExprs[Array + i]);
  }

  bool isParenTypeId() const { return TypeIdParens.isValid(); }
  SourceRange getTypeIdParens() const { return TypeIdParens; }

  bool isGlobalNew() const { return GlobalNew; }
  bool hasInitializer() const { return Initializer; }

  /// Answers whether the usual array deallocation function for the
  /// allocated type expects the size of the allocation as a
  /// parameter.
  bool doesUsualArrayDeleteWantSize() const {
    return UsualArrayDeleteWantsSize;
  }

  unsigned getNumConstructorArgs() const { return NumConstructorArgs; }
  
  Expr **getConstructorArgs() {
    return reinterpret_cast<Expr **>(SubExprs + Array + NumPlacementArgs);
  }
  
  Expr *getConstructorArg(unsigned i) {
    assert(i < NumConstructorArgs && "Index out of range");
    return cast<Expr>(SubExprs[Array + NumPlacementArgs + i]);
  }
  const Expr *getConstructorArg(unsigned i) const {
    assert(i < NumConstructorArgs && "Index out of range");
    return cast<Expr>(SubExprs[Array + NumPlacementArgs + i]);
  }

  /// \brief Whether the new expression refers a constructor that was
  /// resolved from an overloaded set having size greater than 1.
  bool hadMultipleCandidates() const { return HadMultipleCandidates; }
  void setHadMultipleCandidates(bool V) { HadMultipleCandidates = V; }

  typedef ExprIterator arg_iterator;
  typedef ConstExprIterator const_arg_iterator;

  arg_iterator placement_arg_begin() {
    return SubExprs + Array;
  }
  arg_iterator placement_arg_end() {
    return SubExprs + Array + getNumPlacementArgs();
  }
  const_arg_iterator placement_arg_begin() const {
    return SubExprs + Array;
  }
  const_arg_iterator placement_arg_end() const {
    return SubExprs + Array + getNumPlacementArgs();
  }

  arg_iterator constructor_arg_begin() {
    return SubExprs + Array + getNumPlacementArgs();
  }
  arg_iterator constructor_arg_end() {
    return SubExprs + Array + getNumPlacementArgs() + getNumConstructorArgs();
  }
  const_arg_iterator constructor_arg_begin() const {
    return SubExprs + Array + getNumPlacementArgs();
  }
  const_arg_iterator constructor_arg_end() const {
    return SubExprs + Array + getNumPlacementArgs() + getNumConstructorArgs();
  }
  
  typedef Stmt **raw_arg_iterator;
  raw_arg_iterator raw_arg_begin() { return SubExprs; }
  raw_arg_iterator raw_arg_end() {
    return SubExprs + Array + getNumPlacementArgs() + getNumConstructorArgs();
  }
  const_arg_iterator raw_arg_begin() const { return SubExprs; }
  const_arg_iterator raw_arg_end() const { return constructor_arg_end(); }

  SourceLocation getStartLoc() const { return StartLoc; }
  SourceLocation getEndLoc() const { return EndLoc; }

  SourceLocation getConstructorLParen() const { return ConstructorLParen; }
  SourceLocation getConstructorRParen() const { return ConstructorRParen; }

  SourceRange getSourceRange() const {
    return SourceRange(StartLoc, EndLoc);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXNewExprClass;
  }
  static bool classof(const CXXNewExpr *) { return true; }

  // Iterators
  child_range children() {
    return child_range(&SubExprs[0],
                       &SubExprs[0] + Array + getNumPlacementArgs()
                         + getNumConstructorArgs());
  }
};

/// CXXDeleteExpr - A delete expression for memory deallocation and destructor
/// calls, e.g. "delete[] pArray".
class CXXDeleteExpr : public Expr {
  // Is this a forced global delete, i.e. "::delete"?
  bool GlobalDelete : 1;
  // Is this the array form of delete, i.e. "delete[]"?
  bool ArrayForm : 1;
  // ArrayFormAsWritten can be different from ArrayForm if 'delete' is applied
  // to pointer-to-array type (ArrayFormAsWritten will be false while ArrayForm
  // will be true).
  bool ArrayFormAsWritten : 1;
  // Does the usual deallocation function for the element type require
  // a size_t argument?
  bool UsualArrayDeleteWantsSize : 1;
  // Points to the operator delete overload that is used. Could be a member.
  FunctionDecl *OperatorDelete;
  // The pointer expression to be deleted.
  Stmt *Argument;
  // Location of the expression.
  SourceLocation Loc;
public:
  CXXDeleteExpr(QualType ty, bool globalDelete, bool arrayForm,
                bool arrayFormAsWritten, bool usualArrayDeleteWantsSize,
                FunctionDecl *operatorDelete, Expr *arg, SourceLocation loc)
    : Expr(CXXDeleteExprClass, ty, VK_RValue, OK_Ordinary, false, false,
           arg->isInstantiationDependent(),
           arg->containsUnexpandedParameterPack()),
      GlobalDelete(globalDelete),
      ArrayForm(arrayForm), ArrayFormAsWritten(arrayFormAsWritten),
      UsualArrayDeleteWantsSize(usualArrayDeleteWantsSize),
      OperatorDelete(operatorDelete), Argument(arg), Loc(loc) { }
  explicit CXXDeleteExpr(EmptyShell Shell)
    : Expr(CXXDeleteExprClass, Shell), OperatorDelete(0), Argument(0) { }

  bool isGlobalDelete() const { return GlobalDelete; }
  bool isArrayForm() const { return ArrayForm; }
  bool isArrayFormAsWritten() const { return ArrayFormAsWritten; }

  /// Answers whether the usual array deallocation function for the
  /// allocated type expects the size of the allocation as a
  /// parameter.  This can be true even if the actual deallocation
  /// function that we're using doesn't want a size.
  bool doesUsualArrayDeleteWantSize() const {
    return UsualArrayDeleteWantsSize;
  }

  FunctionDecl *getOperatorDelete() const { return OperatorDelete; }

  Expr *getArgument() { return cast<Expr>(Argument); }
  const Expr *getArgument() const { return cast<Expr>(Argument); }

  /// \brief Retrieve the type being destroyed.  If the type being
  /// destroyed is a dependent type which may or may not be a pointer,
  /// return an invalid type.
  QualType getDestroyedType() const;
  
  SourceRange getSourceRange() const {
    return SourceRange(Loc, Argument->getLocEnd());
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXDeleteExprClass;
  }
  static bool classof(const CXXDeleteExpr *) { return true; }

  // Iterators
  child_range children() { return child_range(&Argument, &Argument+1); }

  friend class ASTStmtReader;
};

/// \brief Structure used to store the type being destroyed by a 
/// pseudo-destructor expression.
class PseudoDestructorTypeStorage {
  /// \brief Either the type source information or the name of the type, if 
  /// it couldn't be resolved due to type-dependence.
  llvm::PointerUnion<TypeSourceInfo *, IdentifierInfo *> Type;
  
  /// \brief The starting source location of the pseudo-destructor type.
  SourceLocation Location;
  
public:
  PseudoDestructorTypeStorage() { }
  
  PseudoDestructorTypeStorage(IdentifierInfo *II, SourceLocation Loc)
    : Type(II), Location(Loc) { }
  
  PseudoDestructorTypeStorage(TypeSourceInfo *Info);
  
  TypeSourceInfo *getTypeSourceInfo() const { 
    return Type.dyn_cast<TypeSourceInfo *>(); 
  }
  
  IdentifierInfo *getIdentifier() const {
    return Type.dyn_cast<IdentifierInfo *>();
  }
  
  SourceLocation getLocation() const { return Location; }
};
  
/// \brief Represents a C++ pseudo-destructor (C++ [expr.pseudo]).
///
/// A pseudo-destructor is an expression that looks like a member access to a
/// destructor of a scalar type, except that scalar types don't have 
/// destructors. For example:
///
/// \code
/// typedef int T;
/// void f(int *p) {
///   p->T::~T();
/// }
/// \endcode
///
/// Pseudo-destructors typically occur when instantiating templates such as:
/// 
/// \code
/// template<typename T>
/// void destroy(T* ptr) {
///   ptr->T::~T();
/// }
/// \endcode
///
/// for scalar types. A pseudo-destructor expression has no run-time semantics
/// beyond evaluating the base expression.
class CXXPseudoDestructorExpr : public Expr {
  /// \brief The base expression (that is being destroyed).
  Stmt *Base;

  /// \brief Whether the operator was an arrow ('->'); otherwise, it was a
  /// period ('.').
  bool IsArrow : 1;

  /// \brief The location of the '.' or '->' operator.
  SourceLocation OperatorLoc;
  
  /// \brief The nested-name-specifier that follows the operator, if present.
  NestedNameSpecifierLoc QualifierLoc;

  /// \brief The type that precedes the '::' in a qualified pseudo-destructor
  /// expression.
  TypeSourceInfo *ScopeType;
  
  /// \brief The location of the '::' in a qualified pseudo-destructor 
  /// expression.
  SourceLocation ColonColonLoc;
  
  /// \brief The location of the '~'.
  SourceLocation TildeLoc;
  
  /// \brief The type being destroyed, or its name if we were unable to 
  /// resolve the name.
  PseudoDestructorTypeStorage DestroyedType;

  friend class ASTStmtReader;
  
public:
  CXXPseudoDestructorExpr(ASTContext &Context,
                          Expr *Base, bool isArrow, SourceLocation OperatorLoc,
                          NestedNameSpecifierLoc QualifierLoc,
                          TypeSourceInfo *ScopeType,
                          SourceLocation ColonColonLoc,
                          SourceLocation TildeLoc,
                          PseudoDestructorTypeStorage DestroyedType);

  explicit CXXPseudoDestructorExpr(EmptyShell Shell)
    : Expr(CXXPseudoDestructorExprClass, Shell),
      Base(0), IsArrow(false), QualifierLoc(), ScopeType(0) { }

  Expr *getBase() const { return cast<Expr>(Base); }

  /// \brief Determines whether this member expression actually had
  /// a C++ nested-name-specifier prior to the name of the member, e.g.,
  /// x->Base::foo.
  bool hasQualifier() const { return QualifierLoc; }

  /// \brief Retrieves the nested-name-specifier that qualifies the type name,
  /// with source-location information.
  NestedNameSpecifierLoc getQualifierLoc() const { return QualifierLoc; }
  
  /// \brief If the member name was qualified, retrieves the
  /// nested-name-specifier that precedes the member name. Otherwise, returns
  /// NULL.
  NestedNameSpecifier *getQualifier() const { 
    return QualifierLoc.getNestedNameSpecifier(); 
  }

  /// \brief Determine whether this pseudo-destructor expression was written
  /// using an '->' (otherwise, it used a '.').
  bool isArrow() const { return IsArrow; }

  /// \brief Retrieve the location of the '.' or '->' operator.
  SourceLocation getOperatorLoc() const { return OperatorLoc; }

  /// \brief Retrieve the scope type in a qualified pseudo-destructor 
  /// expression.
  ///
  /// Pseudo-destructor expressions can have extra qualification within them
  /// that is not part of the nested-name-specifier, e.g., \c p->T::~T().
  /// Here, if the object type of the expression is (or may be) a scalar type,
  /// \p T may also be a scalar type and, therefore, cannot be part of a 
  /// nested-name-specifier. It is stored as the "scope type" of the pseudo-
  /// destructor expression.
  TypeSourceInfo *getScopeTypeInfo() const { return ScopeType; }
  
  /// \brief Retrieve the location of the '::' in a qualified pseudo-destructor
  /// expression.
  SourceLocation getColonColonLoc() const { return ColonColonLoc; }
  
  /// \brief Retrieve the location of the '~'.
  SourceLocation getTildeLoc() const { return TildeLoc; }
  
  /// \brief Retrieve the source location information for the type
  /// being destroyed.
  ///
  /// This type-source information is available for non-dependent 
  /// pseudo-destructor expressions and some dependent pseudo-destructor
  /// expressions. Returns NULL if we only have the identifier for a
  /// dependent pseudo-destructor expression.
  TypeSourceInfo *getDestroyedTypeInfo() const { 
    return DestroyedType.getTypeSourceInfo(); 
  }
  
  /// \brief In a dependent pseudo-destructor expression for which we do not
  /// have full type information on the destroyed type, provides the name
  /// of the destroyed type.
  IdentifierInfo *getDestroyedTypeIdentifier() const {
    return DestroyedType.getIdentifier();
  }
  
  /// \brief Retrieve the type being destroyed.
  QualType getDestroyedType() const;
  
  /// \brief Retrieve the starting location of the type being destroyed.
  SourceLocation getDestroyedTypeLoc() const { 
    return DestroyedType.getLocation(); 
  }

  /// \brief Set the name of destroyed type for a dependent pseudo-destructor
  /// expression.
  void setDestroyedType(IdentifierInfo *II, SourceLocation Loc) {
    DestroyedType = PseudoDestructorTypeStorage(II, Loc);
  }

  /// \brief Set the destroyed type.
  void setDestroyedType(TypeSourceInfo *Info) {
    DestroyedType = PseudoDestructorTypeStorage(Info);
  }

  SourceRange getSourceRange() const;

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXPseudoDestructorExprClass;
  }
  static bool classof(const CXXPseudoDestructorExpr *) { return true; }

  // Iterators
  child_range children() { return child_range(&Base, &Base + 1); }
};

/// UnaryTypeTraitExpr - A GCC or MS unary type trait, as used in the
/// implementation of TR1/C++0x type trait templates.
/// Example:
/// __is_pod(int) == true
/// __is_enum(std::string) == false
class UnaryTypeTraitExpr : public Expr {
  /// UTT - The trait. A UnaryTypeTrait enum in MSVC compat unsigned.
  unsigned UTT : 31;
  /// The value of the type trait. Unspecified if dependent.
  bool Value : 1;

  /// Loc - The location of the type trait keyword.
  SourceLocation Loc;

  /// RParen - The location of the closing paren.
  SourceLocation RParen;

  /// The type being queried.
  TypeSourceInfo *QueriedType;

public:
  UnaryTypeTraitExpr(SourceLocation loc, UnaryTypeTrait utt, 
                     TypeSourceInfo *queried, bool value,
                     SourceLocation rparen, QualType ty)
    : Expr(UnaryTypeTraitExprClass, ty, VK_RValue, OK_Ordinary,
           false,  queried->getType()->isDependentType(),
           queried->getType()->isInstantiationDependentType(),
           queried->getType()->containsUnexpandedParameterPack()),
      UTT(utt), Value(value), Loc(loc), RParen(rparen), QueriedType(queried) { }

  explicit UnaryTypeTraitExpr(EmptyShell Empty)
    : Expr(UnaryTypeTraitExprClass, Empty), UTT(0), Value(false),
      QueriedType() { }

  SourceRange getSourceRange() const { return SourceRange(Loc, RParen);}

  UnaryTypeTrait getTrait() const { return static_cast<UnaryTypeTrait>(UTT); }

  QualType getQueriedType() const { return QueriedType->getType(); }

  TypeSourceInfo *getQueriedTypeSourceInfo() const { return QueriedType; }
  
  bool getValue() const { return Value; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == UnaryTypeTraitExprClass;
  }
  static bool classof(const UnaryTypeTraitExpr *) { return true; }

  // Iterators
  child_range children() { return child_range(); }

  friend class ASTStmtReader;
};

/// BinaryTypeTraitExpr - A GCC or MS binary type trait, as used in the
/// implementation of TR1/C++0x type trait templates.
/// Example:
/// __is_base_of(Base, Derived) == true
class BinaryTypeTraitExpr : public Expr {
  /// BTT - The trait. A BinaryTypeTrait enum in MSVC compat unsigned.
  unsigned BTT : 8;

  /// The value of the type trait. Unspecified if dependent.
  bool Value : 1;

  /// Loc - The location of the type trait keyword.
  SourceLocation Loc;

  /// RParen - The location of the closing paren.
  SourceLocation RParen;

  /// The lhs type being queried.
  TypeSourceInfo *LhsType;

  /// The rhs type being queried.
  TypeSourceInfo *RhsType;

public:
  BinaryTypeTraitExpr(SourceLocation loc, BinaryTypeTrait btt, 
                     TypeSourceInfo *lhsType, TypeSourceInfo *rhsType, 
                     bool value, SourceLocation rparen, QualType ty)
    : Expr(BinaryTypeTraitExprClass, ty, VK_RValue, OK_Ordinary, false, 
           lhsType->getType()->isDependentType() ||
           rhsType->getType()->isDependentType(),
           (lhsType->getType()->isInstantiationDependentType() ||
            rhsType->getType()->isInstantiationDependentType()),
           (lhsType->getType()->containsUnexpandedParameterPack() ||
            rhsType->getType()->containsUnexpandedParameterPack())),
      BTT(btt), Value(value), Loc(loc), RParen(rparen),
      LhsType(lhsType), RhsType(rhsType) { }


  explicit BinaryTypeTraitExpr(EmptyShell Empty)
    : Expr(BinaryTypeTraitExprClass, Empty), BTT(0), Value(false),
      LhsType(), RhsType() { }

  SourceRange getSourceRange() const {
    return SourceRange(Loc, RParen);
  }

  BinaryTypeTrait getTrait() const {
    return static_cast<BinaryTypeTrait>(BTT);
  }

  QualType getLhsType() const { return LhsType->getType(); }
  QualType getRhsType() const { return RhsType->getType(); }

  TypeSourceInfo *getLhsTypeSourceInfo() const { return LhsType; }
  TypeSourceInfo *getRhsTypeSourceInfo() const { return RhsType; }
  
  bool getValue() const { assert(!isTypeDependent()); return Value; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == BinaryTypeTraitExprClass;
  }
  static bool classof(const BinaryTypeTraitExpr *) { return true; }

  // Iterators
  child_range children() { return child_range(); }

  friend class ASTStmtReader;
};

/// ArrayTypeTraitExpr - An Embarcadero array type trait, as used in the
/// implementation of __array_rank and __array_extent.
/// Example:
/// __array_rank(int[10][20]) == 2
/// __array_extent(int, 1)    == 20
class ArrayTypeTraitExpr : public Expr {
  /// ATT - The trait. An ArrayTypeTrait enum in MSVC compat unsigned.
  unsigned ATT : 2;

  /// The value of the type trait. Unspecified if dependent.
  uint64_t Value;

  /// The array dimension being queried, or -1 if not used
  Expr *Dimension;

  /// Loc - The location of the type trait keyword.
  SourceLocation Loc;

  /// RParen - The location of the closing paren.
  SourceLocation RParen;

  /// The type being queried.
  TypeSourceInfo *QueriedType;

public:
  ArrayTypeTraitExpr(SourceLocation loc, ArrayTypeTrait att,
                     TypeSourceInfo *queried, uint64_t value,
                     Expr *dimension, SourceLocation rparen, QualType ty)
    : Expr(ArrayTypeTraitExprClass, ty, VK_RValue, OK_Ordinary,
           false, queried->getType()->isDependentType(),
           (queried->getType()->isInstantiationDependentType() ||
            (dimension && dimension->isInstantiationDependent())),
           queried->getType()->containsUnexpandedParameterPack()),
      ATT(att), Value(value), Dimension(dimension),
      Loc(loc), RParen(rparen), QueriedType(queried) { }


  explicit ArrayTypeTraitExpr(EmptyShell Empty)
    : Expr(ArrayTypeTraitExprClass, Empty), ATT(0), Value(false),
      QueriedType() { }

  virtual ~ArrayTypeTraitExpr() { }

  virtual SourceRange getSourceRange() const { return SourceRange(Loc, RParen); }

  ArrayTypeTrait getTrait() const { return static_cast<ArrayTypeTrait>(ATT); }

  QualType getQueriedType() const { return QueriedType->getType(); }

  TypeSourceInfo *getQueriedTypeSourceInfo() const { return QueriedType; }

  uint64_t getValue() const { assert(!isTypeDependent()); return Value; }

  Expr *getDimensionExpression() const { return Dimension; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ArrayTypeTraitExprClass;
  }
  static bool classof(const ArrayTypeTraitExpr *) { return true; }

  // Iterators
  child_range children() { return child_range(); }

  friend class ASTStmtReader;
};

/// ExpressionTraitExpr - An expression trait intrinsic
/// Example:
/// __is_lvalue_expr(std::cout) == true
/// __is_lvalue_expr(1) == false
class ExpressionTraitExpr : public Expr {
  /// ET - The trait. A ExpressionTrait enum in MSVC compat unsigned.
  unsigned ET : 31;
  /// The value of the type trait. Unspecified if dependent.
  bool Value : 1;

  /// Loc - The location of the type trait keyword.
  SourceLocation Loc;

  /// RParen - The location of the closing paren.
  SourceLocation RParen;

  Expr* QueriedExpression;
public:
  ExpressionTraitExpr(SourceLocation loc, ExpressionTrait et, 
                     Expr *queried, bool value,
                     SourceLocation rparen, QualType resultType)
    : Expr(ExpressionTraitExprClass, resultType, VK_RValue, OK_Ordinary,
           false, // Not type-dependent
           // Value-dependent if the argument is type-dependent.
           queried->isTypeDependent(),
           queried->isInstantiationDependent(),
           queried->containsUnexpandedParameterPack()),
      ET(et), Value(value), Loc(loc), RParen(rparen), QueriedExpression(queried) { }

  explicit ExpressionTraitExpr(EmptyShell Empty)
    : Expr(ExpressionTraitExprClass, Empty), ET(0), Value(false),
      QueriedExpression() { }

  SourceRange getSourceRange() const { return SourceRange(Loc, RParen);}

  ExpressionTrait getTrait() const { return static_cast<ExpressionTrait>(ET); }

  Expr *getQueriedExpression() const { return QueriedExpression; }

  bool getValue() const { return Value; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ExpressionTraitExprClass;
  }
  static bool classof(const ExpressionTraitExpr *) { return true; }

  // Iterators
  child_range children() { return child_range(); }

  friend class ASTStmtReader;
};


/// \brief A reference to an overloaded function set, either an
/// \t UnresolvedLookupExpr or an \t UnresolvedMemberExpr.
class OverloadExpr : public Expr {
  /// The results.  These are undesugared, which is to say, they may
  /// include UsingShadowDecls.  Access is relative to the naming
  /// class.
  // FIXME: Allocate this data after the OverloadExpr subclass.
  DeclAccessPair *Results;
  unsigned NumResults;

  /// The common name of these declarations.
  DeclarationNameInfo NameInfo;

  /// \brief The nested-name-specifier that qualifies the name, if any.
  NestedNameSpecifierLoc QualifierLoc;

protected:
  /// True if the name was a template-id.
  bool HasExplicitTemplateArgs;

  OverloadExpr(StmtClass K, ASTContext &C,
               NestedNameSpecifierLoc QualifierLoc,
               const DeclarationNameInfo &NameInfo,
               const TemplateArgumentListInfo *TemplateArgs,
               UnresolvedSetIterator Begin, UnresolvedSetIterator End,
               bool KnownDependent,
               bool KnownInstantiationDependent,
               bool KnownContainsUnexpandedParameterPack);

  OverloadExpr(StmtClass K, EmptyShell Empty)
    : Expr(K, Empty), Results(0), NumResults(0),
      QualifierLoc(), HasExplicitTemplateArgs(false) { }

  void initializeResults(ASTContext &C,
                         UnresolvedSetIterator Begin,
                         UnresolvedSetIterator End);

public:
  struct FindResult {
    OverloadExpr *Expression;
    bool IsAddressOfOperand;
    bool HasFormOfMemberPointer;
  };

  /// Finds the overloaded expression in the given expression of
  /// OverloadTy.
  ///
  /// \return the expression (which must be there) and true if it has
  /// the particular form of a member pointer expression
  static FindResult find(Expr *E) {
    assert(E->getType()->isSpecificBuiltinType(BuiltinType::Overload));

    FindResult Result;

    E = E->IgnoreParens();
    if (isa<UnaryOperator>(E)) {
      assert(cast<UnaryOperator>(E)->getOpcode() == UO_AddrOf);
      E = cast<UnaryOperator>(E)->getSubExpr();
      OverloadExpr *Ovl = cast<OverloadExpr>(E->IgnoreParens());

      Result.HasFormOfMemberPointer = (E == Ovl && Ovl->getQualifier());
      Result.IsAddressOfOperand = true;
      Result.Expression = Ovl;
    } else {
      Result.HasFormOfMemberPointer = false;
      Result.IsAddressOfOperand = false;
      Result.Expression = cast<OverloadExpr>(E);
    }

    return Result;
  }

  /// Gets the naming class of this lookup, if any.
  CXXRecordDecl *getNamingClass() const;

  typedef UnresolvedSetImpl::iterator decls_iterator;
  decls_iterator decls_begin() const { return UnresolvedSetIterator(Results); }
  decls_iterator decls_end() const { 
    return UnresolvedSetIterator(Results + NumResults);
  }
  
  /// Gets the number of declarations in the unresolved set.
  unsigned getNumDecls() const { return NumResults; }

  /// Gets the full name info.
  const DeclarationNameInfo &getNameInfo() const { return NameInfo; }

  /// Gets the name looked up.
  DeclarationName getName() const { return NameInfo.getName(); }

  /// Gets the location of the name.
  SourceLocation getNameLoc() const { return NameInfo.getLoc(); }

  /// Fetches the nested-name qualifier, if one was given.
  NestedNameSpecifier *getQualifier() const { 
    return QualifierLoc.getNestedNameSpecifier(); 
  }

  /// Fetches the nested-name qualifier with source-location information, if 
  /// one was given.
  NestedNameSpecifierLoc getQualifierLoc() const { return QualifierLoc; }

  /// \brief Determines whether this expression had an explicit
  /// template argument list, e.g. f<int>.
  bool hasExplicitTemplateArgs() const { return HasExplicitTemplateArgs; }

  ASTTemplateArgumentListInfo &getExplicitTemplateArgs(); // defined far below

  const ASTTemplateArgumentListInfo &getExplicitTemplateArgs() const {
    return const_cast<OverloadExpr*>(this)->getExplicitTemplateArgs();
  }

  /// \brief Retrieves the optional explicit template arguments.
  /// This points to the same data as getExplicitTemplateArgs(), but
  /// returns null if there are no explicit template arguments.
  const ASTTemplateArgumentListInfo *getOptionalExplicitTemplateArgs() {
    if (!hasExplicitTemplateArgs()) return 0;
    return &getExplicitTemplateArgs();
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == UnresolvedLookupExprClass ||
           T->getStmtClass() == UnresolvedMemberExprClass;
  }
  static bool classof(const OverloadExpr *) { return true; }

  friend class ASTStmtReader;
  friend class ASTStmtWriter;
};

/// \brief A reference to a name which we were able to look up during
/// parsing but could not resolve to a specific declaration.  This
/// arises in several ways:
///   * we might be waiting for argument-dependent lookup
///   * the name might resolve to an overloaded function
/// and eventually:
///   * the lookup might have included a function template
/// These never include UnresolvedUsingValueDecls, which are always
/// class members and therefore appear only in
/// UnresolvedMemberLookupExprs.
class UnresolvedLookupExpr : public OverloadExpr {
  /// True if these lookup results should be extended by
  /// argument-dependent lookup if this is the operand of a function
  /// call.
  bool RequiresADL;

  /// True if namespace ::std should be considered an associated namespace
  /// for the purposes of argument-dependent lookup. See C++0x [stmt.ranged]p1.
  bool StdIsAssociatedNamespace;

  /// True if these lookup results are overloaded.  This is pretty
  /// trivially rederivable if we urgently need to kill this field.
  bool Overloaded;

  /// The naming class (C++ [class.access.base]p5) of the lookup, if
  /// any.  This can generally be recalculated from the context chain,
  /// but that can be fairly expensive for unqualified lookups.  If we
  /// want to improve memory use here, this could go in a union
  /// against the qualified-lookup bits.
  CXXRecordDecl *NamingClass;

  UnresolvedLookupExpr(ASTContext &C, 
                       CXXRecordDecl *NamingClass,
                       NestedNameSpecifierLoc QualifierLoc,
                       const DeclarationNameInfo &NameInfo,
                       bool RequiresADL, bool Overloaded, 
                       const TemplateArgumentListInfo *TemplateArgs,
                       UnresolvedSetIterator Begin, UnresolvedSetIterator End,
                       bool StdIsAssociatedNamespace)
    : OverloadExpr(UnresolvedLookupExprClass, C, QualifierLoc, NameInfo, 
                   TemplateArgs, Begin, End, false, false, false),
      RequiresADL(RequiresADL),
      StdIsAssociatedNamespace(StdIsAssociatedNamespace),
      Overloaded(Overloaded), NamingClass(NamingClass)
  {}

  UnresolvedLookupExpr(EmptyShell Empty)
    : OverloadExpr(UnresolvedLookupExprClass, Empty),
      RequiresADL(false), StdIsAssociatedNamespace(false), Overloaded(false),
      NamingClass(0)
  {}

  friend class ASTStmtReader;
  
public:
  static UnresolvedLookupExpr *Create(ASTContext &C,
                                      CXXRecordDecl *NamingClass,
                                      NestedNameSpecifierLoc QualifierLoc,
                                      const DeclarationNameInfo &NameInfo,
                                      bool ADL, bool Overloaded,
                                      UnresolvedSetIterator Begin, 
                                      UnresolvedSetIterator End,
                                      bool StdIsAssociatedNamespace = false) {
    assert((ADL || !StdIsAssociatedNamespace) &&
           "std considered associated namespace when not performing ADL");
    return new(C) UnresolvedLookupExpr(C, NamingClass, QualifierLoc, NameInfo, 
                                       ADL, Overloaded, 0, Begin, End,
                                       StdIsAssociatedNamespace);
  }

  static UnresolvedLookupExpr *Create(ASTContext &C,
                                      CXXRecordDecl *NamingClass,
                                      NestedNameSpecifierLoc QualifierLoc,
                                      const DeclarationNameInfo &NameInfo,
                                      bool ADL,
                                      const TemplateArgumentListInfo &Args,
                                      UnresolvedSetIterator Begin, 
                                      UnresolvedSetIterator End);

  static UnresolvedLookupExpr *CreateEmpty(ASTContext &C,
                                           bool HasExplicitTemplateArgs,
                                           unsigned NumTemplateArgs);

  /// True if this declaration should be extended by
  /// argument-dependent lookup.
  bool requiresADL() const { return RequiresADL; }

  /// True if namespace ::std should be artificially added to the set of
  /// associated namespaecs for argument-dependent lookup purposes.
  bool isStdAssociatedNamespace() const { return StdIsAssociatedNamespace; }

  /// True if this lookup is overloaded.
  bool isOverloaded() const { return Overloaded; }

  /// Gets the 'naming class' (in the sense of C++0x
  /// [class.access.base]p5) of the lookup.  This is the scope
  /// that was looked in to find these results.
  CXXRecordDecl *getNamingClass() const { return NamingClass; }

  // Note that, inconsistently with the explicit-template-argument AST
  // nodes, users are *forbidden* from calling these methods on objects
  // without explicit template arguments.

  ASTTemplateArgumentListInfo &getExplicitTemplateArgs() {
    assert(hasExplicitTemplateArgs());
    return *reinterpret_cast<ASTTemplateArgumentListInfo*>(this + 1);
  }

  /// Gets a reference to the explicit template argument list.
  const ASTTemplateArgumentListInfo &getExplicitTemplateArgs() const {
    assert(hasExplicitTemplateArgs());
    return *reinterpret_cast<const ASTTemplateArgumentListInfo*>(this + 1);
  }

  /// \brief Retrieves the optional explicit template arguments.
  /// This points to the same data as getExplicitTemplateArgs(), but
  /// returns null if there are no explicit template arguments.
  const ASTTemplateArgumentListInfo *getOptionalExplicitTemplateArgs() {
    if (!hasExplicitTemplateArgs()) return 0;
    return &getExplicitTemplateArgs();
  }

  /// \brief Copies the template arguments (if present) into the given
  /// structure.
  void copyTemplateArgumentsInto(TemplateArgumentListInfo &List) const {
    getExplicitTemplateArgs().copyInto(List);
  }
  
  SourceLocation getLAngleLoc() const {
    return getExplicitTemplateArgs().LAngleLoc;
  }

  SourceLocation getRAngleLoc() const {
    return getExplicitTemplateArgs().RAngleLoc;
  }

  TemplateArgumentLoc const *getTemplateArgs() const {
    return getExplicitTemplateArgs().getTemplateArgs();
  }

  unsigned getNumTemplateArgs() const {
    return getExplicitTemplateArgs().NumTemplateArgs;
  }

  SourceRange getSourceRange() const {
    SourceRange Range(getNameInfo().getSourceRange());
    if (getQualifierLoc()) 
      Range.setBegin(getQualifierLoc().getBeginLoc());
    if (hasExplicitTemplateArgs()) 
      Range.setEnd(getRAngleLoc());
    return Range;
  }

  child_range children() { return child_range(); }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == UnresolvedLookupExprClass;
  }
  static bool classof(const UnresolvedLookupExpr *) { return true; }
};

/// \brief A qualified reference to a name whose declaration cannot
/// yet be resolved.
///
/// DependentScopeDeclRefExpr is similar to DeclRefExpr in that
/// it expresses a reference to a declaration such as
/// X<T>::value. The difference, however, is that an
/// DependentScopeDeclRefExpr node is used only within C++ templates when
/// the qualification (e.g., X<T>::) refers to a dependent type. In
/// this case, X<T>::value cannot resolve to a declaration because the
/// declaration will differ from on instantiation of X<T> to the
/// next. Therefore, DependentScopeDeclRefExpr keeps track of the
/// qualifier (X<T>::) and the name of the entity being referenced
/// ("value"). Such expressions will instantiate to a DeclRefExpr once the
/// declaration can be found.
class DependentScopeDeclRefExpr : public Expr {
  /// \brief The nested-name-specifier that qualifies this unresolved
  /// declaration name.
  NestedNameSpecifierLoc QualifierLoc;
  
  /// The name of the entity we will be referencing.
  DeclarationNameInfo NameInfo;

  /// \brief Whether the name includes explicit template arguments.
  bool HasExplicitTemplateArgs;

  DependentScopeDeclRefExpr(QualType T,
                            NestedNameSpecifierLoc QualifierLoc,
                            const DeclarationNameInfo &NameInfo,
                            const TemplateArgumentListInfo *Args);

public:
  static DependentScopeDeclRefExpr *Create(ASTContext &C,
                                           NestedNameSpecifierLoc QualifierLoc,
                                           const DeclarationNameInfo &NameInfo,
                              const TemplateArgumentListInfo *TemplateArgs = 0);

  static DependentScopeDeclRefExpr *CreateEmpty(ASTContext &C,
                                                bool HasExplicitTemplateArgs,
                                                unsigned NumTemplateArgs);

  /// \brief Retrieve the name that this expression refers to.
  const DeclarationNameInfo &getNameInfo() const { return NameInfo; }

  /// \brief Retrieve the name that this expression refers to.
  DeclarationName getDeclName() const { return NameInfo.getName(); }

  /// \brief Retrieve the location of the name within the expression.
  SourceLocation getLocation() const { return NameInfo.getLoc(); }

  /// \brief Retrieve the nested-name-specifier that qualifies the
  /// name, with source location information.
  NestedNameSpecifierLoc getQualifierLoc() const { return QualifierLoc; }
  
  
  /// \brief Retrieve the nested-name-specifier that qualifies this
  /// declaration.
  NestedNameSpecifier *getQualifier() const { 
    return QualifierLoc.getNestedNameSpecifier(); 
  }

  /// Determines whether this lookup had explicit template arguments.
  bool hasExplicitTemplateArgs() const { return HasExplicitTemplateArgs; }

  // Note that, inconsistently with the explicit-template-argument AST
  // nodes, users are *forbidden* from calling these methods on objects
  // without explicit template arguments.

  ASTTemplateArgumentListInfo &getExplicitTemplateArgs() {
    assert(hasExplicitTemplateArgs());
    return *reinterpret_cast<ASTTemplateArgumentListInfo*>(this + 1);
  }

  /// Gets a reference to the explicit template argument list.
  const ASTTemplateArgumentListInfo &getExplicitTemplateArgs() const {
    assert(hasExplicitTemplateArgs());
    return *reinterpret_cast<const ASTTemplateArgumentListInfo*>(this + 1);
  }

  /// \brief Retrieves the optional explicit template arguments.
  /// This points to the same data as getExplicitTemplateArgs(), but
  /// returns null if there are no explicit template arguments.
  const ASTTemplateArgumentListInfo *getOptionalExplicitTemplateArgs() {
    if (!hasExplicitTemplateArgs()) return 0;
    return &getExplicitTemplateArgs();
  }

  /// \brief Copies the template arguments (if present) into the given
  /// structure.
  void copyTemplateArgumentsInto(TemplateArgumentListInfo &List) const {
    getExplicitTemplateArgs().copyInto(List);
  }
  
  SourceLocation getLAngleLoc() const {
    return getExplicitTemplateArgs().LAngleLoc;
  }

  SourceLocation getRAngleLoc() const {
    return getExplicitTemplateArgs().RAngleLoc;
  }

  TemplateArgumentLoc const *getTemplateArgs() const {
    return getExplicitTemplateArgs().getTemplateArgs();
  }

  unsigned getNumTemplateArgs() const {
    return getExplicitTemplateArgs().NumTemplateArgs;
  }

  SourceRange getSourceRange() const {
    SourceRange Range(QualifierLoc.getBeginLoc(), getLocation());
    if (hasExplicitTemplateArgs())
      Range.setEnd(getRAngleLoc());
    return Range;
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == DependentScopeDeclRefExprClass;
  }
  static bool classof(const DependentScopeDeclRefExpr *) { return true; }

  child_range children() { return child_range(); }

  friend class ASTStmtReader;
  friend class ASTStmtWriter;
};

/// Represents an expression --- generally a full-expression --- which
/// introduces cleanups to be run at the end of the sub-expression's
/// evaluation.  The most common source of expression-introduced
/// cleanups is temporary objects in C++, but several other C++
/// expressions can create cleanups.
class ExprWithCleanups : public Expr {
  Stmt *SubExpr;

  CXXTemporary **Temps;
  unsigned NumTemps;

  ExprWithCleanups(ASTContext &C, Expr *SubExpr,
                   CXXTemporary **Temps, unsigned NumTemps);
  
public:
  ExprWithCleanups(EmptyShell Empty)
    : Expr(ExprWithCleanupsClass, Empty),
      SubExpr(0), Temps(0), NumTemps(0) {}
                         
  static ExprWithCleanups *Create(ASTContext &C, Expr *SubExpr,
                                        CXXTemporary **Temps, 
                                        unsigned NumTemps);

  unsigned getNumTemporaries() const { return NumTemps; }
  void setNumTemporaries(ASTContext &C, unsigned N);
    
  CXXTemporary *getTemporary(unsigned i) {
    assert(i < NumTemps && "Index out of range");
    return Temps[i];
  }
  const CXXTemporary *getTemporary(unsigned i) const {
    return const_cast<ExprWithCleanups*>(this)->getTemporary(i);
  }
  void setTemporary(unsigned i, CXXTemporary *T) {
    assert(i < NumTemps && "Index out of range");
    Temps[i] = T;
  }

  Expr *getSubExpr() { return cast<Expr>(SubExpr); }
  const Expr *getSubExpr() const { return cast<Expr>(SubExpr); }
  void setSubExpr(Expr *E) { SubExpr = E; }

  SourceRange getSourceRange() const { 
    return SubExpr->getSourceRange();
  }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ExprWithCleanupsClass;
  }
  static bool classof(const ExprWithCleanups *) { return true; }

  // Iterators
  child_range children() { return child_range(&SubExpr, &SubExpr + 1); }
};

/// \brief Describes an explicit type conversion that uses functional
/// notion but could not be resolved because one or more arguments are
/// type-dependent.
///
/// The explicit type conversions expressed by
/// CXXUnresolvedConstructExpr have the form \c T(a1, a2, ..., aN),
/// where \c T is some type and \c a1, a2, ..., aN are values, and
/// either \C T is a dependent type or one or more of the \c a's is
/// type-dependent. For example, this would occur in a template such
/// as:
///
/// \code
///   template<typename T, typename A1>
///   inline T make_a(const A1& a1) {
///     return T(a1);
///   }
/// \endcode
///
/// When the returned expression is instantiated, it may resolve to a
/// constructor call, conversion function call, or some kind of type
/// conversion.
class CXXUnresolvedConstructExpr : public Expr {
  /// \brief The type being constructed.
  TypeSourceInfo *Type;
  
  /// \brief The location of the left parentheses ('(').
  SourceLocation LParenLoc;

  /// \brief The location of the right parentheses (')').
  SourceLocation RParenLoc;

  /// \brief The number of arguments used to construct the type.
  unsigned NumArgs;

  CXXUnresolvedConstructExpr(TypeSourceInfo *Type,
                             SourceLocation LParenLoc,
                             Expr **Args,
                             unsigned NumArgs,
                             SourceLocation RParenLoc);

  CXXUnresolvedConstructExpr(EmptyShell Empty, unsigned NumArgs)
    : Expr(CXXUnresolvedConstructExprClass, Empty), Type(), NumArgs(NumArgs) { }

  friend class ASTStmtReader;
  
public:
  static CXXUnresolvedConstructExpr *Create(ASTContext &C,
                                            TypeSourceInfo *Type,
                                            SourceLocation LParenLoc,
                                            Expr **Args,
                                            unsigned NumArgs,
                                            SourceLocation RParenLoc);

  static CXXUnresolvedConstructExpr *CreateEmpty(ASTContext &C,
                                                 unsigned NumArgs);

  /// \brief Retrieve the type that is being constructed, as specified
  /// in the source code.
  QualType getTypeAsWritten() const { return Type->getType(); }

  /// \brief Retrieve the type source information for the type being 
  /// constructed.
  TypeSourceInfo *getTypeSourceInfo() const { return Type; }
  
  /// \brief Retrieve the location of the left parentheses ('(') that
  /// precedes the argument list.
  SourceLocation getLParenLoc() const { return LParenLoc; }
  void setLParenLoc(SourceLocation L) { LParenLoc = L; }

  /// \brief Retrieve the location of the right parentheses (')') that
  /// follows the argument list.
  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation L) { RParenLoc = L; }

  /// \brief Retrieve the number of arguments.
  unsigned arg_size() const { return NumArgs; }

  typedef Expr** arg_iterator;
  arg_iterator arg_begin() { return reinterpret_cast<Expr**>(this + 1); }
  arg_iterator arg_end() { return arg_begin() + NumArgs; }

  typedef const Expr* const * const_arg_iterator;
  const_arg_iterator arg_begin() const {
    return reinterpret_cast<const Expr* const *>(this + 1);
  }
  const_arg_iterator arg_end() const {
    return arg_begin() + NumArgs;
  }

  Expr *getArg(unsigned I) {
    assert(I < NumArgs && "Argument index out-of-range");
    return *(arg_begin() + I);
  }

  const Expr *getArg(unsigned I) const {
    assert(I < NumArgs && "Argument index out-of-range");
    return *(arg_begin() + I);
  }

  void setArg(unsigned I, Expr *E) {
    assert(I < NumArgs && "Argument index out-of-range");
    *(arg_begin() + I) = E;
  }

  SourceRange getSourceRange() const;
  
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXUnresolvedConstructExprClass;
  }
  static bool classof(const CXXUnresolvedConstructExpr *) { return true; }

  // Iterators
  child_range children() {
    Stmt **begin = reinterpret_cast<Stmt**>(this+1);
    return child_range(begin, begin + NumArgs);
  }
};

/// \brief Represents a C++ member access expression where the actual
/// member referenced could not be resolved because the base
/// expression or the member name was dependent.
///
/// Like UnresolvedMemberExprs, these can be either implicit or
/// explicit accesses.  It is only possible to get one of these with
/// an implicit access if a qualifier is provided.
class CXXDependentScopeMemberExpr : public Expr {
  /// \brief The expression for the base pointer or class reference,
  /// e.g., the \c x in x.f.  Can be null in implicit accesses.
  Stmt *Base;

  /// \brief The type of the base expression.  Never null, even for
  /// implicit accesses.
  QualType BaseType;

  /// \brief Whether this member expression used the '->' operator or
  /// the '.' operator.
  bool IsArrow : 1;

  /// \brief Whether this member expression has explicitly-specified template
  /// arguments.
  bool HasExplicitTemplateArgs : 1;

  /// \brief The location of the '->' or '.' operator.
  SourceLocation OperatorLoc;

  /// \brief The nested-name-specifier that precedes the member name, if any.
  NestedNameSpecifierLoc QualifierLoc;

  /// \brief In a qualified member access expression such as t->Base::f, this
  /// member stores the resolves of name lookup in the context of the member
  /// access expression, to be used at instantiation time.
  ///
  /// FIXME: This member, along with the QualifierLoc, could
  /// be stuck into a structure that is optionally allocated at the end of
  /// the CXXDependentScopeMemberExpr, to save space in the common case.
  NamedDecl *FirstQualifierFoundInScope;

  /// \brief The member to which this member expression refers, which
  /// can be name, overloaded operator, or destructor.
  /// FIXME: could also be a template-id
  DeclarationNameInfo MemberNameInfo;

  CXXDependentScopeMemberExpr(ASTContext &C,
                          Expr *Base, QualType BaseType, bool IsArrow,
                          SourceLocation OperatorLoc,
                          NestedNameSpecifierLoc QualifierLoc,
                          NamedDecl *FirstQualifierFoundInScope,
                          DeclarationNameInfo MemberNameInfo,
                          const TemplateArgumentListInfo *TemplateArgs);

public:
  CXXDependentScopeMemberExpr(ASTContext &C,
                              Expr *Base, QualType BaseType,
                              bool IsArrow,
                              SourceLocation OperatorLoc,
                              NestedNameSpecifierLoc QualifierLoc,
                              NamedDecl *FirstQualifierFoundInScope,
                              DeclarationNameInfo MemberNameInfo);

  static CXXDependentScopeMemberExpr *
  Create(ASTContext &C,
         Expr *Base, QualType BaseType, bool IsArrow,
         SourceLocation OperatorLoc,
         NestedNameSpecifierLoc QualifierLoc,
         NamedDecl *FirstQualifierFoundInScope,
         DeclarationNameInfo MemberNameInfo,
         const TemplateArgumentListInfo *TemplateArgs);

  static CXXDependentScopeMemberExpr *
  CreateEmpty(ASTContext &C, bool HasExplicitTemplateArgs, 
              unsigned NumTemplateArgs);

  /// \brief True if this is an implicit access, i.e. one in which the
  /// member being accessed was not written in the source.  The source
  /// location of the operator is invalid in this case.
  bool isImplicitAccess() const;

  /// \brief Retrieve the base object of this member expressions,
  /// e.g., the \c x in \c x.m.
  Expr *getBase() const {
    assert(!isImplicitAccess());
    return cast<Expr>(Base);
  }

  QualType getBaseType() const { return BaseType; }

  /// \brief Determine whether this member expression used the '->'
  /// operator; otherwise, it used the '.' operator.
  bool isArrow() const { return IsArrow; }

  /// \brief Retrieve the location of the '->' or '.' operator.
  SourceLocation getOperatorLoc() const { return OperatorLoc; }

  /// \brief Retrieve the nested-name-specifier that qualifies the member
  /// name.
  NestedNameSpecifier *getQualifier() const { 
    return QualifierLoc.getNestedNameSpecifier(); 
  }

  /// \brief Retrieve the nested-name-specifier that qualifies the member
  /// name, with source location information.
  NestedNameSpecifierLoc getQualifierLoc() const { return QualifierLoc; }
  
  
  /// \brief Retrieve the first part of the nested-name-specifier that was
  /// found in the scope of the member access expression when the member access
  /// was initially parsed.
  ///
  /// This function only returns a useful result when member access expression
  /// uses a qualified member name, e.g., "x.Base::f". Here, the declaration
  /// returned by this function describes what was found by unqualified name
  /// lookup for the identifier "Base" within the scope of the member access
  /// expression itself. At template instantiation time, this information is
  /// combined with the results of name lookup into the type of the object
  /// expression itself (the class type of x).
  NamedDecl *getFirstQualifierFoundInScope() const {
    return FirstQualifierFoundInScope;
  }

  /// \brief Retrieve the name of the member that this expression
  /// refers to.
  const DeclarationNameInfo &getMemberNameInfo() const {
    return MemberNameInfo;
  }

  /// \brief Retrieve the name of the member that this expression
  /// refers to.
  DeclarationName getMember() const { return MemberNameInfo.getName(); }

  // \brief Retrieve the location of the name of the member that this
  // expression refers to.
  SourceLocation getMemberLoc() const { return MemberNameInfo.getLoc(); }

  /// \brief Determines whether this member expression actually had a C++
  /// template argument list explicitly specified, e.g., x.f<int>.
  bool hasExplicitTemplateArgs() const {
    return HasExplicitTemplateArgs;
  }

  /// \brief Retrieve the explicit template argument list that followed the
  /// member template name, if any.
  ASTTemplateArgumentListInfo &getExplicitTemplateArgs() {
    assert(HasExplicitTemplateArgs);
    return *reinterpret_cast<ASTTemplateArgumentListInfo *>(this + 1);
  }

  /// \brief Retrieve the explicit template argument list that followed the
  /// member template name, if any.
  const ASTTemplateArgumentListInfo &getExplicitTemplateArgs() const {
    return const_cast<CXXDependentScopeMemberExpr *>(this)
             ->getExplicitTemplateArgs();
  }

  /// \brief Retrieves the optional explicit template arguments.
  /// This points to the same data as getExplicitTemplateArgs(), but
  /// returns null if there are no explicit template arguments.
  const ASTTemplateArgumentListInfo *getOptionalExplicitTemplateArgs() {
    if (!hasExplicitTemplateArgs()) return 0;
    return &getExplicitTemplateArgs();
  }

  /// \brief Copies the template arguments (if present) into the given
  /// structure.
  void copyTemplateArgumentsInto(TemplateArgumentListInfo &List) const {
    getExplicitTemplateArgs().copyInto(List);
  }

  /// \brief Initializes the template arguments using the given structure.
  void initializeTemplateArgumentsFrom(const TemplateArgumentListInfo &List) {
    getExplicitTemplateArgs().initializeFrom(List);
  }

  /// \brief Retrieve the location of the left angle bracket following the
  /// member name ('<'), if any.
  SourceLocation getLAngleLoc() const {
    return getExplicitTemplateArgs().LAngleLoc;
  }

  /// \brief Retrieve the template arguments provided as part of this
  /// template-id.
  const TemplateArgumentLoc *getTemplateArgs() const {
    return getExplicitTemplateArgs().getTemplateArgs();
  }

  /// \brief Retrieve the number of template arguments provided as part of this
  /// template-id.
  unsigned getNumTemplateArgs() const {
    return getExplicitTemplateArgs().NumTemplateArgs;
  }

  /// \brief Retrieve the location of the right angle bracket following the
  /// template arguments ('>').
  SourceLocation getRAngleLoc() const {
    return getExplicitTemplateArgs().RAngleLoc;
  }

  SourceRange getSourceRange() const {
    SourceRange Range;
    if (!isImplicitAccess())
      Range.setBegin(Base->getSourceRange().getBegin());
    else if (getQualifier())
      Range.setBegin(getQualifierLoc().getBeginLoc());
    else
      Range.setBegin(MemberNameInfo.getBeginLoc());

    if (hasExplicitTemplateArgs())
      Range.setEnd(getRAngleLoc());
    else
      Range.setEnd(MemberNameInfo.getEndLoc());
    return Range;
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXDependentScopeMemberExprClass;
  }
  static bool classof(const CXXDependentScopeMemberExpr *) { return true; }

  // Iterators
  child_range children() {
    if (isImplicitAccess()) return child_range();
    return child_range(&Base, &Base + 1);
  }

  friend class ASTStmtReader;
  friend class ASTStmtWriter;
};

/// \brief Represents a C++ member access expression for which lookup
/// produced a set of overloaded functions.
///
/// The member access may be explicit or implicit:
///    struct A {
///      int a, b;
///      int explicitAccess() { return this->a + this->A::b; }
///      int implicitAccess() { return a + A::b; }
///    };
///
/// In the final AST, an explicit access always becomes a MemberExpr.
/// An implicit access may become either a MemberExpr or a
/// DeclRefExpr, depending on whether the member is static.
class UnresolvedMemberExpr : public OverloadExpr {
  /// \brief Whether this member expression used the '->' operator or
  /// the '.' operator.
  bool IsArrow : 1;

  /// \brief Whether the lookup results contain an unresolved using
  /// declaration.
  bool HasUnresolvedUsing : 1;

  /// \brief The expression for the base pointer or class reference,
  /// e.g., the \c x in x.f.  This can be null if this is an 'unbased'
  /// member expression
  Stmt *Base;

  /// \brief The type of the base expression;  never null.
  QualType BaseType;

  /// \brief The location of the '->' or '.' operator.
  SourceLocation OperatorLoc;

  UnresolvedMemberExpr(ASTContext &C, bool HasUnresolvedUsing,
                       Expr *Base, QualType BaseType, bool IsArrow,
                       SourceLocation OperatorLoc,
                       NestedNameSpecifierLoc QualifierLoc,
                       const DeclarationNameInfo &MemberNameInfo,
                       const TemplateArgumentListInfo *TemplateArgs,
                       UnresolvedSetIterator Begin, UnresolvedSetIterator End);
  
  UnresolvedMemberExpr(EmptyShell Empty)
    : OverloadExpr(UnresolvedMemberExprClass, Empty), IsArrow(false),
      HasUnresolvedUsing(false), Base(0) { }

  friend class ASTStmtReader;
  
public:
  static UnresolvedMemberExpr *
  Create(ASTContext &C, bool HasUnresolvedUsing,
         Expr *Base, QualType BaseType, bool IsArrow,
         SourceLocation OperatorLoc,
         NestedNameSpecifierLoc QualifierLoc,
         const DeclarationNameInfo &MemberNameInfo,
         const TemplateArgumentListInfo *TemplateArgs,
         UnresolvedSetIterator Begin, UnresolvedSetIterator End);

  static UnresolvedMemberExpr *
  CreateEmpty(ASTContext &C, bool HasExplicitTemplateArgs,
              unsigned NumTemplateArgs);

  /// \brief True if this is an implicit access, i.e. one in which the
  /// member being accessed was not written in the source.  The source
  /// location of the operator is invalid in this case.
  bool isImplicitAccess() const;

  /// \brief Retrieve the base object of this member expressions,
  /// e.g., the \c x in \c x.m.
  Expr *getBase() {
    assert(!isImplicitAccess());
    return cast<Expr>(Base);
  }
  const Expr *getBase() const {
    assert(!isImplicitAccess());
    return cast<Expr>(Base);
  }

  QualType getBaseType() const { return BaseType; }

  /// \brief Determine whether the lookup results contain an unresolved using
  /// declaration.
  bool hasUnresolvedUsing() const { return HasUnresolvedUsing; }

  /// \brief Determine whether this member expression used the '->'
  /// operator; otherwise, it used the '.' operator.
  bool isArrow() const { return IsArrow; }

  /// \brief Retrieve the location of the '->' or '.' operator.
  SourceLocation getOperatorLoc() const { return OperatorLoc; }

  /// \brief Retrieves the naming class of this lookup.
  CXXRecordDecl *getNamingClass() const;

  /// \brief Retrieve the full name info for the member that this expression
  /// refers to.
  const DeclarationNameInfo &getMemberNameInfo() const { return getNameInfo(); }

  /// \brief Retrieve the name of the member that this expression
  /// refers to.
  DeclarationName getMemberName() const { return getName(); }

  // \brief Retrieve the location of the name of the member that this
  // expression refers to.
  SourceLocation getMemberLoc() const { return getNameLoc(); }

  /// \brief Retrieve the explicit template argument list that followed the
  /// member template name.
  ASTTemplateArgumentListInfo &getExplicitTemplateArgs() {
    assert(hasExplicitTemplateArgs());
    return *reinterpret_cast<ASTTemplateArgumentListInfo *>(this + 1);
  }

  /// \brief Retrieve the explicit template argument list that followed the
  /// member template name, if any.
  const ASTTemplateArgumentListInfo &getExplicitTemplateArgs() const {
    assert(hasExplicitTemplateArgs());
    return *reinterpret_cast<const ASTTemplateArgumentListInfo *>(this + 1);
  }

  /// \brief Retrieves the optional explicit template arguments.
  /// This points to the same data as getExplicitTemplateArgs(), but
  /// returns null if there are no explicit template arguments.
  const ASTTemplateArgumentListInfo *getOptionalExplicitTemplateArgs() {
    if (!hasExplicitTemplateArgs()) return 0;
    return &getExplicitTemplateArgs();
  }

  /// \brief Copies the template arguments into the given structure.
  void copyTemplateArgumentsInto(TemplateArgumentListInfo &List) const {
    getExplicitTemplateArgs().copyInto(List);
  }

  /// \brief Retrieve the location of the left angle bracket following
  /// the member name ('<').
  SourceLocation getLAngleLoc() const {
    return getExplicitTemplateArgs().LAngleLoc;
  }

  /// \brief Retrieve the template arguments provided as part of this
  /// template-id.
  const TemplateArgumentLoc *getTemplateArgs() const {
    return getExplicitTemplateArgs().getTemplateArgs();
  }

  /// \brief Retrieve the number of template arguments provided as
  /// part of this template-id.
  unsigned getNumTemplateArgs() const {
    return getExplicitTemplateArgs().NumTemplateArgs;
  }

  /// \brief Retrieve the location of the right angle bracket
  /// following the template arguments ('>').
  SourceLocation getRAngleLoc() const {
    return getExplicitTemplateArgs().RAngleLoc;
  }

  SourceRange getSourceRange() const {
    SourceRange Range = getMemberNameInfo().getSourceRange();
    if (!isImplicitAccess())
      Range.setBegin(Base->getSourceRange().getBegin());
    else if (getQualifierLoc())
      Range.setBegin(getQualifierLoc().getBeginLoc());

    if (hasExplicitTemplateArgs())
      Range.setEnd(getRAngleLoc());
    return Range;
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == UnresolvedMemberExprClass;
  }
  static bool classof(const UnresolvedMemberExpr *) { return true; }

  // Iterators
  child_range children() {
    if (isImplicitAccess()) return child_range();
    return child_range(&Base, &Base + 1);
  }
};

/// \brief Represents a C++0x noexcept expression (C++ [expr.unary.noexcept]).
///
/// The noexcept expression tests whether a given expression might throw. Its
/// result is a boolean constant.
class CXXNoexceptExpr : public Expr {
  bool Value : 1;
  Stmt *Operand;
  SourceRange Range;

  friend class ASTStmtReader;

public:
  CXXNoexceptExpr(QualType Ty, Expr *Operand, CanThrowResult Val,
                  SourceLocation Keyword, SourceLocation RParen)
    : Expr(CXXNoexceptExprClass, Ty, VK_RValue, OK_Ordinary,
           /*TypeDependent*/false,
           /*ValueDependent*/Val == CT_Dependent,
           Val == CT_Dependent || Operand->isInstantiationDependent(),
           Operand->containsUnexpandedParameterPack()),
      Value(Val == CT_Cannot), Operand(Operand), Range(Keyword, RParen)
  { }

  CXXNoexceptExpr(EmptyShell Empty)
    : Expr(CXXNoexceptExprClass, Empty)
  { }

  Expr *getOperand() const { return static_cast<Expr*>(Operand); }

  SourceRange getSourceRange() const { return Range; }

  bool getValue() const { return Value; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXNoexceptExprClass;
  }
  static bool classof(const CXXNoexceptExpr *) { return true; }

  // Iterators
  child_range children() { return child_range(&Operand, &Operand + 1); }
};

/// \brief Represents a C++0x pack expansion that produces a sequence of 
/// expressions.
///
/// A pack expansion expression contains a pattern (which itself is an
/// expression) followed by an ellipsis. For example:
///
/// \code
/// template<typename F, typename ...Types>
/// void forward(F f, Types &&...args) {
///   f(static_cast<Types&&>(args)...);
/// }
/// \endcode
///
/// Here, the argument to the function object \c f is a pack expansion whose
/// pattern is \c static_cast<Types&&>(args). When the \c forward function 
/// template is instantiated, the pack expansion will instantiate to zero or
/// or more function arguments to the function object \c f.
class PackExpansionExpr : public Expr {
  SourceLocation EllipsisLoc;
  
  /// \brief The number of expansions that will be produced by this pack
  /// expansion expression, if known.
  ///
  /// When zero, the number of expansions is not known. Otherwise, this value
  /// is the number of expansions + 1.
  unsigned NumExpansions;
  
  Stmt *Pattern;
  
  friend class ASTStmtReader;
  friend class ASTStmtWriter;
  
public:
  PackExpansionExpr(QualType T, Expr *Pattern, SourceLocation EllipsisLoc,
                    llvm::Optional<unsigned> NumExpansions)
    : Expr(PackExpansionExprClass, T, Pattern->getValueKind(), 
           Pattern->getObjectKind(), /*TypeDependent=*/true, 
           /*ValueDependent=*/true, /*InstantiationDependent=*/true,
           /*ContainsUnexpandedParameterPack=*/false),
      EllipsisLoc(EllipsisLoc),
      NumExpansions(NumExpansions? *NumExpansions + 1 : 0),
      Pattern(Pattern) { }

  PackExpansionExpr(EmptyShell Empty) : Expr(PackExpansionExprClass, Empty) { }
  
  /// \brief Retrieve the pattern of the pack expansion.
  Expr *getPattern() { return reinterpret_cast<Expr *>(Pattern); }

  /// \brief Retrieve the pattern of the pack expansion.
  const Expr *getPattern() const { return reinterpret_cast<Expr *>(Pattern); }

  /// \brief Retrieve the location of the ellipsis that describes this pack
  /// expansion.
  SourceLocation getEllipsisLoc() const { return EllipsisLoc; }
  
  /// \brief Determine the number of expansions that will be produced when 
  /// this pack expansion is instantiated, if already known.
  llvm::Optional<unsigned> getNumExpansions() const {
    if (NumExpansions)
      return NumExpansions - 1;
    
    return llvm::Optional<unsigned>();
  }
  
  SourceRange getSourceRange() const {
    return SourceRange(Pattern->getLocStart(), EllipsisLoc);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == PackExpansionExprClass;
  }
  static bool classof(const PackExpansionExpr *) { return true; }
  
  // Iterators
  child_range children() {
    return child_range(&Pattern, &Pattern + 1);
  }
};
  
inline ASTTemplateArgumentListInfo &OverloadExpr::getExplicitTemplateArgs() {
  if (isa<UnresolvedLookupExpr>(this))
    return cast<UnresolvedLookupExpr>(this)->getExplicitTemplateArgs();
  else
    return cast<UnresolvedMemberExpr>(this)->getExplicitTemplateArgs();
}

/// \brief Represents an expression that computes the length of a parameter 
/// pack.
///
/// \code
/// template<typename ...Types>
/// struct count {
///   static const unsigned value = sizeof...(Types);
/// };
/// \endcode
class SizeOfPackExpr : public Expr {
  /// \brief The location of the 'sizeof' keyword.
  SourceLocation OperatorLoc;
  
  /// \brief The location of the name of the parameter pack.
  SourceLocation PackLoc;
  
  /// \brief The location of the closing parenthesis.
  SourceLocation RParenLoc;
  
  /// \brief The length of the parameter pack, if known.
  ///
  /// When this expression is value-dependent, the length of the parameter pack
  /// is unknown. When this expression is not value-dependent, the length is
  /// known.
  unsigned Length;
  
  /// \brief The parameter pack itself.
  NamedDecl *Pack;
  
  friend class ASTStmtReader;
  friend class ASTStmtWriter;
  
public:
  /// \brief Creates a value-dependent expression that computes the length of
  /// the given parameter pack.
  SizeOfPackExpr(QualType SizeType, SourceLocation OperatorLoc, NamedDecl *Pack, 
                 SourceLocation PackLoc, SourceLocation RParenLoc)
    : Expr(SizeOfPackExprClass, SizeType, VK_RValue, OK_Ordinary,
           /*TypeDependent=*/false, /*ValueDependent=*/true,
           /*InstantiationDependent=*/true,
           /*ContainsUnexpandedParameterPack=*/false),
      OperatorLoc(OperatorLoc), PackLoc(PackLoc), RParenLoc(RParenLoc),
      Length(0), Pack(Pack) { }

  /// \brief Creates an expression that computes the length of
  /// the given parameter pack, which is already known.
  SizeOfPackExpr(QualType SizeType, SourceLocation OperatorLoc, NamedDecl *Pack, 
                 SourceLocation PackLoc, SourceLocation RParenLoc,
                 unsigned Length)
  : Expr(SizeOfPackExprClass, SizeType, VK_RValue, OK_Ordinary,
         /*TypeDependent=*/false, /*ValueDependent=*/false,
         /*InstantiationDependent=*/false,
         /*ContainsUnexpandedParameterPack=*/false),
    OperatorLoc(OperatorLoc), PackLoc(PackLoc), RParenLoc(RParenLoc),
    Length(Length), Pack(Pack) { }

  /// \brief Create an empty expression.
  SizeOfPackExpr(EmptyShell Empty) : Expr(SizeOfPackExprClass, Empty) { }
  
  /// \brief Determine the location of the 'sizeof' keyword.
  SourceLocation getOperatorLoc() const { return OperatorLoc; }

  /// \brief Determine the location of the parameter pack.
  SourceLocation getPackLoc() const { return PackLoc; }
  
  /// \brief Determine the location of the right parenthesis.
  SourceLocation getRParenLoc() const { return RParenLoc; }
  
  /// \brief Retrieve the parameter pack.
  NamedDecl *getPack() const { return Pack; }
  
  /// \brief Retrieve the length of the parameter pack.
  ///
  /// This routine may only be invoked when the expression is not 
  /// value-dependent.
  unsigned getPackLength() const {
    assert(!isValueDependent() && 
           "Cannot get the length of a value-dependent pack size expression");
    return Length;
  }
  
  SourceRange getSourceRange() const {
    return SourceRange(OperatorLoc, RParenLoc);
  }
  
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == SizeOfPackExprClass;
  }
  static bool classof(const SizeOfPackExpr *) { return true; }
  
  // Iterators
  child_range children() { return child_range(); }
};

/// \brief Represents a reference to a non-type template parameter
/// that has been substituted with a template argument.
class SubstNonTypeTemplateParmExpr : public Expr {
  /// \brief The replaced parameter.
  NonTypeTemplateParmDecl *Param;

  /// \brief The replacement expression.
  Stmt *Replacement;

  /// \brief The location of the non-type template parameter reference.
  SourceLocation NameLoc;

  friend class ASTReader;
  friend class ASTStmtReader;
  explicit SubstNonTypeTemplateParmExpr(EmptyShell Empty) 
    : Expr(SubstNonTypeTemplateParmExprClass, Empty) { }

public:
  SubstNonTypeTemplateParmExpr(QualType type, 
                               ExprValueKind valueKind,
                               SourceLocation loc,
                               NonTypeTemplateParmDecl *param,
                               Expr *replacement)
    : Expr(SubstNonTypeTemplateParmExprClass, type, valueKind, OK_Ordinary,
           replacement->isTypeDependent(), replacement->isValueDependent(),
           replacement->isInstantiationDependent(),
           replacement->containsUnexpandedParameterPack()),
      Param(param), Replacement(replacement), NameLoc(loc) {}

  SourceLocation getNameLoc() const { return NameLoc; }
  SourceRange getSourceRange() const { return NameLoc; }

  Expr *getReplacement() const { return cast<Expr>(Replacement); }
    
  NonTypeTemplateParmDecl *getParameter() const { return Param; }

  static bool classof(const Stmt *s) {
    return s->getStmtClass() == SubstNonTypeTemplateParmExprClass;
  }
  static bool classof(const SubstNonTypeTemplateParmExpr *) { 
    return true; 
  }
  
  // Iterators
  child_range children() { return child_range(&Replacement, &Replacement+1); }
};

/// \brief Represents a reference to a non-type template parameter pack that
/// has been substituted with a non-template argument pack.
///
/// When a pack expansion in the source code contains multiple parameter packs
/// and those parameter packs correspond to different levels of template
/// parameter lists, this node node is used to represent a non-type template 
/// parameter pack from an outer level, which has already had its argument pack
/// substituted but that still lives within a pack expansion that itself
/// could not be instantiated. When actually performing a substitution into
/// that pack expansion (e.g., when all template parameters have corresponding
/// arguments), this type will be replaced with the appropriate underlying
/// expression at the current pack substitution index.
class SubstNonTypeTemplateParmPackExpr : public Expr {
  /// \brief The non-type template parameter pack itself.
  NonTypeTemplateParmDecl *Param;
  
  /// \brief A pointer to the set of template arguments that this
  /// parameter pack is instantiated with.
  const TemplateArgument *Arguments;
  
  /// \brief The number of template arguments in \c Arguments.
  unsigned NumArguments;
  
  /// \brief The location of the non-type template parameter pack reference.
  SourceLocation NameLoc;
  
  friend class ASTReader;
  friend class ASTStmtReader;
  explicit SubstNonTypeTemplateParmPackExpr(EmptyShell Empty) 
    : Expr(SubstNonTypeTemplateParmPackExprClass, Empty) { }
  
public:
  SubstNonTypeTemplateParmPackExpr(QualType T, 
                                   NonTypeTemplateParmDecl *Param,
                                   SourceLocation NameLoc,
                                   const TemplateArgument &ArgPack);
  
  /// \brief Retrieve the non-type template parameter pack being substituted.
  NonTypeTemplateParmDecl *getParameterPack() const { return Param; }

  /// \brief Retrieve the location of the parameter pack name.
  SourceLocation getParameterPackLocation() const { return NameLoc; }
  
  /// \brief Retrieve the template argument pack containing the substituted
  /// template arguments.
  TemplateArgument getArgumentPack() const;

  SourceRange getSourceRange() const { return NameLoc; }
  
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == SubstNonTypeTemplateParmPackExprClass;
  }
  static bool classof(const SubstNonTypeTemplateParmPackExpr *) { 
    return true; 
  }
  
  // Iterators
  child_range children() { return child_range(); }
};

/// \brief Represents a prvalue temporary that written into memory so that
/// a reference can bind to it.
///
/// Prvalue expressions are materialized when they need to have an address
/// in memory for a reference to bind to. This happens when binding a
/// reference to the result of a conversion, e.g.,
///
/// \code
/// const int &r = 1.0;
/// \endcode
///
/// Here, 1.0 is implicitly converted to an \c int. That resulting \c int is
/// then materialized via a \c MaterializeTemporaryExpr, and the reference
/// binds to the temporary. \c MaterializeTemporaryExprs are always glvalues
/// (either an lvalue or an xvalue, depending on the kind of reference binding
/// to it), maintaining the invariant that references always bind to glvalues.
class MaterializeTemporaryExpr : public Expr {
  /// \brief The temporary-generating expression whose value will be
  /// materialized.
 Stmt *Temporary;
  
  friend class ASTStmtReader;
  friend class ASTStmtWriter;
  
public:
  MaterializeTemporaryExpr(QualType T, Expr *Temporary, 
                           bool BoundToLvalueReference)
    : Expr(MaterializeTemporaryExprClass, T,
           BoundToLvalueReference? VK_LValue : VK_XValue, OK_Ordinary,
           Temporary->isTypeDependent(), Temporary->isValueDependent(),
           Temporary->isInstantiationDependent(),
           Temporary->containsUnexpandedParameterPack()),
      Temporary(Temporary) { }
  
  MaterializeTemporaryExpr(EmptyShell Empty) 
    : Expr(MaterializeTemporaryExprClass, Empty) { }
  
  /// \brief Retrieve the temporary-generating subexpression whose value will
  /// be materialized into a glvalue.
  Expr *GetTemporaryExpr() const { return reinterpret_cast<Expr *>(Temporary); }
  
  /// \brief Determine whether this materialized temporary is bound to an
  /// lvalue reference; otherwise, it's bound to an rvalue reference.
  bool isBoundToLvalueReference() const { 
    return getValueKind() == VK_LValue;
  }
  
  SourceRange getSourceRange() const { return Temporary->getSourceRange(); }
  
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == MaterializeTemporaryExprClass;
  }
  static bool classof(const MaterializeTemporaryExpr *) { 
    return true; 
  }
  
  // Iterators
  child_range children() { return child_range(&Temporary, &Temporary + 1); }
};
  
}  // end namespace clang

#endif
