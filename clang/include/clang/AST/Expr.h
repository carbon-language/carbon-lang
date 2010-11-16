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

#include "clang/AST/APValue.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/AST/DeclAccessPair.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/ASTVector.h"
#include "clang/AST/UsuallyTinyPtrVector.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <vector>

namespace clang {
  class ASTContext;
  class APValue;
  class Decl;
  class IdentifierInfo;
  class ParmVarDecl;
  class NamedDecl;
  class ValueDecl;
  class BlockDecl;
  class CXXBaseSpecifier;
  class CXXOperatorCallExpr;
  class CXXMemberCallExpr;
  class TemplateArgumentLoc;
  class TemplateArgumentListInfo;

/// \brief A simple array of base specifiers.
typedef llvm::SmallVector<CXXBaseSpecifier*, 4> CXXCastPath;

/// Expr - This represents one expression.  Note that Expr's are subclasses of
/// Stmt.  This allows an expression to be transparently used any place a Stmt
/// is required.
///
class Expr : public Stmt {
  QualType TR;

  virtual void ANCHOR(); // key function.

protected:
  Expr(StmtClass SC, QualType T, bool TD, bool VD) : Stmt(SC) {
    ExprBits.TypeDependent = TD;
    ExprBits.ValueDependent = VD;
    ExprBits.ValueKind = 0;
    setType(T);
  }

  /// \brief Construct an empty expression.
  explicit Expr(StmtClass SC, EmptyShell) : Stmt(SC) { }

  /// getValueKind - The value kind that this cast produces.
  ExprValueKind getValueKind() const {
    return static_cast<ExprValueKind>(ExprBits.ValueKind);
  }

  /// setValueKind - Set the value kind this cast produces.
  void setValueKind(ExprValueKind Cat) { ExprBits.ValueKind = Cat; }

public:
  QualType getType() const { return TR; }
  void setType(QualType t) {
    // In C++, the type of an expression is always adjusted so that it
    // will not have reference type an expression will never have
    // reference type (C++ [expr]p6). Use
    // QualType::getNonReferenceType() to retrieve the non-reference
    // type. Additionally, inspect Expr::isLvalue to determine whether
    // an expression that is adjusted in this manner should be
    // considered an lvalue.
    assert((t.isNull() || !t->isReferenceType()) &&
           "Expressions can't have reference type");

    TR = t;
  }

  /// isValueDependent - Determines whether this expression is
  /// value-dependent (C++ [temp.dep.constexpr]). For example, the
  /// array bound of "Chars" in the following example is
  /// value-dependent.
  /// @code
  /// template<int Size, char (&Chars)[Size]> struct meta_string;
  /// @endcode
  bool isValueDependent() const { return ExprBits.ValueDependent; }

  /// \brief Set whether this expression is value-dependent or not.
  void setValueDependent(bool VD) { ExprBits.ValueDependent = VD; }

  /// isTypeDependent - Determines whether this expression is
  /// type-dependent (C++ [temp.dep.expr]), which means that its type
  /// could change from one template instantiation to the next. For
  /// example, the expressions "x" and "x + y" are type-dependent in
  /// the following code, but "y" is not type-dependent:
  /// @code
  /// template<typename T>
  /// void add(T x, int y) {
  ///   x + y;
  /// }
  /// @endcode
  bool isTypeDependent() const { return ExprBits.TypeDependent; }

  /// \brief Set whether this expression is type-dependent or not.
  void setTypeDependent(bool TD) { ExprBits.TypeDependent = TD; }

  /// SourceLocation tokens are not useful in isolation - they are low level
  /// value objects created/interpreted by SourceManager. We assume AST
  /// clients will have a pointer to the respective SourceManager.
  virtual SourceRange getSourceRange() const = 0;

  /// getExprLoc - Return the preferred location for the arrow when diagnosing
  /// a problem with a generic expression.
  virtual SourceLocation getExprLoc() const { return getLocStart(); }

  /// isUnusedResultAWarning - Return true if this immediate expression should
  /// be warned about if the result is unused.  If so, fill in Loc and Ranges
  /// with location to warn on and the source range[s] to report with the
  /// warning.
  bool isUnusedResultAWarning(SourceLocation &Loc, SourceRange &R1,
                              SourceRange &R2, ASTContext &Ctx) const;

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
  ///  - b ? x : y, where x and y are lvalues of suitable types [C++]
  ///
  enum isLvalueResult {
    LV_Valid,
    LV_NotObjectType,
    LV_IncompleteVoidType,
    LV_DuplicateVectorComponents,
    LV_InvalidExpression,
    LV_MemberFunction,
    LV_SubObjCPropertySetting,
    LV_ClassTemporary
  };
  isLvalueResult isLvalue(ASTContext &Ctx) const;

  /// isModifiableLvalue - C99 6.3.2.1: an lvalue that does not have array type,
  /// does not have an incomplete type, does not have a const-qualified type,
  /// and if it is a structure or union, does not have any member (including,
  /// recursively, any member or element of all contained aggregates or unions)
  /// with a const-qualified type.
  ///
  /// \param Loc [in] [out] - A source location which *may* be filled
  /// in with the location of the expression making this a
  /// non-modifiable lvalue, if specified.
  enum isModifiableLvalueResult {
    MLV_Valid,
    MLV_NotObjectType,
    MLV_IncompleteVoidType,
    MLV_DuplicateVectorComponents,
    MLV_InvalidExpression,
    MLV_LValueCast,           // Specialized form of MLV_InvalidExpression.
    MLV_IncompleteType,
    MLV_ConstQualified,
    MLV_ArrayType,
    MLV_NotBlockQualified,
    MLV_ReadonlyProperty,
    MLV_NoSetterProperty,
    MLV_MemberFunction,
    MLV_SubObjCPropertySetting,
    MLV_ClassTemporary
  };
  isModifiableLvalueResult isModifiableLvalue(ASTContext &Ctx,
                                              SourceLocation *Loc = 0) const;

  /// \brief The return type of classify(). Represents the C++0x expression
  ///        taxonomy.
  class Classification {
  public:
    /// \brief The various classification results. Most of these mean prvalue.
    enum Kinds {
      CL_LValue,
      CL_XValue,
      CL_Function, // Functions cannot be lvalues in C.
      CL_Void, // Void cannot be an lvalue in C.
      CL_DuplicateVectorComponents, // A vector shuffle with dupes.
      CL_MemberFunction, // An expression referring to a member function
      CL_SubObjCPropertySetting,
      CL_ClassTemporary, // A prvalue of class type
      CL_PRValue // A prvalue for any other reason, of any other type
    };
    /// \brief The results of modification testing.
    enum ModifiableType {
      CM_Untested, // testModifiable was false.
      CM_Modifiable,
      CM_RValue, // Not modifiable because it's an rvalue
      CM_Function, // Not modifiable because it's a function; C++ only
      CM_LValueCast, // Same as CM_RValue, but indicates GCC cast-as-lvalue ext
      CM_NotBlockQualified, // Not captured in the closure
      CM_NoSetterProperty,// Implicit assignment to ObjC property without setter
      CM_ConstQualified,
      CM_ArrayType,
      CM_IncompleteType
    };

  private:
    friend class Expr;

    unsigned short Kind;
    unsigned short Modifiable;

    explicit Classification(Kinds k, ModifiableType m)
      : Kind(k), Modifiable(m)
    {}

  public:
    Classification() {}

    Kinds getKind() const { return static_cast<Kinds>(Kind); }
    ModifiableType getModifiable() const {
      assert(Modifiable != CM_Untested && "Did not test for modifiability.");
      return static_cast<ModifiableType>(Modifiable);
    }
    bool isLValue() const { return Kind == CL_LValue; }
    bool isXValue() const { return Kind == CL_XValue; }
    bool isGLValue() const { return Kind <= CL_XValue; }
    bool isPRValue() const { return Kind >= CL_Function; }
    bool isRValue() const { return Kind >= CL_XValue; }
    bool isModifiable() const { return getModifiable() == CM_Modifiable; }
  };
  /// \brief Classify - Classify this expression according to the C++0x
  ///        expression taxonomy.
  ///
  /// C++0x defines ([basic.lval]) a new taxonomy of expressions to replace the
  /// old lvalue vs rvalue. This function determines the type of expression this
  /// is. There are three expression types:
  /// - lvalues are classical lvalues as in C++03.
  /// - prvalues are equivalent to rvalues in C++03.
  /// - xvalues are expressions yielding unnamed rvalue references, e.g. a
  ///   function returning an rvalue reference.
  /// lvalues and xvalues are collectively referred to as glvalues, while
  /// prvalues and xvalues together form rvalues.
  Classification Classify(ASTContext &Ctx) const {
    return ClassifyImpl(Ctx, 0);
  }

  /// \brief ClassifyModifiable - Classify this expression according to the
  ///        C++0x expression taxonomy, and see if it is valid on the left side
  ///        of an assignment.
  ///
  /// This function extends classify in that it also tests whether the
  /// expression is modifiable (C99 6.3.2.1p1).
  /// \param Loc A source location that might be filled with a relevant location
  ///            if the expression is not modifiable.
  Classification ClassifyModifiable(ASTContext &Ctx, SourceLocation &Loc) const{
    return ClassifyImpl(Ctx, &Loc);
  }

private:
  Classification ClassifyImpl(ASTContext &Ctx, SourceLocation *Loc) const;

public:

  /// \brief If this expression refers to a bit-field, retrieve the
  /// declaration of that bit-field.
  FieldDecl *getBitField();

  const FieldDecl *getBitField() const {
    return const_cast<Expr*>(this)->getBitField();
  }

  /// \brief Returns whether this expression refers to a vector element.
  bool refersToVectorElement() const;
  
  /// isKnownToHaveBooleanValue - Return true if this is an integer expression
  /// that is known to return 0 or 1.  This happens for _Bool/bool expressions
  /// but also int expressions which are produced by things like comparisons in
  /// C.
  bool isKnownToHaveBooleanValue() const;
  
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
  /// isConstantInitializer - Returns true if this expression is a constant
  /// initializer, which can be emitted at compile-time.
  bool isConstantInitializer(ASTContext &Ctx, bool ForRef) const;

  /// EvalResult is a struct with detailed info about an evaluated expression.
  struct EvalResult {
    /// Val - This is the value the expression can be folded to.
    APValue Val;

    /// HasSideEffects - Whether the evaluated expression has side effects.
    /// For example, (f() && 0) can be folded, but it still has side effects.
    bool HasSideEffects;

    /// Diag - If the expression is unfoldable, then Diag contains a note
    /// diagnostic indicating why it's not foldable. DiagLoc indicates a caret
    /// position for the error, and DiagExpr is the expression that caused
    /// the error.
    /// If the expression is foldable, but not an integer constant expression,
    /// Diag contains a note diagnostic that describes why it isn't an integer
    /// constant expression. If the expression *is* an integer constant
    /// expression, then Diag will be zero.
    unsigned Diag;
    const Expr *DiagExpr;
    SourceLocation DiagLoc;

    EvalResult() : HasSideEffects(false), Diag(0), DiagExpr(0) {}

    // isGlobalLValue - Return true if the evaluated lvalue expression
    // is global.
    bool isGlobalLValue() const;
    // hasSideEffects - Return true if the evaluated expression has
    // side effects.
    bool hasSideEffects() const {
      return HasSideEffects;
    }
  };

  /// Evaluate - Return true if this is a constant which we can fold using
  /// any crazy technique (that has nothing to do with language standards) that
  /// we want to.  If this function returns true, it returns the folded constant
  /// in Result.
  bool Evaluate(EvalResult &Result, ASTContext &Ctx) const;

  /// EvaluateAsBooleanCondition - Return true if this is a constant
  /// which we we can fold and convert to a boolean condition using
  /// any crazy technique that we want to.
  bool EvaluateAsBooleanCondition(bool &Result, ASTContext &Ctx) const;

  /// isEvaluatable - Call Evaluate to see if this expression can be constant
  /// folded, but discard the result.
  bool isEvaluatable(ASTContext &Ctx) const;

  /// HasSideEffects - This routine returns true for all those expressions
  /// which must be evaluated each time and must not be optimized away 
  /// or evaluated at compile time. Example is a function call, volatile
  /// variable read.
  bool HasSideEffects(ASTContext &Ctx) const;
  
  /// EvaluateAsInt - Call Evaluate and return the folded integer. This
  /// must be called on an expression that constant folds to an integer.
  llvm::APSInt EvaluateAsInt(ASTContext &Ctx) const;

  /// EvaluateAsLValue - Evaluate an expression to see if it's a lvalue
  /// with link time known address.
  bool EvaluateAsLValue(EvalResult &Result, ASTContext &Ctx) const;

  /// EvaluateAsLValue - Evaluate an expression to see if it's a lvalue.
  bool EvaluateAsAnyLValue(EvalResult &Result, ASTContext &Ctx) const;

  /// \brief Enumeration used to describe how \c isNullPointerConstant()
  /// should cope with value-dependent expressions.
  enum NullPointerConstantValueDependence {
    /// \brief Specifies that the expression should never be value-dependent.
    NPC_NeverValueDependent = 0,
    
    /// \brief Specifies that a value-dependent expression of integral or
    /// dependent type should be considered a null pointer constant.
    NPC_ValueDependentIsNull,
    
    /// \brief Specifies that a value-dependent expression should be considered
    /// to never be a null pointer constant.
    NPC_ValueDependentIsNotNull
  };
  
  /// isNullPointerConstant - C99 6.3.2.3p3 -  Return true if this is either an
  /// integer constant expression with the value zero, or if this is one that is
  /// cast to void*.
  bool isNullPointerConstant(ASTContext &Ctx,
                             NullPointerConstantValueDependence NPC) const;

  /// isOBJCGCCandidate - Return true if this expression may be used in a read/
  /// write barrier.
  bool isOBJCGCCandidate(ASTContext &Ctx) const;

  /// \brief Returns true if this expression is a bound member function.
  bool isBoundMemberFunction(ASTContext &Ctx) const;

  /// \brief Result type of CanThrow().
  enum CanThrowResult {
    CT_Cannot,
    CT_Dependent,
    CT_Can
  };
  /// \brief Test if this expression, if evaluated, might throw, according to
  ///        the rules of C++ [expr.unary.noexcept].
  CanThrowResult CanThrow(ASTContext &C) const;

  /// IgnoreParens - Ignore parentheses.  If this Expr is a ParenExpr, return
  ///  its subexpression.  If that subexpression is also a ParenExpr,
  ///  then this method recursively returns its subexpression, and so forth.
  ///  Otherwise, the method returns the current Expr.
  Expr *IgnoreParens();

  /// IgnoreParenCasts - Ignore parentheses and casts.  Strip off any ParenExpr
  /// or CastExprs, returning their operand.
  Expr *IgnoreParenCasts();

  /// IgnoreParenImpCasts - Ignore parentheses and implicit casts.  Strip off any
  /// ParenExpr or ImplicitCastExprs, returning their operand.
  Expr *IgnoreParenImpCasts();

  const Expr *IgnoreParenImpCasts() const {
    return const_cast<Expr*>(this)->IgnoreParenImpCasts();
  }

  /// IgnoreParenNoopCasts - Ignore parentheses and casts that do not change the
  /// value (including ptr->int casts of the same size).  Strip off any
  /// ParenExpr or CastExprs, returning their operand.
  Expr *IgnoreParenNoopCasts(ASTContext &Ctx);

  /// \brief Determine whether this expression is a default function argument.
  ///
  /// Default arguments are implicitly generated in the abstract syntax tree
  /// by semantic analysis for function calls, object constructions, etc. in 
  /// C++. Default arguments are represented by \c CXXDefaultArgExpr nodes;
  /// this routine also looks through any implicit casts to determine whether
  /// the expression is a default argument.
  bool isDefaultArgument() const;
  
  /// \brief Determine whether the result of this expression is a
  /// temporary object of the given class type.
  bool isTemporaryObject(ASTContext &Ctx, const CXXRecordDecl *TempTy) const;

  const Expr *IgnoreParens() const {
    return const_cast<Expr*>(this)->IgnoreParens();
  }
  const Expr *IgnoreParenCasts() const {
    return const_cast<Expr*>(this)->IgnoreParenCasts();
  }
  const Expr *IgnoreParenNoopCasts(ASTContext &Ctx) const {
    return const_cast<Expr*>(this)->IgnoreParenNoopCasts(Ctx);
  }

  static bool hasAnyTypeDependentArguments(Expr** Exprs, unsigned NumExprs);
  static bool hasAnyValueDependentArguments(Expr** Exprs, unsigned NumExprs);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() >= firstExprConstant &&
           T->getStmtClass() <= lastExprConstant;
  }
  static bool classof(const Expr *) { return true; }
};


//===----------------------------------------------------------------------===//
// Primary Expressions.
//===----------------------------------------------------------------------===//

/// \brief Represents the qualifier that may precede a C++ name, e.g., the
/// "std::" in "std::sort".
struct NameQualifier {
  /// \brief The nested name specifier.
  NestedNameSpecifier *NNS;
  
  /// \brief The source range covered by the nested name specifier.
  SourceRange Range;
};

/// \brief Represents an explicit template argument list in C++, e.g.,
/// the "<int>" in "sort<int>".
struct ExplicitTemplateArgumentList {
  /// \brief The source location of the left angle bracket ('<');
  SourceLocation LAngleLoc;
  
  /// \brief The source location of the right angle bracket ('>');
  SourceLocation RAngleLoc;
  
  /// \brief The number of template arguments in TemplateArgs.
  /// The actual template arguments (if any) are stored after the
  /// ExplicitTemplateArgumentList structure.
  unsigned NumTemplateArgs;
  
  /// \brief Retrieve the template arguments
  TemplateArgumentLoc *getTemplateArgs() {
    return reinterpret_cast<TemplateArgumentLoc *> (this + 1);
  }
  
  /// \brief Retrieve the template arguments
  const TemplateArgumentLoc *getTemplateArgs() const {
    return reinterpret_cast<const TemplateArgumentLoc *> (this + 1);
  }

  void initializeFrom(const TemplateArgumentListInfo &List);
  void copyInto(TemplateArgumentListInfo &List) const;
  static std::size_t sizeFor(unsigned NumTemplateArgs);
  static std::size_t sizeFor(const TemplateArgumentListInfo &List);
};
  
/// DeclRefExpr - [C99 6.5.1p2] - A reference to a declared variable, function,
/// enum, etc.
class DeclRefExpr : public Expr {
  enum {
    // Flag on DecoratedD that specifies when this declaration reference 
    // expression has a C++ nested-name-specifier.
    HasQualifierFlag = 0x01,
    // Flag on DecoratedD that specifies when this declaration reference 
    // expression has an explicit C++ template argument list.
    HasExplicitTemplateArgumentListFlag = 0x02
  };
  
  // DecoratedD - The declaration that we are referencing, plus two bits to 
  // indicate whether (1) the declaration's name was explicitly qualified and
  // (2) the declaration's name was followed by an explicit template 
  // argument list.
  llvm::PointerIntPair<ValueDecl *, 2> DecoratedD;

  // Loc - The location of the declaration name itself.
  SourceLocation Loc;

  /// DNLoc - Provides source/type location info for the
  /// declaration name embedded in DecoratedD.
  DeclarationNameLoc DNLoc;

  /// \brief Retrieve the qualifier that preceded the declaration name, if any.
  NameQualifier *getNameQualifier() {
    if ((DecoratedD.getInt() & HasQualifierFlag) == 0)
      return 0;
    
    return reinterpret_cast<NameQualifier *> (this + 1);
  }
  
  /// \brief Retrieve the qualifier that preceded the member name, if any.
  const NameQualifier *getNameQualifier() const {
    return const_cast<DeclRefExpr *>(this)->getNameQualifier();
  }

  DeclRefExpr(NestedNameSpecifier *Qualifier, SourceRange QualifierRange,
              ValueDecl *D, SourceLocation NameLoc,
              const TemplateArgumentListInfo *TemplateArgs,
              QualType T);

  DeclRefExpr(NestedNameSpecifier *Qualifier, SourceRange QualifierRange,
              ValueDecl *D, const DeclarationNameInfo &NameInfo,
              const TemplateArgumentListInfo *TemplateArgs,
              QualType T);

  /// \brief Construct an empty declaration reference expression.
  explicit DeclRefExpr(EmptyShell Empty)
    : Expr(DeclRefExprClass, Empty) { }
  
  /// \brief Computes the type- and value-dependence flags for this
  /// declaration reference expression.
  void computeDependence();

public:
  DeclRefExpr(ValueDecl *d, QualType t, SourceLocation l) :
    Expr(DeclRefExprClass, t, false, false), DecoratedD(d, 0), Loc(l) {
    computeDependence();
  }

  static DeclRefExpr *Create(ASTContext &Context,
                             NestedNameSpecifier *Qualifier,
                             SourceRange QualifierRange,
                             ValueDecl *D,
                             SourceLocation NameLoc,
                             QualType T,
                             const TemplateArgumentListInfo *TemplateArgs = 0);

  static DeclRefExpr *Create(ASTContext &Context,
                             NestedNameSpecifier *Qualifier,
                             SourceRange QualifierRange,
                             ValueDecl *D,
                             const DeclarationNameInfo &NameInfo,
                             QualType T,
                             const TemplateArgumentListInfo *TemplateArgs = 0);

  /// \brief Construct an empty declaration reference expression.
  static DeclRefExpr *CreateEmpty(ASTContext &Context,
                                  bool HasQualifier, unsigned NumTemplateArgs);
  
  ValueDecl *getDecl() { return DecoratedD.getPointer(); }
  const ValueDecl *getDecl() const { return DecoratedD.getPointer(); }
  void setDecl(ValueDecl *NewD) { DecoratedD.setPointer(NewD); }

  DeclarationNameInfo getNameInfo() const {
    return DeclarationNameInfo(getDecl()->getDeclName(), Loc, DNLoc);
  }

  SourceLocation getLocation() const { return Loc; }
  void setLocation(SourceLocation L) { Loc = L; }
  virtual SourceRange getSourceRange() const;

  /// \brief Determine whether this declaration reference was preceded by a
  /// C++ nested-name-specifier, e.g., \c N::foo.
  bool hasQualifier() const { return DecoratedD.getInt() & HasQualifierFlag; }
  
  /// \brief If the name was qualified, retrieves the source range of
  /// the nested-name-specifier that precedes the name. Otherwise,
  /// returns an empty source range.
  SourceRange getQualifierRange() const {
    if (!hasQualifier())
      return SourceRange();
    
    return getNameQualifier()->Range;
  }
  
  /// \brief If the name was qualified, retrieves the nested-name-specifier 
  /// that precedes the name. Otherwise, returns NULL.
  NestedNameSpecifier *getQualifier() const {
    if (!hasQualifier())
      return 0;
    
    return getNameQualifier()->NNS;
  }
  
  bool hasExplicitTemplateArgs() const {
    return (DecoratedD.getInt() & HasExplicitTemplateArgumentListFlag);
  }
  
  /// \brief Retrieve the explicit template argument list that followed the
  /// member template name.
  ExplicitTemplateArgumentList &getExplicitTemplateArgs() {
    assert(hasExplicitTemplateArgs());

    if ((DecoratedD.getInt() & HasQualifierFlag) == 0)
      return *reinterpret_cast<ExplicitTemplateArgumentList *>(this + 1);
    
    return *reinterpret_cast<ExplicitTemplateArgumentList *>(
                                                      getNameQualifier() + 1);
  }
  
  /// \brief Retrieve the explicit template argument list that followed the
  /// member template name.
  const ExplicitTemplateArgumentList &getExplicitTemplateArgs() const {
    return const_cast<DeclRefExpr *>(this)->getExplicitTemplateArgs();
  }

  /// \brief Retrieves the optional explicit template arguments.
  /// This points to the same data as getExplicitTemplateArgs(), but
  /// returns null if there are no explicit template arguments.
  const ExplicitTemplateArgumentList *getExplicitTemplateArgsOpt() const {
    if (!hasExplicitTemplateArgs()) return 0;
    return &getExplicitTemplateArgs();
  }
  
  /// \brief Copies the template arguments (if present) into the given
  /// structure.
  void copyTemplateArgumentsInto(TemplateArgumentListInfo &List) const {
    if (hasExplicitTemplateArgs())
      getExplicitTemplateArgs().copyInto(List);
  }
  
  /// \brief Retrieve the location of the left angle bracket following the
  /// member name ('<'), if any.
  SourceLocation getLAngleLoc() const {
    if (!hasExplicitTemplateArgs())
      return SourceLocation();
    
    return getExplicitTemplateArgs().LAngleLoc;
  }
  
  /// \brief Retrieve the template arguments provided as part of this
  /// template-id.
  const TemplateArgumentLoc *getTemplateArgs() const {
    if (!hasExplicitTemplateArgs())
      return 0;
    
    return getExplicitTemplateArgs().getTemplateArgs();
  }
  
  /// \brief Retrieve the number of template arguments provided as part of this
  /// template-id.
  unsigned getNumTemplateArgs() const {
    if (!hasExplicitTemplateArgs())
      return 0;
    
    return getExplicitTemplateArgs().NumTemplateArgs;
  }
  
  /// \brief Retrieve the location of the right angle bracket following the
  /// template arguments ('>').
  SourceLocation getRAngleLoc() const {
    if (!hasExplicitTemplateArgs())
      return SourceLocation();
    
    return getExplicitTemplateArgs().RAngleLoc;
  }
  
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == DeclRefExprClass;
  }
  static bool classof(const DeclRefExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  friend class ASTStmtReader;
  friend class ASTStmtWriter;
};

/// PredefinedExpr - [C99 6.4.2.2] - A predefined identifier such as __func__.
class PredefinedExpr : public Expr {
public:
  enum IdentType {
    Func,
    Function,
    PrettyFunction,
    /// PrettyFunctionNoVirtual - The same as PrettyFunction, except that the
    /// 'virtual' keyword is omitted for virtual member functions.
    PrettyFunctionNoVirtual
  };

private:
  SourceLocation Loc;
  IdentType Type;
public:
  PredefinedExpr(SourceLocation l, QualType type, IdentType IT)
    : Expr(PredefinedExprClass, type, type->isDependentType(), 
           type->isDependentType()), Loc(l), Type(IT) {}

  /// \brief Construct an empty predefined expression.
  explicit PredefinedExpr(EmptyShell Empty)
    : Expr(PredefinedExprClass, Empty) { }

  IdentType getIdentType() const { return Type; }
  void setIdentType(IdentType IT) { Type = IT; }

  SourceLocation getLocation() const { return Loc; }
  void setLocation(SourceLocation L) { Loc = L; }

  static std::string ComputeName(IdentType IT, const Decl *CurrentDecl);

  virtual SourceRange getSourceRange() const { return SourceRange(Loc); }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == PredefinedExprClass;
  }
  static bool classof(const PredefinedExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// \brief Used by IntegerLiteral/FloatingLiteral to store the numeric without
/// leaking memory.
///
/// For large floats/integers, APFloat/APInt will allocate memory from the heap
/// to represent these numbers.  Unfortunately, when we use a BumpPtrAllocator
/// to allocate IntegerLiteral/FloatingLiteral nodes the memory associated with
/// the APFloat/APInt values will never get freed. APNumericStorage uses
/// ASTContext's allocator for memory allocation.
class APNumericStorage {
  unsigned BitWidth;
  union {
    uint64_t VAL;    ///< Used to store the <= 64 bits integer value.
    uint64_t *pVal;  ///< Used to store the >64 bits integer value.
  };

  bool hasAllocation() const { return llvm::APInt::getNumWords(BitWidth) > 1; }

  APNumericStorage(const APNumericStorage&); // do not implement
  APNumericStorage& operator=(const APNumericStorage&); // do not implement

protected:
  APNumericStorage() : BitWidth(0), VAL(0) { }

  llvm::APInt getIntValue() const {
    unsigned NumWords = llvm::APInt::getNumWords(BitWidth);
    if (NumWords > 1)
      return llvm::APInt(BitWidth, NumWords, pVal);
    else
      return llvm::APInt(BitWidth, VAL);
  }
  void setIntValue(ASTContext &C, const llvm::APInt &Val);
};

class APIntStorage : public APNumericStorage {
public:  
  llvm::APInt getValue() const { return getIntValue(); } 
  void setValue(ASTContext &C, const llvm::APInt &Val) { setIntValue(C, Val); }
};

class APFloatStorage : public APNumericStorage {
public:  
  llvm::APFloat getValue() const { return llvm::APFloat(getIntValue()); } 
  void setValue(ASTContext &C, const llvm::APFloat &Val) {
    setIntValue(C, Val.bitcastToAPInt());
  }
};

class IntegerLiteral : public Expr {
  APIntStorage Num;
  SourceLocation Loc;

  /// \brief Construct an empty integer literal.
  explicit IntegerLiteral(EmptyShell Empty)
    : Expr(IntegerLiteralClass, Empty) { }

public:
  // type should be IntTy, LongTy, LongLongTy, UnsignedIntTy, UnsignedLongTy,
  // or UnsignedLongLongTy
  IntegerLiteral(ASTContext &C, const llvm::APInt &V,
                 QualType type, SourceLocation l)
    : Expr(IntegerLiteralClass, type, false, false), Loc(l) {
    assert(type->isIntegerType() && "Illegal type in IntegerLiteral");
    setValue(C, V);
  }

  // type should be IntTy, LongTy, LongLongTy, UnsignedIntTy, UnsignedLongTy,
  // or UnsignedLongLongTy
  static IntegerLiteral *Create(ASTContext &C, const llvm::APInt &V,
                                QualType type, SourceLocation l);
  static IntegerLiteral *Create(ASTContext &C, EmptyShell Empty);

  llvm::APInt getValue() const { return Num.getValue(); }
  virtual SourceRange getSourceRange() const { return SourceRange(Loc); }

  /// \brief Retrieve the location of the literal.
  SourceLocation getLocation() const { return Loc; }

  void setValue(ASTContext &C, const llvm::APInt &Val) { Num.setValue(C, Val); }
  void setLocation(SourceLocation Location) { Loc = Location; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == IntegerLiteralClass;
  }
  static bool classof(const IntegerLiteral *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

class CharacterLiteral : public Expr {
  unsigned Value;
  SourceLocation Loc;
  bool IsWide;
public:
  // type should be IntTy
  CharacterLiteral(unsigned value, bool iswide, QualType type, SourceLocation l)
    : Expr(CharacterLiteralClass, type, false, false), Value(value), Loc(l),
      IsWide(iswide) {
  }

  /// \brief Construct an empty character literal.
  CharacterLiteral(EmptyShell Empty) : Expr(CharacterLiteralClass, Empty) { }

  SourceLocation getLocation() const { return Loc; }
  bool isWide() const { return IsWide; }

  virtual SourceRange getSourceRange() const { return SourceRange(Loc); }

  unsigned getValue() const { return Value; }

  void setLocation(SourceLocation Location) { Loc = Location; }
  void setWide(bool W) { IsWide = W; }
  void setValue(unsigned Val) { Value = Val; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CharacterLiteralClass;
  }
  static bool classof(const CharacterLiteral *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

class FloatingLiteral : public Expr {
  APFloatStorage Num;
  bool IsExact : 1;
  SourceLocation Loc;

  FloatingLiteral(ASTContext &C, const llvm::APFloat &V, bool isexact,
                  QualType Type, SourceLocation L)
    : Expr(FloatingLiteralClass, Type, false, false),
      IsExact(isexact), Loc(L) {
    setValue(C, V);
  }

  /// \brief Construct an empty floating-point literal.
  explicit FloatingLiteral(EmptyShell Empty)
    : Expr(FloatingLiteralClass, Empty), IsExact(false) { }

public:
  static FloatingLiteral *Create(ASTContext &C, const llvm::APFloat &V,
                                 bool isexact, QualType Type, SourceLocation L);
  static FloatingLiteral *Create(ASTContext &C, EmptyShell Empty);

  llvm::APFloat getValue() const { return Num.getValue(); }
  void setValue(ASTContext &C, const llvm::APFloat &Val) {
    Num.setValue(C, Val);
  }

  bool isExact() const { return IsExact; }
  void setExact(bool E) { IsExact = E; }

  /// getValueAsApproximateDouble - This returns the value as an inaccurate
  /// double.  Note that this may cause loss of precision, but is useful for
  /// debugging dumps, etc.
  double getValueAsApproximateDouble() const;

  SourceLocation getLocation() const { return Loc; }
  void setLocation(SourceLocation L) { Loc = L; }

  virtual SourceRange getSourceRange() const { return SourceRange(Loc); }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == FloatingLiteralClass;
  }
  static bool classof(const FloatingLiteral *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
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
    : Expr(ImaginaryLiteralClass, Ty, false, false), Val(val) {}

  /// \brief Build an empty imaginary literal.
  explicit ImaginaryLiteral(EmptyShell Empty)
    : Expr(ImaginaryLiteralClass, Empty) { }

  const Expr *getSubExpr() const { return cast<Expr>(Val); }
  Expr *getSubExpr() { return cast<Expr>(Val); }
  void setSubExpr(Expr *E) { Val = E; }

  virtual SourceRange getSourceRange() const { return Val->getSourceRange(); }
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ImaginaryLiteralClass;
  }
  static bool classof(const ImaginaryLiteral *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// StringLiteral - This represents a string literal expression, e.g. "foo"
/// or L"bar" (wide strings).  The actual string is returned by getStrData()
/// is NOT null-terminated, and the length of the string is determined by
/// calling getByteLength().  The C type for a string is always a
/// ConstantArrayType.  In C++, the char type is const qualified, in C it is
/// not.
///
/// Note that strings in C can be formed by concatenation of multiple string
/// literal pptokens in translation phase #6.  This keeps track of the locations
/// of each of these pieces.
///
/// Strings in C can also be truncated and extended by assigning into arrays,
/// e.g. with constructs like:
///   char X[2] = "foobar";
/// In this case, getByteLength() will return 6, but the string literal will
/// have type "char[2]".
class StringLiteral : public Expr {
  const char *StrData;
  unsigned ByteLength;
  bool IsWide;
  unsigned NumConcatenated;
  SourceLocation TokLocs[1];

  StringLiteral(QualType Ty) : Expr(StringLiteralClass, Ty, false, false) {}

public:
  /// This is the "fully general" constructor that allows representation of
  /// strings formed from multiple concatenated tokens.
  static StringLiteral *Create(ASTContext &C, const char *StrData,
                               unsigned ByteLength, bool Wide, QualType Ty,
                               const SourceLocation *Loc, unsigned NumStrs);

  /// Simple constructor for string literals made from one token.
  static StringLiteral *Create(ASTContext &C, const char *StrData,
                               unsigned ByteLength,
                               bool Wide, QualType Ty, SourceLocation Loc) {
    return Create(C, StrData, ByteLength, Wide, Ty, &Loc, 1);
  }

  /// \brief Construct an empty string literal.
  static StringLiteral *CreateEmpty(ASTContext &C, unsigned NumStrs);

  llvm::StringRef getString() const {
    return llvm::StringRef(StrData, ByteLength);
  }

  unsigned getByteLength() const { return ByteLength; }

  /// \brief Sets the string data to the given string data.
  void setString(ASTContext &C, llvm::StringRef Str);

  bool isWide() const { return IsWide; }
  void setWide(bool W) { IsWide = W; }

  bool containsNonAsciiOrNull() const {
    llvm::StringRef Str = getString();
    for (unsigned i = 0, e = Str.size(); i != e; ++i)
      if (!isascii(Str[i]) || !Str[i])
        return true;
    return false;
  }
  /// getNumConcatenated - Get the number of string literal tokens that were
  /// concatenated in translation phase #6 to form this string literal.
  unsigned getNumConcatenated() const { return NumConcatenated; }

  SourceLocation getStrTokenLoc(unsigned TokNum) const {
    assert(TokNum < NumConcatenated && "Invalid tok number");
    return TokLocs[TokNum];
  }
  void setStrTokenLoc(unsigned TokNum, SourceLocation L) {
    assert(TokNum < NumConcatenated && "Invalid tok number");
    TokLocs[TokNum] = L;
  }

  typedef const SourceLocation *tokloc_iterator;
  tokloc_iterator tokloc_begin() const { return TokLocs; }
  tokloc_iterator tokloc_end() const { return TokLocs+NumConcatenated; }

  virtual SourceRange getSourceRange() const {
    return SourceRange(TokLocs[0], TokLocs[NumConcatenated-1]);
  }
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == StringLiteralClass;
  }
  static bool classof(const StringLiteral *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// ParenExpr - This represents a parethesized expression, e.g. "(1)".  This
/// AST node is only formed if full location information is requested.
class ParenExpr : public Expr {
  SourceLocation L, R;
  Stmt *Val;
public:
  ParenExpr(SourceLocation l, SourceLocation r, Expr *val)
    : Expr(ParenExprClass, val->getType(),
           val->isTypeDependent(), val->isValueDependent()),
      L(l), R(r), Val(val) {}

  /// \brief Construct an empty parenthesized expression.
  explicit ParenExpr(EmptyShell Empty)
    : Expr(ParenExprClass, Empty) { }

  const Expr *getSubExpr() const { return cast<Expr>(Val); }
  Expr *getSubExpr() { return cast<Expr>(Val); }
  void setSubExpr(Expr *E) { Val = E; }

  virtual SourceRange getSourceRange() const { return SourceRange(L, R); }

  /// \brief Get the location of the left parentheses '('.
  SourceLocation getLParen() const { return L; }
  void setLParen(SourceLocation Loc) { L = Loc; }

  /// \brief Get the location of the right parentheses ')'.
  SourceLocation getRParen() const { return R; }
  void setRParen(SourceLocation Loc) { R = Loc; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ParenExprClass;
  }
  static bool classof(const ParenExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};


/// UnaryOperator - This represents the unary-expression's (except sizeof and
/// alignof), the postinc/postdec operators from postfix-expression, and various
/// extensions.
///
/// Notes on various nodes:
///
/// Real/Imag - These return the real/imag part of a complex operand.  If
///   applied to a non-complex value, the former returns its operand and the
///   later returns zero in the type of the operand.
///
class UnaryOperator : public Expr {
public:
  typedef UnaryOperatorKind Opcode;

private:
  unsigned Opc : 5;
  SourceLocation Loc;
  Stmt *Val;
public:

  UnaryOperator(Expr *input, Opcode opc, QualType type, SourceLocation l)
    : Expr(UnaryOperatorClass, type,
           input->isTypeDependent() || type->isDependentType(),
           input->isValueDependent()),
      Opc(opc), Loc(l), Val(input) {}

  /// \brief Build an empty unary operator.
  explicit UnaryOperator(EmptyShell Empty)
    : Expr(UnaryOperatorClass, Empty), Opc(UO_AddrOf) { }

  Opcode getOpcode() const { return static_cast<Opcode>(Opc); }
  void setOpcode(Opcode O) { Opc = O; }

  Expr *getSubExpr() const { return cast<Expr>(Val); }
  void setSubExpr(Expr *E) { Val = E; }

  /// getOperatorLoc - Return the location of the operator.
  SourceLocation getOperatorLoc() const { return Loc; }
  void setOperatorLoc(SourceLocation L) { Loc = L; }

  /// isPostfix - Return true if this is a postfix operation, like x++.
  static bool isPostfix(Opcode Op) {
    return Op == UO_PostInc || Op == UO_PostDec;
  }

  /// isPostfix - Return true if this is a prefix operation, like --x.
  static bool isPrefix(Opcode Op) {
    return Op == UO_PreInc || Op == UO_PreDec;
  }

  bool isPrefix() const { return isPrefix(getOpcode()); }
  bool isPostfix() const { return isPostfix(getOpcode()); }
  bool isIncrementOp() const {
    return Opc == UO_PreInc || Opc == UO_PostInc;
  }
  bool isIncrementDecrementOp() const {
    return Opc <= UO_PreDec;
  }
  static bool isArithmeticOp(Opcode Op) {
    return Op >= UO_Plus && Op <= UO_LNot;
  }
  bool isArithmeticOp() const { return isArithmeticOp(getOpcode()); }

  /// getOpcodeStr - Turn an Opcode enum value into the punctuation char it
  /// corresponds to, e.g. "sizeof" or "[pre]++"
  static const char *getOpcodeStr(Opcode Op);

  /// \brief Retrieve the unary opcode that corresponds to the given
  /// overloaded operator.
  static Opcode getOverloadedOpcode(OverloadedOperatorKind OO, bool Postfix);

  /// \brief Retrieve the overloaded operator kind that corresponds to
  /// the given unary opcode.
  static OverloadedOperatorKind getOverloadedOperator(Opcode Opc);

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

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// OffsetOfExpr - [C99 7.17] - This represents an expression of the form
/// offsetof(record-type, member-designator). For example, given:
/// @code
/// struct S {
///   float f;
///   double d;    
/// };
/// struct T {
///   int i;
///   struct S s[10];
/// };
/// @endcode
/// we can represent and evaluate the expression @c offsetof(struct T, s[2].d). 

class OffsetOfExpr : public Expr {
public:
  // __builtin_offsetof(type, identifier(.identifier|[expr])*)
  class OffsetOfNode {
  public:
    /// \brief The kind of offsetof node we have.
    enum Kind {
      /// \brief An index into an array.
      Array = 0x00,
      /// \brief A field.
      Field = 0x01,
      /// \brief A field in a dependent type, known only by its name.
      Identifier = 0x02,
      /// \brief An implicit indirection through a C++ base class, when the
      /// field found is in a base class.
      Base = 0x03
    };

  private:
    enum { MaskBits = 2, Mask = 0x03 };
    
    /// \brief The source range that covers this part of the designator.
    SourceRange Range;
    
    /// \brief The data describing the designator, which comes in three
    /// different forms, depending on the lower two bits.
    ///   - An unsigned index into the array of Expr*'s stored after this node 
    ///     in memory, for [constant-expression] designators.
    ///   - A FieldDecl*, for references to a known field.
    ///   - An IdentifierInfo*, for references to a field with a given name
    ///     when the class type is dependent.
    ///   - A CXXBaseSpecifier*, for references that look at a field in a 
    ///     base class.
    uintptr_t Data;
    
  public:
    /// \brief Create an offsetof node that refers to an array element.
    OffsetOfNode(SourceLocation LBracketLoc, unsigned Index, 
                 SourceLocation RBracketLoc)
      : Range(LBracketLoc, RBracketLoc), Data((Index << 2) | Array) { }
    
    /// \brief Create an offsetof node that refers to a field.
    OffsetOfNode(SourceLocation DotLoc, FieldDecl *Field, 
                 SourceLocation NameLoc)
      : Range(DotLoc.isValid()? DotLoc : NameLoc, NameLoc), 
        Data(reinterpret_cast<uintptr_t>(Field) | OffsetOfNode::Field) { }
    
    /// \brief Create an offsetof node that refers to an identifier.
    OffsetOfNode(SourceLocation DotLoc, IdentifierInfo *Name,
                 SourceLocation NameLoc)
      : Range(DotLoc.isValid()? DotLoc : NameLoc, NameLoc), 
        Data(reinterpret_cast<uintptr_t>(Name) | Identifier) { }

    /// \brief Create an offsetof node that refers into a C++ base class.
    explicit OffsetOfNode(const CXXBaseSpecifier *Base)
      : Range(), Data(reinterpret_cast<uintptr_t>(Base) | OffsetOfNode::Base) {}
    
    /// \brief Determine what kind of offsetof node this is.
    Kind getKind() const { 
      return static_cast<Kind>(Data & Mask);
    }
    
    /// \brief For an array element node, returns the index into the array
    /// of expressions.
    unsigned getArrayExprIndex() const {
      assert(getKind() == Array);
      return Data >> 2;
    }

    /// \brief For a field offsetof node, returns the field.
    FieldDecl *getField() const {
      assert(getKind() == Field);
      return reinterpret_cast<FieldDecl *>(Data & ~(uintptr_t)Mask);
    }
    
    /// \brief For a field or identifier offsetof node, returns the name of
    /// the field.
    IdentifierInfo *getFieldName() const;
    
    /// \brief For a base class node, returns the base specifier.
    CXXBaseSpecifier *getBase() const {
      assert(getKind() == Base);
      return reinterpret_cast<CXXBaseSpecifier *>(Data & ~(uintptr_t)Mask);      
    }
    
    /// \brief Retrieve the source range that covers this offsetof node.
    ///
    /// For an array element node, the source range contains the locations of
    /// the square brackets. For a field or identifier node, the source range
    /// contains the location of the period (if there is one) and the 
    /// identifier.
    SourceRange getRange() const { return Range; }
  };

private:
  
  SourceLocation OperatorLoc, RParenLoc;
  // Base type;
  TypeSourceInfo *TSInfo;
  // Number of sub-components (i.e. instances of OffsetOfNode).
  unsigned NumComps;
  // Number of sub-expressions (i.e. array subscript expressions).
  unsigned NumExprs;
  
  OffsetOfExpr(ASTContext &C, QualType type, 
               SourceLocation OperatorLoc, TypeSourceInfo *tsi,
               OffsetOfNode* compsPtr, unsigned numComps, 
               Expr** exprsPtr, unsigned numExprs,
               SourceLocation RParenLoc);

  explicit OffsetOfExpr(unsigned numComps, unsigned numExprs)
    : Expr(OffsetOfExprClass, EmptyShell()),
      TSInfo(0), NumComps(numComps), NumExprs(numExprs) {}  

public:
  
  static OffsetOfExpr *Create(ASTContext &C, QualType type, 
                              SourceLocation OperatorLoc, TypeSourceInfo *tsi, 
                              OffsetOfNode* compsPtr, unsigned numComps, 
                              Expr** exprsPtr, unsigned numExprs,
                              SourceLocation RParenLoc);

  static OffsetOfExpr *CreateEmpty(ASTContext &C, 
                                   unsigned NumComps, unsigned NumExprs);

  /// getOperatorLoc - Return the location of the operator.
  SourceLocation getOperatorLoc() const { return OperatorLoc; }
  void setOperatorLoc(SourceLocation L) { OperatorLoc = L; }

  /// \brief Return the location of the right parentheses.
  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation R) { RParenLoc = R; }
  
  TypeSourceInfo *getTypeSourceInfo() const {
    return TSInfo;
  }
  void setTypeSourceInfo(TypeSourceInfo *tsi) {
    TSInfo = tsi;
  }
  
  const OffsetOfNode &getComponent(unsigned Idx) {
    assert(Idx < NumComps && "Subscript out of range");
    return reinterpret_cast<OffsetOfNode *> (this + 1)[Idx];
  }

  void setComponent(unsigned Idx, OffsetOfNode ON) {
    assert(Idx < NumComps && "Subscript out of range");
    reinterpret_cast<OffsetOfNode *> (this + 1)[Idx] = ON;
  }
  
  unsigned getNumComponents() const {
    return NumComps;
  }

  Expr* getIndexExpr(unsigned Idx) {
    assert(Idx < NumExprs && "Subscript out of range");
    return reinterpret_cast<Expr **>(
                    reinterpret_cast<OffsetOfNode *>(this+1) + NumComps)[Idx];
  }

  void setIndexExpr(unsigned Idx, Expr* E) {
    assert(Idx < NumComps && "Subscript out of range");
    reinterpret_cast<Expr **>(
                reinterpret_cast<OffsetOfNode *>(this+1) + NumComps)[Idx] = E;
  }
  
  unsigned getNumExpressions() const {
    return NumExprs;
  }

  virtual SourceRange getSourceRange() const {
    return SourceRange(OperatorLoc, RParenLoc);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OffsetOfExprClass;
  }

  static bool classof(const OffsetOfExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// SizeOfAlignOfExpr - [C99 6.5.3.4] - This is for sizeof/alignof, both of
/// types and expressions.
class SizeOfAlignOfExpr : public Expr {
  bool isSizeof : 1;  // true if sizeof, false if alignof.
  bool isType : 1;    // true if operand is a type, false if an expression
  union {
    TypeSourceInfo *Ty;
    Stmt *Ex;
  } Argument;
  SourceLocation OpLoc, RParenLoc;

public:
  SizeOfAlignOfExpr(bool issizeof, TypeSourceInfo *TInfo,
                    QualType resultType, SourceLocation op,
                    SourceLocation rp) :
      Expr(SizeOfAlignOfExprClass, resultType,
           false, // Never type-dependent (C++ [temp.dep.expr]p3).
           // Value-dependent if the argument is type-dependent.
           TInfo->getType()->isDependentType()),
      isSizeof(issizeof), isType(true), OpLoc(op), RParenLoc(rp) {
    Argument.Ty = TInfo;
  }

  SizeOfAlignOfExpr(bool issizeof, Expr *E,
                    QualType resultType, SourceLocation op,
                    SourceLocation rp) :
      Expr(SizeOfAlignOfExprClass, resultType,
           false, // Never type-dependent (C++ [temp.dep.expr]p3).
           // Value-dependent if the argument is type-dependent.
           E->isTypeDependent()),
      isSizeof(issizeof), isType(false), OpLoc(op), RParenLoc(rp) {
    Argument.Ex = E;
  }

  /// \brief Construct an empty sizeof/alignof expression.
  explicit SizeOfAlignOfExpr(EmptyShell Empty)
    : Expr(SizeOfAlignOfExprClass, Empty) { }

  bool isSizeOf() const { return isSizeof; }
  void setSizeof(bool S) { isSizeof = S; }

  bool isArgumentType() const { return isType; }
  QualType getArgumentType() const {
    return getArgumentTypeInfo()->getType();
  }
  TypeSourceInfo *getArgumentTypeInfo() const {
    assert(isArgumentType() && "calling getArgumentType() when arg is expr");
    return Argument.Ty;
  }
  Expr *getArgumentExpr() {
    assert(!isArgumentType() && "calling getArgumentExpr() when arg is type");
    return static_cast<Expr*>(Argument.Ex);
  }
  const Expr *getArgumentExpr() const {
    return const_cast<SizeOfAlignOfExpr*>(this)->getArgumentExpr();
  }

  void setArgument(Expr *E) { Argument.Ex = E; isType = false; }
  void setArgument(TypeSourceInfo *TInfo) {
    Argument.Ty = TInfo;
    isType = true;
  }

  /// Gets the argument type, or the type of the argument expression, whichever
  /// is appropriate.
  QualType getTypeOfArgument() const {
    return isArgumentType() ? getArgumentType() : getArgumentExpr()->getType();
  }

  SourceLocation getOperatorLoc() const { return OpLoc; }
  void setOperatorLoc(SourceLocation L) { OpLoc = L; }

  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation L) { RParenLoc = L; }

  virtual SourceRange getSourceRange() const {
    return SourceRange(OpLoc, RParenLoc);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == SizeOfAlignOfExprClass;
  }
  static bool classof(const SizeOfAlignOfExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
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
  : Expr(ArraySubscriptExprClass, t,
         lhs->isTypeDependent() || rhs->isTypeDependent(),
         lhs->isValueDependent() || rhs->isValueDependent()),
    RBracketLoc(rbracketloc) {
    SubExprs[LHS] = lhs;
    SubExprs[RHS] = rhs;
  }

  /// \brief Create an empty array subscript expression.
  explicit ArraySubscriptExpr(EmptyShell Shell)
    : Expr(ArraySubscriptExprClass, Shell) { }

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
  void setLHS(Expr *E) { SubExprs[LHS] = E; }

  Expr *getRHS() { return cast<Expr>(SubExprs[RHS]); }
  const Expr *getRHS() const { return cast<Expr>(SubExprs[RHS]); }
  void setRHS(Expr *E) { SubExprs[RHS] = E; }

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

  SourceLocation getRBracketLoc() const { return RBracketLoc; }
  void setRBracketLoc(SourceLocation L) { RBracketLoc = L; }

  virtual SourceLocation getExprLoc() const { return getBase()->getExprLoc(); }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ArraySubscriptExprClass;
  }
  static bool classof(const ArraySubscriptExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};


/// CallExpr - Represents a function call (C99 6.5.2.2, C++ [expr.call]).
/// CallExpr itself represents a normal function call, e.g., "f(x, 2)",
/// while its subclasses may represent alternative syntax that (semantically)
/// results in a function call. For example, CXXOperatorCallExpr is
/// a subclass for overloaded operator calls that use operator syntax, e.g.,
/// "str1 + str2" to resolve to a function call.
class CallExpr : public Expr {
  enum { FN=0, ARGS_START=1 };
  Stmt **SubExprs;
  unsigned NumArgs;
  SourceLocation RParenLoc;

protected:
  // This version of the constructor is for derived classes.
  CallExpr(ASTContext& C, StmtClass SC, Expr *fn, Expr **args, unsigned numargs,
           QualType t, SourceLocation rparenloc);

public:
  CallExpr(ASTContext& C, Expr *fn, Expr **args, unsigned numargs, QualType t,
           SourceLocation rparenloc);

  /// \brief Build an empty call expression.
  CallExpr(ASTContext &C, StmtClass SC, EmptyShell Empty);

  const Expr *getCallee() const { return cast<Expr>(SubExprs[FN]); }
  Expr *getCallee() { return cast<Expr>(SubExprs[FN]); }
  void setCallee(Expr *F) { SubExprs[FN] = F; }

  Decl *getCalleeDecl();
  const Decl *getCalleeDecl() const {
    return const_cast<CallExpr*>(this)->getCalleeDecl();
  }

  /// \brief If the callee is a FunctionDecl, return it. Otherwise return 0.
  FunctionDecl *getDirectCallee();
  const FunctionDecl *getDirectCallee() const {
    return const_cast<CallExpr*>(this)->getDirectCallee();
  }

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
  void setNumArgs(ASTContext& C, unsigned NumArgs);

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
  unsigned isBuiltinCall(ASTContext &Context) const;

  /// getCallReturnType - Get the return type of the call expr. This is not
  /// always the type of the expr itself, if the return type is a reference
  /// type.
  QualType getCallReturnType() const;

  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation L) { RParenLoc = L; }

  virtual SourceRange getSourceRange() const {
    return SourceRange(getCallee()->getLocStart(), RParenLoc);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() >= firstCallExprConstant &&
           T->getStmtClass() <= lastCallExprConstant;
  }
  static bool classof(const CallExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// MemberExpr - [C99 6.5.2.3] Structure and Union Members.  X->F and X.F.
///
class MemberExpr : public Expr {
  /// Extra data stored in some member expressions.
  struct MemberNameQualifier : public NameQualifier {
    DeclAccessPair FoundDecl;
  };

  /// Base - the expression for the base pointer or structure references.  In
  /// X.F, this is "X".
  Stmt *Base;

  /// MemberDecl - This is the decl being referenced by the field/member name.
  /// In X.F, this is the decl referenced by F.
  ValueDecl *MemberDecl;

  /// MemberLoc - This is the location of the member name.
  SourceLocation MemberLoc;

  /// MemberDNLoc - Provides source/type location info for the
  /// declaration name embedded in MemberDecl.
  DeclarationNameLoc MemberDNLoc;

  /// IsArrow - True if this is "X->F", false if this is "X.F".
  bool IsArrow : 1;

  /// \brief True if this member expression used a nested-name-specifier to
  /// refer to the member, e.g., "x->Base::f", or found its member via a using
  /// declaration.  When true, a MemberNameQualifier
  /// structure is allocated immediately after the MemberExpr.
  bool HasQualifierOrFoundDecl : 1;

  /// \brief True if this member expression specified a template argument list
  /// explicitly, e.g., x->f<int>. When true, an ExplicitTemplateArgumentList
  /// structure (and its TemplateArguments) are allocated immediately after
  /// the MemberExpr or, if the member expression also has a qualifier, after
  /// the MemberNameQualifier structure.
  bool HasExplicitTemplateArgumentList : 1;

  /// \brief Retrieve the qualifier that preceded the member name, if any.
  MemberNameQualifier *getMemberQualifier() {
    assert(HasQualifierOrFoundDecl);
    return reinterpret_cast<MemberNameQualifier *> (this + 1);
  }

  /// \brief Retrieve the qualifier that preceded the member name, if any.
  const MemberNameQualifier *getMemberQualifier() const {
    return const_cast<MemberExpr *>(this)->getMemberQualifier();
  }

public:
  MemberExpr(Expr *base, bool isarrow, ValueDecl *memberdecl,
             const DeclarationNameInfo &NameInfo, QualType ty)
    : Expr(MemberExprClass, ty,
           base->isTypeDependent(), base->isValueDependent()),
      Base(base), MemberDecl(memberdecl), MemberLoc(NameInfo.getLoc()),
      MemberDNLoc(NameInfo.getInfo()), IsArrow(isarrow),
      HasQualifierOrFoundDecl(false), HasExplicitTemplateArgumentList(false) {
    assert(memberdecl->getDeclName() == NameInfo.getName());
  }

  // NOTE: this constructor should be used only when it is known that
  // the member name can not provide additional syntactic info
  // (i.e., source locations for C++ operator names or type source info
  // for constructors, destructors and conversion oeprators).
  MemberExpr(Expr *base, bool isarrow, ValueDecl *memberdecl,
             SourceLocation l, QualType ty)
    : Expr(MemberExprClass, ty,
           base->isTypeDependent(), base->isValueDependent()),
      Base(base), MemberDecl(memberdecl), MemberLoc(l), MemberDNLoc(),
      IsArrow(isarrow),
      HasQualifierOrFoundDecl(false), HasExplicitTemplateArgumentList(false) {}

  static MemberExpr *Create(ASTContext &C, Expr *base, bool isarrow,
                            NestedNameSpecifier *qual, SourceRange qualrange,
                            ValueDecl *memberdecl, DeclAccessPair founddecl,
                            DeclarationNameInfo MemberNameInfo,
                            const TemplateArgumentListInfo *targs,
                            QualType ty);

  void setBase(Expr *E) { Base = E; }
  Expr *getBase() const { return cast<Expr>(Base); }

  /// \brief Retrieve the member declaration to which this expression refers.
  ///
  /// The returned declaration will either be a FieldDecl or (in C++)
  /// a CXXMethodDecl.
  ValueDecl *getMemberDecl() const { return MemberDecl; }
  void setMemberDecl(ValueDecl *D) { MemberDecl = D; }

  /// \brief Retrieves the declaration found by lookup.
  DeclAccessPair getFoundDecl() const {
    if (!HasQualifierOrFoundDecl)
      return DeclAccessPair::make(getMemberDecl(),
                                  getMemberDecl()->getAccess());
    return getMemberQualifier()->FoundDecl;
  }

  /// \brief Determines whether this member expression actually had
  /// a C++ nested-name-specifier prior to the name of the member, e.g.,
  /// x->Base::foo.
  bool hasQualifier() const { return getQualifier() != 0; }

  /// \brief If the member name was qualified, retrieves the source range of
  /// the nested-name-specifier that precedes the member name. Otherwise,
  /// returns an empty source range.
  SourceRange getQualifierRange() const {
    if (!HasQualifierOrFoundDecl)
      return SourceRange();

    return getMemberQualifier()->Range;
  }

  /// \brief If the member name was qualified, retrieves the
  /// nested-name-specifier that precedes the member name. Otherwise, returns
  /// NULL.
  NestedNameSpecifier *getQualifier() const {
    if (!HasQualifierOrFoundDecl)
      return 0;

    return getMemberQualifier()->NNS;
  }

  /// \brief Determines whether this member expression actually had a C++
  /// template argument list explicitly specified, e.g., x.f<int>.
  bool hasExplicitTemplateArgs() const {
    return HasExplicitTemplateArgumentList;
  }

  /// \brief Copies the template arguments (if present) into the given
  /// structure.
  void copyTemplateArgumentsInto(TemplateArgumentListInfo &List) const {
    if (hasExplicitTemplateArgs())
      getExplicitTemplateArgs().copyInto(List);
  }

  /// \brief Retrieve the explicit template argument list that
  /// follow the member template name.  This must only be called on an
  /// expression with explicit template arguments.
  ExplicitTemplateArgumentList &getExplicitTemplateArgs() {
    assert(HasExplicitTemplateArgumentList);
    if (!HasQualifierOrFoundDecl)
      return *reinterpret_cast<ExplicitTemplateArgumentList *>(this + 1);

    return *reinterpret_cast<ExplicitTemplateArgumentList *>(
                                                      getMemberQualifier() + 1);
  }

  /// \brief Retrieve the explicit template argument list that
  /// followed the member template name.  This must only be called on
  /// an expression with explicit template arguments.
  const ExplicitTemplateArgumentList &getExplicitTemplateArgs() const {
    return const_cast<MemberExpr *>(this)->getExplicitTemplateArgs();
  }

  /// \brief Retrieves the optional explicit template arguments.
  /// This points to the same data as getExplicitTemplateArgs(), but
  /// returns null if there are no explicit template arguments.
  const ExplicitTemplateArgumentList *getOptionalExplicitTemplateArgs() const {
    if (!hasExplicitTemplateArgs()) return 0;
    return &getExplicitTemplateArgs();
  }
  
  /// \brief Retrieve the location of the left angle bracket following the
  /// member name ('<'), if any.
  SourceLocation getLAngleLoc() const {
    if (!HasExplicitTemplateArgumentList)
      return SourceLocation();

    return getExplicitTemplateArgs().LAngleLoc;
  }

  /// \brief Retrieve the template arguments provided as part of this
  /// template-id.
  const TemplateArgumentLoc *getTemplateArgs() const {
    if (!HasExplicitTemplateArgumentList)
      return 0;

    return getExplicitTemplateArgs().getTemplateArgs();
  }

  /// \brief Retrieve the number of template arguments provided as part of this
  /// template-id.
  unsigned getNumTemplateArgs() const {
    if (!HasExplicitTemplateArgumentList)
      return 0;

    return getExplicitTemplateArgs().NumTemplateArgs;
  }

  /// \brief Retrieve the location of the right angle bracket following the
  /// template arguments ('>').
  SourceLocation getRAngleLoc() const {
    if (!HasExplicitTemplateArgumentList)
      return SourceLocation();

    return getExplicitTemplateArgs().RAngleLoc;
  }

  /// \brief Retrieve the member declaration name info.
  DeclarationNameInfo getMemberNameInfo() const {
    return DeclarationNameInfo(MemberDecl->getDeclName(),
                               MemberLoc, MemberDNLoc);
  }

  bool isArrow() const { return IsArrow; }
  void setArrow(bool A) { IsArrow = A; }

  /// getMemberLoc - Return the location of the "member", in X->F, it is the
  /// location of 'F'.
  SourceLocation getMemberLoc() const { return MemberLoc; }
  void setMemberLoc(SourceLocation L) { MemberLoc = L; }

  virtual SourceRange getSourceRange() const {
    // If we have an implicit base (like a C++ implicit this),
    // make sure not to return its location
    SourceLocation EndLoc = (HasExplicitTemplateArgumentList)
      ? getRAngleLoc() : getMemberNameInfo().getEndLoc();

    SourceLocation BaseLoc = getBase()->getLocStart();
    if (BaseLoc.isInvalid())
      return SourceRange(MemberLoc, EndLoc);
    return SourceRange(BaseLoc, EndLoc);
  }

  virtual SourceLocation getExprLoc() const { return MemberLoc; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == MemberExprClass;
  }
  static bool classof(const MemberExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();

  friend class ASTReader;
  friend class ASTStmtWriter;
};

/// CompoundLiteralExpr - [C99 6.5.2.5]
///
class CompoundLiteralExpr : public Expr {
  /// LParenLoc - If non-null, this is the location of the left paren in a
  /// compound literal like "(int){4}".  This can be null if this is a
  /// synthesized compound expression.
  SourceLocation LParenLoc;

  /// The type as written.  This can be an incomplete array type, in
  /// which case the actual expression type will be different.
  TypeSourceInfo *TInfo;
  Stmt *Init;
  bool FileScope;
public:
  // FIXME: Can compound literals be value-dependent?
  CompoundLiteralExpr(SourceLocation lparenloc, TypeSourceInfo *tinfo,
                      QualType T, Expr *init, bool fileScope)
    : Expr(CompoundLiteralExprClass, T,
           tinfo->getType()->isDependentType(), false),
      LParenLoc(lparenloc), TInfo(tinfo), Init(init), FileScope(fileScope) {}

  /// \brief Construct an empty compound literal.
  explicit CompoundLiteralExpr(EmptyShell Empty)
    : Expr(CompoundLiteralExprClass, Empty) { }

  const Expr *getInitializer() const { return cast<Expr>(Init); }
  Expr *getInitializer() { return cast<Expr>(Init); }
  void setInitializer(Expr *E) { Init = E; }

  bool isFileScope() const { return FileScope; }
  void setFileScope(bool FS) { FileScope = FS; }

  SourceLocation getLParenLoc() const { return LParenLoc; }
  void setLParenLoc(SourceLocation L) { LParenLoc = L; }

  TypeSourceInfo *getTypeSourceInfo() const { return TInfo; }
  void setTypeSourceInfo(TypeSourceInfo* tinfo) { TInfo = tinfo; }

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
};

/// CastExpr - Base class for type casts, including both implicit
/// casts (ImplicitCastExpr) and explicit casts that have some
/// representation in the source code (ExplicitCastExpr's derived
/// classes).
class CastExpr : public Expr {
public:
  typedef clang::CastKind CastKind;

private:
  Stmt *Op;

  void CheckCastConsistency() const {
#ifndef NDEBUG
    switch (getCastKind()) {
    case CK_DerivedToBase:
    case CK_UncheckedDerivedToBase:
    case CK_DerivedToBaseMemberPointer:
    case CK_BaseToDerived:
    case CK_BaseToDerivedMemberPointer:
      assert(!path_empty() && "Cast kind should have a base path!");
      break;

    // These should not have an inheritance path.
    case CK_BitCast:
    case CK_Dynamic:
    case CK_ToUnion:
    case CK_ArrayToPointerDecay:
    case CK_FunctionToPointerDecay:
    case CK_NullToMemberPointer:
    case CK_NullToPointer:
    case CK_ConstructorConversion:
    case CK_IntegralToPointer:
    case CK_PointerToIntegral:
    case CK_ToVoid:
    case CK_VectorSplat:
    case CK_IntegralCast:
    case CK_IntegralToFloating:
    case CK_FloatingToIntegral:
    case CK_FloatingCast:
    case CK_AnyPointerToObjCPointerCast:
    case CK_AnyPointerToBlockPointerCast:
    case CK_ObjCObjectLValueCast:
    case CK_FloatingRealToComplex:
    case CK_FloatingComplexToReal:
    case CK_FloatingComplexCast:
    case CK_FloatingComplexToIntegralComplex:
    case CK_IntegralRealToComplex:
    case CK_IntegralComplexToReal:
    case CK_IntegralComplexCast:
    case CK_IntegralComplexToFloatingComplex:
      assert(!getType()->isBooleanType() && "unheralded conversion to bool");
      // fallthrough to check for null base path

    case CK_Dependent:
    case CK_NoOp:
    case CK_PointerToBoolean:
    case CK_IntegralToBoolean:
    case CK_FloatingToBoolean:
    case CK_MemberPointerToBoolean:
    case CK_FloatingComplexToBoolean:
    case CK_IntegralComplexToBoolean:
    case CK_LValueBitCast:            // -> bool&
    case CK_UserDefinedConversion:    // operator bool()
      assert(path_empty() && "Cast kind should not have a base path!");
      break;
    }
#endif
  }

  const CXXBaseSpecifier * const *path_buffer() const {
    return const_cast<CastExpr*>(this)->path_buffer();
  }
  CXXBaseSpecifier **path_buffer();

protected:
  CastExpr(StmtClass SC, QualType ty, const CastKind kind, Expr *op,
           unsigned BasePathSize) :
    Expr(SC, ty,
         // Cast expressions are type-dependent if the type is
         // dependent (C++ [temp.dep.expr]p3).
         ty->isDependentType(),
         // Cast expressions are value-dependent if the type is
         // dependent or if the subexpression is value-dependent.
         ty->isDependentType() || (op && op->isValueDependent())),
    Op(op) {
    assert(kind != CK_Invalid && "creating cast with invalid cast kind");
    CastExprBits.Kind = kind;
    CastExprBits.BasePathSize = BasePathSize;
    CheckCastConsistency();
  }

  /// \brief Construct an empty cast.
  CastExpr(StmtClass SC, EmptyShell Empty, unsigned BasePathSize)
    : Expr(SC, Empty) {
    CastExprBits.BasePathSize = BasePathSize;
  }

public:
  CastKind getCastKind() const { return (CastKind) CastExprBits.Kind; }
  void setCastKind(CastKind K) { CastExprBits.Kind = K; }
  const char *getCastKindName() const;

  Expr *getSubExpr() { return cast<Expr>(Op); }
  const Expr *getSubExpr() const { return cast<Expr>(Op); }
  void setSubExpr(Expr *E) { Op = E; }

  /// \brief Retrieve the cast subexpression as it was written in the source
  /// code, looking through any implicit casts or other intermediate nodes
  /// introduced by semantic analysis.
  Expr *getSubExprAsWritten();
  const Expr *getSubExprAsWritten() const {
    return const_cast<CastExpr *>(this)->getSubExprAsWritten();
  }

  typedef CXXBaseSpecifier **path_iterator;
  typedef const CXXBaseSpecifier * const *path_const_iterator;
  bool path_empty() const { return CastExprBits.BasePathSize == 0; }
  unsigned path_size() const { return CastExprBits.BasePathSize; }
  path_iterator path_begin() { return path_buffer(); }
  path_iterator path_end() { return path_buffer() + path_size(); }
  path_const_iterator path_begin() const { return path_buffer(); }
  path_const_iterator path_end() const { return path_buffer() + path_size(); }

  void setCastPath(const CXXCastPath &Path);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() >= firstCastExprConstant &&
           T->getStmtClass() <= lastCastExprConstant;
  }
  static bool classof(const CastExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// ImplicitCastExpr - Allows us to explicitly represent implicit type
/// conversions, which have no direct representation in the original
/// source code. For example: converting T[]->T*, void f()->void
/// (*f)(), float->double, short->int, etc.
///
/// In C, implicit casts always produce rvalues. However, in C++, an
/// implicit cast whose result is being bound to a reference will be
/// an lvalue or xvalue. For example:
///
/// @code
/// class Base { };
/// class Derived : public Base { };
/// Derived &&ref();
/// void f(Derived d) {
///   Base& b = d; // initializer is an ImplicitCastExpr
///                // to an lvalue of type Base
///   Base&& r = ref(); // initializer is an ImplicitCastExpr
///                     // to an xvalue of type Base
/// }
/// @endcode
class ImplicitCastExpr : public CastExpr {
private:
  ImplicitCastExpr(QualType ty, CastKind kind, Expr *op,
                   unsigned BasePathLength, ExprValueKind VK)
    : CastExpr(ImplicitCastExprClass, ty, kind, op, BasePathLength) {
    setValueKind(VK);
  }

  /// \brief Construct an empty implicit cast.
  explicit ImplicitCastExpr(EmptyShell Shell, unsigned PathSize)
    : CastExpr(ImplicitCastExprClass, Shell, PathSize) { }

public:
  enum OnStack_t { OnStack };
  ImplicitCastExpr(OnStack_t _, QualType ty, CastKind kind, Expr *op,
                   ExprValueKind VK)
    : CastExpr(ImplicitCastExprClass, ty, kind, op, 0) {
    setValueKind(VK);
  }

  static ImplicitCastExpr *Create(ASTContext &Context, QualType T,
                                  CastKind Kind, Expr *Operand,
                                  const CXXCastPath *BasePath,
                                  ExprValueKind Cat);

  static ImplicitCastExpr *CreateEmpty(ASTContext &Context, unsigned PathSize);

  virtual SourceRange getSourceRange() const {
    return getSubExpr()->getSourceRange();
  }

  using Expr::getValueKind;
  using Expr::setValueKind;

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ImplicitCastExprClass;
  }
  static bool classof(const ImplicitCastExpr *) { return true; }
};

/// ExplicitCastExpr - An explicit cast written in the source
/// code.
///
/// This class is effectively an abstract class, because it provides
/// the basic representation of an explicitly-written cast without
/// specifying which kind of cast (C cast, functional cast, static
/// cast, etc.) was written; specific derived classes represent the
/// particular style of cast and its location information.
///
/// Unlike implicit casts, explicit cast nodes have two different
/// types: the type that was written into the source code, and the
/// actual type of the expression as determined by semantic
/// analysis. These types may differ slightly. For example, in C++ one
/// can cast to a reference type, which indicates that the resulting
/// expression will be an lvalue or xvalue. The reference type, however,
/// will not be used as the type of the expression.
class ExplicitCastExpr : public CastExpr {
  /// TInfo - Source type info for the (written) type
  /// this expression is casting to.
  TypeSourceInfo *TInfo;

protected:
  ExplicitCastExpr(StmtClass SC, QualType exprTy, CastKind kind,
                   Expr *op, unsigned PathSize, TypeSourceInfo *writtenTy)
    : CastExpr(SC, exprTy, kind, op, PathSize), TInfo(writtenTy) {}

  /// \brief Construct an empty explicit cast.
  ExplicitCastExpr(StmtClass SC, EmptyShell Shell, unsigned PathSize)
    : CastExpr(SC, Shell, PathSize) { }

public:
  /// getTypeInfoAsWritten - Returns the type source info for the type
  /// that this expression is casting to.
  TypeSourceInfo *getTypeInfoAsWritten() const { return TInfo; }
  void setTypeInfoAsWritten(TypeSourceInfo *writtenTy) { TInfo = writtenTy; }

  /// getTypeAsWritten - Returns the type that this expression is
  /// casting to, as written in the source code.
  QualType getTypeAsWritten() const { return TInfo->getType(); }

  static bool classof(const Stmt *T) {
     return T->getStmtClass() >= firstExplicitCastExprConstant &&
            T->getStmtClass() <= lastExplicitCastExprConstant;
  }
  static bool classof(const ExplicitCastExpr *) { return true; }
};

/// CStyleCastExpr - An explicit cast in C (C99 6.5.4) or a C-style
/// cast in C++ (C++ [expr.cast]), which uses the syntax
/// (Type)expr. For example: @c (int)f.
class CStyleCastExpr : public ExplicitCastExpr {
  SourceLocation LPLoc; // the location of the left paren
  SourceLocation RPLoc; // the location of the right paren

  CStyleCastExpr(QualType exprTy, CastKind kind, Expr *op,
                 unsigned PathSize, TypeSourceInfo *writtenTy,
                 SourceLocation l, SourceLocation r)
    : ExplicitCastExpr(CStyleCastExprClass, exprTy, kind, op, PathSize,
                       writtenTy), LPLoc(l), RPLoc(r) {}

  /// \brief Construct an empty C-style explicit cast.
  explicit CStyleCastExpr(EmptyShell Shell, unsigned PathSize)
    : ExplicitCastExpr(CStyleCastExprClass, Shell, PathSize) { }

public:
  static CStyleCastExpr *Create(ASTContext &Context, QualType T, CastKind K,
                                Expr *Op, const CXXCastPath *BasePath,
                                TypeSourceInfo *WrittenTy, SourceLocation L,
                                SourceLocation R);

  static CStyleCastExpr *CreateEmpty(ASTContext &Context, unsigned PathSize);

  SourceLocation getLParenLoc() const { return LPLoc; }
  void setLParenLoc(SourceLocation L) { LPLoc = L; }

  SourceLocation getRParenLoc() const { return RPLoc; }
  void setRParenLoc(SourceLocation L) { RPLoc = L; }

  virtual SourceRange getSourceRange() const {
    return SourceRange(LPLoc, getSubExpr()->getSourceRange().getEnd());
  }
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CStyleCastExprClass;
  }
  static bool classof(const CStyleCastExpr *) { return true; }
};

/// \brief A builtin binary operation expression such as "x + y" or "x <= y".
///
/// This expression node kind describes a builtin binary operation,
/// such as "x + y" for integer values "x" and "y". The operands will
/// already have been converted to appropriate types (e.g., by
/// performing promotions or conversions).
///
/// In C++, where operators may be overloaded, a different kind of
/// expression node (CXXOperatorCallExpr) is used to express the
/// invocation of an overloaded operator with operator syntax. Within
/// a C++ template, whether BinaryOperator or CXXOperatorCallExpr is
/// used to store an expression "x + y" depends on the subexpressions
/// for x and y. If neither x or y is type-dependent, and the "+"
/// operator resolves to a built-in operation, BinaryOperator will be
/// used to express the computation (x and y may still be
/// value-dependent). If either x or y is type-dependent, or if the
/// "+" resolves to an overloaded operator, CXXOperatorCallExpr will
/// be used to express the computation.
class BinaryOperator : public Expr {
public:
  typedef BinaryOperatorKind Opcode;

private:
  unsigned Opc : 6;
  SourceLocation OpLoc;

  enum { LHS, RHS, END_EXPR };
  Stmt* SubExprs[END_EXPR];
public:

  BinaryOperator(Expr *lhs, Expr *rhs, Opcode opc, QualType ResTy,
                 SourceLocation opLoc)
    : Expr(BinaryOperatorClass, ResTy,
           lhs->isTypeDependent() || rhs->isTypeDependent(),
           lhs->isValueDependent() || rhs->isValueDependent()),
      Opc(opc), OpLoc(opLoc) {
    SubExprs[LHS] = lhs;
    SubExprs[RHS] = rhs;
    assert(!isCompoundAssignmentOp() &&
           "Use ArithAssignBinaryOperator for compound assignments");
  }

  /// \brief Construct an empty binary operator.
  explicit BinaryOperator(EmptyShell Empty)
    : Expr(BinaryOperatorClass, Empty), Opc(BO_Comma) { }

  SourceLocation getOperatorLoc() const { return OpLoc; }
  void setOperatorLoc(SourceLocation L) { OpLoc = L; }

  Opcode getOpcode() const { return static_cast<Opcode>(Opc); }
  void setOpcode(Opcode O) { Opc = O; }

  Expr *getLHS() const { return cast<Expr>(SubExprs[LHS]); }
  void setLHS(Expr *E) { SubExprs[LHS] = E; }
  Expr *getRHS() const { return cast<Expr>(SubExprs[RHS]); }
  void setRHS(Expr *E) { SubExprs[RHS] = E; }

  virtual SourceRange getSourceRange() const {
    return SourceRange(getLHS()->getLocStart(), getRHS()->getLocEnd());
  }

  /// getOpcodeStr - Turn an Opcode enum value into the punctuation char it
  /// corresponds to, e.g. "<<=".
  static const char *getOpcodeStr(Opcode Op);

  const char *getOpcodeStr() const { return getOpcodeStr(getOpcode()); }

  /// \brief Retrieve the binary opcode that corresponds to the given
  /// overloaded operator.
  static Opcode getOverloadedOpcode(OverloadedOperatorKind OO);

  /// \brief Retrieve the overloaded operator kind that corresponds to
  /// the given binary opcode.
  static OverloadedOperatorKind getOverloadedOperator(Opcode Opc);

  /// predicates to categorize the respective opcodes.
  bool isPtrMemOp() const { return Opc == BO_PtrMemD || Opc == BO_PtrMemI; }
  bool isMultiplicativeOp() const { return Opc >= BO_Mul && Opc <= BO_Rem; }
  static bool isAdditiveOp(Opcode Opc) { return Opc == BO_Add || Opc==BO_Sub; }
  bool isAdditiveOp() const { return isAdditiveOp(getOpcode()); }
  static bool isShiftOp(Opcode Opc) { return Opc == BO_Shl || Opc == BO_Shr; }
  bool isShiftOp() const { return isShiftOp(getOpcode()); }

  static bool isBitwiseOp(Opcode Opc) { return Opc >= BO_And && Opc <= BO_Or; }
  bool isBitwiseOp() const { return isBitwiseOp(getOpcode()); }

  static bool isRelationalOp(Opcode Opc) { return Opc >= BO_LT && Opc<=BO_GE; }
  bool isRelationalOp() const { return isRelationalOp(getOpcode()); }

  static bool isEqualityOp(Opcode Opc) { return Opc == BO_EQ || Opc == BO_NE; }
  bool isEqualityOp() const { return isEqualityOp(getOpcode()); }

  static bool isComparisonOp(Opcode Opc) { return Opc >= BO_LT && Opc<=BO_NE; }
  bool isComparisonOp() const { return isComparisonOp(getOpcode()); }

  static bool isLogicalOp(Opcode Opc) { return Opc == BO_LAnd || Opc==BO_LOr; }
  bool isLogicalOp() const { return isLogicalOp(getOpcode()); }

  bool isAssignmentOp() const { return Opc >= BO_Assign && Opc <= BO_OrAssign; }
  bool isCompoundAssignmentOp() const {
    return Opc > BO_Assign && Opc <= BO_OrAssign;
  }
  bool isShiftAssignOp() const {
    return Opc == BO_ShlAssign || Opc == BO_ShrAssign;
  }

  static bool classof(const Stmt *S) {
    return S->getStmtClass() >= firstBinaryOperatorConstant &&
           S->getStmtClass() <= lastBinaryOperatorConstant;
  }
  static bool classof(const BinaryOperator *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();

protected:
  BinaryOperator(Expr *lhs, Expr *rhs, Opcode opc, QualType ResTy,
                 SourceLocation opLoc, bool dead)
    : Expr(CompoundAssignOperatorClass, ResTy,
           lhs->isTypeDependent() || rhs->isTypeDependent(),
           lhs->isValueDependent() || rhs->isValueDependent()),
      Opc(opc), OpLoc(opLoc) {
    SubExprs[LHS] = lhs;
    SubExprs[RHS] = rhs;
  }

  BinaryOperator(StmtClass SC, EmptyShell Empty)
    : Expr(SC, Empty), Opc(BO_MulAssign) { }
};

/// CompoundAssignOperator - For compound assignments (e.g. +=), we keep
/// track of the type the operation is performed in.  Due to the semantics of
/// these operators, the operands are promoted, the aritmetic performed, an
/// implicit conversion back to the result type done, then the assignment takes
/// place.  This captures the intermediate type which the computation is done
/// in.
class CompoundAssignOperator : public BinaryOperator {
  QualType ComputationLHSType;
  QualType ComputationResultType;
public:
  CompoundAssignOperator(Expr *lhs, Expr *rhs, Opcode opc,
                         QualType ResType, QualType CompLHSType,
                         QualType CompResultType,
                         SourceLocation OpLoc)
    : BinaryOperator(lhs, rhs, opc, ResType, OpLoc, true),
      ComputationLHSType(CompLHSType),
      ComputationResultType(CompResultType) {
    assert(isCompoundAssignmentOp() &&
           "Only should be used for compound assignments");
  }

  /// \brief Build an empty compound assignment operator expression.
  explicit CompoundAssignOperator(EmptyShell Empty)
    : BinaryOperator(CompoundAssignOperatorClass, Empty) { }

  // The two computation types are the type the LHS is converted
  // to for the computation and the type of the result; the two are
  // distinct in a few cases (specifically, int+=ptr and ptr-=ptr).
  QualType getComputationLHSType() const { return ComputationLHSType; }
  void setComputationLHSType(QualType T) { ComputationLHSType = T; }

  QualType getComputationResultType() const { return ComputationResultType; }
  void setComputationResultType(QualType T) { ComputationResultType = T; }

  static bool classof(const CompoundAssignOperator *) { return true; }
  static bool classof(const Stmt *S) {
    return S->getStmtClass() == CompoundAssignOperatorClass;
  }
};

/// ConditionalOperator - The ?: operator.  Note that LHS may be null when the
/// GNU "missing LHS" extension is in use.
///
class ConditionalOperator : public Expr {
  enum { COND, LHS, RHS, END_EXPR };
  Stmt* SubExprs[END_EXPR]; // Left/Middle/Right hand sides.
  Stmt* Save;
  SourceLocation QuestionLoc, ColonLoc;
public:
  ConditionalOperator(Expr *cond, SourceLocation QLoc, Expr *lhs,
                      SourceLocation CLoc, Expr *rhs, Expr *save, QualType t)
    : Expr(ConditionalOperatorClass, t,
           // FIXME: the type of the conditional operator doesn't
           // depend on the type of the conditional, but the standard
           // seems to imply that it could. File a bug!
           ((lhs && lhs->isTypeDependent()) || (rhs && rhs->isTypeDependent())),
           (cond->isValueDependent() ||
            (lhs && lhs->isValueDependent()) ||
            (rhs && rhs->isValueDependent()))),
      QuestionLoc(QLoc),
      ColonLoc(CLoc) {
    SubExprs[COND] = cond;
    SubExprs[LHS] = lhs;
    SubExprs[RHS] = rhs;
    Save = save;
  }

  /// \brief Build an empty conditional operator.
  explicit ConditionalOperator(EmptyShell Empty)
    : Expr(ConditionalOperatorClass, Empty) { }

  // getCond - Return the expression representing the condition for
  //  the ?: operator.
  Expr *getCond() const { return cast<Expr>(SubExprs[COND]); }
  void setCond(Expr *E) { SubExprs[COND] = E; }

  // getTrueExpr - Return the subexpression representing the value of the ?:
  //  expression if the condition evaluates to true.  
  Expr *getTrueExpr() const {
    return cast<Expr>(SubExprs[LHS]);
  }

  // getFalseExpr - Return the subexpression representing the value of the ?:
  // expression if the condition evaluates to false. This is the same as getRHS.
  Expr *getFalseExpr() const { return cast<Expr>(SubExprs[RHS]); }
  
  // getSaveExpr - In most cases this value will be null. Except a GCC extension 
  // allows the left subexpression to be omitted, and instead of that condition 
  // be returned. e.g: x ?: y is shorthand for x ? x : y, except that the 
  // expression "x" is only evaluated once. Under this senario, this function
  // returns the original, non-converted condition expression for the ?:operator
  Expr *getSaveExpr() const { return Save? cast<Expr>(Save) : (Expr*)0; }

  Expr *getLHS() const { return Save ? 0 : cast<Expr>(SubExprs[LHS]); }
  void setLHS(Expr *E) { SubExprs[LHS] = E; }

  Expr *getRHS() const { return cast<Expr>(SubExprs[RHS]); }
  void setRHS(Expr *E) { SubExprs[RHS] = E; }

  Expr *getSAVE() const { return Save? cast<Expr>(Save) : (Expr*)0; }
  void setSAVE(Expr *E) { Save = E; }
  
  SourceLocation getQuestionLoc() const { return QuestionLoc; }
  void setQuestionLoc(SourceLocation L) { QuestionLoc = L; }

  SourceLocation getColonLoc() const { return ColonLoc; }
  void setColonLoc(SourceLocation L) { ColonLoc = L; }

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
};

/// AddrLabelExpr - The GNU address of label extension, representing &&label.
class AddrLabelExpr : public Expr {
  SourceLocation AmpAmpLoc, LabelLoc;
  LabelStmt *Label;
public:
  AddrLabelExpr(SourceLocation AALoc, SourceLocation LLoc, LabelStmt *L,
                QualType t)
    : Expr(AddrLabelExprClass, t, false, false),
      AmpAmpLoc(AALoc), LabelLoc(LLoc), Label(L) {}

  /// \brief Build an empty address of a label expression.
  explicit AddrLabelExpr(EmptyShell Empty)
    : Expr(AddrLabelExprClass, Empty) { }

  SourceLocation getAmpAmpLoc() const { return AmpAmpLoc; }
  void setAmpAmpLoc(SourceLocation L) { AmpAmpLoc = L; }
  SourceLocation getLabelLoc() const { return LabelLoc; }
  void setLabelLoc(SourceLocation L) { LabelLoc = L; }

  virtual SourceRange getSourceRange() const {
    return SourceRange(AmpAmpLoc, LabelLoc);
  }

  LabelStmt *getLabel() const { return Label; }
  void setLabel(LabelStmt *S) { Label = S; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == AddrLabelExprClass;
  }
  static bool classof(const AddrLabelExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// StmtExpr - This is the GNU Statement Expression extension: ({int X=4; X;}).
/// The StmtExpr contains a single CompoundStmt node, which it evaluates and
/// takes the value of the last subexpression.
class StmtExpr : public Expr {
  Stmt *SubStmt;
  SourceLocation LParenLoc, RParenLoc;
public:
  // FIXME: Does type-dependence need to be computed differently?
  StmtExpr(CompoundStmt *substmt, QualType T,
           SourceLocation lp, SourceLocation rp) :
    Expr(StmtExprClass, T, T->isDependentType(), false),
    SubStmt(substmt), LParenLoc(lp), RParenLoc(rp) { }

  /// \brief Build an empty statement expression.
  explicit StmtExpr(EmptyShell Empty) : Expr(StmtExprClass, Empty) { }

  CompoundStmt *getSubStmt() { return cast<CompoundStmt>(SubStmt); }
  const CompoundStmt *getSubStmt() const { return cast<CompoundStmt>(SubStmt); }
  void setSubStmt(CompoundStmt *S) { SubStmt = S; }

  virtual SourceRange getSourceRange() const {
    return SourceRange(LParenLoc, RParenLoc);
  }

  SourceLocation getLParenLoc() const { return LParenLoc; }
  void setLParenLoc(SourceLocation L) { LParenLoc = L; }
  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation L) { RParenLoc = L; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == StmtExprClass;
  }
  static bool classof(const StmtExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// TypesCompatibleExpr - GNU builtin-in function __builtin_types_compatible_p.
/// This AST node represents a function that returns 1 if two *types* (not
/// expressions) are compatible. The result of this built-in function can be
/// used in integer constant expressions.
class TypesCompatibleExpr : public Expr {
  TypeSourceInfo *TInfo1;
  TypeSourceInfo *TInfo2;
  SourceLocation BuiltinLoc, RParenLoc;
public:
  TypesCompatibleExpr(QualType ReturnType, SourceLocation BLoc,
                      TypeSourceInfo *tinfo1, TypeSourceInfo *tinfo2,
                      SourceLocation RP) :
    Expr(TypesCompatibleExprClass, ReturnType, false, false),
    TInfo1(tinfo1), TInfo2(tinfo2), BuiltinLoc(BLoc), RParenLoc(RP) {}

  /// \brief Build an empty __builtin_type_compatible_p expression.
  explicit TypesCompatibleExpr(EmptyShell Empty)
    : Expr(TypesCompatibleExprClass, Empty) { }

  TypeSourceInfo *getArgTInfo1() const { return TInfo1; }
  void setArgTInfo1(TypeSourceInfo *TInfo) { TInfo1 = TInfo; }
  TypeSourceInfo *getArgTInfo2() const { return TInfo2; }
  void setArgTInfo2(TypeSourceInfo *TInfo) { TInfo2 = TInfo; }

  QualType getArgType1() const { return TInfo1->getType(); }
  QualType getArgType2() const { return TInfo2->getType(); }

  SourceLocation getBuiltinLoc() const { return BuiltinLoc; }
  void setBuiltinLoc(SourceLocation L) { BuiltinLoc = L; }

  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation L) { RParenLoc = L; }

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
  // FIXME: Can a shufflevector be value-dependent?  Does type-dependence need
  // to be computed differently?
  ShuffleVectorExpr(ASTContext &C, Expr **args, unsigned nexpr,
                    QualType Type, SourceLocation BLoc,
                    SourceLocation RP) :
    Expr(ShuffleVectorExprClass, Type, Type->isDependentType(), false),
    BuiltinLoc(BLoc), RParenLoc(RP), NumExprs(nexpr) {

    SubExprs = new (C) Stmt*[nexpr];
    for (unsigned i = 0; i < nexpr; i++)
      SubExprs[i] = args[i];
  }

  /// \brief Build an empty vector-shuffle expression.
  explicit ShuffleVectorExpr(EmptyShell Empty)
    : Expr(ShuffleVectorExprClass, Empty), SubExprs(0) { }

  SourceLocation getBuiltinLoc() const { return BuiltinLoc; }
  void setBuiltinLoc(SourceLocation L) { BuiltinLoc = L; }

  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation L) { RParenLoc = L; }

  virtual SourceRange getSourceRange() const {
    return SourceRange(BuiltinLoc, RParenLoc);
  }
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ShuffleVectorExprClass;
  }
  static bool classof(const ShuffleVectorExpr *) { return true; }

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

  void setExprs(ASTContext &C, Expr ** Exprs, unsigned NumExprs);

  unsigned getShuffleMaskIdx(ASTContext &Ctx, unsigned N) {
    assert((N < NumExprs - 2) && "Shuffle idx out of range!");
    return getExpr(N+2)->EvaluateAsInt(Ctx).getZExtValue();
  }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// ChooseExpr - GNU builtin-in function __builtin_choose_expr.
/// This AST node is similar to the conditional operator (?:) in C, with
/// the following exceptions:
/// - the test expression must be a integer constant expression.
/// - the expression returned acts like the chosen subexpression in every
///   visible way: the type is the same as that of the chosen subexpression,
///   and all predicates (whether it's an l-value, whether it's an integer
///   constant expression, etc.) return the same result as for the chosen
///   sub-expression.
class ChooseExpr : public Expr {
  enum { COND, LHS, RHS, END_EXPR };
  Stmt* SubExprs[END_EXPR]; // Left/Middle/Right hand sides.
  SourceLocation BuiltinLoc, RParenLoc;
public:
  ChooseExpr(SourceLocation BLoc, Expr *cond, Expr *lhs, Expr *rhs, QualType t,
             SourceLocation RP, bool TypeDependent, bool ValueDependent)
    : Expr(ChooseExprClass, t, TypeDependent, ValueDependent),
      BuiltinLoc(BLoc), RParenLoc(RP) {
      SubExprs[COND] = cond;
      SubExprs[LHS] = lhs;
      SubExprs[RHS] = rhs;
    }

  /// \brief Build an empty __builtin_choose_expr.
  explicit ChooseExpr(EmptyShell Empty) : Expr(ChooseExprClass, Empty) { }

  /// isConditionTrue - Return whether the condition is true (i.e. not
  /// equal to zero).
  bool isConditionTrue(ASTContext &C) const;

  /// getChosenSubExpr - Return the subexpression chosen according to the
  /// condition.
  Expr *getChosenSubExpr(ASTContext &C) const {
    return isConditionTrue(C) ? getLHS() : getRHS();
  }

  Expr *getCond() const { return cast<Expr>(SubExprs[COND]); }
  void setCond(Expr *E) { SubExprs[COND] = E; }
  Expr *getLHS() const { return cast<Expr>(SubExprs[LHS]); }
  void setLHS(Expr *E) { SubExprs[LHS] = E; }
  Expr *getRHS() const { return cast<Expr>(SubExprs[RHS]); }
  void setRHS(Expr *E) { SubExprs[RHS] = E; }

  SourceLocation getBuiltinLoc() const { return BuiltinLoc; }
  void setBuiltinLoc(SourceLocation L) { BuiltinLoc = L; }

  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation L) { RParenLoc = L; }

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

/// GNUNullExpr - Implements the GNU __null extension, which is a name
/// for a null pointer constant that has integral type (e.g., int or
/// long) and is the same size and alignment as a pointer. The __null
/// extension is typically only used by system headers, which define
/// NULL as __null in C++ rather than using 0 (which is an integer
/// that may not match the size of a pointer).
class GNUNullExpr : public Expr {
  /// TokenLoc - The location of the __null keyword.
  SourceLocation TokenLoc;

public:
  GNUNullExpr(QualType Ty, SourceLocation Loc)
    : Expr(GNUNullExprClass, Ty, false, false), TokenLoc(Loc) { }

  /// \brief Build an empty GNU __null expression.
  explicit GNUNullExpr(EmptyShell Empty) : Expr(GNUNullExprClass, Empty) { }

  /// getTokenLocation - The location of the __null token.
  SourceLocation getTokenLocation() const { return TokenLoc; }
  void setTokenLocation(SourceLocation L) { TokenLoc = L; }

  virtual SourceRange getSourceRange() const {
    return SourceRange(TokenLoc);
  }
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == GNUNullExprClass;
  }
  static bool classof(const GNUNullExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// VAArgExpr, used for the builtin function __builtin_va_arg.
class VAArgExpr : public Expr {
  Stmt *Val;
  TypeSourceInfo *TInfo;
  SourceLocation BuiltinLoc, RParenLoc;
public:
  VAArgExpr(SourceLocation BLoc, Expr* e, TypeSourceInfo *TInfo,
            SourceLocation RPLoc, QualType t)
    : Expr(VAArgExprClass, t, t->isDependentType(), false),
      Val(e), TInfo(TInfo),
      BuiltinLoc(BLoc),
      RParenLoc(RPLoc) { }

  /// \brief Create an empty __builtin_va_arg expression.
  explicit VAArgExpr(EmptyShell Empty) : Expr(VAArgExprClass, Empty) { }

  const Expr *getSubExpr() const { return cast<Expr>(Val); }
  Expr *getSubExpr() { return cast<Expr>(Val); }
  void setSubExpr(Expr *E) { Val = E; }

  TypeSourceInfo *getWrittenTypeInfo() const { return TInfo; }
  void setWrittenTypeInfo(TypeSourceInfo *TI) { TInfo = TI; }

  SourceLocation getBuiltinLoc() const { return BuiltinLoc; }
  void setBuiltinLoc(SourceLocation L) { BuiltinLoc = L; }

  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation L) { RParenLoc = L; }

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

/// @brief Describes an C or C++ initializer list.
///
/// InitListExpr describes an initializer list, which can be used to
/// initialize objects of different types, including
/// struct/class/union types, arrays, and vectors. For example:
///
/// @code
/// struct foo x = { 1, { 2, 3 } };
/// @endcode
///
/// Prior to semantic analysis, an initializer list will represent the
/// initializer list as written by the user, but will have the
/// placeholder type "void". This initializer list is called the
/// syntactic form of the initializer, and may contain C99 designated
/// initializers (represented as DesignatedInitExprs), initializations
/// of subobject members without explicit braces, and so on. Clients
/// interested in the original syntax of the initializer list should
/// use the syntactic form of the initializer list.
///
/// After semantic analysis, the initializer list will represent the
/// semantic form of the initializer, where the initializations of all
/// subobjects are made explicit with nested InitListExpr nodes and
/// C99 designators have been eliminated by placing the designated
/// initializations into the subobject they initialize. Additionally,
/// any "holes" in the initialization, where no initializer has been
/// specified for a particular subobject, will be replaced with
/// implicitly-generated ImplicitValueInitExpr expressions that
/// value-initialize the subobjects. Note, however, that the
/// initializer lists may still have fewer initializers than there are
/// elements to initialize within the object.
///
/// Given the semantic form of the initializer list, one can retrieve
/// the original syntactic form of that initializer list (if it
/// exists) using getSyntacticForm(). Since many initializer lists
/// have the same syntactic and semantic forms, getSyntacticForm() may
/// return NULL, indicating that the current initializer list also
/// serves as its syntactic form.
class InitListExpr : public Expr {
  // FIXME: Eliminate this vector in favor of ASTContext allocation
  typedef ASTVector<Stmt *> InitExprsTy;
  InitExprsTy InitExprs;
  SourceLocation LBraceLoc, RBraceLoc;

  /// Contains the initializer list that describes the syntactic form
  /// written in the source code.
  InitListExpr *SyntacticForm;

  /// If this initializer list initializes a union, specifies which
  /// field within the union will be initialized.
  FieldDecl *UnionFieldInit;

  /// Whether this initializer list originally had a GNU array-range
  /// designator in it. This is a temporary marker used by CodeGen.
  bool HadArrayRangeDesignator;

public:
  InitListExpr(ASTContext &C, SourceLocation lbraceloc,
               Expr **initexprs, unsigned numinits,
               SourceLocation rbraceloc);

  /// \brief Build an empty initializer list.
  explicit InitListExpr(ASTContext &C, EmptyShell Empty)
    : Expr(InitListExprClass, Empty), InitExprs(C) { }

  unsigned getNumInits() const { return InitExprs.size(); }

  const Expr *getInit(unsigned Init) const {
    assert(Init < getNumInits() && "Initializer access out of range!");
    return cast_or_null<Expr>(InitExprs[Init]);
  }

  Expr *getInit(unsigned Init) {
    assert(Init < getNumInits() && "Initializer access out of range!");
    return cast_or_null<Expr>(InitExprs[Init]);
  }

  void setInit(unsigned Init, Expr *expr) {
    assert(Init < getNumInits() && "Initializer access out of range!");
    InitExprs[Init] = expr;
  }

  /// \brief Reserve space for some number of initializers.
  void reserveInits(ASTContext &C, unsigned NumInits);

  /// @brief Specify the number of initializers
  ///
  /// If there are more than @p NumInits initializers, the remaining
  /// initializers will be destroyed. If there are fewer than @p
  /// NumInits initializers, NULL expressions will be added for the
  /// unknown initializers.
  void resizeInits(ASTContext &Context, unsigned NumInits);

  /// @brief Updates the initializer at index @p Init with the new
  /// expression @p expr, and returns the old expression at that
  /// location.
  ///
  /// When @p Init is out of range for this initializer list, the
  /// initializer list will be extended with NULL expressions to
  /// accomodate the new entry.
  Expr *updateInit(ASTContext &C, unsigned Init, Expr *expr);

  /// \brief If this initializes a union, specifies which field in the
  /// union to initialize.
  ///
  /// Typically, this field is the first named field within the
  /// union. However, a designated initializer can specify the
  /// initialization of a different field within the union.
  FieldDecl *getInitializedFieldInUnion() { return UnionFieldInit; }
  void setInitializedFieldInUnion(FieldDecl *FD) { UnionFieldInit = FD; }

  // Explicit InitListExpr's originate from source code (and have valid source
  // locations). Implicit InitListExpr's are created by the semantic analyzer.
  bool isExplicit() {
    return LBraceLoc.isValid() && RBraceLoc.isValid();
  }

  SourceLocation getLBraceLoc() const { return LBraceLoc; }
  void setLBraceLoc(SourceLocation Loc) { LBraceLoc = Loc; }
  SourceLocation getRBraceLoc() const { return RBraceLoc; }
  void setRBraceLoc(SourceLocation Loc) { RBraceLoc = Loc; }

  /// @brief Retrieve the initializer list that describes the
  /// syntactic form of the initializer.
  ///
  ///
  InitListExpr *getSyntacticForm() const { return SyntacticForm; }
  void setSyntacticForm(InitListExpr *Init) { SyntacticForm = Init; }

  bool hadArrayRangeDesignator() const { return HadArrayRangeDesignator; }
  void sawArrayRangeDesignator(bool ARD = true) {
    HadArrayRangeDesignator = ARD;
  }

  virtual SourceRange getSourceRange() const;

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == InitListExprClass;
  }
  static bool classof(const InitListExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();

  typedef InitExprsTy::iterator iterator;
  typedef InitExprsTy::const_iterator const_iterator;
  typedef InitExprsTy::reverse_iterator reverse_iterator;
  typedef InitExprsTy::const_reverse_iterator const_reverse_iterator;

  iterator begin() { return InitExprs.begin(); }
  const_iterator begin() const { return InitExprs.begin(); }
  iterator end() { return InitExprs.end(); }
  const_iterator end() const { return InitExprs.end(); }
  reverse_iterator rbegin() { return InitExprs.rbegin(); }
  const_reverse_iterator rbegin() const { return InitExprs.rbegin(); }
  reverse_iterator rend() { return InitExprs.rend(); }
  const_reverse_iterator rend() const { return InitExprs.rend(); }
};

/// @brief Represents a C99 designated initializer expression.
///
/// A designated initializer expression (C99 6.7.8) contains one or
/// more designators (which can be field designators, array
/// designators, or GNU array-range designators) followed by an
/// expression that initializes the field or element(s) that the
/// designators refer to. For example, given:
///
/// @code
/// struct point {
///   double x;
///   double y;
/// };
/// struct point ptarray[10] = { [2].y = 1.0, [2].x = 2.0, [0].x = 1.0 };
/// @endcode
///
/// The InitListExpr contains three DesignatedInitExprs, the first of
/// which covers @c [2].y=1.0. This DesignatedInitExpr will have two
/// designators, one array designator for @c [2] followed by one field
/// designator for @c .y. The initalization expression will be 1.0.
class DesignatedInitExpr : public Expr {
public:
  /// \brief Forward declaration of the Designator class.
  class Designator;

private:
  /// The location of the '=' or ':' prior to the actual initializer
  /// expression.
  SourceLocation EqualOrColonLoc;

  /// Whether this designated initializer used the GNU deprecated
  /// syntax rather than the C99 '=' syntax.
  bool GNUSyntax : 1;

  /// The number of designators in this initializer expression.
  unsigned NumDesignators : 15;

  /// \brief The designators in this designated initialization
  /// expression.
  Designator *Designators;

  /// The number of subexpressions of this initializer expression,
  /// which contains both the initializer and any additional
  /// expressions used by array and array-range designators.
  unsigned NumSubExprs : 16;


  DesignatedInitExpr(ASTContext &C, QualType Ty, unsigned NumDesignators,
                     const Designator *Designators,
                     SourceLocation EqualOrColonLoc, bool GNUSyntax,
                     Expr **IndexExprs, unsigned NumIndexExprs,
                     Expr *Init);

  explicit DesignatedInitExpr(unsigned NumSubExprs)
    : Expr(DesignatedInitExprClass, EmptyShell()),
      NumDesignators(0), Designators(0), NumSubExprs(NumSubExprs) { }

public:
  /// A field designator, e.g., ".x".
  struct FieldDesignator {
    /// Refers to the field that is being initialized. The low bit
    /// of this field determines whether this is actually a pointer
    /// to an IdentifierInfo (if 1) or a FieldDecl (if 0). When
    /// initially constructed, a field designator will store an
    /// IdentifierInfo*. After semantic analysis has resolved that
    /// name, the field designator will instead store a FieldDecl*.
    uintptr_t NameOrField;

    /// The location of the '.' in the designated initializer.
    unsigned DotLoc;

    /// The location of the field name in the designated initializer.
    unsigned FieldLoc;
  };

  /// An array or GNU array-range designator, e.g., "[9]" or "[10..15]".
  struct ArrayOrRangeDesignator {
    /// Location of the first index expression within the designated
    /// initializer expression's list of subexpressions.
    unsigned Index;
    /// The location of the '[' starting the array range designator.
    unsigned LBracketLoc;
    /// The location of the ellipsis separating the start and end
    /// indices. Only valid for GNU array-range designators.
    unsigned EllipsisLoc;
    /// The location of the ']' terminating the array range designator.
    unsigned RBracketLoc;
  };

  /// @brief Represents a single C99 designator.
  ///
  /// @todo This class is infuriatingly similar to clang::Designator,
  /// but minor differences (storing indices vs. storing pointers)
  /// keep us from reusing it. Try harder, later, to rectify these
  /// differences.
  class Designator {
    /// @brief The kind of designator this describes.
    enum {
      FieldDesignator,
      ArrayDesignator,
      ArrayRangeDesignator
    } Kind;

    union {
      /// A field designator, e.g., ".x".
      struct FieldDesignator Field;
      /// An array or GNU array-range designator, e.g., "[9]" or "[10..15]".
      struct ArrayOrRangeDesignator ArrayOrRange;
    };
    friend class DesignatedInitExpr;

  public:
    Designator() {}

    /// @brief Initializes a field designator.
    Designator(const IdentifierInfo *FieldName, SourceLocation DotLoc,
               SourceLocation FieldLoc)
      : Kind(FieldDesignator) {
      Field.NameOrField = reinterpret_cast<uintptr_t>(FieldName) | 0x01;
      Field.DotLoc = DotLoc.getRawEncoding();
      Field.FieldLoc = FieldLoc.getRawEncoding();
    }

    /// @brief Initializes an array designator.
    Designator(unsigned Index, SourceLocation LBracketLoc,
               SourceLocation RBracketLoc)
      : Kind(ArrayDesignator) {
      ArrayOrRange.Index = Index;
      ArrayOrRange.LBracketLoc = LBracketLoc.getRawEncoding();
      ArrayOrRange.EllipsisLoc = SourceLocation().getRawEncoding();
      ArrayOrRange.RBracketLoc = RBracketLoc.getRawEncoding();
    }

    /// @brief Initializes a GNU array-range designator.
    Designator(unsigned Index, SourceLocation LBracketLoc,
               SourceLocation EllipsisLoc, SourceLocation RBracketLoc)
      : Kind(ArrayRangeDesignator) {
      ArrayOrRange.Index = Index;
      ArrayOrRange.LBracketLoc = LBracketLoc.getRawEncoding();
      ArrayOrRange.EllipsisLoc = EllipsisLoc.getRawEncoding();
      ArrayOrRange.RBracketLoc = RBracketLoc.getRawEncoding();
    }

    bool isFieldDesignator() const { return Kind == FieldDesignator; }
    bool isArrayDesignator() const { return Kind == ArrayDesignator; }
    bool isArrayRangeDesignator() const { return Kind == ArrayRangeDesignator; }

    IdentifierInfo * getFieldName();

    FieldDecl *getField() {
      assert(Kind == FieldDesignator && "Only valid on a field designator");
      if (Field.NameOrField & 0x01)
        return 0;
      else
        return reinterpret_cast<FieldDecl *>(Field.NameOrField);
    }

    void setField(FieldDecl *FD) {
      assert(Kind == FieldDesignator && "Only valid on a field designator");
      Field.NameOrField = reinterpret_cast<uintptr_t>(FD);
    }

    SourceLocation getDotLoc() const {
      assert(Kind == FieldDesignator && "Only valid on a field designator");
      return SourceLocation::getFromRawEncoding(Field.DotLoc);
    }

    SourceLocation getFieldLoc() const {
      assert(Kind == FieldDesignator && "Only valid on a field designator");
      return SourceLocation::getFromRawEncoding(Field.FieldLoc);
    }

    SourceLocation getLBracketLoc() const {
      assert((Kind == ArrayDesignator || Kind == ArrayRangeDesignator) &&
             "Only valid on an array or array-range designator");
      return SourceLocation::getFromRawEncoding(ArrayOrRange.LBracketLoc);
    }

    SourceLocation getRBracketLoc() const {
      assert((Kind == ArrayDesignator || Kind == ArrayRangeDesignator) &&
             "Only valid on an array or array-range designator");
      return SourceLocation::getFromRawEncoding(ArrayOrRange.RBracketLoc);
    }

    SourceLocation getEllipsisLoc() const {
      assert(Kind == ArrayRangeDesignator &&
             "Only valid on an array-range designator");
      return SourceLocation::getFromRawEncoding(ArrayOrRange.EllipsisLoc);
    }

    unsigned getFirstExprIndex() const {
      assert((Kind == ArrayDesignator || Kind == ArrayRangeDesignator) &&
             "Only valid on an array or array-range designator");
      return ArrayOrRange.Index;
    }

    SourceLocation getStartLocation() const {
      if (Kind == FieldDesignator)
        return getDotLoc().isInvalid()? getFieldLoc() : getDotLoc();
      else
        return getLBracketLoc();
    }
  };

  static DesignatedInitExpr *Create(ASTContext &C, Designator *Designators,
                                    unsigned NumDesignators,
                                    Expr **IndexExprs, unsigned NumIndexExprs,
                                    SourceLocation EqualOrColonLoc,
                                    bool GNUSyntax, Expr *Init);

  static DesignatedInitExpr *CreateEmpty(ASTContext &C, unsigned NumIndexExprs);

  /// @brief Returns the number of designators in this initializer.
  unsigned size() const { return NumDesignators; }

  // Iterator access to the designators.
  typedef Designator* designators_iterator;
  designators_iterator designators_begin() { return Designators; }
  designators_iterator designators_end() {
    return Designators + NumDesignators;
  }

  Designator *getDesignator(unsigned Idx) { return &designators_begin()[Idx]; }

  void setDesignators(ASTContext &C, const Designator *Desigs, 
                      unsigned NumDesigs);

  Expr *getArrayIndex(const Designator& D);
  Expr *getArrayRangeStart(const Designator& D);
  Expr *getArrayRangeEnd(const Designator& D);

  /// @brief Retrieve the location of the '=' that precedes the
  /// initializer value itself, if present.
  SourceLocation getEqualOrColonLoc() const { return EqualOrColonLoc; }
  void setEqualOrColonLoc(SourceLocation L) { EqualOrColonLoc = L; }

  /// @brief Determines whether this designated initializer used the
  /// deprecated GNU syntax for designated initializers.
  bool usesGNUSyntax() const { return GNUSyntax; }
  void setGNUSyntax(bool GNU) { GNUSyntax = GNU; }

  /// @brief Retrieve the initializer value.
  Expr *getInit() const {
    return cast<Expr>(*const_cast<DesignatedInitExpr*>(this)->child_begin());
  }

  void setInit(Expr *init) {
    *child_begin() = init;
  }

  /// \brief Retrieve the total number of subexpressions in this
  /// designated initializer expression, including the actual
  /// initialized value and any expressions that occur within array
  /// and array-range designators.
  unsigned getNumSubExprs() const { return NumSubExprs; }

  Expr *getSubExpr(unsigned Idx) {
    assert(Idx < NumSubExprs && "Subscript out of range");
    char* Ptr = static_cast<char*>(static_cast<void *>(this));
    Ptr += sizeof(DesignatedInitExpr);
    return reinterpret_cast<Expr**>(reinterpret_cast<void**>(Ptr))[Idx];
  }

  void setSubExpr(unsigned Idx, Expr *E) {
    assert(Idx < NumSubExprs && "Subscript out of range");
    char* Ptr = static_cast<char*>(static_cast<void *>(this));
    Ptr += sizeof(DesignatedInitExpr);
    reinterpret_cast<Expr**>(reinterpret_cast<void**>(Ptr))[Idx] = E;
  }

  /// \brief Replaces the designator at index @p Idx with the series
  /// of designators in [First, Last).
  void ExpandDesignator(ASTContext &C, unsigned Idx, const Designator *First,
                        const Designator *Last);

  virtual SourceRange getSourceRange() const;

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == DesignatedInitExprClass;
  }
  static bool classof(const DesignatedInitExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// \brief Represents an implicitly-generated value initialization of
/// an object of a given type.
///
/// Implicit value initializations occur within semantic initializer
/// list expressions (InitListExpr) as placeholders for subobject
/// initializations not explicitly specified by the user.
///
/// \see InitListExpr
class ImplicitValueInitExpr : public Expr {
public:
  explicit ImplicitValueInitExpr(QualType ty)
    : Expr(ImplicitValueInitExprClass, ty, false, false) { }

  /// \brief Construct an empty implicit value initialization.
  explicit ImplicitValueInitExpr(EmptyShell Empty)
    : Expr(ImplicitValueInitExprClass, Empty) { }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ImplicitValueInitExprClass;
  }
  static bool classof(const ImplicitValueInitExpr *) { return true; }

  virtual SourceRange getSourceRange() const {
    return SourceRange();
  }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};


class ParenListExpr : public Expr {
  Stmt **Exprs;
  unsigned NumExprs;
  SourceLocation LParenLoc, RParenLoc;

public:
  ParenListExpr(ASTContext& C, SourceLocation lparenloc, Expr **exprs,
                unsigned numexprs, SourceLocation rparenloc);

  /// \brief Build an empty paren list.
  explicit ParenListExpr(EmptyShell Empty) : Expr(ParenListExprClass, Empty) { }

  unsigned getNumExprs() const { return NumExprs; }

  const Expr* getExpr(unsigned Init) const {
    assert(Init < getNumExprs() && "Initializer access out of range!");
    return cast_or_null<Expr>(Exprs[Init]);
  }

  Expr* getExpr(unsigned Init) {
    assert(Init < getNumExprs() && "Initializer access out of range!");
    return cast_or_null<Expr>(Exprs[Init]);
  }

  Expr **getExprs() { return reinterpret_cast<Expr **>(Exprs); }

  SourceLocation getLParenLoc() const { return LParenLoc; }
  SourceLocation getRParenLoc() const { return RParenLoc; }

  virtual SourceRange getSourceRange() const {
    return SourceRange(LParenLoc, RParenLoc);
  }
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ParenListExprClass;
  }
  static bool classof(const ParenListExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();

  friend class ASTStmtReader;
  friend class ASTStmtWriter;
};


//===----------------------------------------------------------------------===//
// Clang Extensions
//===----------------------------------------------------------------------===//


/// ExtVectorElementExpr - This represents access to specific elements of a
/// vector, and may occur on the left hand side or right hand side.  For example
/// the following is legal:  "V.xy = V.zw" if V is a 4 element extended vector.
///
/// Note that the base may have either vector or pointer to vector type, just
/// like a struct field reference.
///
class ExtVectorElementExpr : public Expr {
  Stmt *Base;
  IdentifierInfo *Accessor;
  SourceLocation AccessorLoc;
public:
  ExtVectorElementExpr(QualType ty, Expr *base, IdentifierInfo &accessor,
                       SourceLocation loc)
    : Expr(ExtVectorElementExprClass, ty, base->isTypeDependent(),
           base->isValueDependent()),
      Base(base), Accessor(&accessor), AccessorLoc(loc) {}

  /// \brief Build an empty vector element expression.
  explicit ExtVectorElementExpr(EmptyShell Empty)
    : Expr(ExtVectorElementExprClass, Empty) { }

  const Expr *getBase() const { return cast<Expr>(Base); }
  Expr *getBase() { return cast<Expr>(Base); }
  void setBase(Expr *E) { Base = E; }

  IdentifierInfo &getAccessor() const { return *Accessor; }
  void setAccessor(IdentifierInfo *II) { Accessor = II; }

  SourceLocation getAccessorLoc() const { return AccessorLoc; }
  void setAccessorLoc(SourceLocation L) { AccessorLoc = L; }

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

  /// isArrow - Return true if the base expression is a pointer to vector,
  /// return false if the base expression is a vector.
  bool isArrow() const;

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ExtVectorElementExprClass;
  }
  static bool classof(const ExtVectorElementExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};


/// BlockExpr - Adaptor class for mixing a BlockDecl with expressions.
/// ^{ statement-body }   or   ^(int arg1, float arg2){ statement-body }
class BlockExpr : public Expr {
protected:
  BlockDecl *TheBlock;
  bool HasBlockDeclRefExprs;
public:
  BlockExpr(BlockDecl *BD, QualType ty, bool hasBlockDeclRefExprs)
    : Expr(BlockExprClass, ty, ty->isDependentType(), false),
      TheBlock(BD), HasBlockDeclRefExprs(hasBlockDeclRefExprs) {}

  /// \brief Build an empty block expression.
  explicit BlockExpr(EmptyShell Empty) : Expr(BlockExprClass, Empty) { }

  const BlockDecl *getBlockDecl() const { return TheBlock; }
  BlockDecl *getBlockDecl() { return TheBlock; }
  void setBlockDecl(BlockDecl *BD) { TheBlock = BD; }

  // Convenience functions for probing the underlying BlockDecl.
  SourceLocation getCaretLocation() const;
  const Stmt *getBody() const;
  Stmt *getBody();

  virtual SourceRange getSourceRange() const {
    return SourceRange(getCaretLocation(), getBody()->getLocEnd());
  }

  /// getFunctionType - Return the underlying function type for this block.
  const FunctionType *getFunctionType() const;

  /// hasBlockDeclRefExprs - Return true iff the block has BlockDeclRefExpr
  /// inside of the block that reference values outside the block.
  bool hasBlockDeclRefExprs() const { return HasBlockDeclRefExprs; }
  void setHasBlockDeclRefExprs(bool BDRE) { HasBlockDeclRefExprs = BDRE; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == BlockExprClass;
  }
  static bool classof(const BlockExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// BlockDeclRefExpr - A reference to a declared variable, function,
/// enum, etc.
class BlockDeclRefExpr : public Expr {
  ValueDecl *D;
  SourceLocation Loc;
  bool IsByRef : 1;
  bool ConstQualAdded : 1;
  Stmt *CopyConstructorVal;
public:
  // FIXME: Fix type/value dependence!
  BlockDeclRefExpr(ValueDecl *d, QualType t, SourceLocation l, bool ByRef,
                   bool constAdded = false,
                   Stmt *copyConstructorVal = 0)
  : Expr(BlockDeclRefExprClass, t, (!t.isNull() && t->isDependentType()),false),
    D(d), Loc(l), IsByRef(ByRef),
    ConstQualAdded(constAdded),  CopyConstructorVal(copyConstructorVal) {}

  // \brief Build an empty reference to a declared variable in a
  // block.
  explicit BlockDeclRefExpr(EmptyShell Empty)
    : Expr(BlockDeclRefExprClass, Empty) { }

  ValueDecl *getDecl() { return D; }
  const ValueDecl *getDecl() const { return D; }
  void setDecl(ValueDecl *VD) { D = VD; }

  SourceLocation getLocation() const { return Loc; }
  void setLocation(SourceLocation L) { Loc = L; }

  virtual SourceRange getSourceRange() const { return SourceRange(Loc); }

  bool isByRef() const { return IsByRef; }
  void setByRef(bool BR) { IsByRef = BR; }

  bool isConstQualAdded() const { return ConstQualAdded; }
  void setConstQualAdded(bool C) { ConstQualAdded = C; }
  
  const Expr *getCopyConstructorExpr() const 
    { return cast_or_null<Expr>(CopyConstructorVal); }
  Expr *getCopyConstructorExpr() 
    { return cast_or_null<Expr>(CopyConstructorVal); }
  void setCopyConstructorExpr(Expr *E) { CopyConstructorVal = E; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == BlockDeclRefExprClass;
  }
  static bool classof(const BlockDeclRefExpr *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// OpaqueValueExpr - An expression referring to an opaque object of a
/// fixed type and value class.  These don't correspond to concrete
/// syntax; instead they're used to express operations (usually copy
/// operations) on values whose source is generally obvious from
/// context.
class OpaqueValueExpr : public Expr {
  friend class ASTStmtReader;
public:
  OpaqueValueExpr(QualType T, ExprValueKind VK)
    : Expr(OpaqueValueExprClass, T, T->isDependentType(),
           T->isDependentType()) {
    setValueKind(VK);
  }

  explicit OpaqueValueExpr(EmptyShell Empty)
    : Expr(OpaqueValueExprClass, Empty) { }

  using Expr::getValueKind;

  virtual SourceRange getSourceRange() const;
  virtual child_iterator child_begin();
  virtual child_iterator child_end();

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OpaqueValueExprClass;
  }
  static bool classof(const OpaqueValueExpr *) { return true; }
};

}  // end namespace clang

#endif
