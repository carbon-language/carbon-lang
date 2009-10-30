//===------- TreeTransform.h - Semantic Tree Transformation -----*- C++ -*-===/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===/
//
//  This file implements a semantic tree transformation that takes a given
//  AST and rebuilds it, possibly transforming some nodes in the process.
//
//===----------------------------------------------------------------------===/
#ifndef LLVM_CLANG_SEMA_TREETRANSFORM_H
#define LLVM_CLANG_SEMA_TREETRANSFORM_H

#include "Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/TypeLocBuilder.h"
#include "clang/Parse/Ownership.h"
#include "clang/Parse/Designator.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>

namespace clang {

/// \brief A semantic tree transformation that allows one to transform one
/// abstract syntax tree into another.
///
/// A new tree transformation is defined by creating a new subclass \c X of
/// \c TreeTransform<X> and then overriding certain operations to provide
/// behavior specific to that transformation. For example, template
/// instantiation is implemented as a tree transformation where the
/// transformation of TemplateTypeParmType nodes involves substituting the
/// template arguments for their corresponding template parameters; a similar
/// transformation is performed for non-type template parameters and
/// template template parameters.
///
/// This tree-transformation template uses static polymorphism to allow
/// subclasses to customize any of its operations. Thus, a subclass can
/// override any of the transformation or rebuild operators by providing an
/// operation with the same signature as the default implementation. The
/// overridding function should not be virtual.
///
/// Semantic tree transformations are split into two stages, either of which
/// can be replaced by a subclass. The "transform" step transforms an AST node
/// or the parts of an AST node using the various transformation functions,
/// then passes the pieces on to the "rebuild" step, which constructs a new AST
/// node of the appropriate kind from the pieces. The default transformation
/// routines recursively transform the operands to composite AST nodes (e.g.,
/// the pointee type of a PointerType node) and, if any of those operand nodes
/// were changed by the transformation, invokes the rebuild operation to create
/// a new AST node.
///
/// Subclasses can customize the transformation at various levels. The
/// most coarse-grained transformations involve replacing TransformType(),
/// TransformExpr(), TransformDecl(), TransformNestedNameSpecifier(),
/// TransformTemplateName(), or TransformTemplateArgument() with entirely
/// new implementations.
///
/// For more fine-grained transformations, subclasses can replace any of the
/// \c TransformXXX functions (where XXX is the name of an AST node, e.g.,
/// PointerType, StmtExpr) to alter the transformation. As mentioned previously,
/// replacing TransformTemplateTypeParmType() allows template instantiation
/// to substitute template arguments for their corresponding template
/// parameters. Additionally, subclasses can override the \c RebuildXXX
/// functions to control how AST nodes are rebuilt when their operands change.
/// By default, \c TreeTransform will invoke semantic analysis to rebuild
/// AST nodes. However, certain other tree transformations (e.g, cloning) may
/// be able to use more efficient rebuild steps.
///
/// There are a handful of other functions that can be overridden, allowing one
/// to avoid traversing nodes that don't need any transformation
/// (\c AlreadyTransformed()), force rebuilding AST nodes even when their
/// operands have not changed (\c AlwaysRebuild()), and customize the
/// default locations and entity names used for type-checking
/// (\c getBaseLocation(), \c getBaseEntity()).
template<typename Derived>
class TreeTransform {
protected:
  Sema &SemaRef;

public:
  typedef Sema::OwningStmtResult OwningStmtResult;
  typedef Sema::OwningExprResult OwningExprResult;
  typedef Sema::StmtArg StmtArg;
  typedef Sema::ExprArg ExprArg;
  typedef Sema::MultiExprArg MultiExprArg;
  typedef Sema::MultiStmtArg MultiStmtArg;

  /// \brief Initializes a new tree transformer.
  TreeTransform(Sema &SemaRef) : SemaRef(SemaRef) { }

  /// \brief Retrieves a reference to the derived class.
  Derived &getDerived() { return static_cast<Derived&>(*this); }

  /// \brief Retrieves a reference to the derived class.
  const Derived &getDerived() const {
    return static_cast<const Derived&>(*this);
  }

  /// \brief Retrieves a reference to the semantic analysis object used for
  /// this tree transform.
  Sema &getSema() const { return SemaRef; }

  /// \brief Whether the transformation should always rebuild AST nodes, even
  /// if none of the children have changed.
  ///
  /// Subclasses may override this function to specify when the transformation
  /// should rebuild all AST nodes.
  bool AlwaysRebuild() { return false; }

  /// \brief Returns the location of the entity being transformed, if that
  /// information was not available elsewhere in the AST.
  ///
  /// By default, returns no source-location information. Subclasses can
  /// provide an alternative implementation that provides better location
  /// information.
  SourceLocation getBaseLocation() { return SourceLocation(); }

  /// \brief Returns the name of the entity being transformed, if that
  /// information was not available elsewhere in the AST.
  ///
  /// By default, returns an empty name. Subclasses can provide an alternative
  /// implementation with a more precise name.
  DeclarationName getBaseEntity() { return DeclarationName(); }

  /// \brief Sets the "base" location and entity when that
  /// information is known based on another transformation.
  ///
  /// By default, the source location and entity are ignored. Subclasses can
  /// override this function to provide a customized implementation.
  void setBase(SourceLocation Loc, DeclarationName Entity) { }

  /// \brief RAII object that temporarily sets the base location and entity
  /// used for reporting diagnostics in types.
  class TemporaryBase {
    TreeTransform &Self;
    SourceLocation OldLocation;
    DeclarationName OldEntity;

  public:
    TemporaryBase(TreeTransform &Self, SourceLocation Location,
                  DeclarationName Entity) : Self(Self) {
      OldLocation = Self.getDerived().getBaseLocation();
      OldEntity = Self.getDerived().getBaseEntity();
      Self.getDerived().setBase(Location, Entity);
    }

    ~TemporaryBase() {
      Self.getDerived().setBase(OldLocation, OldEntity);
    }
  };

  /// \brief Determine whether the given type \p T has already been
  /// transformed.
  ///
  /// Subclasses can provide an alternative implementation of this routine
  /// to short-circuit evaluation when it is known that a given type will
  /// not change. For example, template instantiation need not traverse
  /// non-dependent types.
  bool AlreadyTransformed(QualType T) {
    return T.isNull();
  }

  /// \brief Transforms the given type into another type.
  ///
  /// By default, this routine transforms a type by creating a
  /// DeclaratorInfo for it and delegating to the appropriate
  /// function.  This is expensive, but we don't mind, because
  /// this method is deprecated anyway;  all users should be
  /// switched to storing DeclaratorInfos.
  ///
  /// \returns the transformed type.
  QualType TransformType(QualType T);

  /// \brief Transforms the given type-with-location into a new
  /// type-with-location.
  ///
  /// By default, this routine transforms a type by delegating to the
  /// appropriate TransformXXXType to build a new type.  Subclasses
  /// may override this function (to take over all type
  /// transformations) or some set of the TransformXXXType functions
  /// to alter the transformation.
  DeclaratorInfo *TransformType(DeclaratorInfo *DI);

  /// \brief Transform the given type-with-location into a new
  /// type, collecting location information in the given builder
  /// as necessary.
  ///
  QualType TransformType(TypeLocBuilder &TLB, TypeLoc TL);

  /// \brief Transform the given statement.
  ///
  /// By default, this routine transforms a statement by delegating to the
  /// appropriate TransformXXXStmt function to transform a specific kind of
  /// statement or the TransformExpr() function to transform an expression.
  /// Subclasses may override this function to transform statements using some
  /// other mechanism.
  ///
  /// \returns the transformed statement.
  OwningStmtResult TransformStmt(Stmt *S);

  /// \brief Transform the given expression.
  ///
  /// By default, this routine transforms an expression by delegating to the
  /// appropriate TransformXXXExpr function to build a new expression.
  /// Subclasses may override this function to transform expressions using some
  /// other mechanism.
  ///
  /// \returns the transformed expression.
  OwningExprResult TransformExpr(Expr *E) {
    return getDerived().TransformExpr(E, /*isAddressOfOperand=*/false);
  }

  /// \brief Transform the given expression.
  ///
  /// By default, this routine transforms an expression by delegating to the
  /// appropriate TransformXXXExpr function to build a new expression.
  /// Subclasses may override this function to transform expressions using some
  /// other mechanism.
  ///
  /// \returns the transformed expression.
  OwningExprResult TransformExpr(Expr *E, bool isAddressOfOperand);

  /// \brief Transform the given declaration, which is referenced from a type
  /// or expression.
  ///
  /// By default, acts as the identity function on declarations. Subclasses
  /// may override this function to provide alternate behavior.
  Decl *TransformDecl(Decl *D) { return D; }

  /// \brief Transform the definition of the given declaration.
  ///
  /// By default, invokes TransformDecl() to transform the declaration.
  /// Subclasses may override this function to provide alternate behavior.
  Decl *TransformDefinition(Decl *D) { return getDerived().TransformDecl(D); }

  /// \brief Transform the given declaration, which was the first part of a
  /// nested-name-specifier in a member access expression.
  ///
  /// This specific declaration transformation only applies to the first 
  /// identifier in a nested-name-specifier of a member access expression, e.g.,
  /// the \c T in \c x->T::member
  ///
  /// By default, invokes TransformDecl() to transform the declaration.
  /// Subclasses may override this function to provide alternate behavior.
  NamedDecl *TransformFirstQualifierInScope(NamedDecl *D, SourceLocation Loc) { 
    return cast_or_null<NamedDecl>(getDerived().TransformDecl(D)); 
  }
  
  /// \brief Transform the given nested-name-specifier.
  ///
  /// By default, transforms all of the types and declarations within the
  /// nested-name-specifier. Subclasses may override this function to provide
  /// alternate behavior.
  NestedNameSpecifier *TransformNestedNameSpecifier(NestedNameSpecifier *NNS,
                                                    SourceRange Range,
                                              QualType ObjectType = QualType(),
                                          NamedDecl *FirstQualifierInScope = 0);

  /// \brief Transform the given declaration name.
  ///
  /// By default, transforms the types of conversion function, constructor,
  /// and destructor names and then (if needed) rebuilds the declaration name.
  /// Identifiers and selectors are returned unmodified. Sublcasses may
  /// override this function to provide alternate behavior.
  DeclarationName TransformDeclarationName(DeclarationName Name,
                                           SourceLocation Loc,
                                           QualType ObjectType = QualType());

  /// \brief Transform the given template name.
  ///
  /// By default, transforms the template name by transforming the declarations
  /// and nested-name-specifiers that occur within the template name.
  /// Subclasses may override this function to provide alternate behavior.
  TemplateName TransformTemplateName(TemplateName Name,
                                     QualType ObjectType = QualType());

  /// \brief Transform the given template argument.
  ///
  /// By default, this operation transforms the type, expression, or
  /// declaration stored within the template argument and constructs a
  /// new template argument from the transformed result. Subclasses may
  /// override this function to provide alternate behavior.
  ///
  /// Returns true if there was an error.
  bool TransformTemplateArgument(const TemplateArgumentLoc &Input,
                                 TemplateArgumentLoc &Output);

  /// \brief Fakes up a TemplateArgumentLoc for a given TemplateArgument.
  void InventTemplateArgumentLoc(const TemplateArgument &Arg,
                                 TemplateArgumentLoc &ArgLoc);

  /// \brief Fakes up a DeclaratorInfo for a type.
  DeclaratorInfo *InventDeclaratorInfo(QualType T) {
    return SemaRef.Context.getTrivialDeclaratorInfo(T,
                       getDerived().getBaseLocation());
  }

#define ABSTRACT_TYPELOC(CLASS, PARENT)
#define TYPELOC(CLASS, PARENT)                                   \
  QualType Transform##CLASS##Type(TypeLocBuilder &TLB, CLASS##TypeLoc T);
#include "clang/AST/TypeLocNodes.def"

  QualType TransformReferenceType(TypeLocBuilder &TLB, ReferenceTypeLoc TL);

  QualType 
  TransformTemplateSpecializationType(const TemplateSpecializationType *T,
                                      QualType ObjectType);

  QualType
  TransformTemplateSpecializationType(TypeLocBuilder &TLB,
                                      TemplateSpecializationTypeLoc TL,
                                      QualType ObjectType);
  
  OwningStmtResult TransformCompoundStmt(CompoundStmt *S, bool IsStmtExpr);

#define STMT(Node, Parent)                        \
  OwningStmtResult Transform##Node(Node *S);
#define EXPR(Node, Parent)                        \
  OwningExprResult Transform##Node(Node *E);
#define ABSTRACT_EXPR(Node, Parent)
#include "clang/AST/StmtNodes.def"

  /// \brief Build a new pointer type given its pointee type.
  ///
  /// By default, performs semantic analysis when building the pointer type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildPointerType(QualType PointeeType, SourceLocation Sigil);

  /// \brief Build a new block pointer type given its pointee type.
  ///
  /// By default, performs semantic analysis when building the block pointer
  /// type. Subclasses may override this routine to provide different behavior.
  QualType RebuildBlockPointerType(QualType PointeeType, SourceLocation Sigil);

  /// \brief Build a new reference type given the type it references.
  ///
  /// By default, performs semantic analysis when building the
  /// reference type. Subclasses may override this routine to provide
  /// different behavior.
  ///
  /// \param LValue whether the type was written with an lvalue sigil
  /// or an rvalue sigil.
  QualType RebuildReferenceType(QualType ReferentType,
                                bool LValue,
                                SourceLocation Sigil);

  /// \brief Build a new member pointer type given the pointee type and the
  /// class type it refers into.
  ///
  /// By default, performs semantic analysis when building the member pointer
  /// type. Subclasses may override this routine to provide different behavior.
  QualType RebuildMemberPointerType(QualType PointeeType, QualType ClassType,
                                    SourceLocation Sigil);

  /// \brief Build a new Objective C object pointer type.
  QualType RebuildObjCObjectPointerType(QualType PointeeType,
                                        SourceLocation Sigil);

  /// \brief Build a new array type given the element type, size
  /// modifier, size of the array (if known), size expression, and index type
  /// qualifiers.
  ///
  /// By default, performs semantic analysis when building the array type.
  /// Subclasses may override this routine to provide different behavior.
  /// Also by default, all of the other Rebuild*Array
  QualType RebuildArrayType(QualType ElementType,
                            ArrayType::ArraySizeModifier SizeMod,
                            const llvm::APInt *Size,
                            Expr *SizeExpr,
                            unsigned IndexTypeQuals,
                            SourceRange BracketsRange);

  /// \brief Build a new constant array type given the element type, size
  /// modifier, (known) size of the array, and index type qualifiers.
  ///
  /// By default, performs semantic analysis when building the array type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildConstantArrayType(QualType ElementType,
                                    ArrayType::ArraySizeModifier SizeMod,
                                    const llvm::APInt &Size,
                                    unsigned IndexTypeQuals,
                                    SourceRange BracketsRange);

  /// \brief Build a new incomplete array type given the element type, size
  /// modifier, and index type qualifiers.
  ///
  /// By default, performs semantic analysis when building the array type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildIncompleteArrayType(QualType ElementType,
                                      ArrayType::ArraySizeModifier SizeMod,
                                      unsigned IndexTypeQuals,
                                      SourceRange BracketsRange);

  /// \brief Build a new variable-length array type given the element type,
  /// size modifier, size expression, and index type qualifiers.
  ///
  /// By default, performs semantic analysis when building the array type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildVariableArrayType(QualType ElementType,
                                    ArrayType::ArraySizeModifier SizeMod,
                                    ExprArg SizeExpr,
                                    unsigned IndexTypeQuals,
                                    SourceRange BracketsRange);

  /// \brief Build a new dependent-sized array type given the element type,
  /// size modifier, size expression, and index type qualifiers.
  ///
  /// By default, performs semantic analysis when building the array type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildDependentSizedArrayType(QualType ElementType,
                                          ArrayType::ArraySizeModifier SizeMod,
                                          ExprArg SizeExpr,
                                          unsigned IndexTypeQuals,
                                          SourceRange BracketsRange);

  /// \brief Build a new vector type given the element type and
  /// number of elements.
  ///
  /// By default, performs semantic analysis when building the vector type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildVectorType(QualType ElementType, unsigned NumElements);

  /// \brief Build a new extended vector type given the element type and
  /// number of elements.
  ///
  /// By default, performs semantic analysis when building the vector type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildExtVectorType(QualType ElementType, unsigned NumElements,
                                SourceLocation AttributeLoc);

  /// \brief Build a new potentially dependently-sized extended vector type
  /// given the element type and number of elements.
  ///
  /// By default, performs semantic analysis when building the vector type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildDependentSizedExtVectorType(QualType ElementType,
                                              ExprArg SizeExpr,
                                              SourceLocation AttributeLoc);

  /// \brief Build a new function type.
  ///
  /// By default, performs semantic analysis when building the function type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildFunctionProtoType(QualType T,
                                    QualType *ParamTypes,
                                    unsigned NumParamTypes,
                                    bool Variadic, unsigned Quals);

  /// \brief Build a new unprototyped function type.
  QualType RebuildFunctionNoProtoType(QualType ResultType);

  /// \brief Build a new typedef type.
  QualType RebuildTypedefType(TypedefDecl *Typedef) {
    return SemaRef.Context.getTypeDeclType(Typedef);
  }

  /// \brief Build a new class/struct/union type.
  QualType RebuildRecordType(RecordDecl *Record) {
    return SemaRef.Context.getTypeDeclType(Record);
  }

  /// \brief Build a new Enum type.
  QualType RebuildEnumType(EnumDecl *Enum) {
    return SemaRef.Context.getTypeDeclType(Enum);
  }

  /// \brief Build a new elaborated type.
  QualType RebuildElaboratedType(QualType T, ElaboratedType::TagKind Tag) {
    return SemaRef.Context.getElaboratedType(T, Tag);
  }

  /// \brief Build a new typeof(expr) type.
  ///
  /// By default, performs semantic analysis when building the typeof type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildTypeOfExprType(ExprArg Underlying);

  /// \brief Build a new typeof(type) type.
  ///
  /// By default, builds a new TypeOfType with the given underlying type.
  QualType RebuildTypeOfType(QualType Underlying);

  /// \brief Build a new C++0x decltype type.
  ///
  /// By default, performs semantic analysis when building the decltype type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildDecltypeType(ExprArg Underlying);

  /// \brief Build a new template specialization type.
  ///
  /// By default, performs semantic analysis when building the template
  /// specialization type. Subclasses may override this routine to provide
  /// different behavior.
  QualType RebuildTemplateSpecializationType(TemplateName Template,
                                             SourceLocation TemplateLoc,
                                             SourceLocation LAngleLoc,
                                             const TemplateArgumentLoc *Args,
                                             unsigned NumArgs,
                                             SourceLocation RAngleLoc);

  /// \brief Build a new qualified name type.
  ///
  /// By default, builds a new QualifiedNameType type from the
  /// nested-name-specifier and the named type. Subclasses may override
  /// this routine to provide different behavior.
  QualType RebuildQualifiedNameType(NestedNameSpecifier *NNS, QualType Named) {
    return SemaRef.Context.getQualifiedNameType(NNS, Named);
  }

  /// \brief Build a new typename type that refers to a template-id.
  ///
  /// By default, builds a new TypenameType type from the nested-name-specifier
  /// and the given type. Subclasses may override this routine to provide
  /// different behavior.
  QualType RebuildTypenameType(NestedNameSpecifier *NNS, QualType T) {
    if (NNS->isDependent())
      return SemaRef.Context.getTypenameType(NNS,
                                          cast<TemplateSpecializationType>(T));

    return SemaRef.Context.getQualifiedNameType(NNS, T);
  }

  /// \brief Build a new typename type that refers to an identifier.
  ///
  /// By default, performs semantic analysis when building the typename type
  /// (or qualified name type). Subclasses may override this routine to provide
  /// different behavior.
  QualType RebuildTypenameType(NestedNameSpecifier *NNS,
                               const IdentifierInfo *Id,
                               SourceRange SR) {
    return SemaRef.CheckTypenameType(NNS, *Id, SR);
  }

  /// \brief Build a new nested-name-specifier given the prefix and an
  /// identifier that names the next step in the nested-name-specifier.
  ///
  /// By default, performs semantic analysis when building the new
  /// nested-name-specifier. Subclasses may override this routine to provide
  /// different behavior.
  NestedNameSpecifier *RebuildNestedNameSpecifier(NestedNameSpecifier *Prefix,
                                                  SourceRange Range,
                                                  IdentifierInfo &II,
                                                  QualType ObjectType,
                                              NamedDecl *FirstQualifierInScope);

  /// \brief Build a new nested-name-specifier given the prefix and the
  /// namespace named in the next step in the nested-name-specifier.
  ///
  /// By default, performs semantic analysis when building the new
  /// nested-name-specifier. Subclasses may override this routine to provide
  /// different behavior.
  NestedNameSpecifier *RebuildNestedNameSpecifier(NestedNameSpecifier *Prefix,
                                                  SourceRange Range,
                                                  NamespaceDecl *NS);

  /// \brief Build a new nested-name-specifier given the prefix and the
  /// type named in the next step in the nested-name-specifier.
  ///
  /// By default, performs semantic analysis when building the new
  /// nested-name-specifier. Subclasses may override this routine to provide
  /// different behavior.
  NestedNameSpecifier *RebuildNestedNameSpecifier(NestedNameSpecifier *Prefix,
                                                  SourceRange Range,
                                                  bool TemplateKW,
                                                  QualType T);

  /// \brief Build a new template name given a nested name specifier, a flag
  /// indicating whether the "template" keyword was provided, and the template
  /// that the template name refers to.
  ///
  /// By default, builds the new template name directly. Subclasses may override
  /// this routine to provide different behavior.
  TemplateName RebuildTemplateName(NestedNameSpecifier *Qualifier,
                                   bool TemplateKW,
                                   TemplateDecl *Template);

  /// \brief Build a new template name given a nested name specifier, a flag
  /// indicating whether the "template" keyword was provided, and a set of
  /// overloaded function templates.
  ///
  /// By default, builds the new template name directly. Subclasses may override
  /// this routine to provide different behavior.
  TemplateName RebuildTemplateName(NestedNameSpecifier *Qualifier,
                                   bool TemplateKW,
                                   OverloadedFunctionDecl *Ovl);

  /// \brief Build a new template name given a nested name specifier and the
  /// name that is referred to as a template.
  ///
  /// By default, performs semantic analysis to determine whether the name can
  /// be resolved to a specific template, then builds the appropriate kind of
  /// template name. Subclasses may override this routine to provide different
  /// behavior.
  TemplateName RebuildTemplateName(NestedNameSpecifier *Qualifier,
                                   const IdentifierInfo &II,
                                   QualType ObjectType);


  /// \brief Build a new compound statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildCompoundStmt(SourceLocation LBraceLoc,
                                       MultiStmtArg Statements,
                                       SourceLocation RBraceLoc,
                                       bool IsStmtExpr) {
    return getSema().ActOnCompoundStmt(LBraceLoc, RBraceLoc, move(Statements),
                                       IsStmtExpr);
  }

  /// \brief Build a new case statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildCaseStmt(SourceLocation CaseLoc,
                                   ExprArg LHS,
                                   SourceLocation EllipsisLoc,
                                   ExprArg RHS,
                                   SourceLocation ColonLoc) {
    return getSema().ActOnCaseStmt(CaseLoc, move(LHS), EllipsisLoc, move(RHS),
                                   ColonLoc);
  }

  /// \brief Attach the body to a new case statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildCaseStmtBody(StmtArg S, StmtArg Body) {
    getSema().ActOnCaseStmtBody(S.get(), move(Body));
    return move(S);
  }

  /// \brief Build a new default statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildDefaultStmt(SourceLocation DefaultLoc,
                                      SourceLocation ColonLoc,
                                      StmtArg SubStmt) {
    return getSema().ActOnDefaultStmt(DefaultLoc, ColonLoc, move(SubStmt),
                                      /*CurScope=*/0);
  }

  /// \brief Build a new label statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildLabelStmt(SourceLocation IdentLoc,
                                    IdentifierInfo *Id,
                                    SourceLocation ColonLoc,
                                    StmtArg SubStmt) {
    return SemaRef.ActOnLabelStmt(IdentLoc, Id, ColonLoc, move(SubStmt));
  }

  /// \brief Build a new "if" statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildIfStmt(SourceLocation IfLoc, Sema::FullExprArg Cond,
                                 StmtArg Then, SourceLocation ElseLoc,
                                 StmtArg Else) {
    return getSema().ActOnIfStmt(IfLoc, Cond, move(Then), ElseLoc, move(Else));
  }

  /// \brief Start building a new switch statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildSwitchStmtStart(ExprArg Cond) {
    return getSema().ActOnStartOfSwitchStmt(move(Cond));
  }

  /// \brief Attach the body to the switch statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildSwitchStmtBody(SourceLocation SwitchLoc,
                                         StmtArg Switch, StmtArg Body) {
    return getSema().ActOnFinishSwitchStmt(SwitchLoc, move(Switch),
                                         move(Body));
  }

  /// \brief Build a new while statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildWhileStmt(SourceLocation WhileLoc,
                                    Sema::FullExprArg Cond,
                                    StmtArg Body) {
    return getSema().ActOnWhileStmt(WhileLoc, Cond, move(Body));
  }

  /// \brief Build a new do-while statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildDoStmt(SourceLocation DoLoc, StmtArg Body,
                                 SourceLocation WhileLoc,
                                 SourceLocation LParenLoc,
                                 ExprArg Cond,
                                 SourceLocation RParenLoc) {
    return getSema().ActOnDoStmt(DoLoc, move(Body), WhileLoc, LParenLoc,
                                 move(Cond), RParenLoc);
  }

  /// \brief Build a new for statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildForStmt(SourceLocation ForLoc,
                                  SourceLocation LParenLoc,
                                  StmtArg Init, ExprArg Cond, ExprArg Inc,
                                  SourceLocation RParenLoc, StmtArg Body) {
    return getSema().ActOnForStmt(ForLoc, LParenLoc, move(Init), move(Cond),
                                  move(Inc), RParenLoc, move(Body));
  }

  /// \brief Build a new goto statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildGotoStmt(SourceLocation GotoLoc,
                                   SourceLocation LabelLoc,
                                   LabelStmt *Label) {
    return getSema().ActOnGotoStmt(GotoLoc, LabelLoc, Label->getID());
  }

  /// \brief Build a new indirect goto statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildIndirectGotoStmt(SourceLocation GotoLoc,
                                           SourceLocation StarLoc,
                                           ExprArg Target) {
    return getSema().ActOnIndirectGotoStmt(GotoLoc, StarLoc, move(Target));
  }

  /// \brief Build a new return statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildReturnStmt(SourceLocation ReturnLoc,
                                     ExprArg Result) {

    return getSema().ActOnReturnStmt(ReturnLoc, move(Result));
  }

  /// \brief Build a new declaration statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildDeclStmt(Decl **Decls, unsigned NumDecls,
                                   SourceLocation StartLoc,
                                   SourceLocation EndLoc) {
    return getSema().Owned(
             new (getSema().Context) DeclStmt(
                                        DeclGroupRef::Create(getSema().Context,
                                                             Decls, NumDecls),
                                              StartLoc, EndLoc));
  }

  /// \brief Build a new C++ exception declaration.
  ///
  /// By default, performs semantic analysis to build the new decaration.
  /// Subclasses may override this routine to provide different behavior.
  VarDecl *RebuildExceptionDecl(VarDecl *ExceptionDecl, QualType T,
                                DeclaratorInfo *Declarator,
                                IdentifierInfo *Name,
                                SourceLocation Loc,
                                SourceRange TypeRange) {
    return getSema().BuildExceptionDeclaration(0, T, Declarator, Name, Loc,
                                               TypeRange);
  }

  /// \brief Build a new C++ catch statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildCXXCatchStmt(SourceLocation CatchLoc,
                                       VarDecl *ExceptionDecl,
                                       StmtArg Handler) {
    return getSema().Owned(
             new (getSema().Context) CXXCatchStmt(CatchLoc, ExceptionDecl,
                                                  Handler.takeAs<Stmt>()));
  }

  /// \brief Build a new C++ try statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildCXXTryStmt(SourceLocation TryLoc,
                                     StmtArg TryBlock,
                                     MultiStmtArg Handlers) {
    return getSema().ActOnCXXTryBlock(TryLoc, move(TryBlock), move(Handlers));
  }

  /// \brief Build a new expression that references a declaration.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildDeclRefExpr(NestedNameSpecifier *Qualifier,
                                      SourceRange QualifierRange,
                                      NamedDecl *ND, SourceLocation Loc) {
    CXXScopeSpec SS;
    SS.setScopeRep(Qualifier);
    SS.setRange(QualifierRange);
    return getSema().BuildDeclarationNameExpr(Loc, ND,
                                              /*FIXME:*/false,
                                              &SS,
                                              /*FIXME:*/false);
  }

  /// \brief Build a new expression in parentheses.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildParenExpr(ExprArg SubExpr, SourceLocation LParen,
                                    SourceLocation RParen) {
    return getSema().ActOnParenExpr(LParen, RParen, move(SubExpr));
  }

  /// \brief Build a new pseudo-destructor expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXPseudoDestructorExpr(ExprArg Base,
                                                  SourceLocation OperatorLoc,
                                                  bool isArrow,
                                              SourceLocation DestroyedTypeLoc,
                                                  QualType DestroyedType,
                                               NestedNameSpecifier *Qualifier,
                                                  SourceRange QualifierRange) {
    CXXScopeSpec SS;
    if (Qualifier) {
      SS.setRange(QualifierRange);
      SS.setScopeRep(Qualifier);
    }

    DeclarationName Name
      = SemaRef.Context.DeclarationNames.getCXXDestructorName(
                               SemaRef.Context.getCanonicalType(DestroyedType));

    return getSema().BuildMemberReferenceExpr(/*Scope=*/0, move(Base),
                                              OperatorLoc,
                                              isArrow? tok::arrow : tok::period,
                                              DestroyedTypeLoc,
                                              Name,
                                              Sema::DeclPtrTy::make((Decl *)0),
                                              &SS);
  }

  /// \brief Build a new unary operator expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildUnaryOperator(SourceLocation OpLoc,
                                        UnaryOperator::Opcode Opc,
                                        ExprArg SubExpr) {
    return getSema().CreateBuiltinUnaryOp(OpLoc, Opc, move(SubExpr));
  }

  /// \brief Build a new sizeof or alignof expression with a type argument.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildSizeOfAlignOf(QualType T, SourceLocation OpLoc,
                                        bool isSizeOf, SourceRange R) {
    return getSema().CreateSizeOfAlignOfExpr(T, OpLoc, isSizeOf, R);
  }

  /// \brief Build a new sizeof or alignof expression with an expression
  /// argument.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildSizeOfAlignOf(ExprArg SubExpr, SourceLocation OpLoc,
                                        bool isSizeOf, SourceRange R) {
    OwningExprResult Result
      = getSema().CreateSizeOfAlignOfExpr((Expr *)SubExpr.get(),
                                          OpLoc, isSizeOf, R);
    if (Result.isInvalid())
      return getSema().ExprError();

    SubExpr.release();
    return move(Result);
  }

  /// \brief Build a new array subscript expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildArraySubscriptExpr(ExprArg LHS,
                                             SourceLocation LBracketLoc,
                                             ExprArg RHS,
                                             SourceLocation RBracketLoc) {
    return getSema().ActOnArraySubscriptExpr(/*Scope=*/0, move(LHS),
                                             LBracketLoc, move(RHS),
                                             RBracketLoc);
  }

  /// \brief Build a new call expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCallExpr(ExprArg Callee, SourceLocation LParenLoc,
                                   MultiExprArg Args,
                                   SourceLocation *CommaLocs,
                                   SourceLocation RParenLoc) {
    return getSema().ActOnCallExpr(/*Scope=*/0, move(Callee), LParenLoc,
                                   move(Args), CommaLocs, RParenLoc);
  }

  /// \brief Build a new member access expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildMemberExpr(ExprArg Base, SourceLocation OpLoc,
                                     bool isArrow,
                                     NestedNameSpecifier *Qualifier,
                                     SourceRange QualifierRange,
                                     SourceLocation MemberLoc,
                                     NamedDecl *Member) {
    if (!Member->getDeclName()) {
      // We have a reference to an unnamed field.
      assert(!Qualifier && "Can't have an unnamed field with a qualifier!");

      MemberExpr *ME =
        new (getSema().Context) MemberExpr(Base.takeAs<Expr>(), isArrow,
                                           Member, MemberLoc,
                                           cast<FieldDecl>(Member)->getType());
      return getSema().Owned(ME);
    }

    CXXScopeSpec SS;
    if (Qualifier) {
      SS.setRange(QualifierRange);
      SS.setScopeRep(Qualifier);
    }

    return getSema().BuildMemberReferenceExpr(/*Scope=*/0, move(Base), OpLoc,
                                              isArrow? tok::arrow : tok::period,
                                              MemberLoc,
                                              Member->getDeclName(),
                                     /*FIXME?*/Sema::DeclPtrTy::make((Decl*)0),
                                              &SS);
  }

  /// \brief Build a new binary operator expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildBinaryOperator(SourceLocation OpLoc,
                                         BinaryOperator::Opcode Opc,
                                         ExprArg LHS, ExprArg RHS) {
    OwningExprResult Result
      = getSema().CreateBuiltinBinOp(OpLoc, Opc, (Expr *)LHS.get(),
                                     (Expr *)RHS.get());
    if (Result.isInvalid())
      return SemaRef.ExprError();

    LHS.release();
    RHS.release();
    return move(Result);
  }

  /// \brief Build a new conditional operator expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildConditionalOperator(ExprArg Cond,
                                              SourceLocation QuestionLoc,
                                              ExprArg LHS,
                                              SourceLocation ColonLoc,
                                              ExprArg RHS) {
    return getSema().ActOnConditionalOp(QuestionLoc, ColonLoc, move(Cond),
                                        move(LHS), move(RHS));
  }

  /// \brief Build a new implicit cast expression.
  ///
  /// By default, builds a new implicit cast without any semantic analysis.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildImplicitCastExpr(QualType T, CastExpr::CastKind Kind,
                                           ExprArg SubExpr, bool isLvalue) {
    ImplicitCastExpr *ICE
      = new (getSema().Context) ImplicitCastExpr(T, Kind,
                                                 (Expr *)SubExpr.release(),
                                                 isLvalue);
    return getSema().Owned(ICE);
  }

  /// \brief Build a new C-style cast expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCStyleCaseExpr(SourceLocation LParenLoc,
                                         QualType ExplicitTy,
                                         SourceLocation RParenLoc,
                                         ExprArg SubExpr) {
    return getSema().ActOnCastExpr(/*Scope=*/0,
                                   LParenLoc,
                                   ExplicitTy.getAsOpaquePtr(),
                                   RParenLoc,
                                   move(SubExpr));
  }

  /// \brief Build a new compound literal expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCompoundLiteralExpr(SourceLocation LParenLoc,
                                              QualType T,
                                              SourceLocation RParenLoc,
                                              ExprArg Init) {
    return getSema().ActOnCompoundLiteral(LParenLoc, T.getAsOpaquePtr(),
                                          RParenLoc, move(Init));
  }

  /// \brief Build a new extended vector element access expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildExtVectorElementExpr(ExprArg Base,
                                               SourceLocation OpLoc,
                                               SourceLocation AccessorLoc,
                                               IdentifierInfo &Accessor) {
    return getSema().ActOnMemberReferenceExpr(/*Scope=*/0, move(Base), OpLoc,
                                              tok::period, AccessorLoc,
                                              Accessor,
                                     /*FIXME?*/Sema::DeclPtrTy::make((Decl*)0));
  }

  /// \brief Build a new initializer list expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildInitList(SourceLocation LBraceLoc,
                                   MultiExprArg Inits,
                                   SourceLocation RBraceLoc) {
    return SemaRef.ActOnInitList(LBraceLoc, move(Inits), RBraceLoc);
  }

  /// \brief Build a new designated initializer expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildDesignatedInitExpr(Designation &Desig,
                                             MultiExprArg ArrayExprs,
                                             SourceLocation EqualOrColonLoc,
                                             bool GNUSyntax,
                                             ExprArg Init) {
    OwningExprResult Result
      = SemaRef.ActOnDesignatedInitializer(Desig, EqualOrColonLoc, GNUSyntax,
                                           move(Init));
    if (Result.isInvalid())
      return SemaRef.ExprError();

    ArrayExprs.release();
    return move(Result);
  }

  /// \brief Build a new value-initialized expression.
  ///
  /// By default, builds the implicit value initialization without performing
  /// any semantic analysis. Subclasses may override this routine to provide
  /// different behavior.
  OwningExprResult RebuildImplicitValueInitExpr(QualType T) {
    return SemaRef.Owned(new (SemaRef.Context) ImplicitValueInitExpr(T));
  }

  /// \brief Build a new \c va_arg expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildVAArgExpr(SourceLocation BuiltinLoc, ExprArg SubExpr,
                                    QualType T, SourceLocation RParenLoc) {
    return getSema().ActOnVAArg(BuiltinLoc, move(SubExpr), T.getAsOpaquePtr(),
                                RParenLoc);
  }

  /// \brief Build a new expression list in parentheses.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildParenListExpr(SourceLocation LParenLoc,
                                        MultiExprArg SubExprs,
                                        SourceLocation RParenLoc) {
    return getSema().ActOnParenListExpr(LParenLoc, RParenLoc, move(SubExprs));
  }

  /// \brief Build a new address-of-label expression.
  ///
  /// By default, performs semantic analysis, using the name of the label
  /// rather than attempting to map the label statement itself.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildAddrLabelExpr(SourceLocation AmpAmpLoc,
                                        SourceLocation LabelLoc,
                                        LabelStmt *Label) {
    return getSema().ActOnAddrLabel(AmpAmpLoc, LabelLoc, Label->getID());
  }

  /// \brief Build a new GNU statement expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildStmtExpr(SourceLocation LParenLoc,
                                   StmtArg SubStmt,
                                   SourceLocation RParenLoc) {
    return getSema().ActOnStmtExpr(LParenLoc, move(SubStmt), RParenLoc);
  }

  /// \brief Build a new __builtin_types_compatible_p expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildTypesCompatibleExpr(SourceLocation BuiltinLoc,
                                              QualType T1, QualType T2,
                                              SourceLocation RParenLoc) {
    return getSema().ActOnTypesCompatibleExpr(BuiltinLoc,
                                              T1.getAsOpaquePtr(),
                                              T2.getAsOpaquePtr(),
                                              RParenLoc);
  }

  /// \brief Build a new __builtin_choose_expr expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildChooseExpr(SourceLocation BuiltinLoc,
                                     ExprArg Cond, ExprArg LHS, ExprArg RHS,
                                     SourceLocation RParenLoc) {
    return SemaRef.ActOnChooseExpr(BuiltinLoc,
                                   move(Cond), move(LHS), move(RHS),
                                   RParenLoc);
  }

  /// \brief Build a new overloaded operator call expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// The semantic analysis provides the behavior of template instantiation,
  /// copying with transformations that turn what looks like an overloaded
  /// operator call into a use of a builtin operator, performing
  /// argument-dependent lookup, etc. Subclasses may override this routine to
  /// provide different behavior.
  OwningExprResult RebuildCXXOperatorCallExpr(OverloadedOperatorKind Op,
                                              SourceLocation OpLoc,
                                              ExprArg Callee,
                                              ExprArg First,
                                              ExprArg Second);

  /// \brief Build a new C++ "named" cast expression, such as static_cast or
  /// reinterpret_cast.
  ///
  /// By default, this routine dispatches to one of the more-specific routines
  /// for a particular named case, e.g., RebuildCXXStaticCastExpr().
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXNamedCastExpr(SourceLocation OpLoc,
                                           Stmt::StmtClass Class,
                                           SourceLocation LAngleLoc,
                                           QualType T,
                                           SourceLocation RAngleLoc,
                                           SourceLocation LParenLoc,
                                           ExprArg SubExpr,
                                           SourceLocation RParenLoc) {
    switch (Class) {
    case Stmt::CXXStaticCastExprClass:
      return getDerived().RebuildCXXStaticCastExpr(OpLoc, LAngleLoc, T,
                                                   RAngleLoc, LParenLoc,
                                                   move(SubExpr), RParenLoc);

    case Stmt::CXXDynamicCastExprClass:
      return getDerived().RebuildCXXDynamicCastExpr(OpLoc, LAngleLoc, T,
                                                    RAngleLoc, LParenLoc,
                                                    move(SubExpr), RParenLoc);

    case Stmt::CXXReinterpretCastExprClass:
      return getDerived().RebuildCXXReinterpretCastExpr(OpLoc, LAngleLoc, T,
                                                        RAngleLoc, LParenLoc,
                                                        move(SubExpr),
                                                        RParenLoc);

    case Stmt::CXXConstCastExprClass:
      return getDerived().RebuildCXXConstCastExpr(OpLoc, LAngleLoc, T,
                                                   RAngleLoc, LParenLoc,
                                                   move(SubExpr), RParenLoc);

    default:
      assert(false && "Invalid C++ named cast");
      break;
    }

    return getSema().ExprError();
  }

  /// \brief Build a new C++ static_cast expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXStaticCastExpr(SourceLocation OpLoc,
                                            SourceLocation LAngleLoc,
                                            QualType T,
                                            SourceLocation RAngleLoc,
                                            SourceLocation LParenLoc,
                                            ExprArg SubExpr,
                                            SourceLocation RParenLoc) {
    return getSema().ActOnCXXNamedCast(OpLoc, tok::kw_static_cast,
                                       LAngleLoc, T.getAsOpaquePtr(), RAngleLoc,
                                       LParenLoc, move(SubExpr), RParenLoc);
  }

  /// \brief Build a new C++ dynamic_cast expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXDynamicCastExpr(SourceLocation OpLoc,
                                             SourceLocation LAngleLoc,
                                             QualType T,
                                             SourceLocation RAngleLoc,
                                             SourceLocation LParenLoc,
                                             ExprArg SubExpr,
                                             SourceLocation RParenLoc) {
    return getSema().ActOnCXXNamedCast(OpLoc, tok::kw_dynamic_cast,
                                       LAngleLoc, T.getAsOpaquePtr(), RAngleLoc,
                                       LParenLoc, move(SubExpr), RParenLoc);
  }

  /// \brief Build a new C++ reinterpret_cast expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXReinterpretCastExpr(SourceLocation OpLoc,
                                                 SourceLocation LAngleLoc,
                                                 QualType T,
                                                 SourceLocation RAngleLoc,
                                                 SourceLocation LParenLoc,
                                                 ExprArg SubExpr,
                                                 SourceLocation RParenLoc) {
    return getSema().ActOnCXXNamedCast(OpLoc, tok::kw_reinterpret_cast,
                                       LAngleLoc, T.getAsOpaquePtr(), RAngleLoc,
                                       LParenLoc, move(SubExpr), RParenLoc);
  }

  /// \brief Build a new C++ const_cast expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXConstCastExpr(SourceLocation OpLoc,
                                           SourceLocation LAngleLoc,
                                           QualType T,
                                           SourceLocation RAngleLoc,
                                           SourceLocation LParenLoc,
                                           ExprArg SubExpr,
                                           SourceLocation RParenLoc) {
    return getSema().ActOnCXXNamedCast(OpLoc, tok::kw_const_cast,
                                       LAngleLoc, T.getAsOpaquePtr(), RAngleLoc,
                                       LParenLoc, move(SubExpr), RParenLoc);
  }

  /// \brief Build a new C++ functional-style cast expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXFunctionalCastExpr(SourceRange TypeRange,
                                                QualType T,
                                                SourceLocation LParenLoc,
                                                ExprArg SubExpr,
                                                SourceLocation RParenLoc) {
    void *Sub = SubExpr.takeAs<Expr>();
    return getSema().ActOnCXXTypeConstructExpr(TypeRange,
                                               T.getAsOpaquePtr(),
                                               LParenLoc,
                                         Sema::MultiExprArg(getSema(), &Sub, 1),
                                               /*CommaLocs=*/0,
                                               RParenLoc);
  }

  /// \brief Build a new C++ typeid(type) expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXTypeidExpr(SourceLocation TypeidLoc,
                                        SourceLocation LParenLoc,
                                        QualType T,
                                        SourceLocation RParenLoc) {
    return getSema().ActOnCXXTypeid(TypeidLoc, LParenLoc, true,
                                    T.getAsOpaquePtr(), RParenLoc);
  }

  /// \brief Build a new C++ typeid(expr) expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXTypeidExpr(SourceLocation TypeidLoc,
                                        SourceLocation LParenLoc,
                                        ExprArg Operand,
                                        SourceLocation RParenLoc) {
    OwningExprResult Result
      = getSema().ActOnCXXTypeid(TypeidLoc, LParenLoc, false, Operand.get(),
                                 RParenLoc);
    if (Result.isInvalid())
      return getSema().ExprError();

    Operand.release(); // FIXME: since ActOnCXXTypeid silently took ownership
    return move(Result);
  }

  /// \brief Build a new C++ "this" expression.
  ///
  /// By default, builds a new "this" expression without performing any
  /// semantic analysis. Subclasses may override this routine to provide
  /// different behavior.
  OwningExprResult RebuildCXXThisExpr(SourceLocation ThisLoc,
                                      QualType ThisType) {
    return getSema().Owned(
                      new (getSema().Context) CXXThisExpr(ThisLoc, ThisType));
  }

  /// \brief Build a new C++ throw expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXThrowExpr(SourceLocation ThrowLoc, ExprArg Sub) {
    return getSema().ActOnCXXThrow(ThrowLoc, move(Sub));
  }

  /// \brief Build a new C++ default-argument expression.
  ///
  /// By default, builds a new default-argument expression, which does not
  /// require any semantic analysis. Subclasses may override this routine to
  /// provide different behavior.
  OwningExprResult RebuildCXXDefaultArgExpr(ParmVarDecl *Param) {
    return getSema().Owned(CXXDefaultArgExpr::Create(getSema().Context, Param));
  }

  /// \brief Build a new C++ zero-initialization expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXZeroInitValueExpr(SourceLocation TypeStartLoc,
                                               SourceLocation LParenLoc,
                                               QualType T,
                                               SourceLocation RParenLoc) {
    return getSema().ActOnCXXTypeConstructExpr(SourceRange(TypeStartLoc),
                                               T.getAsOpaquePtr(), LParenLoc,
                                               MultiExprArg(getSema(), 0, 0),
                                               0, RParenLoc);
  }

  /// \brief Build a new C++ conditional declaration expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXConditionDeclExpr(SourceLocation StartLoc,
                                               SourceLocation EqLoc,
                                               VarDecl *Var) {
    return SemaRef.Owned(new (SemaRef.Context) CXXConditionDeclExpr(StartLoc,
                                                                    EqLoc,
                                                                    Var));
  }

  /// \brief Build a new C++ "new" expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXNewExpr(SourceLocation StartLoc,
                                     bool UseGlobal,
                                     SourceLocation PlacementLParen,
                                     MultiExprArg PlacementArgs,
                                     SourceLocation PlacementRParen,
                                     bool ParenTypeId,
                                     QualType AllocType,
                                     SourceLocation TypeLoc,
                                     SourceRange TypeRange,
                                     ExprArg ArraySize,
                                     SourceLocation ConstructorLParen,
                                     MultiExprArg ConstructorArgs,
                                     SourceLocation ConstructorRParen) {
    return getSema().BuildCXXNew(StartLoc, UseGlobal,
                                 PlacementLParen,
                                 move(PlacementArgs),
                                 PlacementRParen,
                                 ParenTypeId,
                                 AllocType,
                                 TypeLoc,
                                 TypeRange,
                                 move(ArraySize),
                                 ConstructorLParen,
                                 move(ConstructorArgs),
                                 ConstructorRParen);
  }

  /// \brief Build a new C++ "delete" expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXDeleteExpr(SourceLocation StartLoc,
                                        bool IsGlobalDelete,
                                        bool IsArrayForm,
                                        ExprArg Operand) {
    return getSema().ActOnCXXDelete(StartLoc, IsGlobalDelete, IsArrayForm,
                                    move(Operand));
  }

  /// \brief Build a new unary type trait expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildUnaryTypeTrait(UnaryTypeTrait Trait,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         QualType T,
                                         SourceLocation RParenLoc) {
    return getSema().ActOnUnaryTypeTrait(Trait, StartLoc, LParenLoc,
                                         T.getAsOpaquePtr(), RParenLoc);
  }

  /// \brief Build a new (previously unresolved) declaration reference
  /// expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildUnresolvedDeclRefExpr(NestedNameSpecifier *NNS,
                                                SourceRange QualifierRange,
                                                DeclarationName Name,
                                                SourceLocation Location,
                                                bool IsAddressOfOperand) {
    CXXScopeSpec SS;
    SS.setRange(QualifierRange);
    SS.setScopeRep(NNS);
    return getSema().ActOnDeclarationNameExpr(/*Scope=*/0,
                                              Location,
                                              Name,
                                              /*Trailing lparen=*/false,
                                              &SS,
                                              IsAddressOfOperand);
  }

  /// \brief Build a new template-id expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildTemplateIdExpr(NestedNameSpecifier *Qualifier,
                                         SourceRange QualifierRange,
                                         TemplateName Template,
                                         SourceLocation TemplateLoc,
                                         SourceLocation LAngleLoc,
                                         TemplateArgumentLoc *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation RAngleLoc) {
    return getSema().BuildTemplateIdExpr(Qualifier, QualifierRange,
                                         Template, TemplateLoc,
                                         LAngleLoc,
                                         TemplateArgs, NumTemplateArgs,
                                         RAngleLoc);
  }

  /// \brief Build a new object-construction expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXConstructExpr(QualType T,
                                           CXXConstructorDecl *Constructor,
                                           bool IsElidable,
                                           MultiExprArg Args) {
    return getSema().BuildCXXConstructExpr(/*FIXME:ConstructLoc*/
                                           SourceLocation(),
                                           T, Constructor, IsElidable,
                                           move(Args));
  }

  /// \brief Build a new object-construction expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXTemporaryObjectExpr(SourceLocation TypeBeginLoc,
                                                 QualType T,
                                                 SourceLocation LParenLoc,
                                                 MultiExprArg Args,
                                                 SourceLocation *Commas,
                                                 SourceLocation RParenLoc) {
    return getSema().ActOnCXXTypeConstructExpr(SourceRange(TypeBeginLoc),
                                               T.getAsOpaquePtr(),
                                               LParenLoc,
                                               move(Args),
                                               Commas,
                                               RParenLoc);
  }

  /// \brief Build a new object-construction expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXUnresolvedConstructExpr(SourceLocation TypeBeginLoc,
                                                     QualType T,
                                                     SourceLocation LParenLoc,
                                                     MultiExprArg Args,
                                                     SourceLocation *Commas,
                                                     SourceLocation RParenLoc) {
    return getSema().ActOnCXXTypeConstructExpr(SourceRange(TypeBeginLoc,
                                                           /*FIXME*/LParenLoc),
                                               T.getAsOpaquePtr(),
                                               LParenLoc,
                                               move(Args),
                                               Commas,
                                               RParenLoc);
  }

  /// \brief Build a new member reference expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXUnresolvedMemberExpr(ExprArg BaseE,
                                                  bool IsArrow,
                                                  SourceLocation OperatorLoc,
                                              NestedNameSpecifier *Qualifier,
                                                  SourceRange QualifierRange,
                                                  DeclarationName Name,
                                                  SourceLocation MemberLoc,
                                             NamedDecl *FirstQualifierInScope) {
    OwningExprResult Base = move(BaseE);
    tok::TokenKind OpKind = IsArrow? tok::arrow : tok::period;

    CXXScopeSpec SS;
    SS.setRange(QualifierRange);
    SS.setScopeRep(Qualifier);

    return SemaRef.BuildMemberReferenceExpr(/*Scope=*/0,
                                            move(Base), OperatorLoc, OpKind,
                                            MemberLoc,
                                            Name,
                                    /*FIXME?*/Sema::DeclPtrTy::make((Decl*)0),
                                            &SS,
                                            FirstQualifierInScope);
  }

  /// \brief Build a new member reference expression with explicit template
  /// arguments.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXUnresolvedMemberExpr(ExprArg BaseE,
                                                  bool IsArrow,
                                                  SourceLocation OperatorLoc,
                                                NestedNameSpecifier *Qualifier,
                                                  SourceRange QualifierRange,
                                                  TemplateName Template,
                                                SourceLocation TemplateNameLoc,
                                              NamedDecl *FirstQualifierInScope,
                                                  SourceLocation LAngleLoc,
                                       const TemplateArgumentLoc *TemplateArgs,
                                                  unsigned NumTemplateArgs,
                                                  SourceLocation RAngleLoc) {
    OwningExprResult Base = move(BaseE);
    tok::TokenKind OpKind = IsArrow? tok::arrow : tok::period;

    CXXScopeSpec SS;
    SS.setRange(QualifierRange);
    SS.setScopeRep(Qualifier);

    // FIXME: We're going to end up looking up the template based on its name,
    // twice! Also, duplicates part of Sema::ActOnMemberTemplateIdReferenceExpr.
    DeclarationName Name;
    if (TemplateDecl *ActualTemplate = Template.getAsTemplateDecl())
      Name = ActualTemplate->getDeclName();
    else if (OverloadedFunctionDecl *Ovl
               = Template.getAsOverloadedFunctionDecl())
      Name = Ovl->getDeclName();
    else
      Name = Template.getAsDependentTemplateName()->getName();

      return SemaRef.BuildMemberReferenceExpr(/*Scope=*/0, move(Base),
                                              OperatorLoc, OpKind,
                                              TemplateNameLoc, Name, true,
                                              LAngleLoc, TemplateArgs,
                                              NumTemplateArgs, RAngleLoc,
                                              Sema::DeclPtrTy(), &SS);
  }

  /// \brief Build a new Objective-C @encode expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildObjCEncodeExpr(SourceLocation AtLoc,
                                         QualType T,
                                         SourceLocation RParenLoc) {
    return SemaRef.Owned(SemaRef.BuildObjCEncodeExpression(AtLoc, T,
                                                           RParenLoc));
  }

  /// \brief Build a new Objective-C protocol expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildObjCProtocolExpr(ObjCProtocolDecl *Protocol,
                                           SourceLocation AtLoc,
                                           SourceLocation ProtoLoc,
                                           SourceLocation LParenLoc,
                                           SourceLocation RParenLoc) {
    return SemaRef.Owned(SemaRef.ParseObjCProtocolExpression(
                                              Protocol->getIdentifier(),
                                                             AtLoc,
                                                             ProtoLoc,
                                                             LParenLoc,
                                                             RParenLoc));
  }

  /// \brief Build a new shuffle vector expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildShuffleVectorExpr(SourceLocation BuiltinLoc,
                                            MultiExprArg SubExprs,
                                            SourceLocation RParenLoc) {
    // Find the declaration for __builtin_shufflevector
    const IdentifierInfo &Name
      = SemaRef.Context.Idents.get("__builtin_shufflevector");
    TranslationUnitDecl *TUDecl = SemaRef.Context.getTranslationUnitDecl();
    DeclContext::lookup_result Lookup = TUDecl->lookup(DeclarationName(&Name));
    assert(Lookup.first != Lookup.second && "No __builtin_shufflevector?");

    // Build a reference to the __builtin_shufflevector builtin
    FunctionDecl *Builtin = cast<FunctionDecl>(*Lookup.first);
    Expr *Callee
      = new (SemaRef.Context) DeclRefExpr(Builtin, Builtin->getType(),
                                          BuiltinLoc, false, false);
    SemaRef.UsualUnaryConversions(Callee);

    // Build the CallExpr
    unsigned NumSubExprs = SubExprs.size();
    Expr **Subs = (Expr **)SubExprs.release();
    CallExpr *TheCall = new (SemaRef.Context) CallExpr(SemaRef.Context, Callee,
                                                       Subs, NumSubExprs,
                                                       Builtin->getResultType(),
                                                       RParenLoc);
    OwningExprResult OwnedCall(SemaRef.Owned(TheCall));

    // Type-check the __builtin_shufflevector expression.
    OwningExprResult Result = SemaRef.SemaBuiltinShuffleVector(TheCall);
    if (Result.isInvalid())
      return SemaRef.ExprError();

    OwnedCall.release();
    return move(Result);
  }
};

template<typename Derived>
Sema::OwningStmtResult TreeTransform<Derived>::TransformStmt(Stmt *S) {
  if (!S)
    return SemaRef.Owned(S);

  switch (S->getStmtClass()) {
  case Stmt::NoStmtClass: break;

  // Transform individual statement nodes
#define STMT(Node, Parent)                                              \
  case Stmt::Node##Class: return getDerived().Transform##Node(cast<Node>(S));
#define EXPR(Node, Parent)
#include "clang/AST/StmtNodes.def"

  // Transform expressions by calling TransformExpr.
#define STMT(Node, Parent)
#define EXPR(Node, Parent) case Stmt::Node##Class:
#include "clang/AST/StmtNodes.def"
    {
      Sema::OwningExprResult E = getDerived().TransformExpr(cast<Expr>(S));
      if (E.isInvalid())
        return getSema().StmtError();

      return getSema().Owned(E.takeAs<Stmt>());
    }
  }

  return SemaRef.Owned(S->Retain());
}


template<typename Derived>
Sema::OwningExprResult TreeTransform<Derived>::TransformExpr(Expr *E,
                                                    bool isAddressOfOperand) {
  if (!E)
    return SemaRef.Owned(E);

  switch (E->getStmtClass()) {
    case Stmt::NoStmtClass: break;
#define STMT(Node, Parent) case Stmt::Node##Class: break;
#define EXPR(Node, Parent)                                              \
    case Stmt::Node##Class: return getDerived().Transform##Node(cast<Node>(E));
#include "clang/AST/StmtNodes.def"
  }

  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
NestedNameSpecifier *
TreeTransform<Derived>::TransformNestedNameSpecifier(NestedNameSpecifier *NNS,
                                                     SourceRange Range,
                                                     QualType ObjectType,
                                             NamedDecl *FirstQualifierInScope) {
  if (!NNS)
    return 0;

  // Transform the prefix of this nested name specifier.
  NestedNameSpecifier *Prefix = NNS->getPrefix();
  if (Prefix) {
    Prefix = getDerived().TransformNestedNameSpecifier(Prefix, Range,
                                                       ObjectType,
                                                       FirstQualifierInScope);
    if (!Prefix)
      return 0;

    // Clear out the object type and the first qualifier in scope; they only
    // apply to the first element in the nested-name-specifier.
    ObjectType = QualType();
    FirstQualifierInScope = 0;
  }

  switch (NNS->getKind()) {
  case NestedNameSpecifier::Identifier:
    assert((Prefix || !ObjectType.isNull()) &&
            "Identifier nested-name-specifier with no prefix or object type");
    if (!getDerived().AlwaysRebuild() && Prefix == NNS->getPrefix() &&
        ObjectType.isNull())
      return NNS;

    return getDerived().RebuildNestedNameSpecifier(Prefix, Range,
                                                   *NNS->getAsIdentifier(),
                                                   ObjectType,
                                                   FirstQualifierInScope);

  case NestedNameSpecifier::Namespace: {
    NamespaceDecl *NS
      = cast_or_null<NamespaceDecl>(
                            getDerived().TransformDecl(NNS->getAsNamespace()));
    if (!getDerived().AlwaysRebuild() &&
        Prefix == NNS->getPrefix() &&
        NS == NNS->getAsNamespace())
      return NNS;

    return getDerived().RebuildNestedNameSpecifier(Prefix, Range, NS);
  }

  case NestedNameSpecifier::Global:
    // There is no meaningful transformation that one could perform on the
    // global scope.
    return NNS;

  case NestedNameSpecifier::TypeSpecWithTemplate:
  case NestedNameSpecifier::TypeSpec: {
    TemporaryBase Rebase(*this, Range.getBegin(), DeclarationName());
    QualType T = getDerived().TransformType(QualType(NNS->getAsType(), 0));
    if (T.isNull())
      return 0;

    if (!getDerived().AlwaysRebuild() &&
        Prefix == NNS->getPrefix() &&
        T == QualType(NNS->getAsType(), 0))
      return NNS;

    return getDerived().RebuildNestedNameSpecifier(Prefix, Range,
                  NNS->getKind() == NestedNameSpecifier::TypeSpecWithTemplate,
                                                   T);
  }
  }

  // Required to silence a GCC warning
  return 0;
}

template<typename Derived>
DeclarationName
TreeTransform<Derived>::TransformDeclarationName(DeclarationName Name,
                                                 SourceLocation Loc,
                                                 QualType ObjectType) {
  if (!Name)
    return Name;

  switch (Name.getNameKind()) {
  case DeclarationName::Identifier:
  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
  case DeclarationName::CXXOperatorName:
  case DeclarationName::CXXUsingDirective:
    return Name;

  case DeclarationName::CXXConstructorName:
  case DeclarationName::CXXDestructorName:
  case DeclarationName::CXXConversionFunctionName: {
    TemporaryBase Rebase(*this, Loc, Name);
    QualType T;
    if (!ObjectType.isNull() && 
        isa<TemplateSpecializationType>(Name.getCXXNameType())) {
      TemplateSpecializationType *SpecType
        = cast<TemplateSpecializationType>(Name.getCXXNameType());
      T = TransformTemplateSpecializationType(SpecType, ObjectType);
    } else
      T = getDerived().TransformType(Name.getCXXNameType());
    if (T.isNull())
      return DeclarationName();

    return SemaRef.Context.DeclarationNames.getCXXSpecialName(
                                                           Name.getNameKind(),
                                          SemaRef.Context.getCanonicalType(T));
  }
  }

  return DeclarationName();
}

template<typename Derived>
TemplateName
TreeTransform<Derived>::TransformTemplateName(TemplateName Name,
                                              QualType ObjectType) {
  if (QualifiedTemplateName *QTN = Name.getAsQualifiedTemplateName()) {
    NestedNameSpecifier *NNS
      = getDerived().TransformNestedNameSpecifier(QTN->getQualifier(),
                      /*FIXME:*/SourceRange(getDerived().getBaseLocation()));
    if (!NNS)
      return TemplateName();

    if (TemplateDecl *Template = QTN->getTemplateDecl()) {
      TemplateDecl *TransTemplate
        = cast_or_null<TemplateDecl>(getDerived().TransformDecl(Template));
      if (!TransTemplate)
        return TemplateName();

      if (!getDerived().AlwaysRebuild() &&
          NNS == QTN->getQualifier() &&
          TransTemplate == Template)
        return Name;

      return getDerived().RebuildTemplateName(NNS, QTN->hasTemplateKeyword(),
                                              TransTemplate);
    }

    OverloadedFunctionDecl *Ovl = QTN->getOverloadedFunctionDecl();
    assert(Ovl && "Not a template name or an overload set?");
    OverloadedFunctionDecl *TransOvl
      = cast_or_null<OverloadedFunctionDecl>(getDerived().TransformDecl(Ovl));
    if (!TransOvl)
      return TemplateName();

    if (!getDerived().AlwaysRebuild() &&
        NNS == QTN->getQualifier() &&
        TransOvl == Ovl)
      return Name;

    return getDerived().RebuildTemplateName(NNS, QTN->hasTemplateKeyword(),
                                            TransOvl);
  }

  if (DependentTemplateName *DTN = Name.getAsDependentTemplateName()) {
    NestedNameSpecifier *NNS
      = getDerived().TransformNestedNameSpecifier(DTN->getQualifier(),
                        /*FIXME:*/SourceRange(getDerived().getBaseLocation()));
    if (!NNS && DTN->getQualifier())
      return TemplateName();

    if (!getDerived().AlwaysRebuild() &&
        NNS == DTN->getQualifier() &&
        ObjectType.isNull())
      return Name;

    return getDerived().RebuildTemplateName(NNS, *DTN->getName(), ObjectType);
  }

  if (TemplateDecl *Template = Name.getAsTemplateDecl()) {
    TemplateDecl *TransTemplate
      = cast_or_null<TemplateDecl>(getDerived().TransformDecl(Template));
    if (!TransTemplate)
      return TemplateName();

    if (!getDerived().AlwaysRebuild() &&
        TransTemplate == Template)
      return Name;

    return TemplateName(TransTemplate);
  }

  OverloadedFunctionDecl *Ovl = Name.getAsOverloadedFunctionDecl();
  assert(Ovl && "Not a template name or an overload set?");
  OverloadedFunctionDecl *TransOvl
    = cast_or_null<OverloadedFunctionDecl>(getDerived().TransformDecl(Ovl));
  if (!TransOvl)
    return TemplateName();

  if (!getDerived().AlwaysRebuild() &&
      TransOvl == Ovl)
    return Name;

  return TemplateName(TransOvl);
}

template<typename Derived>
void TreeTransform<Derived>::InventTemplateArgumentLoc(
                                         const TemplateArgument &Arg,
                                         TemplateArgumentLoc &Output) {
  SourceLocation Loc = getDerived().getBaseLocation();
  switch (Arg.getKind()) {
  case TemplateArgument::Null:
    llvm::llvm_unreachable("null template argument in TreeTransform");
    break;

  case TemplateArgument::Type:
    Output = TemplateArgumentLoc(Arg,
               SemaRef.Context.getTrivialDeclaratorInfo(Arg.getAsType(), Loc));
                                            
    break;

  case TemplateArgument::Expression:
    Output = TemplateArgumentLoc(Arg, Arg.getAsExpr());
    break;

  case TemplateArgument::Declaration:
  case TemplateArgument::Integral:
  case TemplateArgument::Pack:
    Output = TemplateArgumentLoc(Arg, TemplateArgumentLocInfo());
    break;
  }
}

template<typename Derived>
bool TreeTransform<Derived>::TransformTemplateArgument(
                                         const TemplateArgumentLoc &Input,
                                         TemplateArgumentLoc &Output) {
  const TemplateArgument &Arg = Input.getArgument();
  switch (Arg.getKind()) {
  case TemplateArgument::Null:
  case TemplateArgument::Integral:
    Output = Input;
    return false;

  case TemplateArgument::Type: {
    DeclaratorInfo *DI = Input.getSourceDeclaratorInfo();
    if (DI == NULL)
      DI = InventDeclaratorInfo(Input.getArgument().getAsType());

    DI = getDerived().TransformType(DI);
    if (!DI) return true;

    Output = TemplateArgumentLoc(TemplateArgument(DI->getType()), DI);
    return false;
  }

  case TemplateArgument::Declaration: {
    // FIXME: we should never have to transform one of these.
    DeclarationName Name;
    if (NamedDecl *ND = dyn_cast<NamedDecl>(Arg.getAsDecl()))
      Name = ND->getDeclName();
    TemporaryBase Rebase(*this, SourceLocation(), Name);
    Decl *D = getDerived().TransformDecl(Arg.getAsDecl());
    if (!D) return true;

    Expr *SourceExpr = Input.getSourceDeclExpression();
    if (SourceExpr) {
      EnterExpressionEvaluationContext Unevaluated(getSema(),
                                                   Action::Unevaluated);
      Sema::OwningExprResult E = getDerived().TransformExpr(SourceExpr);
      if (E.isInvalid())
        SourceExpr = NULL;
      else {
        SourceExpr = E.takeAs<Expr>();
        SourceExpr->Retain();
      }
    }

    Output = TemplateArgumentLoc(TemplateArgument(D), SourceExpr);
    return false;
  }

  case TemplateArgument::Expression: {
    // Template argument expressions are not potentially evaluated.
    EnterExpressionEvaluationContext Unevaluated(getSema(),
                                                 Action::Unevaluated);

    Expr *InputExpr = Input.getSourceExpression();
    if (!InputExpr) InputExpr = Input.getArgument().getAsExpr();

    Sema::OwningExprResult E
      = getDerived().TransformExpr(InputExpr);
    if (E.isInvalid()) return true;

    Expr *ETaken = E.takeAs<Expr>();
    ETaken->Retain();
    Output = TemplateArgumentLoc(TemplateArgument(ETaken), ETaken);
    return false;
  }

  case TemplateArgument::Pack: {
    llvm::SmallVector<TemplateArgument, 4> TransformedArgs;
    TransformedArgs.reserve(Arg.pack_size());
    for (TemplateArgument::pack_iterator A = Arg.pack_begin(),
                                      AEnd = Arg.pack_end();
         A != AEnd; ++A) {

      // FIXME: preserve source information here when we start
      // caring about parameter packs.

      TemplateArgumentLoc InputArg;
      TemplateArgumentLoc OutputArg;
      getDerived().InventTemplateArgumentLoc(*A, InputArg);
      if (getDerived().TransformTemplateArgument(InputArg, OutputArg))
        return true;

      TransformedArgs.push_back(OutputArg.getArgument());
    }
    TemplateArgument Result;
    Result.setArgumentPack(TransformedArgs.data(), TransformedArgs.size(),
                           true);
    Output = TemplateArgumentLoc(Result, Input.getLocInfo());
    return false;
  }
  }

  // Work around bogus GCC warning
  return true;
}

//===----------------------------------------------------------------------===//
// Type transformation
//===----------------------------------------------------------------------===//

template<typename Derived>
QualType TreeTransform<Derived>::TransformType(QualType T) {
  if (getDerived().AlreadyTransformed(T))
    return T;

  // Temporary workaround.  All of these transformations should
  // eventually turn into transformations on TypeLocs.
  DeclaratorInfo *DI = getSema().Context.CreateDeclaratorInfo(T);
  DI->getTypeLoc().initialize(getDerived().getBaseLocation());
  
  DeclaratorInfo *NewDI = getDerived().TransformType(DI);

  if (!NewDI)
    return QualType();

  return NewDI->getType();
}

template<typename Derived>
DeclaratorInfo *TreeTransform<Derived>::TransformType(DeclaratorInfo *DI) {
  if (getDerived().AlreadyTransformed(DI->getType()))
    return DI;

  TypeLocBuilder TLB;

  TypeLoc TL = DI->getTypeLoc();
  TLB.reserve(TL.getFullDataSize());

  QualType Result = getDerived().TransformType(TLB, TL);
  if (Result.isNull())
    return 0;

  return TLB.getDeclaratorInfo(SemaRef.Context, Result);
}

template<typename Derived>
QualType
TreeTransform<Derived>::TransformType(TypeLocBuilder &TLB, TypeLoc T) {
  switch (T.getTypeLocClass()) {
#define ABSTRACT_TYPELOC(CLASS, PARENT)
#define TYPELOC(CLASS, PARENT) \
  case TypeLoc::CLASS: \
    return getDerived().Transform##CLASS##Type(TLB, cast<CLASS##TypeLoc>(T));
#include "clang/AST/TypeLocNodes.def"
  }

  llvm::llvm_unreachable("unhandled type loc!");
  return QualType();
}

/// FIXME: By default, this routine adds type qualifiers only to types
/// that can have qualifiers, and silently suppresses those qualifiers
/// that are not permitted (e.g., qualifiers on reference or function
/// types). This is the right thing for template instantiation, but
/// probably not for other clients.
template<typename Derived>
QualType
TreeTransform<Derived>::TransformQualifiedType(TypeLocBuilder &TLB,
                                               QualifiedTypeLoc T) {
  Qualifiers Quals = T.getType().getQualifiers();

  QualType Result = getDerived().TransformType(TLB, T.getUnqualifiedLoc());
  if (Result.isNull())
    return QualType();

  // Silently suppress qualifiers if the result type can't be qualified.
  // FIXME: this is the right thing for template instantiation, but
  // probably not for other clients.
  if (Result->isFunctionType() || Result->isReferenceType())
    return Result;

  Result = SemaRef.Context.getQualifiedType(Result, Quals);

  TLB.push<QualifiedTypeLoc>(Result);

  // No location information to preserve.

  return Result;
}

template <class TyLoc> static inline
QualType TransformTypeSpecType(TypeLocBuilder &TLB, TyLoc T) {
  TyLoc NewT = TLB.push<TyLoc>(T.getType());
  NewT.setNameLoc(T.getNameLoc());
  return T.getType();
}

// Ugly metaprogramming macros because I couldn't be bothered to make
// the equivalent template version work.
#define TransformPointerLikeType(TypeClass) do { \
  QualType PointeeType                                       \
    = getDerived().TransformType(TLB, TL.getPointeeLoc());   \
  if (PointeeType.isNull())                                  \
    return QualType();                                       \
                                                             \
  QualType Result = TL.getType();                            \
  if (getDerived().AlwaysRebuild() ||                        \
      PointeeType != TL.getPointeeLoc().getType()) {         \
    Result = getDerived().Rebuild##TypeClass(PointeeType,    \
                                          TL.getSigilLoc()); \
    if (Result.isNull())                                     \
      return QualType();                                     \
  }                                                          \
                                                             \
  TypeClass##Loc NewT = TLB.push<TypeClass##Loc>(Result);    \
  NewT.setSigilLoc(TL.getSigilLoc());                        \
                                                             \
  return Result;                                             \
} while(0)

template<typename Derived>
QualType TreeTransform<Derived>::TransformBuiltinType(TypeLocBuilder &TLB,
                                                      BuiltinTypeLoc T) {
  return TransformTypeSpecType(TLB, T);
}

template<typename Derived>
QualType
TreeTransform<Derived>::TransformFixedWidthIntType(TypeLocBuilder &TLB,
                                                   FixedWidthIntTypeLoc T) {
  return TransformTypeSpecType(TLB, T);
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformComplexType(TypeLocBuilder &TLB,
                                                      ComplexTypeLoc T) {
  // FIXME: recurse?
  return TransformTypeSpecType(TLB, T);
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformPointerType(TypeLocBuilder &TLB,
                                                      PointerTypeLoc TL) {
  TransformPointerLikeType(PointerType);
}

template<typename Derived>
QualType
TreeTransform<Derived>::TransformBlockPointerType(TypeLocBuilder &TLB,
                                                  BlockPointerTypeLoc TL) {
  TransformPointerLikeType(BlockPointerType);
}

/// Transforms a reference type.  Note that somewhat paradoxically we
/// don't care whether the type itself is an l-value type or an r-value
/// type;  we only care if the type was *written* as an l-value type
/// or an r-value type.
template<typename Derived>
QualType
TreeTransform<Derived>::TransformReferenceType(TypeLocBuilder &TLB,
                                               ReferenceTypeLoc TL) {
  const ReferenceType *T = TL.getTypePtr();

  // Note that this works with the pointee-as-written.
  QualType PointeeType = getDerived().TransformType(TLB, TL.getPointeeLoc());
  if (PointeeType.isNull())
    return QualType();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      PointeeType != T->getPointeeTypeAsWritten()) {
    Result = getDerived().RebuildReferenceType(PointeeType,
                                               T->isSpelledAsLValue(),
                                               TL.getSigilLoc());
    if (Result.isNull())
      return QualType();
  }

  // r-value references can be rebuilt as l-value references.
  ReferenceTypeLoc NewTL;
  if (isa<LValueReferenceType>(Result))
    NewTL = TLB.push<LValueReferenceTypeLoc>(Result);
  else
    NewTL = TLB.push<RValueReferenceTypeLoc>(Result);
  NewTL.setSigilLoc(TL.getSigilLoc());

  return Result;
}

template<typename Derived>
QualType
TreeTransform<Derived>::TransformLValueReferenceType(TypeLocBuilder &TLB,
                                                 LValueReferenceTypeLoc TL) {
  return TransformReferenceType(TLB, TL);
}

template<typename Derived>
QualType
TreeTransform<Derived>::TransformRValueReferenceType(TypeLocBuilder &TLB,
                                                 RValueReferenceTypeLoc TL) {
  return TransformReferenceType(TLB, TL);
}

template<typename Derived>
QualType
TreeTransform<Derived>::TransformMemberPointerType(TypeLocBuilder &TLB,
                                                   MemberPointerTypeLoc TL) {
  MemberPointerType *T = TL.getTypePtr();

  QualType PointeeType = getDerived().TransformType(TLB, TL.getPointeeLoc());
  if (PointeeType.isNull())
    return QualType();

  // TODO: preserve source information for this.
  QualType ClassType
    = getDerived().TransformType(QualType(T->getClass(), 0));
  if (ClassType.isNull())
    return QualType();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      PointeeType != T->getPointeeType() ||
      ClassType != QualType(T->getClass(), 0)) {
    Result = getDerived().RebuildMemberPointerType(PointeeType, ClassType,
                                                   TL.getStarLoc());
    if (Result.isNull())
      return QualType();
  }

  MemberPointerTypeLoc NewTL = TLB.push<MemberPointerTypeLoc>(Result);
  NewTL.setSigilLoc(TL.getSigilLoc());

  return Result;
}

template<typename Derived>
QualType
TreeTransform<Derived>::TransformConstantArrayType(TypeLocBuilder &TLB,
                                                   ConstantArrayTypeLoc TL) {
  ConstantArrayType *T = TL.getTypePtr();
  QualType ElementType = getDerived().TransformType(TLB, TL.getElementLoc());
  if (ElementType.isNull())
    return QualType();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      ElementType != T->getElementType()) {
    Result = getDerived().RebuildConstantArrayType(ElementType,
                                                   T->getSizeModifier(),
                                                   T->getSize(),
                                             T->getIndexTypeCVRQualifiers(),
                                                   TL.getBracketsRange());
    if (Result.isNull())
      return QualType();
  }
  
  ConstantArrayTypeLoc NewTL = TLB.push<ConstantArrayTypeLoc>(Result);
  NewTL.setLBracketLoc(TL.getLBracketLoc());
  NewTL.setRBracketLoc(TL.getRBracketLoc());

  Expr *Size = TL.getSizeExpr();
  if (Size) {
    EnterExpressionEvaluationContext Unevaluated(SemaRef, Action::Unevaluated);
    Size = getDerived().TransformExpr(Size).template takeAs<Expr>();
  }
  NewTL.setSizeExpr(Size);

  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformIncompleteArrayType(
                                              TypeLocBuilder &TLB,
                                              IncompleteArrayTypeLoc TL) {
  IncompleteArrayType *T = TL.getTypePtr();
  QualType ElementType = getDerived().TransformType(TLB, TL.getElementLoc());
  if (ElementType.isNull())
    return QualType();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      ElementType != T->getElementType()) {
    Result = getDerived().RebuildIncompleteArrayType(ElementType,
                                                     T->getSizeModifier(),
                                           T->getIndexTypeCVRQualifiers(),
                                                     TL.getBracketsRange());
    if (Result.isNull())
      return QualType();
  }
  
  IncompleteArrayTypeLoc NewTL = TLB.push<IncompleteArrayTypeLoc>(Result);
  NewTL.setLBracketLoc(TL.getLBracketLoc());
  NewTL.setRBracketLoc(TL.getRBracketLoc());
  NewTL.setSizeExpr(0);

  return Result;
}

template<typename Derived>
QualType
TreeTransform<Derived>::TransformVariableArrayType(TypeLocBuilder &TLB,
                                                   VariableArrayTypeLoc TL) {
  VariableArrayType *T = TL.getTypePtr();
  QualType ElementType = getDerived().TransformType(TLB, TL.getElementLoc());
  if (ElementType.isNull())
    return QualType();

  // Array bounds are not potentially evaluated contexts
  EnterExpressionEvaluationContext Unevaluated(SemaRef, Action::Unevaluated);

  Sema::OwningExprResult SizeResult
    = getDerived().TransformExpr(T->getSizeExpr());
  if (SizeResult.isInvalid())
    return QualType();

  Expr *Size = static_cast<Expr*>(SizeResult.get());

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      ElementType != T->getElementType() ||
      Size != T->getSizeExpr()) {
    Result = getDerived().RebuildVariableArrayType(ElementType,
                                                   T->getSizeModifier(),
                                                   move(SizeResult),
                                             T->getIndexTypeCVRQualifiers(),
                                                   TL.getBracketsRange());
    if (Result.isNull())
      return QualType();
  }
  else SizeResult.take();
  
  VariableArrayTypeLoc NewTL = TLB.push<VariableArrayTypeLoc>(Result);
  NewTL.setLBracketLoc(TL.getLBracketLoc());
  NewTL.setRBracketLoc(TL.getRBracketLoc());
  NewTL.setSizeExpr(Size);

  return Result;
}

template<typename Derived>
QualType
TreeTransform<Derived>::TransformDependentSizedArrayType(TypeLocBuilder &TLB,
                                             DependentSizedArrayTypeLoc TL) {
  DependentSizedArrayType *T = TL.getTypePtr();
  QualType ElementType = getDerived().TransformType(TLB, TL.getElementLoc());
  if (ElementType.isNull())
    return QualType();

  // Array bounds are not potentially evaluated contexts
  EnterExpressionEvaluationContext Unevaluated(SemaRef, Action::Unevaluated);

  Sema::OwningExprResult SizeResult
    = getDerived().TransformExpr(T->getSizeExpr());
  if (SizeResult.isInvalid())
    return QualType();

  Expr *Size = static_cast<Expr*>(SizeResult.get());

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      ElementType != T->getElementType() ||
      Size != T->getSizeExpr()) {
    Result = getDerived().RebuildDependentSizedArrayType(ElementType,
                                                         T->getSizeModifier(),
                                                         move(SizeResult),
                                                T->getIndexTypeCVRQualifiers(),
                                                        TL.getBracketsRange());
    if (Result.isNull())
      return QualType();
  }
  else SizeResult.take();

  // We might have any sort of array type now, but fortunately they
  // all have the same location layout.
  ArrayTypeLoc NewTL = TLB.push<ArrayTypeLoc>(Result);
  NewTL.setLBracketLoc(TL.getLBracketLoc());
  NewTL.setRBracketLoc(TL.getRBracketLoc());
  NewTL.setSizeExpr(Size);

  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformDependentSizedExtVectorType(
                                      TypeLocBuilder &TLB,
                                      DependentSizedExtVectorTypeLoc TL) {
  DependentSizedExtVectorType *T = TL.getTypePtr();

  // FIXME: ext vector locs should be nested
  QualType ElementType = getDerived().TransformType(T->getElementType());
  if (ElementType.isNull())
    return QualType();

  // Vector sizes are not potentially evaluated contexts
  EnterExpressionEvaluationContext Unevaluated(SemaRef, Action::Unevaluated);

  Sema::OwningExprResult Size = getDerived().TransformExpr(T->getSizeExpr());
  if (Size.isInvalid())
    return QualType();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      ElementType != T->getElementType() ||
      Size.get() != T->getSizeExpr()) {
    Result = getDerived().RebuildDependentSizedExtVectorType(ElementType,
                                                         move(Size),
                                                         T->getAttributeLoc());
    if (Result.isNull())
      return QualType();
  }
  else Size.take();

  // Result might be dependent or not.
  if (isa<DependentSizedExtVectorType>(Result)) {
    DependentSizedExtVectorTypeLoc NewTL
      = TLB.push<DependentSizedExtVectorTypeLoc>(Result);
    NewTL.setNameLoc(TL.getNameLoc());
  } else {
    ExtVectorTypeLoc NewTL = TLB.push<ExtVectorTypeLoc>(Result);
    NewTL.setNameLoc(TL.getNameLoc());
  }

  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformVectorType(TypeLocBuilder &TLB,
                                                     VectorTypeLoc TL) {
  VectorType *T = TL.getTypePtr();
  QualType ElementType = getDerived().TransformType(T->getElementType());
  if (ElementType.isNull())
    return QualType();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      ElementType != T->getElementType()) {
    Result = getDerived().RebuildVectorType(ElementType, T->getNumElements());
    if (Result.isNull())
      return QualType();
  }
  
  VectorTypeLoc NewTL = TLB.push<VectorTypeLoc>(Result);
  NewTL.setNameLoc(TL.getNameLoc());

  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformExtVectorType(TypeLocBuilder &TLB,
                                                        ExtVectorTypeLoc TL) {
  VectorType *T = TL.getTypePtr();
  QualType ElementType = getDerived().TransformType(T->getElementType());
  if (ElementType.isNull())
    return QualType();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      ElementType != T->getElementType()) {
    Result = getDerived().RebuildExtVectorType(ElementType,
                                               T->getNumElements(),
                                               /*FIXME*/ SourceLocation());
    if (Result.isNull())
      return QualType();
  }
  
  ExtVectorTypeLoc NewTL = TLB.push<ExtVectorTypeLoc>(Result);
  NewTL.setNameLoc(TL.getNameLoc());

  return Result;
}

template<typename Derived>
QualType
TreeTransform<Derived>::TransformFunctionProtoType(TypeLocBuilder &TLB,
                                                   FunctionProtoTypeLoc TL) {
  FunctionProtoType *T = TL.getTypePtr();
  QualType ResultType = getDerived().TransformType(TLB, TL.getResultLoc());
  if (ResultType.isNull())
    return QualType();

  // Transform the parameters.
  llvm::SmallVector<QualType, 4> ParamTypes;
  llvm::SmallVector<ParmVarDecl*, 4> ParamDecls;
  for (unsigned i = 0, e = TL.getNumArgs(); i != e; ++i) {
    ParmVarDecl *OldParm = TL.getArg(i);

    QualType NewType;
    ParmVarDecl *NewParm;

    if (OldParm) {
      DeclaratorInfo *OldDI = OldParm->getDeclaratorInfo();
      assert(OldDI->getType() == T->getArgType(i));

      DeclaratorInfo *NewDI = getDerived().TransformType(OldDI);
      if (!NewDI)
        return QualType();

      if (NewDI == OldDI)
        NewParm = OldParm;
      else
        NewParm = ParmVarDecl::Create(SemaRef.Context,
                                      OldParm->getDeclContext(),
                                      OldParm->getLocation(),
                                      OldParm->getIdentifier(),
                                      NewDI->getType(),
                                      NewDI,
                                      OldParm->getStorageClass(),
                                      /* DefArg */ NULL);
      NewType = NewParm->getType();

    // Deal with the possibility that we don't have a parameter
    // declaration for this parameter.
    } else {
      NewParm = 0;

      QualType OldType = T->getArgType(i);
      NewType = getDerived().TransformType(OldType);
      if (NewType.isNull())
        return QualType();
    }

    ParamTypes.push_back(NewType);
    ParamDecls.push_back(NewParm);
  }

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      ResultType != T->getResultType() ||
      !std::equal(T->arg_type_begin(), T->arg_type_end(), ParamTypes.begin())) {
    Result = getDerived().RebuildFunctionProtoType(ResultType,
                                                   ParamTypes.data(),
                                                   ParamTypes.size(),
                                                   T->isVariadic(),
                                                   T->getTypeQuals());
    if (Result.isNull())
      return QualType();
  }

  FunctionProtoTypeLoc NewTL = TLB.push<FunctionProtoTypeLoc>(Result);
  NewTL.setLParenLoc(TL.getLParenLoc());
  NewTL.setRParenLoc(TL.getRParenLoc());
  for (unsigned i = 0, e = NewTL.getNumArgs(); i != e; ++i)
    NewTL.setArg(i, ParamDecls[i]);

  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformFunctionNoProtoType(
                                                 TypeLocBuilder &TLB,
                                                 FunctionNoProtoTypeLoc TL) {
  FunctionNoProtoType *T = TL.getTypePtr();
  QualType ResultType = getDerived().TransformType(TLB, TL.getResultLoc());
  if (ResultType.isNull())
    return QualType();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      ResultType != T->getResultType())
    Result = getDerived().RebuildFunctionNoProtoType(ResultType);

  FunctionNoProtoTypeLoc NewTL = TLB.push<FunctionNoProtoTypeLoc>(Result);
  NewTL.setLParenLoc(TL.getLParenLoc());
  NewTL.setRParenLoc(TL.getRParenLoc());

  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformTypedefType(TypeLocBuilder &TLB,
                                                      TypedefTypeLoc TL) {
  TypedefType *T = TL.getTypePtr();
  TypedefDecl *Typedef
    = cast_or_null<TypedefDecl>(getDerived().TransformDecl(T->getDecl()));
  if (!Typedef)
    return QualType();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      Typedef != T->getDecl()) {
    Result = getDerived().RebuildTypedefType(Typedef);
    if (Result.isNull())
      return QualType();
  }

  TypedefTypeLoc NewTL = TLB.push<TypedefTypeLoc>(Result);
  NewTL.setNameLoc(TL.getNameLoc());

  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformTypeOfExprType(TypeLocBuilder &TLB,
                                                      TypeOfExprTypeLoc TL) {
  TypeOfExprType *T = TL.getTypePtr();

  // typeof expressions are not potentially evaluated contexts
  EnterExpressionEvaluationContext Unevaluated(SemaRef, Action::Unevaluated);

  Sema::OwningExprResult E = getDerived().TransformExpr(T->getUnderlyingExpr());
  if (E.isInvalid())
    return QualType();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      E.get() != T->getUnderlyingExpr()) {
    Result = getDerived().RebuildTypeOfExprType(move(E));
    if (Result.isNull())
      return QualType();
  }
  else E.take();

  TypeOfExprTypeLoc NewTL = TLB.push<TypeOfExprTypeLoc>(Result);
  NewTL.setNameLoc(TL.getNameLoc());

  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformTypeOfType(TypeLocBuilder &TLB,
                                                     TypeOfTypeLoc TL) {
  TypeOfType *T = TL.getTypePtr();

  // FIXME: should be an inner type, or at least have a DeclaratorInfo.
  QualType Underlying = getDerived().TransformType(T->getUnderlyingType());
  if (Underlying.isNull())
    return QualType();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      Underlying != T->getUnderlyingType()) {
    Result = getDerived().RebuildTypeOfType(Underlying);
    if (Result.isNull())
      return QualType();
  }

  TypeOfTypeLoc NewTL = TLB.push<TypeOfTypeLoc>(Result);
  NewTL.setNameLoc(TL.getNameLoc());

  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformDecltypeType(TypeLocBuilder &TLB,
                                                       DecltypeTypeLoc TL) {
  DecltypeType *T = TL.getTypePtr();

  // decltype expressions are not potentially evaluated contexts
  EnterExpressionEvaluationContext Unevaluated(SemaRef, Action::Unevaluated);

  Sema::OwningExprResult E = getDerived().TransformExpr(T->getUnderlyingExpr());
  if (E.isInvalid())
    return QualType();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      E.get() != T->getUnderlyingExpr()) {
    Result = getDerived().RebuildDecltypeType(move(E));
    if (Result.isNull())
      return QualType();
  }
  else E.take();

  DecltypeTypeLoc NewTL = TLB.push<DecltypeTypeLoc>(Result);
  NewTL.setNameLoc(TL.getNameLoc());

  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformRecordType(TypeLocBuilder &TLB,
                                                     RecordTypeLoc TL) {
  RecordType *T = TL.getTypePtr();
  RecordDecl *Record
    = cast_or_null<RecordDecl>(getDerived().TransformDecl(T->getDecl()));
  if (!Record)
    return QualType();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      Record != T->getDecl()) {
    Result = getDerived().RebuildRecordType(Record);
    if (Result.isNull())
      return QualType();
  }

  RecordTypeLoc NewTL = TLB.push<RecordTypeLoc>(Result);
  NewTL.setNameLoc(TL.getNameLoc());

  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformEnumType(TypeLocBuilder &TLB,
                                                   EnumTypeLoc TL) {
  EnumType *T = TL.getTypePtr();
  EnumDecl *Enum
    = cast_or_null<EnumDecl>(getDerived().TransformDecl(T->getDecl()));
  if (!Enum)
    return QualType();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      Enum != T->getDecl()) {
    Result = getDerived().RebuildEnumType(Enum);
    if (Result.isNull())
      return QualType();
  }

  EnumTypeLoc NewTL = TLB.push<EnumTypeLoc>(Result);
  NewTL.setNameLoc(TL.getNameLoc());

  return Result;
}

template <typename Derived>
QualType TreeTransform<Derived>::TransformElaboratedType(TypeLocBuilder &TLB,
                                                      ElaboratedTypeLoc TL) {
  ElaboratedType *T = TL.getTypePtr();

  // FIXME: this should be a nested type.
  QualType Underlying = getDerived().TransformType(T->getUnderlyingType());
  if (Underlying.isNull())
    return QualType();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      Underlying != T->getUnderlyingType()) {
    Result = getDerived().RebuildElaboratedType(Underlying, T->getTagKind());
    if (Result.isNull())
      return QualType();
  }

  ElaboratedTypeLoc NewTL = TLB.push<ElaboratedTypeLoc>(Result);
  NewTL.setNameLoc(TL.getNameLoc());

  return Result;
}


template<typename Derived>
QualType TreeTransform<Derived>::TransformTemplateTypeParmType(
                                                TypeLocBuilder &TLB,
                                                TemplateTypeParmTypeLoc TL) {
  return TransformTypeSpecType(TLB, TL);
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformSubstTemplateTypeParmType(
                                         TypeLocBuilder &TLB,
                                         SubstTemplateTypeParmTypeLoc TL) {
  return TransformTypeSpecType(TLB, TL);
}

template<typename Derived>
inline QualType 
TreeTransform<Derived>::TransformTemplateSpecializationType(
                                          TypeLocBuilder &TLB,
                                          TemplateSpecializationTypeLoc TL) {
  return TransformTemplateSpecializationType(TLB, TL, QualType());
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformTemplateSpecializationType(
                                      const TemplateSpecializationType *TST,
                                                        QualType ObjectType) {
  // FIXME: this entire method is a temporary workaround; callers
  // should be rewritten to provide real type locs.

  // Fake up a TemplateSpecializationTypeLoc.
  TypeLocBuilder TLB;
  TemplateSpecializationTypeLoc TL
    = TLB.push<TemplateSpecializationTypeLoc>(QualType(TST, 0));

  SourceLocation BaseLoc = getDerived().getBaseLocation();

  TL.setTemplateNameLoc(BaseLoc);
  TL.setLAngleLoc(BaseLoc);
  TL.setRAngleLoc(BaseLoc);
  for (unsigned i = 0, e = TL.getNumArgs(); i != e; ++i) {
    const TemplateArgument &TA = TST->getArg(i);
    TemplateArgumentLoc TAL;
    getDerived().InventTemplateArgumentLoc(TA, TAL);
    TL.setArgLocInfo(i, TAL.getLocInfo());
  }

  TypeLocBuilder IgnoredTLB;
  return TransformTemplateSpecializationType(IgnoredTLB, TL, ObjectType);
}
  
template<typename Derived>
QualType TreeTransform<Derived>::TransformTemplateSpecializationType(
                                                        TypeLocBuilder &TLB,
                                           TemplateSpecializationTypeLoc TL,
                                                        QualType ObjectType) {
  const TemplateSpecializationType *T = TL.getTypePtr();

  TemplateName Template
    = getDerived().TransformTemplateName(T->getTemplateName(), ObjectType);
  if (Template.isNull())
    return QualType();

  llvm::SmallVector<TemplateArgumentLoc, 4> NewTemplateArgs(T->getNumArgs());
  for (unsigned i = 0, e = T->getNumArgs(); i != e; ++i)
    if (getDerived().TransformTemplateArgument(TL.getArgLoc(i),
                                               NewTemplateArgs[i]))
      return QualType();

  // FIXME: maybe don't rebuild if all the template arguments are the same.

  QualType Result =
    getDerived().RebuildTemplateSpecializationType(Template,
                                                   TL.getTemplateNameLoc(),
                                                   TL.getLAngleLoc(),
                                                   NewTemplateArgs.data(),
                                                   NewTemplateArgs.size(),
                                                   TL.getRAngleLoc());

  if (!Result.isNull()) {
    TemplateSpecializationTypeLoc NewTL
      = TLB.push<TemplateSpecializationTypeLoc>(Result);
    NewTL.setTemplateNameLoc(TL.getTemplateNameLoc());
    NewTL.setLAngleLoc(TL.getLAngleLoc());
    NewTL.setRAngleLoc(TL.getRAngleLoc());
    for (unsigned i = 0, e = NewTemplateArgs.size(); i != e; ++i)
      NewTL.setArgLocInfo(i, NewTemplateArgs[i].getLocInfo());
  }

  return Result;
}

template<typename Derived>
QualType
TreeTransform<Derived>::TransformQualifiedNameType(TypeLocBuilder &TLB,
                                                   QualifiedNameTypeLoc TL) {
  QualifiedNameType *T = TL.getTypePtr();
  NestedNameSpecifier *NNS
    = getDerived().TransformNestedNameSpecifier(T->getQualifier(),
                                                SourceRange());
  if (!NNS)
    return QualType();

  QualType Named = getDerived().TransformType(T->getNamedType());
  if (Named.isNull())
    return QualType();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      NNS != T->getQualifier() ||
      Named != T->getNamedType()) {
    Result = getDerived().RebuildQualifiedNameType(NNS, Named);
    if (Result.isNull())
      return QualType();
  }

  QualifiedNameTypeLoc NewTL = TLB.push<QualifiedNameTypeLoc>(Result);
  NewTL.setNameLoc(TL.getNameLoc());

  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformTypenameType(TypeLocBuilder &TLB,
                                                       TypenameTypeLoc TL) {
  TypenameType *T = TL.getTypePtr();

  /* FIXME: preserve source information better than this */
  SourceRange SR(TL.getNameLoc());

  NestedNameSpecifier *NNS
    = getDerived().TransformNestedNameSpecifier(T->getQualifier(), SR);
  if (!NNS)
    return QualType();

  QualType Result;

  if (const TemplateSpecializationType *TemplateId = T->getTemplateId()) {
    QualType NewTemplateId
      = getDerived().TransformType(QualType(TemplateId, 0));
    if (NewTemplateId.isNull())
      return QualType();

    if (!getDerived().AlwaysRebuild() &&
        NNS == T->getQualifier() &&
        NewTemplateId == QualType(TemplateId, 0))
      return QualType(T, 0);

    Result = getDerived().RebuildTypenameType(NNS, NewTemplateId);
  } else {
    Result = getDerived().RebuildTypenameType(NNS, T->getIdentifier(), SR);
  }
  if (Result.isNull())
    return QualType();

  TypenameTypeLoc NewTL = TLB.push<TypenameTypeLoc>(Result);
  NewTL.setNameLoc(TL.getNameLoc());

  return Result;
}

template<typename Derived>
QualType
TreeTransform<Derived>::TransformObjCInterfaceType(TypeLocBuilder &TLB,
                                                   ObjCInterfaceTypeLoc TL) {
  assert(false && "TransformObjCInterfaceType unimplemented");
  return QualType();
}

template<typename Derived>
QualType
TreeTransform<Derived>::TransformObjCObjectPointerType(TypeLocBuilder &TLB,
                                               ObjCObjectPointerTypeLoc TL) {
  assert(false && "TransformObjCObjectPointerType unimplemented");
  return QualType();
}

//===----------------------------------------------------------------------===//
// Statement transformation
//===----------------------------------------------------------------------===//
template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformNullStmt(NullStmt *S) {
  return SemaRef.Owned(S->Retain());
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformCompoundStmt(CompoundStmt *S) {
  return getDerived().TransformCompoundStmt(S, false);
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformCompoundStmt(CompoundStmt *S,
                                              bool IsStmtExpr) {
  bool SubStmtChanged = false;
  ASTOwningVector<&ActionBase::DeleteStmt> Statements(getSema());
  for (CompoundStmt::body_iterator B = S->body_begin(), BEnd = S->body_end();
       B != BEnd; ++B) {
    OwningStmtResult Result = getDerived().TransformStmt(*B);
    if (Result.isInvalid())
      return getSema().StmtError();

    SubStmtChanged = SubStmtChanged || Result.get() != *B;
    Statements.push_back(Result.takeAs<Stmt>());
  }

  if (!getDerived().AlwaysRebuild() &&
      !SubStmtChanged)
    return SemaRef.Owned(S->Retain());

  return getDerived().RebuildCompoundStmt(S->getLBracLoc(),
                                          move_arg(Statements),
                                          S->getRBracLoc(),
                                          IsStmtExpr);
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformCaseStmt(CaseStmt *S) {
  // The case value expressions are not potentially evaluated.
  EnterExpressionEvaluationContext Unevaluated(SemaRef, Action::Unevaluated);

  // Transform the left-hand case value.
  OwningExprResult LHS = getDerived().TransformExpr(S->getLHS());
  if (LHS.isInvalid())
    return SemaRef.StmtError();

  // Transform the right-hand case value (for the GNU case-range extension).
  OwningExprResult RHS = getDerived().TransformExpr(S->getRHS());
  if (RHS.isInvalid())
    return SemaRef.StmtError();

  // Build the case statement.
  // Case statements are always rebuilt so that they will attached to their
  // transformed switch statement.
  OwningStmtResult Case = getDerived().RebuildCaseStmt(S->getCaseLoc(),
                                                       move(LHS),
                                                       S->getEllipsisLoc(),
                                                       move(RHS),
                                                       S->getColonLoc());
  if (Case.isInvalid())
    return SemaRef.StmtError();

  // Transform the statement following the case
  OwningStmtResult SubStmt = getDerived().TransformStmt(S->getSubStmt());
  if (SubStmt.isInvalid())
    return SemaRef.StmtError();

  // Attach the body to the case statement
  return getDerived().RebuildCaseStmtBody(move(Case), move(SubStmt));
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformDefaultStmt(DefaultStmt *S) {
  // Transform the statement following the default case
  OwningStmtResult SubStmt = getDerived().TransformStmt(S->getSubStmt());
  if (SubStmt.isInvalid())
    return SemaRef.StmtError();

  // Default statements are always rebuilt
  return getDerived().RebuildDefaultStmt(S->getDefaultLoc(), S->getColonLoc(),
                                         move(SubStmt));
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformLabelStmt(LabelStmt *S) {
  OwningStmtResult SubStmt = getDerived().TransformStmt(S->getSubStmt());
  if (SubStmt.isInvalid())
    return SemaRef.StmtError();

  // FIXME: Pass the real colon location in.
  SourceLocation ColonLoc = SemaRef.PP.getLocForEndOfToken(S->getIdentLoc());
  return getDerived().RebuildLabelStmt(S->getIdentLoc(), S->getID(), ColonLoc,
                                       move(SubStmt));
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformIfStmt(IfStmt *S) {
  // Transform the condition
  OwningExprResult Cond = getDerived().TransformExpr(S->getCond());
  if (Cond.isInvalid())
    return SemaRef.StmtError();

  Sema::FullExprArg FullCond(getSema().FullExpr(Cond));

  // Transform the "then" branch.
  OwningStmtResult Then = getDerived().TransformStmt(S->getThen());
  if (Then.isInvalid())
    return SemaRef.StmtError();

  // Transform the "else" branch.
  OwningStmtResult Else = getDerived().TransformStmt(S->getElse());
  if (Else.isInvalid())
    return SemaRef.StmtError();

  if (!getDerived().AlwaysRebuild() &&
      FullCond->get() == S->getCond() &&
      Then.get() == S->getThen() &&
      Else.get() == S->getElse())
    return SemaRef.Owned(S->Retain());

  return getDerived().RebuildIfStmt(S->getIfLoc(), FullCond, move(Then),
                                    S->getElseLoc(), move(Else));
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformSwitchStmt(SwitchStmt *S) {
  // Transform the condition.
  OwningExprResult Cond = getDerived().TransformExpr(S->getCond());
  if (Cond.isInvalid())
    return SemaRef.StmtError();

  // Rebuild the switch statement.
  OwningStmtResult Switch = getDerived().RebuildSwitchStmtStart(move(Cond));
  if (Switch.isInvalid())
    return SemaRef.StmtError();

  // Transform the body of the switch statement.
  OwningStmtResult Body = getDerived().TransformStmt(S->getBody());
  if (Body.isInvalid())
    return SemaRef.StmtError();

  // Complete the switch statement.
  return getDerived().RebuildSwitchStmtBody(S->getSwitchLoc(), move(Switch),
                                            move(Body));
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformWhileStmt(WhileStmt *S) {
  // Transform the condition
  OwningExprResult Cond = getDerived().TransformExpr(S->getCond());
  if (Cond.isInvalid())
    return SemaRef.StmtError();

  Sema::FullExprArg FullCond(getSema().FullExpr(Cond));

  // Transform the body
  OwningStmtResult Body = getDerived().TransformStmt(S->getBody());
  if (Body.isInvalid())
    return SemaRef.StmtError();

  if (!getDerived().AlwaysRebuild() &&
      FullCond->get() == S->getCond() &&
      Body.get() == S->getBody())
    return SemaRef.Owned(S->Retain());

  return getDerived().RebuildWhileStmt(S->getWhileLoc(), FullCond, move(Body));
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformDoStmt(DoStmt *S) {
  // Transform the condition
  OwningExprResult Cond = getDerived().TransformExpr(S->getCond());
  if (Cond.isInvalid())
    return SemaRef.StmtError();

  // Transform the body
  OwningStmtResult Body = getDerived().TransformStmt(S->getBody());
  if (Body.isInvalid())
    return SemaRef.StmtError();

  if (!getDerived().AlwaysRebuild() &&
      Cond.get() == S->getCond() &&
      Body.get() == S->getBody())
    return SemaRef.Owned(S->Retain());

  return getDerived().RebuildDoStmt(S->getDoLoc(), move(Body), S->getWhileLoc(),
                                    /*FIXME:*/S->getWhileLoc(), move(Cond),
                                    S->getRParenLoc());
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformForStmt(ForStmt *S) {
  // Transform the initialization statement
  OwningStmtResult Init = getDerived().TransformStmt(S->getInit());
  if (Init.isInvalid())
    return SemaRef.StmtError();

  // Transform the condition
  OwningExprResult Cond = getDerived().TransformExpr(S->getCond());
  if (Cond.isInvalid())
    return SemaRef.StmtError();

  // Transform the increment
  OwningExprResult Inc = getDerived().TransformExpr(S->getInc());
  if (Inc.isInvalid())
    return SemaRef.StmtError();

  // Transform the body
  OwningStmtResult Body = getDerived().TransformStmt(S->getBody());
  if (Body.isInvalid())
    return SemaRef.StmtError();

  if (!getDerived().AlwaysRebuild() &&
      Init.get() == S->getInit() &&
      Cond.get() == S->getCond() &&
      Inc.get() == S->getInc() &&
      Body.get() == S->getBody())
    return SemaRef.Owned(S->Retain());

  return getDerived().RebuildForStmt(S->getForLoc(), S->getLParenLoc(),
                                     move(Init), move(Cond), move(Inc),
                                     S->getRParenLoc(), move(Body));
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformGotoStmt(GotoStmt *S) {
  // Goto statements must always be rebuilt, to resolve the label.
  return getDerived().RebuildGotoStmt(S->getGotoLoc(), S->getLabelLoc(),
                                      S->getLabel());
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformIndirectGotoStmt(IndirectGotoStmt *S) {
  OwningExprResult Target = getDerived().TransformExpr(S->getTarget());
  if (Target.isInvalid())
    return SemaRef.StmtError();

  if (!getDerived().AlwaysRebuild() &&
      Target.get() == S->getTarget())
    return SemaRef.Owned(S->Retain());

  return getDerived().RebuildIndirectGotoStmt(S->getGotoLoc(), S->getStarLoc(),
                                              move(Target));
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformContinueStmt(ContinueStmt *S) {
  return SemaRef.Owned(S->Retain());
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformBreakStmt(BreakStmt *S) {
  return SemaRef.Owned(S->Retain());
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformReturnStmt(ReturnStmt *S) {
  Sema::OwningExprResult Result = getDerived().TransformExpr(S->getRetValue());
  if (Result.isInvalid())
    return SemaRef.StmtError();

  // FIXME: We always rebuild the return statement because there is no way
  // to tell whether the return type of the function has changed.
  return getDerived().RebuildReturnStmt(S->getReturnLoc(), move(Result));
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformDeclStmt(DeclStmt *S) {
  bool DeclChanged = false;
  llvm::SmallVector<Decl *, 4> Decls;
  for (DeclStmt::decl_iterator D = S->decl_begin(), DEnd = S->decl_end();
       D != DEnd; ++D) {
    Decl *Transformed = getDerived().TransformDefinition(*D);
    if (!Transformed)
      return SemaRef.StmtError();

    if (Transformed != *D)
      DeclChanged = true;

    Decls.push_back(Transformed);
  }

  if (!getDerived().AlwaysRebuild() && !DeclChanged)
    return SemaRef.Owned(S->Retain());

  return getDerived().RebuildDeclStmt(Decls.data(), Decls.size(),
                                      S->getStartLoc(), S->getEndLoc());
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformSwitchCase(SwitchCase *S) {
  assert(false && "SwitchCase is abstract and cannot be transformed");
  return SemaRef.Owned(S->Retain());
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformAsmStmt(AsmStmt *S) {
  // FIXME: Implement!
  assert(false && "Inline assembly cannot be transformed");
  return SemaRef.Owned(S->Retain());
}


template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformObjCAtTryStmt(ObjCAtTryStmt *S) {
  // FIXME: Implement this
  assert(false && "Cannot transform an Objective-C @try statement");
  return SemaRef.Owned(S->Retain());
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformObjCAtCatchStmt(ObjCAtCatchStmt *S) {
  // FIXME: Implement this
  assert(false && "Cannot transform an Objective-C @catch statement");
  return SemaRef.Owned(S->Retain());
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformObjCAtFinallyStmt(ObjCAtFinallyStmt *S) {
  // FIXME: Implement this
  assert(false && "Cannot transform an Objective-C @finally statement");
  return SemaRef.Owned(S->Retain());
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformObjCAtThrowStmt(ObjCAtThrowStmt *S) {
  // FIXME: Implement this
  assert(false && "Cannot transform an Objective-C @throw statement");
  return SemaRef.Owned(S->Retain());
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformObjCAtSynchronizedStmt(
                                                  ObjCAtSynchronizedStmt *S) {
  // FIXME: Implement this
  assert(false && "Cannot transform an Objective-C @synchronized statement");
  return SemaRef.Owned(S->Retain());
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformObjCForCollectionStmt(
                                                  ObjCForCollectionStmt *S) {
  // FIXME: Implement this
  assert(false && "Cannot transform an Objective-C for-each statement");
  return SemaRef.Owned(S->Retain());
}


template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformCXXCatchStmt(CXXCatchStmt *S) {
  // Transform the exception declaration, if any.
  VarDecl *Var = 0;
  if (S->getExceptionDecl()) {
    VarDecl *ExceptionDecl = S->getExceptionDecl();
    TemporaryBase Rebase(*this, ExceptionDecl->getLocation(),
                         ExceptionDecl->getDeclName());

    QualType T = getDerived().TransformType(ExceptionDecl->getType());
    if (T.isNull())
      return SemaRef.StmtError();

    Var = getDerived().RebuildExceptionDecl(ExceptionDecl,
                                            T,
                                            ExceptionDecl->getDeclaratorInfo(),
                                            ExceptionDecl->getIdentifier(),
                                            ExceptionDecl->getLocation(),
                                            /*FIXME: Inaccurate*/
                                    SourceRange(ExceptionDecl->getLocation()));
    if (!Var || Var->isInvalidDecl()) {
      if (Var)
        Var->Destroy(SemaRef.Context);
      return SemaRef.StmtError();
    }
  }

  // Transform the actual exception handler.
  OwningStmtResult Handler = getDerived().TransformStmt(S->getHandlerBlock());
  if (Handler.isInvalid()) {
    if (Var)
      Var->Destroy(SemaRef.Context);
    return SemaRef.StmtError();
  }

  if (!getDerived().AlwaysRebuild() &&
      !Var &&
      Handler.get() == S->getHandlerBlock())
    return SemaRef.Owned(S->Retain());

  return getDerived().RebuildCXXCatchStmt(S->getCatchLoc(),
                                          Var,
                                          move(Handler));
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformCXXTryStmt(CXXTryStmt *S) {
  // Transform the try block itself.
  OwningStmtResult TryBlock
    = getDerived().TransformCompoundStmt(S->getTryBlock());
  if (TryBlock.isInvalid())
    return SemaRef.StmtError();

  // Transform the handlers.
  bool HandlerChanged = false;
  ASTOwningVector<&ActionBase::DeleteStmt> Handlers(SemaRef);
  for (unsigned I = 0, N = S->getNumHandlers(); I != N; ++I) {
    OwningStmtResult Handler
      = getDerived().TransformCXXCatchStmt(S->getHandler(I));
    if (Handler.isInvalid())
      return SemaRef.StmtError();

    HandlerChanged = HandlerChanged || Handler.get() != S->getHandler(I);
    Handlers.push_back(Handler.takeAs<Stmt>());
  }

  if (!getDerived().AlwaysRebuild() &&
      TryBlock.get() == S->getTryBlock() &&
      !HandlerChanged)
    return SemaRef.Owned(S->Retain());

  return getDerived().RebuildCXXTryStmt(S->getTryLoc(), move(TryBlock),
                                        move_arg(Handlers));
}

//===----------------------------------------------------------------------===//
// Expression transformation
//===----------------------------------------------------------------------===//
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformPredefinedExpr(PredefinedExpr *E) {
  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformDeclRefExpr(DeclRefExpr *E) {
  NestedNameSpecifier *Qualifier = 0;
  if (E->getQualifier()) {
    Qualifier = getDerived().TransformNestedNameSpecifier(E->getQualifier(),
                                                       E->getQualifierRange());
    if (!Qualifier)
      return SemaRef.ExprError();
  }
  
  NamedDecl *ND
    = dyn_cast_or_null<NamedDecl>(getDerived().TransformDecl(E->getDecl()));
  if (!ND)
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() && 
      Qualifier == E->getQualifier() &&
      ND == E->getDecl() &&
      !E->hasExplicitTemplateArgumentList())
    return SemaRef.Owned(E->Retain());

  // FIXME: We're losing the explicit template arguments in this transformation.

  llvm::SmallVector<TemplateArgumentLoc, 4> TransArgs(E->getNumTemplateArgs());
  for (unsigned I = 0, N = E->getNumTemplateArgs(); I != N; ++I) {
    if (getDerived().TransformTemplateArgument(E->getTemplateArgs()[I],
                                               TransArgs[I]))
      return SemaRef.ExprError();
  }
  
  // FIXME: Pass the qualifier/qualifier range along.
  return getDerived().RebuildDeclRefExpr(Qualifier, E->getQualifierRange(),
                                         ND, E->getLocation());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformIntegerLiteral(IntegerLiteral *E) {
  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformFloatingLiteral(FloatingLiteral *E) {
  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformImaginaryLiteral(ImaginaryLiteral *E) {
  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformStringLiteral(StringLiteral *E) {
  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCharacterLiteral(CharacterLiteral *E) {
  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformParenExpr(ParenExpr *E) {
  OwningExprResult SubExpr = getDerived().TransformExpr(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() && SubExpr.get() == E->getSubExpr())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildParenExpr(move(SubExpr), E->getLParen(),
                                       E->getRParen());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformUnaryOperator(UnaryOperator *E) {
  OwningExprResult SubExpr = getDerived().TransformExpr(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() && SubExpr.get() == E->getSubExpr())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildUnaryOperator(E->getOperatorLoc(),
                                           E->getOpcode(),
                                           move(SubExpr));
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformSizeOfAlignOfExpr(SizeOfAlignOfExpr *E) {
  if (E->isArgumentType()) {
    TemporaryBase Rebase(*this, E->getOperatorLoc(), DeclarationName());

    QualType T = getDerived().TransformType(E->getArgumentType());
    if (T.isNull())
      return SemaRef.ExprError();

    if (!getDerived().AlwaysRebuild() && T == E->getArgumentType())
      return SemaRef.Owned(E->Retain());

    return getDerived().RebuildSizeOfAlignOf(T, E->getOperatorLoc(),
                                             E->isSizeOf(),
                                             E->getSourceRange());
  }

  Sema::OwningExprResult SubExpr(SemaRef);
  {
    // C++0x [expr.sizeof]p1:
    //   The operand is either an expression, which is an unevaluated operand
    //   [...]
    EnterExpressionEvaluationContext Unevaluated(SemaRef, Action::Unevaluated);

    SubExpr = getDerived().TransformExpr(E->getArgumentExpr());
    if (SubExpr.isInvalid())
      return SemaRef.ExprError();

    if (!getDerived().AlwaysRebuild() && SubExpr.get() == E->getArgumentExpr())
      return SemaRef.Owned(E->Retain());
  }

  return getDerived().RebuildSizeOfAlignOf(move(SubExpr), E->getOperatorLoc(),
                                           E->isSizeOf(),
                                           E->getSourceRange());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformArraySubscriptExpr(ArraySubscriptExpr *E) {
  OwningExprResult LHS = getDerived().TransformExpr(E->getLHS());
  if (LHS.isInvalid())
    return SemaRef.ExprError();

  OwningExprResult RHS = getDerived().TransformExpr(E->getRHS());
  if (RHS.isInvalid())
    return SemaRef.ExprError();


  if (!getDerived().AlwaysRebuild() &&
      LHS.get() == E->getLHS() &&
      RHS.get() == E->getRHS())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildArraySubscriptExpr(move(LHS),
                                           /*FIXME:*/E->getLHS()->getLocStart(),
                                                move(RHS),
                                                E->getRBracketLoc());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCallExpr(CallExpr *E) {
  // Transform the callee.
  OwningExprResult Callee = getDerived().TransformExpr(E->getCallee());
  if (Callee.isInvalid())
    return SemaRef.ExprError();

  // Transform arguments.
  bool ArgChanged = false;
  ASTOwningVector<&ActionBase::DeleteExpr> Args(SemaRef);
  llvm::SmallVector<SourceLocation, 4> FakeCommaLocs;
  for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I) {
    OwningExprResult Arg = getDerived().TransformExpr(E->getArg(I));
    if (Arg.isInvalid())
      return SemaRef.ExprError();

    // FIXME: Wrong source location information for the ','.
    FakeCommaLocs.push_back(
       SemaRef.PP.getLocForEndOfToken(E->getArg(I)->getSourceRange().getEnd()));

    ArgChanged = ArgChanged || Arg.get() != E->getArg(I);
    Args.push_back(Arg.takeAs<Expr>());
  }

  if (!getDerived().AlwaysRebuild() &&
      Callee.get() == E->getCallee() &&
      !ArgChanged)
    return SemaRef.Owned(E->Retain());

  // FIXME: Wrong source location information for the '('.
  SourceLocation FakeLParenLoc
    = ((Expr *)Callee.get())->getSourceRange().getBegin();
  return getDerived().RebuildCallExpr(move(Callee), FakeLParenLoc,
                                      move_arg(Args),
                                      FakeCommaLocs.data(),
                                      E->getRParenLoc());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformMemberExpr(MemberExpr *E) {
  OwningExprResult Base = getDerived().TransformExpr(E->getBase());
  if (Base.isInvalid())
    return SemaRef.ExprError();

  NestedNameSpecifier *Qualifier = 0;
  if (E->hasQualifier()) {
    Qualifier
      = getDerived().TransformNestedNameSpecifier(E->getQualifier(),
                                                  E->getQualifierRange());
    if (Qualifier == 0)
      return SemaRef.ExprError();
  }

  NamedDecl *Member
    = cast_or_null<NamedDecl>(getDerived().TransformDecl(E->getMemberDecl()));
  if (!Member)
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      Base.get() == E->getBase() &&
      Qualifier == E->getQualifier() &&
      Member == E->getMemberDecl())
    return SemaRef.Owned(E->Retain());

  // FIXME: Bogus source location for the operator
  SourceLocation FakeOperatorLoc
    = SemaRef.PP.getLocForEndOfToken(E->getBase()->getSourceRange().getEnd());

  return getDerived().RebuildMemberExpr(move(Base), FakeOperatorLoc,
                                        E->isArrow(),
                                        Qualifier,
                                        E->getQualifierRange(),
                                        E->getMemberLoc(),
                                        Member);
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCastExpr(CastExpr *E) {
  assert(false && "Cannot transform abstract class");
  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformBinaryOperator(BinaryOperator *E) {
  OwningExprResult LHS = getDerived().TransformExpr(E->getLHS());
  if (LHS.isInvalid())
    return SemaRef.ExprError();

  OwningExprResult RHS = getDerived().TransformExpr(E->getRHS());
  if (RHS.isInvalid())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      LHS.get() == E->getLHS() &&
      RHS.get() == E->getRHS())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildBinaryOperator(E->getOperatorLoc(), E->getOpcode(),
                                            move(LHS), move(RHS));
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCompoundAssignOperator(
                                                  CompoundAssignOperator *E) {
  return getDerived().TransformBinaryOperator(E);
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformConditionalOperator(ConditionalOperator *E) {
  OwningExprResult Cond = getDerived().TransformExpr(E->getCond());
  if (Cond.isInvalid())
    return SemaRef.ExprError();

  OwningExprResult LHS = getDerived().TransformExpr(E->getLHS());
  if (LHS.isInvalid())
    return SemaRef.ExprError();

  OwningExprResult RHS = getDerived().TransformExpr(E->getRHS());
  if (RHS.isInvalid())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      Cond.get() == E->getCond() &&
      LHS.get() == E->getLHS() &&
      RHS.get() == E->getRHS())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildConditionalOperator(move(Cond),
                                                 E->getQuestionLoc(),
                                                 move(LHS),
                                                 E->getColonLoc(),
                                                 move(RHS));
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformImplicitCastExpr(ImplicitCastExpr *E) {
  TemporaryBase Rebase(*this, E->getLocStart(), DeclarationName());

  // FIXME: Will we ever have type information here? It seems like we won't,
  // so do we even need to transform the type?
  QualType T = getDerived().TransformType(E->getType());
  if (T.isNull())
    return SemaRef.ExprError();

  OwningExprResult SubExpr = getDerived().TransformExpr(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      T == E->getType() &&
      SubExpr.get() == E->getSubExpr())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildImplicitCastExpr(T, E->getCastKind(),
                                              move(SubExpr),
                                              E->isLvalueCast());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformExplicitCastExpr(ExplicitCastExpr *E) {
  assert(false && "Cannot transform abstract class");
  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCStyleCastExpr(CStyleCastExpr *E) {
  QualType T;
  {
    // FIXME: Source location isn't quite accurate.
    SourceLocation TypeStartLoc
      = SemaRef.PP.getLocForEndOfToken(E->getLParenLoc());
    TemporaryBase Rebase(*this, TypeStartLoc, DeclarationName());

    T = getDerived().TransformType(E->getTypeAsWritten());
    if (T.isNull())
      return SemaRef.ExprError();
  }

  OwningExprResult SubExpr = getDerived().TransformExpr(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      T == E->getTypeAsWritten() &&
      SubExpr.get() == E->getSubExpr())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildCStyleCaseExpr(E->getLParenLoc(), T,
                                            E->getRParenLoc(),
                                            move(SubExpr));
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCompoundLiteralExpr(CompoundLiteralExpr *E) {
  QualType T;
  {
    // FIXME: Source location isn't quite accurate.
    SourceLocation FakeTypeLoc
      = SemaRef.PP.getLocForEndOfToken(E->getLParenLoc());
    TemporaryBase Rebase(*this, FakeTypeLoc, DeclarationName());

    T = getDerived().TransformType(E->getType());
    if (T.isNull())
      return SemaRef.ExprError();
  }

  OwningExprResult Init = getDerived().TransformExpr(E->getInitializer());
  if (Init.isInvalid())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      T == E->getType() &&
      Init.get() == E->getInitializer())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildCompoundLiteralExpr(E->getLParenLoc(), T,
                                   /*FIXME:*/E->getInitializer()->getLocEnd(),
                                                 move(Init));
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformExtVectorElementExpr(ExtVectorElementExpr *E) {
  OwningExprResult Base = getDerived().TransformExpr(E->getBase());
  if (Base.isInvalid())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      Base.get() == E->getBase())
    return SemaRef.Owned(E->Retain());

  // FIXME: Bad source location
  SourceLocation FakeOperatorLoc
    = SemaRef.PP.getLocForEndOfToken(E->getBase()->getLocEnd());
  return getDerived().RebuildExtVectorElementExpr(move(Base), FakeOperatorLoc,
                                                  E->getAccessorLoc(),
                                                  E->getAccessor());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformInitListExpr(InitListExpr *E) {
  bool InitChanged = false;

  ASTOwningVector<&ActionBase::DeleteExpr, 4> Inits(SemaRef);
  for (unsigned I = 0, N = E->getNumInits(); I != N; ++I) {
    OwningExprResult Init = getDerived().TransformExpr(E->getInit(I));
    if (Init.isInvalid())
      return SemaRef.ExprError();

    InitChanged = InitChanged || Init.get() != E->getInit(I);
    Inits.push_back(Init.takeAs<Expr>());
  }

  if (!getDerived().AlwaysRebuild() && !InitChanged)
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildInitList(E->getLBraceLoc(), move_arg(Inits),
                                      E->getRBraceLoc());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformDesignatedInitExpr(DesignatedInitExpr *E) {
  Designation Desig;

  // transform the initializer value
  OwningExprResult Init = getDerived().TransformExpr(E->getInit());
  if (Init.isInvalid())
    return SemaRef.ExprError();

  // transform the designators.
  ASTOwningVector<&ActionBase::DeleteExpr, 4> ArrayExprs(SemaRef);
  bool ExprChanged = false;
  for (DesignatedInitExpr::designators_iterator D = E->designators_begin(),
                                             DEnd = E->designators_end();
       D != DEnd; ++D) {
    if (D->isFieldDesignator()) {
      Desig.AddDesignator(Designator::getField(D->getFieldName(),
                                               D->getDotLoc(),
                                               D->getFieldLoc()));
      continue;
    }

    if (D->isArrayDesignator()) {
      OwningExprResult Index = getDerived().TransformExpr(E->getArrayIndex(*D));
      if (Index.isInvalid())
        return SemaRef.ExprError();

      Desig.AddDesignator(Designator::getArray(Index.get(),
                                               D->getLBracketLoc()));

      ExprChanged = ExprChanged || Init.get() != E->getArrayIndex(*D);
      ArrayExprs.push_back(Index.release());
      continue;
    }

    assert(D->isArrayRangeDesignator() && "New kind of designator?");
    OwningExprResult Start
      = getDerived().TransformExpr(E->getArrayRangeStart(*D));
    if (Start.isInvalid())
      return SemaRef.ExprError();

    OwningExprResult End = getDerived().TransformExpr(E->getArrayRangeEnd(*D));
    if (End.isInvalid())
      return SemaRef.ExprError();

    Desig.AddDesignator(Designator::getArrayRange(Start.get(),
                                                  End.get(),
                                                  D->getLBracketLoc(),
                                                  D->getEllipsisLoc()));

    ExprChanged = ExprChanged || Start.get() != E->getArrayRangeStart(*D) ||
      End.get() != E->getArrayRangeEnd(*D);

    ArrayExprs.push_back(Start.release());
    ArrayExprs.push_back(End.release());
  }

  if (!getDerived().AlwaysRebuild() &&
      Init.get() == E->getInit() &&
      !ExprChanged)
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildDesignatedInitExpr(Desig, move_arg(ArrayExprs),
                                                E->getEqualOrColonLoc(),
                                                E->usesGNUSyntax(), move(Init));
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformImplicitValueInitExpr(
                                                    ImplicitValueInitExpr *E) {
  TemporaryBase Rebase(*this, E->getLocStart(), DeclarationName());
  
  // FIXME: Will we ever have proper type location here? Will we actually
  // need to transform the type?
  QualType T = getDerived().TransformType(E->getType());
  if (T.isNull())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      T == E->getType())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildImplicitValueInitExpr(T);
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformVAArgExpr(VAArgExpr *E) {
  // FIXME: Do we want the type as written?
  QualType T;

  {
    // FIXME: Source location isn't quite accurate.
    TemporaryBase Rebase(*this, E->getBuiltinLoc(), DeclarationName());
    T = getDerived().TransformType(E->getType());
    if (T.isNull())
      return SemaRef.ExprError();
  }

  OwningExprResult SubExpr = getDerived().TransformExpr(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      T == E->getType() &&
      SubExpr.get() == E->getSubExpr())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildVAArgExpr(E->getBuiltinLoc(), move(SubExpr),
                                       T, E->getRParenLoc());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformParenListExpr(ParenListExpr *E) {
  bool ArgumentChanged = false;
  ASTOwningVector<&ActionBase::DeleteExpr, 4> Inits(SemaRef);
  for (unsigned I = 0, N = E->getNumExprs(); I != N; ++I) {
    OwningExprResult Init = getDerived().TransformExpr(E->getExpr(I));
    if (Init.isInvalid())
      return SemaRef.ExprError();

    ArgumentChanged = ArgumentChanged || Init.get() != E->getExpr(I);
    Inits.push_back(Init.takeAs<Expr>());
  }

  return getDerived().RebuildParenListExpr(E->getLParenLoc(),
                                           move_arg(Inits),
                                           E->getRParenLoc());
}

/// \brief Transform an address-of-label expression.
///
/// By default, the transformation of an address-of-label expression always
/// rebuilds the expression, so that the label identifier can be resolved to
/// the corresponding label statement by semantic analysis.
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformAddrLabelExpr(AddrLabelExpr *E) {
  return getDerived().RebuildAddrLabelExpr(E->getAmpAmpLoc(), E->getLabelLoc(),
                                           E->getLabel());
}

template<typename Derived>
Sema::OwningExprResult TreeTransform<Derived>::TransformStmtExpr(StmtExpr *E) {
  OwningStmtResult SubStmt
    = getDerived().TransformCompoundStmt(E->getSubStmt(), true);
  if (SubStmt.isInvalid())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      SubStmt.get() == E->getSubStmt())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildStmtExpr(E->getLParenLoc(),
                                      move(SubStmt),
                                      E->getRParenLoc());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformTypesCompatibleExpr(TypesCompatibleExpr *E) {
  QualType T1, T2;
  {
    // FIXME: Source location isn't quite accurate.
    TemporaryBase Rebase(*this, E->getBuiltinLoc(), DeclarationName());

    T1 = getDerived().TransformType(E->getArgType1());
    if (T1.isNull())
      return SemaRef.ExprError();

    T2 = getDerived().TransformType(E->getArgType2());
    if (T2.isNull())
      return SemaRef.ExprError();
  }

  if (!getDerived().AlwaysRebuild() &&
      T1 == E->getArgType1() &&
      T2 == E->getArgType2())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildTypesCompatibleExpr(E->getBuiltinLoc(),
                                                 T1, T2, E->getRParenLoc());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformChooseExpr(ChooseExpr *E) {
  OwningExprResult Cond = getDerived().TransformExpr(E->getCond());
  if (Cond.isInvalid())
    return SemaRef.ExprError();

  OwningExprResult LHS = getDerived().TransformExpr(E->getLHS());
  if (LHS.isInvalid())
    return SemaRef.ExprError();

  OwningExprResult RHS = getDerived().TransformExpr(E->getRHS());
  if (RHS.isInvalid())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      Cond.get() == E->getCond() &&
      LHS.get() == E->getLHS() &&
      RHS.get() == E->getRHS())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildChooseExpr(E->getBuiltinLoc(),
                                        move(Cond), move(LHS), move(RHS),
                                        E->getRParenLoc());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformGNUNullExpr(GNUNullExpr *E) {
  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXOperatorCallExpr(CXXOperatorCallExpr *E) {
  OwningExprResult Callee = getDerived().TransformExpr(E->getCallee());
  if (Callee.isInvalid())
    return SemaRef.ExprError();

  OwningExprResult First = getDerived().TransformExpr(E->getArg(0));
  if (First.isInvalid())
    return SemaRef.ExprError();

  OwningExprResult Second(SemaRef);
  if (E->getNumArgs() == 2) {
    Second = getDerived().TransformExpr(E->getArg(1));
    if (Second.isInvalid())
      return SemaRef.ExprError();
  }

  if (!getDerived().AlwaysRebuild() &&
      Callee.get() == E->getCallee() &&
      First.get() == E->getArg(0) &&
      (E->getNumArgs() != 2 || Second.get() == E->getArg(1)))
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildCXXOperatorCallExpr(E->getOperator(),
                                                 E->getOperatorLoc(),
                                                 move(Callee),
                                                 move(First),
                                                 move(Second));
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXMemberCallExpr(CXXMemberCallExpr *E) {
  return getDerived().TransformCallExpr(E);
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXNamedCastExpr(CXXNamedCastExpr *E) {
  QualType ExplicitTy;
  {
    // FIXME: Source location isn't quite accurate.
    SourceLocation TypeStartLoc
      = SemaRef.PP.getLocForEndOfToken(E->getOperatorLoc());
    TemporaryBase Rebase(*this, TypeStartLoc, DeclarationName());

    ExplicitTy = getDerived().TransformType(E->getTypeAsWritten());
    if (ExplicitTy.isNull())
      return SemaRef.ExprError();
  }

  OwningExprResult SubExpr = getDerived().TransformExpr(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      ExplicitTy == E->getTypeAsWritten() &&
      SubExpr.get() == E->getSubExpr())
    return SemaRef.Owned(E->Retain());

  // FIXME: Poor source location information here.
  SourceLocation FakeLAngleLoc
    = SemaRef.PP.getLocForEndOfToken(E->getOperatorLoc());
  SourceLocation FakeRAngleLoc = E->getSubExpr()->getSourceRange().getBegin();
  SourceLocation FakeRParenLoc
    = SemaRef.PP.getLocForEndOfToken(
                                  E->getSubExpr()->getSourceRange().getEnd());
  return getDerived().RebuildCXXNamedCastExpr(E->getOperatorLoc(),
                                              E->getStmtClass(),
                                              FakeLAngleLoc,
                                              ExplicitTy,
                                              FakeRAngleLoc,
                                              FakeRAngleLoc,
                                              move(SubExpr),
                                              FakeRParenLoc);
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXStaticCastExpr(CXXStaticCastExpr *E) {
  return getDerived().TransformCXXNamedCastExpr(E);
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXDynamicCastExpr(CXXDynamicCastExpr *E) {
  return getDerived().TransformCXXNamedCastExpr(E);
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXReinterpretCastExpr(
                                                CXXReinterpretCastExpr *E) {
  return getDerived().TransformCXXNamedCastExpr(E);
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXConstCastExpr(CXXConstCastExpr *E) {
  return getDerived().TransformCXXNamedCastExpr(E);
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXFunctionalCastExpr(
                                                   CXXFunctionalCastExpr *E) {
  QualType ExplicitTy;
  {
    TemporaryBase Rebase(*this, E->getTypeBeginLoc(), DeclarationName());

    ExplicitTy = getDerived().TransformType(E->getTypeAsWritten());
    if (ExplicitTy.isNull())
      return SemaRef.ExprError();
  }

  OwningExprResult SubExpr = getDerived().TransformExpr(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      ExplicitTy == E->getTypeAsWritten() &&
      SubExpr.get() == E->getSubExpr())
    return SemaRef.Owned(E->Retain());

  // FIXME: The end of the type's source range is wrong
  return getDerived().RebuildCXXFunctionalCastExpr(
                                  /*FIXME:*/SourceRange(E->getTypeBeginLoc()),
                                                   ExplicitTy,
                                      /*FIXME:*/E->getSubExpr()->getLocStart(),
                                                   move(SubExpr),
                                                   E->getRParenLoc());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXTypeidExpr(CXXTypeidExpr *E) {
  if (E->isTypeOperand()) {
    TemporaryBase Rebase(*this, /*FIXME*/E->getLocStart(), DeclarationName());

    QualType T = getDerived().TransformType(E->getTypeOperand());
    if (T.isNull())
      return SemaRef.ExprError();

    if (!getDerived().AlwaysRebuild() &&
        T == E->getTypeOperand())
      return SemaRef.Owned(E->Retain());

    return getDerived().RebuildCXXTypeidExpr(E->getLocStart(),
                                             /*FIXME:*/E->getLocStart(),
                                             T,
                                             E->getLocEnd());
  }

  // We don't know whether the expression is potentially evaluated until
  // after we perform semantic analysis, so the expression is potentially
  // potentially evaluated.
  EnterExpressionEvaluationContext Unevaluated(SemaRef,
                                      Action::PotentiallyPotentiallyEvaluated);

  OwningExprResult SubExpr = getDerived().TransformExpr(E->getExprOperand());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      SubExpr.get() == E->getExprOperand())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildCXXTypeidExpr(E->getLocStart(),
                                           /*FIXME:*/E->getLocStart(),
                                           move(SubExpr),
                                           E->getLocEnd());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXBoolLiteralExpr(CXXBoolLiteralExpr *E) {
  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXNullPtrLiteralExpr(
                                                  CXXNullPtrLiteralExpr *E) {
  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXThisExpr(CXXThisExpr *E) {
  TemporaryBase Rebase(*this, E->getLocStart(), DeclarationName());

  QualType T = getDerived().TransformType(E->getType());
  if (T.isNull())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      T == E->getType())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildCXXThisExpr(E->getLocStart(), T);
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXThrowExpr(CXXThrowExpr *E) {
  OwningExprResult SubExpr = getDerived().TransformExpr(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      SubExpr.get() == E->getSubExpr())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildCXXThrowExpr(E->getThrowLoc(), move(SubExpr));
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXDefaultArgExpr(CXXDefaultArgExpr *E) {
  ParmVarDecl *Param
    = cast_or_null<ParmVarDecl>(getDerived().TransformDecl(E->getParam()));
  if (!Param)
    return SemaRef.ExprError();

  if (getDerived().AlwaysRebuild() &&
      Param == E->getParam())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildCXXDefaultArgExpr(Param);
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXZeroInitValueExpr(CXXZeroInitValueExpr *E) {
  TemporaryBase Rebase(*this, E->getTypeBeginLoc(), DeclarationName());

  QualType T = getDerived().TransformType(E->getType());
  if (T.isNull())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      T == E->getType())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildCXXZeroInitValueExpr(E->getTypeBeginLoc(),
                                                /*FIXME:*/E->getTypeBeginLoc(),
                                                  T,
                                                  E->getRParenLoc());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXConditionDeclExpr(CXXConditionDeclExpr *E) {
  VarDecl *Var
    = cast_or_null<VarDecl>(getDerived().TransformDefinition(E->getVarDecl()));
  if (!Var)
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      Var == E->getVarDecl())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildCXXConditionDeclExpr(E->getStartLoc(),
                                                  /*FIXME:*/E->getStartLoc(),
                                                  Var);
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXNewExpr(CXXNewExpr *E) {
  // Transform the type that we're allocating
  TemporaryBase Rebase(*this, E->getLocStart(), DeclarationName());
  QualType AllocType = getDerived().TransformType(E->getAllocatedType());
  if (AllocType.isNull())
    return SemaRef.ExprError();

  // Transform the size of the array we're allocating (if any).
  OwningExprResult ArraySize = getDerived().TransformExpr(E->getArraySize());
  if (ArraySize.isInvalid())
    return SemaRef.ExprError();

  // Transform the placement arguments (if any).
  bool ArgumentChanged = false;
  ASTOwningVector<&ActionBase::DeleteExpr> PlacementArgs(SemaRef);
  for (unsigned I = 0, N = E->getNumPlacementArgs(); I != N; ++I) {
    OwningExprResult Arg = getDerived().TransformExpr(E->getPlacementArg(I));
    if (Arg.isInvalid())
      return SemaRef.ExprError();

    ArgumentChanged = ArgumentChanged || Arg.get() != E->getPlacementArg(I);
    PlacementArgs.push_back(Arg.take());
  }

  // transform the constructor arguments (if any).
  ASTOwningVector<&ActionBase::DeleteExpr> ConstructorArgs(SemaRef);
  for (unsigned I = 0, N = E->getNumConstructorArgs(); I != N; ++I) {
    OwningExprResult Arg = getDerived().TransformExpr(E->getConstructorArg(I));
    if (Arg.isInvalid())
      return SemaRef.ExprError();

    ArgumentChanged = ArgumentChanged || Arg.get() != E->getConstructorArg(I);
    ConstructorArgs.push_back(Arg.take());
  }

  if (!getDerived().AlwaysRebuild() &&
      AllocType == E->getAllocatedType() &&
      ArraySize.get() == E->getArraySize() &&
      !ArgumentChanged)
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildCXXNewExpr(E->getLocStart(),
                                        E->isGlobalNew(),
                                        /*FIXME:*/E->getLocStart(),
                                        move_arg(PlacementArgs),
                                        /*FIXME:*/E->getLocStart(),
                                        E->isParenTypeId(),
                                        AllocType,
                                        /*FIXME:*/E->getLocStart(),
                                        /*FIXME:*/SourceRange(),
                                        move(ArraySize),
                                        /*FIXME:*/E->getLocStart(),
                                        move_arg(ConstructorArgs),
                                        E->getLocEnd());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXDeleteExpr(CXXDeleteExpr *E) {
  OwningExprResult Operand = getDerived().TransformExpr(E->getArgument());
  if (Operand.isInvalid())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      Operand.get() == E->getArgument())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildCXXDeleteExpr(E->getLocStart(),
                                           E->isGlobalDelete(),
                                           E->isArrayForm(),
                                           move(Operand));
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXPseudoDestructorExpr(
                                                CXXPseudoDestructorExpr *E) {
  OwningExprResult Base = getDerived().TransformExpr(E->getBase());
  if (Base.isInvalid())
    return SemaRef.ExprError();

  NestedNameSpecifier *Qualifier
    = getDerived().TransformNestedNameSpecifier(E->getQualifier(),
                                                E->getQualifierRange());
  if (E->getQualifier() && !Qualifier)
    return SemaRef.ExprError();

  QualType DestroyedType;
  {
    TemporaryBase Rebase(*this, E->getDestroyedTypeLoc(), DeclarationName());
    DestroyedType = getDerived().TransformType(E->getDestroyedType());
    if (DestroyedType.isNull())
      return SemaRef.ExprError();
  }

  if (!getDerived().AlwaysRebuild() &&
      Base.get() == E->getBase() &&
      Qualifier == E->getQualifier() &&
      DestroyedType == E->getDestroyedType())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildCXXPseudoDestructorExpr(move(Base),
                                                     E->getOperatorLoc(),
                                                     E->isArrow(),
                                                     E->getDestroyedTypeLoc(),
                                                     DestroyedType,
                                                     Qualifier,
                                                     E->getQualifierRange());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformUnresolvedFunctionNameExpr(
                                              UnresolvedFunctionNameExpr *E) {
  // There is no transformation we can apply to an unresolved function name.
  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformUnaryTypeTraitExpr(UnaryTypeTraitExpr *E) {
  TemporaryBase Rebase(*this, /*FIXME*/E->getLocStart(), DeclarationName());

  QualType T = getDerived().TransformType(E->getQueriedType());
  if (T.isNull())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      T == E->getQueriedType())
    return SemaRef.Owned(E->Retain());

  // FIXME: Bad location information
  SourceLocation FakeLParenLoc
    = SemaRef.PP.getLocForEndOfToken(E->getLocStart());

  return getDerived().RebuildUnaryTypeTrait(E->getTrait(),
                                            E->getLocStart(),
                                            /*FIXME:*/FakeLParenLoc,
                                            T,
                                            E->getLocEnd());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformUnresolvedDeclRefExpr(
                                                    UnresolvedDeclRefExpr *E) {
  NestedNameSpecifier *NNS
    = getDerived().TransformNestedNameSpecifier(E->getQualifier(),
                                                E->getQualifierRange());
  if (!NNS)
    return SemaRef.ExprError();

  DeclarationName Name
    = getDerived().TransformDeclarationName(E->getDeclName(), E->getLocation());
  if (!Name)
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      NNS == E->getQualifier() &&
      Name == E->getDeclName())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildUnresolvedDeclRefExpr(NNS,
                                                   E->getQualifierRange(),
                                                   Name,
                                                   E->getLocation(),
                                                   /*FIXME:*/false);
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformTemplateIdRefExpr(TemplateIdRefExpr *E) {
  TemporaryBase Rebase(*this, E->getTemplateNameLoc(), DeclarationName());
  
  TemplateName Template
    = getDerived().TransformTemplateName(E->getTemplateName());
  if (Template.isNull())
    return SemaRef.ExprError();

  NestedNameSpecifier *Qualifier = 0;
  if (E->getQualifier()) {
    Qualifier = getDerived().TransformNestedNameSpecifier(E->getQualifier(),
                                                      E->getQualifierRange());
    if (!Qualifier)
      return SemaRef.ExprError();
  }
  
  llvm::SmallVector<TemplateArgumentLoc, 4> TransArgs(E->getNumTemplateArgs());
  for (unsigned I = 0, N = E->getNumTemplateArgs(); I != N; ++I) {
    if (getDerived().TransformTemplateArgument(E->getTemplateArgs()[I],
                                               TransArgs[I]))
      return SemaRef.ExprError();
  }

  // FIXME: Would like to avoid rebuilding if nothing changed, but we can't
  // compare template arguments (yet).

  // FIXME: It's possible that we'll find out now that the template name
  // actually refers to a type, in which case the caller is actually dealing
  // with a functional cast. Give a reasonable error message!
  return getDerived().RebuildTemplateIdExpr(Qualifier, E->getQualifierRange(),
                                            Template, E->getTemplateNameLoc(),
                                            E->getLAngleLoc(),
                                            TransArgs.data(),
                                            TransArgs.size(),
                                            E->getRAngleLoc());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXConstructExpr(CXXConstructExpr *E) {
  TemporaryBase Rebase(*this, /*FIXME*/E->getLocStart(), DeclarationName());

  QualType T = getDerived().TransformType(E->getType());
  if (T.isNull())
    return SemaRef.ExprError();

  CXXConstructorDecl *Constructor
    = cast_or_null<CXXConstructorDecl>(
                              getDerived().TransformDecl(E->getConstructor()));
  if (!Constructor)
    return SemaRef.ExprError();

  bool ArgumentChanged = false;
  ASTOwningVector<&ActionBase::DeleteExpr> Args(SemaRef);
  for (CXXConstructExpr::arg_iterator Arg = E->arg_begin(),
       ArgEnd = E->arg_end();
       Arg != ArgEnd; ++Arg) {
    OwningExprResult TransArg = getDerived().TransformExpr(*Arg);
    if (TransArg.isInvalid())
      return SemaRef.ExprError();

    ArgumentChanged = ArgumentChanged || TransArg.get() != *Arg;
    Args.push_back(TransArg.takeAs<Expr>());
  }

  if (!getDerived().AlwaysRebuild() &&
      T == E->getType() &&
      Constructor == E->getConstructor() &&
      !ArgumentChanged)
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildCXXConstructExpr(T, Constructor, E->isElidable(),
                                              move_arg(Args));
}

/// \brief Transform a C++ temporary-binding expression.
///
/// The transformation of a temporary-binding expression always attempts to
/// bind a new temporary variable to its subexpression, even if the
/// subexpression itself did not change, because the temporary variable itself
/// must be unique.
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXBindTemporaryExpr(CXXBindTemporaryExpr *E) {
  OwningExprResult SubExpr = getDerived().TransformExpr(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();

  return SemaRef.MaybeBindToTemporary(SubExpr.takeAs<Expr>());
}

/// \brief Transform a C++ expression that contains temporaries that should
/// be destroyed after the expression is evaluated.
///
/// The transformation of a full expression always attempts to build a new
/// CXXExprWithTemporaries expression, even if the
/// subexpression itself did not change, because it will need to capture the
/// the new temporary variables introduced in the subexpression.
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXExprWithTemporaries(
                                                CXXExprWithTemporaries *E) {
  OwningExprResult SubExpr = getDerived().TransformExpr(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();

  return SemaRef.Owned(
           SemaRef.MaybeCreateCXXExprWithTemporaries(SubExpr.takeAs<Expr>(),
                                               E->shouldDestroyTemporaries()));
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXTemporaryObjectExpr(
                                                  CXXTemporaryObjectExpr *E) {
  TemporaryBase Rebase(*this, E->getTypeBeginLoc(), DeclarationName());
  QualType T = getDerived().TransformType(E->getType());
  if (T.isNull())
    return SemaRef.ExprError();

  CXXConstructorDecl *Constructor
    = cast_or_null<CXXConstructorDecl>(
                            getDerived().TransformDecl(E->getConstructor()));
  if (!Constructor)
    return SemaRef.ExprError();

  bool ArgumentChanged = false;
  ASTOwningVector<&ActionBase::DeleteExpr> Args(SemaRef);
  Args.reserve(E->getNumArgs());
  for (CXXTemporaryObjectExpr::arg_iterator Arg = E->arg_begin(),
                                         ArgEnd = E->arg_end();
       Arg != ArgEnd; ++Arg) {
    OwningExprResult TransArg = getDerived().TransformExpr(*Arg);
    if (TransArg.isInvalid())
      return SemaRef.ExprError();

    ArgumentChanged = ArgumentChanged || TransArg.get() != *Arg;
    Args.push_back((Expr *)TransArg.release());
  }

  if (!getDerived().AlwaysRebuild() &&
      T == E->getType() &&
      Constructor == E->getConstructor() &&
      !ArgumentChanged)
    return SemaRef.Owned(E->Retain());

  // FIXME: Bogus location information
  SourceLocation CommaLoc;
  if (Args.size() > 1) {
    Expr *First = (Expr *)Args[0];
    CommaLoc
      = SemaRef.PP.getLocForEndOfToken(First->getSourceRange().getEnd());
  }
  return getDerived().RebuildCXXTemporaryObjectExpr(E->getTypeBeginLoc(),
                                                    T,
                                                /*FIXME:*/E->getTypeBeginLoc(),
                                                    move_arg(Args),
                                                    &CommaLoc,
                                                    E->getLocEnd());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXUnresolvedConstructExpr(
                                            CXXUnresolvedConstructExpr *E) {
  TemporaryBase Rebase(*this, E->getTypeBeginLoc(), DeclarationName());
  QualType T = getDerived().TransformType(E->getTypeAsWritten());
  if (T.isNull())
    return SemaRef.ExprError();

  bool ArgumentChanged = false;
  ASTOwningVector<&ActionBase::DeleteExpr> Args(SemaRef);
  llvm::SmallVector<SourceLocation, 8> FakeCommaLocs;
  for (CXXUnresolvedConstructExpr::arg_iterator Arg = E->arg_begin(),
                                             ArgEnd = E->arg_end();
       Arg != ArgEnd; ++Arg) {
    OwningExprResult TransArg = getDerived().TransformExpr(*Arg);
    if (TransArg.isInvalid())
      return SemaRef.ExprError();

    ArgumentChanged = ArgumentChanged || TransArg.get() != *Arg;
    FakeCommaLocs.push_back(
                        SemaRef.PP.getLocForEndOfToken((*Arg)->getLocEnd()));
    Args.push_back(TransArg.takeAs<Expr>());
  }

  if (!getDerived().AlwaysRebuild() &&
      T == E->getTypeAsWritten() &&
      !ArgumentChanged)
    return SemaRef.Owned(E->Retain());

  // FIXME: we're faking the locations of the commas
  return getDerived().RebuildCXXUnresolvedConstructExpr(E->getTypeBeginLoc(),
                                                        T,
                                                        E->getLParenLoc(),
                                                        move_arg(Args),
                                                        FakeCommaLocs.data(),
                                                        E->getRParenLoc());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXUnresolvedMemberExpr(
                                                  CXXUnresolvedMemberExpr *E) {
  // Transform the base of the expression.
  OwningExprResult Base = getDerived().TransformExpr(E->getBase());
  if (Base.isInvalid())
    return SemaRef.ExprError();

  // Start the member reference and compute the object's type.
  Sema::TypeTy *ObjectType = 0;
  Base = SemaRef.ActOnStartCXXMemberReference(0, move(Base),
                                              E->getOperatorLoc(),
                                      E->isArrow()? tok::arrow : tok::period,
                                              ObjectType);
  if (Base.isInvalid())
    return SemaRef.ExprError();

  // Transform the first part of the nested-name-specifier that qualifies
  // the member name.
  NamedDecl *FirstQualifierInScope
    = getDerived().TransformFirstQualifierInScope(
                                          E->getFirstQualifierFoundInScope(),
                                          E->getQualifierRange().getBegin());

  NestedNameSpecifier *Qualifier = 0;
  if (E->getQualifier()) {
    Qualifier = getDerived().TransformNestedNameSpecifier(E->getQualifier(),
                                                      E->getQualifierRange(),
                                      QualType::getFromOpaquePtr(ObjectType),
                                                        FirstQualifierInScope);
    if (!Qualifier)
      return SemaRef.ExprError();
  }

  DeclarationName Name
    = getDerived().TransformDeclarationName(E->getMember(), E->getMemberLoc(),
                                       QualType::getFromOpaquePtr(ObjectType));
  if (!Name)
    return SemaRef.ExprError();

  if (!E->hasExplicitTemplateArgumentList()) {
    // This is a reference to a member without an explicitly-specified
    // template argument list. Optimize for this common case.
    if (!getDerived().AlwaysRebuild() &&
        Base.get() == E->getBase() &&
        Qualifier == E->getQualifier() &&
        Name == E->getMember() &&
        FirstQualifierInScope == E->getFirstQualifierFoundInScope())
      return SemaRef.Owned(E->Retain());

    return getDerived().RebuildCXXUnresolvedMemberExpr(move(Base),
                                                       E->isArrow(),
                                                       E->getOperatorLoc(),
                                                       Qualifier,
                                                       E->getQualifierRange(),
                                                       Name,
                                                       E->getMemberLoc(),
                                                       FirstQualifierInScope);
  }

  // FIXME: This is an ugly hack, which forces the same template name to
  // be looked up multiple times. Yuck!
  // FIXME: This also won't work for, e.g., x->template operator+<int>
  TemplateName OrigTemplateName
    = SemaRef.Context.getDependentTemplateName(0, Name.getAsIdentifierInfo());

  TemplateName Template
    = getDerived().TransformTemplateName(OrigTemplateName,
                                       QualType::getFromOpaquePtr(ObjectType));
  if (Template.isNull())
    return SemaRef.ExprError();

  llvm::SmallVector<TemplateArgumentLoc, 4> TransArgs(E->getNumTemplateArgs());
  for (unsigned I = 0, N = E->getNumTemplateArgs(); I != N; ++I) {
    if (getDerived().TransformTemplateArgument(E->getTemplateArgs()[I],
                                               TransArgs[I]))
      return SemaRef.ExprError();
  }

  return getDerived().RebuildCXXUnresolvedMemberExpr(move(Base),
                                                     E->isArrow(),
                                                     E->getOperatorLoc(),
                                                     Qualifier,
                                                     E->getQualifierRange(),
                                                     Template,
                                                     E->getMemberLoc(),
                                                     FirstQualifierInScope,
                                                     E->getLAngleLoc(),
                                                     TransArgs.data(),
                                                     TransArgs.size(),
                                                     E->getRAngleLoc());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformObjCStringLiteral(ObjCStringLiteral *E) {
  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformObjCEncodeExpr(ObjCEncodeExpr *E) {
  // FIXME: poor source location
  TemporaryBase Rebase(*this, E->getAtLoc(), DeclarationName());
  QualType EncodedType = getDerived().TransformType(E->getEncodedType());
  if (EncodedType.isNull())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      EncodedType == E->getEncodedType())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildObjCEncodeExpr(E->getAtLoc(),
                                            EncodedType,
                                            E->getRParenLoc());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformObjCMessageExpr(ObjCMessageExpr *E) {
  // FIXME: Implement this!
  assert(false && "Cannot transform Objective-C expressions yet");
  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformObjCSelectorExpr(ObjCSelectorExpr *E) {
  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformObjCProtocolExpr(ObjCProtocolExpr *E) {
  ObjCProtocolDecl *Protocol
    = cast_or_null<ObjCProtocolDecl>(
                                getDerived().TransformDecl(E->getProtocol()));
  if (!Protocol)
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      Protocol == E->getProtocol())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildObjCProtocolExpr(Protocol,
                                              E->getAtLoc(),
                                              /*FIXME:*/E->getAtLoc(),
                                              /*FIXME:*/E->getAtLoc(),
                                              E->getRParenLoc());

}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformObjCIvarRefExpr(ObjCIvarRefExpr *E) {
  // FIXME: Implement this!
  assert(false && "Cannot transform Objective-C expressions yet");
  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformObjCPropertyRefExpr(ObjCPropertyRefExpr *E) {
  // FIXME: Implement this!
  assert(false && "Cannot transform Objective-C expressions yet");
  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformObjCImplicitSetterGetterRefExpr(
                                          ObjCImplicitSetterGetterRefExpr *E) {
  // FIXME: Implement this!
  assert(false && "Cannot transform Objective-C expressions yet");
  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformObjCSuperExpr(ObjCSuperExpr *E) {
  // FIXME: Implement this!
  assert(false && "Cannot transform Objective-C expressions yet");
  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformObjCIsaExpr(ObjCIsaExpr *E) {
  // FIXME: Implement this!
  assert(false && "Cannot transform Objective-C expressions yet");
  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformShuffleVectorExpr(ShuffleVectorExpr *E) {
  bool ArgumentChanged = false;
  ASTOwningVector<&ActionBase::DeleteExpr> SubExprs(SemaRef);
  for (unsigned I = 0, N = E->getNumSubExprs(); I != N; ++I) {
    OwningExprResult SubExpr = getDerived().TransformExpr(E->getExpr(I));
    if (SubExpr.isInvalid())
      return SemaRef.ExprError();

    ArgumentChanged = ArgumentChanged || SubExpr.get() != E->getExpr(I);
    SubExprs.push_back(SubExpr.takeAs<Expr>());
  }

  if (!getDerived().AlwaysRebuild() &&
      !ArgumentChanged)
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildShuffleVectorExpr(E->getBuiltinLoc(),
                                               move_arg(SubExprs),
                                               E->getRParenLoc());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformBlockExpr(BlockExpr *E) {
  // FIXME: Implement this!
  assert(false && "Cannot transform block expressions yet");
  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformBlockDeclRefExpr(BlockDeclRefExpr *E) {
  // FIXME: Implement this!
  assert(false && "Cannot transform block-related expressions yet");
  return SemaRef.Owned(E->Retain());
}

//===----------------------------------------------------------------------===//
// Type reconstruction
//===----------------------------------------------------------------------===//

template<typename Derived>
QualType TreeTransform<Derived>::RebuildPointerType(QualType PointeeType,
                                                    SourceLocation Star) {
  return SemaRef.BuildPointerType(PointeeType, Qualifiers(), Star,
                                  getDerived().getBaseEntity());
}

template<typename Derived>
QualType TreeTransform<Derived>::RebuildBlockPointerType(QualType PointeeType,
                                                         SourceLocation Star) {
  return SemaRef.BuildBlockPointerType(PointeeType, Qualifiers(), Star,
                                       getDerived().getBaseEntity());
}

template<typename Derived>
QualType
TreeTransform<Derived>::RebuildReferenceType(QualType ReferentType,
                                             bool WrittenAsLValue,
                                             SourceLocation Sigil) {
  return SemaRef.BuildReferenceType(ReferentType, WrittenAsLValue, Qualifiers(),
                                    Sigil, getDerived().getBaseEntity());
}

template<typename Derived>
QualType
TreeTransform<Derived>::RebuildMemberPointerType(QualType PointeeType,
                                                 QualType ClassType,
                                                 SourceLocation Sigil) {
  return SemaRef.BuildMemberPointerType(PointeeType, ClassType, Qualifiers(),
                                        Sigil, getDerived().getBaseEntity());
}

template<typename Derived>
QualType
TreeTransform<Derived>::RebuildObjCObjectPointerType(QualType PointeeType,
                                                     SourceLocation Sigil) {
  return SemaRef.BuildPointerType(PointeeType, Qualifiers(), Sigil,
                                  getDerived().getBaseEntity());
}

template<typename Derived>
QualType
TreeTransform<Derived>::RebuildArrayType(QualType ElementType,
                                         ArrayType::ArraySizeModifier SizeMod,
                                         const llvm::APInt *Size,
                                         Expr *SizeExpr,
                                         unsigned IndexTypeQuals,
                                         SourceRange BracketsRange) {
  if (SizeExpr || !Size)
    return SemaRef.BuildArrayType(ElementType, SizeMod, SizeExpr,
                                  IndexTypeQuals, BracketsRange,
                                  getDerived().getBaseEntity());

  QualType Types[] = {
    SemaRef.Context.UnsignedCharTy, SemaRef.Context.UnsignedShortTy,
    SemaRef.Context.UnsignedIntTy, SemaRef.Context.UnsignedLongTy,
    SemaRef.Context.UnsignedLongLongTy, SemaRef.Context.UnsignedInt128Ty
  };
  const unsigned NumTypes = sizeof(Types) / sizeof(QualType);
  QualType SizeType;
  for (unsigned I = 0; I != NumTypes; ++I)
    if (Size->getBitWidth() == SemaRef.Context.getIntWidth(Types[I])) {
      SizeType = Types[I];
      break;
    }

  if (SizeType.isNull())
    SizeType = SemaRef.Context.getFixedWidthIntType(Size->getBitWidth(), false);

  IntegerLiteral ArraySize(*Size, SizeType, /*FIXME*/BracketsRange.getBegin());
  return SemaRef.BuildArrayType(ElementType, SizeMod, &ArraySize,
                                IndexTypeQuals, BracketsRange,
                                getDerived().getBaseEntity());
}

template<typename Derived>
QualType
TreeTransform<Derived>::RebuildConstantArrayType(QualType ElementType,
                                                 ArrayType::ArraySizeModifier SizeMod,
                                                 const llvm::APInt &Size,
                                                 unsigned IndexTypeQuals,
                                                 SourceRange BracketsRange) {
  return getDerived().RebuildArrayType(ElementType, SizeMod, &Size, 0,
                                        IndexTypeQuals, BracketsRange);
}

template<typename Derived>
QualType
TreeTransform<Derived>::RebuildIncompleteArrayType(QualType ElementType,
                                          ArrayType::ArraySizeModifier SizeMod,
                                                 unsigned IndexTypeQuals,
                                                   SourceRange BracketsRange) {
  return getDerived().RebuildArrayType(ElementType, SizeMod, 0, 0,
                                       IndexTypeQuals, BracketsRange);
}

template<typename Derived>
QualType
TreeTransform<Derived>::RebuildVariableArrayType(QualType ElementType,
                                          ArrayType::ArraySizeModifier SizeMod,
                                                 ExprArg SizeExpr,
                                                 unsigned IndexTypeQuals,
                                                 SourceRange BracketsRange) {
  return getDerived().RebuildArrayType(ElementType, SizeMod, 0,
                                       SizeExpr.takeAs<Expr>(),
                                       IndexTypeQuals, BracketsRange);
}

template<typename Derived>
QualType
TreeTransform<Derived>::RebuildDependentSizedArrayType(QualType ElementType,
                                          ArrayType::ArraySizeModifier SizeMod,
                                                       ExprArg SizeExpr,
                                                       unsigned IndexTypeQuals,
                                                   SourceRange BracketsRange) {
  return getDerived().RebuildArrayType(ElementType, SizeMod, 0,
                                       SizeExpr.takeAs<Expr>(),
                                       IndexTypeQuals, BracketsRange);
}

template<typename Derived>
QualType TreeTransform<Derived>::RebuildVectorType(QualType ElementType,
                                                   unsigned NumElements) {
  // FIXME: semantic checking!
  return SemaRef.Context.getVectorType(ElementType, NumElements);
}

template<typename Derived>
QualType TreeTransform<Derived>::RebuildExtVectorType(QualType ElementType,
                                                      unsigned NumElements,
                                                 SourceLocation AttributeLoc) {
  llvm::APInt numElements(SemaRef.Context.getIntWidth(SemaRef.Context.IntTy),
                          NumElements, true);
  IntegerLiteral *VectorSize
    = new (SemaRef.Context) IntegerLiteral(numElements, SemaRef.Context.IntTy,
                                           AttributeLoc);
  return SemaRef.BuildExtVectorType(ElementType, SemaRef.Owned(VectorSize),
                                    AttributeLoc);
}

template<typename Derived>
QualType
TreeTransform<Derived>::RebuildDependentSizedExtVectorType(QualType ElementType,
                                                           ExprArg SizeExpr,
                                                  SourceLocation AttributeLoc) {
  return SemaRef.BuildExtVectorType(ElementType, move(SizeExpr), AttributeLoc);
}

template<typename Derived>
QualType TreeTransform<Derived>::RebuildFunctionProtoType(QualType T,
                                                          QualType *ParamTypes,
                                                        unsigned NumParamTypes,
                                                          bool Variadic,
                                                          unsigned Quals) {
  return SemaRef.BuildFunctionType(T, ParamTypes, NumParamTypes, Variadic,
                                   Quals,
                                   getDerived().getBaseLocation(),
                                   getDerived().getBaseEntity());
}

template<typename Derived>
QualType TreeTransform<Derived>::RebuildFunctionNoProtoType(QualType T) {
  return SemaRef.Context.getFunctionNoProtoType(T);
}

template<typename Derived>
QualType TreeTransform<Derived>::RebuildTypeOfExprType(ExprArg E) {
  return SemaRef.BuildTypeofExprType(E.takeAs<Expr>());
}

template<typename Derived>
QualType TreeTransform<Derived>::RebuildTypeOfType(QualType Underlying) {
  return SemaRef.Context.getTypeOfType(Underlying);
}

template<typename Derived>
QualType TreeTransform<Derived>::RebuildDecltypeType(ExprArg E) {
  return SemaRef.BuildDecltypeType(E.takeAs<Expr>());
}

template<typename Derived>
QualType TreeTransform<Derived>::RebuildTemplateSpecializationType(
                                                      TemplateName Template,
                                             SourceLocation TemplateNameLoc,
                                                   SourceLocation LAngleLoc,
                                            const TemplateArgumentLoc *Args,
                                                           unsigned NumArgs,
                                                   SourceLocation RAngleLoc) {
  return SemaRef.CheckTemplateIdType(Template, TemplateNameLoc, LAngleLoc,
                                     Args, NumArgs, RAngleLoc);
}

template<typename Derived>
NestedNameSpecifier *
TreeTransform<Derived>::RebuildNestedNameSpecifier(NestedNameSpecifier *Prefix,
                                                   SourceRange Range,
                                                   IdentifierInfo &II,
                                                   QualType ObjectType,
                                             NamedDecl *FirstQualifierInScope) {
  CXXScopeSpec SS;
  // FIXME: The source location information is all wrong.
  SS.setRange(Range);
  SS.setScopeRep(Prefix);
  return static_cast<NestedNameSpecifier *>(
                    SemaRef.BuildCXXNestedNameSpecifier(0, SS, Range.getEnd(),
                                                        Range.getEnd(), II,
                                                        ObjectType,
                                                        FirstQualifierInScope,
                                                        false));
}

template<typename Derived>
NestedNameSpecifier *
TreeTransform<Derived>::RebuildNestedNameSpecifier(NestedNameSpecifier *Prefix,
                                                   SourceRange Range,
                                                   NamespaceDecl *NS) {
  return NestedNameSpecifier::Create(SemaRef.Context, Prefix, NS);
}

template<typename Derived>
NestedNameSpecifier *
TreeTransform<Derived>::RebuildNestedNameSpecifier(NestedNameSpecifier *Prefix,
                                                   SourceRange Range,
                                                   bool TemplateKW,
                                                   QualType T) {
  if (T->isDependentType() || T->isRecordType() ||
      (SemaRef.getLangOptions().CPlusPlus0x && T->isEnumeralType())) {
    assert(!T.hasQualifiers() && "Can't get cv-qualifiers here");
    return NestedNameSpecifier::Create(SemaRef.Context, Prefix, TemplateKW,
                                       T.getTypePtr());
  }

  SemaRef.Diag(Range.getBegin(), diag::err_nested_name_spec_non_tag) << T;
  return 0;
}

template<typename Derived>
TemplateName
TreeTransform<Derived>::RebuildTemplateName(NestedNameSpecifier *Qualifier,
                                            bool TemplateKW,
                                            TemplateDecl *Template) {
  return SemaRef.Context.getQualifiedTemplateName(Qualifier, TemplateKW,
                                                  Template);
}

template<typename Derived>
TemplateName
TreeTransform<Derived>::RebuildTemplateName(NestedNameSpecifier *Qualifier,
                                            bool TemplateKW,
                                            OverloadedFunctionDecl *Ovl) {
  return SemaRef.Context.getQualifiedTemplateName(Qualifier, TemplateKW, Ovl);
}

template<typename Derived>
TemplateName
TreeTransform<Derived>::RebuildTemplateName(NestedNameSpecifier *Qualifier,
                                            const IdentifierInfo &II,
                                            QualType ObjectType) {
  CXXScopeSpec SS;
  SS.setRange(SourceRange(getDerived().getBaseLocation()));
  SS.setScopeRep(Qualifier);
  return getSema().ActOnDependentTemplateName(
                                      /*FIXME:*/getDerived().getBaseLocation(),
                                              II,
                                      /*FIXME:*/getDerived().getBaseLocation(),
                                              SS,
                                              ObjectType.getAsOpaquePtr())
           .template getAsVal<TemplateName>();
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::RebuildCXXOperatorCallExpr(OverloadedOperatorKind Op,
                                                   SourceLocation OpLoc,
                                                   ExprArg Callee,
                                                   ExprArg First,
                                                   ExprArg Second) {
  Expr *FirstExpr = (Expr *)First.get();
  Expr *SecondExpr = (Expr *)Second.get();
  DeclRefExpr *DRE
    = cast<DeclRefExpr>(((Expr *)Callee.get())->IgnoreParenCasts());
  bool isPostIncDec = SecondExpr && (Op == OO_PlusPlus || Op == OO_MinusMinus);

  // Determine whether this should be a builtin operation.
  if (Op == OO_Subscript) {
    if (!FirstExpr->getType()->isOverloadableType() &&
        !SecondExpr->getType()->isOverloadableType())
      return getSema().CreateBuiltinArraySubscriptExpr(move(First),
                                                       DRE->getLocStart(),
                                                       move(Second), OpLoc);
  } else if (SecondExpr == 0 || isPostIncDec) {
    if (!FirstExpr->getType()->isOverloadableType()) {
      // The argument is not of overloadable type, so try to create a
      // built-in unary operation.
      UnaryOperator::Opcode Opc
        = UnaryOperator::getOverloadedOpcode(Op, isPostIncDec);

      return getSema().CreateBuiltinUnaryOp(OpLoc, Opc, move(First));
    }
  } else {
    if (!FirstExpr->getType()->isOverloadableType() &&
        !SecondExpr->getType()->isOverloadableType()) {
      // Neither of the arguments is an overloadable type, so try to
      // create a built-in binary operation.
      BinaryOperator::Opcode Opc = BinaryOperator::getOverloadedOpcode(Op);
      OwningExprResult Result
        = SemaRef.CreateBuiltinBinOp(OpLoc, Opc, FirstExpr, SecondExpr);
      if (Result.isInvalid())
        return SemaRef.ExprError();

      First.release();
      Second.release();
      return move(Result);
    }
  }

  // Compute the transformed set of functions (and function templates) to be
  // used during overload resolution.
  Sema::FunctionSet Functions;

  // FIXME: Do we have to check
  // IsAcceptableNonMemberOperatorCandidate for each of these?
  for (OverloadIterator F(DRE->getDecl()), FEnd; F != FEnd; ++F)
    Functions.insert(*F);

  // Add any functions found via argument-dependent lookup.
  Expr *Args[2] = { FirstExpr, SecondExpr };
  unsigned NumArgs = 1 + (SecondExpr != 0);
  DeclarationName OpName
    = SemaRef.Context.DeclarationNames.getCXXOperatorName(Op);
  SemaRef.ArgumentDependentLookup(OpName, /*Operator*/true, Args, NumArgs,
                                  Functions);

  // Create the overloaded operator invocation for unary operators.
  if (NumArgs == 1 || isPostIncDec) {
    UnaryOperator::Opcode Opc
      = UnaryOperator::getOverloadedOpcode(Op, isPostIncDec);
    return SemaRef.CreateOverloadedUnaryOp(OpLoc, Opc, Functions, move(First));
  }

  if (Op == OO_Subscript)
    return SemaRef.CreateOverloadedArraySubscriptExpr(DRE->getLocStart(), OpLoc,
                                                      move(First),move(Second));

  // Create the overloaded operator invocation for binary operators.
  BinaryOperator::Opcode Opc =
    BinaryOperator::getOverloadedOpcode(Op);
  OwningExprResult Result
    = SemaRef.CreateOverloadedBinOp(OpLoc, Opc, Functions, Args[0], Args[1]);
  if (Result.isInvalid())
    return SemaRef.ExprError();

  First.release();
  Second.release();
  return move(Result);
}

} // end namespace clang

#endif // LLVM_CLANG_SEMA_TREETRANSFORM_H
