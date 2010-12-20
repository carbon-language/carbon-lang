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

#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/Designator.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/ErrorHandling.h"
#include "TypeLocBuilder.h"
#include <algorithm>

namespace clang {
using namespace sema;

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
  /// \brief Initializes a new tree transformer.
  TreeTransform(Sema &SemaRef) : SemaRef(SemaRef) { }

  /// \brief Retrieves a reference to the derived class.
  Derived &getDerived() { return static_cast<Derived&>(*this); }

  /// \brief Retrieves a reference to the derived class.
  const Derived &getDerived() const {
    return static_cast<const Derived&>(*this);
  }

  static inline ExprResult Owned(Expr *E) { return E; }
  static inline StmtResult Owned(Stmt *S) { return S; }

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

  /// \brief Determine whether the given call argument should be dropped, e.g.,
  /// because it is a default argument.
  ///
  /// Subclasses can provide an alternative implementation of this routine to
  /// determine which kinds of call arguments get dropped. By default,
  /// CXXDefaultArgument nodes are dropped (prior to transformation).
  bool DropCallArgument(Expr *E) {
    return E->isDefaultArgument();
  }
  
  /// \brief Transforms the given type into another type.
  ///
  /// By default, this routine transforms a type by creating a
  /// TypeSourceInfo for it and delegating to the appropriate
  /// function.  This is expensive, but we don't mind, because
  /// this method is deprecated anyway;  all users should be
  /// switched to storing TypeSourceInfos.
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
  TypeSourceInfo *TransformType(TypeSourceInfo *DI);

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
  StmtResult TransformStmt(Stmt *S);

  /// \brief Transform the given expression.
  ///
  /// By default, this routine transforms an expression by delegating to the
  /// appropriate TransformXXXExpr function to build a new expression.
  /// Subclasses may override this function to transform expressions using some
  /// other mechanism.
  ///
  /// \returns the transformed expression.
  ExprResult TransformExpr(Expr *E);

  /// \brief Transform the given declaration, which is referenced from a type
  /// or expression.
  ///
  /// By default, acts as the identity function on declarations. Subclasses
  /// may override this function to provide alternate behavior.
  Decl *TransformDecl(SourceLocation Loc, Decl *D) { return D; }

  /// \brief Transform the definition of the given declaration.
  ///
  /// By default, invokes TransformDecl() to transform the declaration.
  /// Subclasses may override this function to provide alternate behavior.
  Decl *TransformDefinition(SourceLocation Loc, Decl *D) { 
    return getDerived().TransformDecl(Loc, D); 
  }

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
    return cast_or_null<NamedDecl>(getDerived().TransformDecl(Loc, D)); 
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
  DeclarationNameInfo
  TransformDeclarationNameInfo(const DeclarationNameInfo &NameInfo);

  /// \brief Transform the given template name.
  ///
  /// By default, transforms the template name by transforming the declarations
  /// and nested-name-specifiers that occur within the template name.
  /// Subclasses may override this function to provide alternate behavior.
  TemplateName TransformTemplateName(TemplateName Name,
                                     QualType ObjectType = QualType(),
                                     NamedDecl *FirstQualifierInScope = 0);

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

  /// \brief Fakes up a TypeSourceInfo for a type.
  TypeSourceInfo *InventTypeSourceInfo(QualType T) {
    return SemaRef.Context.getTrivialTypeSourceInfo(T,
                       getDerived().getBaseLocation());
  }

#define ABSTRACT_TYPELOC(CLASS, PARENT)
#define TYPELOC(CLASS, PARENT)                                   \
  QualType Transform##CLASS##Type(TypeLocBuilder &TLB, CLASS##TypeLoc T);
#include "clang/AST/TypeLocNodes.def"

  QualType 
  TransformTemplateSpecializationType(TypeLocBuilder &TLB,
                                      TemplateSpecializationTypeLoc TL,
                                      TemplateName Template);

  QualType 
  TransformDependentTemplateSpecializationType(TypeLocBuilder &TLB,
                                      DependentTemplateSpecializationTypeLoc TL,
                                               NestedNameSpecifier *Prefix);

  /// \brief Transforms the parameters of a function type into the
  /// given vectors.
  ///
  /// The result vectors should be kept in sync; null entries in the
  /// variables vector are acceptable.
  ///
  /// Return true on error.
  bool TransformFunctionTypeParams(FunctionProtoTypeLoc TL,
                                   llvm::SmallVectorImpl<QualType> &PTypes,
                                   llvm::SmallVectorImpl<ParmVarDecl*> &PVars);

  /// \brief Transforms a single function-type parameter.  Return null
  /// on error.
  ParmVarDecl *TransformFunctionTypeParam(ParmVarDecl *OldParm);

  QualType TransformReferenceType(TypeLocBuilder &TLB, ReferenceTypeLoc TL);

  StmtResult TransformCompoundStmt(CompoundStmt *S, bool IsStmtExpr);
  ExprResult TransformCXXNamedCastExpr(CXXNamedCastExpr *E);

#define STMT(Node, Parent)                        \
  StmtResult Transform##Node(Node *S);
#define EXPR(Node, Parent)                        \
  ExprResult Transform##Node(Node *E);
#define ABSTRACT_STMT(Stmt)
#include "clang/AST/StmtNodes.inc"

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
                                    Expr *SizeExpr,
                                    unsigned IndexTypeQuals,
                                    SourceRange BracketsRange);

  /// \brief Build a new dependent-sized array type given the element type,
  /// size modifier, size expression, and index type qualifiers.
  ///
  /// By default, performs semantic analysis when building the array type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildDependentSizedArrayType(QualType ElementType,
                                          ArrayType::ArraySizeModifier SizeMod,
                                          Expr *SizeExpr,
                                          unsigned IndexTypeQuals,
                                          SourceRange BracketsRange);

  /// \brief Build a new vector type given the element type and
  /// number of elements.
  ///
  /// By default, performs semantic analysis when building the vector type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildVectorType(QualType ElementType, unsigned NumElements,
                             VectorType::VectorKind VecKind);

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
                                              Expr *SizeExpr,
                                              SourceLocation AttributeLoc);

  /// \brief Build a new function type.
  ///
  /// By default, performs semantic analysis when building the function type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildFunctionProtoType(QualType T,
                                    QualType *ParamTypes,
                                    unsigned NumParamTypes,
                                    bool Variadic, unsigned Quals,
                                    const FunctionType::ExtInfo &Info);

  /// \brief Build a new unprototyped function type.
  QualType RebuildFunctionNoProtoType(QualType ResultType);

  /// \brief Rebuild an unresolved typename type, given the decl that
  /// the UnresolvedUsingTypenameDecl was transformed to.
  QualType RebuildUnresolvedUsingType(Decl *D);

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

  /// \brief Build a new typeof(expr) type.
  ///
  /// By default, performs semantic analysis when building the typeof type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildTypeOfExprType(Expr *Underlying, SourceLocation Loc);

  /// \brief Build a new typeof(type) type.
  ///
  /// By default, builds a new TypeOfType with the given underlying type.
  QualType RebuildTypeOfType(QualType Underlying);

  /// \brief Build a new C++0x decltype type.
  ///
  /// By default, performs semantic analysis when building the decltype type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildDecltypeType(Expr *Underlying, SourceLocation Loc);

  /// \brief Build a new template specialization type.
  ///
  /// By default, performs semantic analysis when building the template
  /// specialization type. Subclasses may override this routine to provide
  /// different behavior.
  QualType RebuildTemplateSpecializationType(TemplateName Template,
                                             SourceLocation TemplateLoc,
                                       const TemplateArgumentListInfo &Args);

  /// \brief Build a new parenthesized type.
  ///
  /// By default, builds a new ParenType type from the inner type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildParenType(QualType InnerType) {
    return SemaRef.Context.getParenType(InnerType);
  }

  /// \brief Build a new qualified name type.
  ///
  /// By default, builds a new ElaboratedType type from the keyword,
  /// the nested-name-specifier and the named type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildElaboratedType(SourceLocation KeywordLoc,
                                 ElaboratedTypeKeyword Keyword,
                                 NestedNameSpecifier *NNS, QualType Named) {
    return SemaRef.Context.getElaboratedType(Keyword, NNS, Named);
  }

  /// \brief Build a new typename type that refers to a template-id.
  ///
  /// By default, builds a new DependentNameType type from the
  /// nested-name-specifier and the given type. Subclasses may override
  /// this routine to provide different behavior.
  QualType RebuildDependentTemplateSpecializationType(
                                    ElaboratedTypeKeyword Keyword,
                                    NestedNameSpecifier *Qualifier,
                                    SourceRange QualifierRange,
                                    const IdentifierInfo *Name,
                                    SourceLocation NameLoc,
                                    const TemplateArgumentListInfo &Args) {
    // Rebuild the template name.
    // TODO: avoid TemplateName abstraction
    TemplateName InstName =
      getDerived().RebuildTemplateName(Qualifier, QualifierRange, *Name, 
                                       QualType(), 0);
    
    if (InstName.isNull())
      return QualType();

    // If it's still dependent, make a dependent specialization.
    if (InstName.getAsDependentTemplateName())
      return SemaRef.Context.getDependentTemplateSpecializationType(
                                          Keyword, Qualifier, Name, Args);

    // Otherwise, make an elaborated type wrapping a non-dependent
    // specialization.
    QualType T =
      getDerived().RebuildTemplateSpecializationType(InstName, NameLoc, Args);
    if (T.isNull()) return QualType();

    // NOTE: NNS is already recorded in template specialization type T.
    return SemaRef.Context.getElaboratedType(Keyword, /*NNS=*/0, T);
  }

  /// \brief Build a new typename type that refers to an identifier.
  ///
  /// By default, performs semantic analysis when building the typename type
  /// (or elaborated type). Subclasses may override this routine to provide
  /// different behavior.
  QualType RebuildDependentNameType(ElaboratedTypeKeyword Keyword,
                                    NestedNameSpecifier *NNS,
                                    const IdentifierInfo *Id,
                                    SourceLocation KeywordLoc,
                                    SourceRange NNSRange,
                                    SourceLocation IdLoc) {
    CXXScopeSpec SS;
    SS.setScopeRep(NNS);
    SS.setRange(NNSRange);

    if (NNS->isDependent()) {
      // If the name is still dependent, just build a new dependent name type.
      if (!SemaRef.computeDeclContext(SS))
        return SemaRef.Context.getDependentNameType(Keyword, NNS, Id);
    }

    if (Keyword == ETK_None || Keyword == ETK_Typename)
      return SemaRef.CheckTypenameType(Keyword, NNS, *Id,
                                       KeywordLoc, NNSRange, IdLoc);

    TagTypeKind Kind = TypeWithKeyword::getTagTypeKindForKeyword(Keyword);

    // We had a dependent elaborated-type-specifier that has been transformed
    // into a non-dependent elaborated-type-specifier. Find the tag we're
    // referring to.
    LookupResult Result(SemaRef, Id, IdLoc, Sema::LookupTagName);
    DeclContext *DC = SemaRef.computeDeclContext(SS, false);
    if (!DC)
      return QualType();

    if (SemaRef.RequireCompleteDeclContext(SS, DC))
      return QualType();

    TagDecl *Tag = 0;
    SemaRef.LookupQualifiedName(Result, DC);
    switch (Result.getResultKind()) {
      case LookupResult::NotFound:
      case LookupResult::NotFoundInCurrentInstantiation:
        break;
        
      case LookupResult::Found:
        Tag = Result.getAsSingle<TagDecl>();
        break;
        
      case LookupResult::FoundOverloaded:
      case LookupResult::FoundUnresolvedValue:
        llvm_unreachable("Tag lookup cannot find non-tags");
        return QualType();
        
      case LookupResult::Ambiguous:
        // Let the LookupResult structure handle ambiguities.
        return QualType();
    }

    if (!Tag) {
      // FIXME: Would be nice to highlight just the source range.
      SemaRef.Diag(IdLoc, diag::err_not_tag_in_scope)
        << Kind << Id << DC;
      return QualType();
    }

    if (!SemaRef.isAcceptableTagRedeclaration(Tag, Kind, IdLoc, *Id)) {
      SemaRef.Diag(KeywordLoc, diag::err_use_with_wrong_tag) << Id;
      SemaRef.Diag(Tag->getLocation(), diag::note_previous_use);
      return QualType();
    }

    // Build the elaborated-type-specifier type.
    QualType T = SemaRef.Context.getTypeDeclType(Tag);
    return SemaRef.Context.getElaboratedType(Keyword, NNS, T);
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

  /// \brief Build a new template name given a nested name specifier and the
  /// name that is referred to as a template.
  ///
  /// By default, performs semantic analysis to determine whether the name can
  /// be resolved to a specific template, then builds the appropriate kind of
  /// template name. Subclasses may override this routine to provide different
  /// behavior.
  TemplateName RebuildTemplateName(NestedNameSpecifier *Qualifier,
                                   SourceRange QualifierRange,
                                   const IdentifierInfo &II,
                                   QualType ObjectType,
                                   NamedDecl *FirstQualifierInScope);

  /// \brief Build a new template name given a nested name specifier and the
  /// overloaded operator name that is referred to as a template.
  ///
  /// By default, performs semantic analysis to determine whether the name can
  /// be resolved to a specific template, then builds the appropriate kind of
  /// template name. Subclasses may override this routine to provide different
  /// behavior.
  TemplateName RebuildTemplateName(NestedNameSpecifier *Qualifier,
                                   OverloadedOperatorKind Operator,
                                   QualType ObjectType);
  
  /// \brief Build a new compound statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildCompoundStmt(SourceLocation LBraceLoc,
                                       MultiStmtArg Statements,
                                       SourceLocation RBraceLoc,
                                       bool IsStmtExpr) {
    return getSema().ActOnCompoundStmt(LBraceLoc, RBraceLoc, Statements,
                                       IsStmtExpr);
  }

  /// \brief Build a new case statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildCaseStmt(SourceLocation CaseLoc,
                                   Expr *LHS,
                                   SourceLocation EllipsisLoc,
                                   Expr *RHS,
                                   SourceLocation ColonLoc) {
    return getSema().ActOnCaseStmt(CaseLoc, LHS, EllipsisLoc, RHS,
                                   ColonLoc);
  }

  /// \brief Attach the body to a new case statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildCaseStmtBody(Stmt *S, Stmt *Body) {
    getSema().ActOnCaseStmtBody(S, Body);
    return S;
  }

  /// \brief Build a new default statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildDefaultStmt(SourceLocation DefaultLoc,
                                      SourceLocation ColonLoc,
                                      Stmt *SubStmt) {
    return getSema().ActOnDefaultStmt(DefaultLoc, ColonLoc, SubStmt,
                                      /*CurScope=*/0);
  }

  /// \brief Build a new label statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildLabelStmt(SourceLocation IdentLoc,
                                    IdentifierInfo *Id,
                                    SourceLocation ColonLoc,
                                    Stmt *SubStmt, bool HasUnusedAttr) {
    return SemaRef.ActOnLabelStmt(IdentLoc, Id, ColonLoc, SubStmt,
                                  HasUnusedAttr);
  }

  /// \brief Build a new "if" statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildIfStmt(SourceLocation IfLoc, Sema::FullExprArg Cond,
                                 VarDecl *CondVar, Stmt *Then, 
                                 SourceLocation ElseLoc, Stmt *Else) {
    return getSema().ActOnIfStmt(IfLoc, Cond, CondVar, Then, ElseLoc, Else);
  }

  /// \brief Start building a new switch statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildSwitchStmtStart(SourceLocation SwitchLoc,
                                          Expr *Cond, VarDecl *CondVar) {
    return getSema().ActOnStartOfSwitchStmt(SwitchLoc, Cond, 
                                            CondVar);
  }

  /// \brief Attach the body to the switch statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildSwitchStmtBody(SourceLocation SwitchLoc,
                                         Stmt *Switch, Stmt *Body) {
    return getSema().ActOnFinishSwitchStmt(SwitchLoc, Switch, Body);
  }

  /// \brief Build a new while statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildWhileStmt(SourceLocation WhileLoc,
                                    Sema::FullExprArg Cond,
                                    VarDecl *CondVar,
                                    Stmt *Body) {
    return getSema().ActOnWhileStmt(WhileLoc, Cond, CondVar, Body);
  }

  /// \brief Build a new do-while statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildDoStmt(SourceLocation DoLoc, Stmt *Body,
                                 SourceLocation WhileLoc,
                                 SourceLocation LParenLoc,
                                 Expr *Cond,
                                 SourceLocation RParenLoc) {
    return getSema().ActOnDoStmt(DoLoc, Body, WhileLoc, LParenLoc,
                                 Cond, RParenLoc);
  }

  /// \brief Build a new for statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildForStmt(SourceLocation ForLoc,
                                  SourceLocation LParenLoc,
                                  Stmt *Init, Sema::FullExprArg Cond, 
                                  VarDecl *CondVar, Sema::FullExprArg Inc,
                                  SourceLocation RParenLoc, Stmt *Body) {
    return getSema().ActOnForStmt(ForLoc, LParenLoc, Init, Cond, 
                                  CondVar,
                                  Inc, RParenLoc, Body);
  }

  /// \brief Build a new goto statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildGotoStmt(SourceLocation GotoLoc,
                                   SourceLocation LabelLoc,
                                   LabelStmt *Label) {
    return getSema().ActOnGotoStmt(GotoLoc, LabelLoc, Label->getID());
  }

  /// \brief Build a new indirect goto statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildIndirectGotoStmt(SourceLocation GotoLoc,
                                           SourceLocation StarLoc,
                                           Expr *Target) {
    return getSema().ActOnIndirectGotoStmt(GotoLoc, StarLoc, Target);
  }

  /// \brief Build a new return statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildReturnStmt(SourceLocation ReturnLoc,
                                     Expr *Result) {

    return getSema().ActOnReturnStmt(ReturnLoc, Result);
  }

  /// \brief Build a new declaration statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildDeclStmt(Decl **Decls, unsigned NumDecls,
                                   SourceLocation StartLoc,
                                   SourceLocation EndLoc) {
    return getSema().Owned(
             new (getSema().Context) DeclStmt(
                                        DeclGroupRef::Create(getSema().Context,
                                                             Decls, NumDecls),
                                              StartLoc, EndLoc));
  }

  /// \brief Build a new inline asm statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildAsmStmt(SourceLocation AsmLoc,
                                  bool IsSimple,
                                  bool IsVolatile,
                                  unsigned NumOutputs,
                                  unsigned NumInputs,
                                  IdentifierInfo **Names,
                                  MultiExprArg Constraints,
                                  MultiExprArg Exprs,
                                  Expr *AsmString,
                                  MultiExprArg Clobbers,
                                  SourceLocation RParenLoc,
                                  bool MSAsm) {
    return getSema().ActOnAsmStmt(AsmLoc, IsSimple, IsVolatile, NumOutputs, 
                                  NumInputs, Names, move(Constraints),
                                  Exprs, AsmString, Clobbers,
                                  RParenLoc, MSAsm);
  }

  /// \brief Build a new Objective-C @try statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildObjCAtTryStmt(SourceLocation AtLoc,
                                        Stmt *TryBody,
                                        MultiStmtArg CatchStmts,
                                        Stmt *Finally) {
    return getSema().ActOnObjCAtTryStmt(AtLoc, TryBody, move(CatchStmts),
                                        Finally);
  }

  /// \brief Rebuild an Objective-C exception declaration.
  ///
  /// By default, performs semantic analysis to build the new declaration.
  /// Subclasses may override this routine to provide different behavior.
  VarDecl *RebuildObjCExceptionDecl(VarDecl *ExceptionDecl,
                                    TypeSourceInfo *TInfo, QualType T) {
    return getSema().BuildObjCExceptionDecl(TInfo, T, 
                                            ExceptionDecl->getIdentifier(), 
                                            ExceptionDecl->getLocation());
  }
  
  /// \brief Build a new Objective-C @catch statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildObjCAtCatchStmt(SourceLocation AtLoc,
                                          SourceLocation RParenLoc,
                                          VarDecl *Var,
                                          Stmt *Body) {
    return getSema().ActOnObjCAtCatchStmt(AtLoc, RParenLoc,
                                          Var, Body);
  }
  
  /// \brief Build a new Objective-C @finally statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildObjCAtFinallyStmt(SourceLocation AtLoc,
                                            Stmt *Body) {
    return getSema().ActOnObjCAtFinallyStmt(AtLoc, Body);
  }
  
  /// \brief Build a new Objective-C @throw statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildObjCAtThrowStmt(SourceLocation AtLoc,
                                          Expr *Operand) {
    return getSema().BuildObjCAtThrowStmt(AtLoc, Operand);
  }
  
  /// \brief Build a new Objective-C @synchronized statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildObjCAtSynchronizedStmt(SourceLocation AtLoc,
                                                 Expr *Object,
                                                 Stmt *Body) {
    return getSema().ActOnObjCAtSynchronizedStmt(AtLoc, Object,
                                                 Body);
  }

  /// \brief Build a new Objective-C fast enumeration statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildObjCForCollectionStmt(SourceLocation ForLoc,
                                          SourceLocation LParenLoc,
                                          Stmt *Element,
                                          Expr *Collection,
                                          SourceLocation RParenLoc,
                                          Stmt *Body) {
    return getSema().ActOnObjCForCollectionStmt(ForLoc, LParenLoc,
                                                Element, 
                                                Collection,
                                                RParenLoc,
                                                Body);
  }
  
  /// \brief Build a new C++ exception declaration.
  ///
  /// By default, performs semantic analysis to build the new decaration.
  /// Subclasses may override this routine to provide different behavior.
  VarDecl *RebuildExceptionDecl(VarDecl *ExceptionDecl, 
                                TypeSourceInfo *Declarator,
                                IdentifierInfo *Name,
                                SourceLocation Loc) {
    return getSema().BuildExceptionDeclaration(0, Declarator, Name, Loc);
  }

  /// \brief Build a new C++ catch statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildCXXCatchStmt(SourceLocation CatchLoc,
                                 VarDecl *ExceptionDecl,
                                 Stmt *Handler) {
    return Owned(new (getSema().Context) CXXCatchStmt(CatchLoc, ExceptionDecl,
                                                      Handler));
  }

  /// \brief Build a new C++ try statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildCXXTryStmt(SourceLocation TryLoc,
                               Stmt *TryBlock,
                               MultiStmtArg Handlers) {
    return getSema().ActOnCXXTryBlock(TryLoc, TryBlock, move(Handlers));
  }

  /// \brief Build a new expression that references a declaration.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildDeclarationNameExpr(const CXXScopeSpec &SS,
                                        LookupResult &R,
                                        bool RequiresADL) {
    return getSema().BuildDeclarationNameExpr(SS, R, RequiresADL);
  }


  /// \brief Build a new expression that references a declaration.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildDeclRefExpr(NestedNameSpecifier *Qualifier,
                                SourceRange QualifierRange,
                                ValueDecl *VD,
                                const DeclarationNameInfo &NameInfo,
                                TemplateArgumentListInfo *TemplateArgs) {
    CXXScopeSpec SS;
    SS.setScopeRep(Qualifier);
    SS.setRange(QualifierRange);

    // FIXME: loses template args.

    return getSema().BuildDeclarationNameExpr(SS, NameInfo, VD);
  }

  /// \brief Build a new expression in parentheses.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildParenExpr(Expr *SubExpr, SourceLocation LParen,
                                    SourceLocation RParen) {
    return getSema().ActOnParenExpr(LParen, RParen, SubExpr);
  }

  /// \brief Build a new pseudo-destructor expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildCXXPseudoDestructorExpr(Expr *Base,
                                                  SourceLocation OperatorLoc,
                                                  bool isArrow,
                                                NestedNameSpecifier *Qualifier,
                                                  SourceRange QualifierRange,
                                                  TypeSourceInfo *ScopeType,
                                                  SourceLocation CCLoc,
                                                  SourceLocation TildeLoc,
                                        PseudoDestructorTypeStorage Destroyed);

  /// \brief Build a new unary operator expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildUnaryOperator(SourceLocation OpLoc,
                                        UnaryOperatorKind Opc,
                                        Expr *SubExpr) {
    return getSema().BuildUnaryOp(/*Scope=*/0, OpLoc, Opc, SubExpr);
  }

  /// \brief Build a new builtin offsetof expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildOffsetOfExpr(SourceLocation OperatorLoc,
                                       TypeSourceInfo *Type,
                                       Sema::OffsetOfComponent *Components,
                                       unsigned NumComponents,
                                       SourceLocation RParenLoc) {
    return getSema().BuildBuiltinOffsetOf(OperatorLoc, Type, Components,
                                          NumComponents, RParenLoc);
  }
  
  /// \brief Build a new sizeof or alignof expression with a type argument.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildSizeOfAlignOf(TypeSourceInfo *TInfo,
                                        SourceLocation OpLoc,
                                        bool isSizeOf, SourceRange R) {
    return getSema().CreateSizeOfAlignOfExpr(TInfo, OpLoc, isSizeOf, R);
  }

  /// \brief Build a new sizeof or alignof expression with an expression
  /// argument.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildSizeOfAlignOf(Expr *SubExpr, SourceLocation OpLoc,
                                        bool isSizeOf, SourceRange R) {
    ExprResult Result
      = getSema().CreateSizeOfAlignOfExpr(SubExpr, OpLoc, isSizeOf, R);
    if (Result.isInvalid())
      return ExprError();

    return move(Result);
  }

  /// \brief Build a new array subscript expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildArraySubscriptExpr(Expr *LHS,
                                             SourceLocation LBracketLoc,
                                             Expr *RHS,
                                             SourceLocation RBracketLoc) {
    return getSema().ActOnArraySubscriptExpr(/*Scope=*/0, LHS,
                                             LBracketLoc, RHS,
                                             RBracketLoc);
  }

  /// \brief Build a new call expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildCallExpr(Expr *Callee, SourceLocation LParenLoc,
                                   MultiExprArg Args,
                                   SourceLocation RParenLoc) {
    return getSema().ActOnCallExpr(/*Scope=*/0, Callee, LParenLoc,
                                   move(Args), RParenLoc);
  }

  /// \brief Build a new member access expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildMemberExpr(Expr *Base, SourceLocation OpLoc,
                               bool isArrow,
                               NestedNameSpecifier *Qualifier,
                               SourceRange QualifierRange,
                               const DeclarationNameInfo &MemberNameInfo,
                               ValueDecl *Member,
                               NamedDecl *FoundDecl,
                        const TemplateArgumentListInfo *ExplicitTemplateArgs,
                               NamedDecl *FirstQualifierInScope) {
    if (!Member->getDeclName()) {
      // We have a reference to an unnamed field.  This is always the
      // base of an anonymous struct/union member access, i.e. the
      // field is always of record type.
      assert(!Qualifier && "Can't have an unnamed field with a qualifier!");
      assert(Member->getType()->isRecordType() &&
             "unnamed member not of record type?");

      if (getSema().PerformObjectMemberConversion(Base, Qualifier,
                                                  FoundDecl, Member))
        return ExprError();

      ExprValueKind VK = isArrow ? VK_LValue : Base->getValueKind();
      MemberExpr *ME =
        new (getSema().Context) MemberExpr(Base, isArrow,
                                           Member, MemberNameInfo,
                                           cast<FieldDecl>(Member)->getType(),
                                           VK, OK_Ordinary);
      return getSema().Owned(ME);
    }

    CXXScopeSpec SS;
    if (Qualifier) {
      SS.setRange(QualifierRange);
      SS.setScopeRep(Qualifier);
    }

    getSema().DefaultFunctionArrayConversion(Base);
    QualType BaseType = Base->getType();

    // FIXME: this involves duplicating earlier analysis in a lot of
    // cases; we should avoid this when possible.
    LookupResult R(getSema(), MemberNameInfo, Sema::LookupMemberName);
    R.addDecl(FoundDecl);
    R.resolveKind();

    return getSema().BuildMemberReferenceExpr(Base, BaseType, OpLoc, isArrow,
                                              SS, FirstQualifierInScope,
                                              R, ExplicitTemplateArgs);
  }

  /// \brief Build a new binary operator expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildBinaryOperator(SourceLocation OpLoc,
                                         BinaryOperatorKind Opc,
                                         Expr *LHS, Expr *RHS) {
    return getSema().BuildBinOp(/*Scope=*/0, OpLoc, Opc, LHS, RHS);
  }

  /// \brief Build a new conditional operator expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildConditionalOperator(Expr *Cond,
                                              SourceLocation QuestionLoc,
                                              Expr *LHS,
                                              SourceLocation ColonLoc,
                                              Expr *RHS) {
    return getSema().ActOnConditionalOp(QuestionLoc, ColonLoc, Cond,
                                        LHS, RHS);
  }

  /// \brief Build a new C-style cast expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildCStyleCastExpr(SourceLocation LParenLoc,
                                         TypeSourceInfo *TInfo,
                                         SourceLocation RParenLoc,
                                         Expr *SubExpr) {
    return getSema().BuildCStyleCastExpr(LParenLoc, TInfo, RParenLoc,
                                         SubExpr);
  }

  /// \brief Build a new compound literal expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildCompoundLiteralExpr(SourceLocation LParenLoc,
                                              TypeSourceInfo *TInfo,
                                              SourceLocation RParenLoc,
                                              Expr *Init) {
    return getSema().BuildCompoundLiteralExpr(LParenLoc, TInfo, RParenLoc,
                                              Init);
  }

  /// \brief Build a new extended vector element access expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildExtVectorElementExpr(Expr *Base,
                                               SourceLocation OpLoc,
                                               SourceLocation AccessorLoc,
                                               IdentifierInfo &Accessor) {

    CXXScopeSpec SS;
    DeclarationNameInfo NameInfo(&Accessor, AccessorLoc);
    return getSema().BuildMemberReferenceExpr(Base, Base->getType(),
                                              OpLoc, /*IsArrow*/ false,
                                              SS, /*FirstQualifierInScope*/ 0,
                                              NameInfo,
                                              /* TemplateArgs */ 0);
  }

  /// \brief Build a new initializer list expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildInitList(SourceLocation LBraceLoc,
                                   MultiExprArg Inits,
                                   SourceLocation RBraceLoc,
                                   QualType ResultTy) {
    ExprResult Result
      = SemaRef.ActOnInitList(LBraceLoc, move(Inits), RBraceLoc);
    if (Result.isInvalid() || ResultTy->isDependentType())
      return move(Result);
    
    // Patch in the result type we were given, which may have been computed
    // when the initial InitListExpr was built.
    InitListExpr *ILE = cast<InitListExpr>((Expr *)Result.get());
    ILE->setType(ResultTy);
    return move(Result);
  }

  /// \brief Build a new designated initializer expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildDesignatedInitExpr(Designation &Desig,
                                             MultiExprArg ArrayExprs,
                                             SourceLocation EqualOrColonLoc,
                                             bool GNUSyntax,
                                             Expr *Init) {
    ExprResult Result
      = SemaRef.ActOnDesignatedInitializer(Desig, EqualOrColonLoc, GNUSyntax,
                                           Init);
    if (Result.isInvalid())
      return ExprError();

    ArrayExprs.release();
    return move(Result);
  }

  /// \brief Build a new value-initialized expression.
  ///
  /// By default, builds the implicit value initialization without performing
  /// any semantic analysis. Subclasses may override this routine to provide
  /// different behavior.
  ExprResult RebuildImplicitValueInitExpr(QualType T) {
    return SemaRef.Owned(new (SemaRef.Context) ImplicitValueInitExpr(T));
  }

  /// \brief Build a new \c va_arg expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildVAArgExpr(SourceLocation BuiltinLoc,
                                    Expr *SubExpr, TypeSourceInfo *TInfo,
                                    SourceLocation RParenLoc) {
    return getSema().BuildVAArgExpr(BuiltinLoc,
                                    SubExpr, TInfo,
                                    RParenLoc);
  }

  /// \brief Build a new expression list in parentheses.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildParenListExpr(SourceLocation LParenLoc,
                                        MultiExprArg SubExprs,
                                        SourceLocation RParenLoc) {
    return getSema().ActOnParenOrParenListExpr(LParenLoc, RParenLoc, 
                                               move(SubExprs));
  }

  /// \brief Build a new address-of-label expression.
  ///
  /// By default, performs semantic analysis, using the name of the label
  /// rather than attempting to map the label statement itself.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildAddrLabelExpr(SourceLocation AmpAmpLoc,
                                        SourceLocation LabelLoc,
                                        LabelStmt *Label) {
    return getSema().ActOnAddrLabel(AmpAmpLoc, LabelLoc, Label->getID());
  }

  /// \brief Build a new GNU statement expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildStmtExpr(SourceLocation LParenLoc,
                                   Stmt *SubStmt,
                                   SourceLocation RParenLoc) {
    return getSema().ActOnStmtExpr(LParenLoc, SubStmt, RParenLoc);
  }

  /// \brief Build a new __builtin_choose_expr expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildChooseExpr(SourceLocation BuiltinLoc,
                                     Expr *Cond, Expr *LHS, Expr *RHS,
                                     SourceLocation RParenLoc) {
    return SemaRef.ActOnChooseExpr(BuiltinLoc,
                                   Cond, LHS, RHS,
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
  ExprResult RebuildCXXOperatorCallExpr(OverloadedOperatorKind Op,
                                              SourceLocation OpLoc,
                                              Expr *Callee,
                                              Expr *First,
                                              Expr *Second);

  /// \brief Build a new C++ "named" cast expression, such as static_cast or
  /// reinterpret_cast.
  ///
  /// By default, this routine dispatches to one of the more-specific routines
  /// for a particular named case, e.g., RebuildCXXStaticCastExpr().
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildCXXNamedCastExpr(SourceLocation OpLoc,
                                           Stmt::StmtClass Class,
                                           SourceLocation LAngleLoc,
                                           TypeSourceInfo *TInfo,
                                           SourceLocation RAngleLoc,
                                           SourceLocation LParenLoc,
                                           Expr *SubExpr,
                                           SourceLocation RParenLoc) {
    switch (Class) {
    case Stmt::CXXStaticCastExprClass:
      return getDerived().RebuildCXXStaticCastExpr(OpLoc, LAngleLoc, TInfo,
                                                   RAngleLoc, LParenLoc,
                                                   SubExpr, RParenLoc);

    case Stmt::CXXDynamicCastExprClass:
      return getDerived().RebuildCXXDynamicCastExpr(OpLoc, LAngleLoc, TInfo,
                                                    RAngleLoc, LParenLoc,
                                                    SubExpr, RParenLoc);

    case Stmt::CXXReinterpretCastExprClass:
      return getDerived().RebuildCXXReinterpretCastExpr(OpLoc, LAngleLoc, TInfo,
                                                        RAngleLoc, LParenLoc,
                                                        SubExpr,
                                                        RParenLoc);

    case Stmt::CXXConstCastExprClass:
      return getDerived().RebuildCXXConstCastExpr(OpLoc, LAngleLoc, TInfo,
                                                   RAngleLoc, LParenLoc,
                                                   SubExpr, RParenLoc);

    default:
      assert(false && "Invalid C++ named cast");
      break;
    }

    return ExprError();
  }

  /// \brief Build a new C++ static_cast expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildCXXStaticCastExpr(SourceLocation OpLoc,
                                            SourceLocation LAngleLoc,
                                            TypeSourceInfo *TInfo,
                                            SourceLocation RAngleLoc,
                                            SourceLocation LParenLoc,
                                            Expr *SubExpr,
                                            SourceLocation RParenLoc) {
    return getSema().BuildCXXNamedCast(OpLoc, tok::kw_static_cast,
                                       TInfo, SubExpr,
                                       SourceRange(LAngleLoc, RAngleLoc),
                                       SourceRange(LParenLoc, RParenLoc));
  }

  /// \brief Build a new C++ dynamic_cast expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildCXXDynamicCastExpr(SourceLocation OpLoc,
                                             SourceLocation LAngleLoc,
                                             TypeSourceInfo *TInfo,
                                             SourceLocation RAngleLoc,
                                             SourceLocation LParenLoc,
                                             Expr *SubExpr,
                                             SourceLocation RParenLoc) {
    return getSema().BuildCXXNamedCast(OpLoc, tok::kw_dynamic_cast,
                                       TInfo, SubExpr,
                                       SourceRange(LAngleLoc, RAngleLoc),
                                       SourceRange(LParenLoc, RParenLoc));
  }

  /// \brief Build a new C++ reinterpret_cast expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildCXXReinterpretCastExpr(SourceLocation OpLoc,
                                                 SourceLocation LAngleLoc,
                                                 TypeSourceInfo *TInfo,
                                                 SourceLocation RAngleLoc,
                                                 SourceLocation LParenLoc,
                                                 Expr *SubExpr,
                                                 SourceLocation RParenLoc) {
    return getSema().BuildCXXNamedCast(OpLoc, tok::kw_reinterpret_cast,
                                       TInfo, SubExpr,
                                       SourceRange(LAngleLoc, RAngleLoc),
                                       SourceRange(LParenLoc, RParenLoc));
  }

  /// \brief Build a new C++ const_cast expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildCXXConstCastExpr(SourceLocation OpLoc,
                                           SourceLocation LAngleLoc,
                                           TypeSourceInfo *TInfo,
                                           SourceLocation RAngleLoc,
                                           SourceLocation LParenLoc,
                                           Expr *SubExpr,
                                           SourceLocation RParenLoc) {
    return getSema().BuildCXXNamedCast(OpLoc, tok::kw_const_cast,
                                       TInfo, SubExpr,
                                       SourceRange(LAngleLoc, RAngleLoc),
                                       SourceRange(LParenLoc, RParenLoc));
  }

  /// \brief Build a new C++ functional-style cast expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildCXXFunctionalCastExpr(TypeSourceInfo *TInfo,
                                          SourceLocation LParenLoc,
                                          Expr *Sub,
                                          SourceLocation RParenLoc) {
    return getSema().BuildCXXTypeConstructExpr(TInfo, LParenLoc,
                                               MultiExprArg(&Sub, 1),
                                               RParenLoc);
  }

  /// \brief Build a new C++ typeid(type) expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildCXXTypeidExpr(QualType TypeInfoType,
                                        SourceLocation TypeidLoc,
                                        TypeSourceInfo *Operand,
                                        SourceLocation RParenLoc) {
    return getSema().BuildCXXTypeId(TypeInfoType, TypeidLoc, Operand, 
                                    RParenLoc);
  }


  /// \brief Build a new C++ typeid(expr) expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildCXXTypeidExpr(QualType TypeInfoType,
                                        SourceLocation TypeidLoc,
                                        Expr *Operand,
                                        SourceLocation RParenLoc) {
    return getSema().BuildCXXTypeId(TypeInfoType, TypeidLoc, Operand,
                                    RParenLoc);
  }

  /// \brief Build a new C++ __uuidof(type) expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildCXXUuidofExpr(QualType TypeInfoType,
                                        SourceLocation TypeidLoc,
                                        TypeSourceInfo *Operand,
                                        SourceLocation RParenLoc) {
    return getSema().BuildCXXUuidof(TypeInfoType, TypeidLoc, Operand, 
                                    RParenLoc);
  }

  /// \brief Build a new C++ __uuidof(expr) expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildCXXUuidofExpr(QualType TypeInfoType,
                                        SourceLocation TypeidLoc,
                                        Expr *Operand,
                                        SourceLocation RParenLoc) {
    return getSema().BuildCXXUuidof(TypeInfoType, TypeidLoc, Operand,
                                    RParenLoc);
  }

  /// \brief Build a new C++ "this" expression.
  ///
  /// By default, builds a new "this" expression without performing any
  /// semantic analysis. Subclasses may override this routine to provide
  /// different behavior.
  ExprResult RebuildCXXThisExpr(SourceLocation ThisLoc,
                                QualType ThisType,
                                bool isImplicit) {
    return getSema().Owned(
                      new (getSema().Context) CXXThisExpr(ThisLoc, ThisType,
                                                          isImplicit));
  }

  /// \brief Build a new C++ throw expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildCXXThrowExpr(SourceLocation ThrowLoc, Expr *Sub) {
    return getSema().ActOnCXXThrow(ThrowLoc, Sub);
  }

  /// \brief Build a new C++ default-argument expression.
  ///
  /// By default, builds a new default-argument expression, which does not
  /// require any semantic analysis. Subclasses may override this routine to
  /// provide different behavior.
  ExprResult RebuildCXXDefaultArgExpr(SourceLocation Loc, 
                                            ParmVarDecl *Param) {
    return getSema().Owned(CXXDefaultArgExpr::Create(getSema().Context, Loc,
                                                     Param));
  }

  /// \brief Build a new C++ zero-initialization expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildCXXScalarValueInitExpr(TypeSourceInfo *TSInfo,
                                           SourceLocation LParenLoc,
                                           SourceLocation RParenLoc) {
    return getSema().BuildCXXTypeConstructExpr(TSInfo, LParenLoc,
                                               MultiExprArg(getSema(), 0, 0),
                                               RParenLoc);
  }

  /// \brief Build a new C++ "new" expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildCXXNewExpr(SourceLocation StartLoc,
                               bool UseGlobal,
                               SourceLocation PlacementLParen,
                               MultiExprArg PlacementArgs,
                               SourceLocation PlacementRParen,
                               SourceRange TypeIdParens,
                               QualType AllocatedType,
                               TypeSourceInfo *AllocatedTypeInfo,
                               Expr *ArraySize,
                               SourceLocation ConstructorLParen,
                               MultiExprArg ConstructorArgs,
                               SourceLocation ConstructorRParen) {
    return getSema().BuildCXXNew(StartLoc, UseGlobal,
                                 PlacementLParen,
                                 move(PlacementArgs),
                                 PlacementRParen,
                                 TypeIdParens,
                                 AllocatedType,
                                 AllocatedTypeInfo,
                                 ArraySize,
                                 ConstructorLParen,
                                 move(ConstructorArgs),
                                 ConstructorRParen);
  }

  /// \brief Build a new C++ "delete" expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildCXXDeleteExpr(SourceLocation StartLoc,
                                        bool IsGlobalDelete,
                                        bool IsArrayForm,
                                        Expr *Operand) {
    return getSema().ActOnCXXDelete(StartLoc, IsGlobalDelete, IsArrayForm,
                                    Operand);
  }

  /// \brief Build a new unary type trait expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildUnaryTypeTrait(UnaryTypeTrait Trait,
                                   SourceLocation StartLoc,
                                   TypeSourceInfo *T,
                                   SourceLocation RParenLoc) {
    return getSema().BuildUnaryTypeTrait(Trait, StartLoc, T, RParenLoc);
  }

  /// \brief Build a new binary type trait expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildBinaryTypeTrait(BinaryTypeTrait Trait,
                                    SourceLocation StartLoc,
                                    TypeSourceInfo *LhsT,
                                    TypeSourceInfo *RhsT,
                                    SourceLocation RParenLoc) {
    return getSema().BuildBinaryTypeTrait(Trait, StartLoc, LhsT, RhsT, RParenLoc);
  }

  /// \brief Build a new (previously unresolved) declaration reference
  /// expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildDependentScopeDeclRefExpr(NestedNameSpecifier *NNS,
                                                SourceRange QualifierRange,
                                       const DeclarationNameInfo &NameInfo,
                              const TemplateArgumentListInfo *TemplateArgs) {
    CXXScopeSpec SS;
    SS.setRange(QualifierRange);
    SS.setScopeRep(NNS);

    if (TemplateArgs)
      return getSema().BuildQualifiedTemplateIdExpr(SS, NameInfo,
                                                    *TemplateArgs);

    return getSema().BuildQualifiedDeclarationNameExpr(SS, NameInfo);
  }

  /// \brief Build a new template-id expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildTemplateIdExpr(const CXXScopeSpec &SS,
                                         LookupResult &R,
                                         bool RequiresADL,
                              const TemplateArgumentListInfo &TemplateArgs) {
    return getSema().BuildTemplateIdExpr(SS, R, RequiresADL, TemplateArgs);
  }

  /// \brief Build a new object-construction expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildCXXConstructExpr(QualType T,
                                           SourceLocation Loc,
                                           CXXConstructorDecl *Constructor,
                                           bool IsElidable,
                                           MultiExprArg Args,
                                           bool RequiresZeroInit,
                             CXXConstructExpr::ConstructionKind ConstructKind,
                                           SourceRange ParenRange) {
    ASTOwningVector<Expr*> ConvertedArgs(SemaRef);
    if (getSema().CompleteConstructorCall(Constructor, move(Args), Loc, 
                                          ConvertedArgs))
      return ExprError();
    
    return getSema().BuildCXXConstructExpr(Loc, T, Constructor, IsElidable,
                                           move_arg(ConvertedArgs),
                                           RequiresZeroInit, ConstructKind,
                                           ParenRange);
  }

  /// \brief Build a new object-construction expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildCXXTemporaryObjectExpr(TypeSourceInfo *TSInfo,
                                           SourceLocation LParenLoc,
                                           MultiExprArg Args,
                                           SourceLocation RParenLoc) {
    return getSema().BuildCXXTypeConstructExpr(TSInfo,
                                               LParenLoc,
                                               move(Args),
                                               RParenLoc);
  }

  /// \brief Build a new object-construction expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildCXXUnresolvedConstructExpr(TypeSourceInfo *TSInfo,
                                               SourceLocation LParenLoc,
                                               MultiExprArg Args,
                                               SourceLocation RParenLoc) {
    return getSema().BuildCXXTypeConstructExpr(TSInfo,
                                               LParenLoc,
                                               move(Args),
                                               RParenLoc);
  }

  /// \brief Build a new member reference expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildCXXDependentScopeMemberExpr(Expr *BaseE,
                                                  QualType BaseType,
                                                  bool IsArrow,
                                                  SourceLocation OperatorLoc,
                                              NestedNameSpecifier *Qualifier,
                                                  SourceRange QualifierRange,
                                            NamedDecl *FirstQualifierInScope,
                                   const DeclarationNameInfo &MemberNameInfo,
                              const TemplateArgumentListInfo *TemplateArgs) {
    CXXScopeSpec SS;
    SS.setRange(QualifierRange);
    SS.setScopeRep(Qualifier);

    return SemaRef.BuildMemberReferenceExpr(BaseE, BaseType,
                                            OperatorLoc, IsArrow,
                                            SS, FirstQualifierInScope,
                                            MemberNameInfo,
                                            TemplateArgs);
  }

  /// \brief Build a new member reference expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildUnresolvedMemberExpr(Expr *BaseE,
                                               QualType BaseType,
                                               SourceLocation OperatorLoc,
                                               bool IsArrow,
                                               NestedNameSpecifier *Qualifier,
                                               SourceRange QualifierRange,
                                               NamedDecl *FirstQualifierInScope,
                                               LookupResult &R,
                                const TemplateArgumentListInfo *TemplateArgs) {
    CXXScopeSpec SS;
    SS.setRange(QualifierRange);
    SS.setScopeRep(Qualifier);

    return SemaRef.BuildMemberReferenceExpr(BaseE, BaseType,
                                            OperatorLoc, IsArrow,
                                            SS, FirstQualifierInScope,
                                            R, TemplateArgs);
  }

  /// \brief Build a new noexcept expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildCXXNoexceptExpr(SourceRange Range, Expr *Arg) {
    return SemaRef.BuildCXXNoexceptExpr(Range.getBegin(), Arg, Range.getEnd());
  }

  /// \brief Build a new Objective-C @encode expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildObjCEncodeExpr(SourceLocation AtLoc,
                                         TypeSourceInfo *EncodeTypeInfo,
                                         SourceLocation RParenLoc) {
    return SemaRef.Owned(SemaRef.BuildObjCEncodeExpression(AtLoc, EncodeTypeInfo,
                                                           RParenLoc));
  }

  /// \brief Build a new Objective-C class message.
  ExprResult RebuildObjCMessageExpr(TypeSourceInfo *ReceiverTypeInfo,
                                          Selector Sel,
                                          SourceLocation SelectorLoc,
                                          ObjCMethodDecl *Method,
                                          SourceLocation LBracLoc, 
                                          MultiExprArg Args,
                                          SourceLocation RBracLoc) {
    return SemaRef.BuildClassMessage(ReceiverTypeInfo,
                                     ReceiverTypeInfo->getType(),
                                     /*SuperLoc=*/SourceLocation(),
                                     Sel, Method, LBracLoc, SelectorLoc,
                                     RBracLoc, move(Args));
  }

  /// \brief Build a new Objective-C instance message.
  ExprResult RebuildObjCMessageExpr(Expr *Receiver,
                                          Selector Sel,
                                          SourceLocation SelectorLoc,
                                          ObjCMethodDecl *Method,
                                          SourceLocation LBracLoc, 
                                          MultiExprArg Args,
                                          SourceLocation RBracLoc) {
    return SemaRef.BuildInstanceMessage(Receiver,
                                        Receiver->getType(),
                                        /*SuperLoc=*/SourceLocation(),
                                        Sel, Method, LBracLoc, SelectorLoc,
                                        RBracLoc, move(Args));
  }

  /// \brief Build a new Objective-C ivar reference expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildObjCIvarRefExpr(Expr *BaseArg, ObjCIvarDecl *Ivar,
                                          SourceLocation IvarLoc,
                                          bool IsArrow, bool IsFreeIvar) {
    // FIXME: We lose track of the IsFreeIvar bit.
    CXXScopeSpec SS;
    Expr *Base = BaseArg;
    LookupResult R(getSema(), Ivar->getDeclName(), IvarLoc,
                   Sema::LookupMemberName);
    ExprResult Result = getSema().LookupMemberExpr(R, Base, IsArrow,
                                                         /*FIME:*/IvarLoc,
                                                         SS, 0,
                                                         false);
    if (Result.isInvalid())
      return ExprError();
    
    if (Result.get())
      return move(Result);
    
    return getSema().BuildMemberReferenceExpr(Base, Base->getType(),
                                              /*FIXME:*/IvarLoc, IsArrow, SS, 
                                              /*FirstQualifierInScope=*/0,
                                              R, 
                                              /*TemplateArgs=*/0);
  }

  /// \brief Build a new Objective-C property reference expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildObjCPropertyRefExpr(Expr *BaseArg, 
                                              ObjCPropertyDecl *Property,
                                              SourceLocation PropertyLoc) {
    CXXScopeSpec SS;
    Expr *Base = BaseArg;
    LookupResult R(getSema(), Property->getDeclName(), PropertyLoc,
                   Sema::LookupMemberName);
    bool IsArrow = false;
    ExprResult Result = getSema().LookupMemberExpr(R, Base, IsArrow,
                                                         /*FIME:*/PropertyLoc,
                                                         SS, 0, false);
    if (Result.isInvalid())
      return ExprError();
    
    if (Result.get())
      return move(Result);
    
    return getSema().BuildMemberReferenceExpr(Base, Base->getType(),
                                              /*FIXME:*/PropertyLoc, IsArrow, 
                                              SS, 
                                              /*FirstQualifierInScope=*/0,
                                              R, 
                                              /*TemplateArgs=*/0);
  }
  
  /// \brief Build a new Objective-C property reference expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildObjCPropertyRefExpr(Expr *Base, QualType T,
                                        ObjCMethodDecl *Getter,
                                        ObjCMethodDecl *Setter,
                                        SourceLocation PropertyLoc) {
    // Since these expressions can only be value-dependent, we do not
    // need to perform semantic analysis again.
    return Owned(
      new (getSema().Context) ObjCPropertyRefExpr(Getter, Setter, T,
                                                  VK_LValue, OK_ObjCProperty,
                                                  PropertyLoc, Base));
  }

  /// \brief Build a new Objective-C "isa" expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildObjCIsaExpr(Expr *BaseArg, SourceLocation IsaLoc,
                                      bool IsArrow) {
    CXXScopeSpec SS;
    Expr *Base = BaseArg;
    LookupResult R(getSema(), &getSema().Context.Idents.get("isa"), IsaLoc,
                   Sema::LookupMemberName);
    ExprResult Result = getSema().LookupMemberExpr(R, Base, IsArrow,
                                                         /*FIME:*/IsaLoc,
                                                         SS, 0, false);
    if (Result.isInvalid())
      return ExprError();
    
    if (Result.get())
      return move(Result);
    
    return getSema().BuildMemberReferenceExpr(Base, Base->getType(),
                                              /*FIXME:*/IsaLoc, IsArrow, SS, 
                                              /*FirstQualifierInScope=*/0,
                                              R, 
                                              /*TemplateArgs=*/0);
  }
  
  /// \brief Build a new shuffle vector expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildShuffleVectorExpr(SourceLocation BuiltinLoc,
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
                                          VK_LValue, BuiltinLoc);
    SemaRef.UsualUnaryConversions(Callee);

    // Build the CallExpr
    unsigned NumSubExprs = SubExprs.size();
    Expr **Subs = (Expr **)SubExprs.release();
    CallExpr *TheCall = new (SemaRef.Context) CallExpr(SemaRef.Context, Callee,
                                                       Subs, NumSubExprs,
                                                   Builtin->getCallResultType(),
                            Expr::getValueKindForType(Builtin->getResultType()),
                                                       RParenLoc);
    ExprResult OwnedCall(SemaRef.Owned(TheCall));

    // Type-check the __builtin_shufflevector expression.
    ExprResult Result = SemaRef.SemaBuiltinShuffleVector(TheCall);
    if (Result.isInvalid())
      return ExprError();

    OwnedCall.release();
    return move(Result);
  }

private:
  QualType TransformTypeInObjectScope(QualType T,
                                      QualType ObjectType,
                                      NamedDecl *FirstQualifierInScope,
                                      NestedNameSpecifier *Prefix);

  TypeSourceInfo *TransformTypeInObjectScope(TypeSourceInfo *T,
                                             QualType ObjectType,
                                             NamedDecl *FirstQualifierInScope,
                                             NestedNameSpecifier *Prefix);
};

template<typename Derived>
StmtResult TreeTransform<Derived>::TransformStmt(Stmt *S) {
  if (!S)
    return SemaRef.Owned(S);

  switch (S->getStmtClass()) {
  case Stmt::NoStmtClass: break;

  // Transform individual statement nodes
#define STMT(Node, Parent)                                              \
  case Stmt::Node##Class: return getDerived().Transform##Node(cast<Node>(S));
#define EXPR(Node, Parent)
#include "clang/AST/StmtNodes.inc"

  // Transform expressions by calling TransformExpr.
#define STMT(Node, Parent)
#define ABSTRACT_STMT(Stmt)
#define EXPR(Node, Parent) case Stmt::Node##Class:
#include "clang/AST/StmtNodes.inc"
    {
      ExprResult E = getDerived().TransformExpr(cast<Expr>(S));
      if (E.isInvalid())
        return StmtError();

      return getSema().ActOnExprStmt(getSema().MakeFullExpr(E.take()));
    }
  }

  return SemaRef.Owned(S);
}


template<typename Derived>
ExprResult TreeTransform<Derived>::TransformExpr(Expr *E) {
  if (!E)
    return SemaRef.Owned(E);

  switch (E->getStmtClass()) {
    case Stmt::NoStmtClass: break;
#define STMT(Node, Parent) case Stmt::Node##Class: break;
#define ABSTRACT_STMT(Stmt)
#define EXPR(Node, Parent)                                              \
    case Stmt::Node##Class: return getDerived().Transform##Node(cast<Node>(E));
#include "clang/AST/StmtNodes.inc"
  }

  return SemaRef.Owned(E);
}

template<typename Derived>
NestedNameSpecifier *
TreeTransform<Derived>::TransformNestedNameSpecifier(NestedNameSpecifier *NNS,
                                                     SourceRange Range,
                                                     QualType ObjectType,
                                             NamedDecl *FirstQualifierInScope) {
  NestedNameSpecifier *Prefix = NNS->getPrefix();

  // Transform the prefix of this nested name specifier.
  if (Prefix) {
    Prefix = getDerived().TransformNestedNameSpecifier(Prefix, Range,
                                                       ObjectType,
                                                       FirstQualifierInScope);
    if (!Prefix)
      return 0;
  }

  switch (NNS->getKind()) {
  case NestedNameSpecifier::Identifier:
    if (Prefix) {
      // The object type and qualifier-in-scope really apply to the
      // leftmost entity.
      ObjectType = QualType();
      FirstQualifierInScope = 0;
    }

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
                                    getDerived().TransformDecl(Range.getBegin(),
                                                       NNS->getAsNamespace()));
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
    QualType T = TransformTypeInObjectScope(QualType(NNS->getAsType(), 0),
                                            ObjectType,
                                            FirstQualifierInScope,
                                            Prefix);
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
DeclarationNameInfo
TreeTransform<Derived>
::TransformDeclarationNameInfo(const DeclarationNameInfo &NameInfo) {
  DeclarationName Name = NameInfo.getName();
  if (!Name)
    return DeclarationNameInfo();

  switch (Name.getNameKind()) {
  case DeclarationName::Identifier:
  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
  case DeclarationName::CXXOperatorName:
  case DeclarationName::CXXLiteralOperatorName:
  case DeclarationName::CXXUsingDirective:
    return NameInfo;

  case DeclarationName::CXXConstructorName:
  case DeclarationName::CXXDestructorName:
  case DeclarationName::CXXConversionFunctionName: {
    TypeSourceInfo *NewTInfo;
    CanQualType NewCanTy;
    if (TypeSourceInfo *OldTInfo = NameInfo.getNamedTypeInfo()) {
      NewTInfo = getDerived().TransformType(OldTInfo);
      if (!NewTInfo)
        return DeclarationNameInfo();
      NewCanTy = SemaRef.Context.getCanonicalType(NewTInfo->getType());
    }
    else {
      NewTInfo = 0;
      TemporaryBase Rebase(*this, NameInfo.getLoc(), Name);
      QualType NewT = getDerived().TransformType(Name.getCXXNameType());
      if (NewT.isNull())
        return DeclarationNameInfo();
      NewCanTy = SemaRef.Context.getCanonicalType(NewT);
    }

    DeclarationName NewName
      = SemaRef.Context.DeclarationNames.getCXXSpecialName(Name.getNameKind(),
                                                           NewCanTy);
    DeclarationNameInfo NewNameInfo(NameInfo);
    NewNameInfo.setName(NewName);
    NewNameInfo.setNamedTypeInfo(NewTInfo);
    return NewNameInfo;
  }
  }

  assert(0 && "Unknown name kind.");
  return DeclarationNameInfo();
}

template<typename Derived>
TemplateName
TreeTransform<Derived>::TransformTemplateName(TemplateName Name,
                                              QualType ObjectType,
                                              NamedDecl *FirstQualifierInScope) {
  SourceLocation Loc = getDerived().getBaseLocation();

  if (QualifiedTemplateName *QTN = Name.getAsQualifiedTemplateName()) {
    NestedNameSpecifier *NNS
      = getDerived().TransformNestedNameSpecifier(QTN->getQualifier(),
                                                  /*FIXME*/ SourceRange(Loc),
                                                  ObjectType,
                                                  FirstQualifierInScope);
    if (!NNS)
      return TemplateName();

    if (TemplateDecl *Template = QTN->getTemplateDecl()) {
      TemplateDecl *TransTemplate
        = cast_or_null<TemplateDecl>(getDerived().TransformDecl(Loc, Template));
      if (!TransTemplate)
        return TemplateName();

      if (!getDerived().AlwaysRebuild() &&
          NNS == QTN->getQualifier() &&
          TransTemplate == Template)
        return Name;

      return getDerived().RebuildTemplateName(NNS, QTN->hasTemplateKeyword(),
                                              TransTemplate);
    }

    // These should be getting filtered out before they make it into the AST.
    llvm_unreachable("overloaded template name survived to here");
  }

  if (DependentTemplateName *DTN = Name.getAsDependentTemplateName()) {
    NestedNameSpecifier *NNS = DTN->getQualifier();
    if (NNS) {
      NNS = getDerived().TransformNestedNameSpecifier(NNS,
                                                  /*FIXME:*/SourceRange(Loc),
                                                      ObjectType,
                                                      FirstQualifierInScope);
      if (!NNS) return TemplateName();

      // These apply to the scope specifier, not the template.
      ObjectType = QualType();
      FirstQualifierInScope = 0;
    }

    if (!getDerived().AlwaysRebuild() &&
        NNS == DTN->getQualifier() &&
        ObjectType.isNull())
      return Name;

    if (DTN->isIdentifier()) {
      // FIXME: Bad range
      SourceRange QualifierRange(getDerived().getBaseLocation());
      return getDerived().RebuildTemplateName(NNS, QualifierRange,
                                              *DTN->getIdentifier(), 
                                              ObjectType,
                                              FirstQualifierInScope);
    }
    
    return getDerived().RebuildTemplateName(NNS, DTN->getOperator(), 
                                            ObjectType);
  }

  if (TemplateDecl *Template = Name.getAsTemplateDecl()) {
    TemplateDecl *TransTemplate
      = cast_or_null<TemplateDecl>(getDerived().TransformDecl(Loc, Template));
    if (!TransTemplate)
      return TemplateName();

    if (!getDerived().AlwaysRebuild() &&
        TransTemplate == Template)
      return Name;

    return TemplateName(TransTemplate);
  }

  // These should be getting filtered out before they reach the AST.
  llvm_unreachable("overloaded function decl survived to here");
  return TemplateName();
}

template<typename Derived>
void TreeTransform<Derived>::InventTemplateArgumentLoc(
                                         const TemplateArgument &Arg,
                                         TemplateArgumentLoc &Output) {
  SourceLocation Loc = getDerived().getBaseLocation();
  switch (Arg.getKind()) {
  case TemplateArgument::Null:
    llvm_unreachable("null template argument in TreeTransform");
    break;

  case TemplateArgument::Type:
    Output = TemplateArgumentLoc(Arg,
               SemaRef.Context.getTrivialTypeSourceInfo(Arg.getAsType(), Loc));
                                            
    break;

  case TemplateArgument::Template:
    Output = TemplateArgumentLoc(Arg, SourceRange(), Loc);
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
    TypeSourceInfo *DI = Input.getTypeSourceInfo();
    if (DI == NULL)
      DI = InventTypeSourceInfo(Input.getArgument().getAsType());

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
    TemporaryBase Rebase(*this, Input.getLocation(), Name);
    Decl *D = getDerived().TransformDecl(Input.getLocation(), Arg.getAsDecl());
    if (!D) return true;

    Expr *SourceExpr = Input.getSourceDeclExpression();
    if (SourceExpr) {
      EnterExpressionEvaluationContext Unevaluated(getSema(),
                                                   Sema::Unevaluated);
      ExprResult E = getDerived().TransformExpr(SourceExpr);
      SourceExpr = (E.isInvalid() ? 0 : E.take());
    }

    Output = TemplateArgumentLoc(TemplateArgument(D), SourceExpr);
    return false;
  }

  case TemplateArgument::Template: {
    TemporaryBase Rebase(*this, Input.getLocation(), DeclarationName());    
    TemplateName Template
      = getDerived().TransformTemplateName(Arg.getAsTemplate());
    if (Template.isNull())
      return true;
    
    Output = TemplateArgumentLoc(TemplateArgument(Template),
                                 Input.getTemplateQualifierRange(),
                                 Input.getTemplateNameLoc());
    return false;
  }
      
  case TemplateArgument::Expression: {
    // Template argument expressions are not potentially evaluated.
    EnterExpressionEvaluationContext Unevaluated(getSema(),
                                                 Sema::Unevaluated);

    Expr *InputExpr = Input.getSourceExpression();
    if (!InputExpr) InputExpr = Input.getArgument().getAsExpr();

    ExprResult E
      = getDerived().TransformExpr(InputExpr);
    if (E.isInvalid()) return true;
    Output = TemplateArgumentLoc(TemplateArgument(E.take()), E.take());
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

    TemplateArgument *TransformedArgsPtr
      = new (getSema().Context) TemplateArgument[TransformedArgs.size()];
    std::copy(TransformedArgs.begin(), TransformedArgs.end(),
              TransformedArgsPtr);
    Output = TemplateArgumentLoc(TemplateArgument(TransformedArgsPtr, 
                                                  TransformedArgs.size()), 
                                 Input.getLocInfo());
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
  TypeSourceInfo *DI = getSema().Context.CreateTypeSourceInfo(T);
  DI->getTypeLoc().initialize(getDerived().getBaseLocation());
  
  TypeSourceInfo *NewDI = getDerived().TransformType(DI);

  if (!NewDI)
    return QualType();

  return NewDI->getType();
}

template<typename Derived>
TypeSourceInfo *TreeTransform<Derived>::TransformType(TypeSourceInfo *DI) {
  if (getDerived().AlreadyTransformed(DI->getType()))
    return DI;

  TypeLocBuilder TLB;

  TypeLoc TL = DI->getTypeLoc();
  TLB.reserve(TL.getFullDataSize());

  QualType Result = getDerived().TransformType(TLB, TL);
  if (Result.isNull())
    return 0;

  return TLB.getTypeSourceInfo(SemaRef.Context, Result);
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

  llvm_unreachable("unhandled type loc!");
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
  Qualifiers Quals = T.getType().getLocalQualifiers();

  QualType Result = getDerived().TransformType(TLB, T.getUnqualifiedLoc());
  if (Result.isNull())
    return QualType();

  // Silently suppress qualifiers if the result type can't be qualified.
  // FIXME: this is the right thing for template instantiation, but
  // probably not for other clients.
  if (Result->isFunctionType() || Result->isReferenceType())
    return Result;

  if (!Quals.empty()) {
    Result = SemaRef.BuildQualifiedType(Result, T.getBeginLoc(), Quals);
    TLB.push<QualifiedTypeLoc>(Result);
    // No location information to preserve.
  }

  return Result;
}

/// \brief Transforms a type that was written in a scope specifier,
/// given an object type, the results of unqualified lookup, and
/// an already-instantiated prefix.
///
/// The object type is provided iff the scope specifier qualifies the
/// member of a dependent member-access expression.  The prefix is
/// provided iff the the scope specifier in which this appears has a
/// prefix.
///
/// This is private to TreeTransform.
template<typename Derived>
QualType
TreeTransform<Derived>::TransformTypeInObjectScope(QualType T,
                                                   QualType ObjectType,
                                                   NamedDecl *UnqualLookup,
                                                  NestedNameSpecifier *Prefix) {
  if (getDerived().AlreadyTransformed(T))
    return T;

  TypeSourceInfo *TSI =
    SemaRef.Context.getTrivialTypeSourceInfo(T, getBaseLocation());

  TSI = getDerived().TransformTypeInObjectScope(TSI, ObjectType,
                                                UnqualLookup, Prefix);
  if (!TSI) return QualType();
  return TSI->getType();
}

template<typename Derived>
TypeSourceInfo *
TreeTransform<Derived>::TransformTypeInObjectScope(TypeSourceInfo *TSI,
                                                   QualType ObjectType,
                                                   NamedDecl *UnqualLookup,
                                                  NestedNameSpecifier *Prefix) {
  // TODO: in some cases, we might be some verification to do here.
  if (ObjectType.isNull())
    return getDerived().TransformType(TSI);

  QualType T = TSI->getType();
  if (getDerived().AlreadyTransformed(T))
    return TSI;

  TypeLocBuilder TLB;
  QualType Result;

  if (isa<TemplateSpecializationType>(T)) {
    TemplateSpecializationTypeLoc TL
      = cast<TemplateSpecializationTypeLoc>(TSI->getTypeLoc());

    TemplateName Template =
      getDerived().TransformTemplateName(TL.getTypePtr()->getTemplateName(),
                                         ObjectType, UnqualLookup);
    if (Template.isNull()) return 0;

    Result = getDerived()
      .TransformTemplateSpecializationType(TLB, TL, Template);
  } else if (isa<DependentTemplateSpecializationType>(T)) {
    DependentTemplateSpecializationTypeLoc TL
      = cast<DependentTemplateSpecializationTypeLoc>(TSI->getTypeLoc());

    Result = getDerived()
      .TransformDependentTemplateSpecializationType(TLB, TL, Prefix);
  } else {
    // Nothing special needs to be done for these.
    Result = getDerived().TransformType(TLB, TSI->getTypeLoc());
  }

  if (Result.isNull()) return 0;
  return TLB.getTypeSourceInfo(SemaRef.Context, Result);
}

template <class TyLoc> static inline
QualType TransformTypeSpecType(TypeLocBuilder &TLB, TyLoc T) {
  TyLoc NewT = TLB.push<TyLoc>(T.getType());
  NewT.setNameLoc(T.getNameLoc());
  return T.getType();
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformBuiltinType(TypeLocBuilder &TLB,
                                                      BuiltinTypeLoc T) {
  BuiltinTypeLoc NewT = TLB.push<BuiltinTypeLoc>(T.getType());
  NewT.setBuiltinLoc(T.getBuiltinLoc());
  if (T.needsExtraLocalData())
    NewT.getWrittenBuiltinSpecs() = T.getWrittenBuiltinSpecs();
  return T.getType();
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
  QualType PointeeType                                      
    = getDerived().TransformType(TLB, TL.getPointeeLoc());  
  if (PointeeType.isNull())
    return QualType();

  QualType Result = TL.getType();
  if (PointeeType->getAs<ObjCObjectType>()) {
    // A dependent pointer type 'T *' has is being transformed such
    // that an Objective-C class type is being replaced for 'T'. The
    // resulting pointer type is an ObjCObjectPointerType, not a
    // PointerType.
    Result = SemaRef.Context.getObjCObjectPointerType(PointeeType);
    
    ObjCObjectPointerTypeLoc NewT = TLB.push<ObjCObjectPointerTypeLoc>(Result);
    NewT.setStarLoc(TL.getStarLoc());
    return Result;
  }

  if (getDerived().AlwaysRebuild() ||
      PointeeType != TL.getPointeeLoc().getType()) {
    Result = getDerived().RebuildPointerType(PointeeType, TL.getSigilLoc());
    if (Result.isNull())
      return QualType();
  }
                                                            
  PointerTypeLoc NewT = TLB.push<PointerTypeLoc>(Result);
  NewT.setSigilLoc(TL.getSigilLoc());
  return Result;  
}

template<typename Derived>
QualType
TreeTransform<Derived>::TransformBlockPointerType(TypeLocBuilder &TLB,
                                                  BlockPointerTypeLoc TL) {
  QualType PointeeType
    = getDerived().TransformType(TLB, TL.getPointeeLoc());  
  if (PointeeType.isNull())                                 
    return QualType();                                      
  
  QualType Result = TL.getType();                           
  if (getDerived().AlwaysRebuild() ||                       
      PointeeType != TL.getPointeeLoc().getType()) {        
    Result = getDerived().RebuildBlockPointerType(PointeeType, 
                                                  TL.getSigilLoc());
    if (Result.isNull())
      return QualType();
  }

  BlockPointerTypeLoc NewT = TLB.push<BlockPointerTypeLoc>(Result);
  NewT.setSigilLoc(TL.getSigilLoc());
  return Result;
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
    EnterExpressionEvaluationContext Unevaluated(SemaRef, Sema::Unevaluated);
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
  EnterExpressionEvaluationContext Unevaluated(SemaRef, Sema::Unevaluated);

  ExprResult SizeResult
    = getDerived().TransformExpr(T->getSizeExpr());
  if (SizeResult.isInvalid())
    return QualType();

  Expr *Size = SizeResult.take();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      ElementType != T->getElementType() ||
      Size != T->getSizeExpr()) {
    Result = getDerived().RebuildVariableArrayType(ElementType,
                                                   T->getSizeModifier(),
                                                   Size,
                                             T->getIndexTypeCVRQualifiers(),
                                                   TL.getBracketsRange());
    if (Result.isNull())
      return QualType();
  }
  
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
  EnterExpressionEvaluationContext Unevaluated(SemaRef, Sema::Unevaluated);

  ExprResult SizeResult
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
                                                         Size,
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
  EnterExpressionEvaluationContext Unevaluated(SemaRef, Sema::Unevaluated);

  ExprResult Size = getDerived().TransformExpr(T->getSizeExpr());
  if (Size.isInvalid())
    return QualType();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      ElementType != T->getElementType() ||
      Size.get() != T->getSizeExpr()) {
    Result = getDerived().RebuildDependentSizedExtVectorType(ElementType,
                                                             Size.take(),
                                                         T->getAttributeLoc());
    if (Result.isNull())
      return QualType();
  }

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
    Result = getDerived().RebuildVectorType(ElementType, T->getNumElements(),
                                            T->getVectorKind());
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
ParmVarDecl *
TreeTransform<Derived>::TransformFunctionTypeParam(ParmVarDecl *OldParm) {
  TypeSourceInfo *OldDI = OldParm->getTypeSourceInfo();
  TypeSourceInfo *NewDI = getDerived().TransformType(OldDI);
  if (!NewDI)
    return 0;

  if (NewDI == OldDI)
    return OldParm;
  else
    return ParmVarDecl::Create(SemaRef.Context,
                               OldParm->getDeclContext(),
                               OldParm->getLocation(),
                               OldParm->getIdentifier(),
                               NewDI->getType(),
                               NewDI,
                               OldParm->getStorageClass(),
                               OldParm->getStorageClassAsWritten(),
                               /* DefArg */ NULL);
}

template<typename Derived>
bool TreeTransform<Derived>::
  TransformFunctionTypeParams(FunctionProtoTypeLoc TL,
                              llvm::SmallVectorImpl<QualType> &PTypes,
                              llvm::SmallVectorImpl<ParmVarDecl*> &PVars) {
  FunctionProtoType *T = TL.getTypePtr();

  for (unsigned i = 0, e = TL.getNumArgs(); i != e; ++i) {
    ParmVarDecl *OldParm = TL.getArg(i);

    QualType NewType;
    ParmVarDecl *NewParm;

    if (OldParm) {
      NewParm = getDerived().TransformFunctionTypeParam(OldParm);
      if (!NewParm)
        return true;
      NewType = NewParm->getType();

    // Deal with the possibility that we don't have a parameter
    // declaration for this parameter.
    } else {
      NewParm = 0;

      QualType OldType = T->getArgType(i);
      NewType = getDerived().TransformType(OldType);
      if (NewType.isNull())
        return true;
    }

    PTypes.push_back(NewType);
    PVars.push_back(NewParm);
  }

  return false;
}

template<typename Derived>
QualType
TreeTransform<Derived>::TransformFunctionProtoType(TypeLocBuilder &TLB,
                                                   FunctionProtoTypeLoc TL) {
  // Transform the parameters and return type.
  //
  // We instantiate in source order, with the return type first followed by
  // the parameters, because users tend to expect this (even if they shouldn't
  // rely on it!).
  //
  // When the function has a trailing return type, we instantiate the
  // parameters before the return type,  since the return type can then refer
  // to the parameters themselves (via decltype, sizeof, etc.).
  //
  llvm::SmallVector<QualType, 4> ParamTypes;
  llvm::SmallVector<ParmVarDecl*, 4> ParamDecls;
  FunctionProtoType *T = TL.getTypePtr();

  QualType ResultType;

  if (TL.getTrailingReturn()) {
    if (getDerived().TransformFunctionTypeParams(TL, ParamTypes, ParamDecls))
      return QualType();

    ResultType = getDerived().TransformType(TLB, TL.getResultLoc());
    if (ResultType.isNull())
      return QualType();
  }
  else {
    ResultType = getDerived().TransformType(TLB, TL.getResultLoc());
    if (ResultType.isNull())
      return QualType();

    if (getDerived().TransformFunctionTypeParams(TL, ParamTypes, ParamDecls))
      return QualType();
  }

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      ResultType != T->getResultType() ||
      !std::equal(T->arg_type_begin(), T->arg_type_end(), ParamTypes.begin())) {
    Result = getDerived().RebuildFunctionProtoType(ResultType,
                                                   ParamTypes.data(),
                                                   ParamTypes.size(),
                                                   T->isVariadic(),
                                                   T->getTypeQuals(),
                                                   T->getExtInfo());
    if (Result.isNull())
      return QualType();
  }

  FunctionProtoTypeLoc NewTL = TLB.push<FunctionProtoTypeLoc>(Result);
  NewTL.setLParenLoc(TL.getLParenLoc());
  NewTL.setRParenLoc(TL.getRParenLoc());
  NewTL.setTrailingReturn(TL.getTrailingReturn());
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
  NewTL.setTrailingReturn(false);

  return Result;
}

template<typename Derived> QualType
TreeTransform<Derived>::TransformUnresolvedUsingType(TypeLocBuilder &TLB,
                                                 UnresolvedUsingTypeLoc TL) {
  UnresolvedUsingType *T = TL.getTypePtr();
  Decl *D = getDerived().TransformDecl(TL.getNameLoc(), T->getDecl());
  if (!D)
    return QualType();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() || D != T->getDecl()) {
    Result = getDerived().RebuildUnresolvedUsingType(D);
    if (Result.isNull())
      return QualType();
  }

  // We might get an arbitrary type spec type back.  We should at
  // least always get a type spec type, though.
  TypeSpecTypeLoc NewTL = TLB.pushTypeSpec(Result);
  NewTL.setNameLoc(TL.getNameLoc());

  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformTypedefType(TypeLocBuilder &TLB,
                                                      TypedefTypeLoc TL) {
  TypedefType *T = TL.getTypePtr();
  TypedefDecl *Typedef
    = cast_or_null<TypedefDecl>(getDerived().TransformDecl(TL.getNameLoc(),
                                                           T->getDecl()));
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
  // typeof expressions are not potentially evaluated contexts
  EnterExpressionEvaluationContext Unevaluated(SemaRef, Sema::Unevaluated);

  ExprResult E = getDerived().TransformExpr(TL.getUnderlyingExpr());
  if (E.isInvalid())
    return QualType();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      E.get() != TL.getUnderlyingExpr()) {
    Result = getDerived().RebuildTypeOfExprType(E.get(), TL.getTypeofLoc());
    if (Result.isNull())
      return QualType();
  }
  else E.take();

  TypeOfExprTypeLoc NewTL = TLB.push<TypeOfExprTypeLoc>(Result);
  NewTL.setTypeofLoc(TL.getTypeofLoc());
  NewTL.setLParenLoc(TL.getLParenLoc());
  NewTL.setRParenLoc(TL.getRParenLoc());

  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformTypeOfType(TypeLocBuilder &TLB,
                                                     TypeOfTypeLoc TL) {
  TypeSourceInfo* Old_Under_TI = TL.getUnderlyingTInfo();
  TypeSourceInfo* New_Under_TI = getDerived().TransformType(Old_Under_TI);
  if (!New_Under_TI)
    return QualType();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() || New_Under_TI != Old_Under_TI) {
    Result = getDerived().RebuildTypeOfType(New_Under_TI->getType());
    if (Result.isNull())
      return QualType();
  }

  TypeOfTypeLoc NewTL = TLB.push<TypeOfTypeLoc>(Result);
  NewTL.setTypeofLoc(TL.getTypeofLoc());
  NewTL.setLParenLoc(TL.getLParenLoc());
  NewTL.setRParenLoc(TL.getRParenLoc());
  NewTL.setUnderlyingTInfo(New_Under_TI);

  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformDecltypeType(TypeLocBuilder &TLB,
                                                       DecltypeTypeLoc TL) {
  DecltypeType *T = TL.getTypePtr();

  // decltype expressions are not potentially evaluated contexts
  EnterExpressionEvaluationContext Unevaluated(SemaRef, Sema::Unevaluated);

  ExprResult E = getDerived().TransformExpr(T->getUnderlyingExpr());
  if (E.isInvalid())
    return QualType();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      E.get() != T->getUnderlyingExpr()) {
    Result = getDerived().RebuildDecltypeType(E.get(), TL.getNameLoc());
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
    = cast_or_null<RecordDecl>(getDerived().TransformDecl(TL.getNameLoc(),
                                                          T->getDecl()));
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
    = cast_or_null<EnumDecl>(getDerived().TransformDecl(TL.getNameLoc(),
                                                        T->getDecl()));
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

template<typename Derived>
QualType TreeTransform<Derived>::TransformInjectedClassNameType(
                                         TypeLocBuilder &TLB,
                                         InjectedClassNameTypeLoc TL) {
  Decl *D = getDerived().TransformDecl(TL.getNameLoc(),
                                       TL.getTypePtr()->getDecl());
  if (!D) return QualType();

  QualType T = SemaRef.Context.getTypeDeclType(cast<TypeDecl>(D));
  TLB.pushTypeSpec(T).setNameLoc(TL.getNameLoc());
  return T;
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
QualType TreeTransform<Derived>::TransformTemplateSpecializationType(
                                                        TypeLocBuilder &TLB,
                                           TemplateSpecializationTypeLoc TL) {
  const TemplateSpecializationType *T = TL.getTypePtr();

  TemplateName Template
    = getDerived().TransformTemplateName(T->getTemplateName());
  if (Template.isNull())
    return QualType();

  return getDerived().TransformTemplateSpecializationType(TLB, TL, Template);
}

template <typename Derived>
QualType TreeTransform<Derived>::TransformTemplateSpecializationType(
                                                        TypeLocBuilder &TLB,
                                           TemplateSpecializationTypeLoc TL,
                                                      TemplateName Template) {
  const TemplateSpecializationType *T = TL.getTypePtr();

  TemplateArgumentListInfo NewTemplateArgs;
  NewTemplateArgs.setLAngleLoc(TL.getLAngleLoc());
  NewTemplateArgs.setRAngleLoc(TL.getRAngleLoc());

  for (unsigned i = 0, e = T->getNumArgs(); i != e; ++i) {
    TemplateArgumentLoc Loc;
    if (getDerived().TransformTemplateArgument(TL.getArgLoc(i), Loc))
      return QualType();
    NewTemplateArgs.addArgument(Loc);
  }

  // FIXME: maybe don't rebuild if all the template arguments are the same.

  QualType Result =
    getDerived().RebuildTemplateSpecializationType(Template,
                                                   TL.getTemplateNameLoc(),
                                                   NewTemplateArgs);

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
TreeTransform<Derived>::TransformElaboratedType(TypeLocBuilder &TLB,
                                                ElaboratedTypeLoc TL) {
  ElaboratedType *T = TL.getTypePtr();

  NestedNameSpecifier *NNS = 0;
  // NOTE: the qualifier in an ElaboratedType is optional.
  if (T->getQualifier() != 0) {
    NNS = getDerived().TransformNestedNameSpecifier(T->getQualifier(),
                                                    TL.getQualifierRange());
    if (!NNS)
      return QualType();
  }

  QualType NamedT = getDerived().TransformType(TLB, TL.getNamedTypeLoc());
  if (NamedT.isNull())
    return QualType();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      NNS != T->getQualifier() ||
      NamedT != T->getNamedType()) {
    Result = getDerived().RebuildElaboratedType(TL.getKeywordLoc(),
                                                T->getKeyword(), NNS, NamedT);
    if (Result.isNull())
      return QualType();
  }

  ElaboratedTypeLoc NewTL = TLB.push<ElaboratedTypeLoc>(Result);
  NewTL.setKeywordLoc(TL.getKeywordLoc());
  NewTL.setQualifierRange(TL.getQualifierRange());

  return Result;
}

template<typename Derived>
QualType
TreeTransform<Derived>::TransformParenType(TypeLocBuilder &TLB,
                                           ParenTypeLoc TL) {
  QualType Inner = getDerived().TransformType(TLB, TL.getInnerLoc());
  if (Inner.isNull())
    return QualType();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      Inner != TL.getInnerLoc().getType()) {
    Result = getDerived().RebuildParenType(Inner);
    if (Result.isNull())
      return QualType();
  }

  ParenTypeLoc NewTL = TLB.push<ParenTypeLoc>(Result);
  NewTL.setLParenLoc(TL.getLParenLoc());
  NewTL.setRParenLoc(TL.getRParenLoc());
  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformDependentNameType(TypeLocBuilder &TLB,
                                                      DependentNameTypeLoc TL) {
  DependentNameType *T = TL.getTypePtr();

  NestedNameSpecifier *NNS
    = getDerived().TransformNestedNameSpecifier(T->getQualifier(),
                                                TL.getQualifierRange());
  if (!NNS)
    return QualType();

  QualType Result
    = getDerived().RebuildDependentNameType(T->getKeyword(), NNS,
                                            T->getIdentifier(),
                                            TL.getKeywordLoc(),
                                            TL.getQualifierRange(),
                                            TL.getNameLoc());
  if (Result.isNull())
    return QualType();

  if (const ElaboratedType* ElabT = Result->getAs<ElaboratedType>()) {
    QualType NamedT = ElabT->getNamedType();
    TLB.pushTypeSpec(NamedT).setNameLoc(TL.getNameLoc());

    ElaboratedTypeLoc NewTL = TLB.push<ElaboratedTypeLoc>(Result);
    NewTL.setKeywordLoc(TL.getKeywordLoc());
    NewTL.setQualifierRange(TL.getQualifierRange());
  } else {
    DependentNameTypeLoc NewTL = TLB.push<DependentNameTypeLoc>(Result);
    NewTL.setKeywordLoc(TL.getKeywordLoc());
    NewTL.setQualifierRange(TL.getQualifierRange());
    NewTL.setNameLoc(TL.getNameLoc());
  }
  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::
          TransformDependentTemplateSpecializationType(TypeLocBuilder &TLB,
                                 DependentTemplateSpecializationTypeLoc TL) {
  DependentTemplateSpecializationType *T = TL.getTypePtr();

  NestedNameSpecifier *NNS
    = getDerived().TransformNestedNameSpecifier(T->getQualifier(),
                                                TL.getQualifierRange());
  if (!NNS)
    return QualType();

  return getDerived()
           .TransformDependentTemplateSpecializationType(TLB, TL, NNS);
}

template<typename Derived>
QualType TreeTransform<Derived>::
          TransformDependentTemplateSpecializationType(TypeLocBuilder &TLB,
                                 DependentTemplateSpecializationTypeLoc TL,
                                                  NestedNameSpecifier *NNS) {
  DependentTemplateSpecializationType *T = TL.getTypePtr();

  TemplateArgumentListInfo NewTemplateArgs;
  NewTemplateArgs.setLAngleLoc(TL.getLAngleLoc());
  NewTemplateArgs.setRAngleLoc(TL.getRAngleLoc());

  for (unsigned I = 0, E = T->getNumArgs(); I != E; ++I) {
    TemplateArgumentLoc Loc;
    if (getDerived().TransformTemplateArgument(TL.getArgLoc(I), Loc))
      return QualType();
    NewTemplateArgs.addArgument(Loc);
  }

  QualType Result
    = getDerived().RebuildDependentTemplateSpecializationType(T->getKeyword(),
                                                              NNS,
                                                        TL.getQualifierRange(),
                                                            T->getIdentifier(),
                                                              TL.getNameLoc(),
                                                              NewTemplateArgs);
  if (Result.isNull())
    return QualType();

  if (const ElaboratedType *ElabT = dyn_cast<ElaboratedType>(Result)) {
    QualType NamedT = ElabT->getNamedType();

    // Copy information relevant to the template specialization.
    TemplateSpecializationTypeLoc NamedTL
      = TLB.push<TemplateSpecializationTypeLoc>(NamedT);
    NamedTL.setLAngleLoc(TL.getLAngleLoc());
    NamedTL.setRAngleLoc(TL.getRAngleLoc());
    for (unsigned I = 0, E = TL.getNumArgs(); I != E; ++I)
      NamedTL.setArgLocInfo(I, TL.getArgLocInfo(I));

    // Copy information relevant to the elaborated type.
    ElaboratedTypeLoc NewTL = TLB.push<ElaboratedTypeLoc>(Result);
    NewTL.setKeywordLoc(TL.getKeywordLoc());
    NewTL.setQualifierRange(TL.getQualifierRange());
  } else {
    TypeLoc NewTL(Result, TL.getOpaqueData());
    TLB.pushFullCopy(NewTL);
  }
  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformPackExpansionType(TypeLocBuilder &TLB,
                                                      PackExpansionTypeLoc TL) {
  // FIXME: Implement!
  getSema().Diag(TL.getEllipsisLoc(), 
                 diag::err_pack_expansion_instantiation_unsupported);
  return QualType();
}

template<typename Derived>
QualType
TreeTransform<Derived>::TransformObjCInterfaceType(TypeLocBuilder &TLB,
                                                   ObjCInterfaceTypeLoc TL) {
  // ObjCInterfaceType is never dependent.
  TLB.pushFullCopy(TL);
  return TL.getType();
}

template<typename Derived>
QualType
TreeTransform<Derived>::TransformObjCObjectType(TypeLocBuilder &TLB,
                                                ObjCObjectTypeLoc TL) {
  // ObjCObjectType is never dependent.
  TLB.pushFullCopy(TL);
  return TL.getType();
}

template<typename Derived>
QualType
TreeTransform<Derived>::TransformObjCObjectPointerType(TypeLocBuilder &TLB,
                                               ObjCObjectPointerTypeLoc TL) {
  // ObjCObjectPointerType is never dependent.
  TLB.pushFullCopy(TL);
  return TL.getType();
}

//===----------------------------------------------------------------------===//
// Statement transformation
//===----------------------------------------------------------------------===//
template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformNullStmt(NullStmt *S) {
  return SemaRef.Owned(S);
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformCompoundStmt(CompoundStmt *S) {
  return getDerived().TransformCompoundStmt(S, false);
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformCompoundStmt(CompoundStmt *S,
                                              bool IsStmtExpr) {
  bool SubStmtInvalid = false;
  bool SubStmtChanged = false;
  ASTOwningVector<Stmt*> Statements(getSema());
  for (CompoundStmt::body_iterator B = S->body_begin(), BEnd = S->body_end();
       B != BEnd; ++B) {
    StmtResult Result = getDerived().TransformStmt(*B);
    if (Result.isInvalid()) {
      // Immediately fail if this was a DeclStmt, since it's very
      // likely that this will cause problems for future statements.
      if (isa<DeclStmt>(*B))
        return StmtError();

      // Otherwise, just keep processing substatements and fail later.
      SubStmtInvalid = true;
      continue;
    }

    SubStmtChanged = SubStmtChanged || Result.get() != *B;
    Statements.push_back(Result.takeAs<Stmt>());
  }

  if (SubStmtInvalid)
    return StmtError();

  if (!getDerived().AlwaysRebuild() &&
      !SubStmtChanged)
    return SemaRef.Owned(S);

  return getDerived().RebuildCompoundStmt(S->getLBracLoc(),
                                          move_arg(Statements),
                                          S->getRBracLoc(),
                                          IsStmtExpr);
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformCaseStmt(CaseStmt *S) {
  ExprResult LHS, RHS;
  {
    // The case value expressions are not potentially evaluated.
    EnterExpressionEvaluationContext Unevaluated(SemaRef, Sema::Unevaluated);

    // Transform the left-hand case value.
    LHS = getDerived().TransformExpr(S->getLHS());
    if (LHS.isInvalid())
      return StmtError();

    // Transform the right-hand case value (for the GNU case-range extension).
    RHS = getDerived().TransformExpr(S->getRHS());
    if (RHS.isInvalid())
      return StmtError();
  }

  // Build the case statement.
  // Case statements are always rebuilt so that they will attached to their
  // transformed switch statement.
  StmtResult Case = getDerived().RebuildCaseStmt(S->getCaseLoc(),
                                                       LHS.get(),
                                                       S->getEllipsisLoc(),
                                                       RHS.get(),
                                                       S->getColonLoc());
  if (Case.isInvalid())
    return StmtError();

  // Transform the statement following the case
  StmtResult SubStmt = getDerived().TransformStmt(S->getSubStmt());
  if (SubStmt.isInvalid())
    return StmtError();

  // Attach the body to the case statement
  return getDerived().RebuildCaseStmtBody(Case.get(), SubStmt.get());
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformDefaultStmt(DefaultStmt *S) {
  // Transform the statement following the default case
  StmtResult SubStmt = getDerived().TransformStmt(S->getSubStmt());
  if (SubStmt.isInvalid())
    return StmtError();

  // Default statements are always rebuilt
  return getDerived().RebuildDefaultStmt(S->getDefaultLoc(), S->getColonLoc(),
                                         SubStmt.get());
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformLabelStmt(LabelStmt *S) {
  StmtResult SubStmt = getDerived().TransformStmt(S->getSubStmt());
  if (SubStmt.isInvalid())
    return StmtError();

  // FIXME: Pass the real colon location in.
  SourceLocation ColonLoc = SemaRef.PP.getLocForEndOfToken(S->getIdentLoc());
  return getDerived().RebuildLabelStmt(S->getIdentLoc(), S->getID(), ColonLoc,
                                       SubStmt.get(), S->HasUnusedAttribute());
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformIfStmt(IfStmt *S) {
  // Transform the condition
  ExprResult Cond;
  VarDecl *ConditionVar = 0;
  if (S->getConditionVariable()) {
    ConditionVar 
      = cast_or_null<VarDecl>(
                   getDerived().TransformDefinition(
                                      S->getConditionVariable()->getLocation(),
                                                    S->getConditionVariable()));
    if (!ConditionVar)
      return StmtError();
  } else {
    Cond = getDerived().TransformExpr(S->getCond());
  
    if (Cond.isInvalid())
      return StmtError();
    
    // Convert the condition to a boolean value.
    if (S->getCond()) {
      ExprResult CondE = getSema().ActOnBooleanCondition(0, 
                                                               S->getIfLoc(), 
                                                               Cond.get());
      if (CondE.isInvalid())
        return StmtError();
    
      Cond = CondE.get();
    }
  }
  
  Sema::FullExprArg FullCond(getSema().MakeFullExpr(Cond.take()));
  if (!S->getConditionVariable() && S->getCond() && !FullCond.get())
    return StmtError();
  
  // Transform the "then" branch.
  StmtResult Then = getDerived().TransformStmt(S->getThen());
  if (Then.isInvalid())
    return StmtError();

  // Transform the "else" branch.
  StmtResult Else = getDerived().TransformStmt(S->getElse());
  if (Else.isInvalid())
    return StmtError();

  if (!getDerived().AlwaysRebuild() &&
      FullCond.get() == S->getCond() &&
      ConditionVar == S->getConditionVariable() &&
      Then.get() == S->getThen() &&
      Else.get() == S->getElse())
    return SemaRef.Owned(S);

  return getDerived().RebuildIfStmt(S->getIfLoc(), FullCond, ConditionVar,
                                    Then.get(),
                                    S->getElseLoc(), Else.get());
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformSwitchStmt(SwitchStmt *S) {
  // Transform the condition.
  ExprResult Cond;
  VarDecl *ConditionVar = 0;
  if (S->getConditionVariable()) {
    ConditionVar 
      = cast_or_null<VarDecl>(
                   getDerived().TransformDefinition(
                                      S->getConditionVariable()->getLocation(),
                                                    S->getConditionVariable()));
    if (!ConditionVar)
      return StmtError();
  } else {
    Cond = getDerived().TransformExpr(S->getCond());
    
    if (Cond.isInvalid())
      return StmtError();
  }

  // Rebuild the switch statement.
  StmtResult Switch
    = getDerived().RebuildSwitchStmtStart(S->getSwitchLoc(), Cond.get(),
                                          ConditionVar);
  if (Switch.isInvalid())
    return StmtError();

  // Transform the body of the switch statement.
  StmtResult Body = getDerived().TransformStmt(S->getBody());
  if (Body.isInvalid())
    return StmtError();

  // Complete the switch statement.
  return getDerived().RebuildSwitchStmtBody(S->getSwitchLoc(), Switch.get(),
                                            Body.get());
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformWhileStmt(WhileStmt *S) {
  // Transform the condition
  ExprResult Cond;
  VarDecl *ConditionVar = 0;
  if (S->getConditionVariable()) {
    ConditionVar 
      = cast_or_null<VarDecl>(
                   getDerived().TransformDefinition(
                                      S->getConditionVariable()->getLocation(),
                                                    S->getConditionVariable()));
    if (!ConditionVar)
      return StmtError();
  } else {
    Cond = getDerived().TransformExpr(S->getCond());
    
    if (Cond.isInvalid())
      return StmtError();

    if (S->getCond()) {
      // Convert the condition to a boolean value.
      ExprResult CondE = getSema().ActOnBooleanCondition(0, 
                                                             S->getWhileLoc(), 
                                                               Cond.get());
      if (CondE.isInvalid())
        return StmtError();
      Cond = CondE;
    }
  }

  Sema::FullExprArg FullCond(getSema().MakeFullExpr(Cond.take()));
  if (!S->getConditionVariable() && S->getCond() && !FullCond.get())
    return StmtError();

  // Transform the body
  StmtResult Body = getDerived().TransformStmt(S->getBody());
  if (Body.isInvalid())
    return StmtError();

  if (!getDerived().AlwaysRebuild() &&
      FullCond.get() == S->getCond() &&
      ConditionVar == S->getConditionVariable() &&
      Body.get() == S->getBody())
    return Owned(S);

  return getDerived().RebuildWhileStmt(S->getWhileLoc(), FullCond,
                                       ConditionVar, Body.get());
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformDoStmt(DoStmt *S) {
  // Transform the body
  StmtResult Body = getDerived().TransformStmt(S->getBody());
  if (Body.isInvalid())
    return StmtError();

  // Transform the condition
  ExprResult Cond = getDerived().TransformExpr(S->getCond());
  if (Cond.isInvalid())
    return StmtError();
  
  if (!getDerived().AlwaysRebuild() &&
      Cond.get() == S->getCond() &&
      Body.get() == S->getBody())
    return SemaRef.Owned(S);

  return getDerived().RebuildDoStmt(S->getDoLoc(), Body.get(), S->getWhileLoc(),
                                    /*FIXME:*/S->getWhileLoc(), Cond.get(),
                                    S->getRParenLoc());
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformForStmt(ForStmt *S) {
  // Transform the initialization statement
  StmtResult Init = getDerived().TransformStmt(S->getInit());
  if (Init.isInvalid())
    return StmtError();

  // Transform the condition
  ExprResult Cond;
  VarDecl *ConditionVar = 0;
  if (S->getConditionVariable()) {
    ConditionVar 
      = cast_or_null<VarDecl>(
                   getDerived().TransformDefinition(
                                      S->getConditionVariable()->getLocation(),
                                                    S->getConditionVariable()));
    if (!ConditionVar)
      return StmtError();
  } else {
    Cond = getDerived().TransformExpr(S->getCond());
    
    if (Cond.isInvalid())
      return StmtError();

    if (S->getCond()) {
      // Convert the condition to a boolean value.
      ExprResult CondE = getSema().ActOnBooleanCondition(0, 
                                                               S->getForLoc(), 
                                                               Cond.get());
      if (CondE.isInvalid())
        return StmtError();

      Cond = CondE.get();
    }
  }

  Sema::FullExprArg FullCond(getSema().MakeFullExpr(Cond.take()));  
  if (!S->getConditionVariable() && S->getCond() && !FullCond.get())
    return StmtError();

  // Transform the increment
  ExprResult Inc = getDerived().TransformExpr(S->getInc());
  if (Inc.isInvalid())
    return StmtError();

  Sema::FullExprArg FullInc(getSema().MakeFullExpr(Inc.get()));
  if (S->getInc() && !FullInc.get())
    return StmtError();

  // Transform the body
  StmtResult Body = getDerived().TransformStmt(S->getBody());
  if (Body.isInvalid())
    return StmtError();

  if (!getDerived().AlwaysRebuild() &&
      Init.get() == S->getInit() &&
      FullCond.get() == S->getCond() &&
      Inc.get() == S->getInc() &&
      Body.get() == S->getBody())
    return SemaRef.Owned(S);

  return getDerived().RebuildForStmt(S->getForLoc(), S->getLParenLoc(),
                                     Init.get(), FullCond, ConditionVar,
                                     FullInc, S->getRParenLoc(), Body.get());
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformGotoStmt(GotoStmt *S) {
  // Goto statements must always be rebuilt, to resolve the label.
  return getDerived().RebuildGotoStmt(S->getGotoLoc(), S->getLabelLoc(),
                                      S->getLabel());
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformIndirectGotoStmt(IndirectGotoStmt *S) {
  ExprResult Target = getDerived().TransformExpr(S->getTarget());
  if (Target.isInvalid())
    return StmtError();

  if (!getDerived().AlwaysRebuild() &&
      Target.get() == S->getTarget())
    return SemaRef.Owned(S);

  return getDerived().RebuildIndirectGotoStmt(S->getGotoLoc(), S->getStarLoc(),
                                              Target.get());
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformContinueStmt(ContinueStmt *S) {
  return SemaRef.Owned(S);
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformBreakStmt(BreakStmt *S) {
  return SemaRef.Owned(S);
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformReturnStmt(ReturnStmt *S) {
  ExprResult Result = getDerived().TransformExpr(S->getRetValue());
  if (Result.isInvalid())
    return StmtError();

  // FIXME: We always rebuild the return statement because there is no way
  // to tell whether the return type of the function has changed.
  return getDerived().RebuildReturnStmt(S->getReturnLoc(), Result.get());
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformDeclStmt(DeclStmt *S) {
  bool DeclChanged = false;
  llvm::SmallVector<Decl *, 4> Decls;
  for (DeclStmt::decl_iterator D = S->decl_begin(), DEnd = S->decl_end();
       D != DEnd; ++D) {
    Decl *Transformed = getDerived().TransformDefinition((*D)->getLocation(),
                                                         *D);
    if (!Transformed)
      return StmtError();

    if (Transformed != *D)
      DeclChanged = true;

    Decls.push_back(Transformed);
  }

  if (!getDerived().AlwaysRebuild() && !DeclChanged)
    return SemaRef.Owned(S);

  return getDerived().RebuildDeclStmt(Decls.data(), Decls.size(),
                                      S->getStartLoc(), S->getEndLoc());
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformSwitchCase(SwitchCase *S) {
  assert(false && "SwitchCase is abstract and cannot be transformed");
  return SemaRef.Owned(S);
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformAsmStmt(AsmStmt *S) {
  
  ASTOwningVector<Expr*> Constraints(getSema());
  ASTOwningVector<Expr*> Exprs(getSema());
  llvm::SmallVector<IdentifierInfo *, 4> Names;

  ExprResult AsmString;
  ASTOwningVector<Expr*> Clobbers(getSema());

  bool ExprsChanged = false;
  
  // Go through the outputs.
  for (unsigned I = 0, E = S->getNumOutputs(); I != E; ++I) {
    Names.push_back(S->getOutputIdentifier(I));
    
    // No need to transform the constraint literal.
    Constraints.push_back(S->getOutputConstraintLiteral(I));
    
    // Transform the output expr.
    Expr *OutputExpr = S->getOutputExpr(I);
    ExprResult Result = getDerived().TransformExpr(OutputExpr);
    if (Result.isInvalid())
      return StmtError();
    
    ExprsChanged |= Result.get() != OutputExpr;
    
    Exprs.push_back(Result.get());
  }
  
  // Go through the inputs.
  for (unsigned I = 0, E = S->getNumInputs(); I != E; ++I) {
    Names.push_back(S->getInputIdentifier(I));
    
    // No need to transform the constraint literal.
    Constraints.push_back(S->getInputConstraintLiteral(I));
    
    // Transform the input expr.
    Expr *InputExpr = S->getInputExpr(I);
    ExprResult Result = getDerived().TransformExpr(InputExpr);
    if (Result.isInvalid())
      return StmtError();
    
    ExprsChanged |= Result.get() != InputExpr;
    
    Exprs.push_back(Result.get());
  }
  
  if (!getDerived().AlwaysRebuild() && !ExprsChanged)
    return SemaRef.Owned(S);

  // Go through the clobbers.
  for (unsigned I = 0, E = S->getNumClobbers(); I != E; ++I)
    Clobbers.push_back(S->getClobber(I));

  // No need to transform the asm string literal.
  AsmString = SemaRef.Owned(S->getAsmString());

  return getDerived().RebuildAsmStmt(S->getAsmLoc(),
                                     S->isSimple(),
                                     S->isVolatile(),
                                     S->getNumOutputs(),
                                     S->getNumInputs(),
                                     Names.data(),
                                     move_arg(Constraints),
                                     move_arg(Exprs),
                                     AsmString.get(),
                                     move_arg(Clobbers),
                                     S->getRParenLoc(),
                                     S->isMSAsm());
}


template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformObjCAtTryStmt(ObjCAtTryStmt *S) {
  // Transform the body of the @try.
  StmtResult TryBody = getDerived().TransformStmt(S->getTryBody());
  if (TryBody.isInvalid())
    return StmtError();
  
  // Transform the @catch statements (if present).
  bool AnyCatchChanged = false;
  ASTOwningVector<Stmt*> CatchStmts(SemaRef);
  for (unsigned I = 0, N = S->getNumCatchStmts(); I != N; ++I) {
    StmtResult Catch = getDerived().TransformStmt(S->getCatchStmt(I));
    if (Catch.isInvalid())
      return StmtError();
    if (Catch.get() != S->getCatchStmt(I))
      AnyCatchChanged = true;
    CatchStmts.push_back(Catch.release());
  }
  
  // Transform the @finally statement (if present).
  StmtResult Finally;
  if (S->getFinallyStmt()) {
    Finally = getDerived().TransformStmt(S->getFinallyStmt());
    if (Finally.isInvalid())
      return StmtError();
  }

  // If nothing changed, just retain this statement.
  if (!getDerived().AlwaysRebuild() &&
      TryBody.get() == S->getTryBody() &&
      !AnyCatchChanged &&
      Finally.get() == S->getFinallyStmt())
    return SemaRef.Owned(S);
  
  // Build a new statement.
  return getDerived().RebuildObjCAtTryStmt(S->getAtTryLoc(), TryBody.get(),
                                           move_arg(CatchStmts), Finally.get());
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformObjCAtCatchStmt(ObjCAtCatchStmt *S) {
  // Transform the @catch parameter, if there is one.
  VarDecl *Var = 0;
  if (VarDecl *FromVar = S->getCatchParamDecl()) {
    TypeSourceInfo *TSInfo = 0;
    if (FromVar->getTypeSourceInfo()) {
      TSInfo = getDerived().TransformType(FromVar->getTypeSourceInfo());
      if (!TSInfo)
        return StmtError();
    }
    
    QualType T;
    if (TSInfo)
      T = TSInfo->getType();
    else {
      T = getDerived().TransformType(FromVar->getType());
      if (T.isNull())
        return StmtError();        
    }
    
    Var = getDerived().RebuildObjCExceptionDecl(FromVar, TSInfo, T);
    if (!Var)
      return StmtError();
  }
  
  StmtResult Body = getDerived().TransformStmt(S->getCatchBody());
  if (Body.isInvalid())
    return StmtError();
  
  return getDerived().RebuildObjCAtCatchStmt(S->getAtCatchLoc(), 
                                             S->getRParenLoc(),
                                             Var, Body.get());
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformObjCAtFinallyStmt(ObjCAtFinallyStmt *S) {
  // Transform the body.
  StmtResult Body = getDerived().TransformStmt(S->getFinallyBody());
  if (Body.isInvalid())
    return StmtError();
  
  // If nothing changed, just retain this statement.
  if (!getDerived().AlwaysRebuild() &&
      Body.get() == S->getFinallyBody())
    return SemaRef.Owned(S);

  // Build a new statement.
  return getDerived().RebuildObjCAtFinallyStmt(S->getAtFinallyLoc(),
                                               Body.get());
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformObjCAtThrowStmt(ObjCAtThrowStmt *S) {
  ExprResult Operand;
  if (S->getThrowExpr()) {
    Operand = getDerived().TransformExpr(S->getThrowExpr());
    if (Operand.isInvalid())
      return StmtError();
  }
  
  if (!getDerived().AlwaysRebuild() &&
      Operand.get() == S->getThrowExpr())
    return getSema().Owned(S);
    
  return getDerived().RebuildObjCAtThrowStmt(S->getThrowLoc(), Operand.get());
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformObjCAtSynchronizedStmt(
                                                  ObjCAtSynchronizedStmt *S) {
  // Transform the object we are locking.
  ExprResult Object = getDerived().TransformExpr(S->getSynchExpr());
  if (Object.isInvalid())
    return StmtError();
  
  // Transform the body.
  StmtResult Body = getDerived().TransformStmt(S->getSynchBody());
  if (Body.isInvalid())
    return StmtError();
  
  // If nothing change, just retain the current statement.
  if (!getDerived().AlwaysRebuild() &&
      Object.get() == S->getSynchExpr() &&
      Body.get() == S->getSynchBody())
    return SemaRef.Owned(S);

  // Build a new statement.
  return getDerived().RebuildObjCAtSynchronizedStmt(S->getAtSynchronizedLoc(),
                                                    Object.get(), Body.get());
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformObjCForCollectionStmt(
                                                  ObjCForCollectionStmt *S) {
  // Transform the element statement.
  StmtResult Element = getDerived().TransformStmt(S->getElement());
  if (Element.isInvalid())
    return StmtError();
  
  // Transform the collection expression.
  ExprResult Collection = getDerived().TransformExpr(S->getCollection());
  if (Collection.isInvalid())
    return StmtError();
  
  // Transform the body.
  StmtResult Body = getDerived().TransformStmt(S->getBody());
  if (Body.isInvalid())
    return StmtError();
  
  // If nothing changed, just retain this statement.
  if (!getDerived().AlwaysRebuild() &&
      Element.get() == S->getElement() &&
      Collection.get() == S->getCollection() &&
      Body.get() == S->getBody())
    return SemaRef.Owned(S);
  
  // Build a new statement.
  return getDerived().RebuildObjCForCollectionStmt(S->getForLoc(),
                                                   /*FIXME:*/S->getForLoc(),
                                                   Element.get(),
                                                   Collection.get(),
                                                   S->getRParenLoc(),
                                                   Body.get());
}


template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformCXXCatchStmt(CXXCatchStmt *S) {
  // Transform the exception declaration, if any.
  VarDecl *Var = 0;
  if (S->getExceptionDecl()) {
    VarDecl *ExceptionDecl = S->getExceptionDecl();
    TypeSourceInfo *T = getDerived().TransformType(
                                            ExceptionDecl->getTypeSourceInfo());
    if (!T)
      return StmtError();

    Var = getDerived().RebuildExceptionDecl(ExceptionDecl, T,
                                            ExceptionDecl->getIdentifier(),
                                            ExceptionDecl->getLocation());
    if (!Var || Var->isInvalidDecl())
      return StmtError();
  }

  // Transform the actual exception handler.
  StmtResult Handler = getDerived().TransformStmt(S->getHandlerBlock());
  if (Handler.isInvalid())
    return StmtError();

  if (!getDerived().AlwaysRebuild() &&
      !Var &&
      Handler.get() == S->getHandlerBlock())
    return SemaRef.Owned(S);

  return getDerived().RebuildCXXCatchStmt(S->getCatchLoc(),
                                          Var,
                                          Handler.get());
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformCXXTryStmt(CXXTryStmt *S) {
  // Transform the try block itself.
  StmtResult TryBlock
    = getDerived().TransformCompoundStmt(S->getTryBlock());
  if (TryBlock.isInvalid())
    return StmtError();

  // Transform the handlers.
  bool HandlerChanged = false;
  ASTOwningVector<Stmt*> Handlers(SemaRef);
  for (unsigned I = 0, N = S->getNumHandlers(); I != N; ++I) {
    StmtResult Handler
      = getDerived().TransformCXXCatchStmt(S->getHandler(I));
    if (Handler.isInvalid())
      return StmtError();

    HandlerChanged = HandlerChanged || Handler.get() != S->getHandler(I);
    Handlers.push_back(Handler.takeAs<Stmt>());
  }

  if (!getDerived().AlwaysRebuild() &&
      TryBlock.get() == S->getTryBlock() &&
      !HandlerChanged)
    return SemaRef.Owned(S);

  return getDerived().RebuildCXXTryStmt(S->getTryLoc(), TryBlock.get(),
                                        move_arg(Handlers));
}

//===----------------------------------------------------------------------===//
// Expression transformation
//===----------------------------------------------------------------------===//
template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformPredefinedExpr(PredefinedExpr *E) {
  return SemaRef.Owned(E);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformDeclRefExpr(DeclRefExpr *E) {
  NestedNameSpecifier *Qualifier = 0;
  if (E->getQualifier()) {
    Qualifier = getDerived().TransformNestedNameSpecifier(E->getQualifier(),
                                                       E->getQualifierRange());
    if (!Qualifier)
      return ExprError();
  }

  ValueDecl *ND
    = cast_or_null<ValueDecl>(getDerived().TransformDecl(E->getLocation(),
                                                         E->getDecl()));
  if (!ND)
    return ExprError();

  DeclarationNameInfo NameInfo = E->getNameInfo();
  if (NameInfo.getName()) {
    NameInfo = getDerived().TransformDeclarationNameInfo(NameInfo);
    if (!NameInfo.getName())
      return ExprError();
  }

  if (!getDerived().AlwaysRebuild() &&
      Qualifier == E->getQualifier() &&
      ND == E->getDecl() &&
      NameInfo.getName() == E->getDecl()->getDeclName() &&
      !E->hasExplicitTemplateArgs()) {

    // Mark it referenced in the new context regardless.
    // FIXME: this is a bit instantiation-specific.
    SemaRef.MarkDeclarationReferenced(E->getLocation(), ND);

    return SemaRef.Owned(E);
  }

  TemplateArgumentListInfo TransArgs, *TemplateArgs = 0;
  if (E->hasExplicitTemplateArgs()) {
    TemplateArgs = &TransArgs;
    TransArgs.setLAngleLoc(E->getLAngleLoc());
    TransArgs.setRAngleLoc(E->getRAngleLoc());
    for (unsigned I = 0, N = E->getNumTemplateArgs(); I != N; ++I) {
      TemplateArgumentLoc Loc;
      if (getDerived().TransformTemplateArgument(E->getTemplateArgs()[I], Loc))
        return ExprError();
      TransArgs.addArgument(Loc);
    }
  }

  return getDerived().RebuildDeclRefExpr(Qualifier, E->getQualifierRange(),
                                         ND, NameInfo, TemplateArgs);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformIntegerLiteral(IntegerLiteral *E) {
  return SemaRef.Owned(E);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformFloatingLiteral(FloatingLiteral *E) {
  return SemaRef.Owned(E);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformImaginaryLiteral(ImaginaryLiteral *E) {
  return SemaRef.Owned(E);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformStringLiteral(StringLiteral *E) {
  return SemaRef.Owned(E);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCharacterLiteral(CharacterLiteral *E) {
  return SemaRef.Owned(E);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformParenExpr(ParenExpr *E) {
  ExprResult SubExpr = getDerived().TransformExpr(E->getSubExpr());
  if (SubExpr.isInvalid())
    return ExprError();

  if (!getDerived().AlwaysRebuild() && SubExpr.get() == E->getSubExpr())
    return SemaRef.Owned(E);

  return getDerived().RebuildParenExpr(SubExpr.get(), E->getLParen(),
                                       E->getRParen());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformUnaryOperator(UnaryOperator *E) {
  ExprResult SubExpr = getDerived().TransformExpr(E->getSubExpr());
  if (SubExpr.isInvalid())
    return ExprError();

  if (!getDerived().AlwaysRebuild() && SubExpr.get() == E->getSubExpr())
    return SemaRef.Owned(E);

  return getDerived().RebuildUnaryOperator(E->getOperatorLoc(),
                                           E->getOpcode(),
                                           SubExpr.get());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformOffsetOfExpr(OffsetOfExpr *E) {
  // Transform the type.
  TypeSourceInfo *Type = getDerived().TransformType(E->getTypeSourceInfo());
  if (!Type)
    return ExprError();
  
  // Transform all of the components into components similar to what the
  // parser uses.
  // FIXME: It would be slightly more efficient in the non-dependent case to 
  // just map FieldDecls, rather than requiring the rebuilder to look for 
  // the fields again. However, __builtin_offsetof is rare enough in 
  // template code that we don't care.
  bool ExprChanged = false;
  typedef Sema::OffsetOfComponent Component;
  typedef OffsetOfExpr::OffsetOfNode Node;
  llvm::SmallVector<Component, 4> Components;
  for (unsigned I = 0, N = E->getNumComponents(); I != N; ++I) {
    const Node &ON = E->getComponent(I);
    Component Comp;
    Comp.isBrackets = true;
    Comp.LocStart = ON.getRange().getBegin();
    Comp.LocEnd = ON.getRange().getEnd();
    switch (ON.getKind()) {
    case Node::Array: {
      Expr *FromIndex = E->getIndexExpr(ON.getArrayExprIndex());
      ExprResult Index = getDerived().TransformExpr(FromIndex);
      if (Index.isInvalid())
        return ExprError();
      
      ExprChanged = ExprChanged || Index.get() != FromIndex;
      Comp.isBrackets = true;
      Comp.U.E = Index.get();
      break;
    }
        
    case Node::Field:
    case Node::Identifier:
      Comp.isBrackets = false;
      Comp.U.IdentInfo = ON.getFieldName();
      if (!Comp.U.IdentInfo)
        continue;
        
      break;
        
    case Node::Base:
      // Will be recomputed during the rebuild.
      continue;
    }
    
    Components.push_back(Comp);
  }
  
  // If nothing changed, retain the existing expression.
  if (!getDerived().AlwaysRebuild() &&
      Type == E->getTypeSourceInfo() &&
      !ExprChanged)
    return SemaRef.Owned(E);
  
  // Build a new offsetof expression.
  return getDerived().RebuildOffsetOfExpr(E->getOperatorLoc(), Type,
                                          Components.data(), Components.size(),
                                          E->getRParenLoc());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformOpaqueValueExpr(OpaqueValueExpr *E) {
  assert(getDerived().AlreadyTransformed(E->getType()) &&
         "opaque value expression requires transformation");
  return SemaRef.Owned(E);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformSizeOfAlignOfExpr(SizeOfAlignOfExpr *E) {
  if (E->isArgumentType()) {
    TypeSourceInfo *OldT = E->getArgumentTypeInfo();

    TypeSourceInfo *NewT = getDerived().TransformType(OldT);
    if (!NewT)
      return ExprError();

    if (!getDerived().AlwaysRebuild() && OldT == NewT)
      return SemaRef.Owned(E);

    return getDerived().RebuildSizeOfAlignOf(NewT, E->getOperatorLoc(),
                                             E->isSizeOf(),
                                             E->getSourceRange());
  }

  ExprResult SubExpr;
  {
    // C++0x [expr.sizeof]p1:
    //   The operand is either an expression, which is an unevaluated operand
    //   [...]
    EnterExpressionEvaluationContext Unevaluated(SemaRef, Sema::Unevaluated);

    SubExpr = getDerived().TransformExpr(E->getArgumentExpr());
    if (SubExpr.isInvalid())
      return ExprError();

    if (!getDerived().AlwaysRebuild() && SubExpr.get() == E->getArgumentExpr())
      return SemaRef.Owned(E);
  }

  return getDerived().RebuildSizeOfAlignOf(SubExpr.get(), E->getOperatorLoc(),
                                           E->isSizeOf(),
                                           E->getSourceRange());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformArraySubscriptExpr(ArraySubscriptExpr *E) {
  ExprResult LHS = getDerived().TransformExpr(E->getLHS());
  if (LHS.isInvalid())
    return ExprError();

  ExprResult RHS = getDerived().TransformExpr(E->getRHS());
  if (RHS.isInvalid())
    return ExprError();


  if (!getDerived().AlwaysRebuild() &&
      LHS.get() == E->getLHS() &&
      RHS.get() == E->getRHS())
    return SemaRef.Owned(E);

  return getDerived().RebuildArraySubscriptExpr(LHS.get(),
                                           /*FIXME:*/E->getLHS()->getLocStart(),
                                                RHS.get(),
                                                E->getRBracketLoc());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCallExpr(CallExpr *E) {
  // Transform the callee.
  ExprResult Callee = getDerived().TransformExpr(E->getCallee());
  if (Callee.isInvalid())
    return ExprError();

  // Transform arguments.
  bool ArgChanged = false;
  ASTOwningVector<Expr*> Args(SemaRef);
  for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I) {
    ExprResult Arg = getDerived().TransformExpr(E->getArg(I));
    if (Arg.isInvalid())
      return ExprError();

    ArgChanged = ArgChanged || Arg.get() != E->getArg(I);
    Args.push_back(Arg.get());
  }

  if (!getDerived().AlwaysRebuild() &&
      Callee.get() == E->getCallee() &&
      !ArgChanged)
    return SemaRef.Owned(E);

  // FIXME: Wrong source location information for the '('.
  SourceLocation FakeLParenLoc
    = ((Expr *)Callee.get())->getSourceRange().getBegin();
  return getDerived().RebuildCallExpr(Callee.get(), FakeLParenLoc,
                                      move_arg(Args),
                                      E->getRParenLoc());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformMemberExpr(MemberExpr *E) {
  ExprResult Base = getDerived().TransformExpr(E->getBase());
  if (Base.isInvalid())
    return ExprError();

  NestedNameSpecifier *Qualifier = 0;
  if (E->hasQualifier()) {
    Qualifier
      = getDerived().TransformNestedNameSpecifier(E->getQualifier(),
                                                  E->getQualifierRange());
    if (Qualifier == 0)
      return ExprError();
  }

  ValueDecl *Member
    = cast_or_null<ValueDecl>(getDerived().TransformDecl(E->getMemberLoc(),
                                                         E->getMemberDecl()));
  if (!Member)
    return ExprError();

  NamedDecl *FoundDecl = E->getFoundDecl();
  if (FoundDecl == E->getMemberDecl()) {
    FoundDecl = Member;
  } else {
    FoundDecl = cast_or_null<NamedDecl>(
                   getDerived().TransformDecl(E->getMemberLoc(), FoundDecl));
    if (!FoundDecl)
      return ExprError();
  }

  if (!getDerived().AlwaysRebuild() &&
      Base.get() == E->getBase() &&
      Qualifier == E->getQualifier() &&
      Member == E->getMemberDecl() &&
      FoundDecl == E->getFoundDecl() &&
      !E->hasExplicitTemplateArgs()) {
    
    // Mark it referenced in the new context regardless.
    // FIXME: this is a bit instantiation-specific.
    SemaRef.MarkDeclarationReferenced(E->getMemberLoc(), Member);
    return SemaRef.Owned(E);
  }

  TemplateArgumentListInfo TransArgs;
  if (E->hasExplicitTemplateArgs()) {
    TransArgs.setLAngleLoc(E->getLAngleLoc());
    TransArgs.setRAngleLoc(E->getRAngleLoc());
    for (unsigned I = 0, N = E->getNumTemplateArgs(); I != N; ++I) {
      TemplateArgumentLoc Loc;
      if (getDerived().TransformTemplateArgument(E->getTemplateArgs()[I], Loc))
        return ExprError();
      TransArgs.addArgument(Loc);
    }
  }
  
  // FIXME: Bogus source location for the operator
  SourceLocation FakeOperatorLoc
    = SemaRef.PP.getLocForEndOfToken(E->getBase()->getSourceRange().getEnd());

  // FIXME: to do this check properly, we will need to preserve the
  // first-qualifier-in-scope here, just in case we had a dependent
  // base (and therefore couldn't do the check) and a
  // nested-name-qualifier (and therefore could do the lookup).
  NamedDecl *FirstQualifierInScope = 0;

  return getDerived().RebuildMemberExpr(Base.get(), FakeOperatorLoc,
                                        E->isArrow(),
                                        Qualifier,
                                        E->getQualifierRange(),
                                        E->getMemberNameInfo(),
                                        Member,
                                        FoundDecl,
                                        (E->hasExplicitTemplateArgs()
                                           ? &TransArgs : 0),
                                        FirstQualifierInScope);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformBinaryOperator(BinaryOperator *E) {
  ExprResult LHS = getDerived().TransformExpr(E->getLHS());
  if (LHS.isInvalid())
    return ExprError();

  ExprResult RHS = getDerived().TransformExpr(E->getRHS());
  if (RHS.isInvalid())
    return ExprError();

  if (!getDerived().AlwaysRebuild() &&
      LHS.get() == E->getLHS() &&
      RHS.get() == E->getRHS())
    return SemaRef.Owned(E);

  return getDerived().RebuildBinaryOperator(E->getOperatorLoc(), E->getOpcode(),
                                            LHS.get(), RHS.get());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCompoundAssignOperator(
                                                      CompoundAssignOperator *E) {
  return getDerived().TransformBinaryOperator(E);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformConditionalOperator(ConditionalOperator *E) {
  ExprResult Cond = getDerived().TransformExpr(E->getCond());
  if (Cond.isInvalid())
    return ExprError();

  ExprResult LHS = getDerived().TransformExpr(E->getLHS());
  if (LHS.isInvalid())
    return ExprError();

  ExprResult RHS = getDerived().TransformExpr(E->getRHS());
  if (RHS.isInvalid())
    return ExprError();

  if (!getDerived().AlwaysRebuild() &&
      Cond.get() == E->getCond() &&
      LHS.get() == E->getLHS() &&
      RHS.get() == E->getRHS())
    return SemaRef.Owned(E);

  return getDerived().RebuildConditionalOperator(Cond.get(),
                                                 E->getQuestionLoc(),
                                                 LHS.get(),
                                                 E->getColonLoc(),
                                                 RHS.get());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformImplicitCastExpr(ImplicitCastExpr *E) {
  // Implicit casts are eliminated during transformation, since they
  // will be recomputed by semantic analysis after transformation.
  return getDerived().TransformExpr(E->getSubExprAsWritten());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCStyleCastExpr(CStyleCastExpr *E) {
  TypeSourceInfo *Type = getDerived().TransformType(E->getTypeInfoAsWritten());
  if (!Type)
    return ExprError();
  
  ExprResult SubExpr
    = getDerived().TransformExpr(E->getSubExprAsWritten());
  if (SubExpr.isInvalid())
    return ExprError();

  if (!getDerived().AlwaysRebuild() &&
      Type == E->getTypeInfoAsWritten() &&
      SubExpr.get() == E->getSubExpr())
    return SemaRef.Owned(E);

  return getDerived().RebuildCStyleCastExpr(E->getLParenLoc(),
                                            Type,
                                            E->getRParenLoc(),
                                            SubExpr.get());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCompoundLiteralExpr(CompoundLiteralExpr *E) {
  TypeSourceInfo *OldT = E->getTypeSourceInfo();
  TypeSourceInfo *NewT = getDerived().TransformType(OldT);
  if (!NewT)
    return ExprError();

  ExprResult Init = getDerived().TransformExpr(E->getInitializer());
  if (Init.isInvalid())
    return ExprError();

  if (!getDerived().AlwaysRebuild() &&
      OldT == NewT &&
      Init.get() == E->getInitializer())
    return SemaRef.Owned(E);

  // Note: the expression type doesn't necessarily match the
  // type-as-written, but that's okay, because it should always be
  // derivable from the initializer.

  return getDerived().RebuildCompoundLiteralExpr(E->getLParenLoc(), NewT,
                                   /*FIXME:*/E->getInitializer()->getLocEnd(),
                                                 Init.get());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformExtVectorElementExpr(ExtVectorElementExpr *E) {
  ExprResult Base = getDerived().TransformExpr(E->getBase());
  if (Base.isInvalid())
    return ExprError();

  if (!getDerived().AlwaysRebuild() &&
      Base.get() == E->getBase())
    return SemaRef.Owned(E);

  // FIXME: Bad source location
  SourceLocation FakeOperatorLoc
    = SemaRef.PP.getLocForEndOfToken(E->getBase()->getLocEnd());
  return getDerived().RebuildExtVectorElementExpr(Base.get(), FakeOperatorLoc,
                                                  E->getAccessorLoc(),
                                                  E->getAccessor());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformInitListExpr(InitListExpr *E) {
  bool InitChanged = false;

  ASTOwningVector<Expr*, 4> Inits(SemaRef);
  for (unsigned I = 0, N = E->getNumInits(); I != N; ++I) {
    ExprResult Init = getDerived().TransformExpr(E->getInit(I));
    if (Init.isInvalid())
      return ExprError();

    InitChanged = InitChanged || Init.get() != E->getInit(I);
    Inits.push_back(Init.get());
  }

  if (!getDerived().AlwaysRebuild() && !InitChanged)
    return SemaRef.Owned(E);

  return getDerived().RebuildInitList(E->getLBraceLoc(), move_arg(Inits),
                                      E->getRBraceLoc(), E->getType());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformDesignatedInitExpr(DesignatedInitExpr *E) {
  Designation Desig;

  // transform the initializer value
  ExprResult Init = getDerived().TransformExpr(E->getInit());
  if (Init.isInvalid())
    return ExprError();

  // transform the designators.
  ASTOwningVector<Expr*, 4> ArrayExprs(SemaRef);
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
      ExprResult Index = getDerived().TransformExpr(E->getArrayIndex(*D));
      if (Index.isInvalid())
        return ExprError();

      Desig.AddDesignator(Designator::getArray(Index.get(),
                                               D->getLBracketLoc()));

      ExprChanged = ExprChanged || Init.get() != E->getArrayIndex(*D);
      ArrayExprs.push_back(Index.release());
      continue;
    }

    assert(D->isArrayRangeDesignator() && "New kind of designator?");
    ExprResult Start
      = getDerived().TransformExpr(E->getArrayRangeStart(*D));
    if (Start.isInvalid())
      return ExprError();

    ExprResult End = getDerived().TransformExpr(E->getArrayRangeEnd(*D));
    if (End.isInvalid())
      return ExprError();

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
    return SemaRef.Owned(E);

  return getDerived().RebuildDesignatedInitExpr(Desig, move_arg(ArrayExprs),
                                                E->getEqualOrColonLoc(),
                                                E->usesGNUSyntax(), Init.get());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformImplicitValueInitExpr(
                                                     ImplicitValueInitExpr *E) {
  TemporaryBase Rebase(*this, E->getLocStart(), DeclarationName());
  
  // FIXME: Will we ever have proper type location here? Will we actually
  // need to transform the type?
  QualType T = getDerived().TransformType(E->getType());
  if (T.isNull())
    return ExprError();

  if (!getDerived().AlwaysRebuild() &&
      T == E->getType())
    return SemaRef.Owned(E);

  return getDerived().RebuildImplicitValueInitExpr(T);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformVAArgExpr(VAArgExpr *E) {
  TypeSourceInfo *TInfo = getDerived().TransformType(E->getWrittenTypeInfo());
  if (!TInfo)
    return ExprError();

  ExprResult SubExpr = getDerived().TransformExpr(E->getSubExpr());
  if (SubExpr.isInvalid())
    return ExprError();

  if (!getDerived().AlwaysRebuild() &&
      TInfo == E->getWrittenTypeInfo() &&
      SubExpr.get() == E->getSubExpr())
    return SemaRef.Owned(E);

  return getDerived().RebuildVAArgExpr(E->getBuiltinLoc(), SubExpr.get(),
                                       TInfo, E->getRParenLoc());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformParenListExpr(ParenListExpr *E) {
  bool ArgumentChanged = false;
  ASTOwningVector<Expr*, 4> Inits(SemaRef);
  for (unsigned I = 0, N = E->getNumExprs(); I != N; ++I) {
    ExprResult Init = getDerived().TransformExpr(E->getExpr(I));
    if (Init.isInvalid())
      return ExprError();

    ArgumentChanged = ArgumentChanged || Init.get() != E->getExpr(I);
    Inits.push_back(Init.get());
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
ExprResult
TreeTransform<Derived>::TransformAddrLabelExpr(AddrLabelExpr *E) {
  return getDerived().RebuildAddrLabelExpr(E->getAmpAmpLoc(), E->getLabelLoc(),
                                           E->getLabel());
}

template<typename Derived>
ExprResult 
TreeTransform<Derived>::TransformStmtExpr(StmtExpr *E) {
  StmtResult SubStmt
    = getDerived().TransformCompoundStmt(E->getSubStmt(), true);
  if (SubStmt.isInvalid())
    return ExprError();

  if (!getDerived().AlwaysRebuild() &&
      SubStmt.get() == E->getSubStmt())
    return SemaRef.Owned(E);

  return getDerived().RebuildStmtExpr(E->getLParenLoc(),
                                      SubStmt.get(),
                                      E->getRParenLoc());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformChooseExpr(ChooseExpr *E) {
  ExprResult Cond = getDerived().TransformExpr(E->getCond());
  if (Cond.isInvalid())
    return ExprError();

  ExprResult LHS = getDerived().TransformExpr(E->getLHS());
  if (LHS.isInvalid())
    return ExprError();

  ExprResult RHS = getDerived().TransformExpr(E->getRHS());
  if (RHS.isInvalid())
    return ExprError();

  if (!getDerived().AlwaysRebuild() &&
      Cond.get() == E->getCond() &&
      LHS.get() == E->getLHS() &&
      RHS.get() == E->getRHS())
    return SemaRef.Owned(E);

  return getDerived().RebuildChooseExpr(E->getBuiltinLoc(),
                                        Cond.get(), LHS.get(), RHS.get(),
                                        E->getRParenLoc());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformGNUNullExpr(GNUNullExpr *E) {
  return SemaRef.Owned(E);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCXXOperatorCallExpr(CXXOperatorCallExpr *E) {
  switch (E->getOperator()) {
  case OO_New:
  case OO_Delete:
  case OO_Array_New:
  case OO_Array_Delete:
    llvm_unreachable("new and delete operators cannot use CXXOperatorCallExpr");
    return ExprError();
    
  case OO_Call: {
    // This is a call to an object's operator().
    assert(E->getNumArgs() >= 1 && "Object call is missing arguments");

    // Transform the object itself.
    ExprResult Object = getDerived().TransformExpr(E->getArg(0));
    if (Object.isInvalid())
      return ExprError();

    // FIXME: Poor location information
    SourceLocation FakeLParenLoc
      = SemaRef.PP.getLocForEndOfToken(
                              static_cast<Expr *>(Object.get())->getLocEnd());

    // Transform the call arguments.
    ASTOwningVector<Expr*> Args(SemaRef);
    for (unsigned I = 1, N = E->getNumArgs(); I != N; ++I) {
      if (getDerived().DropCallArgument(E->getArg(I)))
        break;
      
      ExprResult Arg = getDerived().TransformExpr(E->getArg(I));
      if (Arg.isInvalid())
        return ExprError();

      Args.push_back(Arg.release());
    }

    return getDerived().RebuildCallExpr(Object.get(), FakeLParenLoc,
                                        move_arg(Args),
                                        E->getLocEnd());
  }

#define OVERLOADED_OPERATOR(Name,Spelling,Token,Unary,Binary,MemberOnly) \
  case OO_##Name:
#define OVERLOADED_OPERATOR_MULTI(Name,Spelling,Unary,Binary,MemberOnly)
#include "clang/Basic/OperatorKinds.def"
  case OO_Subscript:
    // Handled below.
    break;

  case OO_Conditional:
    llvm_unreachable("conditional operator is not actually overloadable");
    return ExprError();

  case OO_None:
  case NUM_OVERLOADED_OPERATORS:
    llvm_unreachable("not an overloaded operator?");
    return ExprError();
  }

  ExprResult Callee = getDerived().TransformExpr(E->getCallee());
  if (Callee.isInvalid())
    return ExprError();

  ExprResult First = getDerived().TransformExpr(E->getArg(0));
  if (First.isInvalid())
    return ExprError();

  ExprResult Second;
  if (E->getNumArgs() == 2) {
    Second = getDerived().TransformExpr(E->getArg(1));
    if (Second.isInvalid())
      return ExprError();
  }

  if (!getDerived().AlwaysRebuild() &&
      Callee.get() == E->getCallee() &&
      First.get() == E->getArg(0) &&
      (E->getNumArgs() != 2 || Second.get() == E->getArg(1)))
    return SemaRef.Owned(E);

  return getDerived().RebuildCXXOperatorCallExpr(E->getOperator(),
                                                 E->getOperatorLoc(),
                                                 Callee.get(),
                                                 First.get(),
                                                 Second.get());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCXXMemberCallExpr(CXXMemberCallExpr *E) {
  return getDerived().TransformCallExpr(E);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCXXNamedCastExpr(CXXNamedCastExpr *E) {
  TypeSourceInfo *Type = getDerived().TransformType(E->getTypeInfoAsWritten());
  if (!Type)
    return ExprError();
  
  ExprResult SubExpr
    = getDerived().TransformExpr(E->getSubExprAsWritten());
  if (SubExpr.isInvalid())
    return ExprError();

  if (!getDerived().AlwaysRebuild() &&
      Type == E->getTypeInfoAsWritten() &&
      SubExpr.get() == E->getSubExpr())
    return SemaRef.Owned(E);

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
                                              Type,
                                              FakeRAngleLoc,
                                              FakeRAngleLoc,
                                              SubExpr.get(),
                                              FakeRParenLoc);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCXXStaticCastExpr(CXXStaticCastExpr *E) {
  return getDerived().TransformCXXNamedCastExpr(E);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCXXDynamicCastExpr(CXXDynamicCastExpr *E) {
  return getDerived().TransformCXXNamedCastExpr(E);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCXXReinterpretCastExpr(
                                                      CXXReinterpretCastExpr *E) {
  return getDerived().TransformCXXNamedCastExpr(E);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCXXConstCastExpr(CXXConstCastExpr *E) {
  return getDerived().TransformCXXNamedCastExpr(E);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCXXFunctionalCastExpr(
                                                     CXXFunctionalCastExpr *E) {
  TypeSourceInfo *Type = getDerived().TransformType(E->getTypeInfoAsWritten());
  if (!Type)
    return ExprError();

  ExprResult SubExpr
    = getDerived().TransformExpr(E->getSubExprAsWritten());
  if (SubExpr.isInvalid())
    return ExprError();

  if (!getDerived().AlwaysRebuild() &&
      Type == E->getTypeInfoAsWritten() &&
      SubExpr.get() == E->getSubExpr())
    return SemaRef.Owned(E);

  return getDerived().RebuildCXXFunctionalCastExpr(Type,
                                      /*FIXME:*/E->getSubExpr()->getLocStart(),
                                                   SubExpr.get(),
                                                   E->getRParenLoc());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCXXTypeidExpr(CXXTypeidExpr *E) {
  if (E->isTypeOperand()) {
    TypeSourceInfo *TInfo
      = getDerived().TransformType(E->getTypeOperandSourceInfo());
    if (!TInfo)
      return ExprError();

    if (!getDerived().AlwaysRebuild() &&
        TInfo == E->getTypeOperandSourceInfo())
      return SemaRef.Owned(E);

    return getDerived().RebuildCXXTypeidExpr(E->getType(),
                                             E->getLocStart(),
                                             TInfo,
                                             E->getLocEnd());
  }

  // We don't know whether the expression is potentially evaluated until
  // after we perform semantic analysis, so the expression is potentially
  // potentially evaluated.
  EnterExpressionEvaluationContext Unevaluated(SemaRef,
                                      Sema::PotentiallyPotentiallyEvaluated);

  ExprResult SubExpr = getDerived().TransformExpr(E->getExprOperand());
  if (SubExpr.isInvalid())
    return ExprError();

  if (!getDerived().AlwaysRebuild() &&
      SubExpr.get() == E->getExprOperand())
    return SemaRef.Owned(E);

  return getDerived().RebuildCXXTypeidExpr(E->getType(),
                                           E->getLocStart(),
                                           SubExpr.get(),
                                           E->getLocEnd());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCXXUuidofExpr(CXXUuidofExpr *E) {
  if (E->isTypeOperand()) {
    TypeSourceInfo *TInfo
      = getDerived().TransformType(E->getTypeOperandSourceInfo());
    if (!TInfo)
      return ExprError();

    if (!getDerived().AlwaysRebuild() &&
        TInfo == E->getTypeOperandSourceInfo())
      return SemaRef.Owned(E);

    return getDerived().RebuildCXXTypeidExpr(E->getType(),
                                             E->getLocStart(),
                                             TInfo,
                                             E->getLocEnd());
  }

  // We don't know whether the expression is potentially evaluated until
  // after we perform semantic analysis, so the expression is potentially
  // potentially evaluated.
  EnterExpressionEvaluationContext Unevaluated(SemaRef, Sema::Unevaluated);

  ExprResult SubExpr = getDerived().TransformExpr(E->getExprOperand());
  if (SubExpr.isInvalid())
    return ExprError();

  if (!getDerived().AlwaysRebuild() &&
      SubExpr.get() == E->getExprOperand())
    return SemaRef.Owned(E);

  return getDerived().RebuildCXXUuidofExpr(E->getType(),
                                           E->getLocStart(),
                                           SubExpr.get(),
                                           E->getLocEnd());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCXXBoolLiteralExpr(CXXBoolLiteralExpr *E) {
  return SemaRef.Owned(E);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCXXNullPtrLiteralExpr(
                                                     CXXNullPtrLiteralExpr *E) {
  return SemaRef.Owned(E);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCXXThisExpr(CXXThisExpr *E) {
  DeclContext *DC = getSema().getFunctionLevelDeclContext();
  CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(DC);
  QualType T = MD->getThisType(getSema().Context);

  if (!getDerived().AlwaysRebuild() && T == E->getType())
    return SemaRef.Owned(E);

  return getDerived().RebuildCXXThisExpr(E->getLocStart(), T, E->isImplicit());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCXXThrowExpr(CXXThrowExpr *E) {
  ExprResult SubExpr = getDerived().TransformExpr(E->getSubExpr());
  if (SubExpr.isInvalid())
    return ExprError();

  if (!getDerived().AlwaysRebuild() &&
      SubExpr.get() == E->getSubExpr())
    return SemaRef.Owned(E);

  return getDerived().RebuildCXXThrowExpr(E->getThrowLoc(), SubExpr.get());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCXXDefaultArgExpr(CXXDefaultArgExpr *E) {
  ParmVarDecl *Param
    = cast_or_null<ParmVarDecl>(getDerived().TransformDecl(E->getLocStart(),
                                                           E->getParam()));
  if (!Param)
    return ExprError();

  if (!getDerived().AlwaysRebuild() &&
      Param == E->getParam())
    return SemaRef.Owned(E);

  return getDerived().RebuildCXXDefaultArgExpr(E->getUsedLocation(), Param);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCXXScalarValueInitExpr(
                                                    CXXScalarValueInitExpr *E) {
  TypeSourceInfo *T = getDerived().TransformType(E->getTypeSourceInfo());
  if (!T)
    return ExprError();
  
  if (!getDerived().AlwaysRebuild() &&
      T == E->getTypeSourceInfo())
    return SemaRef.Owned(E);

  return getDerived().RebuildCXXScalarValueInitExpr(T, 
                                          /*FIXME:*/T->getTypeLoc().getEndLoc(),
                                                    E->getRParenLoc());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCXXNewExpr(CXXNewExpr *E) {
  // Transform the type that we're allocating
  TypeSourceInfo *AllocTypeInfo
    = getDerived().TransformType(E->getAllocatedTypeSourceInfo());
  if (!AllocTypeInfo)
    return ExprError();

  // Transform the size of the array we're allocating (if any).
  ExprResult ArraySize = getDerived().TransformExpr(E->getArraySize());
  if (ArraySize.isInvalid())
    return ExprError();

  // Transform the placement arguments (if any).
  bool ArgumentChanged = false;
  ASTOwningVector<Expr*> PlacementArgs(SemaRef);
  for (unsigned I = 0, N = E->getNumPlacementArgs(); I != N; ++I) {
    if (getDerived().DropCallArgument(E->getPlacementArg(I))) {
      ArgumentChanged = true;
      break;
    }

    ExprResult Arg = getDerived().TransformExpr(E->getPlacementArg(I));
    if (Arg.isInvalid())
      return ExprError();

    ArgumentChanged = ArgumentChanged || Arg.get() != E->getPlacementArg(I);
    PlacementArgs.push_back(Arg.take());
  }

  // transform the constructor arguments (if any).
  ASTOwningVector<Expr*> ConstructorArgs(SemaRef);
  for (unsigned I = 0, N = E->getNumConstructorArgs(); I != N; ++I) {
    if (getDerived().DropCallArgument(E->getConstructorArg(I))) {
      ArgumentChanged = true;
      break;
    }
    
    ExprResult Arg = getDerived().TransformExpr(E->getConstructorArg(I));
    if (Arg.isInvalid())
      return ExprError();

    ArgumentChanged = ArgumentChanged || Arg.get() != E->getConstructorArg(I);
    ConstructorArgs.push_back(Arg.take());
  }

  // Transform constructor, new operator, and delete operator.
  CXXConstructorDecl *Constructor = 0;
  if (E->getConstructor()) {
    Constructor = cast_or_null<CXXConstructorDecl>(
                                   getDerived().TransformDecl(E->getLocStart(),
                                                         E->getConstructor()));
    if (!Constructor)
      return ExprError();
  }

  FunctionDecl *OperatorNew = 0;
  if (E->getOperatorNew()) {
    OperatorNew = cast_or_null<FunctionDecl>(
                                 getDerived().TransformDecl(E->getLocStart(),
                                                         E->getOperatorNew()));
    if (!OperatorNew)
      return ExprError();
  }

  FunctionDecl *OperatorDelete = 0;
  if (E->getOperatorDelete()) {
    OperatorDelete = cast_or_null<FunctionDecl>(
                                   getDerived().TransformDecl(E->getLocStart(),
                                                       E->getOperatorDelete()));
    if (!OperatorDelete)
      return ExprError();
  }
  
  if (!getDerived().AlwaysRebuild() &&
      AllocTypeInfo == E->getAllocatedTypeSourceInfo() &&
      ArraySize.get() == E->getArraySize() &&
      Constructor == E->getConstructor() &&
      OperatorNew == E->getOperatorNew() &&
      OperatorDelete == E->getOperatorDelete() &&
      !ArgumentChanged) {
    // Mark any declarations we need as referenced.
    // FIXME: instantiation-specific.
    if (Constructor)
      SemaRef.MarkDeclarationReferenced(E->getLocStart(), Constructor);
    if (OperatorNew)
      SemaRef.MarkDeclarationReferenced(E->getLocStart(), OperatorNew);
    if (OperatorDelete)
      SemaRef.MarkDeclarationReferenced(E->getLocStart(), OperatorDelete);
    return SemaRef.Owned(E);
  }

  QualType AllocType = AllocTypeInfo->getType();
  if (!ArraySize.get()) {
    // If no array size was specified, but the new expression was
    // instantiated with an array type (e.g., "new T" where T is
    // instantiated with "int[4]"), extract the outer bound from the
    // array type as our array size. We do this with constant and
    // dependently-sized array types.
    const ArrayType *ArrayT = SemaRef.Context.getAsArrayType(AllocType);
    if (!ArrayT) {
      // Do nothing
    } else if (const ConstantArrayType *ConsArrayT
                                     = dyn_cast<ConstantArrayType>(ArrayT)) {
      ArraySize 
        = SemaRef.Owned(IntegerLiteral::Create(SemaRef.Context,
                                               ConsArrayT->getSize(), 
                                               SemaRef.Context.getSizeType(),
                                               /*FIXME:*/E->getLocStart()));
      AllocType = ConsArrayT->getElementType();
    } else if (const DependentSizedArrayType *DepArrayT
                              = dyn_cast<DependentSizedArrayType>(ArrayT)) {
      if (DepArrayT->getSizeExpr()) {
        ArraySize = SemaRef.Owned(DepArrayT->getSizeExpr());
        AllocType = DepArrayT->getElementType();
      }
    }
  }
  
  return getDerived().RebuildCXXNewExpr(E->getLocStart(),
                                        E->isGlobalNew(),
                                        /*FIXME:*/E->getLocStart(),
                                        move_arg(PlacementArgs),
                                        /*FIXME:*/E->getLocStart(),
                                        E->getTypeIdParens(),
                                        AllocType,
                                        AllocTypeInfo,
                                        ArraySize.get(),
                                        /*FIXME:*/E->getLocStart(),
                                        move_arg(ConstructorArgs),
                                        E->getLocEnd());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCXXDeleteExpr(CXXDeleteExpr *E) {
  ExprResult Operand = getDerived().TransformExpr(E->getArgument());
  if (Operand.isInvalid())
    return ExprError();

  // Transform the delete operator, if known.
  FunctionDecl *OperatorDelete = 0;
  if (E->getOperatorDelete()) {
    OperatorDelete = cast_or_null<FunctionDecl>(
                                   getDerived().TransformDecl(E->getLocStart(),
                                                       E->getOperatorDelete()));
    if (!OperatorDelete)
      return ExprError();
  }
  
  if (!getDerived().AlwaysRebuild() &&
      Operand.get() == E->getArgument() &&
      OperatorDelete == E->getOperatorDelete()) {
    // Mark any declarations we need as referenced.
    // FIXME: instantiation-specific.
    if (OperatorDelete)
      SemaRef.MarkDeclarationReferenced(E->getLocStart(), OperatorDelete);
    
    if (!E->getArgument()->isTypeDependent()) {
      QualType Destroyed = SemaRef.Context.getBaseElementType(
                                                         E->getDestroyedType());
      if (const RecordType *DestroyedRec = Destroyed->getAs<RecordType>()) {
        CXXRecordDecl *Record = cast<CXXRecordDecl>(DestroyedRec->getDecl());
        SemaRef.MarkDeclarationReferenced(E->getLocStart(), 
                                          SemaRef.LookupDestructor(Record));
      }
    }
    
    return SemaRef.Owned(E);
  }

  return getDerived().RebuildCXXDeleteExpr(E->getLocStart(),
                                           E->isGlobalDelete(),
                                           E->isArrayForm(),
                                           Operand.get());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCXXPseudoDestructorExpr(
                                                     CXXPseudoDestructorExpr *E) {
  ExprResult Base = getDerived().TransformExpr(E->getBase());
  if (Base.isInvalid())
    return ExprError();

  ParsedType ObjectTypePtr;
  bool MayBePseudoDestructor = false;
  Base = SemaRef.ActOnStartCXXMemberReference(0, Base.get(), 
                                              E->getOperatorLoc(),
                                        E->isArrow()? tok::arrow : tok::period,
                                              ObjectTypePtr,
                                              MayBePseudoDestructor);
  if (Base.isInvalid())
    return ExprError();
                                              
  QualType ObjectType = ObjectTypePtr.get();
  NestedNameSpecifier *Qualifier = E->getQualifier();
  if (Qualifier) {
    Qualifier
      = getDerived().TransformNestedNameSpecifier(E->getQualifier(),
                                                  E->getQualifierRange(),
                                                  ObjectType);
    if (!Qualifier)
      return ExprError();
  }

  PseudoDestructorTypeStorage Destroyed;
  if (E->getDestroyedTypeInfo()) {
    TypeSourceInfo *DestroyedTypeInfo
      = getDerived().TransformTypeInObjectScope(E->getDestroyedTypeInfo(),
                                                ObjectType, 0, Qualifier);
    if (!DestroyedTypeInfo)
      return ExprError();
    Destroyed = DestroyedTypeInfo;
  } else if (ObjectType->isDependentType()) {
    // We aren't likely to be able to resolve the identifier down to a type
    // now anyway, so just retain the identifier.
    Destroyed = PseudoDestructorTypeStorage(E->getDestroyedTypeIdentifier(),
                                            E->getDestroyedTypeLoc());
  } else {
    // Look for a destructor known with the given name.
    CXXScopeSpec SS;
    if (Qualifier) {
      SS.setScopeRep(Qualifier);
      SS.setRange(E->getQualifierRange());
    }
    
    ParsedType T = SemaRef.getDestructorName(E->getTildeLoc(),
                                              *E->getDestroyedTypeIdentifier(),
                                                E->getDestroyedTypeLoc(),
                                                /*Scope=*/0,
                                                SS, ObjectTypePtr,
                                                false);
    if (!T)
      return ExprError();
    
    Destroyed
      = SemaRef.Context.getTrivialTypeSourceInfo(SemaRef.GetTypeFromParser(T),
                                                 E->getDestroyedTypeLoc());
  }

  TypeSourceInfo *ScopeTypeInfo = 0;
  if (E->getScopeTypeInfo()) {
    ScopeTypeInfo = getDerived().TransformType(E->getScopeTypeInfo());
    if (!ScopeTypeInfo)
      return ExprError();
  }
  
  return getDerived().RebuildCXXPseudoDestructorExpr(Base.get(),
                                                     E->getOperatorLoc(),
                                                     E->isArrow(),
                                                     Qualifier,
                                                     E->getQualifierRange(),
                                                     ScopeTypeInfo,
                                                     E->getColonColonLoc(),
                                                     E->getTildeLoc(),
                                                     Destroyed);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformUnresolvedLookupExpr(
                                                  UnresolvedLookupExpr *Old) {
  TemporaryBase Rebase(*this, Old->getNameLoc(), DeclarationName());

  LookupResult R(SemaRef, Old->getName(), Old->getNameLoc(),
                 Sema::LookupOrdinaryName);

  // Transform all the decls.
  for (UnresolvedLookupExpr::decls_iterator I = Old->decls_begin(),
         E = Old->decls_end(); I != E; ++I) {
    NamedDecl *InstD = static_cast<NamedDecl*>(
                                 getDerived().TransformDecl(Old->getNameLoc(),
                                                            *I));
    if (!InstD) {
      // Silently ignore these if a UsingShadowDecl instantiated to nothing.
      // This can happen because of dependent hiding.
      if (isa<UsingShadowDecl>(*I))
        continue;
      else
        return ExprError();
    }

    // Expand using declarations.
    if (isa<UsingDecl>(InstD)) {
      UsingDecl *UD = cast<UsingDecl>(InstD);
      for (UsingDecl::shadow_iterator I = UD->shadow_begin(),
             E = UD->shadow_end(); I != E; ++I)
        R.addDecl(*I);
      continue;
    }

    R.addDecl(InstD);
  }

  // Resolve a kind, but don't do any further analysis.  If it's
  // ambiguous, the callee needs to deal with it.
  R.resolveKind();

  // Rebuild the nested-name qualifier, if present.
  CXXScopeSpec SS;
  NestedNameSpecifier *Qualifier = 0;
  if (Old->getQualifier()) {
    Qualifier = getDerived().TransformNestedNameSpecifier(Old->getQualifier(),
                                                    Old->getQualifierRange());
    if (!Qualifier)
      return ExprError();
    
    SS.setScopeRep(Qualifier);
    SS.setRange(Old->getQualifierRange());
  } 
  
  if (Old->getNamingClass()) {
    CXXRecordDecl *NamingClass
      = cast_or_null<CXXRecordDecl>(getDerived().TransformDecl(
                                                            Old->getNameLoc(),
                                                        Old->getNamingClass()));
    if (!NamingClass)
      return ExprError();
    
    R.setNamingClass(NamingClass);
  }

  // If we have no template arguments, it's a normal declaration name.
  if (!Old->hasExplicitTemplateArgs())
    return getDerived().RebuildDeclarationNameExpr(SS, R, Old->requiresADL());

  // If we have template arguments, rebuild them, then rebuild the
  // templateid expression.
  TemplateArgumentListInfo TransArgs(Old->getLAngleLoc(), Old->getRAngleLoc());
  for (unsigned I = 0, N = Old->getNumTemplateArgs(); I != N; ++I) {
    TemplateArgumentLoc Loc;
    if (getDerived().TransformTemplateArgument(Old->getTemplateArgs()[I], Loc))
      return ExprError();
    TransArgs.addArgument(Loc);
  }

  return getDerived().RebuildTemplateIdExpr(SS, R, Old->requiresADL(),
                                            TransArgs);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformUnaryTypeTraitExpr(UnaryTypeTraitExpr *E) {
  TypeSourceInfo *T = getDerived().TransformType(E->getQueriedTypeSourceInfo());
  if (!T)
    return ExprError();

  if (!getDerived().AlwaysRebuild() &&
      T == E->getQueriedTypeSourceInfo())
    return SemaRef.Owned(E);

  return getDerived().RebuildUnaryTypeTrait(E->getTrait(),
                                            E->getLocStart(),
                                            T,
                                            E->getLocEnd());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformBinaryTypeTraitExpr(BinaryTypeTraitExpr *E) {
  TypeSourceInfo *LhsT = getDerived().TransformType(E->getLhsTypeSourceInfo());
  if (!LhsT)
    return ExprError();

  TypeSourceInfo *RhsT = getDerived().TransformType(E->getRhsTypeSourceInfo());
  if (!RhsT)
    return ExprError();

  if (!getDerived().AlwaysRebuild() &&
      LhsT == E->getLhsTypeSourceInfo() && RhsT == E->getRhsTypeSourceInfo())
    return SemaRef.Owned(E);

  return getDerived().RebuildBinaryTypeTrait(E->getTrait(),
                                            E->getLocStart(),
                                            LhsT, RhsT,
                                            E->getLocEnd());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformDependentScopeDeclRefExpr(
                                               DependentScopeDeclRefExpr *E) {
  NestedNameSpecifier *NNS
    = getDerived().TransformNestedNameSpecifier(E->getQualifier(),
                                                E->getQualifierRange());
  if (!NNS)
    return ExprError();

  // TODO: If this is a conversion-function-id, verify that the
  // destination type name (if present) resolves the same way after
  // instantiation as it did in the local scope.

  DeclarationNameInfo NameInfo
    = getDerived().TransformDeclarationNameInfo(E->getNameInfo());
  if (!NameInfo.getName())
    return ExprError();

  if (!E->hasExplicitTemplateArgs()) {
    if (!getDerived().AlwaysRebuild() &&
        NNS == E->getQualifier() &&
        // Note: it is sufficient to compare the Name component of NameInfo:
        // if name has not changed, DNLoc has not changed either.
        NameInfo.getName() == E->getDeclName())
      return SemaRef.Owned(E);

    return getDerived().RebuildDependentScopeDeclRefExpr(NNS,
                                                         E->getQualifierRange(),
                                                         NameInfo,
                                                         /*TemplateArgs*/ 0);
  }

  TemplateArgumentListInfo TransArgs(E->getLAngleLoc(), E->getRAngleLoc());
  for (unsigned I = 0, N = E->getNumTemplateArgs(); I != N; ++I) {
    TemplateArgumentLoc Loc;
    if (getDerived().TransformTemplateArgument(E->getTemplateArgs()[I], Loc))
      return ExprError();
    TransArgs.addArgument(Loc);
  }

  return getDerived().RebuildDependentScopeDeclRefExpr(NNS,
                                                       E->getQualifierRange(),
                                                       NameInfo,
                                                       &TransArgs);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCXXConstructExpr(CXXConstructExpr *E) {
  // CXXConstructExprs are always implicit, so when we have a
  // 1-argument construction we just transform that argument.
  if (E->getNumArgs() == 1 ||
      (E->getNumArgs() > 1 && getDerived().DropCallArgument(E->getArg(1))))
    return getDerived().TransformExpr(E->getArg(0));

  TemporaryBase Rebase(*this, /*FIXME*/E->getLocStart(), DeclarationName());

  QualType T = getDerived().TransformType(E->getType());
  if (T.isNull())
    return ExprError();

  CXXConstructorDecl *Constructor
    = cast_or_null<CXXConstructorDecl>(
                                getDerived().TransformDecl(E->getLocStart(),
                                                         E->getConstructor()));
  if (!Constructor)
    return ExprError();

  bool ArgumentChanged = false;
  ASTOwningVector<Expr*> Args(SemaRef);
  for (CXXConstructExpr::arg_iterator Arg = E->arg_begin(),
       ArgEnd = E->arg_end();
       Arg != ArgEnd; ++Arg) {
    if (getDerived().DropCallArgument(*Arg)) {
      ArgumentChanged = true;
      break;
    }

    ExprResult TransArg = getDerived().TransformExpr(*Arg);
    if (TransArg.isInvalid())
      return ExprError();

    ArgumentChanged = ArgumentChanged || TransArg.get() != *Arg;
    Args.push_back(TransArg.get());
  }

  if (!getDerived().AlwaysRebuild() &&
      T == E->getType() &&
      Constructor == E->getConstructor() &&
      !ArgumentChanged) {
    // Mark the constructor as referenced.
    // FIXME: Instantiation-specific
    SemaRef.MarkDeclarationReferenced(E->getLocStart(), Constructor);
    return SemaRef.Owned(E);
  }

  return getDerived().RebuildCXXConstructExpr(T, /*FIXME:*/E->getLocStart(),
                                              Constructor, E->isElidable(),
                                              move_arg(Args),
                                              E->requiresZeroInitialization(),
                                              E->getConstructionKind(),
                                              E->getParenRange());
}

/// \brief Transform a C++ temporary-binding expression.
///
/// Since CXXBindTemporaryExpr nodes are implicitly generated, we just
/// transform the subexpression and return that.
template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCXXBindTemporaryExpr(CXXBindTemporaryExpr *E) {
  return getDerived().TransformExpr(E->getSubExpr());
}

/// \brief Transform a C++ expression that contains cleanups that should
/// be run after the expression is evaluated.
///
/// Since ExprWithCleanups nodes are implicitly generated, we
/// just transform the subexpression and return that.
template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformExprWithCleanups(ExprWithCleanups *E) {
  return getDerived().TransformExpr(E->getSubExpr());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCXXTemporaryObjectExpr(
                                                    CXXTemporaryObjectExpr *E) {
  TypeSourceInfo *T = getDerived().TransformType(E->getTypeSourceInfo());
  if (!T)
    return ExprError();

  CXXConstructorDecl *Constructor
    = cast_or_null<CXXConstructorDecl>(
                                  getDerived().TransformDecl(E->getLocStart(), 
                                                         E->getConstructor()));
  if (!Constructor)
    return ExprError();

  bool ArgumentChanged = false;
  ASTOwningVector<Expr*> Args(SemaRef);
  Args.reserve(E->getNumArgs());
  for (CXXTemporaryObjectExpr::arg_iterator Arg = E->arg_begin(),
                                         ArgEnd = E->arg_end();
       Arg != ArgEnd; ++Arg) {
    if (getDerived().DropCallArgument(*Arg)) {
      ArgumentChanged = true;
      break;
    }

    ExprResult TransArg = getDerived().TransformExpr(*Arg);
    if (TransArg.isInvalid())
      return ExprError();

    ArgumentChanged = ArgumentChanged || TransArg.get() != *Arg;
    Args.push_back((Expr *)TransArg.release());
  }

  if (!getDerived().AlwaysRebuild() &&
      T == E->getTypeSourceInfo() &&
      Constructor == E->getConstructor() &&
      !ArgumentChanged) {
    // FIXME: Instantiation-specific
    SemaRef.MarkDeclarationReferenced(E->getLocStart(), Constructor);
    return SemaRef.MaybeBindToTemporary(E);
  }
  
  return getDerived().RebuildCXXTemporaryObjectExpr(T,
                                          /*FIXME:*/T->getTypeLoc().getEndLoc(),
                                                    move_arg(Args),
                                                    E->getLocEnd());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCXXUnresolvedConstructExpr(
                                                  CXXUnresolvedConstructExpr *E) {
  TypeSourceInfo *T = getDerived().TransformType(E->getTypeSourceInfo());
  if (!T)
    return ExprError();

  bool ArgumentChanged = false;
  ASTOwningVector<Expr*> Args(SemaRef);
  for (CXXUnresolvedConstructExpr::arg_iterator Arg = E->arg_begin(),
                                             ArgEnd = E->arg_end();
       Arg != ArgEnd; ++Arg) {
    ExprResult TransArg = getDerived().TransformExpr(*Arg);
    if (TransArg.isInvalid())
      return ExprError();

    ArgumentChanged = ArgumentChanged || TransArg.get() != *Arg;
    Args.push_back(TransArg.get());
  }

  if (!getDerived().AlwaysRebuild() &&
      T == E->getTypeSourceInfo() &&
      !ArgumentChanged)
    return SemaRef.Owned(E);

  // FIXME: we're faking the locations of the commas
  return getDerived().RebuildCXXUnresolvedConstructExpr(T,
                                                        E->getLParenLoc(),
                                                        move_arg(Args),
                                                        E->getRParenLoc());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCXXDependentScopeMemberExpr(
                                             CXXDependentScopeMemberExpr *E) {
  // Transform the base of the expression.
  ExprResult Base((Expr*) 0);
  Expr *OldBase;
  QualType BaseType;
  QualType ObjectType;
  if (!E->isImplicitAccess()) {
    OldBase = E->getBase();
    Base = getDerived().TransformExpr(OldBase);
    if (Base.isInvalid())
      return ExprError();

    // Start the member reference and compute the object's type.
    ParsedType ObjectTy;
    bool MayBePseudoDestructor = false;
    Base = SemaRef.ActOnStartCXXMemberReference(0, Base.get(),
                                                E->getOperatorLoc(),
                                      E->isArrow()? tok::arrow : tok::period,
                                                ObjectTy,
                                                MayBePseudoDestructor);
    if (Base.isInvalid())
      return ExprError();

    ObjectType = ObjectTy.get();
    BaseType = ((Expr*) Base.get())->getType();
  } else {
    OldBase = 0;
    BaseType = getDerived().TransformType(E->getBaseType());
    ObjectType = BaseType->getAs<PointerType>()->getPointeeType();
  }

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
                                                      ObjectType,
                                                      FirstQualifierInScope);
    if (!Qualifier)
      return ExprError();
  }

  // TODO: If this is a conversion-function-id, verify that the
  // destination type name (if present) resolves the same way after
  // instantiation as it did in the local scope.

  DeclarationNameInfo NameInfo
    = getDerived().TransformDeclarationNameInfo(E->getMemberNameInfo());
  if (!NameInfo.getName())
    return ExprError();

  if (!E->hasExplicitTemplateArgs()) {
    // This is a reference to a member without an explicitly-specified
    // template argument list. Optimize for this common case.
    if (!getDerived().AlwaysRebuild() &&
        Base.get() == OldBase &&
        BaseType == E->getBaseType() &&
        Qualifier == E->getQualifier() &&
        NameInfo.getName() == E->getMember() &&
        FirstQualifierInScope == E->getFirstQualifierFoundInScope())
      return SemaRef.Owned(E);

    return getDerived().RebuildCXXDependentScopeMemberExpr(Base.get(),
                                                       BaseType,
                                                       E->isArrow(),
                                                       E->getOperatorLoc(),
                                                       Qualifier,
                                                       E->getQualifierRange(),
                                                       FirstQualifierInScope,
                                                       NameInfo,
                                                       /*TemplateArgs*/ 0);
  }

  TemplateArgumentListInfo TransArgs(E->getLAngleLoc(), E->getRAngleLoc());
  for (unsigned I = 0, N = E->getNumTemplateArgs(); I != N; ++I) {
    TemplateArgumentLoc Loc;
    if (getDerived().TransformTemplateArgument(E->getTemplateArgs()[I], Loc))
      return ExprError();
    TransArgs.addArgument(Loc);
  }

  return getDerived().RebuildCXXDependentScopeMemberExpr(Base.get(),
                                                     BaseType,
                                                     E->isArrow(),
                                                     E->getOperatorLoc(),
                                                     Qualifier,
                                                     E->getQualifierRange(),
                                                     FirstQualifierInScope,
                                                     NameInfo,
                                                     &TransArgs);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformUnresolvedMemberExpr(UnresolvedMemberExpr *Old) {
  // Transform the base of the expression.
  ExprResult Base((Expr*) 0);
  QualType BaseType;
  if (!Old->isImplicitAccess()) {
    Base = getDerived().TransformExpr(Old->getBase());
    if (Base.isInvalid())
      return ExprError();
    BaseType = ((Expr*) Base.get())->getType();
  } else {
    BaseType = getDerived().TransformType(Old->getBaseType());
  }

  NestedNameSpecifier *Qualifier = 0;
  if (Old->getQualifier()) {
    Qualifier
      = getDerived().TransformNestedNameSpecifier(Old->getQualifier(),
                                                  Old->getQualifierRange());
    if (Qualifier == 0)
      return ExprError();
  }

  LookupResult R(SemaRef, Old->getMemberNameInfo(),
                 Sema::LookupOrdinaryName);

  // Transform all the decls.
  for (UnresolvedMemberExpr::decls_iterator I = Old->decls_begin(),
         E = Old->decls_end(); I != E; ++I) {
    NamedDecl *InstD = static_cast<NamedDecl*>(
                                getDerived().TransformDecl(Old->getMemberLoc(),
                                                           *I));
    if (!InstD) {
      // Silently ignore these if a UsingShadowDecl instantiated to nothing.
      // This can happen because of dependent hiding.
      if (isa<UsingShadowDecl>(*I))
        continue;
      else
        return ExprError();
    }

    // Expand using declarations.
    if (isa<UsingDecl>(InstD)) {
      UsingDecl *UD = cast<UsingDecl>(InstD);
      for (UsingDecl::shadow_iterator I = UD->shadow_begin(),
             E = UD->shadow_end(); I != E; ++I)
        R.addDecl(*I);
      continue;
    }

    R.addDecl(InstD);
  }

  R.resolveKind();

  // Determine the naming class.
  if (Old->getNamingClass()) {
    CXXRecordDecl *NamingClass 
      = cast_or_null<CXXRecordDecl>(getDerived().TransformDecl(
                                                          Old->getMemberLoc(),
                                                        Old->getNamingClass()));
    if (!NamingClass)
      return ExprError();
    
    R.setNamingClass(NamingClass);
  }
  
  TemplateArgumentListInfo TransArgs;
  if (Old->hasExplicitTemplateArgs()) {
    TransArgs.setLAngleLoc(Old->getLAngleLoc());
    TransArgs.setRAngleLoc(Old->getRAngleLoc());
    for (unsigned I = 0, N = Old->getNumTemplateArgs(); I != N; ++I) {
      TemplateArgumentLoc Loc;
      if (getDerived().TransformTemplateArgument(Old->getTemplateArgs()[I],
                                                 Loc))
        return ExprError();
      TransArgs.addArgument(Loc);
    }
  }

  // FIXME: to do this check properly, we will need to preserve the
  // first-qualifier-in-scope here, just in case we had a dependent
  // base (and therefore couldn't do the check) and a
  // nested-name-qualifier (and therefore could do the lookup).
  NamedDecl *FirstQualifierInScope = 0;
  
  return getDerived().RebuildUnresolvedMemberExpr(Base.get(),
                                                  BaseType,
                                                  Old->getOperatorLoc(),
                                                  Old->isArrow(),
                                                  Qualifier,
                                                  Old->getQualifierRange(),
                                                  FirstQualifierInScope,
                                                  R,
                                              (Old->hasExplicitTemplateArgs()
                                                  ? &TransArgs : 0));
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformCXXNoexceptExpr(CXXNoexceptExpr *E) {
  ExprResult SubExpr = getDerived().TransformExpr(E->getOperand());
  if (SubExpr.isInvalid())
    return ExprError();

  if (!getDerived().AlwaysRebuild() && SubExpr.get() == E->getOperand())
    return SemaRef.Owned(E);

  return getDerived().RebuildCXXNoexceptExpr(E->getSourceRange(),SubExpr.get());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformObjCStringLiteral(ObjCStringLiteral *E) {
  return SemaRef.Owned(E);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformObjCEncodeExpr(ObjCEncodeExpr *E) {
  TypeSourceInfo *EncodedTypeInfo
    = getDerived().TransformType(E->getEncodedTypeSourceInfo());
  if (!EncodedTypeInfo)
    return ExprError();

  if (!getDerived().AlwaysRebuild() &&
      EncodedTypeInfo == E->getEncodedTypeSourceInfo())
    return SemaRef.Owned(E);

  return getDerived().RebuildObjCEncodeExpr(E->getAtLoc(),
                                            EncodedTypeInfo,
                                            E->getRParenLoc());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformObjCMessageExpr(ObjCMessageExpr *E) {
  // Transform arguments.
  bool ArgChanged = false;
  ASTOwningVector<Expr*> Args(SemaRef);
  for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I) {
    ExprResult Arg = getDerived().TransformExpr(E->getArg(I));
    if (Arg.isInvalid())
      return ExprError();
    
    ArgChanged = ArgChanged || Arg.get() != E->getArg(I);
    Args.push_back(Arg.get());
  }

  if (E->getReceiverKind() == ObjCMessageExpr::Class) {
    // Class message: transform the receiver type.
    TypeSourceInfo *ReceiverTypeInfo
      = getDerived().TransformType(E->getClassReceiverTypeInfo());
    if (!ReceiverTypeInfo)
      return ExprError();
    
    // If nothing changed, just retain the existing message send.
    if (!getDerived().AlwaysRebuild() &&
        ReceiverTypeInfo == E->getClassReceiverTypeInfo() && !ArgChanged)
      return SemaRef.Owned(E);

    // Build a new class message send.
    return getDerived().RebuildObjCMessageExpr(ReceiverTypeInfo,
                                               E->getSelector(),
                                               E->getSelectorLoc(),
                                               E->getMethodDecl(),
                                               E->getLeftLoc(),
                                               move_arg(Args),
                                               E->getRightLoc());
  }

  // Instance message: transform the receiver
  assert(E->getReceiverKind() == ObjCMessageExpr::Instance &&
         "Only class and instance messages may be instantiated");
  ExprResult Receiver
    = getDerived().TransformExpr(E->getInstanceReceiver());
  if (Receiver.isInvalid())
    return ExprError();

  // If nothing changed, just retain the existing message send.
  if (!getDerived().AlwaysRebuild() &&
      Receiver.get() == E->getInstanceReceiver() && !ArgChanged)
    return SemaRef.Owned(E);
  
  // Build a new instance message send.
  return getDerived().RebuildObjCMessageExpr(Receiver.get(),
                                             E->getSelector(),
                                             E->getSelectorLoc(),
                                             E->getMethodDecl(),
                                             E->getLeftLoc(),
                                             move_arg(Args),
                                             E->getRightLoc());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformObjCSelectorExpr(ObjCSelectorExpr *E) {
  return SemaRef.Owned(E);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformObjCProtocolExpr(ObjCProtocolExpr *E) {
  return SemaRef.Owned(E);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformObjCIvarRefExpr(ObjCIvarRefExpr *E) {
  // Transform the base expression.
  ExprResult Base = getDerived().TransformExpr(E->getBase());
  if (Base.isInvalid())
    return ExprError();

  // We don't need to transform the ivar; it will never change.
  
  // If nothing changed, just retain the existing expression.
  if (!getDerived().AlwaysRebuild() &&
      Base.get() == E->getBase())
    return SemaRef.Owned(E);
  
  return getDerived().RebuildObjCIvarRefExpr(Base.get(), E->getDecl(),
                                             E->getLocation(),
                                             E->isArrow(), E->isFreeIvar());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformObjCPropertyRefExpr(ObjCPropertyRefExpr *E) {
  // 'super' and types never change. Property never changes. Just
  // retain the existing expression.
  if (!E->isObjectReceiver())
    return SemaRef.Owned(E);
  
  // Transform the base expression.
  ExprResult Base = getDerived().TransformExpr(E->getBase());
  if (Base.isInvalid())
    return ExprError();
  
  // We don't need to transform the property; it will never change.
  
  // If nothing changed, just retain the existing expression.
  if (!getDerived().AlwaysRebuild() &&
      Base.get() == E->getBase())
    return SemaRef.Owned(E);

  if (E->isExplicitProperty())
    return getDerived().RebuildObjCPropertyRefExpr(Base.get(),
                                                   E->getExplicitProperty(),
                                                   E->getLocation());

  return getDerived().RebuildObjCPropertyRefExpr(Base.get(),
                                                 E->getType(),
                                                 E->getImplicitPropertyGetter(),
                                                 E->getImplicitPropertySetter(),
                                                 E->getLocation());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformObjCIsaExpr(ObjCIsaExpr *E) {
  // Transform the base expression.
  ExprResult Base = getDerived().TransformExpr(E->getBase());
  if (Base.isInvalid())
    return ExprError();
  
  // If nothing changed, just retain the existing expression.
  if (!getDerived().AlwaysRebuild() &&
      Base.get() == E->getBase())
    return SemaRef.Owned(E);
  
  return getDerived().RebuildObjCIsaExpr(Base.get(), E->getIsaMemberLoc(),
                                         E->isArrow());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformShuffleVectorExpr(ShuffleVectorExpr *E) {
  bool ArgumentChanged = false;
  ASTOwningVector<Expr*> SubExprs(SemaRef);
  for (unsigned I = 0, N = E->getNumSubExprs(); I != N; ++I) {
    ExprResult SubExpr = getDerived().TransformExpr(E->getExpr(I));
    if (SubExpr.isInvalid())
      return ExprError();

    ArgumentChanged = ArgumentChanged || SubExpr.get() != E->getExpr(I);
    SubExprs.push_back(SubExpr.get());
  }

  if (!getDerived().AlwaysRebuild() &&
      !ArgumentChanged)
    return SemaRef.Owned(E);

  return getDerived().RebuildShuffleVectorExpr(E->getBuiltinLoc(),
                                               move_arg(SubExprs),
                                               E->getRParenLoc());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformBlockExpr(BlockExpr *E) {
  SourceLocation CaretLoc(E->getExprLoc());
  
  SemaRef.ActOnBlockStart(CaretLoc, /*Scope=*/0);
  BlockScopeInfo *CurBlock = SemaRef.getCurBlock();
  CurBlock->TheDecl->setIsVariadic(E->getBlockDecl()->isVariadic());
  llvm::SmallVector<ParmVarDecl*, 4> Params;
  llvm::SmallVector<QualType, 4> ParamTypes;
  
  // Parameter substitution.
  const BlockDecl *BD = E->getBlockDecl();
  for (BlockDecl::param_const_iterator P = BD->param_begin(),
       EN = BD->param_end(); P != EN; ++P) {
    ParmVarDecl *OldParm = (*P);
    ParmVarDecl *NewParm = getDerived().TransformFunctionTypeParam(OldParm);
    QualType NewType = NewParm->getType();
    Params.push_back(NewParm);
    ParamTypes.push_back(NewParm->getType());
  }
  
  const FunctionType *BExprFunctionType = E->getFunctionType();
  QualType BExprResultType = BExprFunctionType->getResultType();
  if (!BExprResultType.isNull()) {
    if (!BExprResultType->isDependentType())
      CurBlock->ReturnType = BExprResultType;
    else if (BExprResultType != SemaRef.Context.DependentTy)
      CurBlock->ReturnType = getDerived().TransformType(BExprResultType);
  }
    
  // Transform the body
  StmtResult Body = getDerived().TransformStmt(E->getBody());
  if (Body.isInvalid())
    return ExprError();
  // Set the parameters on the block decl.
  if (!Params.empty())
    CurBlock->TheDecl->setParams(Params.data(), Params.size());
    
  QualType FunctionType = getDerived().RebuildFunctionProtoType(
                                                        CurBlock->ReturnType,
                                                        ParamTypes.data(),
                                                        ParamTypes.size(),
                                                        BD->isVariadic(),
                                                        0,
                                               BExprFunctionType->getExtInfo());
  
  CurBlock->FunctionType = FunctionType;
  return SemaRef.ActOnBlockStmtExpr(CaretLoc, Body.get(), /*Scope=*/0);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformBlockDeclRefExpr(BlockDeclRefExpr *E) {
  NestedNameSpecifier *Qualifier = 0;
    
  ValueDecl *ND
  = cast_or_null<ValueDecl>(getDerived().TransformDecl(E->getLocation(),
                                                       E->getDecl()));
  if (!ND)
    return ExprError();

  if (!getDerived().AlwaysRebuild() &&
      ND == E->getDecl()) {
    // Mark it referenced in the new context regardless.
    // FIXME: this is a bit instantiation-specific.
    SemaRef.MarkDeclarationReferenced(E->getLocation(), ND);
    
    return SemaRef.Owned(E);
  }
  
  DeclarationNameInfo NameInfo(E->getDecl()->getDeclName(), E->getLocation());
  return getDerived().RebuildDeclRefExpr(Qualifier, SourceLocation(),
                                         ND, NameInfo, 0);
}

//===----------------------------------------------------------------------===//
// Type reconstruction
//===----------------------------------------------------------------------===//

template<typename Derived>
QualType TreeTransform<Derived>::RebuildPointerType(QualType PointeeType,
                                                    SourceLocation Star) {
  return SemaRef.BuildPointerType(PointeeType, Star,
                                  getDerived().getBaseEntity());
}

template<typename Derived>
QualType TreeTransform<Derived>::RebuildBlockPointerType(QualType PointeeType,
                                                         SourceLocation Star) {
  return SemaRef.BuildBlockPointerType(PointeeType, Star,
                                       getDerived().getBaseEntity());
}

template<typename Derived>
QualType
TreeTransform<Derived>::RebuildReferenceType(QualType ReferentType,
                                             bool WrittenAsLValue,
                                             SourceLocation Sigil) {
  return SemaRef.BuildReferenceType(ReferentType, WrittenAsLValue,
                                    Sigil, getDerived().getBaseEntity());
}

template<typename Derived>
QualType
TreeTransform<Derived>::RebuildMemberPointerType(QualType PointeeType,
                                                 QualType ClassType,
                                                 SourceLocation Sigil) {
  return SemaRef.BuildMemberPointerType(PointeeType, ClassType,
                                        Sigil, getDerived().getBaseEntity());
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

  IntegerLiteral ArraySize(SemaRef.Context, *Size, SizeType,
                           /*FIXME*/BracketsRange.getBegin());
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
                                                 Expr *SizeExpr,
                                                 unsigned IndexTypeQuals,
                                                 SourceRange BracketsRange) {
  return getDerived().RebuildArrayType(ElementType, SizeMod, 0,
                                       SizeExpr,
                                       IndexTypeQuals, BracketsRange);
}

template<typename Derived>
QualType
TreeTransform<Derived>::RebuildDependentSizedArrayType(QualType ElementType,
                                          ArrayType::ArraySizeModifier SizeMod,
                                                       Expr *SizeExpr,
                                                       unsigned IndexTypeQuals,
                                                   SourceRange BracketsRange) {
  return getDerived().RebuildArrayType(ElementType, SizeMod, 0,
                                       SizeExpr,
                                       IndexTypeQuals, BracketsRange);
}

template<typename Derived>
QualType TreeTransform<Derived>::RebuildVectorType(QualType ElementType,
                                               unsigned NumElements,
                                               VectorType::VectorKind VecKind) {
  // FIXME: semantic checking!
  return SemaRef.Context.getVectorType(ElementType, NumElements, VecKind);
}

template<typename Derived>
QualType TreeTransform<Derived>::RebuildExtVectorType(QualType ElementType,
                                                      unsigned NumElements,
                                                 SourceLocation AttributeLoc) {
  llvm::APInt numElements(SemaRef.Context.getIntWidth(SemaRef.Context.IntTy),
                          NumElements, true);
  IntegerLiteral *VectorSize
    = IntegerLiteral::Create(SemaRef.Context, numElements, SemaRef.Context.IntTy,
                             AttributeLoc);
  return SemaRef.BuildExtVectorType(ElementType, VectorSize, AttributeLoc);
}

template<typename Derived>
QualType
TreeTransform<Derived>::RebuildDependentSizedExtVectorType(QualType ElementType,
                                                           Expr *SizeExpr,
                                                  SourceLocation AttributeLoc) {
  return SemaRef.BuildExtVectorType(ElementType, SizeExpr, AttributeLoc);
}

template<typename Derived>
QualType TreeTransform<Derived>::RebuildFunctionProtoType(QualType T,
                                                          QualType *ParamTypes,
                                                        unsigned NumParamTypes,
                                                          bool Variadic,
                                                          unsigned Quals,
                                            const FunctionType::ExtInfo &Info) {
  return SemaRef.BuildFunctionType(T, ParamTypes, NumParamTypes, Variadic,
                                   Quals,
                                   getDerived().getBaseLocation(),
                                   getDerived().getBaseEntity(),
                                   Info);
}

template<typename Derived>
QualType TreeTransform<Derived>::RebuildFunctionNoProtoType(QualType T) {
  return SemaRef.Context.getFunctionNoProtoType(T);
}

template<typename Derived>
QualType TreeTransform<Derived>::RebuildUnresolvedUsingType(Decl *D) {
  assert(D && "no decl found");
  if (D->isInvalidDecl()) return QualType();

  // FIXME: Doesn't account for ObjCInterfaceDecl!
  TypeDecl *Ty;
  if (isa<UsingDecl>(D)) {
    UsingDecl *Using = cast<UsingDecl>(D);
    assert(Using->isTypeName() &&
           "UnresolvedUsingTypenameDecl transformed to non-typename using");

    // A valid resolved using typename decl points to exactly one type decl.
    assert(++Using->shadow_begin() == Using->shadow_end());
    Ty = cast<TypeDecl>((*Using->shadow_begin())->getTargetDecl());
    
  } else {
    assert(isa<UnresolvedUsingTypenameDecl>(D) &&
           "UnresolvedUsingTypenameDecl transformed to non-using decl");
    Ty = cast<UnresolvedUsingTypenameDecl>(D);
  }

  return SemaRef.Context.getTypeDeclType(Ty);
}

template<typename Derived>
QualType TreeTransform<Derived>::RebuildTypeOfExprType(Expr *E,
                                                       SourceLocation Loc) {
  return SemaRef.BuildTypeofExprType(E, Loc);
}

template<typename Derived>
QualType TreeTransform<Derived>::RebuildTypeOfType(QualType Underlying) {
  return SemaRef.Context.getTypeOfType(Underlying);
}

template<typename Derived>
QualType TreeTransform<Derived>::RebuildDecltypeType(Expr *E,
                                                     SourceLocation Loc) {
  return SemaRef.BuildDecltypeType(E, Loc);
}

template<typename Derived>
QualType TreeTransform<Derived>::RebuildTemplateSpecializationType(
                                                      TemplateName Template,
                                             SourceLocation TemplateNameLoc,
                               const TemplateArgumentListInfo &TemplateArgs) {
  return SemaRef.CheckTemplateIdType(Template, TemplateNameLoc, TemplateArgs);
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
                                                        false, false));
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
    assert(!T.hasLocalQualifiers() && "Can't get cv-qualifiers here");
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
                                            SourceRange QualifierRange,
                                            const IdentifierInfo &II,
                                            QualType ObjectType,
                                            NamedDecl *FirstQualifierInScope) {
  CXXScopeSpec SS;
  SS.setRange(QualifierRange);
  SS.setScopeRep(Qualifier);
  UnqualifiedId Name;
  Name.setIdentifier(&II, /*FIXME:*/getDerived().getBaseLocation());
  Sema::TemplateTy Template;
  getSema().ActOnDependentTemplateName(/*Scope=*/0,
                                       /*FIXME:*/getDerived().getBaseLocation(),
                                       SS,
                                       Name,
                                       ParsedType::make(ObjectType),
                                       /*EnteringContext=*/false,
                                       Template);
  return Template.get();
}

template<typename Derived>
TemplateName
TreeTransform<Derived>::RebuildTemplateName(NestedNameSpecifier *Qualifier,
                                            OverloadedOperatorKind Operator,
                                            QualType ObjectType) {
  CXXScopeSpec SS;
  SS.setRange(SourceRange(getDerived().getBaseLocation()));
  SS.setScopeRep(Qualifier);
  UnqualifiedId Name;
  SourceLocation SymbolLocations[3]; // FIXME: Bogus location information.
  Name.setOperatorFunctionId(/*FIXME:*/getDerived().getBaseLocation(),
                             Operator, SymbolLocations);
  Sema::TemplateTy Template;
  getSema().ActOnDependentTemplateName(/*Scope=*/0,
                                       /*FIXME:*/getDerived().getBaseLocation(),
                                       SS,
                                       Name,
                                       ParsedType::make(ObjectType),
                                       /*EnteringContext=*/false,
                                       Template);
  return Template.template getAsVal<TemplateName>();
}
  
template<typename Derived>
ExprResult
TreeTransform<Derived>::RebuildCXXOperatorCallExpr(OverloadedOperatorKind Op,
                                                   SourceLocation OpLoc,
                                                   Expr *OrigCallee,
                                                   Expr *First,
                                                   Expr *Second) {
  Expr *Callee = OrigCallee->IgnoreParenCasts();
  bool isPostIncDec = Second && (Op == OO_PlusPlus || Op == OO_MinusMinus);

  // Determine whether this should be a builtin operation.
  if (Op == OO_Subscript) {
    if (!First->getType()->isOverloadableType() &&
        !Second->getType()->isOverloadableType())
      return getSema().CreateBuiltinArraySubscriptExpr(First,
                                                       Callee->getLocStart(),
                                                       Second, OpLoc);
  } else if (Op == OO_Arrow) {
    // -> is never a builtin operation.
    return SemaRef.BuildOverloadedArrowExpr(0, First, OpLoc);
  } else if (Second == 0 || isPostIncDec) {
    if (!First->getType()->isOverloadableType()) {
      // The argument is not of overloadable type, so try to create a
      // built-in unary operation.
      UnaryOperatorKind Opc
        = UnaryOperator::getOverloadedOpcode(Op, isPostIncDec);

      return getSema().CreateBuiltinUnaryOp(OpLoc, Opc, First);
    }
  } else {
    if (!First->getType()->isOverloadableType() &&
        !Second->getType()->isOverloadableType()) {
      // Neither of the arguments is an overloadable type, so try to
      // create a built-in binary operation.
      BinaryOperatorKind Opc = BinaryOperator::getOverloadedOpcode(Op);
      ExprResult Result
        = SemaRef.CreateBuiltinBinOp(OpLoc, Opc, First, Second);
      if (Result.isInvalid())
        return ExprError();

      return move(Result);
    }
  }

  // Compute the transformed set of functions (and function templates) to be
  // used during overload resolution.
  UnresolvedSet<16> Functions;

  if (UnresolvedLookupExpr *ULE = dyn_cast<UnresolvedLookupExpr>(Callee)) {
    assert(ULE->requiresADL());

    // FIXME: Do we have to check
    // IsAcceptableNonMemberOperatorCandidate for each of these?
    Functions.append(ULE->decls_begin(), ULE->decls_end());
  } else {
    Functions.addDecl(cast<DeclRefExpr>(Callee)->getDecl());
  }

  // Add any functions found via argument-dependent lookup.
  Expr *Args[2] = { First, Second };
  unsigned NumArgs = 1 + (Second != 0);

  // Create the overloaded operator invocation for unary operators.
  if (NumArgs == 1 || isPostIncDec) {
    UnaryOperatorKind Opc
      = UnaryOperator::getOverloadedOpcode(Op, isPostIncDec);
    return SemaRef.CreateOverloadedUnaryOp(OpLoc, Opc, Functions, First);
  }

  if (Op == OO_Subscript)
    return SemaRef.CreateOverloadedArraySubscriptExpr(Callee->getLocStart(),
                                                      OpLoc,
                                                      First,
                                                      Second);

  // Create the overloaded operator invocation for binary operators.
  BinaryOperatorKind Opc = BinaryOperator::getOverloadedOpcode(Op);
  ExprResult Result
    = SemaRef.CreateOverloadedBinOp(OpLoc, Opc, Functions, Args[0], Args[1]);
  if (Result.isInvalid())
    return ExprError();

  return move(Result);
}

template<typename Derived>
ExprResult 
TreeTransform<Derived>::RebuildCXXPseudoDestructorExpr(Expr *Base,
                                                     SourceLocation OperatorLoc,
                                                       bool isArrow,
                                                 NestedNameSpecifier *Qualifier,
                                                     SourceRange QualifierRange,
                                                     TypeSourceInfo *ScopeType,
                                                       SourceLocation CCLoc,
                                                       SourceLocation TildeLoc,
                                        PseudoDestructorTypeStorage Destroyed) {
  CXXScopeSpec SS;
  if (Qualifier) {
    SS.setRange(QualifierRange);
    SS.setScopeRep(Qualifier);
  }

  QualType BaseType = Base->getType();
  if (Base->isTypeDependent() || Destroyed.getIdentifier() ||
      (!isArrow && !BaseType->getAs<RecordType>()) ||
      (isArrow && BaseType->getAs<PointerType>() && 
       !BaseType->getAs<PointerType>()->getPointeeType()
                                              ->template getAs<RecordType>())){
    // This pseudo-destructor expression is still a pseudo-destructor.
    return SemaRef.BuildPseudoDestructorExpr(Base, OperatorLoc,
                                             isArrow? tok::arrow : tok::period,
                                             SS, ScopeType, CCLoc, TildeLoc,
                                             Destroyed,
                                             /*FIXME?*/true);
  }

  TypeSourceInfo *DestroyedType = Destroyed.getTypeSourceInfo();
  DeclarationName Name(SemaRef.Context.DeclarationNames.getCXXDestructorName(
                 SemaRef.Context.getCanonicalType(DestroyedType->getType())));
  DeclarationNameInfo NameInfo(Name, Destroyed.getLocation());
  NameInfo.setNamedTypeInfo(DestroyedType);

  // FIXME: the ScopeType should be tacked onto SS.

  return getSema().BuildMemberReferenceExpr(Base, BaseType,
                                            OperatorLoc, isArrow,
                                            SS, /*FIXME: FirstQualifier*/ 0,
                                            NameInfo,
                                            /*TemplateArgs*/ 0);
}

} // end namespace clang

#endif // LLVM_CLANG_SEMA_TREETRANSFORM_H
