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
#include "Lookup.h"
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
  typedef Sema::DeclPtrTy DeclPtrTy;
  
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
  QualType TransformType(QualType T, QualType ObjectType = QualType());

  /// \brief Transforms the given type-with-location into a new
  /// type-with-location.
  ///
  /// By default, this routine transforms a type by delegating to the
  /// appropriate TransformXXXType to build a new type.  Subclasses
  /// may override this function (to take over all type
  /// transformations) or some set of the TransformXXXType functions
  /// to alter the transformation.
  TypeSourceInfo *TransformType(TypeSourceInfo *DI, 
                                QualType ObjectType = QualType());

  /// \brief Transform the given type-with-location into a new
  /// type, collecting location information in the given builder
  /// as necessary.
  ///
  QualType TransformType(TypeLocBuilder &TLB, TypeLoc TL, 
                         QualType ObjectType = QualType());

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
  OwningExprResult TransformExpr(Expr *E);

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

  /// \brief Fakes up a TypeSourceInfo for a type.
  TypeSourceInfo *InventTypeSourceInfo(QualType T) {
    return SemaRef.Context.getTrivialTypeSourceInfo(T,
                       getDerived().getBaseLocation());
  }

#define ABSTRACT_TYPELOC(CLASS, PARENT)
#define TYPELOC(CLASS, PARENT)                                   \
  QualType Transform##CLASS##Type(TypeLocBuilder &TLB, CLASS##TypeLoc T, \
                                  QualType ObjectType = QualType());
#include "clang/AST/TypeLocNodes.def"

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

  QualType TransformReferenceType(TypeLocBuilder &TLB, ReferenceTypeLoc TL, 
                                  QualType ObjectType);

  QualType 
  TransformTemplateSpecializationType(const TemplateSpecializationType *T,
                                      QualType ObjectType);

  OwningStmtResult TransformCompoundStmt(CompoundStmt *S, bool IsStmtExpr);
  OwningExprResult TransformCXXNamedCastExpr(CXXNamedCastExpr *E);

#define STMT(Node, Parent)                        \
  OwningStmtResult Transform##Node(Node *S);
#define EXPR(Node, Parent)                        \
  OwningExprResult Transform##Node(Node *E);
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
  QualType RebuildVectorType(QualType ElementType, unsigned NumElements,
    VectorType::AltiVecSpecific AltiVecSpec);

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
                                       const TemplateArgumentListInfo &Args);

  /// \brief Build a new qualified name type.
  ///
  /// By default, builds a new ElaboratedType type from the keyword,
  /// the nested-name-specifier and the named type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildElaboratedType(ElaboratedTypeKeyword Keyword,
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
                                    NestedNameSpecifier *NNS,
                                    const IdentifierInfo *Name,
                                    SourceLocation NameLoc,
                                    const TemplateArgumentListInfo &Args) {
    // Rebuild the template name.
    // TODO: avoid TemplateName abstraction
    TemplateName InstName =
      getDerived().RebuildTemplateName(NNS, *Name, QualType());
    
    if (InstName.isNull())
      return QualType();

    // If it's still dependent, make a dependent specialization.
    if (InstName.getAsDependentTemplateName())
      return SemaRef.Context.getDependentTemplateSpecializationType(
                                          Keyword, NNS, Name, Args);

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
                                   const IdentifierInfo &II,
                                   QualType ObjectType);

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
                                 VarDecl *CondVar, StmtArg Then, 
                                 SourceLocation ElseLoc, StmtArg Else) {
    return getSema().ActOnIfStmt(IfLoc, Cond, DeclPtrTy::make(CondVar), 
                                 move(Then), ElseLoc, move(Else));
  }

  /// \brief Start building a new switch statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildSwitchStmtStart(SourceLocation SwitchLoc,
                                          Sema::ExprArg Cond, 
                                          VarDecl *CondVar) {
    return getSema().ActOnStartOfSwitchStmt(SwitchLoc, move(Cond), 
                                            DeclPtrTy::make(CondVar));
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
                                    VarDecl *CondVar,
                                    StmtArg Body) {
    return getSema().ActOnWhileStmt(WhileLoc, Cond, 
                                    DeclPtrTy::make(CondVar), move(Body));
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
                                  StmtArg Init, Sema::FullExprArg Cond, 
                                  VarDecl *CondVar, Sema::FullExprArg Inc,
                                  SourceLocation RParenLoc, StmtArg Body) {
    return getSema().ActOnForStmt(ForLoc, LParenLoc, move(Init), Cond, 
                                  DeclPtrTy::make(CondVar),
                                  Inc, RParenLoc, move(Body));
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

  /// \brief Build a new inline asm statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildAsmStmt(SourceLocation AsmLoc,
                                  bool IsSimple,
                                  bool IsVolatile,
                                  unsigned NumOutputs,
                                  unsigned NumInputs,
                                  IdentifierInfo **Names,
                                  MultiExprArg Constraints,
                                  MultiExprArg Exprs,
                                  ExprArg AsmString,
                                  MultiExprArg Clobbers,
                                  SourceLocation RParenLoc,
                                  bool MSAsm) {
    return getSema().ActOnAsmStmt(AsmLoc, IsSimple, IsVolatile, NumOutputs, 
                                  NumInputs, Names, move(Constraints),
                                  move(Exprs), move(AsmString), move(Clobbers),
                                  RParenLoc, MSAsm);
  }

  /// \brief Build a new Objective-C @try statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildObjCAtTryStmt(SourceLocation AtLoc,
                                        StmtArg TryBody,
                                        MultiStmtArg CatchStmts,
                                        StmtArg Finally) {
    return getSema().ActOnObjCAtTryStmt(AtLoc, move(TryBody), move(CatchStmts),
                                        move(Finally));
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
  OwningStmtResult RebuildObjCAtCatchStmt(SourceLocation AtLoc,
                                          SourceLocation RParenLoc,
                                          VarDecl *Var,
                                          StmtArg Body) {
    return getSema().ActOnObjCAtCatchStmt(AtLoc, RParenLoc,
                                          Sema::DeclPtrTy::make(Var),
                                          move(Body));
  }
  
  /// \brief Build a new Objective-C @finally statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildObjCAtFinallyStmt(SourceLocation AtLoc,
                                            StmtArg Body) {
    return getSema().ActOnObjCAtFinallyStmt(AtLoc, move(Body));
  }
  
  /// \brief Build a new Objective-C @throw statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildObjCAtThrowStmt(SourceLocation AtLoc,
                                          ExprArg Operand) {
    return getSema().BuildObjCAtThrowStmt(AtLoc, move(Operand));
  }
  
  /// \brief Build a new Objective-C @synchronized statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildObjCAtSynchronizedStmt(SourceLocation AtLoc,
                                                 ExprArg Object,
                                                 StmtArg Body) {
    return getSema().ActOnObjCAtSynchronizedStmt(AtLoc, move(Object),
                                                 move(Body));
  }

  /// \brief Build a new Objective-C fast enumeration statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  OwningStmtResult RebuildObjCForCollectionStmt(SourceLocation ForLoc,
                                                SourceLocation LParenLoc,
                                                StmtArg Element,
                                                ExprArg Collection,
                                                SourceLocation RParenLoc,
                                                StmtArg Body) {
    return getSema().ActOnObjCForCollectionStmt(ForLoc, LParenLoc,
                                                move(Element), 
                                                move(Collection),
                                                RParenLoc,
                                                move(Body));
  }
  
  /// \brief Build a new C++ exception declaration.
  ///
  /// By default, performs semantic analysis to build the new decaration.
  /// Subclasses may override this routine to provide different behavior.
  VarDecl *RebuildExceptionDecl(VarDecl *ExceptionDecl, QualType T,
                                TypeSourceInfo *Declarator,
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
  OwningExprResult RebuildDeclarationNameExpr(const CXXScopeSpec &SS,
                                              LookupResult &R,
                                              bool RequiresADL) {
    return getSema().BuildDeclarationNameExpr(SS, R, RequiresADL);
  }


  /// \brief Build a new expression that references a declaration.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildDeclRefExpr(NestedNameSpecifier *Qualifier,
                                      SourceRange QualifierRange,
                                      ValueDecl *VD, SourceLocation Loc,
                                      TemplateArgumentListInfo *TemplateArgs) {
    CXXScopeSpec SS;
    SS.setScopeRep(Qualifier);
    SS.setRange(QualifierRange);

    // FIXME: loses template args.
    
    return getSema().BuildDeclarationNameExpr(SS, Loc, VD);
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
  OwningExprResult RebuildUnaryOperator(SourceLocation OpLoc,
                                        UnaryOperator::Opcode Opc,
                                        ExprArg SubExpr) {
    return getSema().BuildUnaryOp(/*Scope=*/0, OpLoc, Opc, move(SubExpr));
  }

  /// \brief Build a new builtin offsetof expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildOffsetOfExpr(SourceLocation OperatorLoc,
                                       TypeSourceInfo *Type,
                                       Action::OffsetOfComponent *Components,
                                       unsigned NumComponents,
                                       SourceLocation RParenLoc) {
    return getSema().BuildBuiltinOffsetOf(OperatorLoc, Type, Components,
                                          NumComponents, RParenLoc);
  }
  
  /// \brief Build a new sizeof or alignof expression with a type argument.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildSizeOfAlignOf(TypeSourceInfo *TInfo,
                                        SourceLocation OpLoc,
                                        bool isSizeOf, SourceRange R) {
    return getSema().CreateSizeOfAlignOfExpr(TInfo, OpLoc, isSizeOf, R);
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
                                     ValueDecl *Member,
                                     NamedDecl *FoundDecl,
                        const TemplateArgumentListInfo *ExplicitTemplateArgs,
                                     NamedDecl *FirstQualifierInScope) {
    if (!Member->getDeclName()) {
      // We have a reference to an unnamed field.
      assert(!Qualifier && "Can't have an unnamed field with a qualifier!");

      Expr *BaseExpr = Base.takeAs<Expr>();
      if (getSema().PerformObjectMemberConversion(BaseExpr, Qualifier,
                                                  FoundDecl, Member))
        return getSema().ExprError();

      MemberExpr *ME =
        new (getSema().Context) MemberExpr(BaseExpr, isArrow,
                                           Member, MemberLoc,
                                           cast<FieldDecl>(Member)->getType());
      return getSema().Owned(ME);
    }

    CXXScopeSpec SS;
    if (Qualifier) {
      SS.setRange(QualifierRange);
      SS.setScopeRep(Qualifier);
    }

    Expr *BaseExpr = Base.takeAs<Expr>();
    getSema().DefaultFunctionArrayConversion(BaseExpr);
    QualType BaseType = BaseExpr->getType();

    // FIXME: this involves duplicating earlier analysis in a lot of
    // cases; we should avoid this when possible.
    LookupResult R(getSema(), Member->getDeclName(), MemberLoc,
                   Sema::LookupMemberName);
    R.addDecl(FoundDecl);
    R.resolveKind();

    return getSema().BuildMemberReferenceExpr(getSema().Owned(BaseExpr),
                                              BaseType, OpLoc, isArrow,
                                              SS, FirstQualifierInScope,
                                              R, ExplicitTemplateArgs);
  }

  /// \brief Build a new binary operator expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildBinaryOperator(SourceLocation OpLoc,
                                         BinaryOperator::Opcode Opc,
                                         ExprArg LHS, ExprArg RHS) {
    return getSema().BuildBinOp(/*Scope=*/0, OpLoc, Opc, 
                                LHS.takeAs<Expr>(), RHS.takeAs<Expr>());
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

  /// \brief Build a new C-style cast expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCStyleCastExpr(SourceLocation LParenLoc,
                                         TypeSourceInfo *TInfo,
                                         SourceLocation RParenLoc,
                                         ExprArg SubExpr) {
    return getSema().BuildCStyleCastExpr(LParenLoc, TInfo, RParenLoc,
                                         move(SubExpr));
  }

  /// \brief Build a new compound literal expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCompoundLiteralExpr(SourceLocation LParenLoc,
                                              TypeSourceInfo *TInfo,
                                              SourceLocation RParenLoc,
                                              ExprArg Init) {
    return getSema().BuildCompoundLiteralExpr(LParenLoc, TInfo, RParenLoc,
                                              move(Init));
  }

  /// \brief Build a new extended vector element access expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildExtVectorElementExpr(ExprArg Base,
                                               SourceLocation OpLoc,
                                               SourceLocation AccessorLoc,
                                               IdentifierInfo &Accessor) {

    CXXScopeSpec SS;
    QualType BaseType = ((Expr*) Base.get())->getType();
    return getSema().BuildMemberReferenceExpr(move(Base), BaseType,
                                              OpLoc, /*IsArrow*/ false,
                                              SS, /*FirstQualifierInScope*/ 0,
                                              DeclarationName(&Accessor),
                                              AccessorLoc,
                                              /* TemplateArgs */ 0);
  }

  /// \brief Build a new initializer list expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildInitList(SourceLocation LBraceLoc,
                                   MultiExprArg Inits,
                                   SourceLocation RBraceLoc,
                                   QualType ResultTy) {
    OwningExprResult Result
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
  OwningExprResult RebuildVAArgExpr(SourceLocation BuiltinLoc,
                                    ExprArg SubExpr, TypeSourceInfo *TInfo,
                                    SourceLocation RParenLoc) {
    return getSema().BuildVAArgExpr(BuiltinLoc,
                                    move(SubExpr), TInfo,
                                    RParenLoc);
  }

  /// \brief Build a new expression list in parentheses.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildParenListExpr(SourceLocation LParenLoc,
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
                                              TypeSourceInfo *TInfo1,
                                              TypeSourceInfo *TInfo2,
                                              SourceLocation RParenLoc) {
    return getSema().BuildTypesCompatibleExpr(BuiltinLoc,
                                              TInfo1, TInfo2,
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
                                           TypeSourceInfo *TInfo,
                                           SourceLocation RAngleLoc,
                                           SourceLocation LParenLoc,
                                           ExprArg SubExpr,
                                           SourceLocation RParenLoc) {
    switch (Class) {
    case Stmt::CXXStaticCastExprClass:
      return getDerived().RebuildCXXStaticCastExpr(OpLoc, LAngleLoc, TInfo,
                                                   RAngleLoc, LParenLoc,
                                                   move(SubExpr), RParenLoc);

    case Stmt::CXXDynamicCastExprClass:
      return getDerived().RebuildCXXDynamicCastExpr(OpLoc, LAngleLoc, TInfo,
                                                    RAngleLoc, LParenLoc,
                                                    move(SubExpr), RParenLoc);

    case Stmt::CXXReinterpretCastExprClass:
      return getDerived().RebuildCXXReinterpretCastExpr(OpLoc, LAngleLoc, TInfo,
                                                        RAngleLoc, LParenLoc,
                                                        move(SubExpr),
                                                        RParenLoc);

    case Stmt::CXXConstCastExprClass:
      return getDerived().RebuildCXXConstCastExpr(OpLoc, LAngleLoc, TInfo,
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
                                            TypeSourceInfo *TInfo,
                                            SourceLocation RAngleLoc,
                                            SourceLocation LParenLoc,
                                            ExprArg SubExpr,
                                            SourceLocation RParenLoc) {
    return getSema().BuildCXXNamedCast(OpLoc, tok::kw_static_cast,
                                       TInfo, move(SubExpr),
                                       SourceRange(LAngleLoc, RAngleLoc),
                                       SourceRange(LParenLoc, RParenLoc));
  }

  /// \brief Build a new C++ dynamic_cast expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXDynamicCastExpr(SourceLocation OpLoc,
                                             SourceLocation LAngleLoc,
                                             TypeSourceInfo *TInfo,
                                             SourceLocation RAngleLoc,
                                             SourceLocation LParenLoc,
                                             ExprArg SubExpr,
                                             SourceLocation RParenLoc) {
    return getSema().BuildCXXNamedCast(OpLoc, tok::kw_dynamic_cast,
                                       TInfo, move(SubExpr),
                                       SourceRange(LAngleLoc, RAngleLoc),
                                       SourceRange(LParenLoc, RParenLoc));
  }

  /// \brief Build a new C++ reinterpret_cast expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXReinterpretCastExpr(SourceLocation OpLoc,
                                                 SourceLocation LAngleLoc,
                                                 TypeSourceInfo *TInfo,
                                                 SourceLocation RAngleLoc,
                                                 SourceLocation LParenLoc,
                                                 ExprArg SubExpr,
                                                 SourceLocation RParenLoc) {
    return getSema().BuildCXXNamedCast(OpLoc, tok::kw_reinterpret_cast,
                                       TInfo, move(SubExpr),
                                       SourceRange(LAngleLoc, RAngleLoc),
                                       SourceRange(LParenLoc, RParenLoc));
  }

  /// \brief Build a new C++ const_cast expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXConstCastExpr(SourceLocation OpLoc,
                                           SourceLocation LAngleLoc,
                                           TypeSourceInfo *TInfo,
                                           SourceLocation RAngleLoc,
                                           SourceLocation LParenLoc,
                                           ExprArg SubExpr,
                                           SourceLocation RParenLoc) {
    return getSema().BuildCXXNamedCast(OpLoc, tok::kw_const_cast,
                                       TInfo, move(SubExpr),
                                       SourceRange(LAngleLoc, RAngleLoc),
                                       SourceRange(LParenLoc, RParenLoc));
  }

  /// \brief Build a new C++ functional-style cast expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXFunctionalCastExpr(SourceRange TypeRange,
                                                TypeSourceInfo *TInfo,
                                                SourceLocation LParenLoc,
                                                ExprArg SubExpr,
                                                SourceLocation RParenLoc) {
    void *Sub = SubExpr.takeAs<Expr>();
    return getSema().ActOnCXXTypeConstructExpr(TypeRange,
                                               TInfo->getType().getAsOpaquePtr(),
                                               LParenLoc,
                                         Sema::MultiExprArg(getSema(), &Sub, 1),
                                               /*CommaLocs=*/0,
                                               RParenLoc);
  }

  /// \brief Build a new C++ typeid(type) expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXTypeidExpr(QualType TypeInfoType,
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
  OwningExprResult RebuildCXXTypeidExpr(QualType TypeInfoType,
                                        SourceLocation TypeidLoc,
                                        ExprArg Operand,
                                        SourceLocation RParenLoc) {
    return getSema().BuildCXXTypeId(TypeInfoType, TypeidLoc, move(Operand),
                                    RParenLoc);
  }

  /// \brief Build a new C++ "this" expression.
  ///
  /// By default, builds a new "this" expression without performing any
  /// semantic analysis. Subclasses may override this routine to provide
  /// different behavior.
  OwningExprResult RebuildCXXThisExpr(SourceLocation ThisLoc,
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
  OwningExprResult RebuildCXXThrowExpr(SourceLocation ThrowLoc, ExprArg Sub) {
    return getSema().ActOnCXXThrow(ThrowLoc, move(Sub));
  }

  /// \brief Build a new C++ default-argument expression.
  ///
  /// By default, builds a new default-argument expression, which does not
  /// require any semantic analysis. Subclasses may override this routine to
  /// provide different behavior.
  OwningExprResult RebuildCXXDefaultArgExpr(SourceLocation Loc, 
                                            ParmVarDecl *Param) {
    return getSema().Owned(CXXDefaultArgExpr::Create(getSema().Context, Loc,
                                                     Param));
  }

  /// \brief Build a new C++ zero-initialization expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXScalarValueInitExpr(SourceLocation TypeStartLoc,
                                               SourceLocation LParenLoc,
                                               QualType T,
                                               SourceLocation RParenLoc) {
    return getSema().ActOnCXXTypeConstructExpr(SourceRange(TypeStartLoc),
                                               T.getAsOpaquePtr(), LParenLoc,
                                               MultiExprArg(getSema(), 0, 0),
                                               0, RParenLoc);
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
                                     SourceRange TypeIdParens,
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
                                 TypeIdParens,
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
  OwningExprResult RebuildDependentScopeDeclRefExpr(NestedNameSpecifier *NNS,
                                                SourceRange QualifierRange,
                                                DeclarationName Name,
                                                SourceLocation Location,
                              const TemplateArgumentListInfo *TemplateArgs) {
    CXXScopeSpec SS;
    SS.setRange(QualifierRange);
    SS.setScopeRep(NNS);

    if (TemplateArgs)
      return getSema().BuildQualifiedTemplateIdExpr(SS, Name, Location,
                                                    *TemplateArgs);

    return getSema().BuildQualifiedDeclarationNameExpr(SS, Name, Location);
  }

  /// \brief Build a new template-id expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildTemplateIdExpr(const CXXScopeSpec &SS,
                                         LookupResult &R,
                                         bool RequiresADL,
                              const TemplateArgumentListInfo &TemplateArgs) {
    return getSema().BuildTemplateIdExpr(SS, R, RequiresADL, TemplateArgs);
  }

  /// \brief Build a new object-construction expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildCXXConstructExpr(QualType T,
                                           SourceLocation Loc,
                                           CXXConstructorDecl *Constructor,
                                           bool IsElidable,
                                           MultiExprArg Args) {
    ASTOwningVector<&ActionBase::DeleteExpr> ConvertedArgs(SemaRef);
    if (getSema().CompleteConstructorCall(Constructor, move(Args), Loc, 
                                          ConvertedArgs))
      return getSema().ExprError();
    
    return getSema().BuildCXXConstructExpr(Loc, T, Constructor, IsElidable,
                                           move_arg(ConvertedArgs));
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
  OwningExprResult RebuildCXXDependentScopeMemberExpr(ExprArg BaseE,
                                                  QualType BaseType,
                                                  bool IsArrow,
                                                  SourceLocation OperatorLoc,
                                              NestedNameSpecifier *Qualifier,
                                                  SourceRange QualifierRange,
                                            NamedDecl *FirstQualifierInScope,
                                                  DeclarationName Name,
                                                  SourceLocation MemberLoc,
                              const TemplateArgumentListInfo *TemplateArgs) {
    CXXScopeSpec SS;
    SS.setRange(QualifierRange);
    SS.setScopeRep(Qualifier);

    return SemaRef.BuildMemberReferenceExpr(move(BaseE), BaseType,
                                            OperatorLoc, IsArrow,
                                            SS, FirstQualifierInScope,
                                            Name, MemberLoc, TemplateArgs);
  }

  /// \brief Build a new member reference expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildUnresolvedMemberExpr(ExprArg BaseE,
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

    return SemaRef.BuildMemberReferenceExpr(move(BaseE), BaseType,
                                            OperatorLoc, IsArrow,
                                            SS, FirstQualifierInScope,
                                            R, TemplateArgs);
  }

  /// \brief Build a new Objective-C @encode expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildObjCEncodeExpr(SourceLocation AtLoc,
                                         TypeSourceInfo *EncodeTypeInfo,
                                         SourceLocation RParenLoc) {
    return SemaRef.Owned(SemaRef.BuildObjCEncodeExpression(AtLoc, EncodeTypeInfo,
                                                           RParenLoc));
  }

  /// \brief Build a new Objective-C class message.
  OwningExprResult RebuildObjCMessageExpr(TypeSourceInfo *ReceiverTypeInfo,
                                          Selector Sel,
                                          ObjCMethodDecl *Method,
                                          SourceLocation LBracLoc, 
                                          MultiExprArg Args,
                                          SourceLocation RBracLoc) {
    return SemaRef.BuildClassMessage(ReceiverTypeInfo,
                                     ReceiverTypeInfo->getType(),
                                     /*SuperLoc=*/SourceLocation(),
                                     Sel, Method, LBracLoc, RBracLoc,
                                     move(Args));
  }

  /// \brief Build a new Objective-C instance message.
  OwningExprResult RebuildObjCMessageExpr(ExprArg Receiver,
                                          Selector Sel,
                                          ObjCMethodDecl *Method,
                                          SourceLocation LBracLoc, 
                                          MultiExprArg Args,
                                          SourceLocation RBracLoc) {
    QualType ReceiverType = static_cast<Expr *>(Receiver.get())->getType();
    return SemaRef.BuildInstanceMessage(move(Receiver),
                                        ReceiverType,
                                        /*SuperLoc=*/SourceLocation(),
                                        Sel, Method, LBracLoc, RBracLoc,
                                        move(Args));
  }

  /// \brief Build a new Objective-C ivar reference expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildObjCIvarRefExpr(ExprArg BaseArg, ObjCIvarDecl *Ivar,
                                          SourceLocation IvarLoc,
                                          bool IsArrow, bool IsFreeIvar) {
    // FIXME: We lose track of the IsFreeIvar bit.
    CXXScopeSpec SS;
    Expr *Base = BaseArg.takeAs<Expr>();
    LookupResult R(getSema(), Ivar->getDeclName(), IvarLoc,
                   Sema::LookupMemberName);
    OwningExprResult Result = getSema().LookupMemberExpr(R, Base, IsArrow,
                                                         /*FIME:*/IvarLoc,
                                                         SS, DeclPtrTy(),
                                                         false);
    if (Result.isInvalid())
      return getSema().ExprError();
    
    if (Result.get())
      return move(Result);
    
    return getSema().BuildMemberReferenceExpr(getSema().Owned(Base), 
                                              Base->getType(),
                                              /*FIXME:*/IvarLoc, IsArrow, SS, 
                                              /*FirstQualifierInScope=*/0,
                                              R, 
                                              /*TemplateArgs=*/0);
  }

  /// \brief Build a new Objective-C property reference expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildObjCPropertyRefExpr(ExprArg BaseArg, 
                                              ObjCPropertyDecl *Property,
                                              SourceLocation PropertyLoc) {
    CXXScopeSpec SS;
    Expr *Base = BaseArg.takeAs<Expr>();
    LookupResult R(getSema(), Property->getDeclName(), PropertyLoc,
                   Sema::LookupMemberName);
    bool IsArrow = false;
    OwningExprResult Result = getSema().LookupMemberExpr(R, Base, IsArrow,
                                                         /*FIME:*/PropertyLoc,
                                                         SS, DeclPtrTy(),
                                                         false);
    if (Result.isInvalid())
      return getSema().ExprError();
    
    if (Result.get())
      return move(Result);
    
    return getSema().BuildMemberReferenceExpr(getSema().Owned(Base), 
                                              Base->getType(),
                                              /*FIXME:*/PropertyLoc, IsArrow, 
                                              SS, 
                                              /*FirstQualifierInScope=*/0,
                                              R, 
                                              /*TemplateArgs=*/0);
  }
  
  /// \brief Build a new Objective-C implicit setter/getter reference 
  /// expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.  
  OwningExprResult RebuildObjCImplicitSetterGetterRefExpr(
                                                        ObjCMethodDecl *Getter,
                                                          QualType T,
                                                        ObjCMethodDecl *Setter,
                                                        SourceLocation NameLoc,
                                                          ExprArg Base) {
    // Since these expressions can only be value-dependent, we do not need to
    // perform semantic analysis again.
    return getSema().Owned(
             new (getSema().Context) ObjCImplicitSetterGetterRefExpr(Getter, T,
                                                                     Setter,
                                                                     NameLoc,
                                                          Base.takeAs<Expr>()));
  }

  /// \brief Build a new Objective-C "isa" expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  OwningExprResult RebuildObjCIsaExpr(ExprArg BaseArg, SourceLocation IsaLoc,
                                      bool IsArrow) {
    CXXScopeSpec SS;
    Expr *Base = BaseArg.takeAs<Expr>();
    LookupResult R(getSema(), &getSema().Context.Idents.get("isa"), IsaLoc,
                   Sema::LookupMemberName);
    OwningExprResult Result = getSema().LookupMemberExpr(R, Base, IsArrow,
                                                         /*FIME:*/IsaLoc,
                                                         SS, DeclPtrTy(),
                                                         false);
    if (Result.isInvalid())
      return getSema().ExprError();
    
    if (Result.get())
      return move(Result);
    
    return getSema().BuildMemberReferenceExpr(getSema().Owned(Base), 
                                              Base->getType(),
                                              /*FIXME:*/IsaLoc, IsArrow, SS, 
                                              /*FirstQualifierInScope=*/0,
                                              R, 
                                              /*TemplateArgs=*/0);
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
                                          BuiltinLoc);
    SemaRef.UsualUnaryConversions(Callee);

    // Build the CallExpr
    unsigned NumSubExprs = SubExprs.size();
    Expr **Subs = (Expr **)SubExprs.release();
    CallExpr *TheCall = new (SemaRef.Context) CallExpr(SemaRef.Context, Callee,
                                                       Subs, NumSubExprs,
                                                   Builtin->getCallResultType(),
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
#include "clang/AST/StmtNodes.inc"

  // Transform expressions by calling TransformExpr.
#define STMT(Node, Parent)
#define ABSTRACT_STMT(Stmt)
#define EXPR(Node, Parent) case Stmt::Node##Class:
#include "clang/AST/StmtNodes.inc"
    {
      Sema::OwningExprResult E = getDerived().TransformExpr(cast<Expr>(S));
      if (E.isInvalid())
        return getSema().StmtError();

      return getSema().ActOnExprStmt(getSema().MakeFullExpr(E));
    }
  }

  return SemaRef.Owned(S->Retain());
}


template<typename Derived>
Sema::OwningExprResult TreeTransform<Derived>::TransformExpr(Expr *E) {
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
    QualType T = getDerived().TransformType(QualType(NNS->getAsType(), 0),
                                            ObjectType);
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
  case DeclarationName::CXXLiteralOperatorName:
  case DeclarationName::CXXUsingDirective:
    return Name;

  case DeclarationName::CXXConstructorName:
  case DeclarationName::CXXDestructorName:
  case DeclarationName::CXXConversionFunctionName: {
    TemporaryBase Rebase(*this, Loc, Name);
    QualType T = getDerived().TransformType(Name.getCXXNameType(), 
                                            ObjectType);
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
  SourceLocation Loc = getDerived().getBaseLocation();

  if (QualifiedTemplateName *QTN = Name.getAsQualifiedTemplateName()) {
    NestedNameSpecifier *NNS
      = getDerived().TransformNestedNameSpecifier(QTN->getQualifier(),
                        /*FIXME:*/SourceRange(getDerived().getBaseLocation()),
                                                  ObjectType);
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
    assert(false && "overloaded template name survived to here");
  }

  if (DependentTemplateName *DTN = Name.getAsDependentTemplateName()) {
    NestedNameSpecifier *NNS
      = getDerived().TransformNestedNameSpecifier(DTN->getQualifier(),
                        /*FIXME:*/SourceRange(getDerived().getBaseLocation()),
                                                  ObjectType);
    if (!NNS && DTN->getQualifier())
      return TemplateName();

    if (!getDerived().AlwaysRebuild() &&
        NNS == DTN->getQualifier() &&
        ObjectType.isNull())
      return Name;

    if (DTN->isIdentifier())
      return getDerived().RebuildTemplateName(NNS, *DTN->getIdentifier(), 
                                              ObjectType);
    
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
  assert(false && "overloaded function decl survived to here");
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
QualType TreeTransform<Derived>::TransformType(QualType T, 
                                               QualType ObjectType) {
  if (getDerived().AlreadyTransformed(T))
    return T;

  // Temporary workaround.  All of these transformations should
  // eventually turn into transformations on TypeLocs.
  TypeSourceInfo *DI = getSema().Context.CreateTypeSourceInfo(T);
  DI->getTypeLoc().initialize(getDerived().getBaseLocation());
  
  TypeSourceInfo *NewDI = getDerived().TransformType(DI, ObjectType);

  if (!NewDI)
    return QualType();

  return NewDI->getType();
}

template<typename Derived>
TypeSourceInfo *TreeTransform<Derived>::TransformType(TypeSourceInfo *DI,
                                                      QualType ObjectType) {
  if (getDerived().AlreadyTransformed(DI->getType()))
    return DI;

  TypeLocBuilder TLB;

  TypeLoc TL = DI->getTypeLoc();
  TLB.reserve(TL.getFullDataSize());

  QualType Result = getDerived().TransformType(TLB, TL, ObjectType);
  if (Result.isNull())
    return 0;

  return TLB.getTypeSourceInfo(SemaRef.Context, Result);
}

template<typename Derived>
QualType
TreeTransform<Derived>::TransformType(TypeLocBuilder &TLB, TypeLoc T,
                                      QualType ObjectType) {
  switch (T.getTypeLocClass()) {
#define ABSTRACT_TYPELOC(CLASS, PARENT)
#define TYPELOC(CLASS, PARENT) \
  case TypeLoc::CLASS: \
    return getDerived().Transform##CLASS##Type(TLB, cast<CLASS##TypeLoc>(T), \
                                               ObjectType);
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
                                               QualifiedTypeLoc T,
                                               QualType ObjectType) {
  Qualifiers Quals = T.getType().getLocalQualifiers();

  QualType Result = getDerived().TransformType(TLB, T.getUnqualifiedLoc(),
                                               ObjectType);
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

template <class TyLoc> static inline
QualType TransformTypeSpecType(TypeLocBuilder &TLB, TyLoc T) {
  TyLoc NewT = TLB.push<TyLoc>(T.getType());
  NewT.setNameLoc(T.getNameLoc());
  return T.getType();
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformBuiltinType(TypeLocBuilder &TLB,
                                                      BuiltinTypeLoc T,
                                                      QualType ObjectType) {
  BuiltinTypeLoc NewT = TLB.push<BuiltinTypeLoc>(T.getType());
  NewT.setBuiltinLoc(T.getBuiltinLoc());
  if (T.needsExtraLocalData())
    NewT.getWrittenBuiltinSpecs() = T.getWrittenBuiltinSpecs();
  return T.getType();
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformComplexType(TypeLocBuilder &TLB,
                                                      ComplexTypeLoc T,
                                                      QualType ObjectType) {
  // FIXME: recurse?
  return TransformTypeSpecType(TLB, T);
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformPointerType(TypeLocBuilder &TLB,
                                                      PointerTypeLoc TL, 
                                                      QualType ObjectType) {
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
                                                  BlockPointerTypeLoc TL,
                                                  QualType ObjectType) {
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
                                               ReferenceTypeLoc TL,
                                               QualType ObjectType) {
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
                                                 LValueReferenceTypeLoc TL,
                                                     QualType ObjectType) {
  return TransformReferenceType(TLB, TL, ObjectType);
}

template<typename Derived>
QualType
TreeTransform<Derived>::TransformRValueReferenceType(TypeLocBuilder &TLB,
                                                 RValueReferenceTypeLoc TL,
                                                     QualType ObjectType) {
  return TransformReferenceType(TLB, TL, ObjectType);
}

template<typename Derived>
QualType
TreeTransform<Derived>::TransformMemberPointerType(TypeLocBuilder &TLB,
                                                   MemberPointerTypeLoc TL,
                                                   QualType ObjectType) {
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
                                                   ConstantArrayTypeLoc TL,
                                                   QualType ObjectType) {
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
                                              IncompleteArrayTypeLoc TL,
                                              QualType ObjectType) {
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
                                                   VariableArrayTypeLoc TL,
                                                   QualType ObjectType) {
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
                                             DependentSizedArrayTypeLoc TL,
                                                        QualType ObjectType) {
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
                                      DependentSizedExtVectorTypeLoc TL,
                                      QualType ObjectType) {
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
                                                     VectorTypeLoc TL,
                                                     QualType ObjectType) {
  VectorType *T = TL.getTypePtr();
  QualType ElementType = getDerived().TransformType(T->getElementType());
  if (ElementType.isNull())
    return QualType();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      ElementType != T->getElementType()) {
    Result = getDerived().RebuildVectorType(ElementType, T->getNumElements(),
      T->getAltiVecSpecific());
    if (Result.isNull())
      return QualType();
  }
  
  VectorTypeLoc NewTL = TLB.push<VectorTypeLoc>(Result);
  NewTL.setNameLoc(TL.getNameLoc());

  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformExtVectorType(TypeLocBuilder &TLB,
                                                        ExtVectorTypeLoc TL,
                                                        QualType ObjectType) {
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
                                                   FunctionProtoTypeLoc TL,
                                                   QualType ObjectType) {
  // Transform the parameters. We do this first for the benefit of template
  // instantiations, so that the ParmVarDecls get/ placed into the template
  // instantiation scope before we transform the function type.
  llvm::SmallVector<QualType, 4> ParamTypes;
  llvm::SmallVector<ParmVarDecl*, 4> ParamDecls;
  if (getDerived().TransformFunctionTypeParams(TL, ParamTypes, ParamDecls))
    return QualType();
  
  FunctionProtoType *T = TL.getTypePtr();
  QualType ResultType = getDerived().TransformType(TLB, TL.getResultLoc());
  if (ResultType.isNull())
    return QualType();
  
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
  for (unsigned i = 0, e = NewTL.getNumArgs(); i != e; ++i)
    NewTL.setArg(i, ParamDecls[i]);

  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformFunctionNoProtoType(
                                                 TypeLocBuilder &TLB,
                                                 FunctionNoProtoTypeLoc TL,
                                                 QualType ObjectType) {
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

template<typename Derived> QualType
TreeTransform<Derived>::TransformUnresolvedUsingType(TypeLocBuilder &TLB,
                                                 UnresolvedUsingTypeLoc TL,
                                                     QualType ObjectType) {
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
                                                      TypedefTypeLoc TL,
                                                      QualType ObjectType) {
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
                                                      TypeOfExprTypeLoc TL,
                                                       QualType ObjectType) {
  // typeof expressions are not potentially evaluated contexts
  EnterExpressionEvaluationContext Unevaluated(SemaRef, Action::Unevaluated);

  Sema::OwningExprResult E = getDerived().TransformExpr(TL.getUnderlyingExpr());
  if (E.isInvalid())
    return QualType();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      E.get() != TL.getUnderlyingExpr()) {
    Result = getDerived().RebuildTypeOfExprType(move(E));
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
                                                     TypeOfTypeLoc TL,
                                                     QualType ObjectType) {
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
                                                       DecltypeTypeLoc TL,
                                                       QualType ObjectType) {
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
                                                     RecordTypeLoc TL,
                                                     QualType ObjectType) {
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
                                                   EnumTypeLoc TL,
                                                   QualType ObjectType) {
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
                                         InjectedClassNameTypeLoc TL,
                                         QualType ObjectType) {
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
                                                TemplateTypeParmTypeLoc TL,
                                                QualType ObjectType) {
  return TransformTypeSpecType(TLB, TL);
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformSubstTemplateTypeParmType(
                                         TypeLocBuilder &TLB,
                                         SubstTemplateTypeParmTypeLoc TL,
                                         QualType ObjectType) {
  return TransformTypeSpecType(TLB, TL);
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
                                                ElaboratedTypeLoc TL,
                                                QualType ObjectType) {
  ElaboratedType *T = TL.getTypePtr();

  NestedNameSpecifier *NNS = 0;
  // NOTE: the qualifier in an ElaboratedType is optional.
  if (T->getQualifier() != 0) {
    NNS = getDerived().TransformNestedNameSpecifier(T->getQualifier(),
                                                    TL.getQualifierRange(),
                                                    ObjectType);
    if (!NNS)
      return QualType();
  }

  QualType NamedT;
  // FIXME: this test is meant to workaround a problem (failing assertion)
  // occurring if directly executing the code in the else branch.
  if (isa<TemplateSpecializationTypeLoc>(TL.getNamedTypeLoc())) {
    TemplateSpecializationTypeLoc OldNamedTL
      = cast<TemplateSpecializationTypeLoc>(TL.getNamedTypeLoc());
    const TemplateSpecializationType* OldTST
      = OldNamedTL.getType()->template getAs<TemplateSpecializationType>();
    NamedT = TransformTemplateSpecializationType(OldTST, ObjectType);
    if (NamedT.isNull())
      return QualType();
    TemplateSpecializationTypeLoc NewNamedTL
      = TLB.push<TemplateSpecializationTypeLoc>(NamedT);
    NewNamedTL.copy(OldNamedTL);
  }
  else {
    NamedT = getDerived().TransformType(TLB, TL.getNamedTypeLoc());
    if (NamedT.isNull())
      return QualType();
  }

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      NNS != T->getQualifier() ||
      NamedT != T->getNamedType()) {
    Result = getDerived().RebuildElaboratedType(T->getKeyword(), NNS, NamedT);
    if (Result.isNull())
      return QualType();
  }

  ElaboratedTypeLoc NewTL = TLB.push<ElaboratedTypeLoc>(Result);
  NewTL.setKeywordLoc(TL.getKeywordLoc());
  NewTL.setQualifierRange(TL.getQualifierRange());

  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformDependentNameType(TypeLocBuilder &TLB,
                                                       DependentNameTypeLoc TL,
                                                       QualType ObjectType) {
  DependentNameType *T = TL.getTypePtr();

  NestedNameSpecifier *NNS
    = getDerived().TransformNestedNameSpecifier(T->getQualifier(),
                                                TL.getQualifierRange(),
                                                ObjectType);
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
                                 DependentTemplateSpecializationTypeLoc TL,
                                                       QualType ObjectType) {
  DependentTemplateSpecializationType *T = TL.getTypePtr();

  NestedNameSpecifier *NNS
    = getDerived().TransformNestedNameSpecifier(T->getQualifier(),
                                                TL.getQualifierRange(),
                                                ObjectType);
  if (!NNS)
    return QualType();

  TemplateArgumentListInfo NewTemplateArgs;
  NewTemplateArgs.setLAngleLoc(TL.getLAngleLoc());
  NewTemplateArgs.setRAngleLoc(TL.getRAngleLoc());

  for (unsigned I = 0, E = T->getNumArgs(); I != E; ++I) {
    TemplateArgumentLoc Loc;
    if (getDerived().TransformTemplateArgument(TL.getArgLoc(I), Loc))
      return QualType();
    NewTemplateArgs.addArgument(Loc);
  }

  QualType Result = getDerived().RebuildDependentTemplateSpecializationType(
                                                     T->getKeyword(),
                                                     NNS,
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
QualType
TreeTransform<Derived>::TransformObjCInterfaceType(TypeLocBuilder &TLB,
                                                   ObjCInterfaceTypeLoc TL,
                                                   QualType ObjectType) {
  // ObjCInterfaceType is never dependent.
  TLB.pushFullCopy(TL);
  return TL.getType();
}

template<typename Derived>
QualType
TreeTransform<Derived>::TransformObjCObjectType(TypeLocBuilder &TLB,
                                                ObjCObjectTypeLoc TL,
                                                QualType ObjectType) {
  // ObjCObjectType is never dependent.
  TLB.pushFullCopy(TL);
  return TL.getType();
}

template<typename Derived>
QualType
TreeTransform<Derived>::TransformObjCObjectPointerType(TypeLocBuilder &TLB,
                                               ObjCObjectPointerTypeLoc TL,
                                                       QualType ObjectType) {
  // ObjCObjectPointerType is never dependent.
  TLB.pushFullCopy(TL);
  return TL.getType();
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
  OwningExprResult LHS(SemaRef), RHS(SemaRef);
  {
    // The case value expressions are not potentially evaluated.
    EnterExpressionEvaluationContext Unevaluated(SemaRef, Action::Unevaluated);

    // Transform the left-hand case value.
    LHS = getDerived().TransformExpr(S->getLHS());
    if (LHS.isInvalid())
      return SemaRef.StmtError();

    // Transform the right-hand case value (for the GNU case-range extension).
    RHS = getDerived().TransformExpr(S->getRHS());
    if (RHS.isInvalid())
      return SemaRef.StmtError();
  }

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
  OwningExprResult Cond(SemaRef);
  VarDecl *ConditionVar = 0;
  if (S->getConditionVariable()) {
    ConditionVar 
      = cast_or_null<VarDecl>(
                   getDerived().TransformDefinition(
                                      S->getConditionVariable()->getLocation(),
                                                    S->getConditionVariable()));
    if (!ConditionVar)
      return SemaRef.StmtError();
  } else {
    Cond = getDerived().TransformExpr(S->getCond());
  
    if (Cond.isInvalid())
      return SemaRef.StmtError();
    
    // Convert the condition to a boolean value.
    if (S->getCond()) {
      OwningExprResult CondE = getSema().ActOnBooleanCondition(0, 
                                                               S->getIfLoc(), 
                                                               move(Cond));
      if (CondE.isInvalid())
        return getSema().StmtError();
    
      Cond = move(CondE);
    }
  }
  
  Sema::FullExprArg FullCond(getSema().MakeFullExpr(Cond));
  if (!S->getConditionVariable() && S->getCond() && !FullCond->get())
    return SemaRef.StmtError();
  
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
      ConditionVar == S->getConditionVariable() &&
      Then.get() == S->getThen() &&
      Else.get() == S->getElse())
    return SemaRef.Owned(S->Retain());

  return getDerived().RebuildIfStmt(S->getIfLoc(), FullCond, ConditionVar,
                                    move(Then),
                                    S->getElseLoc(), move(Else));
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformSwitchStmt(SwitchStmt *S) {
  // Transform the condition.
  OwningExprResult Cond(SemaRef);
  VarDecl *ConditionVar = 0;
  if (S->getConditionVariable()) {
    ConditionVar 
      = cast_or_null<VarDecl>(
                   getDerived().TransformDefinition(
                                      S->getConditionVariable()->getLocation(),
                                                    S->getConditionVariable()));
    if (!ConditionVar)
      return SemaRef.StmtError();
  } else {
    Cond = getDerived().TransformExpr(S->getCond());
    
    if (Cond.isInvalid())
      return SemaRef.StmtError();
  }

  // Rebuild the switch statement.
  OwningStmtResult Switch
    = getDerived().RebuildSwitchStmtStart(S->getSwitchLoc(), move(Cond),
                                          ConditionVar);
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
  OwningExprResult Cond(SemaRef);
  VarDecl *ConditionVar = 0;
  if (S->getConditionVariable()) {
    ConditionVar 
      = cast_or_null<VarDecl>(
                   getDerived().TransformDefinition(
                                      S->getConditionVariable()->getLocation(),
                                                    S->getConditionVariable()));
    if (!ConditionVar)
      return SemaRef.StmtError();
  } else {
    Cond = getDerived().TransformExpr(S->getCond());
    
    if (Cond.isInvalid())
      return SemaRef.StmtError();

    if (S->getCond()) {
      // Convert the condition to a boolean value.
      OwningExprResult CondE = getSema().ActOnBooleanCondition(0, 
                                                             S->getWhileLoc(), 
                                                               move(Cond));
      if (CondE.isInvalid())
        return getSema().StmtError();
      Cond = move(CondE);
    }
  }

  Sema::FullExprArg FullCond(getSema().MakeFullExpr(Cond));
  if (!S->getConditionVariable() && S->getCond() && !FullCond->get())
    return SemaRef.StmtError();

  // Transform the body
  OwningStmtResult Body = getDerived().TransformStmt(S->getBody());
  if (Body.isInvalid())
    return SemaRef.StmtError();

  if (!getDerived().AlwaysRebuild() &&
      FullCond->get() == S->getCond() &&
      ConditionVar == S->getConditionVariable() &&
      Body.get() == S->getBody())
    return SemaRef.Owned(S->Retain());

  return getDerived().RebuildWhileStmt(S->getWhileLoc(), FullCond,
                                       ConditionVar, move(Body));
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformDoStmt(DoStmt *S) {
  // Transform the body
  OwningStmtResult Body = getDerived().TransformStmt(S->getBody());
  if (Body.isInvalid())
    return SemaRef.StmtError();

  // Transform the condition
  OwningExprResult Cond = getDerived().TransformExpr(S->getCond());
  if (Cond.isInvalid())
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
  OwningExprResult Cond(SemaRef);
  VarDecl *ConditionVar = 0;
  if (S->getConditionVariable()) {
    ConditionVar 
      = cast_or_null<VarDecl>(
                   getDerived().TransformDefinition(
                                      S->getConditionVariable()->getLocation(),
                                                    S->getConditionVariable()));
    if (!ConditionVar)
      return SemaRef.StmtError();
  } else {
    Cond = getDerived().TransformExpr(S->getCond());
    
    if (Cond.isInvalid())
      return SemaRef.StmtError();

    if (S->getCond()) {
      // Convert the condition to a boolean value.
      OwningExprResult CondE = getSema().ActOnBooleanCondition(0, 
                                                               S->getForLoc(), 
                                                               move(Cond));
      if (CondE.isInvalid())
        return getSema().StmtError();

      Cond = move(CondE);
    }
  }

  Sema::FullExprArg FullCond(getSema().MakeFullExpr(Cond));  
  if (!S->getConditionVariable() && S->getCond() && !FullCond->get())
    return SemaRef.StmtError();

  // Transform the increment
  OwningExprResult Inc = getDerived().TransformExpr(S->getInc());
  if (Inc.isInvalid())
    return SemaRef.StmtError();

  Sema::FullExprArg FullInc(getSema().MakeFullExpr(Inc));
  if (S->getInc() && !FullInc->get())
    return SemaRef.StmtError();

  // Transform the body
  OwningStmtResult Body = getDerived().TransformStmt(S->getBody());
  if (Body.isInvalid())
    return SemaRef.StmtError();

  if (!getDerived().AlwaysRebuild() &&
      Init.get() == S->getInit() &&
      FullCond->get() == S->getCond() &&
      Inc.get() == S->getInc() &&
      Body.get() == S->getBody())
    return SemaRef.Owned(S->Retain());

  return getDerived().RebuildForStmt(S->getForLoc(), S->getLParenLoc(),
                                     move(Init), FullCond, ConditionVar,
                                     FullInc, S->getRParenLoc(), move(Body));
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
    Decl *Transformed = getDerived().TransformDefinition((*D)->getLocation(),
                                                         *D);
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
  
  ASTOwningVector<&ActionBase::DeleteExpr> Constraints(getSema());
  ASTOwningVector<&ActionBase::DeleteExpr> Exprs(getSema());
  llvm::SmallVector<IdentifierInfo *, 4> Names;

  OwningExprResult AsmString(SemaRef);
  ASTOwningVector<&ActionBase::DeleteExpr> Clobbers(getSema());

  bool ExprsChanged = false;
  
  // Go through the outputs.
  for (unsigned I = 0, E = S->getNumOutputs(); I != E; ++I) {
    Names.push_back(S->getOutputIdentifier(I));
    
    // No need to transform the constraint literal.
    Constraints.push_back(S->getOutputConstraintLiteral(I)->Retain());
    
    // Transform the output expr.
    Expr *OutputExpr = S->getOutputExpr(I);
    OwningExprResult Result = getDerived().TransformExpr(OutputExpr);
    if (Result.isInvalid())
      return SemaRef.StmtError();
    
    ExprsChanged |= Result.get() != OutputExpr;
    
    Exprs.push_back(Result.takeAs<Expr>());
  }
  
  // Go through the inputs.
  for (unsigned I = 0, E = S->getNumInputs(); I != E; ++I) {
    Names.push_back(S->getInputIdentifier(I));
    
    // No need to transform the constraint literal.
    Constraints.push_back(S->getInputConstraintLiteral(I)->Retain());
    
    // Transform the input expr.
    Expr *InputExpr = S->getInputExpr(I);
    OwningExprResult Result = getDerived().TransformExpr(InputExpr);
    if (Result.isInvalid())
      return SemaRef.StmtError();
    
    ExprsChanged |= Result.get() != InputExpr;
    
    Exprs.push_back(Result.takeAs<Expr>());
  }
  
  if (!getDerived().AlwaysRebuild() && !ExprsChanged)
    return SemaRef.Owned(S->Retain());

  // Go through the clobbers.
  for (unsigned I = 0, E = S->getNumClobbers(); I != E; ++I)
    Clobbers.push_back(S->getClobber(I)->Retain());

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
                                     move(AsmString),
                                     move_arg(Clobbers),
                                     S->getRParenLoc(),
                                     S->isMSAsm());
}


template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformObjCAtTryStmt(ObjCAtTryStmt *S) {
  // Transform the body of the @try.
  OwningStmtResult TryBody = getDerived().TransformStmt(S->getTryBody());
  if (TryBody.isInvalid())
    return SemaRef.StmtError();
  
  // Transform the @catch statements (if present).
  bool AnyCatchChanged = false;
  ASTOwningVector<&ActionBase::DeleteStmt> CatchStmts(SemaRef);
  for (unsigned I = 0, N = S->getNumCatchStmts(); I != N; ++I) {
    OwningStmtResult Catch = getDerived().TransformStmt(S->getCatchStmt(I));
    if (Catch.isInvalid())
      return SemaRef.StmtError();
    if (Catch.get() != S->getCatchStmt(I))
      AnyCatchChanged = true;
    CatchStmts.push_back(Catch.release());
  }
  
  // Transform the @finally statement (if present).
  OwningStmtResult Finally(SemaRef);
  if (S->getFinallyStmt()) {
    Finally = getDerived().TransformStmt(S->getFinallyStmt());
    if (Finally.isInvalid())
      return SemaRef.StmtError();
  }

  // If nothing changed, just retain this statement.
  if (!getDerived().AlwaysRebuild() &&
      TryBody.get() == S->getTryBody() &&
      !AnyCatchChanged &&
      Finally.get() == S->getFinallyStmt())
    return SemaRef.Owned(S->Retain());
  
  // Build a new statement.
  return getDerived().RebuildObjCAtTryStmt(S->getAtTryLoc(), move(TryBody),
                                           move_arg(CatchStmts), move(Finally));
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformObjCAtCatchStmt(ObjCAtCatchStmt *S) {
  // Transform the @catch parameter, if there is one.
  VarDecl *Var = 0;
  if (VarDecl *FromVar = S->getCatchParamDecl()) {
    TypeSourceInfo *TSInfo = 0;
    if (FromVar->getTypeSourceInfo()) {
      TSInfo = getDerived().TransformType(FromVar->getTypeSourceInfo());
      if (!TSInfo)
        return SemaRef.StmtError();
    }
    
    QualType T;
    if (TSInfo)
      T = TSInfo->getType();
    else {
      T = getDerived().TransformType(FromVar->getType());
      if (T.isNull())
        return SemaRef.StmtError();        
    }
    
    Var = getDerived().RebuildObjCExceptionDecl(FromVar, TSInfo, T);
    if (!Var)
      return SemaRef.StmtError();
  }
  
  OwningStmtResult Body = getDerived().TransformStmt(S->getCatchBody());
  if (Body.isInvalid())
    return SemaRef.StmtError();
  
  return getDerived().RebuildObjCAtCatchStmt(S->getAtCatchLoc(), 
                                             S->getRParenLoc(),
                                             Var, move(Body));
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformObjCAtFinallyStmt(ObjCAtFinallyStmt *S) {
  // Transform the body.
  OwningStmtResult Body = getDerived().TransformStmt(S->getFinallyBody());
  if (Body.isInvalid())
    return SemaRef.StmtError();
  
  // If nothing changed, just retain this statement.
  if (!getDerived().AlwaysRebuild() &&
      Body.get() == S->getFinallyBody())
    return SemaRef.Owned(S->Retain());

  // Build a new statement.
  return getDerived().RebuildObjCAtFinallyStmt(S->getAtFinallyLoc(),
                                               move(Body));
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformObjCAtThrowStmt(ObjCAtThrowStmt *S) {
  OwningExprResult Operand(SemaRef);
  if (S->getThrowExpr()) {
    Operand = getDerived().TransformExpr(S->getThrowExpr());
    if (Operand.isInvalid())
      return getSema().StmtError();
  }
  
  if (!getDerived().AlwaysRebuild() &&
      Operand.get() == S->getThrowExpr())
    return getSema().Owned(S->Retain());
    
  return getDerived().RebuildObjCAtThrowStmt(S->getThrowLoc(), move(Operand));
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformObjCAtSynchronizedStmt(
                                                  ObjCAtSynchronizedStmt *S) {
  // Transform the object we are locking.
  OwningExprResult Object = getDerived().TransformExpr(S->getSynchExpr());
  if (Object.isInvalid())
    return SemaRef.StmtError();
  
  // Transform the body.
  OwningStmtResult Body = getDerived().TransformStmt(S->getSynchBody());
  if (Body.isInvalid())
    return SemaRef.StmtError();
  
  // If nothing change, just retain the current statement.
  if (!getDerived().AlwaysRebuild() &&
      Object.get() == S->getSynchExpr() &&
      Body.get() == S->getSynchBody())
    return SemaRef.Owned(S->Retain());

  // Build a new statement.
  return getDerived().RebuildObjCAtSynchronizedStmt(S->getAtSynchronizedLoc(),
                                                    move(Object), move(Body));
}

template<typename Derived>
Sema::OwningStmtResult
TreeTransform<Derived>::TransformObjCForCollectionStmt(
                                                  ObjCForCollectionStmt *S) {
  // Transform the element statement.
  OwningStmtResult Element = getDerived().TransformStmt(S->getElement());
  if (Element.isInvalid())
    return SemaRef.StmtError();
  
  // Transform the collection expression.
  OwningExprResult Collection = getDerived().TransformExpr(S->getCollection());
  if (Collection.isInvalid())
    return SemaRef.StmtError();
  
  // Transform the body.
  OwningStmtResult Body = getDerived().TransformStmt(S->getBody());
  if (Body.isInvalid())
    return SemaRef.StmtError();
  
  // If nothing changed, just retain this statement.
  if (!getDerived().AlwaysRebuild() &&
      Element.get() == S->getElement() &&
      Collection.get() == S->getCollection() &&
      Body.get() == S->getBody())
    return SemaRef.Owned(S->Retain());
  
  // Build a new statement.
  return getDerived().RebuildObjCForCollectionStmt(S->getForLoc(),
                                                   /*FIXME:*/S->getForLoc(),
                                                   move(Element),
                                                   move(Collection),
                                                   S->getRParenLoc(),
                                                   move(Body));
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
                                            ExceptionDecl->getTypeSourceInfo(),
                                            ExceptionDecl->getIdentifier(),
                                            ExceptionDecl->getLocation(),
                                            /*FIXME: Inaccurate*/
                                    SourceRange(ExceptionDecl->getLocation()));
    if (!Var || Var->isInvalidDecl())
      return SemaRef.StmtError();
  }

  // Transform the actual exception handler.
  OwningStmtResult Handler = getDerived().TransformStmt(S->getHandlerBlock());
  if (Handler.isInvalid())
    return SemaRef.StmtError();

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

  ValueDecl *ND
    = cast_or_null<ValueDecl>(getDerived().TransformDecl(E->getLocation(),
                                                         E->getDecl()));
  if (!ND)
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() && 
      Qualifier == E->getQualifier() &&
      ND == E->getDecl() &&
      !E->hasExplicitTemplateArgumentList()) {

    // Mark it referenced in the new context regardless.
    // FIXME: this is a bit instantiation-specific.
    SemaRef.MarkDeclarationReferenced(E->getLocation(), ND);

    return SemaRef.Owned(E->Retain());
  }

  TemplateArgumentListInfo TransArgs, *TemplateArgs = 0;
  if (E->hasExplicitTemplateArgumentList()) {
    TemplateArgs = &TransArgs;
    TransArgs.setLAngleLoc(E->getLAngleLoc());
    TransArgs.setRAngleLoc(E->getRAngleLoc());
    for (unsigned I = 0, N = E->getNumTemplateArgs(); I != N; ++I) {
      TemplateArgumentLoc Loc;
      if (getDerived().TransformTemplateArgument(E->getTemplateArgs()[I], Loc))
        return SemaRef.ExprError();
      TransArgs.addArgument(Loc);
    }
  }

  return getDerived().RebuildDeclRefExpr(Qualifier, E->getQualifierRange(),
                                         ND, E->getLocation(), TemplateArgs);
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
TreeTransform<Derived>::TransformOffsetOfExpr(OffsetOfExpr *E) {
  // Transform the type.
  TypeSourceInfo *Type = getDerived().TransformType(E->getTypeSourceInfo());
  if (!Type)
    return getSema().ExprError();
  
  // Transform all of the components into components similar to what the
  // parser uses.
  // FIXME: It would be slightly more efficient in the non-dependent case to 
  // just map FieldDecls, rather than requiring the rebuilder to look for 
  // the fields again. However, __builtin_offsetof is rare enough in 
  // template code that we don't care.
  bool ExprChanged = false;
  typedef Action::OffsetOfComponent Component;
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
      OwningExprResult Index = getDerived().TransformExpr(FromIndex);
      if (Index.isInvalid())
        return getSema().ExprError();
      
      ExprChanged = ExprChanged || Index.get() != FromIndex;
      Comp.isBrackets = true;
      Comp.U.E = Index.takeAs<Expr>(); // FIXME: leaked
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
    return SemaRef.Owned(E->Retain());
  
  // Build a new offsetof expression.
  return getDerived().RebuildOffsetOfExpr(E->getOperatorLoc(), Type,
                                          Components.data(), Components.size(),
                                          E->getRParenLoc());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformSizeOfAlignOfExpr(SizeOfAlignOfExpr *E) {
  if (E->isArgumentType()) {
    TypeSourceInfo *OldT = E->getArgumentTypeInfo();

    TypeSourceInfo *NewT = getDerived().TransformType(OldT);
    if (!NewT)
      return SemaRef.ExprError();

    if (!getDerived().AlwaysRebuild() && OldT == NewT)
      return SemaRef.Owned(E->Retain());

    return getDerived().RebuildSizeOfAlignOf(NewT, E->getOperatorLoc(),
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

  ValueDecl *Member
    = cast_or_null<ValueDecl>(getDerived().TransformDecl(E->getMemberLoc(),
                                                         E->getMemberDecl()));
  if (!Member)
    return SemaRef.ExprError();

  NamedDecl *FoundDecl = E->getFoundDecl();
  if (FoundDecl == E->getMemberDecl()) {
    FoundDecl = Member;
  } else {
    FoundDecl = cast_or_null<NamedDecl>(
                   getDerived().TransformDecl(E->getMemberLoc(), FoundDecl));
    if (!FoundDecl)
      return SemaRef.ExprError();
  }

  if (!getDerived().AlwaysRebuild() &&
      Base.get() == E->getBase() &&
      Qualifier == E->getQualifier() &&
      Member == E->getMemberDecl() &&
      FoundDecl == E->getFoundDecl() &&
      !E->hasExplicitTemplateArgumentList()) {
    
    // Mark it referenced in the new context regardless.
    // FIXME: this is a bit instantiation-specific.
    SemaRef.MarkDeclarationReferenced(E->getMemberLoc(), Member);
    return SemaRef.Owned(E->Retain());
  }

  TemplateArgumentListInfo TransArgs;
  if (E->hasExplicitTemplateArgumentList()) {
    TransArgs.setLAngleLoc(E->getLAngleLoc());
    TransArgs.setRAngleLoc(E->getRAngleLoc());
    for (unsigned I = 0, N = E->getNumTemplateArgs(); I != N; ++I) {
      TemplateArgumentLoc Loc;
      if (getDerived().TransformTemplateArgument(E->getTemplateArgs()[I], Loc))
        return SemaRef.ExprError();
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

  return getDerived().RebuildMemberExpr(move(Base), FakeOperatorLoc,
                                        E->isArrow(),
                                        Qualifier,
                                        E->getQualifierRange(),
                                        E->getMemberLoc(),
                                        Member,
                                        FoundDecl,
                                        (E->hasExplicitTemplateArgumentList()
                                           ? &TransArgs : 0),
                                        FirstQualifierInScope);
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
  // Implicit casts are eliminated during transformation, since they
  // will be recomputed by semantic analysis after transformation.
  return getDerived().TransformExpr(E->getSubExprAsWritten());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCStyleCastExpr(CStyleCastExpr *E) {
  TypeSourceInfo *OldT;
  TypeSourceInfo *NewT;
  {
    // FIXME: Source location isn't quite accurate.
    SourceLocation TypeStartLoc
      = SemaRef.PP.getLocForEndOfToken(E->getLParenLoc());
    TemporaryBase Rebase(*this, TypeStartLoc, DeclarationName());

    OldT = E->getTypeInfoAsWritten();
    NewT = getDerived().TransformType(OldT);
    if (!NewT)
      return SemaRef.ExprError();
  }

  OwningExprResult SubExpr
    = getDerived().TransformExpr(E->getSubExprAsWritten());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      OldT == NewT &&
      SubExpr.get() == E->getSubExpr())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildCStyleCastExpr(E->getLParenLoc(),
                                            NewT,
                                            E->getRParenLoc(),
                                            move(SubExpr));
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCompoundLiteralExpr(CompoundLiteralExpr *E) {
  TypeSourceInfo *OldT = E->getTypeSourceInfo();
  TypeSourceInfo *NewT = getDerived().TransformType(OldT);
  if (!NewT)
    return SemaRef.ExprError();

  OwningExprResult Init = getDerived().TransformExpr(E->getInitializer());
  if (Init.isInvalid())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      OldT == NewT &&
      Init.get() == E->getInitializer())
    return SemaRef.Owned(E->Retain());

  // Note: the expression type doesn't necessarily match the
  // type-as-written, but that's okay, because it should always be
  // derivable from the initializer.

  return getDerived().RebuildCompoundLiteralExpr(E->getLParenLoc(), NewT,
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
                                      E->getRBraceLoc(), E->getType());
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
  TypeSourceInfo *TInfo = getDerived().TransformType(E->getWrittenTypeInfo());
  if (!TInfo)
    return SemaRef.ExprError();

  OwningExprResult SubExpr = getDerived().TransformExpr(E->getSubExpr());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      TInfo == E->getWrittenTypeInfo() &&
      SubExpr.get() == E->getSubExpr())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildVAArgExpr(E->getBuiltinLoc(), move(SubExpr),
                                       TInfo, E->getRParenLoc());
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
Sema::OwningExprResult 
TreeTransform<Derived>::TransformStmtExpr(StmtExpr *E) {
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
  TypeSourceInfo *TInfo1;
  TypeSourceInfo *TInfo2;
  
  TInfo1 = getDerived().TransformType(E->getArgTInfo1());
  if (!TInfo1)
    return SemaRef.ExprError();

  TInfo2 = getDerived().TransformType(E->getArgTInfo2());
  if (!TInfo2)
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      TInfo1 == E->getArgTInfo1() &&
      TInfo2 == E->getArgTInfo2())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildTypesCompatibleExpr(E->getBuiltinLoc(),
                                                 TInfo1, TInfo2,
                                                 E->getRParenLoc());
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
  switch (E->getOperator()) {
  case OO_New:
  case OO_Delete:
  case OO_Array_New:
  case OO_Array_Delete:
    llvm_unreachable("new and delete operators cannot use CXXOperatorCallExpr");
    return SemaRef.ExprError();
    
  case OO_Call: {
    // This is a call to an object's operator().
    assert(E->getNumArgs() >= 1 && "Object call is missing arguments");

    // Transform the object itself.
    OwningExprResult Object = getDerived().TransformExpr(E->getArg(0));
    if (Object.isInvalid())
      return SemaRef.ExprError();

    // FIXME: Poor location information
    SourceLocation FakeLParenLoc
      = SemaRef.PP.getLocForEndOfToken(
                              static_cast<Expr *>(Object.get())->getLocEnd());

    // Transform the call arguments.
    ASTOwningVector<&ActionBase::DeleteExpr> Args(SemaRef);
    llvm::SmallVector<SourceLocation, 4> FakeCommaLocs;
    for (unsigned I = 1, N = E->getNumArgs(); I != N; ++I) {
      if (getDerived().DropCallArgument(E->getArg(I)))
        break;
      
      OwningExprResult Arg = getDerived().TransformExpr(E->getArg(I));
      if (Arg.isInvalid())
        return SemaRef.ExprError();

      // FIXME: Poor source location information.
      SourceLocation FakeCommaLoc
        = SemaRef.PP.getLocForEndOfToken(
                                 static_cast<Expr *>(Arg.get())->getLocEnd());
      FakeCommaLocs.push_back(FakeCommaLoc);
      Args.push_back(Arg.release());
    }

    return getDerived().RebuildCallExpr(move(Object), FakeLParenLoc,
                                        move_arg(Args),
                                        FakeCommaLocs.data(),
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
    return SemaRef.ExprError();

  case OO_None:
  case NUM_OVERLOADED_OPERATORS:
    llvm_unreachable("not an overloaded operator?");
    return SemaRef.ExprError();
  }

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
  TypeSourceInfo *OldT;
  TypeSourceInfo *NewT;
  {
    // FIXME: Source location isn't quite accurate.
    SourceLocation TypeStartLoc
      = SemaRef.PP.getLocForEndOfToken(E->getOperatorLoc());
    TemporaryBase Rebase(*this, TypeStartLoc, DeclarationName());

    OldT = E->getTypeInfoAsWritten();
    NewT = getDerived().TransformType(OldT);
    if (!NewT)
      return SemaRef.ExprError();
  }

  OwningExprResult SubExpr
    = getDerived().TransformExpr(E->getSubExprAsWritten());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      OldT == NewT &&
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
                                              NewT,
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
  TypeSourceInfo *OldT;
  TypeSourceInfo *NewT;
  {
    TemporaryBase Rebase(*this, E->getTypeBeginLoc(), DeclarationName());

    OldT = E->getTypeInfoAsWritten();
    NewT = getDerived().TransformType(OldT);
    if (!NewT)
      return SemaRef.ExprError();
  }

  OwningExprResult SubExpr
    = getDerived().TransformExpr(E->getSubExprAsWritten());
  if (SubExpr.isInvalid())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      OldT == NewT &&
      SubExpr.get() == E->getSubExpr())
    return SemaRef.Owned(E->Retain());

  // FIXME: The end of the type's source range is wrong
  return getDerived().RebuildCXXFunctionalCastExpr(
                                  /*FIXME:*/SourceRange(E->getTypeBeginLoc()),
                                                   NewT,
                                      /*FIXME:*/E->getSubExpr()->getLocStart(),
                                                   move(SubExpr),
                                                   E->getRParenLoc());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXTypeidExpr(CXXTypeidExpr *E) {
  if (E->isTypeOperand()) {
    TypeSourceInfo *TInfo
      = getDerived().TransformType(E->getTypeOperandSourceInfo());
    if (!TInfo)
      return SemaRef.ExprError();

    if (!getDerived().AlwaysRebuild() &&
        TInfo == E->getTypeOperandSourceInfo())
      return SemaRef.Owned(E->Retain());

    return getDerived().RebuildCXXTypeidExpr(E->getType(),
                                             E->getLocStart(),
                                             TInfo,
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

  return getDerived().RebuildCXXTypeidExpr(E->getType(),
                                           E->getLocStart(),
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

  return getDerived().RebuildCXXThisExpr(E->getLocStart(), T, E->isImplicit());
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
    = cast_or_null<ParmVarDecl>(getDerived().TransformDecl(E->getLocStart(),
                                                           E->getParam()));
  if (!Param)
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      Param == E->getParam())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildCXXDefaultArgExpr(E->getUsedLocation(), Param);
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXScalarValueInitExpr(CXXScalarValueInitExpr *E) {
  TemporaryBase Rebase(*this, E->getTypeBeginLoc(), DeclarationName());

  QualType T = getDerived().TransformType(E->getType());
  if (T.isNull())
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      T == E->getType())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildCXXScalarValueInitExpr(E->getTypeBeginLoc(),
                                                 /*FIXME:*/E->getTypeBeginLoc(),
                                                    T,
                                                    E->getRParenLoc());
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
    if (getDerived().DropCallArgument(E->getConstructorArg(I)))
      break;
    
    OwningExprResult Arg = getDerived().TransformExpr(E->getConstructorArg(I));
    if (Arg.isInvalid())
      return SemaRef.ExprError();

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
      return SemaRef.ExprError();
  }

  FunctionDecl *OperatorNew = 0;
  if (E->getOperatorNew()) {
    OperatorNew = cast_or_null<FunctionDecl>(
                                 getDerived().TransformDecl(E->getLocStart(),
                                                         E->getOperatorNew()));
    if (!OperatorNew)
      return SemaRef.ExprError();
  }

  FunctionDecl *OperatorDelete = 0;
  if (E->getOperatorDelete()) {
    OperatorDelete = cast_or_null<FunctionDecl>(
                                   getDerived().TransformDecl(E->getLocStart(),
                                                       E->getOperatorDelete()));
    if (!OperatorDelete)
      return SemaRef.ExprError();
  }
  
  if (!getDerived().AlwaysRebuild() &&
      AllocType == E->getAllocatedType() &&
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
    return SemaRef.Owned(E->Retain());
  }

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
        = SemaRef.Owned(new (SemaRef.Context) IntegerLiteral(
                                                  ConsArrayT->getSize(), 
                                                  SemaRef.Context.getSizeType(),
                                                  /*FIXME:*/E->getLocStart()));
      AllocType = ConsArrayT->getElementType();
    } else if (const DependentSizedArrayType *DepArrayT
                              = dyn_cast<DependentSizedArrayType>(ArrayT)) {
      if (DepArrayT->getSizeExpr()) {
        ArraySize = SemaRef.Owned(DepArrayT->getSizeExpr()->Retain());
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

  // Transform the delete operator, if known.
  FunctionDecl *OperatorDelete = 0;
  if (E->getOperatorDelete()) {
    OperatorDelete = cast_or_null<FunctionDecl>(
                                   getDerived().TransformDecl(E->getLocStart(),
                                                       E->getOperatorDelete()));
    if (!OperatorDelete)
      return SemaRef.ExprError();
  }
  
  if (!getDerived().AlwaysRebuild() &&
      Operand.get() == E->getArgument() &&
      OperatorDelete == E->getOperatorDelete()) {
    // Mark any declarations we need as referenced.
    // FIXME: instantiation-specific.
    if (OperatorDelete)
      SemaRef.MarkDeclarationReferenced(E->getLocStart(), OperatorDelete);
    return SemaRef.Owned(E->Retain());
  }

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

  Sema::TypeTy *ObjectTypePtr = 0;
  bool MayBePseudoDestructor = false;
  Base = SemaRef.ActOnStartCXXMemberReference(0, move(Base), 
                                              E->getOperatorLoc(),
                                        E->isArrow()? tok::arrow : tok::period,
                                              ObjectTypePtr,
                                              MayBePseudoDestructor);
  if (Base.isInvalid())
    return SemaRef.ExprError();
                                              
  QualType ObjectType = QualType::getFromOpaquePtr(ObjectTypePtr);
  NestedNameSpecifier *Qualifier
    = getDerived().TransformNestedNameSpecifier(E->getQualifier(),
                                                E->getQualifierRange(),
                                                ObjectType);
  if (E->getQualifier() && !Qualifier)
    return SemaRef.ExprError();

  PseudoDestructorTypeStorage Destroyed;
  if (E->getDestroyedTypeInfo()) {
    TypeSourceInfo *DestroyedTypeInfo
      = getDerived().TransformType(E->getDestroyedTypeInfo(), ObjectType);
    if (!DestroyedTypeInfo)
      return SemaRef.ExprError();
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
    
    Sema::TypeTy *T = SemaRef.getDestructorName(E->getTildeLoc(),
                                              *E->getDestroyedTypeIdentifier(),
                                                E->getDestroyedTypeLoc(),
                                                /*Scope=*/0,
                                                SS, ObjectTypePtr,
                                                false);
    if (!T)
      return SemaRef.ExprError();
    
    Destroyed
      = SemaRef.Context.getTrivialTypeSourceInfo(SemaRef.GetTypeFromParser(T),
                                                 E->getDestroyedTypeLoc());
  }

  TypeSourceInfo *ScopeTypeInfo = 0;
  if (E->getScopeTypeInfo()) {
    ScopeTypeInfo = getDerived().TransformType(E->getScopeTypeInfo(), 
                                               ObjectType);
    if (!ScopeTypeInfo)
      return SemaRef.ExprError();
  }
  
  return getDerived().RebuildCXXPseudoDestructorExpr(move(Base),
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
Sema::OwningExprResult
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
        return SemaRef.ExprError();
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
      return SemaRef.ExprError();
    
    SS.setScopeRep(Qualifier);
    SS.setRange(Old->getQualifierRange());
  } 
  
  if (Old->getNamingClass()) {
    CXXRecordDecl *NamingClass
      = cast_or_null<CXXRecordDecl>(getDerived().TransformDecl(
                                                            Old->getNameLoc(),
                                                        Old->getNamingClass()));
    if (!NamingClass)
      return SemaRef.ExprError();
    
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
      return SemaRef.ExprError();
    TransArgs.addArgument(Loc);
  }

  return getDerived().RebuildTemplateIdExpr(SS, R, Old->requiresADL(),
                                            TransArgs);
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
TreeTransform<Derived>::TransformDependentScopeDeclRefExpr(
                                                  DependentScopeDeclRefExpr *E) {
  NestedNameSpecifier *NNS
    = getDerived().TransformNestedNameSpecifier(E->getQualifier(),
                                                E->getQualifierRange());
  if (!NNS)
    return SemaRef.ExprError();

  DeclarationName Name
    = getDerived().TransformDeclarationName(E->getDeclName(), E->getLocation());
  if (!Name)
    return SemaRef.ExprError();

  if (!E->hasExplicitTemplateArgs()) {
    if (!getDerived().AlwaysRebuild() &&
        NNS == E->getQualifier() &&
        Name == E->getDeclName())
      return SemaRef.Owned(E->Retain());

    return getDerived().RebuildDependentScopeDeclRefExpr(NNS,
                                                         E->getQualifierRange(),
                                                         Name, E->getLocation(),
                                                         /*TemplateArgs*/ 0);
  }

  TemplateArgumentListInfo TransArgs(E->getLAngleLoc(), E->getRAngleLoc());
  for (unsigned I = 0, N = E->getNumTemplateArgs(); I != N; ++I) {
    TemplateArgumentLoc Loc;
    if (getDerived().TransformTemplateArgument(E->getTemplateArgs()[I], Loc))
      return SemaRef.ExprError();
    TransArgs.addArgument(Loc);
  }

  return getDerived().RebuildDependentScopeDeclRefExpr(NNS,
                                                       E->getQualifierRange(),
                                                       Name, E->getLocation(),
                                                       &TransArgs);
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXConstructExpr(CXXConstructExpr *E) {
  // CXXConstructExprs are always implicit, so when we have a
  // 1-argument construction we just transform that argument.
  if (E->getNumArgs() == 1 ||
      (E->getNumArgs() > 1 && getDerived().DropCallArgument(E->getArg(1))))
    return getDerived().TransformExpr(E->getArg(0));

  TemporaryBase Rebase(*this, /*FIXME*/E->getLocStart(), DeclarationName());

  QualType T = getDerived().TransformType(E->getType());
  if (T.isNull())
    return SemaRef.ExprError();

  CXXConstructorDecl *Constructor
    = cast_or_null<CXXConstructorDecl>(
                                getDerived().TransformDecl(E->getLocStart(),
                                                         E->getConstructor()));
  if (!Constructor)
    return SemaRef.ExprError();

  bool ArgumentChanged = false;
  ASTOwningVector<&ActionBase::DeleteExpr> Args(SemaRef);
  for (CXXConstructExpr::arg_iterator Arg = E->arg_begin(),
       ArgEnd = E->arg_end();
       Arg != ArgEnd; ++Arg) {
    if (getDerived().DropCallArgument(*Arg)) {
      ArgumentChanged = true;
      break;
    }

    OwningExprResult TransArg = getDerived().TransformExpr(*Arg);
    if (TransArg.isInvalid())
      return SemaRef.ExprError();

    ArgumentChanged = ArgumentChanged || TransArg.get() != *Arg;
    Args.push_back(TransArg.takeAs<Expr>());
  }

  if (!getDerived().AlwaysRebuild() &&
      T == E->getType() &&
      Constructor == E->getConstructor() &&
      !ArgumentChanged) {
    // Mark the constructor as referenced.
    // FIXME: Instantiation-specific
    SemaRef.MarkDeclarationReferenced(E->getLocStart(), Constructor);
    return SemaRef.Owned(E->Retain());
  }

  return getDerived().RebuildCXXConstructExpr(T, /*FIXME:*/E->getLocStart(),
                                              Constructor, E->isElidable(),
                                              move_arg(Args));
}

/// \brief Transform a C++ temporary-binding expression.
///
/// Since CXXBindTemporaryExpr nodes are implicitly generated, we just
/// transform the subexpression and return that.
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXBindTemporaryExpr(CXXBindTemporaryExpr *E) {
  return getDerived().TransformExpr(E->getSubExpr());
}

/// \brief Transform a C++ reference-binding expression.
///
/// Since CXXBindReferenceExpr nodes are implicitly generated, we just
/// transform the subexpression and return that.
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXBindReferenceExpr(CXXBindReferenceExpr *E) {
  return getDerived().TransformExpr(E->getSubExpr());
}

/// \brief Transform a C++ expression that contains temporaries that should
/// be destroyed after the expression is evaluated.
///
/// Since CXXExprWithTemporaries nodes are implicitly generated, we
/// just transform the subexpression and return that.
template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformCXXExprWithTemporaries(
                                                    CXXExprWithTemporaries *E) {
  return getDerived().TransformExpr(E->getSubExpr());
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
                                  getDerived().TransformDecl(E->getLocStart(), 
                                                         E->getConstructor()));
  if (!Constructor)
    return SemaRef.ExprError();

  bool ArgumentChanged = false;
  ASTOwningVector<&ActionBase::DeleteExpr> Args(SemaRef);
  Args.reserve(E->getNumArgs());
  for (CXXTemporaryObjectExpr::arg_iterator Arg = E->arg_begin(),
                                         ArgEnd = E->arg_end();
       Arg != ArgEnd; ++Arg) {
    if (getDerived().DropCallArgument(*Arg)) {
      ArgumentChanged = true;
      break;
    }

    OwningExprResult TransArg = getDerived().TransformExpr(*Arg);
    if (TransArg.isInvalid())
      return SemaRef.ExprError();

    ArgumentChanged = ArgumentChanged || TransArg.get() != *Arg;
    Args.push_back((Expr *)TransArg.release());
  }

  if (!getDerived().AlwaysRebuild() &&
      T == E->getType() &&
      Constructor == E->getConstructor() &&
      !ArgumentChanged) {
    // FIXME: Instantiation-specific
    SemaRef.MarkDeclarationReferenced(E->getTypeBeginLoc(), Constructor);
    return SemaRef.MaybeBindToTemporary(E->Retain());
  }

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
TreeTransform<Derived>::TransformCXXDependentScopeMemberExpr(
                                                     CXXDependentScopeMemberExpr *E) {
  // Transform the base of the expression.
  OwningExprResult Base(SemaRef, (Expr*) 0);
  Expr *OldBase;
  QualType BaseType;
  QualType ObjectType;
  if (!E->isImplicitAccess()) {
    OldBase = E->getBase();
    Base = getDerived().TransformExpr(OldBase);
    if (Base.isInvalid())
      return SemaRef.ExprError();

    // Start the member reference and compute the object's type.
    Sema::TypeTy *ObjectTy = 0;
    bool MayBePseudoDestructor = false;
    Base = SemaRef.ActOnStartCXXMemberReference(0, move(Base),
                                                E->getOperatorLoc(),
                                      E->isArrow()? tok::arrow : tok::period,
                                                ObjectTy,
                                                MayBePseudoDestructor);
    if (Base.isInvalid())
      return SemaRef.ExprError();

    ObjectType = QualType::getFromOpaquePtr(ObjectTy);
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
      return SemaRef.ExprError();
  }

  DeclarationName Name
    = getDerived().TransformDeclarationName(E->getMember(), E->getMemberLoc(),
                                            ObjectType);
  if (!Name)
    return SemaRef.ExprError();

  if (!E->hasExplicitTemplateArgs()) {
    // This is a reference to a member without an explicitly-specified
    // template argument list. Optimize for this common case.
    if (!getDerived().AlwaysRebuild() &&
        Base.get() == OldBase &&
        BaseType == E->getBaseType() &&
        Qualifier == E->getQualifier() &&
        Name == E->getMember() &&
        FirstQualifierInScope == E->getFirstQualifierFoundInScope())
      return SemaRef.Owned(E->Retain());

    return getDerived().RebuildCXXDependentScopeMemberExpr(move(Base),
                                                       BaseType,
                                                       E->isArrow(),
                                                       E->getOperatorLoc(),
                                                       Qualifier,
                                                       E->getQualifierRange(),
                                                       FirstQualifierInScope,
                                                       Name,
                                                       E->getMemberLoc(),
                                                       /*TemplateArgs*/ 0);
  }

  TemplateArgumentListInfo TransArgs(E->getLAngleLoc(), E->getRAngleLoc());
  for (unsigned I = 0, N = E->getNumTemplateArgs(); I != N; ++I) {
    TemplateArgumentLoc Loc;
    if (getDerived().TransformTemplateArgument(E->getTemplateArgs()[I], Loc))
      return SemaRef.ExprError();
    TransArgs.addArgument(Loc);
  }

  return getDerived().RebuildCXXDependentScopeMemberExpr(move(Base),
                                                     BaseType,
                                                     E->isArrow(),
                                                     E->getOperatorLoc(),
                                                     Qualifier,
                                                     E->getQualifierRange(),
                                                     FirstQualifierInScope,
                                                     Name,
                                                     E->getMemberLoc(),
                                                     &TransArgs);
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformUnresolvedMemberExpr(UnresolvedMemberExpr *Old) {
  // Transform the base of the expression.
  OwningExprResult Base(SemaRef, (Expr*) 0);
  QualType BaseType;
  if (!Old->isImplicitAccess()) {
    Base = getDerived().TransformExpr(Old->getBase());
    if (Base.isInvalid())
      return SemaRef.ExprError();
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
      return SemaRef.ExprError();
  }

  LookupResult R(SemaRef, Old->getMemberName(), Old->getMemberLoc(),
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
        return SemaRef.ExprError();
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
      return SemaRef.ExprError();
    
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
        return SemaRef.ExprError();
      TransArgs.addArgument(Loc);
    }
  }

  // FIXME: to do this check properly, we will need to preserve the
  // first-qualifier-in-scope here, just in case we had a dependent
  // base (and therefore couldn't do the check) and a
  // nested-name-qualifier (and therefore could do the lookup).
  NamedDecl *FirstQualifierInScope = 0;
  
  return getDerived().RebuildUnresolvedMemberExpr(move(Base),
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
Sema::OwningExprResult
TreeTransform<Derived>::TransformObjCStringLiteral(ObjCStringLiteral *E) {
  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformObjCEncodeExpr(ObjCEncodeExpr *E) {
  TypeSourceInfo *EncodedTypeInfo
    = getDerived().TransformType(E->getEncodedTypeSourceInfo());
  if (!EncodedTypeInfo)
    return SemaRef.ExprError();

  if (!getDerived().AlwaysRebuild() &&
      EncodedTypeInfo == E->getEncodedTypeSourceInfo())
    return SemaRef.Owned(E->Retain());

  return getDerived().RebuildObjCEncodeExpr(E->getAtLoc(),
                                            EncodedTypeInfo,
                                            E->getRParenLoc());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformObjCMessageExpr(ObjCMessageExpr *E) {
  // Transform arguments.
  bool ArgChanged = false;
  ASTOwningVector<&ActionBase::DeleteExpr> Args(SemaRef);
  for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I) {
    OwningExprResult Arg = getDerived().TransformExpr(E->getArg(I));
    if (Arg.isInvalid())
      return SemaRef.ExprError();
    
    ArgChanged = ArgChanged || Arg.get() != E->getArg(I);
    Args.push_back(Arg.takeAs<Expr>());
  }

  if (E->getReceiverKind() == ObjCMessageExpr::Class) {
    // Class message: transform the receiver type.
    TypeSourceInfo *ReceiverTypeInfo
      = getDerived().TransformType(E->getClassReceiverTypeInfo());
    if (!ReceiverTypeInfo)
      return SemaRef.ExprError();
    
    // If nothing changed, just retain the existing message send.
    if (!getDerived().AlwaysRebuild() &&
        ReceiverTypeInfo == E->getClassReceiverTypeInfo() && !ArgChanged)
      return SemaRef.Owned(E->Retain());

    // Build a new class message send.
    return getDerived().RebuildObjCMessageExpr(ReceiverTypeInfo,
                                               E->getSelector(),
                                               E->getMethodDecl(),
                                               E->getLeftLoc(),
                                               move_arg(Args),
                                               E->getRightLoc());
  }

  // Instance message: transform the receiver
  assert(E->getReceiverKind() == ObjCMessageExpr::Instance &&
         "Only class and instance messages may be instantiated");
  OwningExprResult Receiver
    = getDerived().TransformExpr(E->getInstanceReceiver());
  if (Receiver.isInvalid())
    return SemaRef.ExprError();

  // If nothing changed, just retain the existing message send.
  if (!getDerived().AlwaysRebuild() &&
      Receiver.get() == E->getInstanceReceiver() && !ArgChanged)
    return SemaRef.Owned(E->Retain());
  
  // Build a new instance message send.
  return getDerived().RebuildObjCMessageExpr(move(Receiver),
                                             E->getSelector(),
                                             E->getMethodDecl(),
                                             E->getLeftLoc(),
                                             move_arg(Args),
                                             E->getRightLoc());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformObjCSelectorExpr(ObjCSelectorExpr *E) {
  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformObjCProtocolExpr(ObjCProtocolExpr *E) {
  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformObjCIvarRefExpr(ObjCIvarRefExpr *E) {
  // Transform the base expression.
  OwningExprResult Base = getDerived().TransformExpr(E->getBase());
  if (Base.isInvalid())
    return SemaRef.ExprError();

  // We don't need to transform the ivar; it will never change.
  
  // If nothing changed, just retain the existing expression.
  if (!getDerived().AlwaysRebuild() &&
      Base.get() == E->getBase())
    return SemaRef.Owned(E->Retain());
  
  return getDerived().RebuildObjCIvarRefExpr(move(Base), E->getDecl(),
                                             E->getLocation(),
                                             E->isArrow(), E->isFreeIvar());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformObjCPropertyRefExpr(ObjCPropertyRefExpr *E) {
  // Transform the base expression.
  OwningExprResult Base = getDerived().TransformExpr(E->getBase());
  if (Base.isInvalid())
    return SemaRef.ExprError();
  
  // We don't need to transform the property; it will never change.
  
  // If nothing changed, just retain the existing expression.
  if (!getDerived().AlwaysRebuild() &&
      Base.get() == E->getBase())
    return SemaRef.Owned(E->Retain());
  
  return getDerived().RebuildObjCPropertyRefExpr(move(Base), E->getProperty(),
                                                 E->getLocation());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformObjCImplicitSetterGetterRefExpr(
                                          ObjCImplicitSetterGetterRefExpr *E) {
  // If this implicit setter/getter refers to class methods, it cannot have any
  // dependent parts. Just retain the existing declaration.
  if (E->getInterfaceDecl())
    return SemaRef.Owned(E->Retain());
  
  // Transform the base expression.
  OwningExprResult Base = getDerived().TransformExpr(E->getBase());
  if (Base.isInvalid())
    return SemaRef.ExprError();
  
  // We don't need to transform the getters/setters; they will never change.
  
  // If nothing changed, just retain the existing expression.
  if (!getDerived().AlwaysRebuild() &&
      Base.get() == E->getBase())
    return SemaRef.Owned(E->Retain());
  
  return getDerived().RebuildObjCImplicitSetterGetterRefExpr(
                                                          E->getGetterMethod(),
                                                             E->getType(),
                                                          E->getSetterMethod(),
                                                             E->getLocation(),
                                                             move(Base));
                                                             
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformObjCSuperExpr(ObjCSuperExpr *E) {
  // Can never occur in a dependent context.
  return SemaRef.Owned(E->Retain());
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformObjCIsaExpr(ObjCIsaExpr *E) {
  // Transform the base expression.
  OwningExprResult Base = getDerived().TransformExpr(E->getBase());
  if (Base.isInvalid())
    return SemaRef.ExprError();
  
  // If nothing changed, just retain the existing expression.
  if (!getDerived().AlwaysRebuild() &&
      Base.get() == E->getBase())
    return SemaRef.Owned(E->Retain());
  
  return getDerived().RebuildObjCIsaExpr(move(Base), E->getIsaMemberLoc(),
                                         E->isArrow());
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
  OwningStmtResult Body = getDerived().TransformStmt(E->getBody());
  if (Body.isInvalid())
    return SemaRef.ExprError();
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
  return SemaRef.ActOnBlockStmtExpr(CaretLoc, move(Body), /*Scope=*/0);
}

template<typename Derived>
Sema::OwningExprResult
TreeTransform<Derived>::TransformBlockDeclRefExpr(BlockDeclRefExpr *E) {
  NestedNameSpecifier *Qualifier = 0;
    
  ValueDecl *ND
  = cast_or_null<ValueDecl>(getDerived().TransformDecl(E->getLocation(),
                                                       E->getDecl()));
  if (!ND)
    return SemaRef.ExprError();
  
  if (!getDerived().AlwaysRebuild() &&
      ND == E->getDecl()) {
    // Mark it referenced in the new context regardless.
    // FIXME: this is a bit instantiation-specific.
    SemaRef.MarkDeclarationReferenced(E->getLocation(), ND);
    
    return SemaRef.Owned(E->Retain());
  }
  
  return getDerived().RebuildDeclRefExpr(Qualifier, SourceLocation(),
                                         ND, E->getLocation(), 0);
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
                                     unsigned NumElements,
                                     VectorType::AltiVecSpecific AltiVecSpec) {
  // FIXME: semantic checking!
  return SemaRef.Context.getVectorType(ElementType, NumElements, AltiVecSpec);
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
                                            const IdentifierInfo &II,
                                            QualType ObjectType) {
  CXXScopeSpec SS;
  SS.setRange(SourceRange(getDerived().getBaseLocation()));
  SS.setScopeRep(Qualifier);
  UnqualifiedId Name;
  Name.setIdentifier(&II, /*FIXME:*/getDerived().getBaseLocation());
  Sema::TemplateTy Template;
  getSema().ActOnDependentTemplateName(/*Scope=*/0,
                                       /*FIXME:*/getDerived().getBaseLocation(),
                                       SS,
                                       Name,
                                       ObjectType.getAsOpaquePtr(),
                                       /*EnteringContext=*/false,
                                       Template);
  return Template.template getAsVal<TemplateName>();
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
                                       ObjectType.getAsOpaquePtr(),
                                       /*EnteringContext=*/false,
                                       Template);
  return Template.template getAsVal<TemplateName>();
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
  Expr *CalleeExpr = ((Expr *)Callee.get())->IgnoreParenCasts();
  bool isPostIncDec = SecondExpr && (Op == OO_PlusPlus || Op == OO_MinusMinus);

  // Determine whether this should be a builtin operation.
  if (Op == OO_Subscript) {
    if (!FirstExpr->getType()->isOverloadableType() &&
        !SecondExpr->getType()->isOverloadableType())
      return getSema().CreateBuiltinArraySubscriptExpr(move(First),
                                                 CalleeExpr->getLocStart(),
                                                       move(Second), OpLoc);
  } else if (Op == OO_Arrow) {
    // -> is never a builtin operation.
    return SemaRef.BuildOverloadedArrowExpr(0, move(First), OpLoc);
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
  UnresolvedSet<16> Functions;

  if (UnresolvedLookupExpr *ULE = dyn_cast<UnresolvedLookupExpr>(CalleeExpr)) {
    assert(ULE->requiresADL());

    // FIXME: Do we have to check
    // IsAcceptableNonMemberOperatorCandidate for each of these?
    Functions.append(ULE->decls_begin(), ULE->decls_end());
  } else {
    Functions.addDecl(cast<DeclRefExpr>(CalleeExpr)->getDecl());
  }

  // Add any functions found via argument-dependent lookup.
  Expr *Args[2] = { FirstExpr, SecondExpr };
  unsigned NumArgs = 1 + (SecondExpr != 0);

  // Create the overloaded operator invocation for unary operators.
  if (NumArgs == 1 || isPostIncDec) {
    UnaryOperator::Opcode Opc
      = UnaryOperator::getOverloadedOpcode(Op, isPostIncDec);
    return SemaRef.CreateOverloadedUnaryOp(OpLoc, Opc, Functions, move(First));
  }

  if (Op == OO_Subscript)
    return SemaRef.CreateOverloadedArraySubscriptExpr(CalleeExpr->getLocStart(),
                                                      OpLoc,
                                                      move(First),
                                                      move(Second));

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

template<typename Derived>
Sema::OwningExprResult 
TreeTransform<Derived>::RebuildCXXPseudoDestructorExpr(ExprArg Base,
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

  Expr *BaseE = (Expr *)Base.get();
  QualType BaseType = BaseE->getType();
  if (BaseE->isTypeDependent() || Destroyed.getIdentifier() ||
      (!isArrow && !BaseType->getAs<RecordType>()) ||
      (isArrow && BaseType->getAs<PointerType>() && 
       !BaseType->getAs<PointerType>()->getPointeeType()
                                              ->template getAs<RecordType>())){
    // This pseudo-destructor expression is still a pseudo-destructor.
    return SemaRef.BuildPseudoDestructorExpr(move(Base), OperatorLoc,
                                             isArrow? tok::arrow : tok::period,
                                             SS, ScopeType, CCLoc, TildeLoc,
                                             Destroyed,
                                             /*FIXME?*/true);
  }
  
  TypeSourceInfo *DestroyedType = Destroyed.getTypeSourceInfo();
  DeclarationName Name
    = SemaRef.Context.DeclarationNames.getCXXDestructorName(
                SemaRef.Context.getCanonicalType(DestroyedType->getType()));
  
  // FIXME: the ScopeType should be tacked onto SS.
  
  return getSema().BuildMemberReferenceExpr(move(Base), BaseType,
                                            OperatorLoc, isArrow,
                                            SS, /*FIXME: FirstQualifier*/ 0,
                                            Name, Destroyed.getLocation(),
                                            /*TemplateArgs*/ 0);
}

} // end namespace clang

#endif // LLVM_CLANG_SEMA_TREETRANSFORM_H
