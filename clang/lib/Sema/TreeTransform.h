//===------- TreeTransform.h - Semantic Tree Transformation -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===//
//
//  This file implements a semantic tree transformation that takes a given
//  AST and rebuilds it, possibly transforming some nodes in the process.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_TREETRANSFORM_H
#define LLVM_CLANG_SEMA_TREETRANSFORM_H

#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/ParsedTemplate.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
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
/// TransformExpr(), TransformDecl(), TransformNestedNameSpecifierLoc(),
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
  /// \brief Private RAII object that helps us forget and then re-remember
  /// the template argument corresponding to a partially-substituted parameter
  /// pack.
  class ForgetPartiallySubstitutedPackRAII {
    Derived &Self;
    TemplateArgument Old;
    
  public:
    ForgetPartiallySubstitutedPackRAII(Derived &Self) : Self(Self) {
      Old = Self.ForgetPartiallySubstitutedPack();
    }
    
    ~ForgetPartiallySubstitutedPackRAII() {
      Self.RememberPartiallySubstitutedPack(Old);
    }
  };
  
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
      
      if (Location.isValid())
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
  
  /// \brief Determine whether we should expand a pack expansion with the
  /// given set of parameter packs into separate arguments by repeatedly
  /// transforming the pattern.
  ///
  /// By default, the transformer never tries to expand pack expansions.
  /// Subclasses can override this routine to provide different behavior.
  ///
  /// \param EllipsisLoc The location of the ellipsis that identifies the
  /// pack expansion.
  ///
  /// \param PatternRange The source range that covers the entire pattern of
  /// the pack expansion.
  ///
  /// \param Unexpanded The set of unexpanded parameter packs within the 
  /// pattern.
  ///
  /// \param NumUnexpanded The number of unexpanded parameter packs in
  /// \p Unexpanded.
  ///
  /// \param ShouldExpand Will be set to \c true if the transformer should
  /// expand the corresponding pack expansions into separate arguments. When
  /// set, \c NumExpansions must also be set.
  ///
  /// \param RetainExpansion Whether the caller should add an unexpanded
  /// pack expansion after all of the expanded arguments. This is used
  /// when extending explicitly-specified template argument packs per
  /// C++0x [temp.arg.explicit]p9.
  ///
  /// \param NumExpansions The number of separate arguments that will be in
  /// the expanded form of the corresponding pack expansion. This is both an
  /// input and an output parameter, which can be set by the caller if the
  /// number of expansions is known a priori (e.g., due to a prior substitution)
  /// and will be set by the callee when the number of expansions is known.
  /// The callee must set this value when \c ShouldExpand is \c true; it may
  /// set this value in other cases.
  ///
  /// \returns true if an error occurred (e.g., because the parameter packs 
  /// are to be instantiated with arguments of different lengths), false 
  /// otherwise. If false, \c ShouldExpand (and possibly \c NumExpansions) 
  /// must be set.
  bool TryExpandParameterPacks(SourceLocation EllipsisLoc,
                               SourceRange PatternRange,
                               const UnexpandedParameterPack *Unexpanded,
                               unsigned NumUnexpanded,
                               bool &ShouldExpand,
                               bool &RetainExpansion,
                               llvm::Optional<unsigned> &NumExpansions) {
    ShouldExpand = false;
    return false;
  }
  
  /// \brief "Forget" about the partially-substituted pack template argument,
  /// when performing an instantiation that must preserve the parameter pack
  /// use.
  ///
  /// This routine is meant to be overridden by the template instantiator.
  TemplateArgument ForgetPartiallySubstitutedPack() {
    return TemplateArgument();
  }
  
  /// \brief "Remember" the partially-substituted pack template argument
  /// after performing an instantiation that must preserve the parameter pack
  /// use.
  ///
  /// This routine is meant to be overridden by the template instantiator.
  void RememberPartiallySubstitutedPack(TemplateArgument Arg) { }
  
  /// \brief Note to the derived class when a function parameter pack is
  /// being expanded.
  void ExpandingFunctionParameterPack(ParmVarDecl *Pack) { }
                                      
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

  /// \brief Transform the given list of expressions.
  ///
  /// This routine transforms a list of expressions by invoking 
  /// \c TransformExpr() for each subexpression. However, it also provides 
  /// support for variadic templates by expanding any pack expansions (if the
  /// derived class permits such expansion) along the way. When pack expansions
  /// are present, the number of outputs may not equal the number of inputs.
  ///
  /// \param Inputs The set of expressions to be transformed.
  ///
  /// \param NumInputs The number of expressions in \c Inputs.
  ///
  /// \param IsCall If \c true, then this transform is being performed on
  /// function-call arguments, and any arguments that should be dropped, will 
  /// be.
  ///
  /// \param Outputs The transformed input expressions will be added to this
  /// vector.
  ///
  /// \param ArgChanged If non-NULL, will be set \c true if any argument changed
  /// due to transformation.
  ///
  /// \returns true if an error occurred, false otherwise.
  bool TransformExprs(Expr **Inputs, unsigned NumInputs, bool IsCall,
                      llvm::SmallVectorImpl<Expr *> &Outputs,
                      bool *ArgChanged = 0);
  
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
  
  /// \brief Transform the given nested-name-specifier with source-location
  /// information.
  ///
  /// By default, transforms all of the types and declarations within the
  /// nested-name-specifier. Subclasses may override this function to provide
  /// alternate behavior.
  NestedNameSpecifierLoc TransformNestedNameSpecifierLoc(
                                                    NestedNameSpecifierLoc NNS,
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
  /// \param SS The nested-name-specifier that qualifies the template
  /// name. This nested-name-specifier must already have been transformed.
  ///
  /// \param Name The template name to transform.
  ///
  /// \param NameLoc The source location of the template name.
  ///
  /// \param ObjectType If we're translating a template name within a member 
  /// access expression, this is the type of the object whose member template
  /// is being referenced.
  ///
  /// \param FirstQualifierInScope If the first part of a nested-name-specifier
  /// also refers to a name within the current (lexical) scope, this is the
  /// declaration it refers to.
  ///
  /// By default, transforms the template name by transforming the declarations
  /// and nested-name-specifiers that occur within the template name.
  /// Subclasses may override this function to provide alternate behavior.
  TemplateName TransformTemplateName(CXXScopeSpec &SS,
                                     TemplateName Name,
                                     SourceLocation NameLoc,                                     
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

  /// \brief Transform the given set of template arguments.
  ///
  /// By default, this operation transforms all of the template arguments 
  /// in the input set using \c TransformTemplateArgument(), and appends
  /// the transformed arguments to the output list.
  ///
  /// Note that this overload of \c TransformTemplateArguments() is merely
  /// a convenience function. Subclasses that wish to override this behavior
  /// should override the iterator-based member template version.
  ///
  /// \param Inputs The set of template arguments to be transformed.
  ///
  /// \param NumInputs The number of template arguments in \p Inputs.
  ///
  /// \param Outputs The set of transformed template arguments output by this
  /// routine.
  ///
  /// Returns true if an error occurred.
  bool TransformTemplateArguments(const TemplateArgumentLoc *Inputs,
                                  unsigned NumInputs,
                                  TemplateArgumentListInfo &Outputs) {
    return TransformTemplateArguments(Inputs, Inputs + NumInputs, Outputs);
  }

  /// \brief Transform the given set of template arguments.
  ///
  /// By default, this operation transforms all of the template arguments 
  /// in the input set using \c TransformTemplateArgument(), and appends
  /// the transformed arguments to the output list. 
  ///
  /// \param First An iterator to the first template argument.
  ///
  /// \param Last An iterator one step past the last template argument.
  ///
  /// \param Outputs The set of transformed template arguments output by this
  /// routine.
  ///
  /// Returns true if an error occurred.
  template<typename InputIterator>
  bool TransformTemplateArguments(InputIterator First,
                                  InputIterator Last,
                                  TemplateArgumentListInfo &Outputs);

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

  StmtResult
  TransformSEHHandler(Stmt *Handler);

  QualType 
  TransformTemplateSpecializationType(TypeLocBuilder &TLB,
                                      TemplateSpecializationTypeLoc TL,
                                      TemplateName Template);

  QualType 
  TransformDependentTemplateSpecializationType(TypeLocBuilder &TLB,
                                      DependentTemplateSpecializationTypeLoc TL,
                                               TemplateName Template,
                                               CXXScopeSpec &SS);

  QualType 
  TransformDependentTemplateSpecializationType(TypeLocBuilder &TLB,
                                               DependentTemplateSpecializationTypeLoc TL,
                                         NestedNameSpecifierLoc QualifierLoc);

  /// \brief Transforms the parameters of a function type into the
  /// given vectors.
  ///
  /// The result vectors should be kept in sync; null entries in the
  /// variables vector are acceptable.
  ///
  /// Return true on error.
  bool TransformFunctionTypeParams(SourceLocation Loc,
                                   ParmVarDecl **Params, unsigned NumParams,
                                   const QualType *ParamTypes,
                                   llvm::SmallVectorImpl<QualType> &PTypes,
                                   llvm::SmallVectorImpl<ParmVarDecl*> *PVars);

  /// \brief Transforms a single function-type parameter.  Return null
  /// on error.
  ///
  /// \param indexAdjustment - A number to add to the parameter's
  ///   scope index;  can be negative
  ParmVarDecl *TransformFunctionTypeParam(ParmVarDecl *OldParm,
                                          int indexAdjustment,
                                        llvm::Optional<unsigned> NumExpansions);

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
                                    RefQualifierKind RefQualifier,
                                    const FunctionType::ExtInfo &Info);

  /// \brief Build a new unprototyped function type.
  QualType RebuildFunctionNoProtoType(QualType ResultType);

  /// \brief Rebuild an unresolved typename type, given the decl that
  /// the UnresolvedUsingTypenameDecl was transformed to.
  QualType RebuildUnresolvedUsingType(Decl *D);

  /// \brief Build a new typedef type.
  QualType RebuildTypedefType(TypedefNameDecl *Typedef) {
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

  /// \brief Build a new unary transform type.
  QualType RebuildUnaryTransformType(QualType BaseType,
                                     UnaryTransformType::UTTKind UKind,
                                     SourceLocation Loc);

  /// \brief Build a new C++0x decltype type.
  ///
  /// By default, performs semantic analysis when building the decltype type.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildDecltypeType(Expr *Underlying, SourceLocation Loc);

  /// \brief Build a new C++0x auto type.
  ///
  /// By default, builds a new AutoType with the given deduced type.
  QualType RebuildAutoType(QualType Deduced) {
    return SemaRef.Context.getAutoType(Deduced);
  }

  /// \brief Build a new template specialization type.
  ///
  /// By default, performs semantic analysis when building the template
  /// specialization type. Subclasses may override this routine to provide
  /// different behavior.
  QualType RebuildTemplateSpecializationType(TemplateName Template,
                                             SourceLocation TemplateLoc,
                                             TemplateArgumentListInfo &Args);

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
                                 NestedNameSpecifierLoc QualifierLoc,
                                 QualType Named) {
    return SemaRef.Context.getElaboratedType(Keyword, 
                                         QualifierLoc.getNestedNameSpecifier(), 
                                             Named);
  }

  /// \brief Build a new typename type that refers to a template-id.
  ///
  /// By default, builds a new DependentNameType type from the
  /// nested-name-specifier and the given type. Subclasses may override
  /// this routine to provide different behavior.
  QualType RebuildDependentTemplateSpecializationType(
                                          ElaboratedTypeKeyword Keyword,
                                          NestedNameSpecifierLoc QualifierLoc,
                                          const IdentifierInfo *Name,
                                          SourceLocation NameLoc,
                                          TemplateArgumentListInfo &Args) {
    // Rebuild the template name.
    // TODO: avoid TemplateName abstraction
    CXXScopeSpec SS;
    SS.Adopt(QualifierLoc);
    TemplateName InstName 
      = getDerived().RebuildTemplateName(SS, *Name, NameLoc, QualType(), 0);
    
    if (InstName.isNull())
      return QualType();
    
    // If it's still dependent, make a dependent specialization.
    if (InstName.getAsDependentTemplateName())
      return SemaRef.Context.getDependentTemplateSpecializationType(Keyword, 
                                          QualifierLoc.getNestedNameSpecifier(), 
                                                                    Name, 
                                                                    Args);
    
    // Otherwise, make an elaborated type wrapping a non-dependent
    // specialization.
    QualType T =
    getDerived().RebuildTemplateSpecializationType(InstName, NameLoc, Args);
    if (T.isNull()) return QualType();
    
    if (Keyword == ETK_None && QualifierLoc.getNestedNameSpecifier() == 0)
      return T;
    
    return SemaRef.Context.getElaboratedType(Keyword, 
                                       QualifierLoc.getNestedNameSpecifier(), 
                                             T);
  }

  /// \brief Build a new typename type that refers to an identifier.
  ///
  /// By default, performs semantic analysis when building the typename type
  /// (or elaborated type). Subclasses may override this routine to provide
  /// different behavior.
  QualType RebuildDependentNameType(ElaboratedTypeKeyword Keyword,
                                    SourceLocation KeywordLoc,
                                    NestedNameSpecifierLoc QualifierLoc,
                                    const IdentifierInfo *Id,
                                    SourceLocation IdLoc) {
    CXXScopeSpec SS;
    SS.Adopt(QualifierLoc);

    if (QualifierLoc.getNestedNameSpecifier()->isDependent()) {
      // If the name is still dependent, just build a new dependent name type.
      if (!SemaRef.computeDeclContext(SS))
        return SemaRef.Context.getDependentNameType(Keyword, 
                                          QualifierLoc.getNestedNameSpecifier(), 
                                                    Id);
    }

    if (Keyword == ETK_None || Keyword == ETK_Typename)
      return SemaRef.CheckTypenameType(Keyword, KeywordLoc, QualifierLoc,
                                       *Id, IdLoc);

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
      // Check where the name exists but isn't a tag type and use that to emit
      // better diagnostics.
      LookupResult Result(SemaRef, Id, IdLoc, Sema::LookupTagName);
      SemaRef.LookupQualifiedName(Result, DC);
      switch (Result.getResultKind()) {
        case LookupResult::Found:
        case LookupResult::FoundOverloaded:
        case LookupResult::FoundUnresolvedValue: {
          NamedDecl *SomeDecl = Result.getRepresentativeDecl();
          unsigned Kind = 0;
          if (isa<TypedefDecl>(SomeDecl)) Kind = 1;
          else if (isa<TypeAliasDecl>(SomeDecl)) Kind = 2;
          else if (isa<ClassTemplateDecl>(SomeDecl)) Kind = 3;
          SemaRef.Diag(IdLoc, diag::err_tag_reference_non_tag) << Kind;
          SemaRef.Diag(SomeDecl->getLocation(), diag::note_declared_at);
          break;
        }
        default:
          // FIXME: Would be nice to highlight just the source range.
          SemaRef.Diag(IdLoc, diag::err_not_tag_in_scope)
            << Kind << Id << DC;
          break;
      }
      return QualType();
    }

    if (!SemaRef.isAcceptableTagRedeclaration(Tag, Kind, IdLoc, *Id)) {
      SemaRef.Diag(KeywordLoc, diag::err_use_with_wrong_tag) << Id;
      SemaRef.Diag(Tag->getLocation(), diag::note_previous_use);
      return QualType();
    }

    // Build the elaborated-type-specifier type.
    QualType T = SemaRef.Context.getTypeDeclType(Tag);
    return SemaRef.Context.getElaboratedType(Keyword, 
                                         QualifierLoc.getNestedNameSpecifier(), 
                                             T);
  }

  /// \brief Build a new pack expansion type.
  ///
  /// By default, builds a new PackExpansionType type from the given pattern.
  /// Subclasses may override this routine to provide different behavior.
  QualType RebuildPackExpansionType(QualType Pattern, 
                                    SourceRange PatternRange,
                                    SourceLocation EllipsisLoc,
                                    llvm::Optional<unsigned> NumExpansions) {
    return getSema().CheckPackExpansion(Pattern, PatternRange, EllipsisLoc,
                                        NumExpansions);
  }

  /// \brief Build a new template name given a nested name specifier, a flag
  /// indicating whether the "template" keyword was provided, and the template
  /// that the template name refers to.
  ///
  /// By default, builds the new template name directly. Subclasses may override
  /// this routine to provide different behavior.
  TemplateName RebuildTemplateName(CXXScopeSpec &SS,
                                   bool TemplateKW,
                                   TemplateDecl *Template);

  /// \brief Build a new template name given a nested name specifier and the
  /// name that is referred to as a template.
  ///
  /// By default, performs semantic analysis to determine whether the name can
  /// be resolved to a specific template, then builds the appropriate kind of
  /// template name. Subclasses may override this routine to provide different
  /// behavior.
  TemplateName RebuildTemplateName(CXXScopeSpec &SS,
                                   const IdentifierInfo &Name,
                                   SourceLocation NameLoc,
                                   QualType ObjectType,
                                   NamedDecl *FirstQualifierInScope);

  /// \brief Build a new template name given a nested name specifier and the
  /// overloaded operator name that is referred to as a template.
  ///
  /// By default, performs semantic analysis to determine whether the name can
  /// be resolved to a specific template, then builds the appropriate kind of
  /// template name. Subclasses may override this routine to provide different
  /// behavior.
  TemplateName RebuildTemplateName(CXXScopeSpec &SS,
                                   OverloadedOperatorKind Operator,
                                   SourceLocation NameLoc,
                                   QualType ObjectType);

  /// \brief Build a new template name given a template template parameter pack
  /// and the 
  ///
  /// By default, performs semantic analysis to determine whether the name can
  /// be resolved to a specific template, then builds the appropriate kind of
  /// template name. Subclasses may override this routine to provide different
  /// behavior.
  TemplateName RebuildTemplateName(TemplateTemplateParmDecl *Param,
                                   const TemplateArgument &ArgPack) {
    return getSema().Context.getSubstTemplateTemplateParmPack(Param, ArgPack);
  }

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
  StmtResult RebuildLabelStmt(SourceLocation IdentLoc, LabelDecl *L,
                              SourceLocation ColonLoc, Stmt *SubStmt) {
    return SemaRef.ActOnLabelStmt(IdentLoc, L, ColonLoc, SubStmt);
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
  StmtResult RebuildWhileStmt(SourceLocation WhileLoc, Sema::FullExprArg Cond,
                              VarDecl *CondVar, Stmt *Body) {
    return getSema().ActOnWhileStmt(WhileLoc, Cond, CondVar, Body);
  }

  /// \brief Build a new do-while statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildDoStmt(SourceLocation DoLoc, Stmt *Body,
                           SourceLocation WhileLoc, SourceLocation LParenLoc,
                           Expr *Cond, SourceLocation RParenLoc) {
    return getSema().ActOnDoStmt(DoLoc, Body, WhileLoc, LParenLoc,
                                 Cond, RParenLoc);
  }

  /// \brief Build a new for statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildForStmt(SourceLocation ForLoc, SourceLocation LParenLoc,
                            Stmt *Init, Sema::FullExprArg Cond, 
                            VarDecl *CondVar, Sema::FullExprArg Inc,
                            SourceLocation RParenLoc, Stmt *Body) {
    return getSema().ActOnForStmt(ForLoc, LParenLoc, Init, Cond, 
                                  CondVar, Inc, RParenLoc, Body);
  }

  /// \brief Build a new goto statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildGotoStmt(SourceLocation GotoLoc, SourceLocation LabelLoc,
                             LabelDecl *Label) {
    return getSema().ActOnGotoStmt(GotoLoc, LabelLoc, Label);
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
  StmtResult RebuildReturnStmt(SourceLocation ReturnLoc, Expr *Result) {
    return getSema().ActOnReturnStmt(ReturnLoc, Result);
  }

  /// \brief Build a new declaration statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildDeclStmt(Decl **Decls, unsigned NumDecls,
                                   SourceLocation StartLoc,
                                   SourceLocation EndLoc) {
    Sema::DeclGroupPtrTy DG = getSema().BuildDeclaratorGroup(Decls, NumDecls);
    return getSema().ActOnDeclStmt(DG, StartLoc, EndLoc);
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
                                            ExceptionDecl->getInnerLocStart(),
                                            ExceptionDecl->getLocation(),
                                            ExceptionDecl->getIdentifier());
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
                                SourceLocation StartLoc,
                                SourceLocation IdLoc,
                                IdentifierInfo *Id) {
    VarDecl *Var = getSema().BuildExceptionDeclaration(0, Declarator,
                                                       StartLoc, IdLoc, Id);
    if (Var)
      getSema().CurContext->addDecl(Var);
    return Var;
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

  /// \brief Build a new C++0x range-based for statement.
  ///
  /// By default, performs semantic analysis to build the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult RebuildCXXForRangeStmt(SourceLocation ForLoc,
                                    SourceLocation ColonLoc,
                                    Stmt *Range, Stmt *BeginEnd,
                                    Expr *Cond, Expr *Inc,
                                    Stmt *LoopVar,
                                    SourceLocation RParenLoc) {
    return getSema().BuildCXXForRangeStmt(ForLoc, ColonLoc, Range, BeginEnd,
                                          Cond, Inc, LoopVar, RParenLoc);
  }
  
  /// \brief Attach body to a C++0x range-based for statement.
  ///
  /// By default, performs semantic analysis to finish the new statement.
  /// Subclasses may override this routine to provide different behavior.
  StmtResult FinishCXXForRangeStmt(Stmt *ForRange, Stmt *Body) {
    return getSema().FinishCXXForRangeStmt(ForRange, Body);
  }
  
  StmtResult RebuildSEHTryStmt(bool IsCXXTry,
                               SourceLocation TryLoc,
                               Stmt *TryBlock,
                               Stmt *Handler) {
    return getSema().ActOnSEHTryBlock(IsCXXTry,TryLoc,TryBlock,Handler);
  }

  StmtResult RebuildSEHExceptStmt(SourceLocation Loc,
                                  Expr *FilterExpr,
                                  Stmt *Block) {
    return getSema().ActOnSEHExceptBlock(Loc,FilterExpr,Block);
  }

  StmtResult RebuildSEHFinallyStmt(SourceLocation Loc,
                                   Stmt *Block) {
    return getSema().ActOnSEHFinallyBlock(Loc,Block);
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
  ExprResult RebuildDeclRefExpr(NestedNameSpecifierLoc QualifierLoc,
                                ValueDecl *VD,
                                const DeclarationNameInfo &NameInfo,
                                TemplateArgumentListInfo *TemplateArgs) {
    CXXScopeSpec SS;
    SS.Adopt(QualifierLoc);

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
                                            CXXScopeSpec &SS,
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
  
  /// \brief Build a new sizeof, alignof or vec_step expression with a 
  /// type argument.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildUnaryExprOrTypeTrait(TypeSourceInfo *TInfo,
                                         SourceLocation OpLoc,
                                         UnaryExprOrTypeTrait ExprKind,
                                         SourceRange R) {
    return getSema().CreateUnaryExprOrTypeTraitExpr(TInfo, OpLoc, ExprKind, R);
  }

  /// \brief Build a new sizeof, alignof or vec step expression with an
  /// expression argument.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildUnaryExprOrTypeTrait(Expr *SubExpr, SourceLocation OpLoc,
                                         UnaryExprOrTypeTrait ExprKind,
                                         SourceRange R) {
    ExprResult Result
      = getSema().CreateUnaryExprOrTypeTraitExpr(SubExpr, ExprKind);
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
                                   SourceLocation RParenLoc,
                                   Expr *ExecConfig = 0) {
    return getSema().ActOnCallExpr(/*Scope=*/0, Callee, LParenLoc,
                                   move(Args), RParenLoc, ExecConfig);
  }

  /// \brief Build a new member access expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildMemberExpr(Expr *Base, SourceLocation OpLoc,
                               bool isArrow,
                               NestedNameSpecifierLoc QualifierLoc,
                               const DeclarationNameInfo &MemberNameInfo,
                               ValueDecl *Member,
                               NamedDecl *FoundDecl,
                        const TemplateArgumentListInfo *ExplicitTemplateArgs,
                               NamedDecl *FirstQualifierInScope) {
    if (!Member->getDeclName()) {
      // We have a reference to an unnamed field.  This is always the
      // base of an anonymous struct/union member access, i.e. the
      // field is always of record type.
      assert(!QualifierLoc && "Can't have an unnamed field with a qualifier!");
      assert(Member->getType()->isRecordType() &&
             "unnamed member not of record type?");

      ExprResult BaseResult =
        getSema().PerformObjectMemberConversion(Base, 
                                                QualifierLoc.getNestedNameSpecifier(),
                                                FoundDecl, Member);
      if (BaseResult.isInvalid())
        return ExprError();
      Base = BaseResult.take();
      ExprValueKind VK = isArrow ? VK_LValue : Base->getValueKind();
      MemberExpr *ME =
        new (getSema().Context) MemberExpr(Base, isArrow,
                                           Member, MemberNameInfo,
                                           cast<FieldDecl>(Member)->getType(),
                                           VK, OK_Ordinary);
      return getSema().Owned(ME);
    }

    CXXScopeSpec SS;
    SS.Adopt(QualifierLoc);

    ExprResult BaseResult = getSema().DefaultFunctionArrayConversion(Base);
    if (BaseResult.isInvalid())
      return ExprError();
    Base = BaseResult.take();
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
                                  SourceLocation LabelLoc, LabelDecl *Label) {
    return getSema().ActOnAddrLabel(AmpAmpLoc, LabelLoc, Label);
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

  /// \brief Build a new generic selection expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildGenericSelectionExpr(SourceLocation KeyLoc,
                                         SourceLocation DefaultLoc,
                                         SourceLocation RParenLoc,
                                         Expr *ControllingExpr,
                                         TypeSourceInfo **Types,
                                         Expr **Exprs,
                                         unsigned NumAssocs) {
    return getSema().CreateGenericSelectionExpr(KeyLoc, DefaultLoc, RParenLoc,
                                                ControllingExpr, Types, Exprs,
                                                NumAssocs);
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

  /// \brief Build a new array type trait expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildArrayTypeTrait(ArrayTypeTrait Trait,
                                   SourceLocation StartLoc,
                                   TypeSourceInfo *TSInfo,
                                   Expr *DimExpr,
                                   SourceLocation RParenLoc) {
    return getSema().BuildArrayTypeTrait(Trait, StartLoc, TSInfo, DimExpr, RParenLoc);
  }

  /// \brief Build a new expression trait expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildExpressionTrait(ExpressionTrait Trait,
                                   SourceLocation StartLoc,
                                   Expr *Queried,
                                   SourceLocation RParenLoc) {
    return getSema().BuildExpressionTrait(Trait, StartLoc, Queried, RParenLoc);
  }

  /// \brief Build a new (previously unresolved) declaration reference
  /// expression.
  ///
  /// By default, performs semantic analysis to build the new expression.
  /// Subclasses may override this routine to provide different behavior.
  ExprResult RebuildDependentScopeDeclRefExpr(
                                          NestedNameSpecifierLoc QualifierLoc,
                                       const DeclarationNameInfo &NameInfo,
                              const TemplateArgumentListInfo *TemplateArgs) {
    CXXScopeSpec SS;
    SS.Adopt(QualifierLoc);

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
                                          NestedNameSpecifierLoc QualifierLoc,
                                            NamedDecl *FirstQualifierInScope,
                                   const DeclarationNameInfo &MemberNameInfo,
                              const TemplateArgumentListInfo *TemplateArgs) {
    CXXScopeSpec SS;
    SS.Adopt(QualifierLoc);

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
                                           NestedNameSpecifierLoc QualifierLoc,
                                               NamedDecl *FirstQualifierInScope,
                                               LookupResult &R,
                                const TemplateArgumentListInfo *TemplateArgs) {
    CXXScopeSpec SS;
    SS.Adopt(QualifierLoc);

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

  /// \brief Build a new expression to compute the length of a parameter pack.
  ExprResult RebuildSizeOfPackExpr(SourceLocation OperatorLoc, NamedDecl *Pack, 
                                   SourceLocation PackLoc, 
                                   SourceLocation RParenLoc,
                                   unsigned Length) {
    return new (SemaRef.Context) SizeOfPackExpr(SemaRef.Context.getSizeType(), 
                                                OperatorLoc, Pack, PackLoc, 
                                                RParenLoc, Length);
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
    ExprResult Base = getSema().Owned(BaseArg);
    LookupResult R(getSema(), Ivar->getDeclName(), IvarLoc,
                   Sema::LookupMemberName);
    ExprResult Result = getSema().LookupMemberExpr(R, Base, IsArrow,
                                                         /*FIME:*/IvarLoc,
                                                         SS, 0,
                                                         false);
    if (Result.isInvalid() || Base.isInvalid())
      return ExprError();
    
    if (Result.get())
      return move(Result);
    
    return getSema().BuildMemberReferenceExpr(Base.get(), Base.get()->getType(),
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
    ExprResult Base = getSema().Owned(BaseArg);
    LookupResult R(getSema(), Property->getDeclName(), PropertyLoc,
                   Sema::LookupMemberName);
    bool IsArrow = false;
    ExprResult Result = getSema().LookupMemberExpr(R, Base, IsArrow,
                                                         /*FIME:*/PropertyLoc,
                                                         SS, 0, false);
    if (Result.isInvalid() || Base.isInvalid())
      return ExprError();
    
    if (Result.get())
      return move(Result);
    
    return getSema().BuildMemberReferenceExpr(Base.get(), Base.get()->getType(),
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
    ExprResult Base = getSema().Owned(BaseArg);
    LookupResult R(getSema(), &getSema().Context.Idents.get("isa"), IsaLoc,
                   Sema::LookupMemberName);
    ExprResult Result = getSema().LookupMemberExpr(R, Base, IsArrow,
                                                         /*FIME:*/IsaLoc,
                                                         SS, 0, false);
    if (Result.isInvalid() || Base.isInvalid())
      return ExprError();
    
    if (Result.get())
      return move(Result);
    
    return getSema().BuildMemberReferenceExpr(Base.get(), Base.get()->getType(),
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
    ExprResult Callee
      = SemaRef.Owned(new (SemaRef.Context) DeclRefExpr(Builtin, Builtin->getType(),
                                                        VK_LValue, BuiltinLoc));
    Callee = SemaRef.UsualUnaryConversions(Callee.take());
    if (Callee.isInvalid())
      return ExprError();

    // Build the CallExpr
    unsigned NumSubExprs = SubExprs.size();
    Expr **Subs = (Expr **)SubExprs.release();
    ExprResult TheCall = SemaRef.Owned(
      new (SemaRef.Context) CallExpr(SemaRef.Context, Callee.take(),
                                                       Subs, NumSubExprs,
                                                   Builtin->getCallResultType(),
                            Expr::getValueKindForType(Builtin->getResultType()),
                                     RParenLoc));

    // Type-check the __builtin_shufflevector expression.
    return SemaRef.SemaBuiltinShuffleVector(cast<CallExpr>(TheCall.take()));
  }

  /// \brief Build a new template argument pack expansion.
  ///
  /// By default, performs semantic analysis to build a new pack expansion
  /// for a template argument. Subclasses may override this routine to provide 
  /// different behavior.
  TemplateArgumentLoc RebuildPackExpansion(TemplateArgumentLoc Pattern,
                                           SourceLocation EllipsisLoc,
                                       llvm::Optional<unsigned> NumExpansions) {
    switch (Pattern.getArgument().getKind()) {
    case TemplateArgument::Expression: {
      ExprResult Result
        = getSema().CheckPackExpansion(Pattern.getSourceExpression(),
                                       EllipsisLoc, NumExpansions);
      if (Result.isInvalid())
        return TemplateArgumentLoc();
          
      return TemplateArgumentLoc(Result.get(), Result.get());
    }
        
    case TemplateArgument::Template:
      return TemplateArgumentLoc(TemplateArgument(
                                          Pattern.getArgument().getAsTemplate(),
                                                  NumExpansions),
                                 Pattern.getTemplateQualifierLoc(),
                                 Pattern.getTemplateNameLoc(),
                                 EllipsisLoc);
        
    case TemplateArgument::Null:
    case TemplateArgument::Integral:
    case TemplateArgument::Declaration:
    case TemplateArgument::Pack:
    case TemplateArgument::TemplateExpansion:
      llvm_unreachable("Pack expansion pattern has no parameter packs");
        
    case TemplateArgument::Type:
      if (TypeSourceInfo *Expansion 
            = getSema().CheckPackExpansion(Pattern.getTypeSourceInfo(),
                                           EllipsisLoc,
                                           NumExpansions))
        return TemplateArgumentLoc(TemplateArgument(Expansion->getType()),
                                   Expansion);
      break;
    }
    
    return TemplateArgumentLoc();
  }
  
  /// \brief Build a new expression pack expansion.
  ///
  /// By default, performs semantic analysis to build a new pack expansion
  /// for an expression. Subclasses may override this routine to provide 
  /// different behavior.
  ExprResult RebuildPackExpansion(Expr *Pattern, SourceLocation EllipsisLoc,
                                  llvm::Optional<unsigned> NumExpansions) {
    return getSema().CheckPackExpansion(Pattern, EllipsisLoc, NumExpansions);
  }
  
private:
  TypeLoc TransformTypeInObjectScope(TypeLoc TL,
                                     QualType ObjectType,
                                     NamedDecl *FirstQualifierInScope,
                                     CXXScopeSpec &SS);

  TypeSourceInfo *TransformTypeInObjectScope(TypeSourceInfo *TSInfo,
                                             QualType ObjectType,
                                             NamedDecl *FirstQualifierInScope,
                                             CXXScopeSpec &SS);
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
#define ABSTRACT_STMT(Node)
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
bool TreeTransform<Derived>::TransformExprs(Expr **Inputs, 
                                            unsigned NumInputs, 
                                            bool IsCall,
                                      llvm::SmallVectorImpl<Expr *> &Outputs,
                                            bool *ArgChanged) {
  for (unsigned I = 0; I != NumInputs; ++I) {
    // If requested, drop call arguments that need to be dropped.
    if (IsCall && getDerived().DropCallArgument(Inputs[I])) {
      if (ArgChanged)
        *ArgChanged = true;
      
      break;
    }
    
    if (PackExpansionExpr *Expansion = dyn_cast<PackExpansionExpr>(Inputs[I])) {
      Expr *Pattern = Expansion->getPattern();
        
      llvm::SmallVector<UnexpandedParameterPack, 2> Unexpanded;
      getSema().collectUnexpandedParameterPacks(Pattern, Unexpanded);
      assert(!Unexpanded.empty() && "Pack expansion without parameter packs?");
      
      // Determine whether the set of unexpanded parameter packs can and should
      // be expanded.
      bool Expand = true;
      bool RetainExpansion = false;
      llvm::Optional<unsigned> OrigNumExpansions
        = Expansion->getNumExpansions();
      llvm::Optional<unsigned> NumExpansions = OrigNumExpansions;
      if (getDerived().TryExpandParameterPacks(Expansion->getEllipsisLoc(),
                                               Pattern->getSourceRange(),
                                               Unexpanded.data(),
                                               Unexpanded.size(),
                                               Expand, RetainExpansion,
                                               NumExpansions))
        return true;
        
      if (!Expand) {
        // The transform has determined that we should perform a simple
        // transformation on the pack expansion, producing another pack 
        // expansion.
        Sema::ArgumentPackSubstitutionIndexRAII SubstIndex(getSema(), -1);
        ExprResult OutPattern = getDerived().TransformExpr(Pattern);
        if (OutPattern.isInvalid())
          return true;
        
        ExprResult Out = getDerived().RebuildPackExpansion(OutPattern.get(), 
                                                Expansion->getEllipsisLoc(),
                                                           NumExpansions);
        if (Out.isInvalid())
          return true;
        
        if (ArgChanged)
          *ArgChanged = true;
        Outputs.push_back(Out.get());
        continue;
      }
      
      // The transform has determined that we should perform an elementwise
      // expansion of the pattern. Do so.
      for (unsigned I = 0; I != *NumExpansions; ++I) {
        Sema::ArgumentPackSubstitutionIndexRAII SubstIndex(getSema(), I);
        ExprResult Out = getDerived().TransformExpr(Pattern);
        if (Out.isInvalid())
          return true;

        if (Out.get()->containsUnexpandedParameterPack()) {
          Out = RebuildPackExpansion(Out.get(), Expansion->getEllipsisLoc(),
                                     OrigNumExpansions);
          if (Out.isInvalid())
            return true;
        }
        
        if (ArgChanged)
          *ArgChanged = true;  
        Outputs.push_back(Out.get());
      }
        
      continue;
    }
    
    ExprResult Result = getDerived().TransformExpr(Inputs[I]);
    if (Result.isInvalid())
      return true;
    
    if (Result.get() != Inputs[I] && ArgChanged)
      *ArgChanged = true;
    
    Outputs.push_back(Result.get());    
  }
  
  return false;
}

template<typename Derived>
NestedNameSpecifierLoc
TreeTransform<Derived>::TransformNestedNameSpecifierLoc(
                                                    NestedNameSpecifierLoc NNS,
                                                     QualType ObjectType,
                                             NamedDecl *FirstQualifierInScope) {
  llvm::SmallVector<NestedNameSpecifierLoc, 4> Qualifiers;
  for (NestedNameSpecifierLoc Qualifier = NNS; Qualifier; 
       Qualifier = Qualifier.getPrefix())
    Qualifiers.push_back(Qualifier);

  CXXScopeSpec SS;
  while (!Qualifiers.empty()) {
    NestedNameSpecifierLoc Q = Qualifiers.pop_back_val();
    NestedNameSpecifier *QNNS = Q.getNestedNameSpecifier();
    
    switch (QNNS->getKind()) {
    case NestedNameSpecifier::Identifier:
      if (SemaRef.BuildCXXNestedNameSpecifier(/*Scope=*/0, 
                                              *QNNS->getAsIdentifier(),
                                              Q.getLocalBeginLoc(), 
                                              Q.getLocalEndLoc(),
                                              ObjectType, false, SS, 
                                              FirstQualifierInScope, false))
        return NestedNameSpecifierLoc();
        
      break;
      
    case NestedNameSpecifier::Namespace: {
      NamespaceDecl *NS
        = cast_or_null<NamespaceDecl>(
                                    getDerived().TransformDecl(
                                                          Q.getLocalBeginLoc(),
                                                       QNNS->getAsNamespace()));
      SS.Extend(SemaRef.Context, NS, Q.getLocalBeginLoc(), Q.getLocalEndLoc());
      break;
    }
      
    case NestedNameSpecifier::NamespaceAlias: {
      NamespaceAliasDecl *Alias
        = cast_or_null<NamespaceAliasDecl>(
                      getDerived().TransformDecl(Q.getLocalBeginLoc(),
                                                 QNNS->getAsNamespaceAlias()));
      SS.Extend(SemaRef.Context, Alias, Q.getLocalBeginLoc(), 
                Q.getLocalEndLoc());
      break;
    }
      
    case NestedNameSpecifier::Global:
      // There is no meaningful transformation that one could perform on the
      // global scope.
      SS.MakeGlobal(SemaRef.Context, Q.getBeginLoc());
      break;
      
    case NestedNameSpecifier::TypeSpecWithTemplate:
    case NestedNameSpecifier::TypeSpec: {
      TypeLoc TL = TransformTypeInObjectScope(Q.getTypeLoc(), ObjectType,
                                              FirstQualifierInScope, SS);
      
      if (!TL)
        return NestedNameSpecifierLoc();
      
      if (TL.getType()->isDependentType() || TL.getType()->isRecordType() ||
          (SemaRef.getLangOptions().CPlusPlus0x && 
           TL.getType()->isEnumeralType())) {
        assert(!TL.getType().hasLocalQualifiers() && 
               "Can't get cv-qualifiers here");
        SS.Extend(SemaRef.Context, /*FIXME:*/SourceLocation(), TL,
                  Q.getLocalEndLoc());
        break;
      }
      // If the nested-name-specifier is an invalid type def, don't emit an
      // error because a previous error should have already been emitted.
      TypedefTypeLoc* TTL = dyn_cast<TypedefTypeLoc>(&TL);
      if (!TTL || !TTL->getTypedefNameDecl()->isInvalidDecl()) {
        SemaRef.Diag(TL.getBeginLoc(), diag::err_nested_name_spec_non_tag) 
          << TL.getType() << SS.getRange();
      }
      return NestedNameSpecifierLoc();
    }
    }
    
    // The qualifier-in-scope and object type only apply to the leftmost entity.
    FirstQualifierInScope = 0;
    ObjectType = QualType();
  }
  
  // Don't rebuild the nested-name-specifier if we don't have to.
  if (SS.getScopeRep() == NNS.getNestedNameSpecifier() && 
      !getDerived().AlwaysRebuild())
    return NNS;
  
  // If we can re-use the source-location data from the original 
  // nested-name-specifier, do so.
  if (SS.location_size() == NNS.getDataLength() &&
      memcmp(SS.location_data(), NNS.getOpaqueData(), SS.location_size()) == 0)
    return NestedNameSpecifierLoc(SS.getScopeRep(), NNS.getOpaqueData());

  // Allocate new nested-name-specifier location information.
  return SS.getWithLocInContext(SemaRef.Context);
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
TreeTransform<Derived>::TransformTemplateName(CXXScopeSpec &SS,
                                              TemplateName Name,
                                              SourceLocation NameLoc,
                                              QualType ObjectType,
                                              NamedDecl *FirstQualifierInScope) {
  if (QualifiedTemplateName *QTN = Name.getAsQualifiedTemplateName()) {
    TemplateDecl *Template = QTN->getTemplateDecl();
    assert(Template && "qualified template name must refer to a template");
    
    TemplateDecl *TransTemplate
      = cast_or_null<TemplateDecl>(getDerived().TransformDecl(NameLoc, 
                                                              Template));
    if (!TransTemplate)
      return TemplateName();
      
    if (!getDerived().AlwaysRebuild() &&
        SS.getScopeRep() == QTN->getQualifier() &&
        TransTemplate == Template)
      return Name;
      
    return getDerived().RebuildTemplateName(SS, QTN->hasTemplateKeyword(),
                                            TransTemplate);
  }
  
  if (DependentTemplateName *DTN = Name.getAsDependentTemplateName()) {
    if (SS.getScopeRep()) {
      // These apply to the scope specifier, not the template.
      ObjectType = QualType();
      FirstQualifierInScope = 0;
    }      
    
    if (!getDerived().AlwaysRebuild() &&
        SS.getScopeRep() == DTN->getQualifier() &&
        ObjectType.isNull())
      return Name;
    
    if (DTN->isIdentifier()) {
      return getDerived().RebuildTemplateName(SS,
                                              *DTN->getIdentifier(), 
                                              NameLoc,
                                              ObjectType,
                                              FirstQualifierInScope);
    }
    
    return getDerived().RebuildTemplateName(SS, DTN->getOperator(), NameLoc,
                                            ObjectType);
  }
  
  if (TemplateDecl *Template = Name.getAsTemplateDecl()) {
    TemplateDecl *TransTemplate
      = cast_or_null<TemplateDecl>(getDerived().TransformDecl(NameLoc, 
                                                              Template));
    if (!TransTemplate)
      return TemplateName();
    
    if (!getDerived().AlwaysRebuild() &&
        TransTemplate == Template)
      return Name;
    
    return TemplateName(TransTemplate);
  }
  
  if (SubstTemplateTemplateParmPackStorage *SubstPack
      = Name.getAsSubstTemplateTemplateParmPack()) {
    TemplateTemplateParmDecl *TransParam
    = cast_or_null<TemplateTemplateParmDecl>(
            getDerived().TransformDecl(NameLoc, SubstPack->getParameterPack()));
    if (!TransParam)
      return TemplateName();
    
    if (!getDerived().AlwaysRebuild() &&
        TransParam == SubstPack->getParameterPack())
      return Name;
    
    return getDerived().RebuildTemplateName(TransParam, 
                                            SubstPack->getArgumentPack());
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
  case TemplateArgument::TemplateExpansion: {
    NestedNameSpecifierLocBuilder Builder;
    TemplateName Template = Arg.getAsTemplate();
    if (DependentTemplateName *DTN = Template.getAsDependentTemplateName())
      Builder.MakeTrivial(SemaRef.Context, DTN->getQualifier(), Loc);
    else if (QualifiedTemplateName *QTN = Template.getAsQualifiedTemplateName())
      Builder.MakeTrivial(SemaRef.Context, QTN->getQualifier(), Loc);
        
    if (Arg.getKind() == TemplateArgument::Template)
      Output = TemplateArgumentLoc(Arg, 
                                   Builder.getWithLocInContext(SemaRef.Context),
                                   Loc);
    else
      Output = TemplateArgumentLoc(Arg, 
                                   Builder.getWithLocInContext(SemaRef.Context),
                                   Loc, Loc);
    
    break;
  }

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
    NestedNameSpecifierLoc QualifierLoc = Input.getTemplateQualifierLoc();
    if (QualifierLoc) {
      QualifierLoc = getDerived().TransformNestedNameSpecifierLoc(QualifierLoc);
      if (!QualifierLoc)
        return true;
    }
    
    CXXScopeSpec SS;
    SS.Adopt(QualifierLoc);
    TemplateName Template
      = getDerived().TransformTemplateName(SS, Arg.getAsTemplate(),
                                           Input.getTemplateNameLoc());
    if (Template.isNull())
      return true;
    
    Output = TemplateArgumentLoc(TemplateArgument(Template), QualifierLoc,
                                 Input.getTemplateNameLoc());
    return false;
  }

  case TemplateArgument::TemplateExpansion:
    llvm_unreachable("Caller should expand pack expansions");

  case TemplateArgument::Expression: {
    // Template argument expressions are not potentially evaluated.
    EnterExpressionEvaluationContext Unevaluated(getSema(),
                                                 Sema::Unevaluated);

    Expr *InputExpr = Input.getSourceExpression();
    if (!InputExpr) InputExpr = Input.getArgument().getAsExpr();

    ExprResult E = getDerived().TransformExpr(InputExpr);
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

/// \brief Iterator adaptor that invents template argument location information
/// for each of the template arguments in its underlying iterator.
template<typename Derived, typename InputIterator>
class TemplateArgumentLocInventIterator {
  TreeTransform<Derived> &Self;
  InputIterator Iter;
  
public:
  typedef TemplateArgumentLoc value_type;
  typedef TemplateArgumentLoc reference;
  typedef typename std::iterator_traits<InputIterator>::difference_type
    difference_type;
  typedef std::input_iterator_tag iterator_category;
  
  class pointer {
    TemplateArgumentLoc Arg;
    
  public:
    explicit pointer(TemplateArgumentLoc Arg) : Arg(Arg) { }
    
    const TemplateArgumentLoc *operator->() const { return &Arg; }
  };
  
  TemplateArgumentLocInventIterator() { }
  
  explicit TemplateArgumentLocInventIterator(TreeTransform<Derived> &Self,
                                             InputIterator Iter)
    : Self(Self), Iter(Iter) { }
  
  TemplateArgumentLocInventIterator &operator++() {
    ++Iter;
    return *this;
  }
  
  TemplateArgumentLocInventIterator operator++(int) {
    TemplateArgumentLocInventIterator Old(*this);
    ++(*this);
    return Old;
  }
  
  reference operator*() const {
    TemplateArgumentLoc Result;
    Self.InventTemplateArgumentLoc(*Iter, Result);
    return Result;
  }
  
  pointer operator->() const { return pointer(**this); }
  
  friend bool operator==(const TemplateArgumentLocInventIterator &X,
                         const TemplateArgumentLocInventIterator &Y) {
    return X.Iter == Y.Iter;
  }

  friend bool operator!=(const TemplateArgumentLocInventIterator &X,
                         const TemplateArgumentLocInventIterator &Y) {
    return X.Iter != Y.Iter;
  }
};
  
template<typename Derived>
template<typename InputIterator>
bool TreeTransform<Derived>::TransformTemplateArguments(InputIterator First,
                                                        InputIterator Last,
                                            TemplateArgumentListInfo &Outputs) {
  for (; First != Last; ++First) {
    TemplateArgumentLoc Out;
    TemplateArgumentLoc In = *First;
    
    if (In.getArgument().getKind() == TemplateArgument::Pack) {
      // Unpack argument packs, which we translate them into separate
      // arguments.
      // FIXME: We could do much better if we could guarantee that the
      // TemplateArgumentLocInfo for the pack expansion would be usable for
      // all of the template arguments in the argument pack.
      typedef TemplateArgumentLocInventIterator<Derived, 
                                                TemplateArgument::pack_iterator>
        PackLocIterator;
      if (TransformTemplateArguments(PackLocIterator(*this, 
                                                 In.getArgument().pack_begin()),
                                     PackLocIterator(*this,
                                                   In.getArgument().pack_end()),
                                     Outputs))
        return true;
      
      continue;
    }
    
    if (In.getArgument().isPackExpansion()) {
      // We have a pack expansion, for which we will be substituting into
      // the pattern.
      SourceLocation Ellipsis;
      llvm::Optional<unsigned> OrigNumExpansions;
      TemplateArgumentLoc Pattern
        = In.getPackExpansionPattern(Ellipsis, OrigNumExpansions, 
                                     getSema().Context);
      
      llvm::SmallVector<UnexpandedParameterPack, 2> Unexpanded;
      getSema().collectUnexpandedParameterPacks(Pattern, Unexpanded);
      assert(!Unexpanded.empty() && "Pack expansion without parameter packs?");
      
      // Determine whether the set of unexpanded parameter packs can and should
      // be expanded.
      bool Expand = true;
      bool RetainExpansion = false;
      llvm::Optional<unsigned> NumExpansions = OrigNumExpansions;
      if (getDerived().TryExpandParameterPacks(Ellipsis,
                                               Pattern.getSourceRange(),
                                               Unexpanded.data(),
                                               Unexpanded.size(),
                                               Expand, 
                                               RetainExpansion,
                                               NumExpansions))
        return true;
      
      if (!Expand) {
        // The transform has determined that we should perform a simple
        // transformation on the pack expansion, producing another pack 
        // expansion.
        TemplateArgumentLoc OutPattern;
        Sema::ArgumentPackSubstitutionIndexRAII SubstIndex(getSema(), -1);
        if (getDerived().TransformTemplateArgument(Pattern, OutPattern))
          return true;
                
        Out = getDerived().RebuildPackExpansion(OutPattern, Ellipsis,
                                                NumExpansions);
        if (Out.getArgument().isNull())
          return true;
        
        Outputs.addArgument(Out);
        continue;
      }
      
      // The transform has determined that we should perform an elementwise
      // expansion of the pattern. Do so.
      for (unsigned I = 0; I != *NumExpansions; ++I) {
        Sema::ArgumentPackSubstitutionIndexRAII SubstIndex(getSema(), I);

        if (getDerived().TransformTemplateArgument(Pattern, Out))
          return true;
        
        if (Out.getArgument().containsUnexpandedParameterPack()) {
          Out = getDerived().RebuildPackExpansion(Out, Ellipsis,
                                                  OrigNumExpansions);
          if (Out.getArgument().isNull())
            return true;
        }
          
        Outputs.addArgument(Out);
      }
      
      // If we're supposed to retain a pack expansion, do so by temporarily
      // forgetting the partially-substituted parameter pack.
      if (RetainExpansion) {
        ForgetPartiallySubstitutedPackRAII Forget(getDerived());
        
        if (getDerived().TransformTemplateArgument(Pattern, Out))
          return true;
        
        Out = getDerived().RebuildPackExpansion(Out, Ellipsis,
                                                OrigNumExpansions);
        if (Out.getArgument().isNull())
          return true;
        
        Outputs.addArgument(Out);
      }
      
      continue;
    }
    
    // The simple case: 
    if (getDerived().TransformTemplateArgument(In, Out))
      return true;
    
    Outputs.addArgument(Out);
  }
  
  return false;

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
  TypeSourceInfo *DI = getSema().Context.getTrivialTypeSourceInfo(T,
                                                getDerived().getBaseLocation());
  
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

template<typename Derived>
TypeLoc
TreeTransform<Derived>::TransformTypeInObjectScope(TypeLoc TL,
                                                   QualType ObjectType,
                                                   NamedDecl *UnqualLookup,
                                                   CXXScopeSpec &SS) {
  QualType T = TL.getType();
  if (getDerived().AlreadyTransformed(T))
    return TL;
  
  TypeLocBuilder TLB;
  QualType Result;
  
  if (isa<TemplateSpecializationType>(T)) {
    TemplateSpecializationTypeLoc SpecTL
      = cast<TemplateSpecializationTypeLoc>(TL);
    
    TemplateName Template =
      getDerived().TransformTemplateName(SS,
                                         SpecTL.getTypePtr()->getTemplateName(),
                                         SpecTL.getTemplateNameLoc(),
                                         ObjectType, UnqualLookup);
    if (Template.isNull()) 
      return TypeLoc();
    
    Result = getDerived().TransformTemplateSpecializationType(TLB, SpecTL, 
                                                              Template);
  } else if (isa<DependentTemplateSpecializationType>(T)) {
    DependentTemplateSpecializationTypeLoc SpecTL
      = cast<DependentTemplateSpecializationTypeLoc>(TL);
    
    TemplateName Template
      = getDerived().RebuildTemplateName(SS, 
                                         *SpecTL.getTypePtr()->getIdentifier(), 
                                         SpecTL.getNameLoc(),
                                         ObjectType, UnqualLookup);
    if (Template.isNull())
      return TypeLoc();
    
    Result = getDerived().TransformDependentTemplateSpecializationType(TLB, 
                                                                       SpecTL,
                                                                     Template,
                                                                       SS);
  } else {
    // Nothing special needs to be done for these.
    Result = getDerived().TransformType(TLB, TL);
  }
  
  if (Result.isNull()) 
    return TypeLoc();
  
  return TLB.getTypeSourceInfo(SemaRef.Context, Result)->getTypeLoc();
}

template<typename Derived>
TypeSourceInfo *
TreeTransform<Derived>::TransformTypeInObjectScope(TypeSourceInfo *TSInfo,
                                                   QualType ObjectType,
                                                   NamedDecl *UnqualLookup,
                                                   CXXScopeSpec &SS) {
  // FIXME: Painfully copy-paste from the above!
  
  QualType T = TSInfo->getType();
  if (getDerived().AlreadyTransformed(T))
    return TSInfo;
  
  TypeLocBuilder TLB;
  QualType Result;
  
  TypeLoc TL = TSInfo->getTypeLoc();
  if (isa<TemplateSpecializationType>(T)) {
    TemplateSpecializationTypeLoc SpecTL
      = cast<TemplateSpecializationTypeLoc>(TL);
    
    TemplateName Template
    = getDerived().TransformTemplateName(SS,
                                         SpecTL.getTypePtr()->getTemplateName(),
                                         SpecTL.getTemplateNameLoc(),
                                         ObjectType, UnqualLookup);
    if (Template.isNull()) 
      return 0;
    
    Result = getDerived().TransformTemplateSpecializationType(TLB, SpecTL, 
                                                              Template);
  } else if (isa<DependentTemplateSpecializationType>(T)) {
    DependentTemplateSpecializationTypeLoc SpecTL
      = cast<DependentTemplateSpecializationTypeLoc>(TL);
    
    TemplateName Template
      = getDerived().RebuildTemplateName(SS, 
                                         *SpecTL.getTypePtr()->getIdentifier(), 
                                         SpecTL.getNameLoc(),
                                         ObjectType, UnqualLookup);
    if (Template.isNull())
      return 0;
    
    Result = getDerived().TransformDependentTemplateSpecializationType(TLB, 
                                                                       SpecTL,
                                                                       Template,
                                                                       SS);
  } else {
    // Nothing special needs to be done for these.
    Result = getDerived().TransformType(TLB, TL);
  }
  
  if (Result.isNull()) 
    return 0;
  
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
  QualType PointeeType = getDerived().TransformType(TLB, TL.getPointeeLoc());
  if (PointeeType.isNull())
    return QualType();

  TypeSourceInfo* OldClsTInfo = TL.getClassTInfo();
  TypeSourceInfo* NewClsTInfo = 0;
  if (OldClsTInfo) {
    NewClsTInfo = getDerived().TransformType(OldClsTInfo);
    if (!NewClsTInfo)
      return QualType();
  }

  const MemberPointerType *T = TL.getTypePtr();
  QualType OldClsType = QualType(T->getClass(), 0);
  QualType NewClsType;
  if (NewClsTInfo)
    NewClsType = NewClsTInfo->getType();
  else {
    NewClsType = getDerived().TransformType(OldClsType);
    if (NewClsType.isNull())
      return QualType();
  }

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      PointeeType != T->getPointeeType() ||
      NewClsType != OldClsType) {
    Result = getDerived().RebuildMemberPointerType(PointeeType, NewClsType,
                                                   TL.getStarLoc());
    if (Result.isNull())
      return QualType();
  }

  MemberPointerTypeLoc NewTL = TLB.push<MemberPointerTypeLoc>(Result);
  NewTL.setSigilLoc(TL.getSigilLoc());
  NewTL.setClassTInfo(NewClsTInfo);

  return Result;
}

template<typename Derived>
QualType
TreeTransform<Derived>::TransformConstantArrayType(TypeLocBuilder &TLB,
                                                   ConstantArrayTypeLoc TL) {
  const ConstantArrayType *T = TL.getTypePtr();
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
  const IncompleteArrayType *T = TL.getTypePtr();
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
  const VariableArrayType *T = TL.getTypePtr();
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
  const DependentSizedArrayType *T = TL.getTypePtr();
  QualType ElementType = getDerived().TransformType(TLB, TL.getElementLoc());
  if (ElementType.isNull())
    return QualType();

  // Array bounds are not potentially evaluated contexts
  EnterExpressionEvaluationContext Unevaluated(SemaRef, Sema::Unevaluated);

  // Prefer the expression from the TypeLoc;  the other may have been uniqued.
  Expr *origSize = TL.getSizeExpr();
  if (!origSize) origSize = T->getSizeExpr();

  ExprResult sizeResult
    = getDerived().TransformExpr(origSize);
  if (sizeResult.isInvalid())
    return QualType();

  Expr *size = sizeResult.get();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      ElementType != T->getElementType() ||
      size != origSize) {
    Result = getDerived().RebuildDependentSizedArrayType(ElementType,
                                                         T->getSizeModifier(),
                                                         size,
                                                T->getIndexTypeCVRQualifiers(),
                                                        TL.getBracketsRange());
    if (Result.isNull())
      return QualType();
  }

  // We might have any sort of array type now, but fortunately they
  // all have the same location layout.
  ArrayTypeLoc NewTL = TLB.push<ArrayTypeLoc>(Result);
  NewTL.setLBracketLoc(TL.getLBracketLoc());
  NewTL.setRBracketLoc(TL.getRBracketLoc());
  NewTL.setSizeExpr(size);

  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformDependentSizedExtVectorType(
                                      TypeLocBuilder &TLB,
                                      DependentSizedExtVectorTypeLoc TL) {
  const DependentSizedExtVectorType *T = TL.getTypePtr();

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
  const VectorType *T = TL.getTypePtr();
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
  const VectorType *T = TL.getTypePtr();
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
TreeTransform<Derived>::TransformFunctionTypeParam(ParmVarDecl *OldParm,
                                                   int indexAdjustment,
                                       llvm::Optional<unsigned> NumExpansions) {
  TypeSourceInfo *OldDI = OldParm->getTypeSourceInfo();
  TypeSourceInfo *NewDI = 0;
  
  if (NumExpansions && isa<PackExpansionType>(OldDI->getType())) {
    // If we're substituting into a pack expansion type and we know the 
    TypeLoc OldTL = OldDI->getTypeLoc();
    PackExpansionTypeLoc OldExpansionTL = cast<PackExpansionTypeLoc>(OldTL);
    
    TypeLocBuilder TLB;
    TypeLoc NewTL = OldDI->getTypeLoc();
    TLB.reserve(NewTL.getFullDataSize());
    
    QualType Result = getDerived().TransformType(TLB, 
                                               OldExpansionTL.getPatternLoc());
    if (Result.isNull())
      return 0;
   
    Result = RebuildPackExpansionType(Result, 
                                OldExpansionTL.getPatternLoc().getSourceRange(), 
                                      OldExpansionTL.getEllipsisLoc(),
                                      NumExpansions);
    if (Result.isNull())
      return 0;
    
    PackExpansionTypeLoc NewExpansionTL
      = TLB.push<PackExpansionTypeLoc>(Result);
    NewExpansionTL.setEllipsisLoc(OldExpansionTL.getEllipsisLoc());
    NewDI = TLB.getTypeSourceInfo(SemaRef.Context, Result);
  } else
    NewDI = getDerived().TransformType(OldDI);
  if (!NewDI)
    return 0;

  if (NewDI == OldDI && indexAdjustment == 0)
    return OldParm;

  ParmVarDecl *newParm = ParmVarDecl::Create(SemaRef.Context,
                                             OldParm->getDeclContext(),
                                             OldParm->getInnerLocStart(),
                                             OldParm->getLocation(),
                                             OldParm->getIdentifier(),
                                             NewDI->getType(),
                                             NewDI,
                                             OldParm->getStorageClass(),
                                             OldParm->getStorageClassAsWritten(),
                                             /* DefArg */ NULL);
  newParm->setScopeInfo(OldParm->getFunctionScopeDepth(),
                        OldParm->getFunctionScopeIndex() + indexAdjustment);
  return newParm;
}

template<typename Derived>
bool TreeTransform<Derived>::
  TransformFunctionTypeParams(SourceLocation Loc,
                              ParmVarDecl **Params, unsigned NumParams,
                              const QualType *ParamTypes,
                              llvm::SmallVectorImpl<QualType> &OutParamTypes,
                              llvm::SmallVectorImpl<ParmVarDecl*> *PVars) {
  int indexAdjustment = 0;

  for (unsigned i = 0; i != NumParams; ++i) {
    if (ParmVarDecl *OldParm = Params[i]) {
      assert(OldParm->getFunctionScopeIndex() == i);

      llvm::Optional<unsigned> NumExpansions;
      ParmVarDecl *NewParm = 0;
      if (OldParm->isParameterPack()) {
        // We have a function parameter pack that may need to be expanded.
        llvm::SmallVector<UnexpandedParameterPack, 2> Unexpanded;

        // Find the parameter packs that could be expanded.
        TypeLoc TL = OldParm->getTypeSourceInfo()->getTypeLoc();
        PackExpansionTypeLoc ExpansionTL = cast<PackExpansionTypeLoc>(TL);
        TypeLoc Pattern = ExpansionTL.getPatternLoc();
        SemaRef.collectUnexpandedParameterPacks(Pattern, Unexpanded);
        assert(Unexpanded.size() > 0 && "Could not find parameter packs!");

        // Determine whether we should expand the parameter packs.
        bool ShouldExpand = false;
        bool RetainExpansion = false;
        llvm::Optional<unsigned> OrigNumExpansions
          = ExpansionTL.getTypePtr()->getNumExpansions();
        NumExpansions = OrigNumExpansions;
        if (getDerived().TryExpandParameterPacks(ExpansionTL.getEllipsisLoc(),
                                                 Pattern.getSourceRange(),
                                                 Unexpanded.data(), 
                                                 Unexpanded.size(),
                                                 ShouldExpand, 
                                                 RetainExpansion,
                                                 NumExpansions)) {
          return true;
        }
        
        if (ShouldExpand) {
          // Expand the function parameter pack into multiple, separate
          // parameters.
          getDerived().ExpandingFunctionParameterPack(OldParm);
          for (unsigned I = 0; I != *NumExpansions; ++I) {
            Sema::ArgumentPackSubstitutionIndexRAII SubstIndex(getSema(), I);
            ParmVarDecl *NewParm 
              = getDerived().TransformFunctionTypeParam(OldParm,
                                                        indexAdjustment++,
                                                        OrigNumExpansions);
            if (!NewParm)
              return true;
            
            OutParamTypes.push_back(NewParm->getType());
            if (PVars)
              PVars->push_back(NewParm);
          }

          // If we're supposed to retain a pack expansion, do so by temporarily
          // forgetting the partially-substituted parameter pack.
          if (RetainExpansion) {
            ForgetPartiallySubstitutedPackRAII Forget(getDerived());
            ParmVarDecl *NewParm 
              = getDerived().TransformFunctionTypeParam(OldParm,
                                                        indexAdjustment++,
                                                        OrigNumExpansions);
            if (!NewParm)
              return true;
            
            OutParamTypes.push_back(NewParm->getType());
            if (PVars)
              PVars->push_back(NewParm);
          }

          // The next parameter should have the same adjustment as the
          // last thing we pushed, but we post-incremented indexAdjustment
          // on every push.  Also, if we push nothing, the adjustment should
          // go down by one.
          indexAdjustment--;

          // We're done with the pack expansion.
          continue;
        }
        
        // We'll substitute the parameter now without expanding the pack 
        // expansion.
        Sema::ArgumentPackSubstitutionIndexRAII SubstIndex(getSema(), -1);
        NewParm = getDerived().TransformFunctionTypeParam(OldParm,
                                                          indexAdjustment,
                                                          NumExpansions);
      } else {
        NewParm = getDerived().TransformFunctionTypeParam(OldParm,
                                                          indexAdjustment,
                                                  llvm::Optional<unsigned>());
      }

      if (!NewParm)
        return true;
      
      OutParamTypes.push_back(NewParm->getType());
      if (PVars)
        PVars->push_back(NewParm);
      continue;
    }

    // Deal with the possibility that we don't have a parameter
    // declaration for this parameter.
    QualType OldType = ParamTypes[i];
    bool IsPackExpansion = false;
    llvm::Optional<unsigned> NumExpansions;
    QualType NewType;
    if (const PackExpansionType *Expansion 
                                       = dyn_cast<PackExpansionType>(OldType)) {
      // We have a function parameter pack that may need to be expanded.
      QualType Pattern = Expansion->getPattern();
      llvm::SmallVector<UnexpandedParameterPack, 2> Unexpanded;
      getSema().collectUnexpandedParameterPacks(Pattern, Unexpanded);
      
      // Determine whether we should expand the parameter packs.
      bool ShouldExpand = false;
      bool RetainExpansion = false;
      if (getDerived().TryExpandParameterPacks(Loc, SourceRange(),
                                               Unexpanded.data(), 
                                               Unexpanded.size(),
                                               ShouldExpand, 
                                               RetainExpansion,
                                               NumExpansions)) {
        return true;
      }
      
      if (ShouldExpand) {
        // Expand the function parameter pack into multiple, separate 
        // parameters.
        for (unsigned I = 0; I != *NumExpansions; ++I) {
          Sema::ArgumentPackSubstitutionIndexRAII SubstIndex(getSema(), I);
          QualType NewType = getDerived().TransformType(Pattern);
          if (NewType.isNull())
            return true;

          OutParamTypes.push_back(NewType);
          if (PVars)
            PVars->push_back(0);
        }
        
        // We're done with the pack expansion.
        continue;
      }
      
      // If we're supposed to retain a pack expansion, do so by temporarily
      // forgetting the partially-substituted parameter pack.
      if (RetainExpansion) {
        ForgetPartiallySubstitutedPackRAII Forget(getDerived());
        QualType NewType = getDerived().TransformType(Pattern);
        if (NewType.isNull())
          return true;
        
        OutParamTypes.push_back(NewType);
        if (PVars)
          PVars->push_back(0);
      }

      // We'll substitute the parameter now without expanding the pack 
      // expansion.
      OldType = Expansion->getPattern();
      IsPackExpansion = true;
      Sema::ArgumentPackSubstitutionIndexRAII SubstIndex(getSema(), -1);
      NewType = getDerived().TransformType(OldType);
    } else {
      NewType = getDerived().TransformType(OldType);
    }
    
    if (NewType.isNull())
      return true;

    if (IsPackExpansion)
      NewType = getSema().Context.getPackExpansionType(NewType,
                                                       NumExpansions);
      
    OutParamTypes.push_back(NewType);
    if (PVars)
      PVars->push_back(0);
  }

#ifndef NDEBUG
  if (PVars) {
    for (unsigned i = 0, e = PVars->size(); i != e; ++i)
      if (ParmVarDecl *parm = (*PVars)[i])
        assert(parm->getFunctionScopeIndex() == i);
  }
#endif

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
  const FunctionProtoType *T = TL.getTypePtr();

  QualType ResultType;

  if (TL.getTrailingReturn()) {
    if (getDerived().TransformFunctionTypeParams(TL.getBeginLoc(), 
                                                 TL.getParmArray(),
                                                 TL.getNumArgs(),
                                             TL.getTypePtr()->arg_type_begin(),                                                
                                                 ParamTypes, &ParamDecls))
      return QualType();

    ResultType = getDerived().TransformType(TLB, TL.getResultLoc());
    if (ResultType.isNull())
      return QualType();
  }
  else {
    ResultType = getDerived().TransformType(TLB, TL.getResultLoc());
    if (ResultType.isNull())
      return QualType();

    if (getDerived().TransformFunctionTypeParams(TL.getBeginLoc(), 
                                                 TL.getParmArray(),
                                                 TL.getNumArgs(),
                                             TL.getTypePtr()->arg_type_begin(),                                                
                                                 ParamTypes, &ParamDecls))
      return QualType();
  }

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      ResultType != T->getResultType() ||
      T->getNumArgs() != ParamTypes.size() ||
      !std::equal(T->arg_type_begin(), T->arg_type_end(), ParamTypes.begin())) {
    Result = getDerived().RebuildFunctionProtoType(ResultType,
                                                   ParamTypes.data(),
                                                   ParamTypes.size(),
                                                   T->isVariadic(),
                                                   T->getTypeQuals(),
                                                   T->getRefQualifier(),
                                                   T->getExtInfo());
    if (Result.isNull())
      return QualType();
  }

  FunctionProtoTypeLoc NewTL = TLB.push<FunctionProtoTypeLoc>(Result);
  NewTL.setLocalRangeBegin(TL.getLocalRangeBegin());
  NewTL.setLocalRangeEnd(TL.getLocalRangeEnd());
  NewTL.setTrailingReturn(TL.getTrailingReturn());
  for (unsigned i = 0, e = NewTL.getNumArgs(); i != e; ++i)
    NewTL.setArg(i, ParamDecls[i]);

  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformFunctionNoProtoType(
                                                 TypeLocBuilder &TLB,
                                                 FunctionNoProtoTypeLoc TL) {
  const FunctionNoProtoType *T = TL.getTypePtr();
  QualType ResultType = getDerived().TransformType(TLB, TL.getResultLoc());
  if (ResultType.isNull())
    return QualType();

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      ResultType != T->getResultType())
    Result = getDerived().RebuildFunctionNoProtoType(ResultType);

  FunctionNoProtoTypeLoc NewTL = TLB.push<FunctionNoProtoTypeLoc>(Result);
  NewTL.setLocalRangeBegin(TL.getLocalRangeBegin());
  NewTL.setLocalRangeEnd(TL.getLocalRangeEnd());
  NewTL.setTrailingReturn(false);

  return Result;
}

template<typename Derived> QualType
TreeTransform<Derived>::TransformUnresolvedUsingType(TypeLocBuilder &TLB,
                                                 UnresolvedUsingTypeLoc TL) {
  const UnresolvedUsingType *T = TL.getTypePtr();
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
  const TypedefType *T = TL.getTypePtr();
  TypedefNameDecl *Typedef
    = cast_or_null<TypedefNameDecl>(getDerived().TransformDecl(TL.getNameLoc(),
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
  const DecltypeType *T = TL.getTypePtr();

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
QualType TreeTransform<Derived>::TransformUnaryTransformType(
                                                            TypeLocBuilder &TLB,
                                                     UnaryTransformTypeLoc TL) {
  QualType Result = TL.getType();
  if (Result->isDependentType()) {
    const UnaryTransformType *T = TL.getTypePtr();
    QualType NewBase =
      getDerived().TransformType(TL.getUnderlyingTInfo())->getType();
    Result = getDerived().RebuildUnaryTransformType(NewBase,
                                                    T->getUTTKind(),
                                                    TL.getKWLoc());
    if (Result.isNull())
      return QualType();
  }

  UnaryTransformTypeLoc NewTL = TLB.push<UnaryTransformTypeLoc>(Result);
  NewTL.setKWLoc(TL.getKWLoc());
  NewTL.setParensRange(TL.getParensRange());
  NewTL.setUnderlyingTInfo(TL.getUnderlyingTInfo());
  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformAutoType(TypeLocBuilder &TLB,
                                                   AutoTypeLoc TL) {
  const AutoType *T = TL.getTypePtr();
  QualType OldDeduced = T->getDeducedType();
  QualType NewDeduced;
  if (!OldDeduced.isNull()) {
    NewDeduced = getDerived().TransformType(OldDeduced);
    if (NewDeduced.isNull())
      return QualType();
  }

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() || NewDeduced != OldDeduced) {
    Result = getDerived().RebuildAutoType(NewDeduced);
    if (Result.isNull())
      return QualType();
  }

  AutoTypeLoc NewTL = TLB.push<AutoTypeLoc>(Result);
  NewTL.setNameLoc(TL.getNameLoc());

  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformRecordType(TypeLocBuilder &TLB,
                                                     RecordTypeLoc TL) {
  const RecordType *T = TL.getTypePtr();
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
  const EnumType *T = TL.getTypePtr();
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
  const SubstTemplateTypeParmType *T = TL.getTypePtr();
  
  // Substitute into the replacement type, which itself might involve something
  // that needs to be transformed. This only tends to occur with default
  // template arguments of template template parameters.
  TemporaryBase Rebase(*this, TL.getNameLoc(), DeclarationName());
  QualType Replacement = getDerived().TransformType(T->getReplacementType());
  if (Replacement.isNull())
    return QualType();
  
  // Always canonicalize the replacement type.
  Replacement = SemaRef.Context.getCanonicalType(Replacement);
  QualType Result
    = SemaRef.Context.getSubstTemplateTypeParmType(T->getReplacedParameter(), 
                                                   Replacement);
  
  // Propagate type-source information.
  SubstTemplateTypeParmTypeLoc NewTL
    = TLB.push<SubstTemplateTypeParmTypeLoc>(Result);
  NewTL.setNameLoc(TL.getNameLoc());
  return Result;

}

template<typename Derived>
QualType TreeTransform<Derived>::TransformSubstTemplateTypeParmPackType(
                                          TypeLocBuilder &TLB,
                                          SubstTemplateTypeParmPackTypeLoc TL) {
  return TransformTypeSpecType(TLB, TL);
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformTemplateSpecializationType(
                                                        TypeLocBuilder &TLB,
                                           TemplateSpecializationTypeLoc TL) {
  const TemplateSpecializationType *T = TL.getTypePtr();

  // The nested-name-specifier never matters in a TemplateSpecializationType,
  // because we can't have a dependent nested-name-specifier anyway.
  CXXScopeSpec SS;
  TemplateName Template
    = getDerived().TransformTemplateName(SS, T->getTemplateName(),
                                         TL.getTemplateNameLoc());
  if (Template.isNull())
    return QualType();

  return getDerived().TransformTemplateSpecializationType(TLB, TL, Template);
}

namespace {
  /// \brief Simple iterator that traverses the template arguments in a 
  /// container that provides a \c getArgLoc() member function.
  ///
  /// This iterator is intended to be used with the iterator form of
  /// \c TreeTransform<Derived>::TransformTemplateArguments().
  template<typename ArgLocContainer>
  class TemplateArgumentLocContainerIterator {
    ArgLocContainer *Container;
    unsigned Index;
    
  public:
    typedef TemplateArgumentLoc value_type;
    typedef TemplateArgumentLoc reference;
    typedef int difference_type;
    typedef std::input_iterator_tag iterator_category;
    
    class pointer {
      TemplateArgumentLoc Arg;
      
    public:
      explicit pointer(TemplateArgumentLoc Arg) : Arg(Arg) { }
      
      const TemplateArgumentLoc *operator->() const {
        return &Arg;
      }
    };
    
    
    TemplateArgumentLocContainerIterator() {}
    
    TemplateArgumentLocContainerIterator(ArgLocContainer &Container,
                                 unsigned Index)
      : Container(&Container), Index(Index) { }
    
    TemplateArgumentLocContainerIterator &operator++() {
      ++Index;
      return *this;
    }
    
    TemplateArgumentLocContainerIterator operator++(int) {
      TemplateArgumentLocContainerIterator Old(*this);
      ++(*this);
      return Old;
    }
    
    TemplateArgumentLoc operator*() const {
      return Container->getArgLoc(Index);
    }
    
    pointer operator->() const {
      return pointer(Container->getArgLoc(Index));
    }
    
    friend bool operator==(const TemplateArgumentLocContainerIterator &X,
                           const TemplateArgumentLocContainerIterator &Y) {
      return X.Container == Y.Container && X.Index == Y.Index;
    }
    
    friend bool operator!=(const TemplateArgumentLocContainerIterator &X,
                           const TemplateArgumentLocContainerIterator &Y) {
      return !(X == Y);
    }
  };
}
  
  
template <typename Derived>
QualType TreeTransform<Derived>::TransformTemplateSpecializationType(
                                                        TypeLocBuilder &TLB,
                                           TemplateSpecializationTypeLoc TL,
                                                      TemplateName Template) {
  TemplateArgumentListInfo NewTemplateArgs;
  NewTemplateArgs.setLAngleLoc(TL.getLAngleLoc());
  NewTemplateArgs.setRAngleLoc(TL.getRAngleLoc());
  typedef TemplateArgumentLocContainerIterator<TemplateSpecializationTypeLoc>
    ArgIterator;
  if (getDerived().TransformTemplateArguments(ArgIterator(TL, 0), 
                                              ArgIterator(TL, TL.getNumArgs()),
                                              NewTemplateArgs))
    return QualType();

  // FIXME: maybe don't rebuild if all the template arguments are the same.

  QualType Result =
    getDerived().RebuildTemplateSpecializationType(Template,
                                                   TL.getTemplateNameLoc(),
                                                   NewTemplateArgs);

  if (!Result.isNull()) {
    // Specializations of template template parameters are represented as
    // TemplateSpecializationTypes, and substitution of type alias templates
    // within a dependent context can transform them into
    // DependentTemplateSpecializationTypes.
    if (isa<DependentTemplateSpecializationType>(Result)) {
      DependentTemplateSpecializationTypeLoc NewTL
        = TLB.push<DependentTemplateSpecializationTypeLoc>(Result);
      NewTL.setKeywordLoc(TL.getTemplateNameLoc());
      NewTL.setQualifierLoc(NestedNameSpecifierLoc());
      NewTL.setNameLoc(TL.getTemplateNameLoc());
      NewTL.setLAngleLoc(TL.getLAngleLoc());
      NewTL.setRAngleLoc(TL.getRAngleLoc());
      for (unsigned i = 0, e = NewTemplateArgs.size(); i != e; ++i)
        NewTL.setArgLocInfo(i, NewTemplateArgs[i].getLocInfo());
      return Result;
    }

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

template <typename Derived>
QualType TreeTransform<Derived>::TransformDependentTemplateSpecializationType(
                                     TypeLocBuilder &TLB,
                                     DependentTemplateSpecializationTypeLoc TL,
                                     TemplateName Template,
                                     CXXScopeSpec &SS) {
  TemplateArgumentListInfo NewTemplateArgs;
  NewTemplateArgs.setLAngleLoc(TL.getLAngleLoc());
  NewTemplateArgs.setRAngleLoc(TL.getRAngleLoc());
  typedef TemplateArgumentLocContainerIterator<
            DependentTemplateSpecializationTypeLoc> ArgIterator;
  if (getDerived().TransformTemplateArguments(ArgIterator(TL, 0), 
                                              ArgIterator(TL, TL.getNumArgs()),
                                              NewTemplateArgs))
    return QualType();
  
  // FIXME: maybe don't rebuild if all the template arguments are the same.
  
  if (DependentTemplateName *DTN = Template.getAsDependentTemplateName()) {
    QualType Result
      = getSema().Context.getDependentTemplateSpecializationType(
                                                TL.getTypePtr()->getKeyword(),
                                                         DTN->getQualifier(),
                                                         DTN->getIdentifier(),
                                                               NewTemplateArgs);
   
    DependentTemplateSpecializationTypeLoc NewTL
      = TLB.push<DependentTemplateSpecializationTypeLoc>(Result);
    NewTL.setKeywordLoc(TL.getKeywordLoc());
    
    NewTL.setQualifierLoc(SS.getWithLocInContext(SemaRef.Context));
    NewTL.setNameLoc(TL.getNameLoc());
    NewTL.setLAngleLoc(TL.getLAngleLoc());
    NewTL.setRAngleLoc(TL.getRAngleLoc());
    for (unsigned i = 0, e = NewTemplateArgs.size(); i != e; ++i)
      NewTL.setArgLocInfo(i, NewTemplateArgs[i].getLocInfo());
    return Result;
  }
      
  QualType Result 
    = getDerived().RebuildTemplateSpecializationType(Template,
                                                     TL.getNameLoc(),
                                                     NewTemplateArgs);
  
  if (!Result.isNull()) {
    /// FIXME: Wrap this in an elaborated-type-specifier?
    TemplateSpecializationTypeLoc NewTL
      = TLB.push<TemplateSpecializationTypeLoc>(Result);
    NewTL.setTemplateNameLoc(TL.getNameLoc());
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
  const ElaboratedType *T = TL.getTypePtr();

  NestedNameSpecifierLoc QualifierLoc;
  // NOTE: the qualifier in an ElaboratedType is optional.
  if (TL.getQualifierLoc()) {
    QualifierLoc 
      = getDerived().TransformNestedNameSpecifierLoc(TL.getQualifierLoc());
    if (!QualifierLoc)
      return QualType();
  }

  QualType NamedT = getDerived().TransformType(TLB, TL.getNamedTypeLoc());
  if (NamedT.isNull())
    return QualType();

  // C++0x [dcl.type.elab]p2:
  //   If the identifier resolves to a typedef-name or the simple-template-id
  //   resolves to an alias template specialization, the
  //   elaborated-type-specifier is ill-formed.
  if (T->getKeyword() != ETK_None && T->getKeyword() != ETK_Typename) {
    if (const TemplateSpecializationType *TST =
          NamedT->getAs<TemplateSpecializationType>()) {
      TemplateName Template = TST->getTemplateName();
      if (TypeAliasTemplateDecl *TAT =
          dyn_cast_or_null<TypeAliasTemplateDecl>(Template.getAsTemplateDecl())) {
        SemaRef.Diag(TL.getNamedTypeLoc().getBeginLoc(),
                     diag::err_tag_reference_non_tag) << 4;
        SemaRef.Diag(TAT->getLocation(), diag::note_declared_at);
      }
    }
  }

  QualType Result = TL.getType();
  if (getDerived().AlwaysRebuild() ||
      QualifierLoc != TL.getQualifierLoc() ||
      NamedT != T->getNamedType()) {
    Result = getDerived().RebuildElaboratedType(TL.getKeywordLoc(),
                                                T->getKeyword(), 
                                                QualifierLoc, NamedT);
    if (Result.isNull())
      return QualType();
  }

  ElaboratedTypeLoc NewTL = TLB.push<ElaboratedTypeLoc>(Result);
  NewTL.setKeywordLoc(TL.getKeywordLoc());
  NewTL.setQualifierLoc(QualifierLoc);
  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformAttributedType(
                                                TypeLocBuilder &TLB,
                                                AttributedTypeLoc TL) {
  const AttributedType *oldType = TL.getTypePtr();
  QualType modifiedType = getDerived().TransformType(TLB, TL.getModifiedLoc());
  if (modifiedType.isNull())
    return QualType();

  QualType result = TL.getType();

  // FIXME: dependent operand expressions?
  if (getDerived().AlwaysRebuild() ||
      modifiedType != oldType->getModifiedType()) {
    // TODO: this is really lame; we should really be rebuilding the
    // equivalent type from first principles.
    QualType equivalentType
      = getDerived().TransformType(oldType->getEquivalentType());
    if (equivalentType.isNull())
      return QualType();
    result = SemaRef.Context.getAttributedType(oldType->getAttrKind(),
                                               modifiedType,
                                               equivalentType);
  }

  AttributedTypeLoc newTL = TLB.push<AttributedTypeLoc>(result);
  newTL.setAttrNameLoc(TL.getAttrNameLoc());
  if (TL.hasAttrOperand())
    newTL.setAttrOperandParensRange(TL.getAttrOperandParensRange());
  if (TL.hasAttrExprOperand())
    newTL.setAttrExprOperand(TL.getAttrExprOperand());
  else if (TL.hasAttrEnumOperand())
    newTL.setAttrEnumOperandLoc(TL.getAttrEnumOperandLoc());

  return result;
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
  const DependentNameType *T = TL.getTypePtr();

  NestedNameSpecifierLoc QualifierLoc
    = getDerived().TransformNestedNameSpecifierLoc(TL.getQualifierLoc());
  if (!QualifierLoc)
    return QualType();

  QualType Result
    = getDerived().RebuildDependentNameType(T->getKeyword(),
                                            TL.getKeywordLoc(),
                                            QualifierLoc,
                                            T->getIdentifier(),
                                            TL.getNameLoc());
  if (Result.isNull())
    return QualType();

  if (const ElaboratedType* ElabT = Result->getAs<ElaboratedType>()) {
    QualType NamedT = ElabT->getNamedType();
    TLB.pushTypeSpec(NamedT).setNameLoc(TL.getNameLoc());

    ElaboratedTypeLoc NewTL = TLB.push<ElaboratedTypeLoc>(Result);
    NewTL.setKeywordLoc(TL.getKeywordLoc());
    NewTL.setQualifierLoc(QualifierLoc);
  } else {
    DependentNameTypeLoc NewTL = TLB.push<DependentNameTypeLoc>(Result);
    NewTL.setKeywordLoc(TL.getKeywordLoc());
    NewTL.setQualifierLoc(QualifierLoc);
    NewTL.setNameLoc(TL.getNameLoc());
  }
  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::
          TransformDependentTemplateSpecializationType(TypeLocBuilder &TLB,
                                 DependentTemplateSpecializationTypeLoc TL) {
  NestedNameSpecifierLoc QualifierLoc;
  if (TL.getQualifierLoc()) {
    QualifierLoc
      = getDerived().TransformNestedNameSpecifierLoc(TL.getQualifierLoc());
    if (!QualifierLoc)
      return QualType();
  }
            
  return getDerived()
           .TransformDependentTemplateSpecializationType(TLB, TL, QualifierLoc);
}

template<typename Derived>
QualType TreeTransform<Derived>::
TransformDependentTemplateSpecializationType(TypeLocBuilder &TLB,
                                   DependentTemplateSpecializationTypeLoc TL,
                                       NestedNameSpecifierLoc QualifierLoc) {
  const DependentTemplateSpecializationType *T = TL.getTypePtr();
  
  TemplateArgumentListInfo NewTemplateArgs;
  NewTemplateArgs.setLAngleLoc(TL.getLAngleLoc());
  NewTemplateArgs.setRAngleLoc(TL.getRAngleLoc());
  
  typedef TemplateArgumentLocContainerIterator<
  DependentTemplateSpecializationTypeLoc> ArgIterator;
  if (getDerived().TransformTemplateArguments(ArgIterator(TL, 0),
                                              ArgIterator(TL, TL.getNumArgs()),
                                              NewTemplateArgs))
    return QualType();
  
  QualType Result
    = getDerived().RebuildDependentTemplateSpecializationType(T->getKeyword(),
                                                              QualifierLoc,
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
    NamedTL.setTemplateNameLoc(TL.getNameLoc());
    NamedTL.setLAngleLoc(TL.getLAngleLoc());
    NamedTL.setRAngleLoc(TL.getRAngleLoc());
    for (unsigned I = 0, E = NewTemplateArgs.size(); I != E; ++I)
      NamedTL.setArgLocInfo(I, NewTemplateArgs[I].getLocInfo());
    
    // Copy information relevant to the elaborated type.
    ElaboratedTypeLoc NewTL = TLB.push<ElaboratedTypeLoc>(Result);
    NewTL.setKeywordLoc(TL.getKeywordLoc());
    NewTL.setQualifierLoc(QualifierLoc);
  } else if (isa<DependentTemplateSpecializationType>(Result)) {
    DependentTemplateSpecializationTypeLoc SpecTL
      = TLB.push<DependentTemplateSpecializationTypeLoc>(Result);
    SpecTL.setKeywordLoc(TL.getKeywordLoc());
    SpecTL.setQualifierLoc(QualifierLoc);
    SpecTL.setNameLoc(TL.getNameLoc());
    SpecTL.setLAngleLoc(TL.getLAngleLoc());
    SpecTL.setRAngleLoc(TL.getRAngleLoc());
    for (unsigned I = 0, E = NewTemplateArgs.size(); I != E; ++I)
      SpecTL.setArgLocInfo(I, NewTemplateArgs[I].getLocInfo());
  } else {
    TemplateSpecializationTypeLoc SpecTL
      = TLB.push<TemplateSpecializationTypeLoc>(Result);
    SpecTL.setTemplateNameLoc(TL.getNameLoc());
    SpecTL.setLAngleLoc(TL.getLAngleLoc());
    SpecTL.setRAngleLoc(TL.getRAngleLoc());
    for (unsigned I = 0, E = NewTemplateArgs.size(); I != E; ++I)
      SpecTL.setArgLocInfo(I, NewTemplateArgs[I].getLocInfo());
  }
  return Result;
}

template<typename Derived>
QualType TreeTransform<Derived>::TransformPackExpansionType(TypeLocBuilder &TLB,
                                                      PackExpansionTypeLoc TL) {
  QualType Pattern                                      
    = getDerived().TransformType(TLB, TL.getPatternLoc());  
  if (Pattern.isNull())
    return QualType();
  
  QualType Result = TL.getType();  
  if (getDerived().AlwaysRebuild() ||
      Pattern != TL.getPatternLoc().getType()) {
    Result = getDerived().RebuildPackExpansionType(Pattern, 
                                           TL.getPatternLoc().getSourceRange(),
                                                   TL.getEllipsisLoc(),
                                           TL.getTypePtr()->getNumExpansions());
    if (Result.isNull())
      return QualType();
  }
  
  PackExpansionTypeLoc NewT = TLB.push<PackExpansionTypeLoc>(Result);
  NewT.setEllipsisLoc(TL.getEllipsisLoc());
  return Result;
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

  Decl *LD = getDerived().TransformDecl(S->getDecl()->getLocation(),
                                        S->getDecl());
  if (!LD)
    return StmtError();
  
  
  // FIXME: Pass the real colon location in.
  return getDerived().RebuildLabelStmt(S->getIdentLoc(),
                                       cast<LabelDecl>(LD), SourceLocation(),
                                       SubStmt.get());
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
      ExprResult CondE = getSema().ActOnBooleanCondition(0, S->getIfLoc(), 
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
      ExprResult CondE = getSema().ActOnBooleanCondition(0, S->getWhileLoc(), 
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
      ExprResult CondE = getSema().ActOnBooleanCondition(0, S->getForLoc(), 
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
  Decl *LD = getDerived().TransformDecl(S->getLabel()->getLocation(),
                                        S->getLabel());
  if (!LD)
    return StmtError();
  
  // Goto statements must always be rebuilt, to resolve the label.
  return getDerived().RebuildGotoStmt(S->getGotoLoc(), S->getLabelLoc(),
                                      cast<LabelDecl>(LD));
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
                                            ExceptionDecl->getInnerLocStart(),
                                            ExceptionDecl->getLocation(),
                                            ExceptionDecl->getIdentifier());
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

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformCXXForRangeStmt(CXXForRangeStmt *S) {
  StmtResult Range = getDerived().TransformStmt(S->getRangeStmt());
  if (Range.isInvalid())
    return StmtError();

  StmtResult BeginEnd = getDerived().TransformStmt(S->getBeginEndStmt());
  if (BeginEnd.isInvalid())
    return StmtError();

  ExprResult Cond = getDerived().TransformExpr(S->getCond());
  if (Cond.isInvalid())
    return StmtError();

  ExprResult Inc = getDerived().TransformExpr(S->getInc());
  if (Inc.isInvalid())
    return StmtError();

  StmtResult LoopVar = getDerived().TransformStmt(S->getLoopVarStmt());
  if (LoopVar.isInvalid())
    return StmtError();

  StmtResult NewStmt = S;
  if (getDerived().AlwaysRebuild() ||
      Range.get() != S->getRangeStmt() ||
      BeginEnd.get() != S->getBeginEndStmt() ||
      Cond.get() != S->getCond() ||
      Inc.get() != S->getInc() ||
      LoopVar.get() != S->getLoopVarStmt())
    NewStmt = getDerived().RebuildCXXForRangeStmt(S->getForLoc(),
                                                  S->getColonLoc(), Range.get(),
                                                  BeginEnd.get(), Cond.get(),
                                                  Inc.get(), LoopVar.get(),
                                                  S->getRParenLoc());

  StmtResult Body = getDerived().TransformStmt(S->getBody());
  if (Body.isInvalid())
    return StmtError();

  // Body has changed but we didn't rebuild the for-range statement. Rebuild
  // it now so we have a new statement to attach the body to.
  if (Body.get() != S->getBody() && NewStmt.get() == S)
    NewStmt = getDerived().RebuildCXXForRangeStmt(S->getForLoc(),
                                                  S->getColonLoc(), Range.get(),
                                                  BeginEnd.get(), Cond.get(),
                                                  Inc.get(), LoopVar.get(),
                                                  S->getRParenLoc());

  if (NewStmt.get() == S)
    return SemaRef.Owned(S);

  return FinishCXXForRangeStmt(NewStmt.get(), Body.get());
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformSEHTryStmt(SEHTryStmt *S) {
  StmtResult TryBlock; //  = getDerived().TransformCompoundStmt(S->getTryBlock());
  if(TryBlock.isInvalid()) return StmtError();

  StmtResult Handler = getDerived().TransformSEHHandler(S->getHandler());
  if(!getDerived().AlwaysRebuild() &&
     TryBlock.get() == S->getTryBlock() &&
     Handler.get() == S->getHandler())
    return SemaRef.Owned(S);

  return getDerived().RebuildSEHTryStmt(S->getIsCXXTry(),
                                        S->getTryLoc(),
                                        TryBlock.take(),
                                        Handler.take());
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformSEHFinallyStmt(SEHFinallyStmt *S) {
  StmtResult Block; //  = getDerived().TransformCompoundStatement(S->getBlock());
  if(Block.isInvalid()) return StmtError();

  return getDerived().RebuildSEHFinallyStmt(S->getFinallyLoc(),
                                            Block.take());
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformSEHExceptStmt(SEHExceptStmt *S) {
  ExprResult FilterExpr = getDerived().TransformExpr(S->getFilterExpr());
  if(FilterExpr.isInvalid()) return StmtError();

  StmtResult Block; //  = getDerived().TransformCompoundStatement(S->getBlock());
  if(Block.isInvalid()) return StmtError();

  return getDerived().RebuildSEHExceptStmt(S->getExceptLoc(),
                                           FilterExpr.take(),
                                           Block.take());
}

template<typename Derived>
StmtResult
TreeTransform<Derived>::TransformSEHHandler(Stmt *Handler) {
  if(isa<SEHFinallyStmt>(Handler))
    return getDerived().TransformSEHFinallyStmt(cast<SEHFinallyStmt>(Handler));
  else
    return getDerived().TransformSEHExceptStmt(cast<SEHExceptStmt>(Handler));
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
  NestedNameSpecifierLoc QualifierLoc;
  if (E->getQualifierLoc()) {
    QualifierLoc
      = getDerived().TransformNestedNameSpecifierLoc(E->getQualifierLoc());
    if (!QualifierLoc)
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
      QualifierLoc == E->getQualifierLoc() &&
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
    if (getDerived().TransformTemplateArguments(E->getTemplateArgs(),
                                                E->getNumTemplateArgs(),
                                                TransArgs))
      return ExprError();
  }

  return getDerived().RebuildDeclRefExpr(QualifierLoc, ND, NameInfo, 
                                         TemplateArgs);
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
TreeTransform<Derived>::TransformGenericSelectionExpr(GenericSelectionExpr *E) {
  ExprResult ControllingExpr =
    getDerived().TransformExpr(E->getControllingExpr());
  if (ControllingExpr.isInvalid())
    return ExprError();

  llvm::SmallVector<Expr *, 4> AssocExprs;
  llvm::SmallVector<TypeSourceInfo *, 4> AssocTypes;
  for (unsigned i = 0; i != E->getNumAssocs(); ++i) {
    TypeSourceInfo *TS = E->getAssocTypeSourceInfo(i);
    if (TS) {
      TypeSourceInfo *AssocType = getDerived().TransformType(TS);
      if (!AssocType)
        return ExprError();
      AssocTypes.push_back(AssocType);
    } else {
      AssocTypes.push_back(0);
    }

    ExprResult AssocExpr = getDerived().TransformExpr(E->getAssocExpr(i));
    if (AssocExpr.isInvalid())
      return ExprError();
    AssocExprs.push_back(AssocExpr.release());
  }

  return getDerived().RebuildGenericSelectionExpr(E->getGenericLoc(),
                                                  E->getDefaultLoc(),
                                                  E->getRParenLoc(),
                                                  ControllingExpr.release(),
                                                  AssocTypes.data(),
                                                  AssocExprs.data(),
                                                  E->getNumAssocs());
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
    Comp.LocStart = ON.getSourceRange().getBegin();
    Comp.LocEnd = ON.getSourceRange().getEnd();
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
TreeTransform<Derived>::TransformUnaryExprOrTypeTraitExpr(
                                                UnaryExprOrTypeTraitExpr *E) {
  if (E->isArgumentType()) {
    TypeSourceInfo *OldT = E->getArgumentTypeInfo();

    TypeSourceInfo *NewT = getDerived().TransformType(OldT);
    if (!NewT)
      return ExprError();

    if (!getDerived().AlwaysRebuild() && OldT == NewT)
      return SemaRef.Owned(E);

    return getDerived().RebuildUnaryExprOrTypeTrait(NewT, E->getOperatorLoc(),
                                                    E->getKind(),
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

  return getDerived().RebuildUnaryExprOrTypeTrait(SubExpr.get(),
                                                  E->getOperatorLoc(),
                                                  E->getKind(),
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
  if (getDerived().TransformExprs(E->getArgs(), E->getNumArgs(), true, Args, 
                                  &ArgChanged))
    return ExprError();
  
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

  NestedNameSpecifierLoc QualifierLoc;
  if (E->hasQualifier()) {
    QualifierLoc
      = getDerived().TransformNestedNameSpecifierLoc(E->getQualifierLoc());
    
    if (!QualifierLoc)
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
      QualifierLoc == E->getQualifierLoc() &&
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
    if (getDerived().TransformTemplateArguments(E->getTemplateArgs(),
                                                E->getNumTemplateArgs(),
                                                TransArgs))
      return ExprError();
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
                                        QualifierLoc,
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
ExprResult TreeTransform<Derived>::
TransformBinaryConditionalOperator(BinaryConditionalOperator *e) {
  // Just rebuild the common and RHS expressions and see whether we
  // get any changes.

  ExprResult commonExpr = getDerived().TransformExpr(e->getCommon());
  if (commonExpr.isInvalid())
    return ExprError();

  ExprResult rhs = getDerived().TransformExpr(e->getFalseExpr());
  if (rhs.isInvalid())
    return ExprError();

  if (!getDerived().AlwaysRebuild() &&
      commonExpr.get() == e->getCommon() &&
      rhs.get() == e->getFalseExpr())
    return SemaRef.Owned(e);

  return getDerived().RebuildConditionalOperator(commonExpr.take(),
                                                 e->getQuestionLoc(),
                                                 0,
                                                 e->getColonLoc(),
                                                 rhs.get());
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
  if (getDerived().TransformExprs(E->getInits(), E->getNumInits(), false, 
                                  Inits, &InitChanged))
    return ExprError();
  
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
  if (TransformExprs(E->getExprs(), E->getNumExprs(), true, Inits,
                     &ArgumentChanged))
    return ExprError();
  
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
  Decl *LD = getDerived().TransformDecl(E->getLabel()->getLocation(),
                                        E->getLabel());
  if (!LD)
    return ExprError();
  
  return getDerived().RebuildAddrLabelExpr(E->getAmpAmpLoc(), E->getLabelLoc(),
                                           cast<LabelDecl>(LD));
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
    if (getDerived().TransformExprs(E->getArgs() + 1, E->getNumArgs() - 1, true, 
                                    Args))
      return ExprError();

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
TreeTransform<Derived>::TransformCUDAKernelCallExpr(CUDAKernelCallExpr *E) {
  // Transform the callee.
  ExprResult Callee = getDerived().TransformExpr(E->getCallee());
  if (Callee.isInvalid())
    return ExprError();

  // Transform exec config.
  ExprResult EC = getDerived().TransformCallExpr(E->getConfig());
  if (EC.isInvalid())
    return ExprError();

  // Transform arguments.
  bool ArgChanged = false;
  ASTOwningVector<Expr*> Args(SemaRef);
  if (getDerived().TransformExprs(E->getArgs(), E->getNumArgs(), true, Args, 
                                  &ArgChanged))
    return ExprError();

  if (!getDerived().AlwaysRebuild() &&
      Callee.get() == E->getCallee() &&
      !ArgChanged)
    return SemaRef.Owned(E);

  // FIXME: Wrong source location information for the '('.
  SourceLocation FakeLParenLoc
    = ((Expr *)Callee.get())->getSourceRange().getBegin();
  return getDerived().RebuildCallExpr(Callee.get(), FakeLParenLoc,
                                      move_arg(Args),
                                      E->getRParenLoc(), EC.get());
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

    return getDerived().RebuildCXXUuidofExpr(E->getType(),
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
  if (getDerived().TransformExprs(E->getPlacementArgs(), 
                                  E->getNumPlacementArgs(), true,
                                  PlacementArgs, &ArgumentChanged))
    return ExprError();  

  // transform the constructor arguments (if any).
  ASTOwningVector<Expr*> ConstructorArgs(SemaRef);
  if (TransformExprs(E->getConstructorArgs(), E->getNumConstructorArgs(), true,
                     ConstructorArgs, &ArgumentChanged))
    return ExprError();  

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
  NestedNameSpecifierLoc QualifierLoc = E->getQualifierLoc();
  if (QualifierLoc) {
    QualifierLoc
      = getDerived().TransformNestedNameSpecifierLoc(QualifierLoc, ObjectType);
    if (!QualifierLoc)
      return ExprError();
  }
  CXXScopeSpec SS;
  SS.Adopt(QualifierLoc);

  PseudoDestructorTypeStorage Destroyed;
  if (E->getDestroyedTypeInfo()) {
    TypeSourceInfo *DestroyedTypeInfo
      = getDerived().TransformTypeInObjectScope(E->getDestroyedTypeInfo(),
                                                ObjectType, 0, SS);
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
                                                     SS,
                                                     ScopeTypeInfo,
                                                     E->getColonColonLoc(),
                                                     E->getTildeLoc(),
                                                     Destroyed);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformUnresolvedLookupExpr(
                                                  UnresolvedLookupExpr *Old) {
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
  if (Old->getQualifierLoc()) {
    NestedNameSpecifierLoc QualifierLoc
      = getDerived().TransformNestedNameSpecifierLoc(Old->getQualifierLoc());
    if (!QualifierLoc)
      return ExprError();
    
    SS.Adopt(QualifierLoc);
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
  if (getDerived().TransformTemplateArguments(Old->getTemplateArgs(),
                                              Old->getNumTemplateArgs(),
                                              TransArgs))
    return ExprError();

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
TreeTransform<Derived>::TransformArrayTypeTraitExpr(ArrayTypeTraitExpr *E) {
  TypeSourceInfo *T = getDerived().TransformType(E->getQueriedTypeSourceInfo());
  if (!T)
    return ExprError();

  if (!getDerived().AlwaysRebuild() &&
      T == E->getQueriedTypeSourceInfo())
    return SemaRef.Owned(E);

  ExprResult SubExpr;
  {
    EnterExpressionEvaluationContext Unevaluated(SemaRef, Sema::Unevaluated);
    SubExpr = getDerived().TransformExpr(E->getDimensionExpression());
    if (SubExpr.isInvalid())
      return ExprError();

    if (!getDerived().AlwaysRebuild() && SubExpr.get() == E->getDimensionExpression())
      return SemaRef.Owned(E);
  }

  return getDerived().RebuildArrayTypeTrait(E->getTrait(),
                                            E->getLocStart(),
                                            T,
                                            SubExpr.get(),
                                            E->getLocEnd());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformExpressionTraitExpr(ExpressionTraitExpr *E) {
  ExprResult SubExpr;
  {
    EnterExpressionEvaluationContext Unevaluated(SemaRef, Sema::Unevaluated);
    SubExpr = getDerived().TransformExpr(E->getQueriedExpression());
    if (SubExpr.isInvalid())
      return ExprError();

    if (!getDerived().AlwaysRebuild() && SubExpr.get() == E->getQueriedExpression())
      return SemaRef.Owned(E);
  }

  return getDerived().RebuildExpressionTrait(
      E->getTrait(), E->getLocStart(), SubExpr.get(), E->getLocEnd());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformDependentScopeDeclRefExpr(
                                               DependentScopeDeclRefExpr *E) {
  NestedNameSpecifierLoc QualifierLoc
  = getDerived().TransformNestedNameSpecifierLoc(E->getQualifierLoc());
  if (!QualifierLoc)
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
        QualifierLoc == E->getQualifierLoc() &&
        // Note: it is sufficient to compare the Name component of NameInfo:
        // if name has not changed, DNLoc has not changed either.
        NameInfo.getName() == E->getDeclName())
      return SemaRef.Owned(E);

    return getDerived().RebuildDependentScopeDeclRefExpr(QualifierLoc,
                                                         NameInfo,
                                                         /*TemplateArgs*/ 0);
  }

  TemplateArgumentListInfo TransArgs(E->getLAngleLoc(), E->getRAngleLoc());
  if (getDerived().TransformTemplateArguments(E->getTemplateArgs(),
                                              E->getNumTemplateArgs(),
                                              TransArgs))
    return ExprError();

  return getDerived().RebuildDependentScopeDeclRefExpr(QualifierLoc,
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
  if (getDerived().TransformExprs(E->getArgs(), E->getNumArgs(), true, Args, 
                                  &ArgumentChanged))
    return ExprError();
  
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
  if (TransformExprs(E->getArgs(), E->getNumArgs(), true, Args, 
                     &ArgumentChanged))
    return ExprError();

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
  Args.reserve(E->arg_size());
  if (getDerived().TransformExprs(E->arg_begin(), E->arg_size(), true, Args, 
                                  &ArgumentChanged))
    return ExprError();
  
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
                                            E->getQualifierLoc().getBeginLoc());

  NestedNameSpecifierLoc QualifierLoc;
  if (E->getQualifier()) {
    QualifierLoc
      = getDerived().TransformNestedNameSpecifierLoc(E->getQualifierLoc(),
                                                     ObjectType,
                                                     FirstQualifierInScope);
    if (!QualifierLoc)
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
        QualifierLoc == E->getQualifierLoc() &&
        NameInfo.getName() == E->getMember() &&
        FirstQualifierInScope == E->getFirstQualifierFoundInScope())
      return SemaRef.Owned(E);

    return getDerived().RebuildCXXDependentScopeMemberExpr(Base.get(),
                                                       BaseType,
                                                       E->isArrow(),
                                                       E->getOperatorLoc(),
                                                       QualifierLoc,
                                                       FirstQualifierInScope,
                                                       NameInfo,
                                                       /*TemplateArgs*/ 0);
  }

  TemplateArgumentListInfo TransArgs(E->getLAngleLoc(), E->getRAngleLoc());
  if (getDerived().TransformTemplateArguments(E->getTemplateArgs(),
                                              E->getNumTemplateArgs(),
                                              TransArgs))
    return ExprError();

  return getDerived().RebuildCXXDependentScopeMemberExpr(Base.get(),
                                                     BaseType,
                                                     E->isArrow(),
                                                     E->getOperatorLoc(),
                                                     QualifierLoc,
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

  NestedNameSpecifierLoc QualifierLoc;
  if (Old->getQualifierLoc()) {
    QualifierLoc
    = getDerived().TransformNestedNameSpecifierLoc(Old->getQualifierLoc());
    if (!QualifierLoc)
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
      else {
        R.clear();
        return ExprError();
      }
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
    if (getDerived().TransformTemplateArguments(Old->getTemplateArgs(),
                                                Old->getNumTemplateArgs(),
                                                TransArgs))
      return ExprError();
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
                                                  QualifierLoc,
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
TreeTransform<Derived>::TransformPackExpansionExpr(PackExpansionExpr *E) {
  ExprResult Pattern = getDerived().TransformExpr(E->getPattern());
  if (Pattern.isInvalid())
    return ExprError();
  
  if (!getDerived().AlwaysRebuild() && Pattern.get() == E->getPattern())
    return SemaRef.Owned(E);

  return getDerived().RebuildPackExpansion(Pattern.get(), E->getEllipsisLoc(),
                                           E->getNumExpansions());
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformSizeOfPackExpr(SizeOfPackExpr *E) {
  // If E is not value-dependent, then nothing will change when we transform it.
  // Note: This is an instantiation-centric view.
  if (!E->isValueDependent())
    return SemaRef.Owned(E);

  // Note: None of the implementations of TryExpandParameterPacks can ever
  // produce a diagnostic when given only a single unexpanded parameter pack,
  // so 
  UnexpandedParameterPack Unexpanded(E->getPack(), E->getPackLoc());
  bool ShouldExpand = false;
  bool RetainExpansion = false;
  llvm::Optional<unsigned> NumExpansions;
  if (getDerived().TryExpandParameterPacks(E->getOperatorLoc(), E->getPackLoc(), 
                                           &Unexpanded, 1, 
                                           ShouldExpand, RetainExpansion,
                                           NumExpansions))
    return ExprError();
  
  if (!ShouldExpand || RetainExpansion)
    return SemaRef.Owned(E);
  
  // We now know the length of the parameter pack, so build a new expression
  // that stores that length.
  return getDerived().RebuildSizeOfPackExpr(E->getOperatorLoc(), E->getPack(), 
                                            E->getPackLoc(), E->getRParenLoc(), 
                                            *NumExpansions);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformSubstNonTypeTemplateParmPackExpr(
                                          SubstNonTypeTemplateParmPackExpr *E) {
  // Default behavior is to do nothing with this transformation.
  return SemaRef.Owned(E);
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
  Args.reserve(E->getNumArgs());
  if (getDerived().TransformExprs(E->getArgs(), E->getNumArgs(), false, Args, 
                                  &ArgChanged))
    return ExprError();
  
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
  SubExprs.reserve(E->getNumSubExprs());
  if (getDerived().TransformExprs(E->getSubExprs(), E->getNumSubExprs(), false, 
                                  SubExprs, &ArgumentChanged))
    return ExprError();

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
  BlockDecl *oldBlock = E->getBlockDecl();
  
  SemaRef.ActOnBlockStart(E->getCaretLocation(), /*Scope=*/0);
  BlockScopeInfo *blockScope = SemaRef.getCurBlock();

  blockScope->TheDecl->setIsVariadic(oldBlock->isVariadic());
  // We built a new blockScopeInfo in call to ActOnBlockStart
  // in above, CapturesCXXThis need be set here from the block
  // expression.
  blockScope->CapturesCXXThis = oldBlock->capturesCXXThis();
  
  llvm::SmallVector<ParmVarDecl*, 4> params;
  llvm::SmallVector<QualType, 4> paramTypes;
  
  // Parameter substitution.
  if (getDerived().TransformFunctionTypeParams(E->getCaretLocation(),
                                               oldBlock->param_begin(),
                                               oldBlock->param_size(),
                                               0, paramTypes, &params))
    return true;

  const FunctionType *exprFunctionType = E->getFunctionType();
  QualType exprResultType = exprFunctionType->getResultType();
  if (!exprResultType.isNull()) {
    if (!exprResultType->isDependentType())
      blockScope->ReturnType = exprResultType;
    else if (exprResultType != getSema().Context.DependentTy)
      blockScope->ReturnType = getDerived().TransformType(exprResultType);
  }
  
  // If the return type has not been determined yet, leave it as a dependent
  // type; it'll get set when we process the body.
  if (blockScope->ReturnType.isNull())
    blockScope->ReturnType = getSema().Context.DependentTy;

  // Don't allow returning a objc interface by value.
  if (blockScope->ReturnType->isObjCObjectType()) {
    getSema().Diag(E->getCaretLocation(), 
                   diag::err_object_cannot_be_passed_returned_by_value) 
      << 0 << blockScope->ReturnType;
    return ExprError();
  }

  QualType functionType = getDerived().RebuildFunctionProtoType(
                                                        blockScope->ReturnType,
                                                        paramTypes.data(),
                                                        paramTypes.size(),
                                                        oldBlock->isVariadic(),
                                                        0, RQ_None,
                                               exprFunctionType->getExtInfo());
  blockScope->FunctionType = functionType;

  // Set the parameters on the block decl.
  if (!params.empty())
    blockScope->TheDecl->setParams(params.data(), params.size());
  
  // If the return type wasn't explicitly set, it will have been marked as a 
  // dependent type (DependentTy); clear out the return type setting so 
  // we will deduce the return type when type-checking the block's body.
  if (blockScope->ReturnType == getSema().Context.DependentTy)
    blockScope->ReturnType = QualType();
  
  // Transform the body
  StmtResult body = getDerived().TransformStmt(E->getBody());
  if (body.isInvalid())
    return ExprError();

#ifndef NDEBUG
  // In builds with assertions, make sure that we captured everything we
  // captured before.
  if (!SemaRef.getDiagnostics().hasErrorOccurred()) {
    for (BlockDecl::capture_iterator i = oldBlock->capture_begin(),
           e = oldBlock->capture_end(); i != e; ++i) {
      VarDecl *oldCapture = i->getVariable();

      // Ignore parameter packs.
      if (isa<ParmVarDecl>(oldCapture) &&
          cast<ParmVarDecl>(oldCapture)->isParameterPack())
        continue;

      VarDecl *newCapture =
        cast<VarDecl>(getDerived().TransformDecl(E->getCaretLocation(),
                                                 oldCapture));
      assert(blockScope->CaptureMap.count(newCapture));
    }
  }
#endif

  return SemaRef.ActOnBlockStmtExpr(E->getCaretLocation(), body.get(),
                                    /*Scope=*/0);
}

template<typename Derived>
ExprResult
TreeTransform<Derived>::TransformBlockDeclRefExpr(BlockDeclRefExpr *E) {
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
  return getDerived().RebuildDeclRefExpr(NestedNameSpecifierLoc(), 
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
                                                  RefQualifierKind RefQualifier,
                                            const FunctionType::ExtInfo &Info) {
  return SemaRef.BuildFunctionType(T, ParamTypes, NumParamTypes, Variadic,
                                   Quals, RefQualifier,
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
QualType TreeTransform<Derived>::RebuildUnaryTransformType(QualType BaseType,
                                            UnaryTransformType::UTTKind UKind,
                                            SourceLocation Loc) {
  return SemaRef.BuildUnaryTransformType(BaseType, UKind, Loc);
}

template<typename Derived>
QualType TreeTransform<Derived>::RebuildTemplateSpecializationType(
                                                      TemplateName Template,
                                             SourceLocation TemplateNameLoc,
                                     TemplateArgumentListInfo &TemplateArgs) {
  return SemaRef.CheckTemplateIdType(Template, TemplateNameLoc, TemplateArgs);
}

template<typename Derived>
TemplateName
TreeTransform<Derived>::RebuildTemplateName(CXXScopeSpec &SS,
                                            bool TemplateKW,
                                            TemplateDecl *Template) {
  return SemaRef.Context.getQualifiedTemplateName(SS.getScopeRep(), TemplateKW,
                                                  Template);
}

template<typename Derived>
TemplateName
TreeTransform<Derived>::RebuildTemplateName(CXXScopeSpec &SS,
                                            const IdentifierInfo &Name,
                                            SourceLocation NameLoc,
                                            QualType ObjectType,
                                            NamedDecl *FirstQualifierInScope) {
  UnqualifiedId TemplateName;
  TemplateName.setIdentifier(&Name, NameLoc);
  Sema::TemplateTy Template;
  getSema().ActOnDependentTemplateName(/*Scope=*/0,
                                       /*FIXME:*/SourceLocation(),
                                       SS,
                                       TemplateName,
                                       ParsedType::make(ObjectType),
                                       /*EnteringContext=*/false,
                                       Template);
  return Template.get();
}

template<typename Derived>
TemplateName
TreeTransform<Derived>::RebuildTemplateName(CXXScopeSpec &SS,
                                            OverloadedOperatorKind Operator,
                                            SourceLocation NameLoc,
                                            QualType ObjectType) {
  UnqualifiedId Name;
  // FIXME: Bogus location information.
  SourceLocation SymbolLocations[3] = { NameLoc, NameLoc, NameLoc };   
  Name.setOperatorFunctionId(NameLoc, Operator, SymbolLocations);
  Sema::TemplateTy Template;
  getSema().ActOnDependentTemplateName(/*Scope=*/0,
                                       /*FIXME:*/SourceLocation(),
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
                                                       CXXScopeSpec &SS,
                                                     TypeSourceInfo *ScopeType,
                                                       SourceLocation CCLoc,
                                                       SourceLocation TildeLoc,
                                        PseudoDestructorTypeStorage Destroyed) {
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
