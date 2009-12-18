//===------- SemaTemplate.cpp - Semantic Analysis for C++ Templates -------===/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===/
//
//  This file implements semantic analysis for C++ templates.
//===----------------------------------------------------------------------===/

#include "Sema.h"
#include "Lookup.h"
#include "TreeTransform.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Parse/Template.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "llvm/ADT/StringExtras.h"
using namespace clang;

/// \brief Determine whether the declaration found is acceptable as the name
/// of a template and, if so, return that template declaration. Otherwise,
/// returns NULL.
static NamedDecl *isAcceptableTemplateName(ASTContext &Context, NamedDecl *D) {
  if (!D)
    return 0;

  if (isa<TemplateDecl>(D))
    return D;

  if (CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(D)) {
    // C++ [temp.local]p1:
    //   Like normal (non-template) classes, class templates have an
    //   injected-class-name (Clause 9). The injected-class-name
    //   can be used with or without a template-argument-list. When
    //   it is used without a template-argument-list, it is
    //   equivalent to the injected-class-name followed by the
    //   template-parameters of the class template enclosed in
    //   <>. When it is used with a template-argument-list, it
    //   refers to the specified class template specialization,
    //   which could be the current specialization or another
    //   specialization.
    if (Record->isInjectedClassName()) {
      Record = cast<CXXRecordDecl>(Record->getDeclContext());
      if (Record->getDescribedClassTemplate())
        return Record->getDescribedClassTemplate();

      if (ClassTemplateSpecializationDecl *Spec
            = dyn_cast<ClassTemplateSpecializationDecl>(Record))
        return Spec->getSpecializedTemplate();
    }

    return 0;
  }

  return 0;
}

static void FilterAcceptableTemplateNames(ASTContext &C, LookupResult &R) {
  LookupResult::Filter filter = R.makeFilter();
  while (filter.hasNext()) {
    NamedDecl *Orig = filter.next();
    NamedDecl *Repl = isAcceptableTemplateName(C, Orig->getUnderlyingDecl());
    if (!Repl)
      filter.erase();
    else if (Repl != Orig)
      filter.replace(Repl);
  }
  filter.done();
}

TemplateNameKind Sema::isTemplateName(Scope *S,
                                      const CXXScopeSpec &SS,
                                      UnqualifiedId &Name,
                                      TypeTy *ObjectTypePtr,
                                      bool EnteringContext,
                                      TemplateTy &TemplateResult) {
  DeclarationName TName;
  
  switch (Name.getKind()) {
  case UnqualifiedId::IK_Identifier:
    TName = DeclarationName(Name.Identifier);
    break;
      
  case UnqualifiedId::IK_OperatorFunctionId:
    TName = Context.DeclarationNames.getCXXOperatorName(
                                              Name.OperatorFunctionId.Operator);
    break;

  case UnqualifiedId::IK_LiteralOperatorId:
    TName = Context.DeclarationNames.getCXXLiteralOperatorName(Name.Identifier);
    break;

  default:
    return TNK_Non_template;
  }

  QualType ObjectType = QualType::getFromOpaquePtr(ObjectTypePtr);

  LookupResult R(*this, TName, SourceLocation(), LookupOrdinaryName);
  R.suppressDiagnostics();
  LookupTemplateName(R, S, SS, ObjectType, EnteringContext);
  if (R.empty())
    return TNK_Non_template;

  TemplateName Template;
  TemplateNameKind TemplateKind;

  unsigned ResultCount = R.end() - R.begin();
  if (ResultCount > 1) {
    // We assume that we'll preserve the qualifier from a function
    // template name in other ways.
    Template = Context.getOverloadedTemplateName(R.begin(), R.end());
    TemplateKind = TNK_Function_template;
  } else {
    TemplateDecl *TD = cast<TemplateDecl>((*R.begin())->getUnderlyingDecl());

    if (SS.isSet() && !SS.isInvalid()) {
      NestedNameSpecifier *Qualifier
        = static_cast<NestedNameSpecifier *>(SS.getScopeRep());
      Template = Context.getQualifiedTemplateName(Qualifier, false, TD);
    } else {
      Template = TemplateName(TD);
    }

    if (isa<FunctionTemplateDecl>(TD))
      TemplateKind = TNK_Function_template;
    else {
      assert(isa<ClassTemplateDecl>(TD) || isa<TemplateTemplateParmDecl>(TD));
      TemplateKind = TNK_Type_template;
    }
  }

  TemplateResult = TemplateTy::make(Template);
  return TemplateKind;
}

void Sema::LookupTemplateName(LookupResult &Found,
                              Scope *S, const CXXScopeSpec &SS,
                              QualType ObjectType,
                              bool EnteringContext) {
  // Determine where to perform name lookup
  DeclContext *LookupCtx = 0;
  bool isDependent = false;
  if (!ObjectType.isNull()) {
    // This nested-name-specifier occurs in a member access expression, e.g.,
    // x->B::f, and we are looking into the type of the object.
    assert(!SS.isSet() && "ObjectType and scope specifier cannot coexist");
    LookupCtx = computeDeclContext(ObjectType);
    isDependent = ObjectType->isDependentType();
    assert((isDependent || !ObjectType->isIncompleteType()) && 
           "Caller should have completed object type");
  } else if (SS.isSet()) {
    // This nested-name-specifier occurs after another nested-name-specifier,
    // so long into the context associated with the prior nested-name-specifier.
    LookupCtx = computeDeclContext(SS, EnteringContext);
    isDependent = isDependentScopeSpecifier(SS);
    
    // The declaration context must be complete.
    if (LookupCtx && RequireCompleteDeclContext(SS))
      return;
  }

  bool ObjectTypeSearchedInScope = false;
  if (LookupCtx) {
    // Perform "qualified" name lookup into the declaration context we
    // computed, which is either the type of the base of a member access
    // expression or the declaration context associated with a prior
    // nested-name-specifier.
    LookupQualifiedName(Found, LookupCtx);

    if (!ObjectType.isNull() && Found.empty()) {
      // C++ [basic.lookup.classref]p1:
      //   In a class member access expression (5.2.5), if the . or -> token is
      //   immediately followed by an identifier followed by a <, the
      //   identifier must be looked up to determine whether the < is the
      //   beginning of a template argument list (14.2) or a less-than operator.
      //   The identifier is first looked up in the class of the object
      //   expression. If the identifier is not found, it is then looked up in
      //   the context of the entire postfix-expression and shall name a class
      //   or function template.
      //
      // FIXME: When we're instantiating a template, do we actually have to
      // look in the scope of the template? Seems fishy...
      if (S) LookupName(Found, S);
      ObjectTypeSearchedInScope = true;
    }
  } else if (isDependent) {
    // We cannot look into a dependent object type or
    return;
  } else {
    // Perform unqualified name lookup in the current scope.
    LookupName(Found, S);
  }

  // FIXME: Cope with ambiguous name-lookup results.
  assert(!Found.isAmbiguous() &&
         "Cannot handle template name-lookup ambiguities");

  FilterAcceptableTemplateNames(Context, Found);
  if (Found.empty())
    return;

  if (S && !ObjectType.isNull() && !ObjectTypeSearchedInScope) {
    // C++ [basic.lookup.classref]p1:
    //   [...] If the lookup in the class of the object expression finds a
    //   template, the name is also looked up in the context of the entire
    //   postfix-expression and [...]
    //
    LookupResult FoundOuter(*this, Found.getLookupName(), Found.getNameLoc(),
                            LookupOrdinaryName);
    LookupName(FoundOuter, S);
    FilterAcceptableTemplateNames(Context, FoundOuter);
    // FIXME: Handle ambiguities in this lookup better

    if (FoundOuter.empty()) {
      //   - if the name is not found, the name found in the class of the
      //     object expression is used, otherwise
    } else if (!FoundOuter.getAsSingle<ClassTemplateDecl>()) {
      //   - if the name is found in the context of the entire
      //     postfix-expression and does not name a class template, the name
      //     found in the class of the object expression is used, otherwise
    } else {
      //   - if the name found is a class template, it must refer to the same
      //     entity as the one found in the class of the object expression,
      //     otherwise the program is ill-formed.
      if (!Found.isSingleResult() ||
          Found.getFoundDecl()->getCanonicalDecl()
            != FoundOuter.getFoundDecl()->getCanonicalDecl()) {
        Diag(Found.getNameLoc(), 
             diag::err_nested_name_member_ref_lookup_ambiguous)
          << Found.getLookupName();
        Diag(Found.getRepresentativeDecl()->getLocation(),
             diag::note_ambig_member_ref_object_type)
          << ObjectType;
        Diag(FoundOuter.getFoundDecl()->getLocation(),
             diag::note_ambig_member_ref_scope);

        // Recover by taking the template that we found in the object
        // expression's type.
      }
    }
  }
}

/// ActOnDependentIdExpression - Handle a dependent id-expression that
/// was just parsed.  This is only possible with an explicit scope
/// specifier naming a dependent type.
Sema::OwningExprResult
Sema::ActOnDependentIdExpression(const CXXScopeSpec &SS,
                                 DeclarationName Name,
                                 SourceLocation NameLoc,
                                 bool isAddressOfOperand,
                           const TemplateArgumentListInfo *TemplateArgs) {
  NestedNameSpecifier *Qualifier
    = static_cast<NestedNameSpecifier*>(SS.getScopeRep());
    
  if (!isAddressOfOperand &&
      isa<CXXMethodDecl>(CurContext) &&
      cast<CXXMethodDecl>(CurContext)->isInstance()) {
    QualType ThisType = cast<CXXMethodDecl>(CurContext)->getThisType(Context);
    
    // Since the 'this' expression is synthesized, we don't need to
    // perform the double-lookup check.
    NamedDecl *FirstQualifierInScope = 0;

    return Owned(CXXDependentScopeMemberExpr::Create(Context,
                                                     /*This*/ 0, ThisType,
                                                     /*IsArrow*/ true,
                                                     /*Op*/ SourceLocation(),
                                                     Qualifier, SS.getRange(),
                                                     FirstQualifierInScope,
                                                     Name, NameLoc,
                                                     TemplateArgs));
  }

  return BuildDependentDeclRefExpr(SS, Name, NameLoc, TemplateArgs);
}

Sema::OwningExprResult
Sema::BuildDependentDeclRefExpr(const CXXScopeSpec &SS,
                                DeclarationName Name,
                                SourceLocation NameLoc,
                                const TemplateArgumentListInfo *TemplateArgs) {
  return Owned(DependentScopeDeclRefExpr::Create(Context,
               static_cast<NestedNameSpecifier*>(SS.getScopeRep()),
                                                 SS.getRange(),
                                                 Name, NameLoc,
                                                 TemplateArgs));
}

/// DiagnoseTemplateParameterShadow - Produce a diagnostic complaining
/// that the template parameter 'PrevDecl' is being shadowed by a new
/// declaration at location Loc. Returns true to indicate that this is
/// an error, and false otherwise.
bool Sema::DiagnoseTemplateParameterShadow(SourceLocation Loc, Decl *PrevDecl) {
  assert(PrevDecl->isTemplateParameter() && "Not a template parameter");

  // Microsoft Visual C++ permits template parameters to be shadowed.
  if (getLangOptions().Microsoft)
    return false;

  // C++ [temp.local]p4:
  //   A template-parameter shall not be redeclared within its
  //   scope (including nested scopes).
  Diag(Loc, diag::err_template_param_shadow)
    << cast<NamedDecl>(PrevDecl)->getDeclName();
  Diag(PrevDecl->getLocation(), diag::note_template_param_here);
  return true;
}

/// AdjustDeclIfTemplate - If the given decl happens to be a template, reset
/// the parameter D to reference the templated declaration and return a pointer
/// to the template declaration. Otherwise, do nothing to D and return null.
TemplateDecl *Sema::AdjustDeclIfTemplate(DeclPtrTy &D) {
  if (TemplateDecl *Temp = dyn_cast_or_null<TemplateDecl>(D.getAs<Decl>())) {
    D = DeclPtrTy::make(Temp->getTemplatedDecl());
    return Temp;
  }
  return 0;
}

static TemplateArgumentLoc translateTemplateArgument(Sema &SemaRef,
                                            const ParsedTemplateArgument &Arg) {
  
  switch (Arg.getKind()) {
  case ParsedTemplateArgument::Type: {
    TypeSourceInfo *DI;
    QualType T = SemaRef.GetTypeFromParser(Arg.getAsType(), &DI);
    if (!DI) 
      DI = SemaRef.Context.getTrivialTypeSourceInfo(T, Arg.getLocation());
    return TemplateArgumentLoc(TemplateArgument(T), DI);
  }
    
  case ParsedTemplateArgument::NonType: {
    Expr *E = static_cast<Expr *>(Arg.getAsExpr());
    return TemplateArgumentLoc(TemplateArgument(E), E);
  }
    
  case ParsedTemplateArgument::Template: {
    TemplateName Template
      = TemplateName::getFromVoidPointer(Arg.getAsTemplate().get());
    return TemplateArgumentLoc(TemplateArgument(Template),
                               Arg.getScopeSpec().getRange(),
                               Arg.getLocation());
  }
  }
  
  llvm_unreachable("Unhandled parsed template argument");
  return TemplateArgumentLoc();
}
                                                     
/// \brief Translates template arguments as provided by the parser
/// into template arguments used by semantic analysis.
void Sema::translateTemplateArguments(const ASTTemplateArgsPtr &TemplateArgsIn,
                                      TemplateArgumentListInfo &TemplateArgs) {
 for (unsigned I = 0, Last = TemplateArgsIn.size(); I != Last; ++I)
   TemplateArgs.addArgument(translateTemplateArgument(*this,
                                                      TemplateArgsIn[I]));
}
                                                     
/// ActOnTypeParameter - Called when a C++ template type parameter
/// (e.g., "typename T") has been parsed. Typename specifies whether
/// the keyword "typename" was used to declare the type parameter
/// (otherwise, "class" was used), and KeyLoc is the location of the
/// "class" or "typename" keyword. ParamName is the name of the
/// parameter (NULL indicates an unnamed template parameter) and
/// ParamName is the location of the parameter name (if any).
/// If the type parameter has a default argument, it will be added
/// later via ActOnTypeParameterDefault.
Sema::DeclPtrTy Sema::ActOnTypeParameter(Scope *S, bool Typename, bool Ellipsis,
                                         SourceLocation EllipsisLoc,
                                         SourceLocation KeyLoc,
                                         IdentifierInfo *ParamName,
                                         SourceLocation ParamNameLoc,
                                         unsigned Depth, unsigned Position) {
  assert(S->isTemplateParamScope() &&
         "Template type parameter not in template parameter scope!");
  bool Invalid = false;

  if (ParamName) {
    NamedDecl *PrevDecl = LookupSingleName(S, ParamName, LookupTagName);
    if (PrevDecl && PrevDecl->isTemplateParameter())
      Invalid = Invalid || DiagnoseTemplateParameterShadow(ParamNameLoc,
                                                           PrevDecl);
  }

  SourceLocation Loc = ParamNameLoc;
  if (!ParamName)
    Loc = KeyLoc;

  TemplateTypeParmDecl *Param
    = TemplateTypeParmDecl::Create(Context, CurContext, Loc,
                                   Depth, Position, ParamName, Typename,
                                   Ellipsis);
  if (Invalid)
    Param->setInvalidDecl();

  if (ParamName) {
    // Add the template parameter into the current scope.
    S->AddDecl(DeclPtrTy::make(Param));
    IdResolver.AddDecl(Param);
  }

  return DeclPtrTy::make(Param);
}

/// ActOnTypeParameterDefault - Adds a default argument (the type
/// Default) to the given template type parameter (TypeParam).
void Sema::ActOnTypeParameterDefault(DeclPtrTy TypeParam,
                                     SourceLocation EqualLoc,
                                     SourceLocation DefaultLoc,
                                     TypeTy *DefaultT) {
  TemplateTypeParmDecl *Parm
    = cast<TemplateTypeParmDecl>(TypeParam.getAs<Decl>());

  TypeSourceInfo *DefaultTInfo;
  GetTypeFromParser(DefaultT, &DefaultTInfo);

  assert(DefaultTInfo && "expected source information for type");

  // C++0x [temp.param]p9:
  // A default template-argument may be specified for any kind of
  // template-parameter that is not a template parameter pack.
  if (Parm->isParameterPack()) {
    Diag(DefaultLoc, diag::err_template_param_pack_default_arg);
    return;
  }

  // C++ [temp.param]p14:
  //   A template-parameter shall not be used in its own default argument.
  // FIXME: Implement this check! Needs a recursive walk over the types.

  // Check the template argument itself.
  if (CheckTemplateArgument(Parm, DefaultTInfo)) {
    Parm->setInvalidDecl();
    return;
  }

  Parm->setDefaultArgument(DefaultTInfo, false);
}

/// \brief Check that the type of a non-type template parameter is
/// well-formed.
///
/// \returns the (possibly-promoted) parameter type if valid;
/// otherwise, produces a diagnostic and returns a NULL type.
QualType
Sema::CheckNonTypeTemplateParameterType(QualType T, SourceLocation Loc) {
  // C++ [temp.param]p4:
  //
  // A non-type template-parameter shall have one of the following
  // (optionally cv-qualified) types:
  //
  //       -- integral or enumeration type,
  if (T->isIntegralType() || T->isEnumeralType() ||
      //   -- pointer to object or pointer to function,
      (T->isPointerType() &&
       (T->getAs<PointerType>()->getPointeeType()->isObjectType() ||
        T->getAs<PointerType>()->getPointeeType()->isFunctionType())) ||
      //   -- reference to object or reference to function,
      T->isReferenceType() ||
      //   -- pointer to member.
      T->isMemberPointerType() ||
      // If T is a dependent type, we can't do the check now, so we
      // assume that it is well-formed.
      T->isDependentType())
    return T;
  // C++ [temp.param]p8:
  //
  //   A non-type template-parameter of type "array of T" or
  //   "function returning T" is adjusted to be of type "pointer to
  //   T" or "pointer to function returning T", respectively.
  else if (T->isArrayType())
    // FIXME: Keep the type prior to promotion?
    return Context.getArrayDecayedType(T);
  else if (T->isFunctionType())
    // FIXME: Keep the type prior to promotion?
    return Context.getPointerType(T);

  Diag(Loc, diag::err_template_nontype_parm_bad_type)
    << T;

  return QualType();
}

/// ActOnNonTypeTemplateParameter - Called when a C++ non-type
/// template parameter (e.g., "int Size" in "template<int Size>
/// class Array") has been parsed. S is the current scope and D is
/// the parsed declarator.
Sema::DeclPtrTy Sema::ActOnNonTypeTemplateParameter(Scope *S, Declarator &D,
                                                    unsigned Depth,
                                                    unsigned Position) {
  TypeSourceInfo *TInfo = 0;
  QualType T = GetTypeForDeclarator(D, S, &TInfo);

  assert(S->isTemplateParamScope() &&
         "Non-type template parameter not in template parameter scope!");
  bool Invalid = false;

  IdentifierInfo *ParamName = D.getIdentifier();
  if (ParamName) {
    NamedDecl *PrevDecl = LookupSingleName(S, ParamName, LookupTagName);
    if (PrevDecl && PrevDecl->isTemplateParameter())
      Invalid = Invalid || DiagnoseTemplateParameterShadow(D.getIdentifierLoc(),
                                                           PrevDecl);
  }

  T = CheckNonTypeTemplateParameterType(T, D.getIdentifierLoc());
  if (T.isNull()) {
    T = Context.IntTy; // Recover with an 'int' type.
    Invalid = true;
  }

  NonTypeTemplateParmDecl *Param
    = NonTypeTemplateParmDecl::Create(Context, CurContext, D.getIdentifierLoc(),
                                      Depth, Position, ParamName, T, TInfo);
  if (Invalid)
    Param->setInvalidDecl();

  if (D.getIdentifier()) {
    // Add the template parameter into the current scope.
    S->AddDecl(DeclPtrTy::make(Param));
    IdResolver.AddDecl(Param);
  }
  return DeclPtrTy::make(Param);
}

/// \brief Adds a default argument to the given non-type template
/// parameter.
void Sema::ActOnNonTypeTemplateParameterDefault(DeclPtrTy TemplateParamD,
                                                SourceLocation EqualLoc,
                                                ExprArg DefaultE) {
  NonTypeTemplateParmDecl *TemplateParm
    = cast<NonTypeTemplateParmDecl>(TemplateParamD.getAs<Decl>());
  Expr *Default = static_cast<Expr *>(DefaultE.get());

  // C++ [temp.param]p14:
  //   A template-parameter shall not be used in its own default argument.
  // FIXME: Implement this check! Needs a recursive walk over the types.

  // Check the well-formedness of the default template argument.
  TemplateArgument Converted;
  if (CheckTemplateArgument(TemplateParm, TemplateParm->getType(), Default,
                            Converted)) {
    TemplateParm->setInvalidDecl();
    return;
  }

  TemplateParm->setDefaultArgument(DefaultE.takeAs<Expr>());
}


/// ActOnTemplateTemplateParameter - Called when a C++ template template
/// parameter (e.g. T in template <template <typename> class T> class array)
/// has been parsed. S is the current scope.
Sema::DeclPtrTy Sema::ActOnTemplateTemplateParameter(Scope* S,
                                                     SourceLocation TmpLoc,
                                                     TemplateParamsTy *Params,
                                                     IdentifierInfo *Name,
                                                     SourceLocation NameLoc,
                                                     unsigned Depth,
                                                     unsigned Position) {
  assert(S->isTemplateParamScope() &&
         "Template template parameter not in template parameter scope!");

  // Construct the parameter object.
  TemplateTemplateParmDecl *Param =
    TemplateTemplateParmDecl::Create(Context, CurContext, TmpLoc, Depth,
                                     Position, Name,
                                     (TemplateParameterList*)Params);

  // Make sure the parameter is valid.
  // FIXME: Decl object is not currently invalidated anywhere so this doesn't
  // do anything yet. However, if the template parameter list or (eventual)
  // default value is ever invalidated, that will propagate here.
  bool Invalid = false;
  if (Invalid) {
    Param->setInvalidDecl();
  }

  // If the tt-param has a name, then link the identifier into the scope
  // and lookup mechanisms.
  if (Name) {
    S->AddDecl(DeclPtrTy::make(Param));
    IdResolver.AddDecl(Param);
  }

  return DeclPtrTy::make(Param);
}

/// \brief Adds a default argument to the given template template
/// parameter.
void Sema::ActOnTemplateTemplateParameterDefault(DeclPtrTy TemplateParamD,
                                                 SourceLocation EqualLoc,
                                        const ParsedTemplateArgument &Default) {
  TemplateTemplateParmDecl *TemplateParm
    = cast<TemplateTemplateParmDecl>(TemplateParamD.getAs<Decl>());
  
  // C++ [temp.param]p14:
  //   A template-parameter shall not be used in its own default argument.
  // FIXME: Implement this check! Needs a recursive walk over the types.

  // Check only that we have a template template argument. We don't want to
  // try to check well-formedness now, because our template template parameter
  // might have dependent types in its template parameters, which we wouldn't
  // be able to match now.
  //
  // If none of the template template parameter's template arguments mention
  // other template parameters, we could actually perform more checking here.
  // However, it isn't worth doing.
  TemplateArgumentLoc DefaultArg = translateTemplateArgument(*this, Default);
  if (DefaultArg.getArgument().getAsTemplate().isNull()) {
    Diag(DefaultArg.getLocation(), diag::err_template_arg_not_class_template)
      << DefaultArg.getSourceRange();
    return;
  }
  
  TemplateParm->setDefaultArgument(DefaultArg);
}

/// ActOnTemplateParameterList - Builds a TemplateParameterList that
/// contains the template parameters in Params/NumParams.
Sema::TemplateParamsTy *
Sema::ActOnTemplateParameterList(unsigned Depth,
                                 SourceLocation ExportLoc,
                                 SourceLocation TemplateLoc,
                                 SourceLocation LAngleLoc,
                                 DeclPtrTy *Params, unsigned NumParams,
                                 SourceLocation RAngleLoc) {
  if (ExportLoc.isValid())
    Diag(ExportLoc, diag::warn_template_export_unsupported);

  return TemplateParameterList::Create(Context, TemplateLoc, LAngleLoc,
                                       (NamedDecl**)Params, NumParams, 
                                       RAngleLoc);
}

Sema::DeclResult
Sema::CheckClassTemplate(Scope *S, unsigned TagSpec, TagUseKind TUK,
                         SourceLocation KWLoc, const CXXScopeSpec &SS,
                         IdentifierInfo *Name, SourceLocation NameLoc,
                         AttributeList *Attr,
                         TemplateParameterList *TemplateParams,
                         AccessSpecifier AS) {
  assert(TemplateParams && TemplateParams->size() > 0 &&
         "No template parameters");
  assert(TUK != TUK_Reference && "Can only declare or define class templates");
  bool Invalid = false;

  // Check that we can declare a template here.
  if (CheckTemplateDeclScope(S, TemplateParams))
    return true;

  TagDecl::TagKind Kind = TagDecl::getTagKindForTypeSpec(TagSpec);
  assert(Kind != TagDecl::TK_enum && "can't build template of enumerated type");

  // There is no such thing as an unnamed class template.
  if (!Name) {
    Diag(KWLoc, diag::err_template_unnamed_class);
    return true;
  }

  // Find any previous declaration with this name.
  DeclContext *SemanticContext;
  LookupResult Previous(*this, Name, NameLoc, LookupOrdinaryName,
                        ForRedeclaration);
  if (SS.isNotEmpty() && !SS.isInvalid()) {
    if (RequireCompleteDeclContext(SS))
      return true;

    SemanticContext = computeDeclContext(SS, true);
    if (!SemanticContext) {
      // FIXME: Produce a reasonable diagnostic here
      return true;
    }

    LookupQualifiedName(Previous, SemanticContext);
  } else {
    SemanticContext = CurContext;
    LookupName(Previous, S);
  }

  assert(!Previous.isAmbiguous() && "Ambiguity in class template redecl?");
  NamedDecl *PrevDecl = 0;
  if (Previous.begin() != Previous.end())
    PrevDecl = *Previous.begin();

  // If there is a previous declaration with the same name, check
  // whether this is a valid redeclaration.
  ClassTemplateDecl *PrevClassTemplate
    = dyn_cast_or_null<ClassTemplateDecl>(PrevDecl);

  // We may have found the injected-class-name of a class template,
  // class template partial specialization, or class template specialization. 
  // In these cases, grab the template that is being defined or specialized.
  if (!PrevClassTemplate && PrevDecl && isa<CXXRecordDecl>(PrevDecl) && 
      cast<CXXRecordDecl>(PrevDecl)->isInjectedClassName()) {
    PrevDecl = cast<CXXRecordDecl>(PrevDecl->getDeclContext());
    PrevClassTemplate 
      = cast<CXXRecordDecl>(PrevDecl)->getDescribedClassTemplate();
    if (!PrevClassTemplate && isa<ClassTemplateSpecializationDecl>(PrevDecl)) {
      PrevClassTemplate
        = cast<ClassTemplateSpecializationDecl>(PrevDecl)
            ->getSpecializedTemplate();
    }
  }

  if (TUK == TUK_Friend) {
    // C++ [namespace.memdef]p3:
    //   [...] When looking for a prior declaration of a class or a function 
    //   declared as a friend, and when the name of the friend class or 
    //   function is neither a qualified name nor a template-id, scopes outside
    //   the innermost enclosing namespace scope are not considered.
    DeclContext *OutermostContext = CurContext;
    while (!OutermostContext->isFileContext())
      OutermostContext = OutermostContext->getLookupParent();

    if (PrevDecl &&
        (OutermostContext->Equals(PrevDecl->getDeclContext()) ||
         OutermostContext->Encloses(PrevDecl->getDeclContext()))) {
      SemanticContext = PrevDecl->getDeclContext();
    } else {
      // Declarations in outer scopes don't matter. However, the outermost
      // context we computed is the semantic context for our new 
      // declaration.
      PrevDecl = PrevClassTemplate = 0;
      SemanticContext = OutermostContext;
    }
    
    if (CurContext->isDependentContext()) {
      // If this is a dependent context, we don't want to link the friend
      // class template to the template in scope, because that would perform
      // checking of the template parameter lists that can't be performed
      // until the outer context is instantiated.
      PrevDecl = PrevClassTemplate = 0;
    }
  } else if (PrevDecl && !isDeclInScope(PrevDecl, SemanticContext, S))
    PrevDecl = PrevClassTemplate = 0;

  if (PrevClassTemplate) {
    // Ensure that the template parameter lists are compatible.
    if (!TemplateParameterListsAreEqual(TemplateParams,
                                   PrevClassTemplate->getTemplateParameters(),
                                        /*Complain=*/true,
                                        TPL_TemplateMatch))
      return true;

    // C++ [temp.class]p4:
    //   In a redeclaration, partial specialization, explicit
    //   specialization or explicit instantiation of a class template,
    //   the class-key shall agree in kind with the original class
    //   template declaration (7.1.5.3).
    RecordDecl *PrevRecordDecl = PrevClassTemplate->getTemplatedDecl();
    if (!isAcceptableTagRedeclaration(PrevRecordDecl, Kind, KWLoc, *Name)) {
      Diag(KWLoc, diag::err_use_with_wrong_tag)
        << Name
        << CodeModificationHint::CreateReplacement(KWLoc,
                            PrevRecordDecl->getKindName());
      Diag(PrevRecordDecl->getLocation(), diag::note_previous_use);
      Kind = PrevRecordDecl->getTagKind();
    }

    // Check for redefinition of this class template.
    if (TUK == TUK_Definition) {
      if (TagDecl *Def = PrevRecordDecl->getDefinition(Context)) {
        Diag(NameLoc, diag::err_redefinition) << Name;
        Diag(Def->getLocation(), diag::note_previous_definition);
        // FIXME: Would it make sense to try to "forget" the previous
        // definition, as part of error recovery?
        return true;
      }
    }
  } else if (PrevDecl && PrevDecl->isTemplateParameter()) {
    // Maybe we will complain about the shadowed template parameter.
    DiagnoseTemplateParameterShadow(NameLoc, PrevDecl);
    // Just pretend that we didn't see the previous declaration.
    PrevDecl = 0;
  } else if (PrevDecl) {
    // C++ [temp]p5:
    //   A class template shall not have the same name as any other
    //   template, class, function, object, enumeration, enumerator,
    //   namespace, or type in the same scope (3.3), except as specified
    //   in (14.5.4).
    Diag(NameLoc, diag::err_redefinition_different_kind) << Name;
    Diag(PrevDecl->getLocation(), diag::note_previous_definition);
    return true;
  }

  // Check the template parameter list of this declaration, possibly
  // merging in the template parameter list from the previous class
  // template declaration.
  if (CheckTemplateParameterList(TemplateParams,
            PrevClassTemplate? PrevClassTemplate->getTemplateParameters() : 0,
                                 TPC_ClassTemplate))
    Invalid = true;

  // FIXME: If we had a scope specifier, we better have a previous template
  // declaration!

  CXXRecordDecl *NewClass =
    CXXRecordDecl::Create(Context, Kind, SemanticContext, NameLoc, Name, KWLoc,
                          PrevClassTemplate?
                            PrevClassTemplate->getTemplatedDecl() : 0,
                          /*DelayTypeCreation=*/true);

  ClassTemplateDecl *NewTemplate
    = ClassTemplateDecl::Create(Context, SemanticContext, NameLoc,
                                DeclarationName(Name), TemplateParams,
                                NewClass, PrevClassTemplate);
  NewClass->setDescribedClassTemplate(NewTemplate);

  // Build the type for the class template declaration now.
  QualType T =
    Context.getTypeDeclType(NewClass,
                            PrevClassTemplate?
                              PrevClassTemplate->getTemplatedDecl() : 0);
  assert(T->isDependentType() && "Class template type is not dependent?");
  (void)T;

  // If we are providing an explicit specialization of a member that is a 
  // class template, make a note of that.
  if (PrevClassTemplate && 
      PrevClassTemplate->getInstantiatedFromMemberTemplate())
    PrevClassTemplate->setMemberSpecialization();
  
  // Set the access specifier.
  if (!Invalid && TUK != TUK_Friend)
    SetMemberAccessSpecifier(NewTemplate, PrevClassTemplate, AS);

  // Set the lexical context of these templates
  NewClass->setLexicalDeclContext(CurContext);
  NewTemplate->setLexicalDeclContext(CurContext);

  if (TUK == TUK_Definition)
    NewClass->startDefinition();

  if (Attr)
    ProcessDeclAttributeList(S, NewClass, Attr);

  if (TUK != TUK_Friend)
    PushOnScopeChains(NewTemplate, S);
  else {
    if (PrevClassTemplate && PrevClassTemplate->getAccess() != AS_none) {
      NewTemplate->setAccess(PrevClassTemplate->getAccess());
      NewClass->setAccess(PrevClassTemplate->getAccess());
    }

    NewTemplate->setObjectOfFriendDecl(/* PreviouslyDeclared = */
                                       PrevClassTemplate != NULL);
    
    // Friend templates are visible in fairly strange ways.
    if (!CurContext->isDependentContext()) {
      DeclContext *DC = SemanticContext->getLookupContext();
      DC->makeDeclVisibleInContext(NewTemplate, /* Recoverable = */ false);
      if (Scope *EnclosingScope = getScopeForDeclContext(S, DC))
        PushOnScopeChains(NewTemplate, EnclosingScope,
                          /* AddToContext = */ false);      
    }
    
    FriendDecl *Friend = FriendDecl::Create(Context, CurContext,
                                            NewClass->getLocation(),
                                            NewTemplate,
                                    /*FIXME:*/NewClass->getLocation());
    Friend->setAccess(AS_public);
    CurContext->addDecl(Friend);
  }

  if (Invalid) {
    NewTemplate->setInvalidDecl();
    NewClass->setInvalidDecl();
  }
  return DeclPtrTy::make(NewTemplate);
}

/// \brief Diagnose the presence of a default template argument on a
/// template parameter, which is ill-formed in certain contexts.
///
/// \returns true if the default template argument should be dropped.
static bool DiagnoseDefaultTemplateArgument(Sema &S, 
                                            Sema::TemplateParamListContext TPC,
                                            SourceLocation ParamLoc,
                                            SourceRange DefArgRange) {
  switch (TPC) {
  case Sema::TPC_ClassTemplate:
    return false;

  case Sema::TPC_FunctionTemplate:
    // C++ [temp.param]p9: 
    //   A default template-argument shall not be specified in a
    //   function template declaration or a function template
    //   definition [...]
    // (This sentence is not in C++0x, per DR226).
    if (!S.getLangOptions().CPlusPlus0x)
      S.Diag(ParamLoc, 
             diag::err_template_parameter_default_in_function_template)
        << DefArgRange;
    return false;

  case Sema::TPC_ClassTemplateMember:
    // C++0x [temp.param]p9:
    //   A default template-argument shall not be specified in the
    //   template-parameter-lists of the definition of a member of a
    //   class template that appears outside of the member's class.
    S.Diag(ParamLoc, diag::err_template_parameter_default_template_member)
      << DefArgRange;
    return true;

  case Sema::TPC_FriendFunctionTemplate:
    // C++ [temp.param]p9:
    //   A default template-argument shall not be specified in a
    //   friend template declaration.
    S.Diag(ParamLoc, diag::err_template_parameter_default_friend_template)
      << DefArgRange;
    return true;

    // FIXME: C++0x [temp.param]p9 allows default template-arguments
    // for friend function templates if there is only a single
    // declaration (and it is a definition). Strange!
  }

  return false;
}

/// \brief Checks the validity of a template parameter list, possibly
/// considering the template parameter list from a previous
/// declaration.
///
/// If an "old" template parameter list is provided, it must be
/// equivalent (per TemplateParameterListsAreEqual) to the "new"
/// template parameter list.
///
/// \param NewParams Template parameter list for a new template
/// declaration. This template parameter list will be updated with any
/// default arguments that are carried through from the previous
/// template parameter list.
///
/// \param OldParams If provided, template parameter list from a
/// previous declaration of the same template. Default template
/// arguments will be merged from the old template parameter list to
/// the new template parameter list.
///
/// \param TPC Describes the context in which we are checking the given
/// template parameter list.
///
/// \returns true if an error occurred, false otherwise.
bool Sema::CheckTemplateParameterList(TemplateParameterList *NewParams,
                                      TemplateParameterList *OldParams,
                                      TemplateParamListContext TPC) {
  bool Invalid = false;

  // C++ [temp.param]p10:
  //   The set of default template-arguments available for use with a
  //   template declaration or definition is obtained by merging the
  //   default arguments from the definition (if in scope) and all
  //   declarations in scope in the same way default function
  //   arguments are (8.3.6).
  bool SawDefaultArgument = false;
  SourceLocation PreviousDefaultArgLoc;

  bool SawParameterPack = false;
  SourceLocation ParameterPackLoc;

  // Dummy initialization to avoid warnings.
  TemplateParameterList::iterator OldParam = NewParams->end();
  if (OldParams)
    OldParam = OldParams->begin();

  for (TemplateParameterList::iterator NewParam = NewParams->begin(),
                                    NewParamEnd = NewParams->end();
       NewParam != NewParamEnd; ++NewParam) {
    // Variables used to diagnose redundant default arguments
    bool RedundantDefaultArg = false;
    SourceLocation OldDefaultLoc;
    SourceLocation NewDefaultLoc;

    // Variables used to diagnose missing default arguments
    bool MissingDefaultArg = false;

    // C++0x [temp.param]p11:
    // If a template parameter of a class template is a template parameter pack,
    // it must be the last template parameter.
    if (SawParameterPack) {
      Diag(ParameterPackLoc,
           diag::err_template_param_pack_must_be_last_template_parameter);
      Invalid = true;
    }

    if (TemplateTypeParmDecl *NewTypeParm
          = dyn_cast<TemplateTypeParmDecl>(*NewParam)) {
      // Check the presence of a default argument here.
      if (NewTypeParm->hasDefaultArgument() && 
          DiagnoseDefaultTemplateArgument(*this, TPC, 
                                          NewTypeParm->getLocation(), 
               NewTypeParm->getDefaultArgumentInfo()->getTypeLoc()
                                                       .getFullSourceRange()))
        NewTypeParm->removeDefaultArgument();

      // Merge default arguments for template type parameters.
      TemplateTypeParmDecl *OldTypeParm
          = OldParams? cast<TemplateTypeParmDecl>(*OldParam) : 0;

      if (NewTypeParm->isParameterPack()) {
        assert(!NewTypeParm->hasDefaultArgument() &&
               "Parameter packs can't have a default argument!");
        SawParameterPack = true;
        ParameterPackLoc = NewTypeParm->getLocation();
      } else if (OldTypeParm && OldTypeParm->hasDefaultArgument() &&
                 NewTypeParm->hasDefaultArgument()) {
        OldDefaultLoc = OldTypeParm->getDefaultArgumentLoc();
        NewDefaultLoc = NewTypeParm->getDefaultArgumentLoc();
        SawDefaultArgument = true;
        RedundantDefaultArg = true;
        PreviousDefaultArgLoc = NewDefaultLoc;
      } else if (OldTypeParm && OldTypeParm->hasDefaultArgument()) {
        // Merge the default argument from the old declaration to the
        // new declaration.
        SawDefaultArgument = true;
        NewTypeParm->setDefaultArgument(OldTypeParm->getDefaultArgumentInfo(),
                                        true);
        PreviousDefaultArgLoc = OldTypeParm->getDefaultArgumentLoc();
      } else if (NewTypeParm->hasDefaultArgument()) {
        SawDefaultArgument = true;
        PreviousDefaultArgLoc = NewTypeParm->getDefaultArgumentLoc();
      } else if (SawDefaultArgument)
        MissingDefaultArg = true;
    } else if (NonTypeTemplateParmDecl *NewNonTypeParm
               = dyn_cast<NonTypeTemplateParmDecl>(*NewParam)) {
      // Check the presence of a default argument here.
      if (NewNonTypeParm->hasDefaultArgument() && 
          DiagnoseDefaultTemplateArgument(*this, TPC, 
                                          NewNonTypeParm->getLocation(), 
                    NewNonTypeParm->getDefaultArgument()->getSourceRange())) {
        NewNonTypeParm->getDefaultArgument()->Destroy(Context);
        NewNonTypeParm->setDefaultArgument(0);
      }

      // Merge default arguments for non-type template parameters
      NonTypeTemplateParmDecl *OldNonTypeParm
        = OldParams? cast<NonTypeTemplateParmDecl>(*OldParam) : 0;
      if (OldNonTypeParm && OldNonTypeParm->hasDefaultArgument() &&
          NewNonTypeParm->hasDefaultArgument()) {
        OldDefaultLoc = OldNonTypeParm->getDefaultArgumentLoc();
        NewDefaultLoc = NewNonTypeParm->getDefaultArgumentLoc();
        SawDefaultArgument = true;
        RedundantDefaultArg = true;
        PreviousDefaultArgLoc = NewDefaultLoc;
      } else if (OldNonTypeParm && OldNonTypeParm->hasDefaultArgument()) {
        // Merge the default argument from the old declaration to the
        // new declaration.
        SawDefaultArgument = true;
        // FIXME: We need to create a new kind of "default argument"
        // expression that points to a previous template template
        // parameter.
        NewNonTypeParm->setDefaultArgument(
                                        OldNonTypeParm->getDefaultArgument());
        PreviousDefaultArgLoc = OldNonTypeParm->getDefaultArgumentLoc();
      } else if (NewNonTypeParm->hasDefaultArgument()) {
        SawDefaultArgument = true;
        PreviousDefaultArgLoc = NewNonTypeParm->getDefaultArgumentLoc();
      } else if (SawDefaultArgument)
        MissingDefaultArg = true;
    } else {
      // Check the presence of a default argument here.
      TemplateTemplateParmDecl *NewTemplateParm
        = cast<TemplateTemplateParmDecl>(*NewParam);
      if (NewTemplateParm->hasDefaultArgument() && 
          DiagnoseDefaultTemplateArgument(*this, TPC, 
                                          NewTemplateParm->getLocation(), 
                     NewTemplateParm->getDefaultArgument().getSourceRange()))
        NewTemplateParm->setDefaultArgument(TemplateArgumentLoc());

      // Merge default arguments for template template parameters
      TemplateTemplateParmDecl *OldTemplateParm
        = OldParams? cast<TemplateTemplateParmDecl>(*OldParam) : 0;
      if (OldTemplateParm && OldTemplateParm->hasDefaultArgument() &&
          NewTemplateParm->hasDefaultArgument()) {
        OldDefaultLoc = OldTemplateParm->getDefaultArgument().getLocation();
        NewDefaultLoc = NewTemplateParm->getDefaultArgument().getLocation();
        SawDefaultArgument = true;
        RedundantDefaultArg = true;
        PreviousDefaultArgLoc = NewDefaultLoc;
      } else if (OldTemplateParm && OldTemplateParm->hasDefaultArgument()) {
        // Merge the default argument from the old declaration to the
        // new declaration.
        SawDefaultArgument = true;
        // FIXME: We need to create a new kind of "default argument" expression
        // that points to a previous template template parameter.
        NewTemplateParm->setDefaultArgument(
                                        OldTemplateParm->getDefaultArgument());
        PreviousDefaultArgLoc
          = OldTemplateParm->getDefaultArgument().getLocation();
      } else if (NewTemplateParm->hasDefaultArgument()) {
        SawDefaultArgument = true;
        PreviousDefaultArgLoc
          = NewTemplateParm->getDefaultArgument().getLocation();
      } else if (SawDefaultArgument)
        MissingDefaultArg = true;
    }

    if (RedundantDefaultArg) {
      // C++ [temp.param]p12:
      //   A template-parameter shall not be given default arguments
      //   by two different declarations in the same scope.
      Diag(NewDefaultLoc, diag::err_template_param_default_arg_redefinition);
      Diag(OldDefaultLoc, diag::note_template_param_prev_default_arg);
      Invalid = true;
    } else if (MissingDefaultArg) {
      // C++ [temp.param]p11:
      //   If a template-parameter has a default template-argument,
      //   all subsequent template-parameters shall have a default
      //   template-argument supplied.
      Diag((*NewParam)->getLocation(),
           diag::err_template_param_default_arg_missing);
      Diag(PreviousDefaultArgLoc, diag::note_template_param_prev_default_arg);
      Invalid = true;
    }

    // If we have an old template parameter list that we're merging
    // in, move on to the next parameter.
    if (OldParams)
      ++OldParam;
  }

  return Invalid;
}

/// \brief Match the given template parameter lists to the given scope
/// specifier, returning the template parameter list that applies to the
/// name.
///
/// \param DeclStartLoc the start of the declaration that has a scope
/// specifier or a template parameter list.
///
/// \param SS the scope specifier that will be matched to the given template
/// parameter lists. This scope specifier precedes a qualified name that is
/// being declared.
///
/// \param ParamLists the template parameter lists, from the outermost to the
/// innermost template parameter lists.
///
/// \param NumParamLists the number of template parameter lists in ParamLists.
///
/// \param IsExplicitSpecialization will be set true if the entity being
/// declared is an explicit specialization, false otherwise.
///
/// \returns the template parameter list, if any, that corresponds to the
/// name that is preceded by the scope specifier @p SS. This template
/// parameter list may be have template parameters (if we're declaring a
/// template) or may have no template parameters (if we're declaring a
/// template specialization), or may be NULL (if we were's declaring isn't
/// itself a template).
TemplateParameterList *
Sema::MatchTemplateParametersToScopeSpecifier(SourceLocation DeclStartLoc,
                                              const CXXScopeSpec &SS,
                                          TemplateParameterList **ParamLists,
                                              unsigned NumParamLists,
                                              bool &IsExplicitSpecialization) {
  IsExplicitSpecialization = false;
  
  // Find the template-ids that occur within the nested-name-specifier. These
  // template-ids will match up with the template parameter lists.
  llvm::SmallVector<const TemplateSpecializationType *, 4>
    TemplateIdsInSpecifier;
  llvm::SmallVector<ClassTemplateSpecializationDecl *, 4>
    ExplicitSpecializationsInSpecifier;
  for (NestedNameSpecifier *NNS = (NestedNameSpecifier *)SS.getScopeRep();
       NNS; NNS = NNS->getPrefix()) {
    const Type *T = NNS->getAsType();
    if (!T) break;

    // C++0x [temp.expl.spec]p17:
    //   A member or a member template may be nested within many
    //   enclosing class templates. In an explicit specialization for
    //   such a member, the member declaration shall be preceded by a
    //   template<> for each enclosing class template that is
    //   explicitly specialized.
    // We interpret this as forbidding typedefs of template
    // specializations in the scope specifiers of out-of-line decls.
    if (const TypedefType *TT = dyn_cast<TypedefType>(T)) {
      const Type *UnderlyingT = TT->LookThroughTypedefs().getTypePtr();
      if (isa<TemplateSpecializationType>(UnderlyingT))
        // FIXME: better source location information.
        Diag(DeclStartLoc, diag::err_typedef_in_def_scope) << QualType(T,0);
      T = UnderlyingT;
    }

    if (const TemplateSpecializationType *SpecType
          = dyn_cast<TemplateSpecializationType>(T)) {
      TemplateDecl *Template = SpecType->getTemplateName().getAsTemplateDecl();
      if (!Template)
        continue; // FIXME: should this be an error? probably...

      if (const RecordType *Record = SpecType->getAs<RecordType>()) {
        ClassTemplateSpecializationDecl *SpecDecl
          = cast<ClassTemplateSpecializationDecl>(Record->getDecl());
        // If the nested name specifier refers to an explicit specialization,
        // we don't need a template<> header.
        if (SpecDecl->getSpecializationKind() == TSK_ExplicitSpecialization) {
          ExplicitSpecializationsInSpecifier.push_back(SpecDecl);
          continue;
        }
      }

      TemplateIdsInSpecifier.push_back(SpecType);
    }
  }

  // Reverse the list of template-ids in the scope specifier, so that we can
  // more easily match up the template-ids and the template parameter lists.
  std::reverse(TemplateIdsInSpecifier.begin(), TemplateIdsInSpecifier.end());

  SourceLocation FirstTemplateLoc = DeclStartLoc;
  if (NumParamLists)
    FirstTemplateLoc = ParamLists[0]->getTemplateLoc();

  // Match the template-ids found in the specifier to the template parameter
  // lists.
  unsigned Idx = 0;
  for (unsigned NumTemplateIds = TemplateIdsInSpecifier.size();
       Idx != NumTemplateIds; ++Idx) {
    QualType TemplateId = QualType(TemplateIdsInSpecifier[Idx], 0);
    bool DependentTemplateId = TemplateId->isDependentType();
    if (Idx >= NumParamLists) {
      // We have a template-id without a corresponding template parameter
      // list.
      if (DependentTemplateId) {
        // FIXME: the location information here isn't great.
        Diag(SS.getRange().getBegin(),
             diag::err_template_spec_needs_template_parameters)
          << TemplateId
          << SS.getRange();
      } else {
        Diag(SS.getRange().getBegin(), diag::err_template_spec_needs_header)
          << SS.getRange()
          << CodeModificationHint::CreateInsertion(FirstTemplateLoc,
                                                   "template<> ");
        IsExplicitSpecialization = true;
      }
      return 0;
    }

    // Check the template parameter list against its corresponding template-id.
    if (DependentTemplateId) {
      TemplateDecl *Template
        = TemplateIdsInSpecifier[Idx]->getTemplateName().getAsTemplateDecl();

      if (ClassTemplateDecl *ClassTemplate
            = dyn_cast<ClassTemplateDecl>(Template)) {
        TemplateParameterList *ExpectedTemplateParams = 0;
        // Is this template-id naming the primary template?
        if (Context.hasSameType(TemplateId,
                             ClassTemplate->getInjectedClassNameType(Context)))
          ExpectedTemplateParams = ClassTemplate->getTemplateParameters();
        // ... or a partial specialization?
        else if (ClassTemplatePartialSpecializationDecl *PartialSpec
                   = ClassTemplate->findPartialSpecialization(TemplateId))
          ExpectedTemplateParams = PartialSpec->getTemplateParameters();

        if (ExpectedTemplateParams)
          TemplateParameterListsAreEqual(ParamLists[Idx],
                                         ExpectedTemplateParams,
                                         true, TPL_TemplateMatch);
      }

      CheckTemplateParameterList(ParamLists[Idx], 0, TPC_ClassTemplateMember);
    } else if (ParamLists[Idx]->size() > 0)
      Diag(ParamLists[Idx]->getTemplateLoc(),
           diag::err_template_param_list_matches_nontemplate)
        << TemplateId
        << ParamLists[Idx]->getSourceRange();
    else
      IsExplicitSpecialization = true;
  }

  // If there were at least as many template-ids as there were template
  // parameter lists, then there are no template parameter lists remaining for
  // the declaration itself.
  if (Idx >= NumParamLists)
    return 0;

  // If there were too many template parameter lists, complain about that now.
  if (Idx != NumParamLists - 1) {
    while (Idx < NumParamLists - 1) {
      bool isExplicitSpecHeader = ParamLists[Idx]->size() == 0;
      Diag(ParamLists[Idx]->getTemplateLoc(),
           isExplicitSpecHeader? diag::warn_template_spec_extra_headers
                               : diag::err_template_spec_extra_headers)
        << SourceRange(ParamLists[Idx]->getTemplateLoc(),
                       ParamLists[Idx]->getRAngleLoc());

      if (isExplicitSpecHeader && !ExplicitSpecializationsInSpecifier.empty()) {
        Diag(ExplicitSpecializationsInSpecifier.back()->getLocation(),
             diag::note_explicit_template_spec_does_not_need_header)
          << ExplicitSpecializationsInSpecifier.back();
        ExplicitSpecializationsInSpecifier.pop_back();
      }
        
      ++Idx;
    }
  }

  // Return the last template parameter list, which corresponds to the
  // entity being declared.
  return ParamLists[NumParamLists - 1];
}

QualType Sema::CheckTemplateIdType(TemplateName Name,
                                   SourceLocation TemplateLoc,
                              const TemplateArgumentListInfo &TemplateArgs) {
  TemplateDecl *Template = Name.getAsTemplateDecl();
  if (!Template) {
    // The template name does not resolve to a template, so we just
    // build a dependent template-id type.
    return Context.getTemplateSpecializationType(Name, TemplateArgs);
  }

  // Check that the template argument list is well-formed for this
  // template.
  TemplateArgumentListBuilder Converted(Template->getTemplateParameters(),
                                        TemplateArgs.size());
  if (CheckTemplateArgumentList(Template, TemplateLoc, TemplateArgs,
                                false, Converted))
    return QualType();

  assert((Converted.structuredSize() ==
            Template->getTemplateParameters()->size()) &&
         "Converted template argument list is too short!");

  QualType CanonType;

  if (Name.isDependent() ||
      TemplateSpecializationType::anyDependentTemplateArguments(
                                                      TemplateArgs)) {
    // This class template specialization is a dependent
    // type. Therefore, its canonical type is another class template
    // specialization type that contains all of the converted
    // arguments in canonical form. This ensures that, e.g., A<T> and
    // A<T, T> have identical types when A is declared as:
    //
    //   template<typename T, typename U = T> struct A;
    TemplateName CanonName = Context.getCanonicalTemplateName(Name);
    CanonType = Context.getTemplateSpecializationType(CanonName,
                                                   Converted.getFlatArguments(),
                                                   Converted.flatSize());

    // FIXME: CanonType is not actually the canonical type, and unfortunately
    // it is a TemplateSpecializationType that we will never use again.
    // In the future, we need to teach getTemplateSpecializationType to only
    // build the canonical type and return that to us.
    CanonType = Context.getCanonicalType(CanonType);
  } else if (ClassTemplateDecl *ClassTemplate
               = dyn_cast<ClassTemplateDecl>(Template)) {
    // Find the class template specialization declaration that
    // corresponds to these arguments.
    llvm::FoldingSetNodeID ID;
    ClassTemplateSpecializationDecl::Profile(ID,
                                             Converted.getFlatArguments(),
                                             Converted.flatSize(),
                                             Context);
    void *InsertPos = 0;
    ClassTemplateSpecializationDecl *Decl
      = ClassTemplate->getSpecializations().FindNodeOrInsertPos(ID, InsertPos);
    if (!Decl) {
      // This is the first time we have referenced this class template
      // specialization. Create the canonical declaration and add it to
      // the set of specializations.
      Decl = ClassTemplateSpecializationDecl::Create(Context,
                                    ClassTemplate->getDeclContext(),
                                    ClassTemplate->getLocation(),
                                    ClassTemplate,
                                    Converted, 0);
      ClassTemplate->getSpecializations().InsertNode(Decl, InsertPos);
      Decl->setLexicalDeclContext(CurContext);
    }

    CanonType = Context.getTypeDeclType(Decl);
  }

  // Build the fully-sugared type for this class template
  // specialization, which refers back to the class template
  // specialization we created or found.
  return Context.getTemplateSpecializationType(Name, TemplateArgs, CanonType);
}

Action::TypeResult
Sema::ActOnTemplateIdType(TemplateTy TemplateD, SourceLocation TemplateLoc,
                          SourceLocation LAngleLoc,
                          ASTTemplateArgsPtr TemplateArgsIn,
                          SourceLocation RAngleLoc) {
  TemplateName Template = TemplateD.getAsVal<TemplateName>();

  // Translate the parser's template argument list in our AST format.
  TemplateArgumentListInfo TemplateArgs(LAngleLoc, RAngleLoc);
  translateTemplateArguments(TemplateArgsIn, TemplateArgs);

  QualType Result = CheckTemplateIdType(Template, TemplateLoc, TemplateArgs);
  TemplateArgsIn.release();

  if (Result.isNull())
    return true;

  TypeSourceInfo *DI = Context.CreateTypeSourceInfo(Result);
  TemplateSpecializationTypeLoc TL
    = cast<TemplateSpecializationTypeLoc>(DI->getTypeLoc());
  TL.setTemplateNameLoc(TemplateLoc);
  TL.setLAngleLoc(LAngleLoc);
  TL.setRAngleLoc(RAngleLoc);
  for (unsigned i = 0, e = TL.getNumArgs(); i != e; ++i)
    TL.setArgLocInfo(i, TemplateArgs[i].getLocInfo());

  return CreateLocInfoType(Result, DI).getAsOpaquePtr();
}

Sema::TypeResult Sema::ActOnTagTemplateIdType(TypeResult TypeResult,
                                              TagUseKind TUK,
                                              DeclSpec::TST TagSpec,
                                              SourceLocation TagLoc) {
  if (TypeResult.isInvalid())
    return Sema::TypeResult();

  // FIXME: preserve source info, ideally without copying the DI.
  TypeSourceInfo *DI;
  QualType Type = GetTypeFromParser(TypeResult.get(), &DI);

  // Verify the tag specifier.
  TagDecl::TagKind TagKind = TagDecl::getTagKindForTypeSpec(TagSpec);

  if (const RecordType *RT = Type->getAs<RecordType>()) {
    RecordDecl *D = RT->getDecl();

    IdentifierInfo *Id = D->getIdentifier();
    assert(Id && "templated class must have an identifier");

    if (!isAcceptableTagRedeclaration(D, TagKind, TagLoc, *Id)) {
      Diag(TagLoc, diag::err_use_with_wrong_tag)
        << Type
        << CodeModificationHint::CreateReplacement(SourceRange(TagLoc),
                                                   D->getKindName());
      Diag(D->getLocation(), diag::note_previous_use);
    }
  }

  QualType ElabType = Context.getElaboratedType(Type, TagKind);

  return ElabType.getAsOpaquePtr();
}

Sema::OwningExprResult Sema::BuildTemplateIdExpr(const CXXScopeSpec &SS,
                                                 LookupResult &R,
                                                 bool RequiresADL,
                                 const TemplateArgumentListInfo &TemplateArgs) {
  // FIXME: Can we do any checking at this point? I guess we could check the
  // template arguments that we have against the template name, if the template
  // name refers to a single template. That's not a terribly common case,
  // though.

  // These should be filtered out by our callers.
  assert(!R.empty() && "empty lookup results when building templateid");
  assert(!R.isAmbiguous() && "ambiguous lookup when building templateid");

  NestedNameSpecifier *Qualifier = 0;
  SourceRange QualifierRange;
  if (SS.isSet()) {
    Qualifier = static_cast<NestedNameSpecifier*>(SS.getScopeRep());
    QualifierRange = SS.getRange();
  }
  
  bool Dependent
    = UnresolvedLookupExpr::ComputeDependence(R.begin(), R.end(),
                                              &TemplateArgs);
  UnresolvedLookupExpr *ULE
    = UnresolvedLookupExpr::Create(Context, Dependent,
                                   Qualifier, QualifierRange,
                                   R.getLookupName(), R.getNameLoc(),
                                   RequiresADL, TemplateArgs);
  for (LookupResult::iterator I = R.begin(), E = R.end(); I != E; ++I)
    ULE->addDecl(*I);

  return Owned(ULE);
}

// We actually only call this from template instantiation.
Sema::OwningExprResult
Sema::BuildQualifiedTemplateIdExpr(const CXXScopeSpec &SS,
                                   DeclarationName Name,
                                   SourceLocation NameLoc,
                             const TemplateArgumentListInfo &TemplateArgs) {
  DeclContext *DC;
  if (!(DC = computeDeclContext(SS, false)) ||
      DC->isDependentContext() ||
      RequireCompleteDeclContext(SS))
    return BuildDependentDeclRefExpr(SS, Name, NameLoc, &TemplateArgs);

  LookupResult R(*this, Name, NameLoc, LookupOrdinaryName);
  LookupTemplateName(R, (Scope*) 0, SS, QualType(), /*Entering*/ false);

  if (R.isAmbiguous())
    return ExprError();
  
  if (R.empty()) {
    Diag(NameLoc, diag::err_template_kw_refers_to_non_template)
      << Name << SS.getRange();
    return ExprError();
  }

  if (ClassTemplateDecl *Temp = R.getAsSingle<ClassTemplateDecl>()) {
    Diag(NameLoc, diag::err_template_kw_refers_to_class_template)
      << (NestedNameSpecifier*) SS.getScopeRep() << Name << SS.getRange();
    Diag(Temp->getLocation(), diag::note_referenced_class_template);
    return ExprError();
  }

  return BuildTemplateIdExpr(SS, R, /* ADL */ false, TemplateArgs);
}

/// \brief Form a dependent template name.
///
/// This action forms a dependent template name given the template
/// name and its (presumably dependent) scope specifier. For
/// example, given "MetaFun::template apply", the scope specifier \p
/// SS will be "MetaFun::", \p TemplateKWLoc contains the location
/// of the "template" keyword, and "apply" is the \p Name.
Sema::TemplateTy
Sema::ActOnDependentTemplateName(SourceLocation TemplateKWLoc,
                                 const CXXScopeSpec &SS,
                                 UnqualifiedId &Name,
                                 TypeTy *ObjectType,
                                 bool EnteringContext) {
  if ((ObjectType &&
       computeDeclContext(QualType::getFromOpaquePtr(ObjectType))) ||
      (SS.isSet() && computeDeclContext(SS, EnteringContext))) {
    // C++0x [temp.names]p5:
    //   If a name prefixed by the keyword template is not the name of
    //   a template, the program is ill-formed. [Note: the keyword
    //   template may not be applied to non-template members of class
    //   templates. -end note ] [ Note: as is the case with the
    //   typename prefix, the template prefix is allowed in cases
    //   where it is not strictly necessary; i.e., when the
    //   nested-name-specifier or the expression on the left of the ->
    //   or . is not dependent on a template-parameter, or the use
    //   does not appear in the scope of a template. -end note]
    //
    // Note: C++03 was more strict here, because it banned the use of
    // the "template" keyword prior to a template-name that was not a
    // dependent name. C++ DR468 relaxed this requirement (the
    // "template" keyword is now permitted). We follow the C++0x
    // rules, even in C++03 mode, retroactively applying the DR.
    TemplateTy Template;
    TemplateNameKind TNK = isTemplateName(0, SS, Name, ObjectType,
                                          EnteringContext, Template);
    if (TNK == TNK_Non_template) {
      Diag(Name.getSourceRange().getBegin(), 
           diag::err_template_kw_refers_to_non_template)
        << GetNameFromUnqualifiedId(Name)
        << Name.getSourceRange();
      return TemplateTy();
    }

    return Template;
  }

  NestedNameSpecifier *Qualifier
    = static_cast<NestedNameSpecifier *>(SS.getScopeRep());
  
  switch (Name.getKind()) {
  case UnqualifiedId::IK_Identifier:
    return TemplateTy::make(Context.getDependentTemplateName(Qualifier, 
                                                             Name.Identifier));
    
  case UnqualifiedId::IK_OperatorFunctionId:
    return TemplateTy::make(Context.getDependentTemplateName(Qualifier,
                                             Name.OperatorFunctionId.Operator));

  case UnqualifiedId::IK_LiteralOperatorId:
    assert(false && "We don't support these; Parse shouldn't have allowed propagation");

  default:
    break;
  }
  
  Diag(Name.getSourceRange().getBegin(), 
       diag::err_template_kw_refers_to_non_template)
    << GetNameFromUnqualifiedId(Name)
    << Name.getSourceRange();
  return TemplateTy();
}

bool Sema::CheckTemplateTypeArgument(TemplateTypeParmDecl *Param,
                                     const TemplateArgumentLoc &AL,
                                     TemplateArgumentListBuilder &Converted) {
  const TemplateArgument &Arg = AL.getArgument();

  // Check template type parameter.
  if (Arg.getKind() != TemplateArgument::Type) {
    // C++ [temp.arg.type]p1:
    //   A template-argument for a template-parameter which is a
    //   type shall be a type-id.

    // We have a template type parameter but the template argument
    // is not a type.
    SourceRange SR = AL.getSourceRange();
    Diag(SR.getBegin(), diag::err_template_arg_must_be_type) << SR;
    Diag(Param->getLocation(), diag::note_template_param_here);

    return true;
  }

  if (CheckTemplateArgument(Param, AL.getTypeSourceInfo()))
    return true;

  // Add the converted template type argument.
  Converted.Append(
                 TemplateArgument(Context.getCanonicalType(Arg.getAsType())));
  return false;
}

/// \brief Substitute template arguments into the default template argument for
/// the given template type parameter.
///
/// \param SemaRef the semantic analysis object for which we are performing
/// the substitution.
///
/// \param Template the template that we are synthesizing template arguments 
/// for.
///
/// \param TemplateLoc the location of the template name that started the
/// template-id we are checking.
///
/// \param RAngleLoc the location of the right angle bracket ('>') that
/// terminates the template-id.
///
/// \param Param the template template parameter whose default we are
/// substituting into.
///
/// \param Converted the list of template arguments provided for template
/// parameters that precede \p Param in the template parameter list.
///
/// \returns the substituted template argument, or NULL if an error occurred.
static TypeSourceInfo *
SubstDefaultTemplateArgument(Sema &SemaRef,
                             TemplateDecl *Template,
                             SourceLocation TemplateLoc,
                             SourceLocation RAngleLoc,
                             TemplateTypeParmDecl *Param,
                             TemplateArgumentListBuilder &Converted) {
  TypeSourceInfo *ArgType = Param->getDefaultArgumentInfo();

  // If the argument type is dependent, instantiate it now based
  // on the previously-computed template arguments.
  if (ArgType->getType()->isDependentType()) {
    TemplateArgumentList TemplateArgs(SemaRef.Context, Converted,
                                      /*TakeArgs=*/false);
    
    MultiLevelTemplateArgumentList AllTemplateArgs
      = SemaRef.getTemplateInstantiationArgs(Template, &TemplateArgs);

    Sema::InstantiatingTemplate Inst(SemaRef, TemplateLoc,
                                     Template, Converted.getFlatArguments(),
                                     Converted.flatSize(),
                                     SourceRange(TemplateLoc, RAngleLoc));
    
    ArgType = SemaRef.SubstType(ArgType, AllTemplateArgs,
                                Param->getDefaultArgumentLoc(),
                                Param->getDeclName());
  }

  return ArgType;
}

/// \brief Substitute template arguments into the default template argument for
/// the given non-type template parameter.
///
/// \param SemaRef the semantic analysis object for which we are performing
/// the substitution.
///
/// \param Template the template that we are synthesizing template arguments 
/// for.
///
/// \param TemplateLoc the location of the template name that started the
/// template-id we are checking.
///
/// \param RAngleLoc the location of the right angle bracket ('>') that
/// terminates the template-id.
///
/// \param Param the non-type template parameter whose default we are
/// substituting into.
///
/// \param Converted the list of template arguments provided for template
/// parameters that precede \p Param in the template parameter list.
///
/// \returns the substituted template argument, or NULL if an error occurred.
static Sema::OwningExprResult
SubstDefaultTemplateArgument(Sema &SemaRef,
                             TemplateDecl *Template,
                             SourceLocation TemplateLoc,
                             SourceLocation RAngleLoc,
                             NonTypeTemplateParmDecl *Param,
                             TemplateArgumentListBuilder &Converted) {
  TemplateArgumentList TemplateArgs(SemaRef.Context, Converted,
                                    /*TakeArgs=*/false);
    
  MultiLevelTemplateArgumentList AllTemplateArgs
    = SemaRef.getTemplateInstantiationArgs(Template, &TemplateArgs);
    
  Sema::InstantiatingTemplate Inst(SemaRef, TemplateLoc,
                                   Template, Converted.getFlatArguments(),
                                   Converted.flatSize(),
                                   SourceRange(TemplateLoc, RAngleLoc));

  return SemaRef.SubstExpr(Param->getDefaultArgument(), AllTemplateArgs);
}

/// \brief Substitute template arguments into the default template argument for
/// the given template template parameter.
///
/// \param SemaRef the semantic analysis object for which we are performing
/// the substitution.
///
/// \param Template the template that we are synthesizing template arguments 
/// for.
///
/// \param TemplateLoc the location of the template name that started the
/// template-id we are checking.
///
/// \param RAngleLoc the location of the right angle bracket ('>') that
/// terminates the template-id.
///
/// \param Param the template template parameter whose default we are
/// substituting into.
///
/// \param Converted the list of template arguments provided for template
/// parameters that precede \p Param in the template parameter list.
///
/// \returns the substituted template argument, or NULL if an error occurred.
static TemplateName
SubstDefaultTemplateArgument(Sema &SemaRef,
                             TemplateDecl *Template,
                             SourceLocation TemplateLoc,
                             SourceLocation RAngleLoc,
                             TemplateTemplateParmDecl *Param,
                             TemplateArgumentListBuilder &Converted) {
  TemplateArgumentList TemplateArgs(SemaRef.Context, Converted,
                                    /*TakeArgs=*/false);
  
  MultiLevelTemplateArgumentList AllTemplateArgs
    = SemaRef.getTemplateInstantiationArgs(Template, &TemplateArgs);
  
  Sema::InstantiatingTemplate Inst(SemaRef, TemplateLoc,
                                   Template, Converted.getFlatArguments(),
                                   Converted.flatSize(),
                                   SourceRange(TemplateLoc, RAngleLoc));
  
  return SemaRef.SubstTemplateName(
                      Param->getDefaultArgument().getArgument().getAsTemplate(),
                              Param->getDefaultArgument().getTemplateNameLoc(), 
                                   AllTemplateArgs);
}

/// \brief If the given template parameter has a default template
/// argument, substitute into that default template argument and
/// return the corresponding template argument.
TemplateArgumentLoc 
Sema::SubstDefaultTemplateArgumentIfAvailable(TemplateDecl *Template,
                                              SourceLocation TemplateLoc,
                                              SourceLocation RAngleLoc,
                                              Decl *Param,
                                     TemplateArgumentListBuilder &Converted) {
  if (TemplateTypeParmDecl *TypeParm = dyn_cast<TemplateTypeParmDecl>(Param)) {
    if (!TypeParm->hasDefaultArgument())
      return TemplateArgumentLoc();

    TypeSourceInfo *DI = SubstDefaultTemplateArgument(*this, Template,
                                                      TemplateLoc,
                                                      RAngleLoc,
                                                      TypeParm,
                                                      Converted);
    if (DI)
      return TemplateArgumentLoc(TemplateArgument(DI->getType()), DI);

    return TemplateArgumentLoc();
  }

  if (NonTypeTemplateParmDecl *NonTypeParm
        = dyn_cast<NonTypeTemplateParmDecl>(Param)) {
    if (!NonTypeParm->hasDefaultArgument())
      return TemplateArgumentLoc();

    OwningExprResult Arg = SubstDefaultTemplateArgument(*this, Template,
                                                        TemplateLoc,
                                                        RAngleLoc,
                                                        NonTypeParm,
                                                        Converted);
    if (Arg.isInvalid())
      return TemplateArgumentLoc();

    Expr *ArgE = Arg.takeAs<Expr>();
    return TemplateArgumentLoc(TemplateArgument(ArgE), ArgE);
  }

  TemplateTemplateParmDecl *TempTempParm
    = cast<TemplateTemplateParmDecl>(Param);
  if (!TempTempParm->hasDefaultArgument())
    return TemplateArgumentLoc();

  TemplateName TName = SubstDefaultTemplateArgument(*this, Template,
                                                    TemplateLoc, 
                                                    RAngleLoc,
                                                    TempTempParm,
                                                    Converted);
  if (TName.isNull())
    return TemplateArgumentLoc();

  return TemplateArgumentLoc(TemplateArgument(TName), 
                TempTempParm->getDefaultArgument().getTemplateQualifierRange(),
                TempTempParm->getDefaultArgument().getTemplateNameLoc());
}

/// \brief Check that the given template argument corresponds to the given
/// template parameter.
bool Sema::CheckTemplateArgument(NamedDecl *Param,
                                 const TemplateArgumentLoc &Arg,
                                 TemplateDecl *Template,
                                 SourceLocation TemplateLoc,
                                 SourceLocation RAngleLoc,
                                 TemplateArgumentListBuilder &Converted) {
  // Check template type parameters.
  if (TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(Param))
    return CheckTemplateTypeArgument(TTP, Arg, Converted);
  
  // Check non-type template parameters.
  if (NonTypeTemplateParmDecl *NTTP =dyn_cast<NonTypeTemplateParmDecl>(Param)) {    
    // Do substitution on the type of the non-type template parameter
    // with the template arguments we've seen thus far.
    QualType NTTPType = NTTP->getType();
    if (NTTPType->isDependentType()) {
      // Do substitution on the type of the non-type template parameter.
      InstantiatingTemplate Inst(*this, TemplateLoc, Template,
                                 NTTP, Converted.getFlatArguments(),
                                 Converted.flatSize(),
                                 SourceRange(TemplateLoc, RAngleLoc));
      
      TemplateArgumentList TemplateArgs(Context, Converted,
                                        /*TakeArgs=*/false);
      NTTPType = SubstType(NTTPType,
                           MultiLevelTemplateArgumentList(TemplateArgs),
                           NTTP->getLocation(),
                           NTTP->getDeclName());
      // If that worked, check the non-type template parameter type
      // for validity.
      if (!NTTPType.isNull())
        NTTPType = CheckNonTypeTemplateParameterType(NTTPType,
                                                     NTTP->getLocation());
      if (NTTPType.isNull())
        return true;
    }
    
    switch (Arg.getArgument().getKind()) {
    case TemplateArgument::Null:
      assert(false && "Should never see a NULL template argument here");
      return true;
      
    case TemplateArgument::Expression: {
      Expr *E = Arg.getArgument().getAsExpr();
      TemplateArgument Result;
      if (CheckTemplateArgument(NTTP, NTTPType, E, Result))
        return true;
      
      Converted.Append(Result);
      break;
    }
      
    case TemplateArgument::Declaration:
    case TemplateArgument::Integral:
      // We've already checked this template argument, so just copy
      // it to the list of converted arguments.
      Converted.Append(Arg.getArgument());
      break;
      
    case TemplateArgument::Template:
      // We were given a template template argument. It may not be ill-formed;
      // see below.
      if (DependentTemplateName *DTN
            = Arg.getArgument().getAsTemplate().getAsDependentTemplateName()) {
        // We have a template argument such as \c T::template X, which we
        // parsed as a template template argument. However, since we now
        // know that we need a non-type template argument, convert this
        // template name into an expression.          
        Expr *E = DependentScopeDeclRefExpr::Create(Context,
                                                    DTN->getQualifier(),
                                               Arg.getTemplateQualifierRange(),
                                                    DTN->getIdentifier(),
                                                    Arg.getTemplateNameLoc());
        
        TemplateArgument Result;
        if (CheckTemplateArgument(NTTP, NTTPType, E, Result))
          return true;
        
        Converted.Append(Result);
        break;
      }
      
      // We have a template argument that actually does refer to a class
      // template, template alias, or template template parameter, and
      // therefore cannot be a non-type template argument.
      Diag(Arg.getLocation(), diag::err_template_arg_must_be_expr)
        << Arg.getSourceRange();
      
      Diag(Param->getLocation(), diag::note_template_param_here);
      return true;
      
    case TemplateArgument::Type: {
      // We have a non-type template parameter but the template
      // argument is a type.
      
      // C++ [temp.arg]p2:
      //   In a template-argument, an ambiguity between a type-id and
      //   an expression is resolved to a type-id, regardless of the
      //   form of the corresponding template-parameter.
      //
      // We warn specifically about this case, since it can be rather
      // confusing for users.
      QualType T = Arg.getArgument().getAsType();
      SourceRange SR = Arg.getSourceRange();
      if (T->isFunctionType())
        Diag(SR.getBegin(), diag::err_template_arg_nontype_ambig) << SR << T;
      else
        Diag(SR.getBegin(), diag::err_template_arg_must_be_expr) << SR;
      Diag(Param->getLocation(), diag::note_template_param_here);
      return true;
    }
      
    case TemplateArgument::Pack:
      llvm_unreachable("Caller must expand template argument packs");
      break;
    }
    
    return false;
  } 
  
  
  // Check template template parameters.
  TemplateTemplateParmDecl *TempParm = cast<TemplateTemplateParmDecl>(Param);
    
  // Substitute into the template parameter list of the template
  // template parameter, since previously-supplied template arguments
  // may appear within the template template parameter.
  {
    // Set up a template instantiation context.
    LocalInstantiationScope Scope(*this);
    InstantiatingTemplate Inst(*this, TemplateLoc, Template,
                               TempParm, Converted.getFlatArguments(),
                               Converted.flatSize(),
                               SourceRange(TemplateLoc, RAngleLoc));
    
    TemplateArgumentList TemplateArgs(Context, Converted,
                                      /*TakeArgs=*/false);
    TempParm = cast_or_null<TemplateTemplateParmDecl>(
                      SubstDecl(TempParm, CurContext, 
                                MultiLevelTemplateArgumentList(TemplateArgs)));
    if (!TempParm)
      return true;
    
    // FIXME: TempParam is leaked.
  }
    
  switch (Arg.getArgument().getKind()) {
  case TemplateArgument::Null:
    assert(false && "Should never see a NULL template argument here");
    return true;
    
  case TemplateArgument::Template:
    if (CheckTemplateArgument(TempParm, Arg))
      return true;
      
    Converted.Append(Arg.getArgument());
    break;
    
  case TemplateArgument::Expression:
  case TemplateArgument::Type:
    // We have a template template parameter but the template
    // argument does not refer to a template.
    Diag(Arg.getLocation(), diag::err_template_arg_must_be_template);
    return true;
      
  case TemplateArgument::Declaration:
    llvm_unreachable(
                       "Declaration argument with template template parameter");
    break;
  case TemplateArgument::Integral:
    llvm_unreachable(
                          "Integral argument with template template parameter");
    break;
    
  case TemplateArgument::Pack:
    llvm_unreachable("Caller must expand template argument packs");
    break;
  }
  
  return false;
}

/// \brief Check that the given template argument list is well-formed
/// for specializing the given template.
bool Sema::CheckTemplateArgumentList(TemplateDecl *Template,
                                     SourceLocation TemplateLoc,
                                const TemplateArgumentListInfo &TemplateArgs,
                                     bool PartialTemplateArgs,
                                     TemplateArgumentListBuilder &Converted) {
  TemplateParameterList *Params = Template->getTemplateParameters();
  unsigned NumParams = Params->size();
  unsigned NumArgs = TemplateArgs.size();
  bool Invalid = false;

  SourceLocation RAngleLoc = TemplateArgs.getRAngleLoc();

  bool HasParameterPack =
    NumParams > 0 && Params->getParam(NumParams - 1)->isTemplateParameterPack();

  if ((NumArgs > NumParams && !HasParameterPack) ||
      (NumArgs < Params->getMinRequiredArguments() &&
       !PartialTemplateArgs)) {
    // FIXME: point at either the first arg beyond what we can handle,
    // or the '>', depending on whether we have too many or too few
    // arguments.
    SourceRange Range;
    if (NumArgs > NumParams)
      Range = SourceRange(TemplateArgs[NumParams].getLocation(), RAngleLoc);
    Diag(TemplateLoc, diag::err_template_arg_list_different_arity)
      << (NumArgs > NumParams)
      << (isa<ClassTemplateDecl>(Template)? 0 :
          isa<FunctionTemplateDecl>(Template)? 1 :
          isa<TemplateTemplateParmDecl>(Template)? 2 : 3)
      << Template << Range;
    Diag(Template->getLocation(), diag::note_template_decl_here)
      << Params->getSourceRange();
    Invalid = true;
  }

  // C++ [temp.arg]p1:
  //   [...] The type and form of each template-argument specified in
  //   a template-id shall match the type and form specified for the
  //   corresponding parameter declared by the template in its
  //   template-parameter-list.
  unsigned ArgIdx = 0;
  for (TemplateParameterList::iterator Param = Params->begin(),
                                       ParamEnd = Params->end();
       Param != ParamEnd; ++Param, ++ArgIdx) {
    if (ArgIdx > NumArgs && PartialTemplateArgs)
      break;

    // If we have a template parameter pack, check every remaining template
    // argument against that template parameter pack.
    if ((*Param)->isTemplateParameterPack()) {
      Converted.BeginPack();
      for (; ArgIdx < NumArgs; ++ArgIdx) {
        if (CheckTemplateArgument(*Param, TemplateArgs[ArgIdx], Template,
                                  TemplateLoc, RAngleLoc, Converted)) {
          Invalid = true;
          break;
        }
      }
      Converted.EndPack();
      continue;
    }
    
    if (ArgIdx < NumArgs) {
      // Check the template argument we were given.
      if (CheckTemplateArgument(*Param, TemplateArgs[ArgIdx], Template, 
                                TemplateLoc, RAngleLoc, Converted))
        return true;
      
      continue;
    }
    
    // We have a default template argument that we will use.
    TemplateArgumentLoc Arg;
    
    // Retrieve the default template argument from the template
    // parameter. For each kind of template parameter, we substitute the
    // template arguments provided thus far and any "outer" template arguments
    // (when the template parameter was part of a nested template) into 
    // the default argument.
    if (TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(*Param)) {
      if (!TTP->hasDefaultArgument()) {
        assert((Invalid || PartialTemplateArgs) && "Missing default argument");
        break;
      }

      TypeSourceInfo *ArgType = SubstDefaultTemplateArgument(*this, 
                                                             Template,
                                                             TemplateLoc,
                                                             RAngleLoc,
                                                             TTP,
                                                             Converted);
      if (!ArgType)
        return true;
                                                             
      Arg = TemplateArgumentLoc(TemplateArgument(ArgType->getType()),
                                ArgType);
    } else if (NonTypeTemplateParmDecl *NTTP
                 = dyn_cast<NonTypeTemplateParmDecl>(*Param)) {
      if (!NTTP->hasDefaultArgument()) {
        assert((Invalid || PartialTemplateArgs) && "Missing default argument");
        break;
      }

      Sema::OwningExprResult E = SubstDefaultTemplateArgument(*this, Template,
                                                              TemplateLoc, 
                                                              RAngleLoc, 
                                                              NTTP, 
                                                              Converted);
      if (E.isInvalid())
        return true;

      Expr *Ex = E.takeAs<Expr>();
      Arg = TemplateArgumentLoc(TemplateArgument(Ex), Ex);
    } else {
      TemplateTemplateParmDecl *TempParm
        = cast<TemplateTemplateParmDecl>(*Param);

      if (!TempParm->hasDefaultArgument()) {
        assert((Invalid || PartialTemplateArgs) && "Missing default argument");
        break;
      }

      TemplateName Name = SubstDefaultTemplateArgument(*this, Template,
                                                       TemplateLoc, 
                                                       RAngleLoc, 
                                                       TempParm,
                                                       Converted);
      if (Name.isNull())
        return true;
      
      Arg = TemplateArgumentLoc(TemplateArgument(Name), 
                  TempParm->getDefaultArgument().getTemplateQualifierRange(),
                  TempParm->getDefaultArgument().getTemplateNameLoc());
    }
    
    // Introduce an instantiation record that describes where we are using
    // the default template argument.
    InstantiatingTemplate Instantiating(*this, RAngleLoc, Template, *Param,
                                        Converted.getFlatArguments(),
                                        Converted.flatSize(),
                                        SourceRange(TemplateLoc, RAngleLoc));    
    
    // Check the default template argument.
    if (CheckTemplateArgument(*Param, Arg, Template, TemplateLoc,
                              RAngleLoc, Converted))
      return true;
  }

  return Invalid;
}

/// \brief Check a template argument against its corresponding
/// template type parameter.
///
/// This routine implements the semantics of C++ [temp.arg.type]. It
/// returns true if an error occurred, and false otherwise.
bool Sema::CheckTemplateArgument(TemplateTypeParmDecl *Param,
                                 TypeSourceInfo *ArgInfo) {
  assert(ArgInfo && "invalid TypeSourceInfo");
  QualType Arg = ArgInfo->getType();

  // C++ [temp.arg.type]p2:
  //   A local type, a type with no linkage, an unnamed type or a type
  //   compounded from any of these types shall not be used as a
  //   template-argument for a template type-parameter.
  //
  // FIXME: Perform the recursive and no-linkage type checks.
  const TagType *Tag = 0;
  if (const EnumType *EnumT = Arg->getAs<EnumType>())
    Tag = EnumT;
  else if (const RecordType *RecordT = Arg->getAs<RecordType>())
    Tag = RecordT;
  if (Tag && Tag->getDecl()->getDeclContext()->isFunctionOrMethod()) {
    SourceRange SR = ArgInfo->getTypeLoc().getFullSourceRange();
    return Diag(SR.getBegin(), diag::err_template_arg_local_type)
      << QualType(Tag, 0) << SR;
  } else if (Tag && !Tag->getDecl()->getDeclName() &&
           !Tag->getDecl()->getTypedefForAnonDecl()) {
    SourceRange SR = ArgInfo->getTypeLoc().getFullSourceRange();
    Diag(SR.getBegin(), diag::err_template_arg_unnamed_type) << SR;
    Diag(Tag->getDecl()->getLocation(), diag::note_template_unnamed_type_here);
    return true;
  }

  return false;
}

/// \brief Checks whether the given template argument is the address
/// of an object or function according to C++ [temp.arg.nontype]p1.
bool Sema::CheckTemplateArgumentAddressOfObjectOrFunction(Expr *Arg,
                                                          NamedDecl *&Entity) {
  bool Invalid = false;

  // See through any implicit casts we added to fix the type.
  while (ImplicitCastExpr *Cast = dyn_cast<ImplicitCastExpr>(Arg))
    Arg = Cast->getSubExpr();

  // C++0x allows nullptr, and there's no further checking to be done for that.
  if (Arg->getType()->isNullPtrType())
    return false;

  // C++ [temp.arg.nontype]p1:
  //
  //   A template-argument for a non-type, non-template
  //   template-parameter shall be one of: [...]
  //
  //     -- the address of an object or function with external
  //        linkage, including function templates and function
  //        template-ids but excluding non-static class members,
  //        expressed as & id-expression where the & is optional if
  //        the name refers to a function or array, or if the
  //        corresponding template-parameter is a reference; or
  DeclRefExpr *DRE = 0;

  // Ignore (and complain about) any excess parentheses.
  while (ParenExpr *Parens = dyn_cast<ParenExpr>(Arg)) {
    if (!Invalid) {
      Diag(Arg->getSourceRange().getBegin(),
           diag::err_template_arg_extra_parens)
        << Arg->getSourceRange();
      Invalid = true;
    }

    Arg = Parens->getSubExpr();
  }

  if (UnaryOperator *UnOp = dyn_cast<UnaryOperator>(Arg)) {
    if (UnOp->getOpcode() == UnaryOperator::AddrOf)
      DRE = dyn_cast<DeclRefExpr>(UnOp->getSubExpr());
  } else
    DRE = dyn_cast<DeclRefExpr>(Arg);

  if (!DRE || !isa<ValueDecl>(DRE->getDecl()))
    return Diag(Arg->getSourceRange().getBegin(),
                diag::err_template_arg_not_object_or_func_form)
      << Arg->getSourceRange();

  // Cannot refer to non-static data members
  if (FieldDecl *Field = dyn_cast<FieldDecl>(DRE->getDecl()))
    return Diag(Arg->getSourceRange().getBegin(), diag::err_template_arg_field)
      << Field << Arg->getSourceRange();

  // Cannot refer to non-static member functions
  if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(DRE->getDecl()))
    if (!Method->isStatic())
      return Diag(Arg->getSourceRange().getBegin(),
                  diag::err_template_arg_method)
        << Method << Arg->getSourceRange();

  // Functions must have external linkage.
  if (FunctionDecl *Func = dyn_cast<FunctionDecl>(DRE->getDecl())) {
    if (Func->getLinkage() != NamedDecl::ExternalLinkage) {
      Diag(Arg->getSourceRange().getBegin(),
           diag::err_template_arg_function_not_extern)
        << Func << Arg->getSourceRange();
      Diag(Func->getLocation(), diag::note_template_arg_internal_object)
        << true;
      return true;
    }

    // Okay: we've named a function with external linkage.
    Entity = Func;
    return Invalid;
  }

  if (VarDecl *Var = dyn_cast<VarDecl>(DRE->getDecl())) {
    if (Var->getLinkage() != NamedDecl::ExternalLinkage) {
      Diag(Arg->getSourceRange().getBegin(),
           diag::err_template_arg_object_not_extern)
        << Var << Arg->getSourceRange();
      Diag(Var->getLocation(), diag::note_template_arg_internal_object)
        << true;
      return true;
    }

    // Okay: we've named an object with external linkage
    Entity = Var;
    return Invalid;
  }

  // We found something else, but we don't know specifically what it is.
  Diag(Arg->getSourceRange().getBegin(),
       diag::err_template_arg_not_object_or_func)
      << Arg->getSourceRange();
  Diag(DRE->getDecl()->getLocation(),
       diag::note_template_arg_refers_here);
  return true;
}

/// \brief Checks whether the given template argument is a pointer to
/// member constant according to C++ [temp.arg.nontype]p1.
bool Sema::CheckTemplateArgumentPointerToMember(Expr *Arg, 
                                                TemplateArgument &Converted) {
  bool Invalid = false;

  // See through any implicit casts we added to fix the type.
  while (ImplicitCastExpr *Cast = dyn_cast<ImplicitCastExpr>(Arg))
    Arg = Cast->getSubExpr();

  // C++0x allows nullptr, and there's no further checking to be done for that.
  if (Arg->getType()->isNullPtrType())
    return false;

  // C++ [temp.arg.nontype]p1:
  //
  //   A template-argument for a non-type, non-template
  //   template-parameter shall be one of: [...]
  //
  //     -- a pointer to member expressed as described in 5.3.1.
  DeclRefExpr *DRE = 0;

  // Ignore (and complain about) any excess parentheses.
  while (ParenExpr *Parens = dyn_cast<ParenExpr>(Arg)) {
    if (!Invalid) {
      Diag(Arg->getSourceRange().getBegin(),
           diag::err_template_arg_extra_parens)
        << Arg->getSourceRange();
      Invalid = true;
    }

    Arg = Parens->getSubExpr();
  }

  // A pointer-to-member constant written &Class::member.
  if (UnaryOperator *UnOp = dyn_cast<UnaryOperator>(Arg)) {
    if (UnOp->getOpcode() == UnaryOperator::AddrOf) {
      DRE = dyn_cast<DeclRefExpr>(UnOp->getSubExpr());
      if (DRE && !DRE->getQualifier())
        DRE = 0;
    }
  } 
  // A constant of pointer-to-member type.
  else if ((DRE = dyn_cast<DeclRefExpr>(Arg))) {
    if (ValueDecl *VD = dyn_cast<ValueDecl>(DRE->getDecl())) {
      if (VD->getType()->isMemberPointerType()) {
        if (isa<NonTypeTemplateParmDecl>(VD) ||
            (isa<VarDecl>(VD) && 
             Context.getCanonicalType(VD->getType()).isConstQualified())) {
          if (Arg->isTypeDependent() || Arg->isValueDependent())
            Converted = TemplateArgument(Arg->Retain());
          else
            Converted = TemplateArgument(VD->getCanonicalDecl());
          return Invalid;
        }
      }
    }
    
    DRE = 0;
  }
  
  if (!DRE)
    return Diag(Arg->getSourceRange().getBegin(),
                diag::err_template_arg_not_pointer_to_member_form)
      << Arg->getSourceRange();

  if (isa<FieldDecl>(DRE->getDecl()) || isa<CXXMethodDecl>(DRE->getDecl())) {
    assert((isa<FieldDecl>(DRE->getDecl()) ||
            !cast<CXXMethodDecl>(DRE->getDecl())->isStatic()) &&
           "Only non-static member pointers can make it here");

    // Okay: this is the address of a non-static member, and therefore
    // a member pointer constant.
    if (Arg->isTypeDependent() || Arg->isValueDependent())
      Converted = TemplateArgument(Arg->Retain());
    else
      Converted = TemplateArgument(DRE->getDecl()->getCanonicalDecl());
    return Invalid;
  }

  // We found something else, but we don't know specifically what it is.
  Diag(Arg->getSourceRange().getBegin(),
       diag::err_template_arg_not_pointer_to_member_form)
      << Arg->getSourceRange();
  Diag(DRE->getDecl()->getLocation(),
       diag::note_template_arg_refers_here);
  return true;
}

/// \brief Check a template argument against its corresponding
/// non-type template parameter.
///
/// This routine implements the semantics of C++ [temp.arg.nontype].
/// It returns true if an error occurred, and false otherwise. \p
/// InstantiatedParamType is the type of the non-type template
/// parameter after it has been instantiated.
///
/// If no error was detected, Converted receives the converted template argument.
bool Sema::CheckTemplateArgument(NonTypeTemplateParmDecl *Param,
                                 QualType InstantiatedParamType, Expr *&Arg,
                                 TemplateArgument &Converted) {
  SourceLocation StartLoc = Arg->getSourceRange().getBegin();

  // If either the parameter has a dependent type or the argument is
  // type-dependent, there's nothing we can check now.
  // FIXME: Add template argument to Converted!
  if (InstantiatedParamType->isDependentType() || Arg->isTypeDependent()) {
    // FIXME: Produce a cloned, canonical expression?
    Converted = TemplateArgument(Arg);
    return false;
  }

  // C++ [temp.arg.nontype]p5:
  //   The following conversions are performed on each expression used
  //   as a non-type template-argument. If a non-type
  //   template-argument cannot be converted to the type of the
  //   corresponding template-parameter then the program is
  //   ill-formed.
  //
  //     -- for a non-type template-parameter of integral or
  //        enumeration type, integral promotions (4.5) and integral
  //        conversions (4.7) are applied.
  QualType ParamType = InstantiatedParamType;
  QualType ArgType = Arg->getType();
  if (ParamType->isIntegralType() || ParamType->isEnumeralType()) {
    // C++ [temp.arg.nontype]p1:
    //   A template-argument for a non-type, non-template
    //   template-parameter shall be one of:
    //
    //     -- an integral constant-expression of integral or enumeration
    //        type; or
    //     -- the name of a non-type template-parameter; or
    SourceLocation NonConstantLoc;
    llvm::APSInt Value;
    if (!ArgType->isIntegralType() && !ArgType->isEnumeralType()) {
      Diag(Arg->getSourceRange().getBegin(),
           diag::err_template_arg_not_integral_or_enumeral)
        << ArgType << Arg->getSourceRange();
      Diag(Param->getLocation(), diag::note_template_param_here);
      return true;
    } else if (!Arg->isValueDependent() &&
               !Arg->isIntegerConstantExpr(Value, Context, &NonConstantLoc)) {
      Diag(NonConstantLoc, diag::err_template_arg_not_ice)
        << ArgType << Arg->getSourceRange();
      return true;
    }

    // FIXME: We need some way to more easily get the unqualified form
    // of the types without going all the way to the
    // canonical type.
    if (Context.getCanonicalType(ParamType).getCVRQualifiers())
      ParamType = Context.getCanonicalType(ParamType).getUnqualifiedType();
    if (Context.getCanonicalType(ArgType).getCVRQualifiers())
      ArgType = Context.getCanonicalType(ArgType).getUnqualifiedType();

    // Try to convert the argument to the parameter's type.
    if (Context.hasSameType(ParamType, ArgType)) {
      // Okay: no conversion necessary
    } else if (IsIntegralPromotion(Arg, ArgType, ParamType) ||
               !ParamType->isEnumeralType()) {
      // This is an integral promotion or conversion.
      ImpCastExprToType(Arg, ParamType, CastExpr::CK_IntegralCast);
    } else {
      // We can't perform this conversion.
      Diag(Arg->getSourceRange().getBegin(),
           diag::err_template_arg_not_convertible)
        << Arg->getType() << InstantiatedParamType << Arg->getSourceRange();
      Diag(Param->getLocation(), diag::note_template_param_here);
      return true;
    }

    QualType IntegerType = Context.getCanonicalType(ParamType);
    if (const EnumType *Enum = IntegerType->getAs<EnumType>())
      IntegerType = Context.getCanonicalType(Enum->getDecl()->getIntegerType());

    if (!Arg->isValueDependent()) {
      // Check that an unsigned parameter does not receive a negative
      // value.
      if (IntegerType->isUnsignedIntegerType()
          && (Value.isSigned() && Value.isNegative())) {
        Diag(Arg->getSourceRange().getBegin(), diag::err_template_arg_negative)
          << Value.toString(10) << Param->getType()
          << Arg->getSourceRange();
        Diag(Param->getLocation(), diag::note_template_param_here);
        return true;
      }

      // Check that we don't overflow the template parameter type.
      unsigned AllowedBits = Context.getTypeSize(IntegerType);
      if (Value.getActiveBits() > AllowedBits) {
        Diag(Arg->getSourceRange().getBegin(),
             diag::err_template_arg_too_large)
          << Value.toString(10) << Param->getType()
          << Arg->getSourceRange();
        Diag(Param->getLocation(), diag::note_template_param_here);
        return true;
      }

      if (Value.getBitWidth() != AllowedBits)
        Value.extOrTrunc(AllowedBits);
      Value.setIsSigned(IntegerType->isSignedIntegerType());
    }

    // Add the value of this argument to the list of converted
    // arguments. We use the bitwidth and signedness of the template
    // parameter.
    if (Arg->isValueDependent()) {
      // The argument is value-dependent. Create a new
      // TemplateArgument with the converted expression.
      Converted = TemplateArgument(Arg);
      return false;
    }

    Converted = TemplateArgument(Value,
                                 ParamType->isEnumeralType() ? ParamType
                                                             : IntegerType);
    return false;
  }

  // Handle pointer-to-function, reference-to-function, and
  // pointer-to-member-function all in (roughly) the same way.
  if (// -- For a non-type template-parameter of type pointer to
      //    function, only the function-to-pointer conversion (4.3) is
      //    applied. If the template-argument represents a set of
      //    overloaded functions (or a pointer to such), the matching
      //    function is selected from the set (13.4).
      // In C++0x, any std::nullptr_t value can be converted.
      (ParamType->isPointerType() &&
       ParamType->getAs<PointerType>()->getPointeeType()->isFunctionType()) ||
      // -- For a non-type template-parameter of type reference to
      //    function, no conversions apply. If the template-argument
      //    represents a set of overloaded functions, the matching
      //    function is selected from the set (13.4).
      (ParamType->isReferenceType() &&
       ParamType->getAs<ReferenceType>()->getPointeeType()->isFunctionType()) ||
      // -- For a non-type template-parameter of type pointer to
      //    member function, no conversions apply. If the
      //    template-argument represents a set of overloaded member
      //    functions, the matching member function is selected from
      //    the set (13.4).
      // Again, C++0x allows a std::nullptr_t value.
      (ParamType->isMemberPointerType() &&
       ParamType->getAs<MemberPointerType>()->getPointeeType()
         ->isFunctionType())) {
    if (Context.hasSameUnqualifiedType(ArgType,
                                       ParamType.getNonReferenceType())) {
      // We don't have to do anything: the types already match.
    } else if (ArgType->isNullPtrType() && (ParamType->isPointerType() ||
                 ParamType->isMemberPointerType())) {
      ArgType = ParamType;
      if (ParamType->isMemberPointerType())
        ImpCastExprToType(Arg, ParamType, CastExpr::CK_NullToMemberPointer);
      else
        ImpCastExprToType(Arg, ParamType, CastExpr::CK_BitCast);
    } else if (ArgType->isFunctionType() && ParamType->isPointerType()) {
      ArgType = Context.getPointerType(ArgType);
      ImpCastExprToType(Arg, ArgType, CastExpr::CK_FunctionToPointerDecay);
    } else if (FunctionDecl *Fn
                 = ResolveAddressOfOverloadedFunction(Arg, ParamType, true)) {
      if (DiagnoseUseOfDecl(Fn, Arg->getSourceRange().getBegin()))
        return true;

      Arg = FixOverloadedFunctionReference(Arg, Fn);
      ArgType = Arg->getType();
      if (ArgType->isFunctionType() && ParamType->isPointerType()) {
        ArgType = Context.getPointerType(Arg->getType());
        ImpCastExprToType(Arg, ArgType, CastExpr::CK_FunctionToPointerDecay);
      }
    }

    if (!Context.hasSameUnqualifiedType(ArgType,
                                        ParamType.getNonReferenceType())) {
      // We can't perform this conversion.
      Diag(Arg->getSourceRange().getBegin(),
           diag::err_template_arg_not_convertible)
        << Arg->getType() << InstantiatedParamType << Arg->getSourceRange();
      Diag(Param->getLocation(), diag::note_template_param_here);
      return true;
    }

    if (ParamType->isMemberPointerType())
      return CheckTemplateArgumentPointerToMember(Arg, Converted);

    NamedDecl *Entity = 0;
    if (CheckTemplateArgumentAddressOfObjectOrFunction(Arg, Entity))
      return true;

    if (Entity)
      Entity = cast<NamedDecl>(Entity->getCanonicalDecl());
    Converted = TemplateArgument(Entity);
    return false;
  }

  if (ParamType->isPointerType()) {
    //   -- for a non-type template-parameter of type pointer to
    //      object, qualification conversions (4.4) and the
    //      array-to-pointer conversion (4.2) are applied.
    // C++0x also allows a value of std::nullptr_t.
    assert(ParamType->getAs<PointerType>()->getPointeeType()->isObjectType() &&
           "Only object pointers allowed here");

    if (ArgType->isNullPtrType()) {
      ArgType = ParamType;
      ImpCastExprToType(Arg, ParamType, CastExpr::CK_BitCast);
    } else if (ArgType->isArrayType()) {
      ArgType = Context.getArrayDecayedType(ArgType);
      ImpCastExprToType(Arg, ArgType, CastExpr::CK_ArrayToPointerDecay);
    }

    if (IsQualificationConversion(ArgType, ParamType)) {
      ArgType = ParamType;
      ImpCastExprToType(Arg, ParamType, CastExpr::CK_NoOp);
    }

    if (!Context.hasSameUnqualifiedType(ArgType, ParamType)) {
      // We can't perform this conversion.
      Diag(Arg->getSourceRange().getBegin(),
           diag::err_template_arg_not_convertible)
        << Arg->getType() << InstantiatedParamType << Arg->getSourceRange();
      Diag(Param->getLocation(), diag::note_template_param_here);
      return true;
    }

    NamedDecl *Entity = 0;
    if (CheckTemplateArgumentAddressOfObjectOrFunction(Arg, Entity))
      return true;

    if (Entity)
      Entity = cast<NamedDecl>(Entity->getCanonicalDecl());
    Converted = TemplateArgument(Entity);
    return false;
  }

  if (const ReferenceType *ParamRefType = ParamType->getAs<ReferenceType>()) {
    //   -- For a non-type template-parameter of type reference to
    //      object, no conversions apply. The type referred to by the
    //      reference may be more cv-qualified than the (otherwise
    //      identical) type of the template-argument. The
    //      template-parameter is bound directly to the
    //      template-argument, which must be an lvalue.
    assert(ParamRefType->getPointeeType()->isObjectType() &&
           "Only object references allowed here");

    if (!Context.hasSameUnqualifiedType(ParamRefType->getPointeeType(), ArgType)) {
      Diag(Arg->getSourceRange().getBegin(),
           diag::err_template_arg_no_ref_bind)
        << InstantiatedParamType << Arg->getType()
        << Arg->getSourceRange();
      Diag(Param->getLocation(), diag::note_template_param_here);
      return true;
    }

    unsigned ParamQuals
      = Context.getCanonicalType(ParamType).getCVRQualifiers();
    unsigned ArgQuals = Context.getCanonicalType(ArgType).getCVRQualifiers();

    if ((ParamQuals | ArgQuals) != ParamQuals) {
      Diag(Arg->getSourceRange().getBegin(),
           diag::err_template_arg_ref_bind_ignores_quals)
        << InstantiatedParamType << Arg->getType()
        << Arg->getSourceRange();
      Diag(Param->getLocation(), diag::note_template_param_here);
      return true;
    }

    NamedDecl *Entity = 0;
    if (CheckTemplateArgumentAddressOfObjectOrFunction(Arg, Entity))
      return true;

    Entity = cast<NamedDecl>(Entity->getCanonicalDecl());
    Converted = TemplateArgument(Entity);
    return false;
  }

  //     -- For a non-type template-parameter of type pointer to data
  //        member, qualification conversions (4.4) are applied.
  // C++0x allows std::nullptr_t values.
  assert(ParamType->isMemberPointerType() && "Only pointers to members remain");

  if (Context.hasSameUnqualifiedType(ParamType, ArgType)) {
    // Types match exactly: nothing more to do here.
  } else if (ArgType->isNullPtrType()) {
    ImpCastExprToType(Arg, ParamType, CastExpr::CK_NullToMemberPointer);
  } else if (IsQualificationConversion(ArgType, ParamType)) {
    ImpCastExprToType(Arg, ParamType, CastExpr::CK_NoOp);
  } else {
    // We can't perform this conversion.
    Diag(Arg->getSourceRange().getBegin(),
         diag::err_template_arg_not_convertible)
      << Arg->getType() << InstantiatedParamType << Arg->getSourceRange();
    Diag(Param->getLocation(), diag::note_template_param_here);
    return true;
  }

  return CheckTemplateArgumentPointerToMember(Arg, Converted);
}

/// \brief Check a template argument against its corresponding
/// template template parameter.
///
/// This routine implements the semantics of C++ [temp.arg.template].
/// It returns true if an error occurred, and false otherwise.
bool Sema::CheckTemplateArgument(TemplateTemplateParmDecl *Param,
                                 const TemplateArgumentLoc &Arg) {
  TemplateName Name = Arg.getArgument().getAsTemplate();
  TemplateDecl *Template = Name.getAsTemplateDecl();
  if (!Template) {
    // Any dependent template name is fine.
    assert(Name.isDependent() && "Non-dependent template isn't a declaration?");
    return false;
  }

  // C++ [temp.arg.template]p1:
  //   A template-argument for a template template-parameter shall be
  //   the name of a class template, expressed as id-expression. Only
  //   primary class templates are considered when matching the
  //   template template argument with the corresponding parameter;
  //   partial specializations are not considered even if their
  //   parameter lists match that of the template template parameter.
  //
  // Note that we also allow template template parameters here, which
  // will happen when we are dealing with, e.g., class template
  // partial specializations.
  if (!isa<ClassTemplateDecl>(Template) &&
      !isa<TemplateTemplateParmDecl>(Template)) {
    assert(isa<FunctionTemplateDecl>(Template) &&
           "Only function templates are possible here");
    Diag(Arg.getLocation(), diag::err_template_arg_not_class_template);
    Diag(Template->getLocation(), diag::note_template_arg_refers_here_func)
      << Template;
  }

  return !TemplateParameterListsAreEqual(Template->getTemplateParameters(),
                                         Param->getTemplateParameters(),
                                         true, 
                                         TPL_TemplateTemplateArgumentMatch,
                                         Arg.getLocation());
}

/// \brief Determine whether the given template parameter lists are
/// equivalent.
///
/// \param New  The new template parameter list, typically written in the
/// source code as part of a new template declaration.
///
/// \param Old  The old template parameter list, typically found via
/// name lookup of the template declared with this template parameter
/// list.
///
/// \param Complain  If true, this routine will produce a diagnostic if
/// the template parameter lists are not equivalent.
///
/// \param Kind describes how we are to match the template parameter lists.
///
/// \param TemplateArgLoc If this source location is valid, then we
/// are actually checking the template parameter list of a template
/// argument (New) against the template parameter list of its
/// corresponding template template parameter (Old). We produce
/// slightly different diagnostics in this scenario.
///
/// \returns True if the template parameter lists are equal, false
/// otherwise.
bool
Sema::TemplateParameterListsAreEqual(TemplateParameterList *New,
                                     TemplateParameterList *Old,
                                     bool Complain,
                                     TemplateParameterListEqualKind Kind,
                                     SourceLocation TemplateArgLoc) {
  if (Old->size() != New->size()) {
    if (Complain) {
      unsigned NextDiag = diag::err_template_param_list_different_arity;
      if (TemplateArgLoc.isValid()) {
        Diag(TemplateArgLoc, diag::err_template_arg_template_params_mismatch);
        NextDiag = diag::note_template_param_list_different_arity;
      }
      Diag(New->getTemplateLoc(), NextDiag)
          << (New->size() > Old->size())
          << (Kind != TPL_TemplateMatch)
          << SourceRange(New->getTemplateLoc(), New->getRAngleLoc());
      Diag(Old->getTemplateLoc(), diag::note_template_prev_declaration)
        << (Kind != TPL_TemplateMatch)
        << SourceRange(Old->getTemplateLoc(), Old->getRAngleLoc());
    }

    return false;
  }

  for (TemplateParameterList::iterator OldParm = Old->begin(),
         OldParmEnd = Old->end(), NewParm = New->begin();
       OldParm != OldParmEnd; ++OldParm, ++NewParm) {
    if ((*OldParm)->getKind() != (*NewParm)->getKind()) {
      if (Complain) {
        unsigned NextDiag = diag::err_template_param_different_kind;
        if (TemplateArgLoc.isValid()) {
          Diag(TemplateArgLoc, diag::err_template_arg_template_params_mismatch);
          NextDiag = diag::note_template_param_different_kind;
        }
        Diag((*NewParm)->getLocation(), NextDiag)
          << (Kind != TPL_TemplateMatch);
        Diag((*OldParm)->getLocation(), diag::note_template_prev_declaration)
          << (Kind != TPL_TemplateMatch);
      }
      return false;
    }

    if (isa<TemplateTypeParmDecl>(*OldParm)) {
      // Okay; all template type parameters are equivalent (since we
      // know we're at the same index).
    } else if (NonTypeTemplateParmDecl *OldNTTP
                 = dyn_cast<NonTypeTemplateParmDecl>(*OldParm)) {
      // The types of non-type template parameters must agree.
      NonTypeTemplateParmDecl *NewNTTP
        = cast<NonTypeTemplateParmDecl>(*NewParm);
      
      // If we are matching a template template argument to a template
      // template parameter and one of the non-type template parameter types
      // is dependent, then we must wait until template instantiation time
      // to actually compare the arguments.
      if (Kind == TPL_TemplateTemplateArgumentMatch &&
          (OldNTTP->getType()->isDependentType() ||
           NewNTTP->getType()->isDependentType()))
        continue;
      
      if (Context.getCanonicalType(OldNTTP->getType()) !=
            Context.getCanonicalType(NewNTTP->getType())) {
        if (Complain) {
          unsigned NextDiag = diag::err_template_nontype_parm_different_type;
          if (TemplateArgLoc.isValid()) {
            Diag(TemplateArgLoc,
                 diag::err_template_arg_template_params_mismatch);
            NextDiag = diag::note_template_nontype_parm_different_type;
          }
          Diag(NewNTTP->getLocation(), NextDiag)
            << NewNTTP->getType()
            << (Kind != TPL_TemplateMatch);
          Diag(OldNTTP->getLocation(),
               diag::note_template_nontype_parm_prev_declaration)
            << OldNTTP->getType();
        }
        return false;
      }
    } else {
      // The template parameter lists of template template
      // parameters must agree.
      assert(isa<TemplateTemplateParmDecl>(*OldParm) &&
             "Only template template parameters handled here");
      TemplateTemplateParmDecl *OldTTP
        = cast<TemplateTemplateParmDecl>(*OldParm);
      TemplateTemplateParmDecl *NewTTP
        = cast<TemplateTemplateParmDecl>(*NewParm);
      if (!TemplateParameterListsAreEqual(NewTTP->getTemplateParameters(),
                                          OldTTP->getTemplateParameters(),
                                          Complain,
              (Kind == TPL_TemplateMatch? TPL_TemplateTemplateParmMatch : Kind),
                                          TemplateArgLoc))
        return false;
    }
  }

  return true;
}

/// \brief Check whether a template can be declared within this scope.
///
/// If the template declaration is valid in this scope, returns
/// false. Otherwise, issues a diagnostic and returns true.
bool
Sema::CheckTemplateDeclScope(Scope *S, TemplateParameterList *TemplateParams) {
  // Find the nearest enclosing declaration scope.
  while ((S->getFlags() & Scope::DeclScope) == 0 ||
         (S->getFlags() & Scope::TemplateParamScope) != 0)
    S = S->getParent();

  // C++ [temp]p2:
  //   A template-declaration can appear only as a namespace scope or
  //   class scope declaration.
  DeclContext *Ctx = static_cast<DeclContext *>(S->getEntity());
  if (Ctx && isa<LinkageSpecDecl>(Ctx) &&
      cast<LinkageSpecDecl>(Ctx)->getLanguage() != LinkageSpecDecl::lang_cxx)
    return Diag(TemplateParams->getTemplateLoc(), diag::err_template_linkage)
             << TemplateParams->getSourceRange();

  while (Ctx && isa<LinkageSpecDecl>(Ctx))
    Ctx = Ctx->getParent();

  if (Ctx && (Ctx->isFileContext() || Ctx->isRecord()))
    return false;

  return Diag(TemplateParams->getTemplateLoc(),
              diag::err_template_outside_namespace_or_class_scope)
    << TemplateParams->getSourceRange();
}

/// \brief Determine what kind of template specialization the given declaration
/// is.
static TemplateSpecializationKind getTemplateSpecializationKind(NamedDecl *D) {
  if (!D)
    return TSK_Undeclared;
  
  if (CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(D))
    return Record->getTemplateSpecializationKind();
  if (FunctionDecl *Function = dyn_cast<FunctionDecl>(D))
    return Function->getTemplateSpecializationKind();
  if (VarDecl *Var = dyn_cast<VarDecl>(D))
    return Var->getTemplateSpecializationKind();
  
  return TSK_Undeclared;
}

/// \brief Check whether a specialization is well-formed in the current 
/// context.
///
/// This routine determines whether a template specialization can be declared
/// in the current context (C++ [temp.expl.spec]p2).
///
/// \param S the semantic analysis object for which this check is being
/// performed.
///
/// \param Specialized the entity being specialized or instantiated, which
/// may be a kind of template (class template, function template, etc.) or
/// a member of a class template (member function, static data member, 
/// member class).
///
/// \param PrevDecl the previous declaration of this entity, if any.
///
/// \param Loc the location of the explicit specialization or instantiation of
/// this entity.
///
/// \param IsPartialSpecialization whether this is a partial specialization of
/// a class template.
///
/// \returns true if there was an error that we cannot recover from, false
/// otherwise.
static bool CheckTemplateSpecializationScope(Sema &S,
                                             NamedDecl *Specialized,
                                             NamedDecl *PrevDecl,
                                             SourceLocation Loc,
                                             bool IsPartialSpecialization) {
  // Keep these "kind" numbers in sync with the %select statements in the
  // various diagnostics emitted by this routine.
  int EntityKind = 0;
  bool isTemplateSpecialization = false;
  if (isa<ClassTemplateDecl>(Specialized)) {
    EntityKind = IsPartialSpecialization? 1 : 0;
    isTemplateSpecialization = true;
  } else if (isa<FunctionTemplateDecl>(Specialized)) {
    EntityKind = 2;
    isTemplateSpecialization = true;
  } else if (isa<CXXMethodDecl>(Specialized))
    EntityKind = 3;
  else if (isa<VarDecl>(Specialized))
    EntityKind = 4;
  else if (isa<RecordDecl>(Specialized))
    EntityKind = 5;
  else {
    S.Diag(Loc, diag::err_template_spec_unknown_kind);
    S.Diag(Specialized->getLocation(), diag::note_specialized_entity);
    return true;
  }

  // C++ [temp.expl.spec]p2:
  //   An explicit specialization shall be declared in the namespace
  //   of which the template is a member, or, for member templates, in
  //   the namespace of which the enclosing class or enclosing class
  //   template is a member. An explicit specialization of a member
  //   function, member class or static data member of a class
  //   template shall be declared in the namespace of which the class
  //   template is a member. Such a declaration may also be a
  //   definition. If the declaration is not a definition, the
  //   specialization may be defined later in the name- space in which
  //   the explicit specialization was declared, or in a namespace
  //   that encloses the one in which the explicit specialization was
  //   declared.
  if (S.CurContext->getLookupContext()->isFunctionOrMethod()) {
    S.Diag(Loc, diag::err_template_spec_decl_function_scope)
      << Specialized;
    return true;
  }

  if (S.CurContext->isRecord() && !IsPartialSpecialization) {
    S.Diag(Loc, diag::err_template_spec_decl_class_scope)
      << Specialized;
    return true;
  }
  
  // C++ [temp.class.spec]p6:
  //   A class template partial specialization may be declared or redeclared
  //   in any namespace scope in which its definition may be defined (14.5.1 
  //   and 14.5.2).  
  bool ComplainedAboutScope = false;
  DeclContext *SpecializedContext 
    = Specialized->getDeclContext()->getEnclosingNamespaceContext();
  DeclContext *DC = S.CurContext->getEnclosingNamespaceContext();
  if ((!PrevDecl || 
       getTemplateSpecializationKind(PrevDecl) == TSK_Undeclared ||
       getTemplateSpecializationKind(PrevDecl) == TSK_ImplicitInstantiation)){
    // There is no prior declaration of this entity, so this
    // specialization must be in the same context as the template
    // itself.
    if (!DC->Equals(SpecializedContext)) {
      if (isa<TranslationUnitDecl>(SpecializedContext))
        S.Diag(Loc, diag::err_template_spec_decl_out_of_scope_global)
        << EntityKind << Specialized;
      else if (isa<NamespaceDecl>(SpecializedContext))
        S.Diag(Loc, diag::err_template_spec_decl_out_of_scope)
        << EntityKind << Specialized
        << cast<NamedDecl>(SpecializedContext);
      
      S.Diag(Specialized->getLocation(), diag::note_specialized_entity);
      ComplainedAboutScope = true;
    }
  }
  
  // Make sure that this redeclaration (or definition) occurs in an enclosing 
  // namespace.
  // Note that HandleDeclarator() performs this check for explicit 
  // specializations of function templates, static data members, and member
  // functions, so we skip the check here for those kinds of entities.
  // FIXME: HandleDeclarator's diagnostics aren't quite as good, though.
  // Should we refactor that check, so that it occurs later?
  if (!ComplainedAboutScope && !DC->Encloses(SpecializedContext) &&
      !(isa<FunctionTemplateDecl>(Specialized) || isa<VarDecl>(Specialized) ||
        isa<FunctionDecl>(Specialized))) {
    if (isa<TranslationUnitDecl>(SpecializedContext))
      S.Diag(Loc, diag::err_template_spec_redecl_global_scope)
        << EntityKind << Specialized;
    else if (isa<NamespaceDecl>(SpecializedContext))
      S.Diag(Loc, diag::err_template_spec_redecl_out_of_scope)
        << EntityKind << Specialized
        << cast<NamedDecl>(SpecializedContext);
  
    S.Diag(Specialized->getLocation(), diag::note_specialized_entity);
  }
  
  // FIXME: check for specialization-after-instantiation errors and such.
  
  return false;
}
                                             
/// \brief Check the non-type template arguments of a class template
/// partial specialization according to C++ [temp.class.spec]p9.
///
/// \param TemplateParams the template parameters of the primary class
/// template.
///
/// \param TemplateArg the template arguments of the class template
/// partial specialization.
///
/// \param MirrorsPrimaryTemplate will be set true if the class
/// template partial specialization arguments are identical to the
/// implicit template arguments of the primary template. This is not
/// necessarily an error (C++0x), and it is left to the caller to diagnose
/// this condition when it is an error.
///
/// \returns true if there was an error, false otherwise.
bool Sema::CheckClassTemplatePartialSpecializationArgs(
                                        TemplateParameterList *TemplateParams,
                             const TemplateArgumentListBuilder &TemplateArgs,
                                        bool &MirrorsPrimaryTemplate) {
  // FIXME: the interface to this function will have to change to
  // accommodate variadic templates.
  MirrorsPrimaryTemplate = true;

  const TemplateArgument *ArgList = TemplateArgs.getFlatArguments();

  for (unsigned I = 0, N = TemplateParams->size(); I != N; ++I) {
    // Determine whether the template argument list of the partial
    // specialization is identical to the implicit argument list of
    // the primary template. The caller may need to diagnostic this as
    // an error per C++ [temp.class.spec]p9b3.
    if (MirrorsPrimaryTemplate) {
      if (TemplateTypeParmDecl *TTP
            = dyn_cast<TemplateTypeParmDecl>(TemplateParams->getParam(I))) {
        if (Context.getCanonicalType(Context.getTypeDeclType(TTP)) !=
              Context.getCanonicalType(ArgList[I].getAsType()))
          MirrorsPrimaryTemplate = false;
      } else if (TemplateTemplateParmDecl *TTP
                   = dyn_cast<TemplateTemplateParmDecl>(
                                                 TemplateParams->getParam(I))) {
        TemplateName Name = ArgList[I].getAsTemplate();
        TemplateTemplateParmDecl *ArgDecl
          = dyn_cast_or_null<TemplateTemplateParmDecl>(Name.getAsTemplateDecl());
        if (!ArgDecl ||
            ArgDecl->getIndex() != TTP->getIndex() ||
            ArgDecl->getDepth() != TTP->getDepth())
          MirrorsPrimaryTemplate = false;
      }
    }

    NonTypeTemplateParmDecl *Param
      = dyn_cast<NonTypeTemplateParmDecl>(TemplateParams->getParam(I));
    if (!Param) {
      continue;
    }

    Expr *ArgExpr = ArgList[I].getAsExpr();
    if (!ArgExpr) {
      MirrorsPrimaryTemplate = false;
      continue;
    }

    // C++ [temp.class.spec]p8:
    //   A non-type argument is non-specialized if it is the name of a
    //   non-type parameter. All other non-type arguments are
    //   specialized.
    //
    // Below, we check the two conditions that only apply to
    // specialized non-type arguments, so skip any non-specialized
    // arguments.
    if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(ArgExpr))
      if (NonTypeTemplateParmDecl *NTTP
            = dyn_cast<NonTypeTemplateParmDecl>(DRE->getDecl())) {
        if (MirrorsPrimaryTemplate &&
            (Param->getIndex() != NTTP->getIndex() ||
             Param->getDepth() != NTTP->getDepth()))
          MirrorsPrimaryTemplate = false;

        continue;
      }

    // C++ [temp.class.spec]p9:
    //   Within the argument list of a class template partial
    //   specialization, the following restrictions apply:
    //     -- A partially specialized non-type argument expression
    //        shall not involve a template parameter of the partial
    //        specialization except when the argument expression is a
    //        simple identifier.
    if (ArgExpr->isTypeDependent() || ArgExpr->isValueDependent()) {
      Diag(ArgExpr->getLocStart(),
           diag::err_dependent_non_type_arg_in_partial_spec)
        << ArgExpr->getSourceRange();
      return true;
    }

    //     -- The type of a template parameter corresponding to a
    //        specialized non-type argument shall not be dependent on a
    //        parameter of the specialization.
    if (Param->getType()->isDependentType()) {
      Diag(ArgExpr->getLocStart(),
           diag::err_dependent_typed_non_type_arg_in_partial_spec)
        << Param->getType()
        << ArgExpr->getSourceRange();
      Diag(Param->getLocation(), diag::note_template_param_here);
      return true;
    }

    MirrorsPrimaryTemplate = false;
  }

  return false;
}

Sema::DeclResult
Sema::ActOnClassTemplateSpecialization(Scope *S, unsigned TagSpec,
                                       TagUseKind TUK,
                                       SourceLocation KWLoc,
                                       const CXXScopeSpec &SS,
                                       TemplateTy TemplateD,
                                       SourceLocation TemplateNameLoc,
                                       SourceLocation LAngleLoc,
                                       ASTTemplateArgsPtr TemplateArgsIn,
                                       SourceLocation RAngleLoc,
                                       AttributeList *Attr,
                               MultiTemplateParamsArg TemplateParameterLists) {
  assert(TUK != TUK_Reference && "References are not specializations");

  // Find the class template we're specializing
  TemplateName Name = TemplateD.getAsVal<TemplateName>();
  ClassTemplateDecl *ClassTemplate
    = dyn_cast_or_null<ClassTemplateDecl>(Name.getAsTemplateDecl());

  if (!ClassTemplate) {
    Diag(TemplateNameLoc, diag::err_not_class_template_specialization)
      << (Name.getAsTemplateDecl() && 
          isa<TemplateTemplateParmDecl>(Name.getAsTemplateDecl()));
    return true;
  }

  bool isExplicitSpecialization = false;
  bool isPartialSpecialization = false;

  // Check the validity of the template headers that introduce this
  // template.
  // FIXME: We probably shouldn't complain about these headers for
  // friend declarations.
  TemplateParameterList *TemplateParams
    = MatchTemplateParametersToScopeSpecifier(TemplateNameLoc, SS,
                        (TemplateParameterList**)TemplateParameterLists.get(),
                                              TemplateParameterLists.size(),
                                              isExplicitSpecialization);
  if (TemplateParams && TemplateParams->size() > 0) {
    isPartialSpecialization = true;

    // C++ [temp.class.spec]p10:
    //   The template parameter list of a specialization shall not
    //   contain default template argument values.
    for (unsigned I = 0, N = TemplateParams->size(); I != N; ++I) {
      Decl *Param = TemplateParams->getParam(I);
      if (TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(Param)) {
        if (TTP->hasDefaultArgument()) {
          Diag(TTP->getDefaultArgumentLoc(),
               diag::err_default_arg_in_partial_spec);
          TTP->removeDefaultArgument();
        }
      } else if (NonTypeTemplateParmDecl *NTTP
                   = dyn_cast<NonTypeTemplateParmDecl>(Param)) {
        if (Expr *DefArg = NTTP->getDefaultArgument()) {
          Diag(NTTP->getDefaultArgumentLoc(),
               diag::err_default_arg_in_partial_spec)
            << DefArg->getSourceRange();
          NTTP->setDefaultArgument(0);
          DefArg->Destroy(Context);
        }
      } else {
        TemplateTemplateParmDecl *TTP = cast<TemplateTemplateParmDecl>(Param);
        if (TTP->hasDefaultArgument()) {
          Diag(TTP->getDefaultArgument().getLocation(),
               diag::err_default_arg_in_partial_spec)
            << TTP->getDefaultArgument().getSourceRange();
          TTP->setDefaultArgument(TemplateArgumentLoc());
        }
      }
    }
  } else if (TemplateParams) {
    if (TUK == TUK_Friend)
      Diag(KWLoc, diag::err_template_spec_friend)
        << CodeModificationHint::CreateRemoval(
                                SourceRange(TemplateParams->getTemplateLoc(),
                                            TemplateParams->getRAngleLoc()))
        << SourceRange(LAngleLoc, RAngleLoc);
    else
      isExplicitSpecialization = true;
  } else if (TUK != TUK_Friend) {
    Diag(KWLoc, diag::err_template_spec_needs_header)
      << CodeModificationHint::CreateInsertion(KWLoc, "template<> ");
    isExplicitSpecialization = true;
  }

  // Check that the specialization uses the same tag kind as the
  // original template.
  TagDecl::TagKind Kind;
  switch (TagSpec) {
  default: assert(0 && "Unknown tag type!");
  case DeclSpec::TST_struct: Kind = TagDecl::TK_struct; break;
  case DeclSpec::TST_union:  Kind = TagDecl::TK_union; break;
  case DeclSpec::TST_class:  Kind = TagDecl::TK_class; break;
  }
  if (!isAcceptableTagRedeclaration(ClassTemplate->getTemplatedDecl(),
                                    Kind, KWLoc,
                                    *ClassTemplate->getIdentifier())) {
    Diag(KWLoc, diag::err_use_with_wrong_tag)
      << ClassTemplate
      << CodeModificationHint::CreateReplacement(KWLoc,
                            ClassTemplate->getTemplatedDecl()->getKindName());
    Diag(ClassTemplate->getTemplatedDecl()->getLocation(),
         diag::note_previous_use);
    Kind = ClassTemplate->getTemplatedDecl()->getTagKind();
  }

  // Translate the parser's template argument list in our AST format.
  TemplateArgumentListInfo TemplateArgs;
  TemplateArgs.setLAngleLoc(LAngleLoc);
  TemplateArgs.setRAngleLoc(RAngleLoc);
  translateTemplateArguments(TemplateArgsIn, TemplateArgs);

  // Check that the template argument list is well-formed for this
  // template.
  TemplateArgumentListBuilder Converted(ClassTemplate->getTemplateParameters(),
                                        TemplateArgs.size());
  if (CheckTemplateArgumentList(ClassTemplate, TemplateNameLoc,
                                TemplateArgs, false, Converted))
    return true;

  assert((Converted.structuredSize() ==
            ClassTemplate->getTemplateParameters()->size()) &&
         "Converted template argument list is too short!");

  // Find the class template (partial) specialization declaration that
  // corresponds to these arguments.
  llvm::FoldingSetNodeID ID;
  if (isPartialSpecialization) {
    bool MirrorsPrimaryTemplate;
    if (CheckClassTemplatePartialSpecializationArgs(
                                         ClassTemplate->getTemplateParameters(),
                                         Converted, MirrorsPrimaryTemplate))
      return true;

    if (MirrorsPrimaryTemplate) {
      // C++ [temp.class.spec]p9b3:
      //
      //   -- The argument list of the specialization shall not be identical
      //      to the implicit argument list of the primary template.
      Diag(TemplateNameLoc, diag::err_partial_spec_args_match_primary_template)
        << (TUK == TUK_Definition)
        << CodeModificationHint::CreateRemoval(SourceRange(LAngleLoc,
                                                           RAngleLoc));
      return CheckClassTemplate(S, TagSpec, TUK, KWLoc, SS,
                                ClassTemplate->getIdentifier(),
                                TemplateNameLoc,
                                Attr,
                                TemplateParams,
                                AS_none);
    }

    // FIXME: Diagnose friend partial specializations

    // FIXME: Template parameter list matters, too
    ClassTemplatePartialSpecializationDecl::Profile(ID,
                                                   Converted.getFlatArguments(),
                                                   Converted.flatSize(),
                                                    Context);
  } else
    ClassTemplateSpecializationDecl::Profile(ID,
                                             Converted.getFlatArguments(),
                                             Converted.flatSize(),
                                             Context);
  void *InsertPos = 0;
  ClassTemplateSpecializationDecl *PrevDecl = 0;

  if (isPartialSpecialization)
    PrevDecl
      = ClassTemplate->getPartialSpecializations().FindNodeOrInsertPos(ID,
                                                                    InsertPos);
  else
    PrevDecl
      = ClassTemplate->getSpecializations().FindNodeOrInsertPos(ID, InsertPos);

  ClassTemplateSpecializationDecl *Specialization = 0;

  // Check whether we can declare a class template specialization in
  // the current scope.
  if (TUK != TUK_Friend &&
      CheckTemplateSpecializationScope(*this, ClassTemplate, PrevDecl, 
                                       TemplateNameLoc, 
                                       isPartialSpecialization))
    return true;
  
  // The canonical type
  QualType CanonType;
  if (PrevDecl && 
      (PrevDecl->getSpecializationKind() == TSK_Undeclared ||
       TUK == TUK_Friend)) {
    // Since the only prior class template specialization with these
    // arguments was referenced but not declared, or we're only
    // referencing this specialization as a friend, reuse that
    // declaration node as our own, updating its source location to
    // reflect our new declaration.
    Specialization = PrevDecl;
    Specialization->setLocation(TemplateNameLoc);
    PrevDecl = 0;
    CanonType = Context.getTypeDeclType(Specialization);
  } else if (isPartialSpecialization) {
    // Build the canonical type that describes the converted template
    // arguments of the class template partial specialization.
    CanonType = Context.getTemplateSpecializationType(
                                                  TemplateName(ClassTemplate),
                                                  Converted.getFlatArguments(),
                                                  Converted.flatSize());

    // Create a new class template partial specialization declaration node.
    ClassTemplatePartialSpecializationDecl *PrevPartial
      = cast_or_null<ClassTemplatePartialSpecializationDecl>(PrevDecl);
    ClassTemplatePartialSpecializationDecl *Partial
      = ClassTemplatePartialSpecializationDecl::Create(Context,
                                             ClassTemplate->getDeclContext(),
                                                       TemplateNameLoc,
                                                       TemplateParams,
                                                       ClassTemplate,
                                                       Converted,
                                                       TemplateArgs,
                                                       PrevPartial);

    if (PrevPartial) {
      ClassTemplate->getPartialSpecializations().RemoveNode(PrevPartial);
      ClassTemplate->getPartialSpecializations().GetOrInsertNode(Partial);
    } else {
      ClassTemplate->getPartialSpecializations().InsertNode(Partial, InsertPos);
    }
    Specialization = Partial;

    // If we are providing an explicit specialization of a member class 
    // template specialization, make a note of that.
    if (PrevPartial && PrevPartial->getInstantiatedFromMember())
      PrevPartial->setMemberSpecialization();
    
    // Check that all of the template parameters of the class template
    // partial specialization are deducible from the template
    // arguments. If not, this class template partial specialization
    // will never be used.
    llvm::SmallVector<bool, 8> DeducibleParams;
    DeducibleParams.resize(TemplateParams->size());
    MarkUsedTemplateParameters(Partial->getTemplateArgs(), true, 
                               TemplateParams->getDepth(),
                               DeducibleParams);
    unsigned NumNonDeducible = 0;
    for (unsigned I = 0, N = DeducibleParams.size(); I != N; ++I)
      if (!DeducibleParams[I])
        ++NumNonDeducible;

    if (NumNonDeducible) {
      Diag(TemplateNameLoc, diag::warn_partial_specs_not_deducible)
        << (NumNonDeducible > 1)
        << SourceRange(TemplateNameLoc, RAngleLoc);
      for (unsigned I = 0, N = DeducibleParams.size(); I != N; ++I) {
        if (!DeducibleParams[I]) {
          NamedDecl *Param = cast<NamedDecl>(TemplateParams->getParam(I));
          if (Param->getDeclName())
            Diag(Param->getLocation(),
                 diag::note_partial_spec_unused_parameter)
              << Param->getDeclName();
          else
            Diag(Param->getLocation(),
                 diag::note_partial_spec_unused_parameter)
              << std::string("<anonymous>");
        }
      }
    }
  } else {
    // Create a new class template specialization declaration node for
    // this explicit specialization or friend declaration.
    Specialization
      = ClassTemplateSpecializationDecl::Create(Context,
                                             ClassTemplate->getDeclContext(),
                                                TemplateNameLoc,
                                                ClassTemplate,
                                                Converted,
                                                PrevDecl);

    if (PrevDecl) {
      ClassTemplate->getSpecializations().RemoveNode(PrevDecl);
      ClassTemplate->getSpecializations().GetOrInsertNode(Specialization);
    } else {
      ClassTemplate->getSpecializations().InsertNode(Specialization,
                                                     InsertPos);
    }

    CanonType = Context.getTypeDeclType(Specialization);
  }

  // C++ [temp.expl.spec]p6:
  //   If a template, a member template or the member of a class template is
  //   explicitly specialized then that specialization shall be declared 
  //   before the first use of that specialization that would cause an implicit
  //   instantiation to take place, in every translation unit in which such a 
  //   use occurs; no diagnostic is required.
  if (PrevDecl && PrevDecl->getPointOfInstantiation().isValid()) {
    SourceRange Range(TemplateNameLoc, RAngleLoc);
    Diag(TemplateNameLoc, diag::err_specialization_after_instantiation)
      << Context.getTypeDeclType(Specialization) << Range;

    Diag(PrevDecl->getPointOfInstantiation(), 
         diag::note_instantiation_required_here)
      << (PrevDecl->getTemplateSpecializationKind() 
                                                != TSK_ImplicitInstantiation);
    return true;
  }
  
  // If this is not a friend, note that this is an explicit specialization.
  if (TUK != TUK_Friend)
    Specialization->setSpecializationKind(TSK_ExplicitSpecialization);

  // Check that this isn't a redefinition of this specialization.
  if (TUK == TUK_Definition) {
    if (RecordDecl *Def = Specialization->getDefinition(Context)) {
      SourceRange Range(TemplateNameLoc, RAngleLoc);
      Diag(TemplateNameLoc, diag::err_redefinition)
        << Context.getTypeDeclType(Specialization) << Range;
      Diag(Def->getLocation(), diag::note_previous_definition);
      Specialization->setInvalidDecl();
      return true;
    }
  }

  // Build the fully-sugared type for this class template
  // specialization as the user wrote in the specialization
  // itself. This means that we'll pretty-print the type retrieved
  // from the specialization's declaration the way that the user
  // actually wrote the specialization, rather than formatting the
  // name based on the "canonical" representation used to store the
  // template arguments in the specialization.
  QualType WrittenTy
    = Context.getTemplateSpecializationType(Name, TemplateArgs, CanonType);
  if (TUK != TUK_Friend)
    Specialization->setTypeAsWritten(WrittenTy);
  TemplateArgsIn.release();

  // C++ [temp.expl.spec]p9:
  //   A template explicit specialization is in the scope of the
  //   namespace in which the template was defined.
  //
  // We actually implement this paragraph where we set the semantic
  // context (in the creation of the ClassTemplateSpecializationDecl),
  // but we also maintain the lexical context where the actual
  // definition occurs.
  Specialization->setLexicalDeclContext(CurContext);

  // We may be starting the definition of this specialization.
  if (TUK == TUK_Definition)
    Specialization->startDefinition();

  if (TUK == TUK_Friend) {
    FriendDecl *Friend = FriendDecl::Create(Context, CurContext,
                                            TemplateNameLoc,
                                            WrittenTy.getTypePtr(),
                                            /*FIXME:*/KWLoc);
    Friend->setAccess(AS_public);
    CurContext->addDecl(Friend);
  } else {
    // Add the specialization into its lexical context, so that it can
    // be seen when iterating through the list of declarations in that
    // context. However, specializations are not found by name lookup.
    CurContext->addDecl(Specialization);
  }
  return DeclPtrTy::make(Specialization);
}

Sema::DeclPtrTy
Sema::ActOnTemplateDeclarator(Scope *S,
                              MultiTemplateParamsArg TemplateParameterLists,
                              Declarator &D) {
  return HandleDeclarator(S, D, move(TemplateParameterLists), false);
}

Sema::DeclPtrTy
Sema::ActOnStartOfFunctionTemplateDef(Scope *FnBodyScope,
                               MultiTemplateParamsArg TemplateParameterLists,
                                      Declarator &D) {
  assert(getCurFunctionDecl() == 0 && "Function parsing confused");
  assert(D.getTypeObject(0).Kind == DeclaratorChunk::Function &&
         "Not a function declarator!");
  DeclaratorChunk::FunctionTypeInfo &FTI = D.getTypeObject(0).Fun;

  if (FTI.hasPrototype) {
    // FIXME: Diagnose arguments without names in C.
  }

  Scope *ParentScope = FnBodyScope->getParent();

  DeclPtrTy DP = HandleDeclarator(ParentScope, D,
                                  move(TemplateParameterLists),
                                  /*IsFunctionDefinition=*/true);
  if (FunctionTemplateDecl *FunctionTemplate
        = dyn_cast_or_null<FunctionTemplateDecl>(DP.getAs<Decl>()))
    return ActOnStartOfFunctionDef(FnBodyScope,
                      DeclPtrTy::make(FunctionTemplate->getTemplatedDecl()));
  if (FunctionDecl *Function = dyn_cast_or_null<FunctionDecl>(DP.getAs<Decl>()))
    return ActOnStartOfFunctionDef(FnBodyScope, DeclPtrTy::make(Function));
  return DeclPtrTy();
}

/// \brief Diagnose cases where we have an explicit template specialization 
/// before/after an explicit template instantiation, producing diagnostics
/// for those cases where they are required and determining whether the 
/// new specialization/instantiation will have any effect.
///
/// \param NewLoc the location of the new explicit specialization or 
/// instantiation.
///
/// \param NewTSK the kind of the new explicit specialization or instantiation.
///
/// \param PrevDecl the previous declaration of the entity.
///
/// \param PrevTSK the kind of the old explicit specialization or instantiatin.
///
/// \param PrevPointOfInstantiation if valid, indicates where the previus 
/// declaration was instantiated (either implicitly or explicitly).
///
/// \param SuppressNew will be set to true to indicate that the new 
/// specialization or instantiation has no effect and should be ignored.
///
/// \returns true if there was an error that should prevent the introduction of
/// the new declaration into the AST, false otherwise.
bool
Sema::CheckSpecializationInstantiationRedecl(SourceLocation NewLoc,
                                             TemplateSpecializationKind NewTSK,
                                             NamedDecl *PrevDecl,
                                             TemplateSpecializationKind PrevTSK,
                                        SourceLocation PrevPointOfInstantiation,
                                             bool &SuppressNew) {
  SuppressNew = false;
  
  switch (NewTSK) {
  case TSK_Undeclared:
  case TSK_ImplicitInstantiation:
    assert(false && "Don't check implicit instantiations here");
    return false;
    
  case TSK_ExplicitSpecialization:
    switch (PrevTSK) {
    case TSK_Undeclared:
    case TSK_ExplicitSpecialization:
      // Okay, we're just specializing something that is either already 
      // explicitly specialized or has merely been mentioned without any
      // instantiation.
      return false;

    case TSK_ImplicitInstantiation:
      if (PrevPointOfInstantiation.isInvalid()) {
        // The declaration itself has not actually been instantiated, so it is
        // still okay to specialize it.
        return false;
      }
      // Fall through
        
    case TSK_ExplicitInstantiationDeclaration:
    case TSK_ExplicitInstantiationDefinition:
      assert((PrevTSK == TSK_ImplicitInstantiation || 
              PrevPointOfInstantiation.isValid()) && 
             "Explicit instantiation without point of instantiation?");
        
      // C++ [temp.expl.spec]p6:
      //   If a template, a member template or the member of a class template 
      //   is explicitly specialized then that specialization shall be declared
      //   before the first use of that specialization that would cause an 
      //   implicit instantiation to take place, in every translation unit in
      //   which such a use occurs; no diagnostic is required.
      Diag(NewLoc, diag::err_specialization_after_instantiation)
        << PrevDecl;
      Diag(PrevPointOfInstantiation, diag::note_instantiation_required_here)
        << (PrevTSK != TSK_ImplicitInstantiation);
      
      return true;
    }
    break;
      
  case TSK_ExplicitInstantiationDeclaration:
    switch (PrevTSK) {
    case TSK_ExplicitInstantiationDeclaration:
      // This explicit instantiation declaration is redundant (that's okay).
      SuppressNew = true;
      return false;
        
    case TSK_Undeclared:
    case TSK_ImplicitInstantiation:
      // We're explicitly instantiating something that may have already been
      // implicitly instantiated; that's fine.
      return false;
        
    case TSK_ExplicitSpecialization:
      // C++0x [temp.explicit]p4:
      //   For a given set of template parameters, if an explicit instantiation
      //   of a template appears after a declaration of an explicit 
      //   specialization for that template, the explicit instantiation has no
      //   effect.
      return false;
        
    case TSK_ExplicitInstantiationDefinition:
      // C++0x [temp.explicit]p10:
      //   If an entity is the subject of both an explicit instantiation 
      //   declaration and an explicit instantiation definition in the same 
      //   translation unit, the definition shall follow the declaration.
      Diag(NewLoc, 
           diag::err_explicit_instantiation_declaration_after_definition);
      Diag(PrevPointOfInstantiation, 
           diag::note_explicit_instantiation_definition_here);
      assert(PrevPointOfInstantiation.isValid() &&
             "Explicit instantiation without point of instantiation?");
      SuppressNew = true;
      return false;
    }
    break;
      
  case TSK_ExplicitInstantiationDefinition:
    switch (PrevTSK) {
    case TSK_Undeclared:
    case TSK_ImplicitInstantiation:
      // We're explicitly instantiating something that may have already been
      // implicitly instantiated; that's fine.
      return false;
        
    case TSK_ExplicitSpecialization:
      // C++ DR 259, C++0x [temp.explicit]p4:
      //   For a given set of template parameters, if an explicit
      //   instantiation of a template appears after a declaration of
      //   an explicit specialization for that template, the explicit
      //   instantiation has no effect.
      //
      // In C++98/03 mode, we only give an extension warning here, because it 
      // is not not harmful to try to explicitly instantiate something that
      // has been explicitly specialized.
      if (!getLangOptions().CPlusPlus0x) {
        Diag(NewLoc, diag::ext_explicit_instantiation_after_specialization)
          << PrevDecl;
        Diag(PrevDecl->getLocation(),
             diag::note_previous_template_specialization);
      }
      SuppressNew = true;
      return false;
        
    case TSK_ExplicitInstantiationDeclaration:
      // We're explicity instantiating a definition for something for which we
      // were previously asked to suppress instantiations. That's fine. 
      return false;
        
    case TSK_ExplicitInstantiationDefinition:
      // C++0x [temp.spec]p5:
      //   For a given template and a given set of template-arguments,
      //     - an explicit instantiation definition shall appear at most once
      //       in a program,
      Diag(NewLoc, diag::err_explicit_instantiation_duplicate)
        << PrevDecl;
      Diag(PrevPointOfInstantiation, 
           diag::note_previous_explicit_instantiation);
      SuppressNew = true;
      return false;        
    }
    break;
  }
  
  assert(false && "Missing specialization/instantiation case?");
         
  return false;
}

/// \brief Perform semantic analysis for the given function template 
/// specialization.
///
/// This routine performs all of the semantic analysis required for an 
/// explicit function template specialization. On successful completion,
/// the function declaration \p FD will become a function template
/// specialization.
///
/// \param FD the function declaration, which will be updated to become a
/// function template specialization.
///
/// \param HasExplicitTemplateArgs whether any template arguments were
/// explicitly provided.
///
/// \param LAngleLoc the location of the left angle bracket ('<'), if
/// template arguments were explicitly provided.
///
/// \param ExplicitTemplateArgs the explicitly-provided template arguments, 
/// if any.
///
/// \param NumExplicitTemplateArgs the number of explicitly-provided template
/// arguments. This number may be zero even when HasExplicitTemplateArgs is
/// true as in, e.g., \c void sort<>(char*, char*);
///
/// \param RAngleLoc the location of the right angle bracket ('>'), if
/// template arguments were explicitly provided.
/// 
/// \param PrevDecl the set of declarations that 
bool 
Sema::CheckFunctionTemplateSpecialization(FunctionDecl *FD,
                        const TemplateArgumentListInfo *ExplicitTemplateArgs,
                                          LookupResult &Previous) {
  // The set of function template specializations that could match this
  // explicit function template specialization.
  typedef llvm::SmallVector<FunctionDecl *, 8> CandidateSet;
  CandidateSet Candidates;
  
  DeclContext *FDLookupContext = FD->getDeclContext()->getLookupContext();
  for (LookupResult::iterator I = Previous.begin(), E = Previous.end();
         I != E; ++I) {
    NamedDecl *Ovl = (*I)->getUnderlyingDecl();
    if (FunctionTemplateDecl *FunTmpl = dyn_cast<FunctionTemplateDecl>(Ovl)) {
      // Only consider templates found within the same semantic lookup scope as 
      // FD.
      if (!FDLookupContext->Equals(Ovl->getDeclContext()->getLookupContext()))
        continue;
      
      // C++ [temp.expl.spec]p11:
      //   A trailing template-argument can be left unspecified in the 
      //   template-id naming an explicit function template specialization 
      //   provided it can be deduced from the function argument type.
      // Perform template argument deduction to determine whether we may be
      // specializing this template.
      // FIXME: It is somewhat wasteful to build
      TemplateDeductionInfo Info(Context);
      FunctionDecl *Specialization = 0;
      if (TemplateDeductionResult TDK
            = DeduceTemplateArguments(FunTmpl, ExplicitTemplateArgs,
                                      FD->getType(),
                                      Specialization,
                                      Info)) {
        // FIXME: Template argument deduction failed; record why it failed, so
        // that we can provide nifty diagnostics.
        (void)TDK;
        continue;
      }
      
      // Record this candidate.
      Candidates.push_back(Specialization);
    }
  }
  
  // Find the most specialized function template.
  FunctionDecl *Specialization = getMostSpecialized(Candidates.data(),
                                                    Candidates.size(),
                                                    TPOC_Other,
                                                    FD->getLocation(),
                  PartialDiagnostic(diag::err_function_template_spec_no_match) 
                    << FD->getDeclName(),
                  PartialDiagnostic(diag::err_function_template_spec_ambiguous)
                    << FD->getDeclName() << (ExplicitTemplateArgs != 0),
                  PartialDiagnostic(diag::note_function_template_spec_matched));
  if (!Specialization)
    return true;
  
  // FIXME: Check if the prior specialization has a point of instantiation.
  // If so, we have run afoul of .
  
  // Check the scope of this explicit specialization.
  if (CheckTemplateSpecializationScope(*this, 
                                       Specialization->getPrimaryTemplate(),
                                       Specialization, FD->getLocation(), 
                                       false))
    return true;

  // C++ [temp.expl.spec]p6:
  //   If a template, a member template or the member of a class template is
  //   explicitly specialized then that specialization shall be declared 
  //   before the first use of that specialization that would cause an implicit
  //   instantiation to take place, in every translation unit in which such a 
  //   use occurs; no diagnostic is required.
  FunctionTemplateSpecializationInfo *SpecInfo
    = Specialization->getTemplateSpecializationInfo();
  assert(SpecInfo && "Function template specialization info missing?");
  if (SpecInfo->getPointOfInstantiation().isValid()) {
    Diag(FD->getLocation(), diag::err_specialization_after_instantiation)
      << FD;
    Diag(SpecInfo->getPointOfInstantiation(), 
         diag::note_instantiation_required_here)
      << (Specialization->getTemplateSpecializationKind() 
                                                != TSK_ImplicitInstantiation);
    return true;
  }
  
  // Mark the prior declaration as an explicit specialization, so that later
  // clients know that this is an explicit specialization.
  SpecInfo->setTemplateSpecializationKind(TSK_ExplicitSpecialization);
  
  // Turn the given function declaration into a function template
  // specialization, with the template arguments from the previous
  // specialization.
  FD->setFunctionTemplateSpecialization(Context, 
                                        Specialization->getPrimaryTemplate(),
                         new (Context) TemplateArgumentList(
                             *Specialization->getTemplateSpecializationArgs()), 
                                        /*InsertPos=*/0, 
                                        TSK_ExplicitSpecialization);
  
  // The "previous declaration" for this function template specialization is
  // the prior function template specialization.
  Previous.clear();
  Previous.addDecl(Specialization);
  return false;
}

/// \brief Perform semantic analysis for the given non-template member
/// specialization.
///
/// This routine performs all of the semantic analysis required for an 
/// explicit member function specialization. On successful completion,
/// the function declaration \p FD will become a member function
/// specialization.
///
/// \param Member the member declaration, which will be updated to become a
/// specialization.
///
/// \param Previous the set of declarations, one of which may be specialized
/// by this function specialization;  the set will be modified to contain the
/// redeclared member.
bool 
Sema::CheckMemberSpecialization(NamedDecl *Member, LookupResult &Previous) {
  assert(!isa<TemplateDecl>(Member) && "Only for non-template members");
         
  // Try to find the member we are instantiating.
  NamedDecl *Instantiation = 0;
  NamedDecl *InstantiatedFrom = 0;
  MemberSpecializationInfo *MSInfo = 0;

  if (Previous.empty()) {
    // Nowhere to look anyway.
  } else if (FunctionDecl *Function = dyn_cast<FunctionDecl>(Member)) {
    for (LookupResult::iterator I = Previous.begin(), E = Previous.end();
           I != E; ++I) {
      NamedDecl *D = (*I)->getUnderlyingDecl();
      if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(D)) {
        if (Context.hasSameType(Function->getType(), Method->getType())) {
          Instantiation = Method;
          InstantiatedFrom = Method->getInstantiatedFromMemberFunction();
          MSInfo = Method->getMemberSpecializationInfo();
          break;
        }
      }
    }
  } else if (isa<VarDecl>(Member)) {
    VarDecl *PrevVar;
    if (Previous.isSingleResult() &&
        (PrevVar = dyn_cast<VarDecl>(Previous.getFoundDecl())))
      if (PrevVar->isStaticDataMember()) {
        Instantiation = PrevVar;
        InstantiatedFrom = PrevVar->getInstantiatedFromStaticDataMember();
        MSInfo = PrevVar->getMemberSpecializationInfo();
      }
  } else if (isa<RecordDecl>(Member)) {
    CXXRecordDecl *PrevRecord;
    if (Previous.isSingleResult() &&
        (PrevRecord = dyn_cast<CXXRecordDecl>(Previous.getFoundDecl()))) {
      Instantiation = PrevRecord;
      InstantiatedFrom = PrevRecord->getInstantiatedFromMemberClass();
      MSInfo = PrevRecord->getMemberSpecializationInfo();
    }
  }
  
  if (!Instantiation) {
    // There is no previous declaration that matches. Since member
    // specializations are always out-of-line, the caller will complain about
    // this mismatch later.
    return false;
  }
  
  // Make sure that this is a specialization of a member.
  if (!InstantiatedFrom) {
    Diag(Member->getLocation(), diag::err_spec_member_not_instantiated)
      << Member;
    Diag(Instantiation->getLocation(), diag::note_specialized_decl);
    return true;
  }
  
  // C++ [temp.expl.spec]p6:
  //   If a template, a member template or the member of a class template is
  //   explicitly specialized then that spe- cialization shall be declared 
  //   before the first use of that specialization that would cause an implicit
  //   instantiation to take place, in every translation unit in which such a 
  //   use occurs; no diagnostic is required.
  assert(MSInfo && "Member specialization info missing?");
  if (MSInfo->getPointOfInstantiation().isValid()) {
    Diag(Member->getLocation(), diag::err_specialization_after_instantiation)
      << Member;
    Diag(MSInfo->getPointOfInstantiation(), 
         diag::note_instantiation_required_here)
      << (MSInfo->getTemplateSpecializationKind() != TSK_ImplicitInstantiation);
    return true;
  }
  
  // Check the scope of this explicit specialization.
  if (CheckTemplateSpecializationScope(*this, 
                                       InstantiatedFrom,
                                       Instantiation, Member->getLocation(), 
                                       false))
    return true;

  // Note that this is an explicit instantiation of a member.
  // the original declaration to note that it is an explicit specialization
  // (if it was previously an implicit instantiation). This latter step
  // makes bookkeeping easier.
  if (isa<FunctionDecl>(Member)) {
    FunctionDecl *InstantiationFunction = cast<FunctionDecl>(Instantiation);
    if (InstantiationFunction->getTemplateSpecializationKind() ==
          TSK_ImplicitInstantiation) {
      InstantiationFunction->setTemplateSpecializationKind(
                                                  TSK_ExplicitSpecialization);
      InstantiationFunction->setLocation(Member->getLocation());
    }
    
    cast<FunctionDecl>(Member)->setInstantiationOfMemberFunction(
                                        cast<CXXMethodDecl>(InstantiatedFrom),
                                                  TSK_ExplicitSpecialization);
  } else if (isa<VarDecl>(Member)) {
    VarDecl *InstantiationVar = cast<VarDecl>(Instantiation);
    if (InstantiationVar->getTemplateSpecializationKind() ==
          TSK_ImplicitInstantiation) {
      InstantiationVar->setTemplateSpecializationKind(
                                                  TSK_ExplicitSpecialization);
      InstantiationVar->setLocation(Member->getLocation());
    }
    
    Context.setInstantiatedFromStaticDataMember(cast<VarDecl>(Member),
                                                cast<VarDecl>(InstantiatedFrom),
                                                TSK_ExplicitSpecialization);
  } else {
    assert(isa<CXXRecordDecl>(Member) && "Only member classes remain");
    CXXRecordDecl *InstantiationClass = cast<CXXRecordDecl>(Instantiation);
    if (InstantiationClass->getTemplateSpecializationKind() ==
          TSK_ImplicitInstantiation) {
      InstantiationClass->setTemplateSpecializationKind(
                                                   TSK_ExplicitSpecialization);
      InstantiationClass->setLocation(Member->getLocation());
    }
    
    cast<CXXRecordDecl>(Member)->setInstantiationOfMemberClass(
                                        cast<CXXRecordDecl>(InstantiatedFrom),
                                                   TSK_ExplicitSpecialization);
  }
             
  // Save the caller the trouble of having to figure out which declaration
  // this specialization matches.
  Previous.clear();
  Previous.addDecl(Instantiation);
  return false;
}

/// \brief Check the scope of an explicit instantiation.
static void CheckExplicitInstantiationScope(Sema &S, NamedDecl *D,
                                            SourceLocation InstLoc,
                                            bool WasQualifiedName) {
  DeclContext *ExpectedContext
    = D->getDeclContext()->getEnclosingNamespaceContext()->getLookupContext();
  DeclContext *CurContext = S.CurContext->getLookupContext();
  
  // C++0x [temp.explicit]p2:
  //   An explicit instantiation shall appear in an enclosing namespace of its 
  //   template.
  //
  // This is DR275, which we do not retroactively apply to C++98/03.
  if (S.getLangOptions().CPlusPlus0x && 
      !CurContext->Encloses(ExpectedContext)) {
    if (NamespaceDecl *NS = dyn_cast<NamespaceDecl>(ExpectedContext))
      S.Diag(InstLoc, diag::err_explicit_instantiation_out_of_scope)
        << D << NS;
    else
      S.Diag(InstLoc, diag::err_explicit_instantiation_must_be_global)
        << D;
    S.Diag(D->getLocation(), diag::note_explicit_instantiation_here);
    return;
  }
  
  // C++0x [temp.explicit]p2:
  //   If the name declared in the explicit instantiation is an unqualified 
  //   name, the explicit instantiation shall appear in the namespace where 
  //   its template is declared or, if that namespace is inline (7.3.1), any
  //   namespace from its enclosing namespace set.
  if (WasQualifiedName)
    return;
  
  if (CurContext->Equals(ExpectedContext))
    return;
  
  S.Diag(InstLoc, diag::err_explicit_instantiation_unqualified_wrong_namespace)
    << D << ExpectedContext;
  S.Diag(D->getLocation(), diag::note_explicit_instantiation_here);
}

/// \brief Determine whether the given scope specifier has a template-id in it.
static bool ScopeSpecifierHasTemplateId(const CXXScopeSpec &SS) {
  if (!SS.isSet())
    return false;
  
  // C++0x [temp.explicit]p2:
  //   If the explicit instantiation is for a member function, a member class 
  //   or a static data member of a class template specialization, the name of
  //   the class template specialization in the qualified-id for the member
  //   name shall be a simple-template-id.
  //
  // C++98 has the same restriction, just worded differently.
  for (NestedNameSpecifier *NNS = (NestedNameSpecifier *)SS.getScopeRep();
       NNS; NNS = NNS->getPrefix())
    if (Type *T = NNS->getAsType())
      if (isa<TemplateSpecializationType>(T))
        return true;

  return false;
}

// Explicit instantiation of a class template specialization
// FIXME: Implement extern template semantics
Sema::DeclResult
Sema::ActOnExplicitInstantiation(Scope *S,
                                 SourceLocation ExternLoc,
                                 SourceLocation TemplateLoc,
                                 unsigned TagSpec,
                                 SourceLocation KWLoc,
                                 const CXXScopeSpec &SS,
                                 TemplateTy TemplateD,
                                 SourceLocation TemplateNameLoc,
                                 SourceLocation LAngleLoc,
                                 ASTTemplateArgsPtr TemplateArgsIn,
                                 SourceLocation RAngleLoc,
                                 AttributeList *Attr) {
  // Find the class template we're specializing
  TemplateName Name = TemplateD.getAsVal<TemplateName>();
  ClassTemplateDecl *ClassTemplate
    = cast<ClassTemplateDecl>(Name.getAsTemplateDecl());

  // Check that the specialization uses the same tag kind as the
  // original template.
  TagDecl::TagKind Kind;
  switch (TagSpec) {
  default: assert(0 && "Unknown tag type!");
  case DeclSpec::TST_struct: Kind = TagDecl::TK_struct; break;
  case DeclSpec::TST_union:  Kind = TagDecl::TK_union; break;
  case DeclSpec::TST_class:  Kind = TagDecl::TK_class; break;
  }
  if (!isAcceptableTagRedeclaration(ClassTemplate->getTemplatedDecl(),
                                    Kind, KWLoc,
                                    *ClassTemplate->getIdentifier())) {
    Diag(KWLoc, diag::err_use_with_wrong_tag)
      << ClassTemplate
      << CodeModificationHint::CreateReplacement(KWLoc,
                            ClassTemplate->getTemplatedDecl()->getKindName());
    Diag(ClassTemplate->getTemplatedDecl()->getLocation(),
         diag::note_previous_use);
    Kind = ClassTemplate->getTemplatedDecl()->getTagKind();
  }

  // C++0x [temp.explicit]p2:
  //   There are two forms of explicit instantiation: an explicit instantiation
  //   definition and an explicit instantiation declaration. An explicit 
  //   instantiation declaration begins with the extern keyword. [...]  
  TemplateSpecializationKind TSK
    = ExternLoc.isInvalid()? TSK_ExplicitInstantiationDefinition
                           : TSK_ExplicitInstantiationDeclaration;
  
  // Translate the parser's template argument list in our AST format.
  TemplateArgumentListInfo TemplateArgs(LAngleLoc, RAngleLoc);
  translateTemplateArguments(TemplateArgsIn, TemplateArgs);

  // Check that the template argument list is well-formed for this
  // template.
  TemplateArgumentListBuilder Converted(ClassTemplate->getTemplateParameters(),
                                        TemplateArgs.size());
  if (CheckTemplateArgumentList(ClassTemplate, TemplateNameLoc,
                                TemplateArgs, false, Converted))
    return true;

  assert((Converted.structuredSize() ==
            ClassTemplate->getTemplateParameters()->size()) &&
         "Converted template argument list is too short!");

  // Find the class template specialization declaration that
  // corresponds to these arguments.
  llvm::FoldingSetNodeID ID;
  ClassTemplateSpecializationDecl::Profile(ID,
                                           Converted.getFlatArguments(),
                                           Converted.flatSize(),
                                           Context);
  void *InsertPos = 0;
  ClassTemplateSpecializationDecl *PrevDecl
    = ClassTemplate->getSpecializations().FindNodeOrInsertPos(ID, InsertPos);

  // C++0x [temp.explicit]p2:
  //   [...] An explicit instantiation shall appear in an enclosing
  //   namespace of its template. [...]
  //
  // This is C++ DR 275.
  CheckExplicitInstantiationScope(*this, ClassTemplate, TemplateNameLoc,
                                  SS.isSet());
  
  ClassTemplateSpecializationDecl *Specialization = 0;

  bool ReusedDecl = false;
  if (PrevDecl) {
    bool SuppressNew = false;
    if (CheckSpecializationInstantiationRedecl(TemplateNameLoc, TSK,
                                               PrevDecl, 
                                              PrevDecl->getSpecializationKind(), 
                                            PrevDecl->getPointOfInstantiation(),
                                               SuppressNew))
      return DeclPtrTy::make(PrevDecl);

    if (SuppressNew)
      return DeclPtrTy::make(PrevDecl);
    
    if (PrevDecl->getSpecializationKind() == TSK_ImplicitInstantiation ||
        PrevDecl->getSpecializationKind() == TSK_Undeclared) {
      // Since the only prior class template specialization with these
      // arguments was referenced but not declared, reuse that
      // declaration node as our own, updating its source location to
      // reflect our new declaration.
      Specialization = PrevDecl;
      Specialization->setLocation(TemplateNameLoc);
      PrevDecl = 0;
      ReusedDecl = true;
    }
  }
  
  if (!Specialization) {
    // Create a new class template specialization declaration node for
    // this explicit specialization.
    Specialization
      = ClassTemplateSpecializationDecl::Create(Context,
                                             ClassTemplate->getDeclContext(),
                                                TemplateNameLoc,
                                                ClassTemplate,
                                                Converted, PrevDecl);

    if (PrevDecl) {
      // Remove the previous declaration from the folding set, since we want
      // to introduce a new declaration.
      ClassTemplate->getSpecializations().RemoveNode(PrevDecl);
      ClassTemplate->getSpecializations().FindNodeOrInsertPos(ID, InsertPos);
    } 
    
    // Insert the new specialization.
    ClassTemplate->getSpecializations().InsertNode(Specialization, InsertPos);
  }

  // Build the fully-sugared type for this explicit instantiation as
  // the user wrote in the explicit instantiation itself. This means
  // that we'll pretty-print the type retrieved from the
  // specialization's declaration the way that the user actually wrote
  // the explicit instantiation, rather than formatting the name based
  // on the "canonical" representation used to store the template
  // arguments in the specialization.
  QualType WrittenTy
    = Context.getTemplateSpecializationType(Name, TemplateArgs,
                                  Context.getTypeDeclType(Specialization));
  Specialization->setTypeAsWritten(WrittenTy);
  TemplateArgsIn.release();

  if (!ReusedDecl) {
    // Add the explicit instantiation into its lexical context. However,
    // since explicit instantiations are never found by name lookup, we
    // just put it into the declaration context directly.
    Specialization->setLexicalDeclContext(CurContext);
    CurContext->addDecl(Specialization);
  }

  // C++ [temp.explicit]p3:
  //   A definition of a class template or class member template
  //   shall be in scope at the point of the explicit instantiation of
  //   the class template or class member template.
  //
  // This check comes when we actually try to perform the
  // instantiation.
  ClassTemplateSpecializationDecl *Def
    = cast_or_null<ClassTemplateSpecializationDecl>(
                                        Specialization->getDefinition(Context));
  if (!Def)
    InstantiateClassTemplateSpecialization(TemplateNameLoc, Specialization, TSK);
  
  // Instantiate the members of this class template specialization.
  Def = cast_or_null<ClassTemplateSpecializationDecl>(
                                       Specialization->getDefinition(Context));
  if (Def)
    InstantiateClassTemplateSpecializationMembers(TemplateNameLoc, Def, TSK);

  return DeclPtrTy::make(Specialization);
}

// Explicit instantiation of a member class of a class template.
Sema::DeclResult
Sema::ActOnExplicitInstantiation(Scope *S,
                                 SourceLocation ExternLoc,
                                 SourceLocation TemplateLoc,
                                 unsigned TagSpec,
                                 SourceLocation KWLoc,
                                 const CXXScopeSpec &SS,
                                 IdentifierInfo *Name,
                                 SourceLocation NameLoc,
                                 AttributeList *Attr) {

  bool Owned = false;
  bool IsDependent = false;
  DeclPtrTy TagD = ActOnTag(S, TagSpec, Action::TUK_Reference,
                            KWLoc, SS, Name, NameLoc, Attr, AS_none,
                            MultiTemplateParamsArg(*this, 0, 0),
                            Owned, IsDependent);
  assert(!IsDependent && "explicit instantiation of dependent name not yet handled");

  if (!TagD)
    return true;

  TagDecl *Tag = cast<TagDecl>(TagD.getAs<Decl>());
  if (Tag->isEnum()) {
    Diag(TemplateLoc, diag::err_explicit_instantiation_enum)
      << Context.getTypeDeclType(Tag);
    return true;
  }

  if (Tag->isInvalidDecl())
    return true;
    
  CXXRecordDecl *Record = cast<CXXRecordDecl>(Tag);
  CXXRecordDecl *Pattern = Record->getInstantiatedFromMemberClass();
  if (!Pattern) {
    Diag(TemplateLoc, diag::err_explicit_instantiation_nontemplate_type)
      << Context.getTypeDeclType(Record);
    Diag(Record->getLocation(), diag::note_nontemplate_decl_here);
    return true;
  }

  // C++0x [temp.explicit]p2:
  //   If the explicit instantiation is for a class or member class, the 
  //   elaborated-type-specifier in the declaration shall include a 
  //   simple-template-id.
  //
  // C++98 has the same restriction, just worded differently.
  if (!ScopeSpecifierHasTemplateId(SS))
    Diag(TemplateLoc, diag::err_explicit_instantiation_without_qualified_id)
      << Record << SS.getRange();
           
  // C++0x [temp.explicit]p2:
  //   There are two forms of explicit instantiation: an explicit instantiation
  //   definition and an explicit instantiation declaration. An explicit 
  //   instantiation declaration begins with the extern keyword. [...]
  TemplateSpecializationKind TSK
    = ExternLoc.isInvalid()? TSK_ExplicitInstantiationDefinition
                           : TSK_ExplicitInstantiationDeclaration;
  
  // C++0x [temp.explicit]p2:
  //   [...] An explicit instantiation shall appear in an enclosing
  //   namespace of its template. [...]
  //
  // This is C++ DR 275.
  CheckExplicitInstantiationScope(*this, Record, NameLoc, true);
  
  // Verify that it is okay to explicitly instantiate here.
  CXXRecordDecl *PrevDecl 
    = cast_or_null<CXXRecordDecl>(Record->getPreviousDeclaration());
  if (!PrevDecl && Record->getDefinition(Context))
    PrevDecl = Record;
  if (PrevDecl) {
    MemberSpecializationInfo *MSInfo = PrevDecl->getMemberSpecializationInfo();
    bool SuppressNew = false;
    assert(MSInfo && "No member specialization information?");
    if (CheckSpecializationInstantiationRedecl(TemplateLoc, TSK, 
                                               PrevDecl,
                                        MSInfo->getTemplateSpecializationKind(),
                                             MSInfo->getPointOfInstantiation(), 
                                               SuppressNew))
      return true;
    if (SuppressNew)
      return TagD;
  }
  
  CXXRecordDecl *RecordDef
    = cast_or_null<CXXRecordDecl>(Record->getDefinition(Context));
  if (!RecordDef) {
    // C++ [temp.explicit]p3:
    //   A definition of a member class of a class template shall be in scope 
    //   at the point of an explicit instantiation of the member class.
    CXXRecordDecl *Def 
      = cast_or_null<CXXRecordDecl>(Pattern->getDefinition(Context));
    if (!Def) {
      Diag(TemplateLoc, diag::err_explicit_instantiation_undefined_member)
        << 0 << Record->getDeclName() << Record->getDeclContext();
      Diag(Pattern->getLocation(), diag::note_forward_declaration)
        << Pattern;
      return true;
    } else {
      if (InstantiateClass(NameLoc, Record, Def,
                           getTemplateInstantiationArgs(Record),
                           TSK))
        return true;

      RecordDef = cast_or_null<CXXRecordDecl>(Record->getDefinition(Context));
      if (!RecordDef)
        return true;
    }
  } 
  
  // Instantiate all of the members of the class.
  InstantiateClassMembers(NameLoc, RecordDef,
                          getTemplateInstantiationArgs(Record), TSK);

  // FIXME: We don't have any representation for explicit instantiations of
  // member classes. Such a representation is not needed for compilation, but it
  // should be available for clients that want to see all of the declarations in
  // the source code.
  return TagD;
}

Sema::DeclResult Sema::ActOnExplicitInstantiation(Scope *S,
                                                  SourceLocation ExternLoc,
                                                  SourceLocation TemplateLoc,
                                                  Declarator &D) {
  // Explicit instantiations always require a name.
  DeclarationName Name = GetNameForDeclarator(D);
  if (!Name) {
    if (!D.isInvalidType())
      Diag(D.getDeclSpec().getSourceRange().getBegin(),
           diag::err_explicit_instantiation_requires_name)
        << D.getDeclSpec().getSourceRange()
        << D.getSourceRange();
    
    return true;
  }

  // The scope passed in may not be a decl scope.  Zip up the scope tree until
  // we find one that is.
  while ((S->getFlags() & Scope::DeclScope) == 0 ||
         (S->getFlags() & Scope::TemplateParamScope) != 0)
    S = S->getParent();

  // Determine the type of the declaration.
  QualType R = GetTypeForDeclarator(D, S, 0);
  if (R.isNull())
    return true;
  
  if (D.getDeclSpec().getStorageClassSpec() == DeclSpec::SCS_typedef) {
    // Cannot explicitly instantiate a typedef.
    Diag(D.getIdentifierLoc(), diag::err_explicit_instantiation_of_typedef)
      << Name;
    return true;
  }

  // C++0x [temp.explicit]p1:
  //   [...] An explicit instantiation of a function template shall not use the
  //   inline or constexpr specifiers.
  // Presumably, this also applies to member functions of class templates as
  // well.
  if (D.getDeclSpec().isInlineSpecified() && getLangOptions().CPlusPlus0x)
    Diag(D.getDeclSpec().getInlineSpecLoc(), 
         diag::err_explicit_instantiation_inline)
      <<CodeModificationHint::CreateRemoval(D.getDeclSpec().getInlineSpecLoc());
  
  // FIXME: check for constexpr specifier.
  
  // C++0x [temp.explicit]p2:
  //   There are two forms of explicit instantiation: an explicit instantiation
  //   definition and an explicit instantiation declaration. An explicit 
  //   instantiation declaration begins with the extern keyword. [...]  
  TemplateSpecializationKind TSK
    = ExternLoc.isInvalid()? TSK_ExplicitInstantiationDefinition
                           : TSK_ExplicitInstantiationDeclaration;
    
  LookupResult Previous(*this, Name, D.getIdentifierLoc(), LookupOrdinaryName);
  LookupParsedName(Previous, S, &D.getCXXScopeSpec());

  if (!R->isFunctionType()) {
    // C++ [temp.explicit]p1:
    //   A [...] static data member of a class template can be explicitly 
    //   instantiated from the member definition associated with its class 
    //   template.
    if (Previous.isAmbiguous())
      return true;
    
    VarDecl *Prev = Previous.getAsSingle<VarDecl>();
    if (!Prev || !Prev->isStaticDataMember()) {
      // We expect to see a data data member here.
      Diag(D.getIdentifierLoc(), diag::err_explicit_instantiation_not_known)
        << Name;
      for (LookupResult::iterator P = Previous.begin(), PEnd = Previous.end();
           P != PEnd; ++P)
        Diag((*P)->getLocation(), diag::note_explicit_instantiation_here);
      return true;
    }
    
    if (!Prev->getInstantiatedFromStaticDataMember()) {
      // FIXME: Check for explicit specialization?
      Diag(D.getIdentifierLoc(), 
           diag::err_explicit_instantiation_data_member_not_instantiated)
        << Prev;
      Diag(Prev->getLocation(), diag::note_explicit_instantiation_here);
      // FIXME: Can we provide a note showing where this was declared?
      return true;
    }
    
    // C++0x [temp.explicit]p2:
    //   If the explicit instantiation is for a member function, a member class 
    //   or a static data member of a class template specialization, the name of
    //   the class template specialization in the qualified-id for the member
    //   name shall be a simple-template-id.
    //
    // C++98 has the same restriction, just worded differently.
    if (!ScopeSpecifierHasTemplateId(D.getCXXScopeSpec()))
      Diag(D.getIdentifierLoc(), 
           diag::err_explicit_instantiation_without_qualified_id)
        << Prev << D.getCXXScopeSpec().getRange();
    
    // Check the scope of this explicit instantiation.
    CheckExplicitInstantiationScope(*this, Prev, D.getIdentifierLoc(), true);
    
    // Verify that it is okay to explicitly instantiate here.
    MemberSpecializationInfo *MSInfo = Prev->getMemberSpecializationInfo();
    assert(MSInfo && "Missing static data member specialization info?");
    bool SuppressNew = false;
    if (CheckSpecializationInstantiationRedecl(D.getIdentifierLoc(), TSK, Prev,
                                        MSInfo->getTemplateSpecializationKind(),
                                              MSInfo->getPointOfInstantiation(), 
                                               SuppressNew))
      return true;
    if (SuppressNew)
      return DeclPtrTy();
    
    // Instantiate static data member.
    Prev->setTemplateSpecializationKind(TSK, D.getIdentifierLoc());
    if (TSK == TSK_ExplicitInstantiationDefinition)
      InstantiateStaticDataMemberDefinition(D.getIdentifierLoc(), Prev, false,
                                            /*DefinitionRequired=*/true);
    
    // FIXME: Create an ExplicitInstantiation node?
    return DeclPtrTy();
  }
  
  // If the declarator is a template-id, translate the parser's template 
  // argument list into our AST format.
  bool HasExplicitTemplateArgs = false;
  TemplateArgumentListInfo TemplateArgs;
  if (D.getName().getKind() == UnqualifiedId::IK_TemplateId) {
    TemplateIdAnnotation *TemplateId = D.getName().TemplateId;
    TemplateArgs.setLAngleLoc(TemplateId->LAngleLoc);
    TemplateArgs.setRAngleLoc(TemplateId->RAngleLoc);
    ASTTemplateArgsPtr TemplateArgsPtr(*this,
                                       TemplateId->getTemplateArgs(),
                                       TemplateId->NumArgs);
    translateTemplateArguments(TemplateArgsPtr, TemplateArgs);
    HasExplicitTemplateArgs = true;
    TemplateArgsPtr.release();
  }
    
  // C++ [temp.explicit]p1:
  //   A [...] function [...] can be explicitly instantiated from its template. 
  //   A member function [...] of a class template can be explicitly 
  //  instantiated from the member definition associated with its class 
  //  template.
  llvm::SmallVector<FunctionDecl *, 8> Matches;
  for (LookupResult::iterator P = Previous.begin(), PEnd = Previous.end();
       P != PEnd; ++P) {
    NamedDecl *Prev = *P;
    if (!HasExplicitTemplateArgs) {
      if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(Prev)) {
        if (Context.hasSameUnqualifiedType(Method->getType(), R)) {
          Matches.clear();
          Matches.push_back(Method);
          break;
        }
      }
    }
    
    FunctionTemplateDecl *FunTmpl = dyn_cast<FunctionTemplateDecl>(Prev);
    if (!FunTmpl)
      continue;

    TemplateDeductionInfo Info(Context);
    FunctionDecl *Specialization = 0;
    if (TemplateDeductionResult TDK
          = DeduceTemplateArguments(FunTmpl,
                               (HasExplicitTemplateArgs ? &TemplateArgs : 0),
                                    R, Specialization, Info)) {
      // FIXME: Keep track of almost-matches?
      (void)TDK;
      continue;
    }
    
    Matches.push_back(Specialization);
  }
  
  // Find the most specialized function template specialization.
  FunctionDecl *Specialization
    = getMostSpecialized(Matches.data(), Matches.size(), TPOC_Other, 
                         D.getIdentifierLoc(), 
          PartialDiagnostic(diag::err_explicit_instantiation_not_known) << Name,
          PartialDiagnostic(diag::err_explicit_instantiation_ambiguous) << Name,
                PartialDiagnostic(diag::note_explicit_instantiation_candidate));

  if (!Specialization)
    return true;
  
  if (Specialization->getTemplateSpecializationKind() == TSK_Undeclared) {
    Diag(D.getIdentifierLoc(), 
         diag::err_explicit_instantiation_member_function_not_instantiated)
      << Specialization
      << (Specialization->getTemplateSpecializationKind() ==
          TSK_ExplicitSpecialization);
    Diag(Specialization->getLocation(), diag::note_explicit_instantiation_here);
    return true;
  } 
  
  FunctionDecl *PrevDecl = Specialization->getPreviousDeclaration();
  if (!PrevDecl && Specialization->isThisDeclarationADefinition())
    PrevDecl = Specialization;

  if (PrevDecl) {
    bool SuppressNew = false;
    if (CheckSpecializationInstantiationRedecl(D.getIdentifierLoc(), TSK,
                                               PrevDecl, 
                                     PrevDecl->getTemplateSpecializationKind(), 
                                          PrevDecl->getPointOfInstantiation(),
                                               SuppressNew))
      return true;
    
    // FIXME: We may still want to build some representation of this
    // explicit specialization.
    if (SuppressNew)
      return DeclPtrTy();
  }

  Specialization->setTemplateSpecializationKind(TSK, D.getIdentifierLoc());
  
  if (TSK == TSK_ExplicitInstantiationDefinition)
    InstantiateFunctionDefinition(D.getIdentifierLoc(), Specialization, 
                                  false, /*DefinitionRequired=*/true);
 
  // C++0x [temp.explicit]p2:
  //   If the explicit instantiation is for a member function, a member class 
  //   or a static data member of a class template specialization, the name of
  //   the class template specialization in the qualified-id for the member
  //   name shall be a simple-template-id.
  //
  // C++98 has the same restriction, just worded differently.
  FunctionTemplateDecl *FunTmpl = Specialization->getPrimaryTemplate();
  if (D.getName().getKind() != UnqualifiedId::IK_TemplateId && !FunTmpl &&
      D.getCXXScopeSpec().isSet() && 
      !ScopeSpecifierHasTemplateId(D.getCXXScopeSpec()))
    Diag(D.getIdentifierLoc(), 
         diag::err_explicit_instantiation_without_qualified_id)
    << Specialization << D.getCXXScopeSpec().getRange();
  
  CheckExplicitInstantiationScope(*this,
                   FunTmpl? (NamedDecl *)FunTmpl 
                          : Specialization->getInstantiatedFromMemberFunction(),
                                  D.getIdentifierLoc(), 
                                  D.getCXXScopeSpec().isSet());
  
  // FIXME: Create some kind of ExplicitInstantiationDecl here.
  return DeclPtrTy();
}

Sema::TypeResult
Sema::ActOnDependentTag(Scope *S, unsigned TagSpec, TagUseKind TUK,
                        const CXXScopeSpec &SS, IdentifierInfo *Name,
                        SourceLocation TagLoc, SourceLocation NameLoc) {
  // This has to hold, because SS is expected to be defined.
  assert(Name && "Expected a name in a dependent tag");

  NestedNameSpecifier *NNS
    = static_cast<NestedNameSpecifier *>(SS.getScopeRep());
  if (!NNS)
    return true;

  QualType T = CheckTypenameType(NNS, *Name, SourceRange(TagLoc, NameLoc));
  if (T.isNull())
    return true;

  TagDecl::TagKind TagKind = TagDecl::getTagKindForTypeSpec(TagSpec);
  QualType ElabType = Context.getElaboratedType(T, TagKind);

  return ElabType.getAsOpaquePtr();
}

Sema::TypeResult
Sema::ActOnTypenameType(SourceLocation TypenameLoc, const CXXScopeSpec &SS,
                        const IdentifierInfo &II, SourceLocation IdLoc) {
  NestedNameSpecifier *NNS
    = static_cast<NestedNameSpecifier *>(SS.getScopeRep());
  if (!NNS)
    return true;

  QualType T = CheckTypenameType(NNS, II, SourceRange(TypenameLoc, IdLoc));
  if (T.isNull())
    return true;
  return T.getAsOpaquePtr();
}

Sema::TypeResult
Sema::ActOnTypenameType(SourceLocation TypenameLoc, const CXXScopeSpec &SS,
                        SourceLocation TemplateLoc, TypeTy *Ty) {
  QualType T = GetTypeFromParser(Ty);
  NestedNameSpecifier *NNS
    = static_cast<NestedNameSpecifier *>(SS.getScopeRep());
  const TemplateSpecializationType *TemplateId
    = T->getAs<TemplateSpecializationType>();
  assert(TemplateId && "Expected a template specialization type");

  if (computeDeclContext(SS, false)) {
    // If we can compute a declaration context, then the "typename"
    // keyword was superfluous. Just build a QualifiedNameType to keep
    // track of the nested-name-specifier.

    // FIXME: Note that the QualifiedNameType had the "typename" keyword!
    return Context.getQualifiedNameType(NNS, T).getAsOpaquePtr();
  }

  return Context.getTypenameType(NNS, TemplateId).getAsOpaquePtr();
}

/// \brief Build the type that describes a C++ typename specifier,
/// e.g., "typename T::type".
QualType
Sema::CheckTypenameType(NestedNameSpecifier *NNS, const IdentifierInfo &II,
                        SourceRange Range) {
  CXXRecordDecl *CurrentInstantiation = 0;
  if (NNS->isDependent()) {
    CurrentInstantiation = getCurrentInstantiationOf(NNS);

    // If the nested-name-specifier does not refer to the current
    // instantiation, then build a typename type.
    if (!CurrentInstantiation)
      return Context.getTypenameType(NNS, &II);

    // The nested-name-specifier refers to the current instantiation, so the
    // "typename" keyword itself is superfluous. In C++03, the program is
    // actually ill-formed. However, DR 382 (in C++0x CD1) allows such
    // extraneous "typename" keywords, and we retroactively apply this DR to
    // C++03 code.
  }

  DeclContext *Ctx = 0;

  if (CurrentInstantiation)
    Ctx = CurrentInstantiation;
  else {
    CXXScopeSpec SS;
    SS.setScopeRep(NNS);
    SS.setRange(Range);
    if (RequireCompleteDeclContext(SS))
      return QualType();

    Ctx = computeDeclContext(SS);
  }
  assert(Ctx && "No declaration context?");

  DeclarationName Name(&II);
  LookupResult Result(*this, Name, Range.getEnd(), LookupOrdinaryName);
  LookupQualifiedName(Result, Ctx);
  unsigned DiagID = 0;
  Decl *Referenced = 0;
  switch (Result.getResultKind()) {
  case LookupResult::NotFound:
    DiagID = diag::err_typename_nested_not_found;
    break;

  case LookupResult::Found:
    if (TypeDecl *Type = dyn_cast<TypeDecl>(Result.getFoundDecl())) {
      // We found a type. Build a QualifiedNameType, since the
      // typename-specifier was just sugar. FIXME: Tell
      // QualifiedNameType that it has a "typename" prefix.
      return Context.getQualifiedNameType(NNS, Context.getTypeDeclType(Type));
    }

    DiagID = diag::err_typename_nested_not_type;
    Referenced = Result.getFoundDecl();
    break;

  case LookupResult::FoundUnresolvedValue:
    llvm_unreachable("unresolved using decl in non-dependent context");
    return QualType();

  case LookupResult::FoundOverloaded:
    DiagID = diag::err_typename_nested_not_type;
    Referenced = *Result.begin();
    break;

  case LookupResult::Ambiguous:
    return QualType();
  }

  // If we get here, it's because name lookup did not find a
  // type. Emit an appropriate diagnostic and return an error.
  Diag(Range.getEnd(), DiagID) << Range << Name << Ctx;
  if (Referenced)
    Diag(Referenced->getLocation(), diag::note_typename_refers_here)
      << Name;
  return QualType();
}

namespace {
  // See Sema::RebuildTypeInCurrentInstantiation
  class CurrentInstantiationRebuilder
    : public TreeTransform<CurrentInstantiationRebuilder> {
    SourceLocation Loc;
    DeclarationName Entity;

  public:
    CurrentInstantiationRebuilder(Sema &SemaRef,
                                  SourceLocation Loc,
                                  DeclarationName Entity)
    : TreeTransform<CurrentInstantiationRebuilder>(SemaRef),
      Loc(Loc), Entity(Entity) { }

    /// \brief Determine whether the given type \p T has already been
    /// transformed.
    ///
    /// For the purposes of type reconstruction, a type has already been
    /// transformed if it is NULL or if it is not dependent.
    bool AlreadyTransformed(QualType T) {
      return T.isNull() || !T->isDependentType();
    }

    /// \brief Returns the location of the entity whose type is being
    /// rebuilt.
    SourceLocation getBaseLocation() { return Loc; }

    /// \brief Returns the name of the entity whose type is being rebuilt.
    DeclarationName getBaseEntity() { return Entity; }

    /// \brief Sets the "base" location and entity when that
    /// information is known based on another transformation.
    void setBase(SourceLocation Loc, DeclarationName Entity) {
      this->Loc = Loc;
      this->Entity = Entity;
    }
      
    /// \brief Transforms an expression by returning the expression itself
    /// (an identity function).
    ///
    /// FIXME: This is completely unsafe; we will need to actually clone the
    /// expressions.
    Sema::OwningExprResult TransformExpr(Expr *E) {
      return getSema().Owned(E);
    }

    /// \brief Transforms a typename type by determining whether the type now
    /// refers to a member of the current instantiation, and then
    /// type-checking and building a QualifiedNameType (when possible).
    QualType TransformTypenameType(TypeLocBuilder &TLB, TypenameTypeLoc TL);
  };
}

QualType
CurrentInstantiationRebuilder::TransformTypenameType(TypeLocBuilder &TLB,
                                                     TypenameTypeLoc TL) {
  TypenameType *T = TL.getTypePtr();

  NestedNameSpecifier *NNS
    = TransformNestedNameSpecifier(T->getQualifier(),
                              /*FIXME:*/SourceRange(getBaseLocation()));
  if (!NNS)
    return QualType();

  // If the nested-name-specifier did not change, and we cannot compute the
  // context corresponding to the nested-name-specifier, then this
  // typename type will not change; exit early.
  CXXScopeSpec SS;
  SS.setRange(SourceRange(getBaseLocation()));
  SS.setScopeRep(NNS);

  QualType Result;
  if (NNS == T->getQualifier() && getSema().computeDeclContext(SS) == 0)
    Result = QualType(T, 0);

  // Rebuild the typename type, which will probably turn into a
  // QualifiedNameType.
  else if (const TemplateSpecializationType *TemplateId = T->getTemplateId()) {
    QualType NewTemplateId
      = TransformType(QualType(TemplateId, 0));
    if (NewTemplateId.isNull())
      return QualType();

    if (NNS == T->getQualifier() &&
        NewTemplateId == QualType(TemplateId, 0))
      Result = QualType(T, 0);
    else
      Result = getDerived().RebuildTypenameType(NNS, NewTemplateId);
  } else
    Result = getDerived().RebuildTypenameType(NNS, T->getIdentifier(),
                                              SourceRange(TL.getNameLoc()));

  TypenameTypeLoc NewTL = TLB.push<TypenameTypeLoc>(Result);
  NewTL.setNameLoc(TL.getNameLoc());
  return Result;
}

/// \brief Rebuilds a type within the context of the current instantiation.
///
/// The type \p T is part of the type of an out-of-line member definition of
/// a class template (or class template partial specialization) that was parsed
/// and constructed before we entered the scope of the class template (or
/// partial specialization thereof). This routine will rebuild that type now
/// that we have entered the declarator's scope, which may produce different
/// canonical types, e.g.,
///
/// \code
/// template<typename T>
/// struct X {
///   typedef T* pointer;
///   pointer data();
/// };
///
/// template<typename T>
/// typename X<T>::pointer X<T>::data() { ... }
/// \endcode
///
/// Here, the type "typename X<T>::pointer" will be created as a TypenameType,
/// since we do not know that we can look into X<T> when we parsed the type.
/// This function will rebuild the type, performing the lookup of "pointer"
/// in X<T> and returning a QualifiedNameType whose canonical type is the same
/// as the canonical type of T*, allowing the return types of the out-of-line
/// definition and the declaration to match.
QualType Sema::RebuildTypeInCurrentInstantiation(QualType T, SourceLocation Loc,
                                                 DeclarationName Name) {
  if (T.isNull() || !T->isDependentType())
    return T;

  CurrentInstantiationRebuilder Rebuilder(*this, Loc, Name);
  return Rebuilder.TransformType(T);
}

/// \brief Produces a formatted string that describes the binding of
/// template parameters to template arguments.
std::string
Sema::getTemplateArgumentBindingsText(const TemplateParameterList *Params,
                                      const TemplateArgumentList &Args) {
  // FIXME: For variadic templates, we'll need to get the structured list.
  return getTemplateArgumentBindingsText(Params, Args.getFlatArgumentList(),
                                         Args.flat_size());
}

std::string
Sema::getTemplateArgumentBindingsText(const TemplateParameterList *Params,
                                      const TemplateArgument *Args,
                                      unsigned NumArgs) {
  std::string Result;

  if (!Params || Params->size() == 0 || NumArgs == 0)
    return Result;
  
  for (unsigned I = 0, N = Params->size(); I != N; ++I) {
    if (I >= NumArgs)
      break;
    
    if (I == 0)
      Result += "[with ";
    else
      Result += ", ";
    
    if (const IdentifierInfo *Id = Params->getParam(I)->getIdentifier()) {
      Result += Id->getName();
    } else {
      Result += '$';
      Result += llvm::utostr(I);
    }
    
    Result += " = ";
    
    switch (Args[I].getKind()) {
      case TemplateArgument::Null:
        Result += "<no value>";
        break;
        
      case TemplateArgument::Type: {
        std::string TypeStr;
        Args[I].getAsType().getAsStringInternal(TypeStr, 
                                                Context.PrintingPolicy);
        Result += TypeStr;
        break;
      }
        
      case TemplateArgument::Declaration: {
        bool Unnamed = true;
        if (NamedDecl *ND = dyn_cast_or_null<NamedDecl>(Args[I].getAsDecl())) {
          if (ND->getDeclName()) {
            Unnamed = false;
            Result += ND->getNameAsString();
          }
        }
        
        if (Unnamed) {
          Result += "<anonymous>";
        }
        break;
      }
        
      case TemplateArgument::Template: {
        std::string Str;
        llvm::raw_string_ostream OS(Str);
        Args[I].getAsTemplate().print(OS, Context.PrintingPolicy);
        Result += OS.str();
        break;
      }
        
      case TemplateArgument::Integral: {
        Result += Args[I].getAsIntegral()->toString(10);
        break;
      }
        
      case TemplateArgument::Expression: {
        assert(false && "No expressions in deduced template arguments!");
        Result += "<expression>";
        break;
      }
        
      case TemplateArgument::Pack:
        // FIXME: Format template argument packs
        Result += "<template argument pack>";
        break;        
    }
  }
  
  Result += ']';
  return Result;
}
