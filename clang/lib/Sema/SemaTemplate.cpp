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
#include "TreeTransform.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/Support/Compiler.h"

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
      Record = cast<CXXRecordDecl>(Record->getCanonicalDecl());
      if (Record->getDescribedClassTemplate())
        return Record->getDescribedClassTemplate();

      if (ClassTemplateSpecializationDecl *Spec
            = dyn_cast<ClassTemplateSpecializationDecl>(Record))
        return Spec->getSpecializedTemplate();
    }

    return 0;
  }

  OverloadedFunctionDecl *Ovl = dyn_cast<OverloadedFunctionDecl>(D);
  if (!Ovl)
    return 0;

  for (OverloadedFunctionDecl::function_iterator F = Ovl->function_begin(),
                                              FEnd = Ovl->function_end();
       F != FEnd; ++F) {
    if (FunctionTemplateDecl *FuncTmpl = dyn_cast<FunctionTemplateDecl>(*F)) {
      // We've found a function template. Determine whether there are
      // any other function templates we need to bundle together in an
      // OverloadedFunctionDecl
      for (++F; F != FEnd; ++F) {
        if (isa<FunctionTemplateDecl>(*F))
          break;
      }

      if (F != FEnd) {
        // Build an overloaded function decl containing only the
        // function templates in Ovl.
        OverloadedFunctionDecl *OvlTemplate
          = OverloadedFunctionDecl::Create(Context,
                                           Ovl->getDeclContext(),
                                           Ovl->getDeclName());
        OvlTemplate->addOverload(FuncTmpl);
        OvlTemplate->addOverload(*F);
        for (++F; F != FEnd; ++F) {
          if (isa<FunctionTemplateDecl>(*F))
            OvlTemplate->addOverload(*F);
        }

        return OvlTemplate;
      }

      return FuncTmpl;
    }
  }

  return 0;
}

TemplateNameKind Sema::isTemplateName(Scope *S,
                                      const IdentifierInfo &II,
                                      SourceLocation IdLoc,
                                      const CXXScopeSpec *SS,
                                      TypeTy *ObjectTypePtr,
                                      bool EnteringContext,
                                      TemplateTy &TemplateResult) {
  // Determine where to perform name lookup
  DeclContext *LookupCtx = 0;
  bool isDependent = false;
  if (ObjectTypePtr) {
    // This nested-name-specifier occurs in a member access expression, e.g.,
    // x->B::f, and we are looking into the type of the object.
    assert((!SS || !SS->isSet()) &&
           "ObjectType and scope specifier cannot coexist");
    QualType ObjectType = QualType::getFromOpaquePtr(ObjectTypePtr);
    LookupCtx = computeDeclContext(ObjectType);
    isDependent = ObjectType->isDependentType();
  } else if (SS && SS->isSet()) {
    // This nested-name-specifier occurs after another nested-name-specifier,
    // so long into the context associated with the prior nested-name-specifier.

    LookupCtx = computeDeclContext(*SS, EnteringContext);
    isDependent = isDependentScopeSpecifier(*SS);
  }

  LookupResult Found;
  bool ObjectTypeSearchedInScope = false;
  if (LookupCtx) {
    // Perform "qualified" name lookup into the declaration context we
    // computed, which is either the type of the base of a member access
    // expression or the declaration context associated with a prior
    // nested-name-specifier.

    // The declaration context must be complete.
    if (!LookupCtx->isDependentContext() && RequireCompleteDeclContext(*SS))
      return TNK_Non_template;

    Found = LookupQualifiedName(LookupCtx, &II, LookupOrdinaryName);

    if (ObjectTypePtr && Found.getKind() == LookupResult::NotFound) {
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
      Found = LookupName(S, &II, LookupOrdinaryName);
      ObjectTypeSearchedInScope = true;
    }
  } else if (isDependent) {
    // We cannot look into a dependent object type or
    return TNK_Non_template;
  } else {
    // Perform unqualified name lookup in the current scope.
    Found = LookupName(S, &II, LookupOrdinaryName);
  }

  // FIXME: Cope with ambiguous name-lookup results.
  assert(!Found.isAmbiguous() &&
         "Cannot handle template name-lookup ambiguities");

  NamedDecl *Template = isAcceptableTemplateName(Context, Found);
  if (!Template)
    return TNK_Non_template;

  if (ObjectTypePtr && !ObjectTypeSearchedInScope) {
    // C++ [basic.lookup.classref]p1:
    //   [...] If the lookup in the class of the object expression finds a
    //   template, the name is also looked up in the context of the entire
    //   postfix-expression and [...]
    //
    LookupResult FoundOuter = LookupName(S, &II, LookupOrdinaryName);
    // FIXME: Handle ambiguities in this lookup better
    NamedDecl *OuterTemplate = isAcceptableTemplateName(Context, FoundOuter);

    if (!OuterTemplate) {
      //   - if the name is not found, the name found in the class of the
      //     object expression is used, otherwise
    } else if (!isa<ClassTemplateDecl>(OuterTemplate)) {
      //   - if the name is found in the context of the entire
      //     postfix-expression and does not name a class template, the name
      //     found in the class of the object expression is used, otherwise
    } else {
      //   - if the name found is a class template, it must refer to the same
      //     entity as the one found in the class of the object expression,
      //     otherwise the program is ill-formed.
      if (OuterTemplate->getCanonicalDecl() != Template->getCanonicalDecl()) {
        Diag(IdLoc, diag::err_nested_name_member_ref_lookup_ambiguous)
          << &II;
        Diag(Template->getLocation(), diag::note_ambig_member_ref_object_type)
          << QualType::getFromOpaquePtr(ObjectTypePtr);
        Diag(OuterTemplate->getLocation(), diag::note_ambig_member_ref_scope);

        // Recover by taking the template that we found in the object
        // expression's type.
      }
    }
  }

  if (SS && SS->isSet() && !SS->isInvalid()) {
    NestedNameSpecifier *Qualifier
      = static_cast<NestedNameSpecifier *>(SS->getScopeRep());
    if (OverloadedFunctionDecl *Ovl
          = dyn_cast<OverloadedFunctionDecl>(Template))
      TemplateResult
        = TemplateTy::make(Context.getQualifiedTemplateName(Qualifier, false,
                                                            Ovl));
    else
      TemplateResult
        = TemplateTy::make(Context.getQualifiedTemplateName(Qualifier, false,
                                                 cast<TemplateDecl>(Template)));
  } else if (OverloadedFunctionDecl *Ovl
               = dyn_cast<OverloadedFunctionDecl>(Template)) {
    TemplateResult = TemplateTy::make(TemplateName(Ovl));
  } else {
    TemplateResult = TemplateTy::make(
                                  TemplateName(cast<TemplateDecl>(Template)));
  }

  if (isa<ClassTemplateDecl>(Template) ||
      isa<TemplateTemplateParmDecl>(Template))
    return TNK_Type_template;

  assert((isa<FunctionTemplateDecl>(Template) ||
          isa<OverloadedFunctionDecl>(Template)) &&
         "Unhandled template kind in Sema::isTemplateName");
  return TNK_Function_template;
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
  if (TemplateDecl *Temp = dyn_cast<TemplateDecl>(D.getAs<Decl>())) {
    D = DeclPtrTy::make(Temp->getTemplatedDecl());
    return Temp;
  }
  return 0;
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
    NamedDecl *PrevDecl = LookupName(S, ParamName, LookupTagName);
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
  // FIXME: Preserve type source info.
  QualType Default = GetTypeFromParser(DefaultT);

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
  if (CheckTemplateArgument(Parm, Default, DefaultLoc)) {
    Parm->setInvalidDecl();
    return;
  }

  Parm->setDefaultArgument(Default, DefaultLoc, false);
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
  DeclaratorInfo *DInfo = 0;
  QualType T = GetTypeForDeclarator(D, S, &DInfo);

  assert(S->isTemplateParamScope() &&
         "Non-type template parameter not in template parameter scope!");
  bool Invalid = false;

  IdentifierInfo *ParamName = D.getIdentifier();
  if (ParamName) {
    NamedDecl *PrevDecl = LookupName(S, ParamName, LookupTagName);
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
                                      Depth, Position, ParamName, T, DInfo);
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
                                                 ExprArg DefaultE) {
  TemplateTemplateParmDecl *TemplateParm
    = cast<TemplateTemplateParmDecl>(TemplateParamD.getAs<Decl>());

  // Since a template-template parameter's default argument is an
  // id-expression, it must be a DeclRefExpr.
  DeclRefExpr *Default
    = cast<DeclRefExpr>(static_cast<Expr *>(DefaultE.get()));

  // C++ [temp.param]p14:
  //   A template-parameter shall not be used in its own default argument.
  // FIXME: Implement this check! Needs a recursive walk over the types.

  // Check the well-formedness of the template argument.
  if (!isa<TemplateDecl>(Default->getDecl())) {
    Diag(Default->getSourceRange().getBegin(),
         diag::err_template_arg_must_be_template)
      << Default->getSourceRange();
    TemplateParm->setInvalidDecl();
    return;
  }
  if (CheckTemplateArgument(TemplateParm, Default)) {
    TemplateParm->setInvalidDecl();
    return;
  }

  DefaultE.release();
  TemplateParm->setDefaultArgument(Default);
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
    Diag(ExportLoc, diag::note_template_export_unsupported);

  return TemplateParameterList::Create(Context, TemplateLoc, LAngleLoc,
                                       (Decl**)Params, NumParams, RAngleLoc);
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

  TagDecl::TagKind Kind;
  switch (TagSpec) {
  default: assert(0 && "Unknown tag type!");
  case DeclSpec::TST_struct: Kind = TagDecl::TK_struct; break;
  case DeclSpec::TST_union:  Kind = TagDecl::TK_union; break;
  case DeclSpec::TST_class:  Kind = TagDecl::TK_class; break;
  }

  // There is no such thing as an unnamed class template.
  if (!Name) {
    Diag(KWLoc, diag::err_template_unnamed_class);
    return true;
  }

  // Find any previous declaration with this name.
  DeclContext *SemanticContext;
  LookupResult Previous;
  if (SS.isNotEmpty() && !SS.isInvalid()) {
    SemanticContext = computeDeclContext(SS, true);
    if (!SemanticContext) {
      // FIXME: Produce a reasonable diagnostic here
      return true;
    }

    Previous = LookupQualifiedName(SemanticContext, Name, LookupOrdinaryName,
                                   true);
  } else {
    SemanticContext = CurContext;
    Previous = LookupName(S, Name, LookupOrdinaryName, true);
  }

  assert(!Previous.isAmbiguous() && "Ambiguity in class template redecl?");
  NamedDecl *PrevDecl = 0;
  if (Previous.begin() != Previous.end())
    PrevDecl = *Previous.begin();

  if (PrevDecl && !isDeclInScope(PrevDecl, SemanticContext, S))
    PrevDecl = 0;

  // If there is a previous declaration with the same name, check
  // whether this is a valid redeclaration.
  ClassTemplateDecl *PrevClassTemplate
    = dyn_cast_or_null<ClassTemplateDecl>(PrevDecl);
  if (PrevClassTemplate) {
    // Ensure that the template parameter lists are compatible.
    if (!TemplateParameterListsAreEqual(TemplateParams,
                                   PrevClassTemplate->getTemplateParameters(),
                                        /*Complain=*/true))
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
            PrevClassTemplate? PrevClassTemplate->getTemplateParameters() : 0))
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

  // Set the access specifier.
  SetMemberAccessSpecifier(NewTemplate, PrevClassTemplate, AS);

  // Set the lexical context of these templates
  NewClass->setLexicalDeclContext(CurContext);
  NewTemplate->setLexicalDeclContext(CurContext);

  if (TUK == TUK_Definition)
    NewClass->startDefinition();

  if (Attr)
    ProcessDeclAttributeList(S, NewClass, Attr);

  PushOnScopeChains(NewTemplate, S);

  if (Invalid) {
    NewTemplate->setInvalidDecl();
    NewClass->setInvalidDecl();
  }
  return DeclPtrTy::make(NewTemplate);
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
/// \returns true if an error occurred, false otherwise.
bool Sema::CheckTemplateParameterList(TemplateParameterList *NewParams,
                                      TemplateParameterList *OldParams) {
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

    // Merge default arguments for template type parameters.
    if (TemplateTypeParmDecl *NewTypeParm
          = dyn_cast<TemplateTypeParmDecl>(*NewParam)) {
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
        NewTypeParm->setDefaultArgument(OldTypeParm->getDefaultArgument(),
                                        OldTypeParm->getDefaultArgumentLoc(),
                                        true);
        PreviousDefaultArgLoc = OldTypeParm->getDefaultArgumentLoc();
      } else if (NewTypeParm->hasDefaultArgument()) {
        SawDefaultArgument = true;
        PreviousDefaultArgLoc = NewTypeParm->getDefaultArgumentLoc();
      } else if (SawDefaultArgument)
        MissingDefaultArg = true;
    } else if (NonTypeTemplateParmDecl *NewNonTypeParm
               = dyn_cast<NonTypeTemplateParmDecl>(*NewParam)) {
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
    // Merge default arguments for template template parameters
      TemplateTemplateParmDecl *NewTemplateParm
        = cast<TemplateTemplateParmDecl>(*NewParam);
      TemplateTemplateParmDecl *OldTemplateParm
        = OldParams? cast<TemplateTemplateParmDecl>(*OldParam) : 0;
      if (OldTemplateParm && OldTemplateParm->hasDefaultArgument() &&
          NewTemplateParm->hasDefaultArgument()) {
        OldDefaultLoc = OldTemplateParm->getDefaultArgumentLoc();
        NewDefaultLoc = NewTemplateParm->getDefaultArgumentLoc();
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
        PreviousDefaultArgLoc = OldTemplateParm->getDefaultArgumentLoc();
      } else if (NewTemplateParm->hasDefaultArgument()) {
        SawDefaultArgument = true;
        PreviousDefaultArgLoc = NewTemplateParm->getDefaultArgumentLoc();
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
                                              unsigned NumParamLists) {
  // Find the template-ids that occur within the nested-name-specifier. These
  // template-ids will match up with the template parameter lists.
  llvm::SmallVector<const TemplateSpecializationType *, 4>
    TemplateIdsInSpecifier;
  for (NestedNameSpecifier *NNS = (NestedNameSpecifier *)SS.getScopeRep();
       NNS; NNS = NNS->getPrefix()) {
    if (const TemplateSpecializationType *SpecType
          = dyn_cast_or_null<TemplateSpecializationType>(NNS->getAsType())) {
      TemplateDecl *Template = SpecType->getTemplateName().getAsTemplateDecl();
      if (!Template)
        continue; // FIXME: should this be an error? probably...

      if (const RecordType *Record = SpecType->getAs<RecordType>()) {
        ClassTemplateSpecializationDecl *SpecDecl
          = cast<ClassTemplateSpecializationDecl>(Record->getDecl());
        // If the nested name specifier refers to an explicit specialization,
        // we don't need a template<> header.
        // FIXME: revisit this approach once we cope with specialization
        // properly.
        if (SpecDecl->getSpecializationKind() == TSK_ExplicitSpecialization)
          continue;
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
                                         true);
      }
    } else if (ParamLists[Idx]->size() > 0)
      Diag(ParamLists[Idx]->getTemplateLoc(),
           diag::err_template_param_list_matches_nontemplate)
        << TemplateId
        << ParamLists[Idx]->getSourceRange();
  }

  // If there were at least as many template-ids as there were template
  // parameter lists, then there are no template parameter lists remaining for
  // the declaration itself.
  if (Idx >= NumParamLists)
    return 0;

  // If there were too many template parameter lists, complain about that now.
  if (Idx != NumParamLists - 1) {
    while (Idx < NumParamLists - 1) {
      Diag(ParamLists[Idx]->getTemplateLoc(),
           diag::err_template_spec_extra_headers)
        << SourceRange(ParamLists[Idx]->getTemplateLoc(),
                       ParamLists[Idx]->getRAngleLoc());
      ++Idx;
    }
  }

  // Return the last template parameter list, which corresponds to the
  // entity being declared.
  return ParamLists[NumParamLists - 1];
}

/// \brief Translates template arguments as provided by the parser
/// into template arguments used by semantic analysis.
static void
translateTemplateArguments(ASTTemplateArgsPtr &TemplateArgsIn,
                           SourceLocation *TemplateArgLocs,
                     llvm::SmallVector<TemplateArgument, 16> &TemplateArgs) {
  TemplateArgs.reserve(TemplateArgsIn.size());

  void **Args = TemplateArgsIn.getArgs();
  bool *ArgIsType = TemplateArgsIn.getArgIsType();
  for (unsigned Arg = 0, Last = TemplateArgsIn.size(); Arg != Last; ++Arg) {
    TemplateArgs.push_back(
      ArgIsType[Arg]? TemplateArgument(TemplateArgLocs[Arg],
                                       //FIXME: Preserve type source info.
                                       Sema::GetTypeFromParser(Args[Arg]))
                    : TemplateArgument(reinterpret_cast<Expr *>(Args[Arg])));
  }
}

QualType Sema::CheckTemplateIdType(TemplateName Name,
                                   SourceLocation TemplateLoc,
                                   SourceLocation LAngleLoc,
                                   const TemplateArgument *TemplateArgs,
                                   unsigned NumTemplateArgs,
                                   SourceLocation RAngleLoc) {
  TemplateDecl *Template = Name.getAsTemplateDecl();
  if (!Template) {
    // The template name does not resolve to a template, so we just
    // build a dependent template-id type.
    return Context.getTemplateSpecializationType(Name, TemplateArgs,
                                                 NumTemplateArgs);
  }

  // Check that the template argument list is well-formed for this
  // template.
  TemplateArgumentListBuilder Converted(Template->getTemplateParameters(),
                                        NumTemplateArgs);
  if (CheckTemplateArgumentList(Template, TemplateLoc, LAngleLoc,
                                TemplateArgs, NumTemplateArgs, RAngleLoc,
                                false, Converted))
    return QualType();

  assert((Converted.structuredSize() ==
            Template->getTemplateParameters()->size()) &&
         "Converted template argument list is too short!");

  QualType CanonType;

  if (TemplateSpecializationType::anyDependentTemplateArguments(
                                                      TemplateArgs,
                                                      NumTemplateArgs)) {
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
    // it is a TemplateTypeSpecializationType that we will never use again.
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
  //FIXME: Preserve type source info.
  return Context.getTemplateSpecializationType(Name, TemplateArgs,
                                               NumTemplateArgs, CanonType);
}

Action::TypeResult
Sema::ActOnTemplateIdType(TemplateTy TemplateD, SourceLocation TemplateLoc,
                          SourceLocation LAngleLoc,
                          ASTTemplateArgsPtr TemplateArgsIn,
                          SourceLocation *TemplateArgLocs,
                          SourceLocation RAngleLoc) {
  TemplateName Template = TemplateD.getAsVal<TemplateName>();

  // Translate the parser's template argument list in our AST format.
  llvm::SmallVector<TemplateArgument, 16> TemplateArgs;
  translateTemplateArguments(TemplateArgsIn, TemplateArgLocs, TemplateArgs);

  QualType Result = CheckTemplateIdType(Template, TemplateLoc, LAngleLoc,
                                        TemplateArgs.data(),
                                        TemplateArgs.size(),
                                        RAngleLoc);
  TemplateArgsIn.release();

  if (Result.isNull())
    return true;

  return Result.getAsOpaquePtr();
}

Sema::TypeResult Sema::ActOnTagTemplateIdType(TypeResult TypeResult,
                                              TagUseKind TUK,
                                              DeclSpec::TST TagSpec,
                                              SourceLocation TagLoc) {
  if (TypeResult.isInvalid())
    return Sema::TypeResult();

  QualType Type = QualType::getFromOpaquePtr(TypeResult.get());

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

Sema::OwningExprResult Sema::BuildTemplateIdExpr(TemplateName Template,
                                                 SourceLocation TemplateNameLoc,
                                                 SourceLocation LAngleLoc,
                                           const TemplateArgument *TemplateArgs,
                                                 unsigned NumTemplateArgs,
                                                 SourceLocation RAngleLoc) {
  // FIXME: Can we do any checking at this point? I guess we could check the
  // template arguments that we have against the template name, if the template
  // name refers to a single template. That's not a terribly common case,
  // though.
  return Owned(TemplateIdRefExpr::Create(Context,
                                         /*FIXME: New type?*/Context.OverloadTy,
                                         /*FIXME: Necessary?*/0,
                                         /*FIXME: Necessary?*/SourceRange(),
                                         Template, TemplateNameLoc, LAngleLoc,
                                         TemplateArgs,
                                         NumTemplateArgs, RAngleLoc));
}

Sema::OwningExprResult Sema::ActOnTemplateIdExpr(TemplateTy TemplateD,
                                                 SourceLocation TemplateNameLoc,
                                                 SourceLocation LAngleLoc,
                                              ASTTemplateArgsPtr TemplateArgsIn,
                                                SourceLocation *TemplateArgLocs,
                                                 SourceLocation RAngleLoc) {
  TemplateName Template = TemplateD.getAsVal<TemplateName>();

  // Translate the parser's template argument list in our AST format.
  llvm::SmallVector<TemplateArgument, 16> TemplateArgs;
  translateTemplateArguments(TemplateArgsIn, TemplateArgLocs, TemplateArgs);
  TemplateArgsIn.release();

  return BuildTemplateIdExpr(Template, TemplateNameLoc, LAngleLoc,
                             TemplateArgs.data(), TemplateArgs.size(),
                             RAngleLoc);
}

Sema::OwningExprResult
Sema::ActOnMemberTemplateIdReferenceExpr(Scope *S, ExprArg Base,
                                         SourceLocation OpLoc,
                                         tok::TokenKind OpKind,
                                         const CXXScopeSpec &SS,
                                         TemplateTy TemplateD,
                                         SourceLocation TemplateNameLoc,
                                         SourceLocation LAngleLoc,
                                         ASTTemplateArgsPtr TemplateArgsIn,
                                         SourceLocation *TemplateArgLocs,
                                         SourceLocation RAngleLoc) {
  TemplateName Template = TemplateD.getAsVal<TemplateName>();

  // FIXME: We're going to end up looking up the template based on its name,
  // twice!
  DeclarationName Name;
  if (TemplateDecl *ActualTemplate = Template.getAsTemplateDecl())
    Name = ActualTemplate->getDeclName();
  else if (OverloadedFunctionDecl *Ovl = Template.getAsOverloadedFunctionDecl())
    Name = Ovl->getDeclName();
  else
    Name = Template.getAsDependentTemplateName()->getName();

  // Translate the parser's template argument list in our AST format.
  llvm::SmallVector<TemplateArgument, 16> TemplateArgs;
  translateTemplateArguments(TemplateArgsIn, TemplateArgLocs, TemplateArgs);
  TemplateArgsIn.release();

  // Do we have the save the actual template name? We might need it...
  return BuildMemberReferenceExpr(S, move(Base), OpLoc, OpKind, TemplateNameLoc,
                                  Name, true, LAngleLoc,
                                  TemplateArgs.data(), TemplateArgs.size(),
                                  RAngleLoc, DeclPtrTy(), &SS);
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
                                 const IdentifierInfo &Name,
                                 SourceLocation NameLoc,
                                 const CXXScopeSpec &SS,
                                 TypeTy *ObjectType) {
  if ((ObjectType &&
       computeDeclContext(QualType::getFromOpaquePtr(ObjectType))) ||
      (SS.isSet() && computeDeclContext(SS, false))) {
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
    TemplateNameKind TNK = isTemplateName(0, Name, NameLoc, &SS, ObjectType,
                                          false, Template);
    if (TNK == TNK_Non_template) {
      Diag(NameLoc, diag::err_template_kw_refers_to_non_template)
        << &Name;
      return TemplateTy();
    }

    return Template;
  }

  NestedNameSpecifier *Qualifier
    = static_cast<NestedNameSpecifier *>(SS.getScopeRep());
  return TemplateTy::make(Context.getDependentTemplateName(Qualifier, &Name));
}

bool Sema::CheckTemplateTypeArgument(TemplateTypeParmDecl *Param,
                                     const TemplateArgument &Arg,
                                     TemplateArgumentListBuilder &Converted) {
  // Check template type parameter.
  if (Arg.getKind() != TemplateArgument::Type) {
    // C++ [temp.arg.type]p1:
    //   A template-argument for a template-parameter which is a
    //   type shall be a type-id.

    // We have a template type parameter but the template argument
    // is not a type.
    Diag(Arg.getLocation(), diag::err_template_arg_must_be_type);
    Diag(Param->getLocation(), diag::note_template_param_here);

    return true;
  }

  if (CheckTemplateArgument(Param, Arg.getAsType(), Arg.getLocation()))
    return true;

  // Add the converted template type argument.
  Converted.Append(
                 TemplateArgument(Arg.getLocation(),
                                  Context.getCanonicalType(Arg.getAsType())));
  return false;
}

/// \brief Check that the given template argument list is well-formed
/// for specializing the given template.
bool Sema::CheckTemplateArgumentList(TemplateDecl *Template,
                                     SourceLocation TemplateLoc,
                                     SourceLocation LAngleLoc,
                                     const TemplateArgument *TemplateArgs,
                                     unsigned NumTemplateArgs,
                                     SourceLocation RAngleLoc,
                                     bool PartialTemplateArgs,
                                     TemplateArgumentListBuilder &Converted) {
  TemplateParameterList *Params = Template->getTemplateParameters();
  unsigned NumParams = Params->size();
  unsigned NumArgs = NumTemplateArgs;
  bool Invalid = false;

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

    // Decode the template argument
    TemplateArgument Arg;
    if (ArgIdx >= NumArgs) {
      // Retrieve the default template argument from the template
      // parameter.
      if (TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(*Param)) {
        if (TTP->isParameterPack()) {
          // We have an empty argument pack.
          Converted.BeginPack();
          Converted.EndPack();
          break;
        }

        if (!TTP->hasDefaultArgument())
          break;

        QualType ArgType = TTP->getDefaultArgument();

        // If the argument type is dependent, instantiate it now based
        // on the previously-computed template arguments.
        if (ArgType->isDependentType()) {
          InstantiatingTemplate Inst(*this, TemplateLoc,
                                     Template, Converted.getFlatArguments(),
                                     Converted.flatSize(),
                                     SourceRange(TemplateLoc, RAngleLoc));

          TemplateArgumentList TemplateArgs(Context, Converted,
                                            /*TakeArgs=*/false);
          ArgType = SubstType(ArgType,
                              MultiLevelTemplateArgumentList(TemplateArgs),
                              TTP->getDefaultArgumentLoc(),
                              TTP->getDeclName());
        }

        if (ArgType.isNull())
          return true;

        Arg = TemplateArgument(TTP->getLocation(), ArgType);
      } else if (NonTypeTemplateParmDecl *NTTP
                   = dyn_cast<NonTypeTemplateParmDecl>(*Param)) {
        if (!NTTP->hasDefaultArgument())
          break;

        InstantiatingTemplate Inst(*this, TemplateLoc,
                                   Template, Converted.getFlatArguments(),
                                   Converted.flatSize(),
                                   SourceRange(TemplateLoc, RAngleLoc));

        TemplateArgumentList TemplateArgs(Context, Converted,
                                          /*TakeArgs=*/false);

        Sema::OwningExprResult E
          = SubstExpr(NTTP->getDefaultArgument(),
                      MultiLevelTemplateArgumentList(TemplateArgs));
        if (E.isInvalid())
          return true;

        Arg = TemplateArgument(E.takeAs<Expr>());
      } else {
        TemplateTemplateParmDecl *TempParm
          = cast<TemplateTemplateParmDecl>(*Param);

        if (!TempParm->hasDefaultArgument())
          break;

        // FIXME: Subst default argument
        Arg = TemplateArgument(TempParm->getDefaultArgument());
      }
    } else {
      // Retrieve the template argument produced by the user.
      Arg = TemplateArgs[ArgIdx];
    }


    if (TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(*Param)) {
      if (TTP->isParameterPack()) {
        Converted.BeginPack();
        // Check all the remaining arguments (if any).
        for (; ArgIdx < NumArgs; ++ArgIdx) {
          if (CheckTemplateTypeArgument(TTP, TemplateArgs[ArgIdx], Converted))
            Invalid = true;
        }

        Converted.EndPack();
      } else {
        if (CheckTemplateTypeArgument(TTP, Arg, Converted))
          Invalid = true;
      }
    } else if (NonTypeTemplateParmDecl *NTTP
                 = dyn_cast<NonTypeTemplateParmDecl>(*Param)) {
      // Check non-type template parameters.

      // Do substitution on the type of the non-type template parameter
      // with the template arguments we've seen thus far.
      QualType NTTPType = NTTP->getType();
      if (NTTPType->isDependentType()) {
        // Do substitution on the type of the non-type template parameter.
        InstantiatingTemplate Inst(*this, TemplateLoc,
                                   Template, Converted.getFlatArguments(),
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
        if (NTTPType.isNull()) {
          Invalid = true;
          break;
        }
      }

      switch (Arg.getKind()) {
      case TemplateArgument::Null:
        assert(false && "Should never see a NULL template argument here");
        break;

      case TemplateArgument::Expression: {
        Expr *E = Arg.getAsExpr();
        TemplateArgument Result;
        if (CheckTemplateArgument(NTTP, NTTPType, E, Result))
          Invalid = true;
        else
          Converted.Append(Result);
        break;
      }

      case TemplateArgument::Declaration:
      case TemplateArgument::Integral:
        // We've already checked this template argument, so just copy
        // it to the list of converted arguments.
        Converted.Append(Arg);
        break;

      case TemplateArgument::Type:
        // We have a non-type template parameter but the template
        // argument is a type.

        // C++ [temp.arg]p2:
        //   In a template-argument, an ambiguity between a type-id and
        //   an expression is resolved to a type-id, regardless of the
        //   form of the corresponding template-parameter.
        //
        // We warn specifically about this case, since it can be rather
        // confusing for users.
        if (Arg.getAsType()->isFunctionType())
          Diag(Arg.getLocation(), diag::err_template_arg_nontype_ambig)
            << Arg.getAsType();
        else
          Diag(Arg.getLocation(), diag::err_template_arg_must_be_expr);
        Diag((*Param)->getLocation(), diag::note_template_param_here);
        Invalid = true;
        break;

      case TemplateArgument::Pack:
        assert(0 && "FIXME: Implement!");
        break;
      }
    } else {
      // Check template template parameters.
      TemplateTemplateParmDecl *TempParm
        = cast<TemplateTemplateParmDecl>(*Param);

      switch (Arg.getKind()) {
      case TemplateArgument::Null:
        assert(false && "Should never see a NULL template argument here");
        break;

      case TemplateArgument::Expression: {
        Expr *ArgExpr = Arg.getAsExpr();
        if (ArgExpr && isa<DeclRefExpr>(ArgExpr) &&
            isa<TemplateDecl>(cast<DeclRefExpr>(ArgExpr)->getDecl())) {
          if (CheckTemplateArgument(TempParm, cast<DeclRefExpr>(ArgExpr)))
            Invalid = true;

          // Add the converted template argument.
          Decl *D
            = cast<DeclRefExpr>(ArgExpr)->getDecl()->getCanonicalDecl();
          Converted.Append(TemplateArgument(Arg.getLocation(), D));
          continue;
        }
      }
        // fall through

      case TemplateArgument::Type: {
        // We have a template template parameter but the template
        // argument does not refer to a template.
        Diag(Arg.getLocation(), diag::err_template_arg_must_be_template);
        Invalid = true;
        break;
      }

      case TemplateArgument::Declaration:
        // We've already checked this template argument, so just copy
        // it to the list of converted arguments.
        Converted.Append(Arg);
        break;

      case TemplateArgument::Integral:
        assert(false && "Integral argument with template template parameter");
        break;

      case TemplateArgument::Pack:
        assert(0 && "FIXME: Implement!");
        break;
      }
    }
  }

  return Invalid;
}

/// \brief Check a template argument against its corresponding
/// template type parameter.
///
/// This routine implements the semantics of C++ [temp.arg.type]. It
/// returns true if an error occurred, and false otherwise.
bool Sema::CheckTemplateArgument(TemplateTypeParmDecl *Param,
                                 QualType Arg, SourceLocation ArgLoc) {
  // C++ [temp.arg.type]p2:
  //   A local type, a type with no linkage, an unnamed type or a type
  //   compounded from any of these types shall not be used as a
  //   template-argument for a template type-parameter.
  //
  // FIXME: Perform the recursive and no-linkage type checks.
  const TagType *Tag = 0;
  if (const EnumType *EnumT = Arg->getAsEnumType())
    Tag = EnumT;
  else if (const RecordType *RecordT = Arg->getAs<RecordType>())
    Tag = RecordT;
  if (Tag && Tag->getDecl()->getDeclContext()->isFunctionOrMethod())
    return Diag(ArgLoc, diag::err_template_arg_local_type)
      << QualType(Tag, 0);
  else if (Tag && !Tag->getDecl()->getDeclName() &&
           !Tag->getDecl()->getTypedefForAnonDecl()) {
    Diag(ArgLoc, diag::err_template_arg_unnamed_type);
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
  if (ImplicitCastExpr *Cast = dyn_cast<ImplicitCastExpr>(Arg))
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
    if (Func->getStorageClass() == FunctionDecl::Static) {
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
    if (!Var->hasGlobalStorage()) {
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
bool
Sema::CheckTemplateArgumentPointerToMember(Expr *Arg, NamedDecl *&Member) {
  bool Invalid = false;

  // See through any implicit casts we added to fix the type.
  if (ImplicitCastExpr *Cast = dyn_cast<ImplicitCastExpr>(Arg))
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
  QualifiedDeclRefExpr *DRE = 0;

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

  if (UnaryOperator *UnOp = dyn_cast<UnaryOperator>(Arg))
    if (UnOp->getOpcode() == UnaryOperator::AddrOf)
      DRE = dyn_cast<QualifiedDeclRefExpr>(UnOp->getSubExpr());

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
    Member = DRE->getDecl();
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
    if (ParamType == ArgType) {
      // Okay: no conversion necessary
    } else if (IsIntegralPromotion(Arg, ArgType, ParamType) ||
               !ParamType->isEnumeralType()) {
      // This is an integral promotion or conversion.
      ImpCastExprToType(Arg, ParamType);
    } else {
      // We can't perform this conversion.
      Diag(Arg->getSourceRange().getBegin(),
           diag::err_template_arg_not_convertible)
        << Arg->getType() << InstantiatedParamType << Arg->getSourceRange();
      Diag(Param->getLocation(), diag::note_template_param_here);
      return true;
    }

    QualType IntegerType = Context.getCanonicalType(ParamType);
    if (const EnumType *Enum = IntegerType->getAsEnumType())
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

    Converted = TemplateArgument(StartLoc, Value,
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
      ImpCastExprToType(Arg, ParamType);
    } else if (ArgType->isFunctionType() && ParamType->isPointerType()) {
      ArgType = Context.getPointerType(ArgType);
      ImpCastExprToType(Arg, ArgType);
    } else if (FunctionDecl *Fn
                 = ResolveAddressOfOverloadedFunction(Arg, ParamType, true)) {
      if (DiagnoseUseOfDecl(Fn, Arg->getSourceRange().getBegin()))
        return true;

      FixOverloadedFunctionReference(Arg, Fn);
      ArgType = Arg->getType();
      if (ArgType->isFunctionType() && ParamType->isPointerType()) {
        ArgType = Context.getPointerType(Arg->getType());
        ImpCastExprToType(Arg, ArgType);
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

    if (ParamType->isMemberPointerType()) {
      NamedDecl *Member = 0;
      if (CheckTemplateArgumentPointerToMember(Arg, Member))
        return true;

      if (Member)
        Member = cast<NamedDecl>(Member->getCanonicalDecl());
      Converted = TemplateArgument(StartLoc, Member);
      return false;
    }

    NamedDecl *Entity = 0;
    if (CheckTemplateArgumentAddressOfObjectOrFunction(Arg, Entity))
      return true;

    if (Entity)
      Entity = cast<NamedDecl>(Entity->getCanonicalDecl());
    Converted = TemplateArgument(StartLoc, Entity);
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
      ImpCastExprToType(Arg, ParamType);
    } else if (ArgType->isArrayType()) {
      ArgType = Context.getArrayDecayedType(ArgType);
      ImpCastExprToType(Arg, ArgType);
    }

    if (IsQualificationConversion(ArgType, ParamType)) {
      ArgType = ParamType;
      ImpCastExprToType(Arg, ParamType);
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
    Converted = TemplateArgument(StartLoc, Entity);
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
    Converted = TemplateArgument(StartLoc, Entity);
    return false;
  }

  //     -- For a non-type template-parameter of type pointer to data
  //        member, qualification conversions (4.4) are applied.
  // C++0x allows std::nullptr_t values.
  assert(ParamType->isMemberPointerType() && "Only pointers to members remain");

  if (Context.hasSameUnqualifiedType(ParamType, ArgType)) {
    // Types match exactly: nothing more to do here.
  } else if (ArgType->isNullPtrType()) {
    ImpCastExprToType(Arg, ParamType);
  } else if (IsQualificationConversion(ArgType, ParamType)) {
    ImpCastExprToType(Arg, ParamType);
  } else {
    // We can't perform this conversion.
    Diag(Arg->getSourceRange().getBegin(),
         diag::err_template_arg_not_convertible)
      << Arg->getType() << InstantiatedParamType << Arg->getSourceRange();
    Diag(Param->getLocation(), diag::note_template_param_here);
    return true;
  }

  NamedDecl *Member = 0;
  if (CheckTemplateArgumentPointerToMember(Arg, Member))
    return true;

  if (Member)
    Member = cast<NamedDecl>(Member->getCanonicalDecl());
  Converted = TemplateArgument(StartLoc, Member);
  return false;
}

/// \brief Check a template argument against its corresponding
/// template template parameter.
///
/// This routine implements the semantics of C++ [temp.arg.template].
/// It returns true if an error occurred, and false otherwise.
bool Sema::CheckTemplateArgument(TemplateTemplateParmDecl *Param,
                                 DeclRefExpr *Arg) {
  assert(isa<TemplateDecl>(Arg->getDecl()) && "Only template decls allowed");
  TemplateDecl *Template = cast<TemplateDecl>(Arg->getDecl());

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
    Diag(Arg->getLocStart(), diag::err_template_arg_not_class_template);
    Diag(Template->getLocation(), diag::note_template_arg_refers_here_func)
      << Template;
  }

  return !TemplateParameterListsAreEqual(Template->getTemplateParameters(),
                                         Param->getTemplateParameters(),
                                         true, true,
                                         Arg->getSourceRange().getBegin());
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
/// \param IsTemplateTemplateParm  If true, this routine is being
/// called to compare the template parameter lists of a template
/// template parameter.
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
                                     bool IsTemplateTemplateParm,
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
          << IsTemplateTemplateParm
          << SourceRange(New->getTemplateLoc(), New->getRAngleLoc());
      Diag(Old->getTemplateLoc(), diag::note_template_prev_declaration)
        << IsTemplateTemplateParm
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
        << IsTemplateTemplateParm;
        Diag((*OldParm)->getLocation(), diag::note_template_prev_declaration)
        << IsTemplateTemplateParm;
      }
      return false;
    }

    if (isa<TemplateTypeParmDecl>(*OldParm)) {
      // Okay; all template type parameters are equivalent (since we
      // know we're at the same index).
#if 0
      // FIXME: Enable this code in debug mode *after* we properly go through
      // and "instantiate" the template parameter lists of template template
      // parameters. It's only after this instantiation that (1) any dependent
      // types within the template parameter list of the template template
      // parameter can be checked, and (2) the template type parameter depths
      // will match up.
      QualType OldParmType
        = Context.getTypeDeclType(cast<TemplateTypeParmDecl>(*OldParm));
      QualType NewParmType
        = Context.getTypeDeclType(cast<TemplateTypeParmDecl>(*NewParm));
      assert(Context.getCanonicalType(OldParmType) ==
             Context.getCanonicalType(NewParmType) &&
             "type parameter mismatch?");
#endif
    } else if (NonTypeTemplateParmDecl *OldNTTP
                 = dyn_cast<NonTypeTemplateParmDecl>(*OldParm)) {
      // The types of non-type template parameters must agree.
      NonTypeTemplateParmDecl *NewNTTP
        = cast<NonTypeTemplateParmDecl>(*NewParm);
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
            << IsTemplateTemplateParm;
          Diag(OldNTTP->getLocation(),
               diag::note_template_nontype_parm_prev_declaration)
            << OldNTTP->getType();
        }
        return false;
      }
    } else {
      // The template parameter lists of template template
      // parameters must agree.
      // FIXME: Could we perform a faster "type" comparison here?
      assert(isa<TemplateTemplateParmDecl>(*OldParm) &&
             "Only template template parameters handled here");
      TemplateTemplateParmDecl *OldTTP
        = cast<TemplateTemplateParmDecl>(*OldParm);
      TemplateTemplateParmDecl *NewTTP
        = cast<TemplateTemplateParmDecl>(*NewParm);
      if (!TemplateParameterListsAreEqual(NewTTP->getTemplateParameters(),
                                          OldTTP->getTemplateParameters(),
                                          Complain,
                                          /*IsTemplateTemplateParm=*/true,
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

/// \brief Check whether a class template specialization or explicit
/// instantiation in the current context is well-formed.
///
/// This routine determines whether a class template specialization or
/// explicit instantiation can be declared in the current context
/// (C++ [temp.expl.spec]p2, C++0x [temp.explicit]p2) and emits
/// appropriate diagnostics if there was an error. It returns true if
// there was an error that we cannot recover from, and false otherwise.
bool
Sema::CheckClassTemplateSpecializationScope(ClassTemplateDecl *ClassTemplate,
                                   ClassTemplateSpecializationDecl *PrevDecl,
                                            SourceLocation TemplateNameLoc,
                                            SourceRange ScopeSpecifierRange,
                                            bool PartialSpecialization,
                                            bool ExplicitInstantiation) {
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
  if (CurContext->getLookupContext()->isFunctionOrMethod()) {
    int Kind = ExplicitInstantiation? 2 : PartialSpecialization? 1 : 0;
    Diag(TemplateNameLoc, diag::err_template_spec_decl_function_scope)
      << Kind << ClassTemplate;
    return true;
  }

  DeclContext *DC = CurContext->getEnclosingNamespaceContext();
  DeclContext *TemplateContext
    = ClassTemplate->getDeclContext()->getEnclosingNamespaceContext();
  if ((!PrevDecl || PrevDecl->getSpecializationKind() == TSK_Undeclared) &&
      !ExplicitInstantiation) {
    // There is no prior declaration of this entity, so this
    // specialization must be in the same context as the template
    // itself.
    if (DC != TemplateContext) {
      if (isa<TranslationUnitDecl>(TemplateContext))
        Diag(TemplateNameLoc, diag::err_template_spec_decl_out_of_scope_global)
          << PartialSpecialization
          << ClassTemplate << ScopeSpecifierRange;
      else if (isa<NamespaceDecl>(TemplateContext))
        Diag(TemplateNameLoc, diag::err_template_spec_decl_out_of_scope)
          << PartialSpecialization << ClassTemplate
          << cast<NamedDecl>(TemplateContext) << ScopeSpecifierRange;

      Diag(ClassTemplate->getLocation(), diag::note_template_decl_here);
    }

    return false;
  }

  // We have a previous declaration of this entity. Make sure that
  // this redeclaration (or definition) occurs in an enclosing namespace.
  if (!CurContext->Encloses(TemplateContext)) {
    // FIXME:  In C++98,  we  would like  to  turn these  errors into  warnings,
    // dependent on a -Wc++0x flag.
    bool SuppressedDiag = false;
    int Kind = ExplicitInstantiation? 2 : PartialSpecialization? 1 : 0;
    if (isa<TranslationUnitDecl>(TemplateContext)) {
      if (!ExplicitInstantiation || getLangOptions().CPlusPlus0x)
        Diag(TemplateNameLoc, diag::err_template_spec_redecl_global_scope)
          << Kind << ClassTemplate << ScopeSpecifierRange;
      else
        SuppressedDiag = true;
    } else if (isa<NamespaceDecl>(TemplateContext)) {
      if (!ExplicitInstantiation || getLangOptions().CPlusPlus0x)
        Diag(TemplateNameLoc, diag::err_template_spec_redecl_out_of_scope)
          << Kind << ClassTemplate
          << cast<NamedDecl>(TemplateContext) << ScopeSpecifierRange;
      else
        SuppressedDiag = true;
    }

    if (!SuppressedDiag)
      Diag(ClassTemplate->getLocation(), diag::note_template_decl_here);
  }

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
        // FIXME: We should settle on either Declaration storage or
        // Expression storage for template template parameters.
        TemplateTemplateParmDecl *ArgDecl
          = dyn_cast_or_null<TemplateTemplateParmDecl>(
                                                  ArgList[I].getAsDecl());
        if (!ArgDecl)
          if (DeclRefExpr *DRE
                = dyn_cast_or_null<DeclRefExpr>(ArgList[I].getAsExpr()))
            ArgDecl = dyn_cast<TemplateTemplateParmDecl>(DRE->getDecl());

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
                                       SourceLocation *TemplateArgLocs,
                                       SourceLocation RAngleLoc,
                                       AttributeList *Attr,
                               MultiTemplateParamsArg TemplateParameterLists) {
  assert(TUK == TUK_Declaration || TUK == TUK_Definition);

  // Find the class template we're specializing
  TemplateName Name = TemplateD.getAsVal<TemplateName>();
  ClassTemplateDecl *ClassTemplate
    = cast<ClassTemplateDecl>(Name.getAsTemplateDecl());

  bool isPartialSpecialization = false;

  // Check the validity of the template headers that introduce this
  // template.
  TemplateParameterList *TemplateParams
    = MatchTemplateParametersToScopeSpecifier(TemplateNameLoc, SS,
                        (TemplateParameterList**)TemplateParameterLists.get(),
                                              TemplateParameterLists.size());
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
          TTP->setDefaultArgument(QualType(), SourceLocation(), false);
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
        if (Expr *DefArg = TTP->getDefaultArgument()) {
          Diag(TTP->getDefaultArgumentLoc(),
               diag::err_default_arg_in_partial_spec)
            << DefArg->getSourceRange();
          TTP->setDefaultArgument(0);
          DefArg->Destroy(Context);
        }
      }
    }
  } else if (!TemplateParams)
    Diag(KWLoc, diag::err_template_spec_needs_header)
      << CodeModificationHint::CreateInsertion(KWLoc, "template<> ");

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
  llvm::SmallVector<TemplateArgument, 16> TemplateArgs;
  translateTemplateArguments(TemplateArgsIn, TemplateArgLocs, TemplateArgs);

  // Check that the template argument list is well-formed for this
  // template.
  TemplateArgumentListBuilder Converted(ClassTemplate->getTemplateParameters(),
                                        TemplateArgs.size());
  if (CheckTemplateArgumentList(ClassTemplate, TemplateNameLoc, LAngleLoc,
                                TemplateArgs.data(), TemplateArgs.size(),
                                RAngleLoc, false, Converted))
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
  if (CheckClassTemplateSpecializationScope(ClassTemplate, PrevDecl,
                                            TemplateNameLoc,
                                            SS.getRange(),
                                            isPartialSpecialization,
                                            /*ExplicitInstantiation=*/false))
    return true;

  // The canonical type
  QualType CanonType;
  if (PrevDecl && PrevDecl->getSpecializationKind() == TSK_Undeclared) {
    // Since the only prior class template specialization with these
    // arguments was referenced but not declared, reuse that
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
    TemplateParameterList *TemplateParams
      = static_cast<TemplateParameterList*>(*TemplateParameterLists.get());
    ClassTemplatePartialSpecializationDecl *PrevPartial
      = cast_or_null<ClassTemplatePartialSpecializationDecl>(PrevDecl);
    ClassTemplatePartialSpecializationDecl *Partial
      = ClassTemplatePartialSpecializationDecl::Create(Context,
                                             ClassTemplate->getDeclContext(),
                                                       TemplateNameLoc,
                                                       TemplateParams,
                                                       ClassTemplate,
                                                       Converted,
                                                       PrevPartial);

    if (PrevPartial) {
      ClassTemplate->getPartialSpecializations().RemoveNode(PrevPartial);
      ClassTemplate->getPartialSpecializations().GetOrInsertNode(Partial);
    } else {
      ClassTemplate->getPartialSpecializations().InsertNode(Partial, InsertPos);
    }
    Specialization = Partial;

    // Check that all of the template parameters of the class template
    // partial specialization are deducible from the template
    // arguments. If not, this class template partial specialization
    // will never be used.
    llvm::SmallVector<bool, 8> DeducibleParams;
    DeducibleParams.resize(TemplateParams->size());
    MarkUsedTemplateParameters(Partial->getTemplateArgs(), true, 
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
    // this explicit specialization.
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

  // Note that this is an explicit specialization.
  Specialization->setSpecializationKind(TSK_ExplicitSpecialization);

  // Check that this isn't a redefinition of this specialization.
  if (TUK == TUK_Definition) {
    if (RecordDecl *Def = Specialization->getDefinition(Context)) {
      // FIXME: Should also handle explicit specialization after implicit
      // instantiation with a special diagnostic.
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
    = Context.getTemplateSpecializationType(Name,
                                            TemplateArgs.data(),
                                            TemplateArgs.size(),
                                            CanonType);
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

  // Add the specialization into its lexical context, so that it can
  // be seen when iterating through the list of declarations in that
  // context. However, specializations are not found by name lookup.
  CurContext->addDecl(Specialization);
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
                                 SourceLocation *TemplateArgLocs,
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
  //   [...] An explicit instantiation shall appear in an enclosing
  //   namespace of its template. [...]
  //
  // This is C++ DR 275.
  if (CheckClassTemplateSpecializationScope(ClassTemplate, 0,
                                            TemplateNameLoc,
                                            SS.getRange(),
                                            /*PartialSpecialization=*/false,
                                            /*ExplicitInstantiation=*/true))
    return true;

  // Translate the parser's template argument list in our AST format.
  llvm::SmallVector<TemplateArgument, 16> TemplateArgs;
  translateTemplateArguments(TemplateArgsIn, TemplateArgLocs, TemplateArgs);

  // Check that the template argument list is well-formed for this
  // template.
  TemplateArgumentListBuilder Converted(ClassTemplate->getTemplateParameters(),
                                        TemplateArgs.size());
  if (CheckTemplateArgumentList(ClassTemplate, TemplateNameLoc, LAngleLoc,
                                TemplateArgs.data(), TemplateArgs.size(),
                                RAngleLoc, false, Converted))
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

  ClassTemplateSpecializationDecl *Specialization = 0;

  bool SpecializationRequiresInstantiation = true;
  if (PrevDecl) {
    if (PrevDecl->getSpecializationKind()
          == TSK_ExplicitInstantiationDefinition) {
      // This particular specialization has already been declared or
      // instantiated. We cannot explicitly instantiate it.
      Diag(TemplateNameLoc, diag::err_explicit_instantiation_duplicate)
        << Context.getTypeDeclType(PrevDecl);
      Diag(PrevDecl->getLocation(),
           diag::note_previous_explicit_instantiation);
      return DeclPtrTy::make(PrevDecl);
    }

    if (PrevDecl->getSpecializationKind() == TSK_ExplicitSpecialization) {
      // C++ DR 259, C++0x [temp.explicit]p4:
      //   For a given set of template parameters, if an explicit
      //   instantiation of a template appears after a declaration of
      //   an explicit specialization for that template, the explicit
      //   instantiation has no effect.
      if (!getLangOptions().CPlusPlus0x) {
        Diag(TemplateNameLoc,
             diag::ext_explicit_instantiation_after_specialization)
          << Context.getTypeDeclType(PrevDecl);
        Diag(PrevDecl->getLocation(),
             diag::note_previous_template_specialization);
      }

      // Create a new class template specialization declaration node
      // for this explicit specialization. This node is only used to
      // record the existence of this explicit instantiation for
      // accurate reproduction of the source code; we don't actually
      // use it for anything, since it is semantically irrelevant.
      Specialization
        = ClassTemplateSpecializationDecl::Create(Context,
                                             ClassTemplate->getDeclContext(),
                                                  TemplateNameLoc,
                                                  ClassTemplate,
                                                  Converted, 0);
      Specialization->setLexicalDeclContext(CurContext);
      CurContext->addDecl(Specialization);
      return DeclPtrTy::make(PrevDecl);
    }

    // If we have already (implicitly) instantiated this
    // specialization, there is less work to do.
    if (PrevDecl->getSpecializationKind() == TSK_ImplicitInstantiation)
      SpecializationRequiresInstantiation = false;

    if (PrevDecl->getSpecializationKind() == TSK_ImplicitInstantiation ||
        PrevDecl->getSpecializationKind() == TSK_Undeclared) {
      // Since the only prior class template specialization with these
      // arguments was referenced but not declared, reuse that
      // declaration node as our own, updating its source location to
      // reflect our new declaration.
      Specialization = PrevDecl;
      Specialization->setLocation(TemplateNameLoc);
      PrevDecl = 0;
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
    = Context.getTemplateSpecializationType(Name,
                                            TemplateArgs.data(),
                                            TemplateArgs.size(),
                                  Context.getTypeDeclType(Specialization));
  Specialization->setTypeAsWritten(WrittenTy);
  TemplateArgsIn.release();

  // Add the explicit instantiation into its lexical context. However,
  // since explicit instantiations are never found by name lookup, we
  // just put it into the declaration context directly.
  Specialization->setLexicalDeclContext(CurContext);
  CurContext->addDecl(Specialization);

  Specialization->setPointOfInstantiation(TemplateNameLoc);

  // C++ [temp.explicit]p3:
  //   A definition of a class template or class member template
  //   shall be in scope at the point of the explicit instantiation of
  //   the class template or class member template.
  //
  // This check comes when we actually try to perform the
  // instantiation.
  TemplateSpecializationKind TSK
    = ExternLoc.isInvalid()? TSK_ExplicitInstantiationDefinition
                           : TSK_ExplicitInstantiationDeclaration;
  if (SpecializationRequiresInstantiation)
    InstantiateClassTemplateSpecialization(Specialization, TSK);
  else // Instantiate the members of this class template specialization.
    InstantiateClassTemplateSpecializationMembers(TemplateLoc, Specialization,
                                                  TSK);

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
  //   [...] An explicit instantiation shall appear in an enclosing
  //   namespace of its template. [...]
  //
  // This is C++ DR 275.
  if (getLangOptions().CPlusPlus0x) {
    // FIXME: In C++98, we would like to turn these errors into warnings,
    // dependent on a -Wc++0x flag.
    DeclContext *PatternContext
      = Pattern->getDeclContext()->getEnclosingNamespaceContext();
    if (!CurContext->Encloses(PatternContext)) {
      Diag(TemplateLoc, diag::err_explicit_instantiation_out_of_scope)
        << Record << cast<NamedDecl>(PatternContext) << SS.getRange();
      Diag(Pattern->getLocation(), diag::note_previous_declaration);
    }
  }

  TemplateSpecializationKind TSK
    = ExternLoc.isInvalid()? TSK_ExplicitInstantiationDefinition
                           : TSK_ExplicitInstantiationDeclaration;

  if (!Record->getDefinition(Context)) {
    // If the class has a definition, instantiate it (and all of its
    // members, recursively).
    Pattern = cast_or_null<CXXRecordDecl>(Pattern->getDefinition(Context));
    if (Pattern && InstantiateClass(TemplateLoc, Record, Pattern,
                                    getTemplateInstantiationArgs(Record),
                                    TSK))
      return true;
  } else // Instantiate all of the members of the class.
    InstantiateClassMembers(TemplateLoc, Record,
                            getTemplateInstantiationArgs(Record), TSK);

  // FIXME: We don't have any representation for explicit instantiations of
  // member classes. Such a representation is not needed for compilation, but it
  // should be available for clients that want to see all of the declarations in
  // the source code.
  return TagD;
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
    = T->getAsTemplateSpecializationType();
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
  LookupResult Result = LookupQualifiedName(Ctx, Name, LookupOrdinaryName,
                                            false);
  unsigned DiagID = 0;
  Decl *Referenced = 0;
  switch (Result.getKind()) {
  case LookupResult::NotFound:
    if (Ctx->isTranslationUnit())
      DiagID = diag::err_typename_nested_not_found_global;
    else
      DiagID = diag::err_typename_nested_not_found;
    break;

  case LookupResult::Found:
    if (TypeDecl *Type = dyn_cast<TypeDecl>(Result.getAsDecl())) {
      // We found a type. Build a QualifiedNameType, since the
      // typename-specifier was just sugar. FIXME: Tell
      // QualifiedNameType that it has a "typename" prefix.
      return Context.getQualifiedNameType(NNS, Context.getTypeDeclType(Type));
    }

    DiagID = diag::err_typename_nested_not_type;
    Referenced = Result.getAsDecl();
    break;

  case LookupResult::FoundOverloaded:
    DiagID = diag::err_typename_nested_not_type;
    Referenced = *Result.begin();
    break;

  case LookupResult::AmbiguousBaseSubobjectTypes:
  case LookupResult::AmbiguousBaseSubobjects:
  case LookupResult::AmbiguousReference:
    DiagnoseAmbiguousLookup(Result, Name, Range.getEnd(), Range);
    return QualType();
  }

  // If we get here, it's because name lookup did not find a
  // type. Emit an appropriate diagnostic and return an error.
  if (NamedDecl *NamedCtx = dyn_cast<NamedDecl>(Ctx))
    Diag(Range.getEnd(), DiagID) << Range << Name << NamedCtx;
  else
    Diag(Range.getEnd(), DiagID) << Range << Name;
  if (Referenced)
    Diag(Referenced->getLocation(), diag::note_typename_refers_here)
      << Name;
  return QualType();
}

namespace {
  // See Sema::RebuildTypeInCurrentInstantiation
  class VISIBILITY_HIDDEN CurrentInstantiationRebuilder
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
    QualType TransformTypenameType(const TypenameType *T);
  };
}

QualType
CurrentInstantiationRebuilder::TransformTypenameType(const TypenameType *T) {
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
  if (NNS == T->getQualifier() && getSema().computeDeclContext(SS) == 0)
    return QualType(T, 0);

  // Rebuild the typename type, which will probably turn into a
  // QualifiedNameType.
  if (const TemplateSpecializationType *TemplateId = T->getTemplateId()) {
    QualType NewTemplateId
      = TransformType(QualType(TemplateId, 0));
    if (NewTemplateId.isNull())
      return QualType();

    if (NNS == T->getQualifier() &&
        NewTemplateId == QualType(TemplateId, 0))
      return QualType(T, 0);

    return getDerived().RebuildTypenameType(NNS, NewTemplateId);
  }

  return getDerived().RebuildTypenameType(NNS, T->getIdentifier());
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
