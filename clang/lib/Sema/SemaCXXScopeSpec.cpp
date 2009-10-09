//===--- SemaCXXScopeSpec.cpp - Semantic Analysis for C++ scope specifiers-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements C++ semantic analysis for scope specifiers.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Parse/DeclSpec.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

/// \brief Compute the DeclContext that is associated with the given type.
///
/// \param T the type for which we are attempting to find a DeclContext.
///
/// \returns the declaration context represented by the type T,
/// or NULL if the declaration context cannot be computed (e.g., because it is
/// dependent and not the current instantiation).
DeclContext *Sema::computeDeclContext(QualType T) {
  if (const TagType *Tag = T->getAs<TagType>())
    return Tag->getDecl();

  return 0;
}

/// \brief Compute the DeclContext that is associated with the given
/// scope specifier.
///
/// \param SS the C++ scope specifier as it appears in the source
///
/// \param EnteringContext when true, we will be entering the context of
/// this scope specifier, so we can retrieve the declaration context of a
/// class template or class template partial specialization even if it is
/// not the current instantiation.
///
/// \returns the declaration context represented by the scope specifier @p SS,
/// or NULL if the declaration context cannot be computed (e.g., because it is
/// dependent and not the current instantiation).
DeclContext *Sema::computeDeclContext(const CXXScopeSpec &SS,
                                      bool EnteringContext) {
  if (!SS.isSet() || SS.isInvalid())
    return 0;

  NestedNameSpecifier *NNS
    = static_cast<NestedNameSpecifier *>(SS.getScopeRep());
  if (NNS->isDependent()) {
    // If this nested-name-specifier refers to the current
    // instantiation, return its DeclContext.
    if (CXXRecordDecl *Record = getCurrentInstantiationOf(NNS))
      return Record;

    if (EnteringContext) {
      if (const TemplateSpecializationType *SpecType
            = dyn_cast_or_null<TemplateSpecializationType>(NNS->getAsType())) {
        // We are entering the context of the nested name specifier, so try to
        // match the nested name specifier to either a primary class template
        // or a class template partial specialization.
        if (ClassTemplateDecl *ClassTemplate
              = dyn_cast_or_null<ClassTemplateDecl>(
                            SpecType->getTemplateName().getAsTemplateDecl())) {
          QualType ContextType
            = Context.getCanonicalType(QualType(SpecType, 0));

          // If the type of the nested name specifier is the same as the
          // injected class name of the named class template, we're entering
          // into that class template definition.
          QualType Injected = ClassTemplate->getInjectedClassNameType(Context);
          if (Context.hasSameType(Injected, ContextType))
            return ClassTemplate->getTemplatedDecl();

          // If the type of the nested name specifier is the same as the
          // type of one of the class template's class template partial
          // specializations, we're entering into the definition of that
          // class template partial specialization.
          if (ClassTemplatePartialSpecializationDecl *PartialSpec
                = ClassTemplate->findPartialSpecialization(ContextType))
            return PartialSpec;
        }
      } else if (const RecordType *RecordT
                   = dyn_cast_or_null<RecordType>(NNS->getAsType())) {
        // The nested name specifier refers to a member of a class template.
        return RecordT->getDecl();
      }
    }

    return 0;
  }

  switch (NNS->getKind()) {
  case NestedNameSpecifier::Identifier:
    assert(false && "Dependent nested-name-specifier has no DeclContext");
    break;

  case NestedNameSpecifier::Namespace:
    return NNS->getAsNamespace();

  case NestedNameSpecifier::TypeSpec:
  case NestedNameSpecifier::TypeSpecWithTemplate: {
    const TagType *Tag = NNS->getAsType()->getAs<TagType>();
    assert(Tag && "Non-tag type in nested-name-specifier");
    return Tag->getDecl();
  } break;

  case NestedNameSpecifier::Global:
    return Context.getTranslationUnitDecl();
  }

  // Required to silence a GCC warning.
  return 0;
}

bool Sema::isDependentScopeSpecifier(const CXXScopeSpec &SS) {
  if (!SS.isSet() || SS.isInvalid())
    return false;

  NestedNameSpecifier *NNS
    = static_cast<NestedNameSpecifier *>(SS.getScopeRep());
  return NNS->isDependent();
}

// \brief Determine whether this C++ scope specifier refers to an
// unknown specialization, i.e., a dependent type that is not the
// current instantiation.
bool Sema::isUnknownSpecialization(const CXXScopeSpec &SS) {
  if (!isDependentScopeSpecifier(SS))
    return false;

  NestedNameSpecifier *NNS
    = static_cast<NestedNameSpecifier *>(SS.getScopeRep());
  return getCurrentInstantiationOf(NNS) == 0;
}

/// \brief If the given nested name specifier refers to the current
/// instantiation, return the declaration that corresponds to that
/// current instantiation (C++0x [temp.dep.type]p1).
///
/// \param NNS a dependent nested name specifier.
CXXRecordDecl *Sema::getCurrentInstantiationOf(NestedNameSpecifier *NNS) {
  assert(getLangOptions().CPlusPlus && "Only callable in C++");
  assert(NNS->isDependent() && "Only dependent nested-name-specifier allowed");

  if (!NNS->getAsType())
    return 0;

  QualType T = QualType(NNS->getAsType(), 0);
  // If the nested name specifier does not refer to a type, then it
  // does not refer to the current instantiation.
  if (T.isNull())
    return 0;

  T = Context.getCanonicalType(T);

  for (DeclContext *Ctx = CurContext; Ctx; Ctx = Ctx->getParent()) {
    // If we've hit a namespace or the global scope, then the
    // nested-name-specifier can't refer to the current instantiation.
    if (Ctx->isFileContext())
      return 0;

    // Skip non-class contexts.
    CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(Ctx);
    if (!Record)
      continue;

    // If this record type is not dependent,
    if (!Record->isDependentType())
      return 0;

    // C++ [temp.dep.type]p1:
    //
    //   In the definition of a class template, a nested class of a
    //   class template, a member of a class template, or a member of a
    //   nested class of a class template, a name refers to the current
    //   instantiation if it is
    //     -- the injected-class-name (9) of the class template or
    //        nested class,
    //     -- in the definition of a primary class template, the name
    //        of the class template followed by the template argument
    //        list of the primary template (as described below)
    //        enclosed in <>,
    //     -- in the definition of a nested class of a class template,
    //        the name of the nested class referenced as a member of
    //        the current instantiation, or
    //     -- in the definition of a partial specialization, the name
    //        of the class template followed by the template argument
    //        list of the partial specialization enclosed in <>. If
    //        the nth template parameter is a parameter pack, the nth
    //        template argument is a pack expansion (14.6.3) whose
    //        pattern is the name of the parameter pack.
    //        (FIXME: parameter packs)
    //
    // All of these options come down to having the
    // nested-name-specifier type that is equivalent to the
    // injected-class-name of one of the types that is currently in
    // our context.
    if (Context.getCanonicalType(Context.getTypeDeclType(Record)) == T)
      return Record;

    if (ClassTemplateDecl *Template = Record->getDescribedClassTemplate()) {
      QualType InjectedClassName
        = Template->getInjectedClassNameType(Context);
      if (T == Context.getCanonicalType(InjectedClassName))
        return Template->getTemplatedDecl();
    }
    // FIXME: check for class template partial specializations
  }

  return 0;
}

/// \brief Require that the context specified by SS be complete.
///
/// If SS refers to a type, this routine checks whether the type is
/// complete enough (or can be made complete enough) for name lookup
/// into the DeclContext. A type that is not yet completed can be
/// considered "complete enough" if it is a class/struct/union/enum
/// that is currently being defined. Or, if we have a type that names
/// a class template specialization that is not a complete type, we
/// will attempt to instantiate that class template.
bool Sema::RequireCompleteDeclContext(const CXXScopeSpec &SS) {
  if (!SS.isSet() || SS.isInvalid())
    return false;

  DeclContext *DC = computeDeclContext(SS, true);
  if (TagDecl *Tag = dyn_cast<TagDecl>(DC)) {
    // If we're currently defining this type, then lookup into the
    // type is okay: don't complain that it isn't complete yet.
    const TagType *TagT = Context.getTypeDeclType(Tag)->getAs<TagType>();
    if (TagT->isBeingDefined())
      return false;

    // The type must be complete.
    return RequireCompleteType(SS.getRange().getBegin(),
                               Context.getTypeDeclType(Tag),
                               PDiag(diag::err_incomplete_nested_name_spec)
                                 << SS.getRange());
  }

  return false;
}

/// ActOnCXXGlobalScopeSpecifier - Return the object that represents the
/// global scope ('::').
Sema::CXXScopeTy *Sema::ActOnCXXGlobalScopeSpecifier(Scope *S,
                                                     SourceLocation CCLoc) {
  return NestedNameSpecifier::GlobalSpecifier(Context);
}

/// \brief Determines whether the given declaration is an valid acceptable
/// result for name lookup of a nested-name-specifier.
bool Sema::isAcceptableNestedNameSpecifier(NamedDecl *SD) {
  if (!SD)
    return false;

  // Namespace and namespace aliases are fine.
  if (isa<NamespaceDecl>(SD) || isa<NamespaceAliasDecl>(SD))
    return true;

  if (!isa<TypeDecl>(SD))
    return false;

  // Determine whether we have a class (or, in C++0x, an enum) or
  // a typedef thereof. If so, build the nested-name-specifier.
  QualType T = Context.getTypeDeclType(cast<TypeDecl>(SD));
  if (T->isDependentType())
    return true;
  else if (TypedefDecl *TD = dyn_cast<TypedefDecl>(SD)) {
    if (TD->getUnderlyingType()->isRecordType() ||
        (Context.getLangOptions().CPlusPlus0x &&
         TD->getUnderlyingType()->isEnumeralType()))
      return true;
  } else if (isa<RecordDecl>(SD) ||
             (Context.getLangOptions().CPlusPlus0x && isa<EnumDecl>(SD)))
    return true;

  return false;
}

/// \brief If the given nested-name-specifier begins with a bare identifier
/// (e.g., Base::), perform name lookup for that identifier as a
/// nested-name-specifier within the given scope, and return the result of that
/// name lookup.
NamedDecl *Sema::FindFirstQualifierInScope(Scope *S, NestedNameSpecifier *NNS) {
  if (!S || !NNS)
    return 0;

  while (NNS->getPrefix())
    NNS = NNS->getPrefix();

  if (NNS->getKind() != NestedNameSpecifier::Identifier)
    return 0;

  LookupResult Found;
  LookupName(Found, S, NNS->getAsIdentifier(), LookupNestedNameSpecifierName);
  assert(!Found.isAmbiguous() && "Cannot handle ambiguities here yet");

  NamedDecl *Result = Found.getAsSingleDecl(Context);
  if (isAcceptableNestedNameSpecifier(Result))
    return Result;

  return 0;
}

/// \brief Build a new nested-name-specifier for "identifier::", as described
/// by ActOnCXXNestedNameSpecifier.
///
/// This routine differs only slightly from ActOnCXXNestedNameSpecifier, in
/// that it contains an extra parameter \p ScopeLookupResult, which provides
/// the result of name lookup within the scope of the nested-name-specifier
/// that was computed at template definitino time.
Sema::CXXScopeTy *Sema::BuildCXXNestedNameSpecifier(Scope *S,
                                                    const CXXScopeSpec &SS,
                                                    SourceLocation IdLoc,
                                                    SourceLocation CCLoc,
                                                    IdentifierInfo &II,
                                                    QualType ObjectType,
                                                  NamedDecl *ScopeLookupResult,
                                                    bool EnteringContext) {
  NestedNameSpecifier *Prefix
    = static_cast<NestedNameSpecifier *>(SS.getScopeRep());

  // Determine where to perform name lookup
  DeclContext *LookupCtx = 0;
  bool isDependent = false;
  if (!ObjectType.isNull()) {
    // This nested-name-specifier occurs in a member access expression, e.g.,
    // x->B::f, and we are looking into the type of the object.
    assert(!SS.isSet() && "ObjectType and scope specifier cannot coexist");
    LookupCtx = computeDeclContext(ObjectType);
    isDependent = ObjectType->isDependentType();
  } else if (SS.isSet()) {
    // This nested-name-specifier occurs after another nested-name-specifier,
    // so long into the context associated with the prior nested-name-specifier.
    LookupCtx = computeDeclContext(SS, EnteringContext);
    isDependent = isDependentScopeSpecifier(SS);
  }

  LookupResult Found;
  bool ObjectTypeSearchedInScope = false;
  if (LookupCtx) {
    // Perform "qualified" name lookup into the declaration context we
    // computed, which is either the type of the base of a member access
    // expression or the declaration context associated with a prior
    // nested-name-specifier.

    // The declaration context must be complete.
    if (!LookupCtx->isDependentContext() && RequireCompleteDeclContext(SS))
      return 0;

    LookupQualifiedName(Found, LookupCtx, &II, LookupNestedNameSpecifierName,
                        false);

    if (!ObjectType.isNull() && Found.getKind() == LookupResult::NotFound) {
      // C++ [basic.lookup.classref]p4:
      //   If the id-expression in a class member access is a qualified-id of
      //   the form
      //
      //        class-name-or-namespace-name::...
      //
      //   the class-name-or-namespace-name following the . or -> operator is
      //   looked up both in the context of the entire postfix-expression and in
      //   the scope of the class of the object expression. If the name is found
      //   only in the scope of the class of the object expression, the name
      //   shall refer to a class-name. If the name is found only in the
      //   context of the entire postfix-expression, the name shall refer to a
      //   class-name or namespace-name. [...]
      //
      // Qualified name lookup into a class will not find a namespace-name,
      // so we do not need to diagnoste that case specifically. However,
      // this qualified name lookup may find nothing. In that case, perform
      // unqualified name lookup in the given scope (if available) or
      // reconstruct the result from when name lookup was performed at template
      // definition time.
      if (S)
        LookupName(Found, S, &II, LookupNestedNameSpecifierName);
      else if (ScopeLookupResult)
        Found.addDecl(ScopeLookupResult);

      ObjectTypeSearchedInScope = true;
    }
  } else if (isDependent) {
    // We were not able to compute the declaration context for a dependent
    // base object type or prior nested-name-specifier, so this
    // nested-name-specifier refers to an unknown specialization. Just build
    // a dependent nested-name-specifier.
    if (!Prefix)
      return NestedNameSpecifier::Create(Context, &II);

    return NestedNameSpecifier::Create(Context, Prefix, &II);
  } else {
    // Perform unqualified name lookup in the current scope.
    LookupName(Found, S, &II, LookupNestedNameSpecifierName);
  }

  // FIXME: Deal with ambiguities cleanly.
  NamedDecl *SD = Found.getAsSingleDecl(Context);
  if (isAcceptableNestedNameSpecifier(SD)) {
    if (!ObjectType.isNull() && !ObjectTypeSearchedInScope) {
      // C++ [basic.lookup.classref]p4:
      //   [...] If the name is found in both contexts, the
      //   class-name-or-namespace-name shall refer to the same entity.
      //
      // We already found the name in the scope of the object. Now, look
      // into the current scope (the scope of the postfix-expression) to
      // see if we can find the same name there. As above, if there is no
      // scope, reconstruct the result from the template instantiation itself.
      NamedDecl *OuterDecl;
      if (S) {
        LookupResult FoundOuter;
        LookupName(FoundOuter, S, &II, LookupNestedNameSpecifierName);
        // FIXME: Handle ambiguities!
        OuterDecl = FoundOuter.getAsSingleDecl(Context);
      } else
        OuterDecl = ScopeLookupResult;

      if (isAcceptableNestedNameSpecifier(OuterDecl) &&
          OuterDecl->getCanonicalDecl() != SD->getCanonicalDecl() &&
          (!isa<TypeDecl>(OuterDecl) || !isa<TypeDecl>(SD) ||
           !Context.hasSameType(
                            Context.getTypeDeclType(cast<TypeDecl>(OuterDecl)),
                               Context.getTypeDeclType(cast<TypeDecl>(SD))))) {
             Diag(IdLoc, diag::err_nested_name_member_ref_lookup_ambiguous)
               << &II;
             Diag(SD->getLocation(), diag::note_ambig_member_ref_object_type)
               << ObjectType;
             Diag(OuterDecl->getLocation(), diag::note_ambig_member_ref_scope);

             // Fall through so that we'll pick the name we found in the object type,
             // since that's probably what the user wanted anyway.
           }
    }

    if (NamespaceDecl *Namespace = dyn_cast<NamespaceDecl>(SD))
      return NestedNameSpecifier::Create(Context, Prefix, Namespace);

    // FIXME: It would be nice to maintain the namespace alias name, then
    // see through that alias when resolving the nested-name-specifier down to
    // a declaration context.
    if (NamespaceAliasDecl *Alias = dyn_cast<NamespaceAliasDecl>(SD))
      return NestedNameSpecifier::Create(Context, Prefix,

                                         Alias->getNamespace());

    QualType T = Context.getTypeDeclType(cast<TypeDecl>(SD));
    return NestedNameSpecifier::Create(Context, Prefix, false,
                                       T.getTypePtr());
  }

  // If we didn't find anything during our lookup, try again with
  // ordinary name lookup, which can help us produce better error
  // messages.
  if (!SD) {
    Found.clear();
    LookupName(Found, S, &II, LookupOrdinaryName);
    SD = Found.getAsSingleDecl(Context);
  }

  unsigned DiagID;
  if (SD)
    DiagID = diag::err_expected_class_or_namespace;
  else if (SS.isSet()) {
    DiagnoseMissingMember(IdLoc, DeclarationName(&II),
                          (NestedNameSpecifier *)SS.getScopeRep(),
                          SS.getRange());
    return 0;
  } else
    DiagID = diag::err_undeclared_var_use;

  if (SS.isSet())
    Diag(IdLoc, DiagID) << &II << SS.getRange();
  else
    Diag(IdLoc, DiagID) << &II;

  return 0;
}

/// ActOnCXXNestedNameSpecifier - Called during parsing of a
/// nested-name-specifier. e.g. for "foo::bar::" we parsed "foo::" and now
/// we want to resolve "bar::". 'SS' is empty or the previously parsed
/// nested-name part ("foo::"), 'IdLoc' is the source location of 'bar',
/// 'CCLoc' is the location of '::' and 'II' is the identifier for 'bar'.
/// Returns a CXXScopeTy* object representing the C++ scope.
Sema::CXXScopeTy *Sema::ActOnCXXNestedNameSpecifier(Scope *S,
                                                    const CXXScopeSpec &SS,
                                                    SourceLocation IdLoc,
                                                    SourceLocation CCLoc,
                                                    IdentifierInfo &II,
                                                    TypeTy *ObjectTypePtr,
                                                    bool EnteringContext) {
  return BuildCXXNestedNameSpecifier(S, SS, IdLoc, CCLoc, II,
                                     QualType::getFromOpaquePtr(ObjectTypePtr),
                                     /*ScopeLookupResult=*/0, EnteringContext);
}

Sema::CXXScopeTy *Sema::ActOnCXXNestedNameSpecifier(Scope *S,
                                                    const CXXScopeSpec &SS,
                                                    TypeTy *Ty,
                                                    SourceRange TypeRange,
                                                    SourceLocation CCLoc) {
  NestedNameSpecifier *Prefix
    = static_cast<NestedNameSpecifier *>(SS.getScopeRep());
  QualType T = GetTypeFromParser(Ty);
  return NestedNameSpecifier::Create(Context, Prefix, /*FIXME:*/false,
                                     T.getTypePtr());
}

/// ActOnCXXEnterDeclaratorScope - Called when a C++ scope specifier (global
/// scope or nested-name-specifier) is parsed, part of a declarator-id.
/// After this method is called, according to [C++ 3.4.3p3], names should be
/// looked up in the declarator-id's scope, until the declarator is parsed and
/// ActOnCXXExitDeclaratorScope is called.
/// The 'SS' should be a non-empty valid CXXScopeSpec.
bool Sema::ActOnCXXEnterDeclaratorScope(Scope *S, const CXXScopeSpec &SS) {
  assert(SS.isSet() && "Parser passed invalid CXXScopeSpec.");
  if (DeclContext *DC = computeDeclContext(SS, true)) {
    // Before we enter a declarator's context, we need to make sure that
    // it is a complete declaration context.
    if (!DC->isDependentContext() && RequireCompleteDeclContext(SS))
      return true;
      
    EnterDeclaratorContext(S, DC);
  }
  
  return false;
}

/// ActOnCXXExitDeclaratorScope - Called when a declarator that previously
/// invoked ActOnCXXEnterDeclaratorScope(), is finished. 'SS' is the same
/// CXXScopeSpec that was passed to ActOnCXXEnterDeclaratorScope as well.
/// Used to indicate that names should revert to being looked up in the
/// defining scope.
void Sema::ActOnCXXExitDeclaratorScope(Scope *S, const CXXScopeSpec &SS) {
  assert(SS.isSet() && "Parser passed invalid CXXScopeSpec.");
  if (SS.isInvalid())
    return;
  if (computeDeclContext(SS, true))
    ExitDeclaratorContext(S);
}
