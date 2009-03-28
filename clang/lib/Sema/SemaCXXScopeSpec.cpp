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
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/Parse/DeclSpec.h"
#include "llvm/ADT/STLExtras.h"
using namespace clang;

/// \brief Compute the DeclContext that is associated with the given
/// scope specifier.
DeclContext *Sema::computeDeclContext(const CXXScopeSpec &SS) {
  if (!SS.isSet() || SS.isInvalid())
    return 0;

  NestedNameSpecifier *NNS 
    = static_cast<NestedNameSpecifier *>(SS.getScopeRep());
  if (NNS->isDependent())
    return 0;

  switch (NNS->getKind()) {
  case NestedNameSpecifier::Identifier:
    assert(false && "Dependent nested-name-specifier has no DeclContext");
    break;

  case NestedNameSpecifier::Namespace:
    return NNS->getAsNamespace();

  case NestedNameSpecifier::TypeSpec:
  case NestedNameSpecifier::TypeSpecWithTemplate: {
    const TagType *Tag = NNS->getAsType()->getAsTagType();
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
  
  DeclContext *DC = computeDeclContext(SS);
  if (TagDecl *Tag = dyn_cast<TagDecl>(DC)) {
    // If we're currently defining this type, then lookup into the
    // type is okay: don't complain that it isn't complete yet.
    const TagType *TagT = Context.getTypeDeclType(Tag)->getAsTagType();
    if (TagT->isBeingDefined())
      return false;

    // The type must be complete.
    return RequireCompleteType(SS.getRange().getBegin(),
                               Context.getTypeDeclType(Tag),
                               diag::err_incomplete_nested_name_spec,
                               SS.getRange());
  }

  return false;
}

/// ActOnCXXGlobalScopeSpecifier - Return the object that represents the
/// global scope ('::').
Sema::CXXScopeTy *Sema::ActOnCXXGlobalScopeSpecifier(Scope *S,
                                                     SourceLocation CCLoc) {
  return NestedNameSpecifier::GlobalSpecifier(Context);
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
                                                    IdentifierInfo &II) {
  NestedNameSpecifier *Prefix 
    = static_cast<NestedNameSpecifier *>(SS.getScopeRep());

  // If the prefix is already dependent, there is no name lookup to
  // perform. Just build the resulting nested-name-specifier.
  if (Prefix && Prefix->isDependent())
    return NestedNameSpecifier::Create(Context, Prefix, &II);

  NamedDecl *SD = LookupParsedName(S, &SS, &II, LookupNestedNameSpecifierName);

  if (SD) {
    if (NamespaceDecl *Namespace = dyn_cast<NamespaceDecl>(SD))
      return NestedNameSpecifier::Create(Context, Prefix, Namespace);

    if (TypeDecl *Type = dyn_cast<TypeDecl>(SD)) {
      // Determine whether we have a class (or, in C++0x, an enum) or
      // a typedef thereof. If so, build the nested-name-specifier.
      QualType T = Context.getTypeDeclType(Type);
      bool AcceptableType = false;
      if (T->isDependentType())
        AcceptableType = true;
      else if (TypedefDecl *TD = dyn_cast<TypedefDecl>(SD)) {
        if (TD->getUnderlyingType()->isRecordType() ||
            (getLangOptions().CPlusPlus0x && 
             TD->getUnderlyingType()->isEnumeralType()))
          AcceptableType = true;
      } else if (isa<RecordDecl>(Type) || 
                 (getLangOptions().CPlusPlus0x && isa<EnumDecl>(Type)))
        AcceptableType = true;

      if (AcceptableType)
        return NestedNameSpecifier::Create(Context, Prefix, false, 
                                           T.getTypePtr());
    }
    
    if (NamespaceAliasDecl *Alias = dyn_cast<NamespaceAliasDecl>(SD))
      return NestedNameSpecifier::Create(Context, Prefix,
                                         Alias->getNamespace());

    // Fall through to produce an error: we found something that isn't
    // a class or a namespace.
  }

  // If we didn't find anything during our lookup, try again with
  // ordinary name lookup, which can help us produce better error
  // messages.
  if (!SD)
    SD = LookupParsedName(S, &SS, &II, LookupOrdinaryName);
  unsigned DiagID;
  if (SD)
    DiagID = diag::err_expected_class_or_namespace;
  else if (SS.isSet())
    DiagID = diag::err_typecheck_no_member;
  else
    DiagID = diag::err_undeclared_var_use;

  if (SS.isSet())
    Diag(IdLoc, DiagID) << &II << SS.getRange();
  else
    Diag(IdLoc, DiagID) << &II;

  return 0;
}

Sema::CXXScopeTy *Sema::ActOnCXXNestedNameSpecifier(Scope *S,
                                                    const CXXScopeSpec &SS,
                                                    TypeTy *Ty,
                                                    SourceRange TypeRange,
                                                    SourceLocation CCLoc) {
  NestedNameSpecifier *Prefix 
    = static_cast<NestedNameSpecifier *>(SS.getScopeRep());
  return NestedNameSpecifier::Create(Context, Prefix, /*FIXME:*/false,
                                QualType::getFromOpaquePtr(Ty).getTypePtr());
}

/// ActOnCXXEnterDeclaratorScope - Called when a C++ scope specifier (global
/// scope or nested-name-specifier) is parsed, part of a declarator-id.
/// After this method is called, according to [C++ 3.4.3p3], names should be
/// looked up in the declarator-id's scope, until the declarator is parsed and
/// ActOnCXXExitDeclaratorScope is called.
/// The 'SS' should be a non-empty valid CXXScopeSpec.
void Sema::ActOnCXXEnterDeclaratorScope(Scope *S, const CXXScopeSpec &SS) {
  assert(SS.isSet() && "Parser passed invalid CXXScopeSpec.");
  assert(PreDeclaratorDC == 0 && "Previous declarator context not popped?");
  PreDeclaratorDC = static_cast<DeclContext*>(S->getEntity());
  CurContext = computeDeclContext(SS);
  assert(CurContext && "No context?");
  S->setEntity(CurContext);
}

/// ActOnCXXExitDeclaratorScope - Called when a declarator that previously
/// invoked ActOnCXXEnterDeclaratorScope(), is finished. 'SS' is the same
/// CXXScopeSpec that was passed to ActOnCXXEnterDeclaratorScope as well.
/// Used to indicate that names should revert to being looked up in the
/// defining scope.
void Sema::ActOnCXXExitDeclaratorScope(Scope *S, const CXXScopeSpec &SS) {
  assert(SS.isSet() && "Parser passed invalid CXXScopeSpec.");
  assert(S->getEntity() == computeDeclContext(SS) && "Context imbalance!");
  S->setEntity(PreDeclaratorDC);
  PreDeclaratorDC = 0;

  // Reset CurContext to the nearest enclosing context.
  while (!S->getEntity() && S->getParent())
    S = S->getParent();
  CurContext = static_cast<DeclContext*>(S->getEntity());
  assert(CurContext && "No context?");
}
