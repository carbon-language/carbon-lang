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
#include "clang/Parse/DeclSpec.h"
#include "llvm/ADT/STLExtras.h"
using namespace clang;


/// ActOnCXXGlobalScopeSpecifier - Return the object that represents the
/// global scope ('::').
Sema::CXXScopeTy *Sema::ActOnCXXGlobalScopeSpecifier(Scope *S,
                                                     SourceLocation CCLoc) {
  return cast<DeclContext>(Context.getTranslationUnitDecl());
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
  NamedDecl *SD = LookupParsedName(S, &SS, &II, LookupNestedNameSpecifierName);

  if (SD) {
    if (TypedefDecl *TD = dyn_cast<TypedefDecl>(SD)) {
      if (const RecordType* Record = TD->getUnderlyingType()->getAsRecordType())
        return cast<DeclContext>(Record->getDecl());
    } else if (isa<NamespaceDecl>(SD) || isa<RecordDecl>(SD)) {
      return cast<DeclContext>(SD);
    }

    // FIXME: Template parameters and dependent types.
    // FIXME: C++0x scoped enums

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
  QualType Type = QualType::getFromOpaquePtr(Ty);
  assert(Type->isRecordType() && 
         "Types in a nested-name-specifier always refer to a record type");
  return cast<DeclContext>(Type->getAsRecordType()->getDecl());
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
  CurContext = static_cast<DeclContext*>(SS.getScopeRep());
  S->setEntity(CurContext);
}

/// ActOnCXXExitDeclaratorScope - Called when a declarator that previously
/// invoked ActOnCXXEnterDeclaratorScope(), is finished. 'SS' is the same
/// CXXScopeSpec that was passed to ActOnCXXEnterDeclaratorScope as well.
/// Used to indicate that names should revert to being looked up in the
/// defining scope.
void Sema::ActOnCXXExitDeclaratorScope(Scope *S, const CXXScopeSpec &SS) {
  assert(SS.isSet() && "Parser passed invalid CXXScopeSpec.");
  assert(S->getEntity() == SS.getScopeRep() && "Context imbalance!");
  S->setEntity(PreDeclaratorDC);
  PreDeclaratorDC = 0;

  // Reset CurContext to the nearest enclosing context.
  while (!S->getEntity() && S->getParent())
    S = S->getParent();
  CurContext = static_cast<DeclContext*>(S->getEntity());
}
