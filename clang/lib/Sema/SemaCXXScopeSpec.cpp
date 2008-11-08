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
#include "clang/Basic/Diagnostic.h"
using namespace clang;


namespace {
  Decl *LookupNestedName(DeclContext *LookupCtx, bool LookInParentCtx,
                         const IdentifierInfo &II, bool &IdIsUndeclared) {
    IdentifierResolver::iterator
      I = IdentifierResolver::begin(&II, LookupCtx, LookInParentCtx),
      E = IdentifierResolver::end();

    if (I == E) {
      IdIsUndeclared = true;
      return 0;
    }
    IdIsUndeclared = false;

    // C++ 3.4.3p1 :
    // During the lookup for a name preceding the :: scope resolution operator,
    // object, function, and enumerator names are ignored. If the name found is
    // not a class-name or namespace-name, the program is ill-formed.

    for (; I != E; ++I) {
      if (TypedefDecl *TD = dyn_cast<TypedefDecl>(*I)) {
        if (TD->getUnderlyingType()->isRecordType())
          break;
        continue;
      }
      if (((*I)->getIdentifierNamespace()&Decl::IDNS_Tag) && !isa<EnumDecl>(*I))
        break;    
    }

    return (I != E ? *I : 0);
  }
} // anonymous namespace

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
                                                    const IdentifierInfo &II) {
  DeclContext *DC = static_cast<DeclContext*>(SS.getScopeRep());
  Decl *SD;
  bool IdIsUndeclared;

  if (DC)
    SD = LookupNestedName(DC, false/*LookInParentCtx*/, II, IdIsUndeclared);
  else
    SD = LookupNestedName(CurContext, true/*LookInParent*/, II, IdIsUndeclared);

  if (SD) {
    if (TypedefDecl *TD = dyn_cast<TypedefDecl>(SD)) {
      assert(TD->getUnderlyingType()->isRecordType() &&"Invalid Scope Decl!");
      SD = TD->getUnderlyingType()->getAsRecordType()->getDecl();
    }

    assert((isa<NamespaceDecl>(SD) || isa<RecordDecl>(SD)) &&
           "Invalid Scope Decl!");
    return cast<DeclContext>(SD);
  }

  unsigned DiagID;
  if (!IdIsUndeclared)
    DiagID = diag::err_expected_class_or_namespace;
  else if (DC)
    DiagID = diag::err_typecheck_no_member;
  else
    DiagID = diag::err_undeclared_var_use;

  if (DC)
    Diag(IdLoc, DiagID, II.getName(), SS.getRange());
  else
    Diag(IdLoc, DiagID, II.getName());

  return 0;
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
  PreDeclaratorDC = CurContext;
  CurContext = static_cast<DeclContext*>(SS.getScopeRep());
}

/// ActOnCXXExitDeclaratorScope - Called when a declarator that previously
/// invoked ActOnCXXEnterDeclaratorScope(), is finished. 'SS' is the same
/// CXXScopeSpec that was passed to ActOnCXXEnterDeclaratorScope as well.
/// Used to indicate that names should revert to being looked up in the
/// defining scope.
void Sema::ActOnCXXExitDeclaratorScope(const CXXScopeSpec &SS) {
  assert(SS.isSet() && "Parser passed invalid CXXScopeSpec.");
  assert(CurContext == static_cast<DeclContext*>(SS.getScopeRep()) &&
         "Context imbalance!");
  CurContext = PreDeclaratorDC;
  PreDeclaratorDC = 0;
}
