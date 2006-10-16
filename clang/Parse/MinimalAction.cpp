//===--- EmptyAction.cpp - Minimalistic action implementation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the EmptyAction interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"
#include "clang/Parse/Declarations.h"
#include "clang/Parse/Scope.h"
using namespace llvm;
using namespace clang;

/// TypedefInfo - A link exists here for each scope that an identifier is
/// defined.
struct TypedefInfo {
  TypedefInfo *Prev;
  bool isTypedef;
};

/// isTypedefName - This looks at the IdentifierInfo::FETokenInfo field to
/// determine whether the name is a typedef or not in this scope.
bool EmptyAction::isTypedefName(const IdentifierInfo &II, Scope *S) const {
  TypedefInfo *TI = II.getFETokenInfo<TypedefInfo>();
  return TI != 0 && TI->isTypedef;
}

/// ParseDeclarator - If this is a typedef declarator, we modify the
/// IdentifierInfo::FETokenInfo field to keep track of this fact, until S is
/// popped.
Action::DeclTy *
EmptyAction::ParseDeclarator(Scope *S, Declarator &D, ExprTy *Init,
                             DeclTy *LastInGroup) {
  // If there is no identifier associated with this declarator, bail out.
  if (D.getIdentifier() == 0) return 0;
  
  // Remember whether or not this declarator is a typedef.
  TypedefInfo *TI = new TypedefInfo();
  TI->isTypedef = D.getDeclSpec().StorageClassSpec == DeclSpec::SCS_typedef;

  // Add this to the linked-list hanging off the identifier.
  IdentifierInfo &II = *D.getIdentifier();
  TI->Prev = II.getFETokenInfo<TypedefInfo>();
  II.setFETokenInfo(TI);
  
  // Remember that this needs to be removed when the scope is popped.
  S->AddDecl(&II);
  return 0;
}

/// PopScope - When a scope is popped, if any typedefs are now out-of-scope,
/// they are removed from the IdentifierInfo::FETokenInfo field.
void EmptyAction::PopScope(SourceLocation Loc, Scope *S) {
  for (Scope::decl_iterator I = S->decl_begin(), E = S->decl_end();
       I != E; ++I) {
    IdentifierInfo &II = *static_cast<IdentifierInfo*>(*I);
    TypedefInfo *TI = II.getFETokenInfo<TypedefInfo>();
    assert(TI && "This decl didn't get pushed??");

    TypedefInfo *Next = TI->Prev;
    delete TI;
    
    II.setFETokenInfo(Next);
  }
}
