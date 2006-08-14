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
//#include "clang/Parse/Declarations.h"
//#include "clang/Parse/Scope.h"
using namespace llvm;
using namespace clang;

/// isTypedefName - This looks at the IdentifierInfo::FETokenInfo field to
/// determine whether the name is a typedef or not in this scope.
bool EmptyAction::isTypedefName(const IdentifierInfo &II, Scope *S) const {
  return true;
}

/// ParseDeclarator - If this is a typedef declarator, we modify the
/// IdentifierInfo::FETokenInfo field to keep track of this fact, until S is
/// popped.
void EmptyAction::ParseDeclarator(SourceLocation Loc, Scope *S, Declarator &D,
                                  ExprTy *Init) {
}

/// PopScope - When a scope is popped, if any typedefs are now out-of-scope,
/// they are removed from the IdentifierInfo::FETokenInfo field.
void EmptyAction::PopScope(SourceLocation Loc, Scope *S) {
  
}
