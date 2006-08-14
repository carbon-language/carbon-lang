//===--- Scope.h - Scope interface ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Scope interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARSE_SCOPE_H
#define LLVM_CLANG_PARSE_SCOPE_H

#include "clang/Parse/Action.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
namespace clang {
    
/// Scope - A scope is a transient data structure that is used while parsing the
/// program.  It assists with resolving identifiers to the appropriate
/// declaration.
///
class Scope {
  /// The parent scope for this scope.  This is null for the translation-unit
  /// scope.
  Scope *Parent;
  
  /// Depth - This is the depth of this scope.  The translation-unit scope has
  /// depth 0.
  unsigned Depth;
  
  /// DeclsInScope - This keeps track of all declarations in this scope.  When
  /// the declaration is added to the scope, it is set as the current
  /// declaration for the identifier in the IdentifierTable.  When the scope is
  /// popped, these declarations are removed from the IdentifierTable's notion
  /// of current declaration.  It is up to the current Action implementation to
  /// implement these semantics.
  SmallVector<Action::DeclTy*, 32> DeclsInScope;
public:
  Scope(Scope *parent) : Parent(parent), Depth(Parent ? Parent->Depth+1 : 0) {
  }
  
  /// getParent - Return the scope that this is nested in.
  ///
  Scope *getParent() const { return Parent; }
  
  typedef SmallVector<Action::DeclTy*, 32>::iterator decl_iterator;
  typedef SmallVector<Action::DeclTy*, 32>::const_iterator decl_const_iterator;
  
  decl_iterator decl_begin() { return DeclsInScope.begin(); }
  decl_iterator decl_end()   { return DeclsInScope.end(); }

  decl_const_iterator decl_begin() const { return DeclsInScope.begin(); }
  decl_const_iterator decl_end()   const { return DeclsInScope.end(); }

  void AddDecl(Action::DeclTy *D) {
    DeclsInScope.push_back(D);
  }
  
};
    
}  // end namespace clang
}  // end namespace llvm

#endif
