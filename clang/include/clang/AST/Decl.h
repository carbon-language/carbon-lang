//===--- Decl.h - Classes for representing declarations ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Decl interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DECL_H
#define LLVM_CLANG_AST_DECL_H

#include "clang/Basic/SourceLocation.h"
#include "clang/Parse/Declarations.h"

namespace llvm {
namespace clang {
class IdentifierInfo;
  
/// Decl - This represents one declaration (or definition), e.g. a variable, 
/// typedef, function, struct, etc.  
///
class Decl {
  /// Identifier - The identifier for this declaration (e.g. the name for the
  /// variable, the tag for a struct).
  IdentifierInfo *Identifier;
  
  /// DeclarationSpecifier - Information about storage class, type specifiers,
  /// etc.
  DeclSpec DeclarationSpecifier;
  
  /// Type.
  /// Kind.
  
  /// Loc - The location of the declaration in the source code.
  ///
  SourceLocation Loc;
  
  /// Scope stack info when parsing, otherwise decl list when scope is popped.
  ///
  Decl *Next;
public:
  Decl(IdentifierInfo *Id, const DeclSpec &DS, SourceLocation loc, Decl *next)
    : Identifier(Id), DeclarationSpecifier(DS), Loc(loc), Next(next) {}
  
  
  const IdentifierInfo *getIdentifier() const { return Identifier; }
  
  const DeclSpec &getDeclSpecs() const { return DeclarationSpecifier; }
  
  Decl *getNext() const { return Next; }
  
  
};
  
}  // end namespace clang
}  // end namespace llvm

#endif
