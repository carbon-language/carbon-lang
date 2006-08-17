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
  
  /// Type.
  /// Kind.
  
  /// Loc - The location of the declaration in the source code.
  ///
  SourceLocation Loc;
  
  /// Next - Decls are chained together in a singly-linked list by their owning
  /// object.  Currently we allow decls to be owned by a translation unit or a
  /// function.  This way we can deallocate a function body and all the
  /// declarations within it.
  Decl *Next;
public:
  Decl(IdentifierInfo *Id, SourceLocation loc, Decl *next)
    : Identifier(Id), Loc(loc), Next(next) {}
  
  
};
  
}  // end namespace clang
}  // end namespace llvm

#endif
