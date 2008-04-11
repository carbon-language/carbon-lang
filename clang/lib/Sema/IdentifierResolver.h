//===- IdentifierResolver.h - Lexical Scope Name lookup ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the IdentifierResolver class,which is used for lexical
// scoped lookup, based on identifier.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_SEMA_IDENTIFIERRESOLVER_H
#define LLVM_CLANG_AST_SEMA_IDENTIFIERRESOLVER_H

namespace clang {
  class IdentifierInfo;
  class NamedDecl;
  class Scope;

/// IdentifierResolver - Keeps track of shadowed decls on enclosing scopes.
/// it manages the shadowing chains of identifiers and implements efficent decl
/// lookup based on an identifier.
class IdentifierResolver {
public:
  IdentifierResolver();
  ~IdentifierResolver();

  /// AddDecl - Link the decl to its shadowed decl chain
  void AddDecl(NamedDecl *D, Scope *S);

  /// AddGlobalDecl - Link the decl at the top of the shadowed decl chain
  void AddGlobalDecl(NamedDecl *D);

  /// RemoveDecl - Unlink the decl from its shadowed decl chain
  /// The decl must already be part of the decl chain.
  void RemoveDecl(NamedDecl *D);

  /// Lookup - Find the non-shadowed decl that belongs to a particular
  /// Decl::IdentifierNamespace.
  NamedDecl *Lookup(const IdentifierInfo *II, unsigned NSI);

private:
  class IdDeclInfoMap;
  IdDeclInfoMap &IdDeclInfos;
};

} // end namespace clang

#endif
