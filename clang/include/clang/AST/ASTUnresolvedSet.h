//===-- ASTUnresolvedSet.h - Unresolved sets of declarations  ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file provides an UnresolvedSet-like class, whose contents are
//  allocated using the allocator associated with an ASTContext.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_ASTUNRESOLVEDSET_H
#define LLVM_CLANG_AST_ASTUNRESOLVEDSET_H

#include "clang/AST/ASTVector.h"
#include "clang/AST/UnresolvedSet.h"

namespace clang {

/// \brief An UnresolvedSet-like class which uses the ASTContext's allocator.
class ASTUnresolvedSet {
  typedef ASTVector<DeclAccessPair> DeclsTy;
  DeclsTy Decls;

  ASTUnresolvedSet(const ASTUnresolvedSet &) LLVM_DELETED_FUNCTION;
  void operator=(const ASTUnresolvedSet &) LLVM_DELETED_FUNCTION;

public:
  ASTUnresolvedSet() {}
  ASTUnresolvedSet(ASTContext &C, unsigned N) : Decls(C, N) {}

  typedef UnresolvedSetIterator iterator;
  typedef UnresolvedSetIterator const_iterator;

  iterator begin() { return iterator(Decls.begin()); }
  iterator end() { return iterator(Decls.end()); }

  const_iterator begin() const { return const_iterator(Decls.begin()); }
  const_iterator end() const { return const_iterator(Decls.end()); }

  void addDecl(ASTContext &C, NamedDecl *D, AccessSpecifier AS) {
    Decls.push_back(DeclAccessPair::make(D, AS), C);
  }

  /// Replaces the given declaration with the new one, once.
  ///
  /// \return true if the set changed
  bool replace(const NamedDecl* Old, NamedDecl *New, AccessSpecifier AS) {
    for (DeclsTy::iterator I = Decls.begin(), E = Decls.end(); I != E; ++I) {
      if (I->getDecl() == Old) {
        I->set(New, AS);
        return true;
      }
    }
    return false;
  }

  void erase(unsigned I) {
    Decls[I] = Decls.back();
    Decls.pop_back();
  }

  void clear() { Decls.clear(); }

  bool empty() const { return Decls.empty(); }
  unsigned size() const { return Decls.size(); }

  void reserve(ASTContext &C, unsigned N) {
    Decls.reserve(C, N);
  }

  void append(ASTContext &C, iterator I, iterator E) {
    Decls.append(C, I.ir, E.ir);
  }

  DeclAccessPair &operator[](unsigned I) { return Decls[I]; }
  const DeclAccessPair &operator[](unsigned I) const { return Decls[I]; }
};
  
} // namespace clang

#endif
