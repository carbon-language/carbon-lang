//===--- DeclReferenceMap.h - Map Decls to their references -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  DeclReferenceMap creates a mapping from Decls to the ASTNodes that
//  reference them.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DECLREFERENCEMAP_H
#define LLVM_CLANG_AST_DECLREFERENCEMAP_H

#include "clang/AST/ASTNode.h"
#include <map>

namespace clang {
  class ASTContext;
  class NamedDecl;
  
/// \brief Maps NamedDecls with the ASTNodes that reference them.
///
/// References are mapped and retrieved using the primary decls
/// (see Decl::getPrimaryDecl()).
class DeclReferenceMap {
public:
  explicit DeclReferenceMap(ASTContext &Ctx);
  
  typedef std::multimap<NamedDecl*, ASTNode> MapTy;

  class astnode_iterator {
    MapTy::iterator I;

    astnode_iterator(MapTy::iterator i) : I(i) { }
    friend class DeclReferenceMap;

  public:
    typedef ASTNode  value_type;
    typedef ASTNode& reference;
    typedef ASTNode* pointer;
    typedef MapTy::iterator::iterator_category iterator_category;
    typedef MapTy::iterator::difference_type   difference_type;

    astnode_iterator() { }

    reference operator*() const { return I->second; }
    pointer operator->() const { return &I->second; }

    astnode_iterator& operator++() {
      ++I;
      return *this;
    }

    astnode_iterator operator++(int) {
      astnode_iterator tmp(*this);
      ++(*this);
      return tmp;
    }

    friend bool operator==(astnode_iterator L, astnode_iterator R) { 
      return L.I == R.I;
    }
    friend bool operator!=(astnode_iterator L, astnode_iterator R) { 
      return L.I != R.I;
    }
  };

  astnode_iterator refs_begin(NamedDecl *D) const;
  astnode_iterator refs_end(NamedDecl *D) const;
  bool refs_empty(NamedDecl *D) const;
  
private:
  mutable MapTy Map;
};
  
} // end clang namespace

#endif
