//===--- DeclReferenceMap.h - Map Decls to their references -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  DeclReferenceMap creates a mapping from Decls to the ASTLocations that
//  reference them.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DECLREFERENCEMAP_H
#define LLVM_CLANG_AST_DECLREFERENCEMAP_H

#include "clang/AST/ASTLocation.h"
#include <map>

namespace clang {
  class ASTContext;
  class NamedDecl;
  
/// \brief Maps NamedDecls with the ASTLocations that reference them.
///
/// References are mapped and retrieved using the primary decls
/// (see Decl::getPrimaryDecl()).
class DeclReferenceMap {
public:
  explicit DeclReferenceMap(ASTContext &Ctx);
  
  typedef std::multimap<NamedDecl*, ASTLocation> MapTy;

  class astlocation_iterator {
    MapTy::iterator I;

    astlocation_iterator(MapTy::iterator i) : I(i) { }
    friend class DeclReferenceMap;

  public:
    typedef ASTLocation  value_type;
    typedef ASTLocation& reference;
    typedef ASTLocation* pointer;
    typedef MapTy::iterator::iterator_category iterator_category;
    typedef MapTy::iterator::difference_type   difference_type;

    astlocation_iterator() { }

    reference operator*() const { return I->second; }
    pointer operator->() const { return &I->second; }

    astlocation_iterator& operator++() {
      ++I;
      return *this;
    }

    astlocation_iterator operator++(int) {
      astlocation_iterator tmp(*this);
      ++(*this);
      return tmp;
    }

    friend bool operator==(astlocation_iterator L, astlocation_iterator R) { 
      return L.I == R.I;
    }
    friend bool operator!=(astlocation_iterator L, astlocation_iterator R) { 
      return L.I != R.I;
    }
  };

  astlocation_iterator refs_begin(NamedDecl *D) const;
  astlocation_iterator refs_end(NamedDecl *D) const;
  bool refs_empty(NamedDecl *D) const;
  
private:
  mutable MapTy Map;
};
  
} // end clang namespace

#endif
