//===--- SelectorMap.h - Maps selectors to methods and messages -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  SelectorMap creates a mapping from selectors to ObjC method declarations
//  and ObjC message expressions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_SELECTORMAP_H
#define LLVM_CLANG_INDEX_SELECTORMAP_H

#include "clang/Index/ASTLocation.h"
#include "clang/Basic/IdentifierTable.h"
#include <map>

namespace clang {
  class ASTContext;
  class ObjCMethodDecl;

namespace idx {
  
/// \brief Maps NamedDecls with the ASTLocations that reference them.
///
/// References are mapped and retrieved using the canonical decls.
class SelectorMap {

  template <typename iter_type>
  class wrap_pair_iterator {
    iter_type I;

    wrap_pair_iterator(iter_type i) : I(i) { }
    friend class SelectorMap;

  public:
    typedef typename iter_type::value_type::second_type value_type;
    typedef value_type& reference;
    typedef value_type* pointer;
    typedef typename iter_type::iterator_category iterator_category;
    typedef typename iter_type::difference_type   difference_type;

    wrap_pair_iterator() { }

    reference operator*() const { return I->second; }
    pointer operator->() const { return &I->second; }

    wrap_pair_iterator& operator++() {
      ++I;
      return *this;
    }

    wrap_pair_iterator operator++(int) {
      wrap_pair_iterator tmp(*this);
      ++(*this);
      return tmp;
    }

    friend bool operator==(wrap_pair_iterator L, wrap_pair_iterator R) { 
      return L.I == R.I;
    }
    friend bool operator!=(wrap_pair_iterator L, wrap_pair_iterator R) { 
      return L.I != R.I;
    }
  };

public:
  explicit SelectorMap(ASTContext &Ctx);
  
  typedef std::multimap<Selector, ObjCMethodDecl *> SelMethMapTy;
  typedef std::multimap<Selector, ASTLocation> SelRefMapTy;

  typedef wrap_pair_iterator<SelMethMapTy::iterator> method_iterator;
  typedef wrap_pair_iterator<SelRefMapTy::iterator> astlocation_iterator;

  method_iterator methods_begin(Selector Sel) const;
  method_iterator methods_end(Selector Sel) const;

  astlocation_iterator refs_begin(Selector Sel) const;
  astlocation_iterator refs_end(Selector Sel) const;
  
private:
  mutable SelMethMapTy SelMethMap;
  mutable SelRefMapTy SelRefMap;
};

} // end idx namespace
  
} // end clang namespace

#endif
