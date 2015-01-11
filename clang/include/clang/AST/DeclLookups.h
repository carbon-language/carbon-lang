//===-- DeclLookups.h - Low-level interface to all names in a DC-*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines DeclContext::all_lookups_iterator.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DECLLOOKUPS_H
#define LLVM_CLANG_AST_DECLLOOKUPS_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclContextInternals.h"
#include "clang/AST/DeclarationName.h"

namespace clang {

/// all_lookups_iterator - An iterator that provides a view over the results
/// of looking up every possible name.
class DeclContext::all_lookups_iterator {
  StoredDeclsMap::iterator It, End;
public:
  typedef lookup_result             value_type;
  typedef lookup_result             reference;
  typedef lookup_result             pointer;
  typedef std::forward_iterator_tag iterator_category;
  typedef std::ptrdiff_t            difference_type;

  all_lookups_iterator() {}
  all_lookups_iterator(StoredDeclsMap::iterator It,
                       StoredDeclsMap::iterator End)
      : It(It), End(End) {}

  DeclarationName getLookupName() const { return It->first; }

  reference operator*() const { return It->second.getLookupResult(); }
  pointer operator->() const { return It->second.getLookupResult(); }

  all_lookups_iterator& operator++() {
    // Filter out using directives. They don't belong as results from name
    // lookup anyways, except as an implementation detail. Users of the API
    // should not expect to get them (or worse, rely on it).
    do {
      ++It;
    } while (It != End &&
             It->first == DeclarationName::getUsingDirectiveName());
             
    return *this;
  }

  all_lookups_iterator operator++(int) {
    all_lookups_iterator tmp(*this);
    ++(*this);
    return tmp;
  }

  friend bool operator==(all_lookups_iterator x, all_lookups_iterator y) {
    return x.It == y.It;
  }
  friend bool operator!=(all_lookups_iterator x, all_lookups_iterator y) {
    return x.It != y.It;
  }
};

inline DeclContext::lookups_range DeclContext::lookups() const {
  DeclContext *Primary = const_cast<DeclContext*>(this)->getPrimaryContext();
  if (Primary->hasExternalVisibleStorage())
    getParentASTContext().getExternalSource()->completeVisibleDeclsMap(Primary);
  if (StoredDeclsMap *Map = Primary->buildLookup())
    return lookups_range(all_lookups_iterator(Map->begin(), Map->end()),
                         all_lookups_iterator(Map->end(), Map->end()));

  // Synthesize an empty range. This requires that two default constructed
  // versions of these iterators form a valid empty range.
  return lookups_range(all_lookups_iterator(), all_lookups_iterator());
}

inline DeclContext::all_lookups_iterator DeclContext::lookups_begin() const {
  return lookups().begin();
}

inline DeclContext::all_lookups_iterator DeclContext::lookups_end() const {
  return lookups().end();
}

inline DeclContext::lookups_range DeclContext::noload_lookups() const {
  DeclContext *Primary = const_cast<DeclContext*>(this)->getPrimaryContext();
  if (StoredDeclsMap *Map = Primary->getLookupPtr())
    return lookups_range(all_lookups_iterator(Map->begin(), Map->end()),
                         all_lookups_iterator(Map->end(), Map->end()));

  // Synthesize an empty range. This requires that two default constructed
  // versions of these iterators form a valid empty range.
  return lookups_range(all_lookups_iterator(), all_lookups_iterator());
}

inline
DeclContext::all_lookups_iterator DeclContext::noload_lookups_begin() const {
  return noload_lookups().begin();
}

inline
DeclContext::all_lookups_iterator DeclContext::noload_lookups_end() const {
  return noload_lookups().end();
}

} // end namespace clang

#endif
