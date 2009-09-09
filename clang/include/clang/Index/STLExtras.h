//===--- STLExtras.h - Helper STL related templates -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Helper templates for using with the STL.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_STLEXTRAS_H
#define LLVM_CLANG_INDEX_STLEXTRAS_H

namespace clang {

namespace idx {

/// \brief Wraps an iterator whose value_type is a pair, and provides
/// pair's second object as the value.
template <typename iter_type>
class pair_value_iterator {
  iter_type I;

public:
  typedef typename iter_type::value_type::second_type value_type;
  typedef value_type& reference;
  typedef value_type* pointer;
  typedef typename iter_type::iterator_category iterator_category;
  typedef typename iter_type::difference_type   difference_type;

  pair_value_iterator() { }
  pair_value_iterator(iter_type i) : I(i) { }

  reference operator*() const { return I->second; }
  pointer operator->() const { return &I->second; }

  pair_value_iterator& operator++() {
    ++I;
    return *this;
  }

  pair_value_iterator operator++(int) {
    pair_value_iterator tmp(*this);
    ++(*this);
    return tmp;
  }

  friend bool operator==(pair_value_iterator L, pair_value_iterator R) {
    return L.I == R.I;
  }
  friend bool operator!=(pair_value_iterator L, pair_value_iterator R) {
    return L.I != R.I;
  }
};

} // end idx namespace

} // end clang namespace

#endif
