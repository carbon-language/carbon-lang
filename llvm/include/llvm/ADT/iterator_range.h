//===- iterator_range.h - A range adaptor for iterators ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This provides a very simple, boring adaptor for a begin and end iterator
/// into a range type. This should be used to build range views that work well
/// with range based for loops and range based constructors.
///
/// Note that code here follows more standards-based coding conventions as it
/// is mirroring proposed interfaces for standardization.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_ITERATOR_RANGE_H
#define LLVM_ADT_ITERATOR_RANGE_H

#include <iterator>
#include <utility>
#include <cassert>

namespace llvm {

template <typename T>
constexpr bool is_random_iterator() {
  return std::is_same<
    typename std::iterator_traits<T>::iterator_category,
    std::random_access_iterator_tag>::value;
}

/// A range adaptor for a pair of iterators.
///
/// This just wraps two iterators into a range-compatible interface. Nothing
/// fancy at all.
template <typename IteratorT>
class iterator_range {
  IteratorT begin_iterator, end_iterator;

public:
  //TODO: Add SFINAE to test that the Container's iterators match the range's
  //      iterators.
  template <typename Container>
  iterator_range(Container &&c)
  //TODO: Consider ADL/non-member begin/end calls.
      : begin_iterator(c.begin()), end_iterator(c.end()) {}
  iterator_range(IteratorT begin_iterator, IteratorT end_iterator)
      : begin_iterator(std::move(begin_iterator)),
        end_iterator(std::move(end_iterator)) {}

  IteratorT begin() const { return begin_iterator; }
  IteratorT end() const { return end_iterator; }
};

/// Convenience function for iterating over sub-ranges.
///
/// This provides a bit of syntactic sugar to make using sub-ranges
/// in for loops a bit easier. Analogous to std::make_pair().
template <class T> iterator_range<T> make_range(T x, T y) {
  return iterator_range<T>(std::move(x), std::move(y));
}

template <typename T> iterator_range<T> make_range(std::pair<T, T> p) {
  return iterator_range<T>(std::move(p.first), std::move(p.second));
}

/// Non-random-iterator version
template <typename T>
auto drop_begin(T &&t, int n) ->
  typename std::enable_if<!is_random_iterator<decltype(adl_begin(t))>(),
  iterator_range<decltype(adl_begin(t))>>::type {
  auto begin = adl_begin(t);
  auto end = adl_end(t);
  for (int i = 0; i < n; i++) {
    assert(begin != end);
    ++begin;
  }
  return make_range(begin, end);
}

/// Optimized version for random iterators
template <typename T>
auto drop_begin(T &&t, int n) ->
  typename std::enable_if<is_random_iterator<decltype(adl_begin(t))>(),
  iterator_range<decltype(adl_begin(t))>>::type {
  auto begin = adl_begin(t);
  auto end = adl_end(t);
  assert(end - begin >= n && "Dropping more elements than exist!");
  return make_range(std::next(begin, n), end);
}

}

#endif
