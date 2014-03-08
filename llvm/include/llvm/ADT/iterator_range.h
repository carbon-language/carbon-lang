//===- iterator_range.h - A range adaptor for iterators ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

#include <utility>

namespace llvm {

template <typename Range>
struct range_traits {
  typedef typename Range::difference_type difference_type;
};

/// \brief A range adaptor for a pair of iterators.
///
/// This just wraps two iterators into a range-compatible interface. Nothing
/// fancy at all.
template <typename IteratorT>
class iterator_range {
  IteratorT begin_iterator, end_iterator;

public:
  // FIXME: We should be using iterator_traits to determine the
  // difference_type, but most of our iterators do not expose anything like it.
  typedef int difference_type;

  iterator_range() {}
  iterator_range(IteratorT begin_iterator, IteratorT end_iterator)
      : begin_iterator(std::move(begin_iterator)),
        end_iterator(std::move(end_iterator)) {}

  IteratorT begin() const { return begin_iterator; }
  IteratorT end() const { return end_iterator; }
};

/// \brief Determine the distance between the end() and begin() iterators of
/// a range. Analogous to std::distance().
template <class Range>
typename range_traits<Range>::difference_type distance(Range R) {
  return std::distance(R.begin(), R.end());
}

/// \brief Copies members of a range into the output iterator provided.
/// Analogous to std::copy.
template <class Range, class OutputIterator>
OutputIterator copy(Range In, OutputIterator Result) {
  return std::copy(In.begin(), In.end(), Result);
}
}

#endif
