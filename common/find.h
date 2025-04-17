// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_FIND_H_
#define CARBON_COMMON_FIND_H_

#include <type_traits>

#include "llvm/ADT/STLExtras.h"

namespace Carbon {

namespace Internal {

template <typename Range>
using RangePointerType = typename std::iterator_traits<decltype(std::begin(
    std::declval<Range>()))>::pointer;

template <typename Range>
using RangeValueType = typename std::iterator_traits<decltype(std::begin(
    std::declval<Range>()))>::value_type;

template <typename Range, typename Pred>
concept IsValidFindPredicate =
    requires(const RangeValueType<Range>& elem, Pred pred) {
      { pred(elem) } -> std::convertible_to<bool>;
    };

template <typename A, typename B>
concept IsComparable = requires(const A& a, const B& b) {
  { a == b } -> std::convertible_to<bool>;
};

}  // namespace Internal

// Finds a value in the given `range` by comparing to `query`. Returns a
// pointer to the value from the range on success, and nullptr if nothing is
// found.
//
// This is similar to `std::find_if()` but returns a pointer to the value
// instead of an iterator that must be tested against `end()`.
template <typename Range, typename Query = Internal::RangeValueType<Range>>
  requires Internal::IsComparable<Query, Internal::RangeValueType<Range>>
auto FindOrNull(Range&& range, const Query& query)
    -> Internal::RangePointerType<Range> {
  auto it = llvm::find(range, query);
  if (it != range.end()) {
    return std::addressof(*it);
  } else {
    return nullptr;
  }
}

// Finds a value in the given `range` by testing the `predicate`. Returns a
// pointer to the value from the range on success, and nullptr if nothing is
// found.
//
// This is similar to `std::find_if()` but returns a pointer to the value
// instead of an iterator that must be tested against `end()`.
template <typename Range, typename Pred>
  requires Internal::IsValidFindPredicate<Range, Pred>
auto FindIfOrNull(Range&& range, Pred predicate)
    -> Internal::RangePointerType<Range> {
  auto it = llvm::find_if(range, predicate);
  if (it != range.end()) {
    return std::addressof(*it);
  } else {
    return nullptr;
  }
}

}  // namespace Carbon

#endif  // CARBON_COMMON_FIND_H_
