// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_FIND_H_
#define CARBON_COMMON_FIND_H_

#include <concepts>
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

template <typename Range>
concept RangeValueHasNoneType = requires {
  { RangeValueType<Range>::None } -> std::convertible_to<RangeValueType<Range>>;
};

}  // namespace Internal

// Finds a value in the given `range` by testing the `predicate`. Returns a
// pointer to the value from the range on success, and nullptr if nothing is
// found.
//
// This is similar to `std::find_if()` but returns a pointer to the value
// instead of an iterator that must be tested against `end()`.
template <typename Range, typename Pred>
  requires Internal::IsValidFindPredicate<Range, Pred>
constexpr auto FindIfOrNull(Range&& range, Pred predicate)
    -> Internal::RangePointerType<Range> {
  auto it = llvm::find_if(range, predicate);
  if (it != range.end()) {
    return std::addressof(*it);
  } else {
    return nullptr;
  }
}

// Finds a value in the given `range` by testing the `predicate` and returns a
// copy of it. If no match is found, returns `T::None` where the input range is
// over values of type `T`.
template <typename Range, typename Pred>
  requires Internal::IsValidFindPredicate<Range, Pred> &&
           Internal::RangeValueHasNoneType<Range> &&
           std::copy_constructible<Internal::RangeValueType<Range>>
constexpr auto FindIfOrNone(Range&& range, Pred predicate)
    -> Internal::RangeValueType<Range> {
  auto it = llvm::find_if(range, predicate);
  if (it != range.end()) {
    return *it;
  } else {
    return Internal::RangeValueType<Range>::None;
  }
}

// Finds a value in the given `range` by comparing to `query`. Returns a
// pointer to the value from the range on success, and nullptr if nothing is
// found.
//
// This is similar to `std::find_if()` but returns a pointer to the value
// instead of an iterator that must be tested against `end()`.
template <typename Range, typename Query = Internal::RangeValueType<Range>>
  requires Internal::IsComparable<Query, Internal::RangeValueType<Range>>
constexpr auto Contains(Range&& range, const Query& query) -> bool {
  return llvm::find(range, query) != range.end();
}

}  // namespace Carbon

#endif  // CARBON_COMMON_FIND_H_
