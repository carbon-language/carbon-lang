//===- Sequence.h - Utility for producing sequences of values ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This routine provides some synthesis utilities to produce sequences of
/// values. The names are intentionally kept very short as they tend to occur
/// in common and widely used contexts.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SEQUENCE_H
#define LLVM_ADT_SEQUENCE_H

#include <cassert>     // assert
#include <iterator>    // std::random_access_iterator_tag
#include <limits>      // std::numeric_limits
#include <type_traits> // std::underlying_type, std::is_enum

#include "llvm/Support/MathExtras.h" // AddOverflow / SubOverflow

namespace llvm {

namespace detail {

// Returns whether a value of type U can be represented with type T.
template <typename T, typename U> bool canTypeFitValue(const U Value) {
  const intmax_t BotT = intmax_t(std::numeric_limits<T>::min());
  const intmax_t BotU = intmax_t(std::numeric_limits<U>::min());
  const uintmax_t TopT = uintmax_t(std::numeric_limits<T>::max());
  const uintmax_t TopU = uintmax_t(std::numeric_limits<U>::max());
  return !((BotT > BotU && Value < static_cast<U>(BotT)) ||
           (TopT < TopU && Value > static_cast<U>(TopT)));
}

// An integer type that asserts when:
// - constructed from a value that doesn't fit into intmax_t,
// - casted to a type that cannot hold the current value,
// - its internal representation overflows.
struct CheckedInt {
  // Integral constructor, asserts if Value cannot be represented as intmax_t.
  template <typename Integral, typename std::enable_if_t<
                                   std::is_integral<Integral>::value, bool> = 0>
  static CheckedInt from(Integral FromValue) {
    if (!canTypeFitValue<intmax_t>(FromValue))
      assertOutOfBounds();
    CheckedInt Result;
    Result.Value = static_cast<intmax_t>(FromValue);
    return Result;
  }

  // Enum constructor, asserts if Value cannot be represented as intmax_t.
  template <typename Enum,
            typename std::enable_if_t<std::is_enum<Enum>::value, bool> = 0>
  static CheckedInt from(Enum FromValue) {
    using type = typename std::underlying_type<Enum>::type;
    return from<type>(static_cast<type>(FromValue));
  }

  // Equality
  bool operator==(const CheckedInt &O) const { return Value == O.Value; }
  bool operator!=(const CheckedInt &O) const { return Value != O.Value; }

  CheckedInt operator+(intmax_t Offset) const {
    CheckedInt Result;
    if (AddOverflow(Value, Offset, Result.Value))
      assertOutOfBounds();
    return Result;
  }

  intmax_t operator-(CheckedInt Other) const {
    intmax_t Result;
    if (SubOverflow(Value, Other.Value, Result))
      assertOutOfBounds();
    return Result;
  }

  // Convert to integral, asserts if Value cannot be represented as Integral.
  template <typename Integral, typename std::enable_if_t<
                                   std::is_integral<Integral>::value, bool> = 0>
  Integral to() const {
    if (!canTypeFitValue<Integral>(Value))
      assertOutOfBounds();
    return static_cast<Integral>(Value);
  }

  // Convert to enum, asserts if Value cannot be represented as Enum's
  // underlying type.
  template <typename Enum,
            typename std::enable_if_t<std::is_enum<Enum>::value, bool> = 0>
  Enum to() const {
    using type = typename std::underlying_type<Enum>::type;
    return Enum(to<type>());
  }

private:
  static void assertOutOfBounds() { assert(false && "Out of bounds"); }

  intmax_t Value;
};

template <typename T, bool IsReverse> struct SafeIntIterator {
  using iterator_category = std::random_access_iterator_tag;
  using value_type = T;
  using difference_type = intmax_t;
  using pointer = T *;
  using reference = T &;

  // Construct from T.
  explicit SafeIntIterator(T Value) : SI(CheckedInt::from<T>(Value)) {}
  // Construct from other direction.
  SafeIntIterator(const SafeIntIterator<T, !IsReverse> &O) : SI(O.SI) {}

  // Dereference
  value_type operator*() const { return SI.to<T>(); }
  // Indexing
  value_type operator[](intmax_t Offset) const { return *(*this + Offset); }

  // Can be compared for equivalence using the equality/inequality operators.
  bool operator==(const SafeIntIterator &O) const { return SI == O.SI; }
  bool operator!=(const SafeIntIterator &O) const { return SI != O.SI; }
  // Comparison
  bool operator<(const SafeIntIterator &O) const { return (*this - O) < 0; }
  bool operator>(const SafeIntIterator &O) const { return (*this - O) > 0; }
  bool operator<=(const SafeIntIterator &O) const { return (*this - O) <= 0; }
  bool operator>=(const SafeIntIterator &O) const { return (*this - O) >= 0; }

  // Pre Increment/Decrement
  void operator++() { offset(1); }
  void operator--() { offset(-1); }

  // Post Increment/Decrement
  SafeIntIterator operator++(int) {
    const auto Copy = *this;
    ++*this;
    return Copy;
  }
  SafeIntIterator operator--(int) {
    const auto Copy = *this;
    --*this;
    return Copy;
  }

  // Compound assignment operators
  void operator+=(intmax_t Offset) { offset(Offset); }
  void operator-=(intmax_t Offset) { offset(-Offset); }

  // Arithmetic
  SafeIntIterator operator+(intmax_t Offset) const { return add(Offset); }
  SafeIntIterator operator-(intmax_t Offset) const { return add(-Offset); }

  // Difference
  intmax_t operator-(const SafeIntIterator &O) const {
    return IsReverse ? O.SI - SI : SI - O.SI;
  }

private:
  SafeIntIterator(const CheckedInt &SI) : SI(SI) {}

  static intmax_t getOffset(intmax_t Offset) {
    return IsReverse ? -Offset : Offset;
  }

  CheckedInt add(intmax_t Offset) const { return SI + getOffset(Offset); }

  void offset(intmax_t Offset) { SI = SI + getOffset(Offset); }

  CheckedInt SI;

  // To allow construction from the other direction.
  template <typename, bool> friend struct SafeIntIterator;
};

} // namespace detail

template <typename T> struct iota_range {
  using value_type = T;
  using reference = T &;
  using const_reference = const T &;
  using iterator = detail::SafeIntIterator<value_type, false>;
  using const_iterator = iterator;
  using reverse_iterator = detail::SafeIntIterator<value_type, true>;
  using const_reverse_iterator = reverse_iterator;
  using difference_type = intmax_t;
  using size_type = std::size_t;

  explicit iota_range(T Begin, T End, bool Inclusive)
      : BeginValue(Begin), PastEndValue(End) {
    assert(Begin <= End && "Begin must be less or equal to End.");
    if (Inclusive)
      ++PastEndValue;
  }

  size_t size() const { return PastEndValue - BeginValue; }
  bool empty() const { return BeginValue == PastEndValue; }

  auto begin() const { return const_iterator(BeginValue); }
  auto end() const { return const_iterator(PastEndValue); }

  auto rbegin() const { return const_reverse_iterator(PastEndValue - 1); }
  auto rend() const { return const_reverse_iterator(BeginValue - 1); }

private:
  static_assert(std::is_integral<T>::value || std::is_enum<T>::value,
                "T must be an integral or enum type");
  static_assert(std::is_same<T, std::remove_cv_t<T>>::value,
                "T must not be const nor volatile");

  iterator BeginValue;
  iterator PastEndValue;
};

/// Iterate over an integral/enum type from Begin up to - but not including -
/// End.
/// Note on enum iteration: `seq` will generate each consecutive value, even if
/// no enumerator with that value exists.
/// Note: Begin and End values have to be within [INTMAX_MIN, INTMAX_MAX] for
/// forward iteration (resp. [INTMAX_MIN + 1, INTMAX_MAX] for reverse
/// iteration).
template <typename T> auto seq(T Begin, T End) {
  return iota_range<T>(Begin, End, false);
}

/// Iterate over an integral/enum type from Begin to End inclusive.
/// Note on enum iteration: `seq_inclusive` will generate each consecutive
/// value, even if no enumerator with that value exists.
/// Note: Begin and End values have to be within [INTMAX_MIN, INTMAX_MAX - 1]
/// for forward iteration (resp. [INTMAX_MIN + 1, INTMAX_MAX - 1] for reverse
/// iteration).
template <typename T> auto seq_inclusive(T Begin, T End) {
  return iota_range<T>(Begin, End, true);
}

} // end namespace llvm

#endif // LLVM_ADT_SEQUENCE_H
