//===-- lib/evaluate/logical.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//----------------------------------------------------------------------------//

#ifndef FORTRAN_EVALUATE_LOGICAL_H_
#define FORTRAN_EVALUATE_LOGICAL_H_

#include "integer.h"
#include <cinttypes>

namespace Fortran::evaluate::value {

template<int BITS, bool IS_LIKE_C = false> class Logical {
public:
  static constexpr int bits{BITS};

  // Module ISO_C_BINDING kind C_BOOL is LOGICAL(KIND=1) and must have
  // C's bit representation (.TRUE. -> 1, .FALSE. -> 0).
  static constexpr bool IsLikeC{BITS <= 8 || IS_LIKE_C};

  constexpr Logical() {}  // .FALSE.
  constexpr Logical(const Logical &that) = default;
  constexpr Logical(bool truth)
    : word_{truth ? canonicalTrue : canonicalFalse} {}
  constexpr Logical &operator=(const Logical &) = default;

  template<int B> constexpr bool operator==(const Logical<B> &that) const {
    return IsTrue() == that.IsTrue();
  }

  // For static expression evaluation, all the bits will have the same value.
  constexpr bool IsTrue() const {
    if constexpr (IsLikeC) {
      return !word_.IsZero();
    } else {
      return word_.BTEST(0);
    }
  }

  constexpr Logical NOT() const { return {word_.IEOR(canonicalTrue)}; }

  constexpr Logical AND(const Logical &that) const {
    return {word_.IAND(that.word_)};
  }

  constexpr Logical OR(const Logical &that) const {
    return {word_.IOR(that.word_)};
  }

  constexpr Logical EQV(const Logical &that) const { return NEQV(that).NOT(); }

  constexpr Logical NEQV(const Logical &that) const {
    return {word_.IEOR(that.word_)};
  }

private:
  using Word = Integer<bits>;
  static constexpr Word canonicalTrue{IsLikeC ? -std::uint64_t{1} : 1};
  static constexpr Word canonicalFalse{0};
  constexpr Logical(const Word &w) : word_{w} {}
  Word word_;
};

extern template class Logical<8>;
extern template class Logical<16>;
extern template class Logical<32>;
extern template class Logical<64>;
}
#endif  // FORTRAN_EVALUATE_LOGICAL_H_
