//===-- lib/decimal/binary-floating-point.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_DECIMAL_BINARY_FLOATING_POINT_H_
#define FORTRAN_DECIMAL_BINARY_FLOATING_POINT_H_

// Access and manipulate the fields of an IEEE-754 binary
// floating-point value via a generalized template.

#include "../common/uint128.h"
#include <cinttypes>
#include <climits>
#include <cstring>
#include <type_traits>

namespace Fortran::decimal {

static constexpr int BitsForPrecision(int prec) {
  switch (prec) {
  case 8: return 16;
  case 11: return 16;
  case 24: return 32;
  case 53: return 64;
  case 64: return 80;
  case 112: return 128;
  default: return -1;
  }
}

// LOG10(2.)*1E12
static constexpr std::int64_t ScaledLogBaseTenOfTwo{301029995664};

template<int PRECISION> struct BinaryFloatingPointNumber {
  static constexpr int precision{PRECISION};
  static constexpr int bits{BitsForPrecision(precision)};
  using RawType = common::HostUnsignedIntType<bits>;
  static_assert(CHAR_BIT * sizeof(RawType) >= bits);
  static constexpr bool implicitMSB{precision != 64 /*x87*/};
  static constexpr int significandBits{precision - implicitMSB};
  static constexpr int exponentBits{bits - 1 - significandBits};
  static constexpr int maxExponent{(1 << exponentBits) - 1};
  static constexpr int exponentBias{maxExponent / 2};
  static constexpr RawType significandMask{(RawType{1} << significandBits) - 1};
  static constexpr int RANGE{static_cast<int>(
      (exponentBias - 1) * ScaledLogBaseTenOfTwo / 1000000000000)};

  constexpr BinaryFloatingPointNumber() {}  // zero
  constexpr BinaryFloatingPointNumber(
      const BinaryFloatingPointNumber &that) = default;
  constexpr BinaryFloatingPointNumber(
      BinaryFloatingPointNumber &&that) = default;
  constexpr BinaryFloatingPointNumber &operator=(
      const BinaryFloatingPointNumber &that) = default;
  constexpr BinaryFloatingPointNumber &operator=(
      BinaryFloatingPointNumber &&that) = default;

  template<typename A> explicit constexpr BinaryFloatingPointNumber(A x) {
    static_assert(sizeof raw <= sizeof x);
    std::memcpy(reinterpret_cast<void *>(&raw),
        reinterpret_cast<const void *>(&x), sizeof raw);
  }

  constexpr int BiasedExponent() const {
    return static_cast<int>(
        (raw >> significandBits) & ((1 << exponentBits) - 1));
  }
  constexpr int UnbiasedExponent() const {
    int biased{BiasedExponent()};
    return biased - exponentBias + (biased == 0);
  }
  constexpr RawType Significand() const { return raw & significandMask; }
  constexpr RawType Fraction() const {
    RawType sig{Significand()};
    if (implicitMSB && BiasedExponent() > 0) {
      sig |= RawType{1} << significandBits;
    }
    return sig;
  }

  constexpr bool IsZero() const {
    return (raw & ((RawType{1} << (bits - 1)) - 1)) == 0;
  }
  constexpr bool IsNaN() const {
    return BiasedExponent() == maxExponent && Significand() != 0;
  }
  constexpr bool IsInfinite() const {
    return BiasedExponent() == maxExponent && Significand() == 0;
  }
  constexpr bool IsMaximalFiniteMagnitude() const {
    return BiasedExponent() == maxExponent - 1 &&
        Significand() == significandMask;
  }
  constexpr bool IsNegative() const { return ((raw >> (bits - 1)) & 1) != 0; }

  constexpr void Negate() { raw ^= RawType{1} << (bits - 1); }

  RawType raw{0};
};
}
#endif
