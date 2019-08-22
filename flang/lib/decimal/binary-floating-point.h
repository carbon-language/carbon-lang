// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FORTRAN_DECIMAL_BINARY_FLOATING_POINT_H_
#define FORTRAN_DECIMAL_BINARY_FLOATING_POINT_H_

// Access and manipulate the fields of an IEEE-754 binary
// floating-point value via a generalized template.

#include <cinttypes>
#include <climits>
#include <cstring>
#include <type_traits>

namespace Fortran::decimal {

template<int BITS> struct HostUnsignedIntTypeHelper {
  using type = std::conditional_t<(BITS <= 8), std::uint8_t,
      std::conditional_t<(BITS <= 16), std::uint16_t,
          std::conditional_t<(BITS <= 32), std::uint32_t,
              std::conditional_t<(BITS <= 64), std::uint64_t, __uint128_t>>>>;
};
template<int BITS>
using HostUnsignedIntType = typename HostUnsignedIntTypeHelper<BITS>::type;

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

template<int PRECISION> struct BinaryFloatingPointNumber {
  static constexpr int precision{PRECISION};
  static constexpr int bits{BitsForPrecision(precision)};
  using RawType = HostUnsignedIntType<bits>;
  static_assert(CHAR_BIT * sizeof(RawType) >= bits);
  static constexpr bool implicitMSB{precision != 64 /*x87*/};
  static constexpr int significandBits{precision - implicitMSB};
  static constexpr int exponentBits{bits - 1 - significandBits};
  static constexpr int maxExponent{(1 << exponentBits) - 1};
  static constexpr int exponentBias{maxExponent / 2};
  static constexpr RawType significandMask{(RawType{1} << significandBits) - 1};

  BinaryFloatingPointNumber() {}  // zero
  BinaryFloatingPointNumber(const BinaryFloatingPointNumber &that) = default;
  BinaryFloatingPointNumber(BinaryFloatingPointNumber &&that) = default;
  BinaryFloatingPointNumber &operator=(
      const BinaryFloatingPointNumber &that) = default;
  BinaryFloatingPointNumber &operator=(
      BinaryFloatingPointNumber &&that) = default;

  template<typename A> explicit constexpr BinaryFloatingPointNumber(A x) {
    static_assert(sizeof raw == sizeof x);
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
  constexpr bool IsNegative() const { return (raw >> (bits - 1)) & 1; }

  constexpr void Negate() { raw ^= RawType{1} << (bits - 1); }

  RawType raw{0};
};
}
#endif
