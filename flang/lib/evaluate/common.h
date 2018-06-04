// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef FORTRAN_EVALUATE_COMMON_H_
#define FORTRAN_EVALUATE_COMMON_H_

#include <cinttypes>

namespace Fortran::evaluate {

// Integers are always ordered; reals may not be.
enum class Ordering { Less, Equal, Greater };
enum class Relation { Less, Equal, Greater, Unordered };

static constexpr Ordering Reverse(Ordering ordering) {
  if (ordering == Ordering::Less) {
    return Ordering::Greater;
  } else if (ordering == Ordering::Greater) {
    return Ordering::Less;
  } else {
    return Ordering::Equal;
  }
}

static constexpr Relation RelationFromOrdering(Ordering ordering) {
  if (ordering == Ordering::Less) {
    return Relation::Less;
  } else if (ordering == Ordering::Greater) {
    return Relation::Greater;
  } else {
    return Relation::Equal;
  }
}

static constexpr Relation Reverse(Relation relation) {
  if (relation == Relation::Less) {
    return Relation::Greater;
  } else if (relation == Relation::Greater) {
    return Relation::Less;
  } else {
    return relation;
  }
}

namespace RealFlag {
enum {
  Ok = 0, Overflow = 1, DivideByZero = 2, InvalidArgument = 4,
  Underflow = 8, Inexact = 16
};
}  // namespace RealFlag

enum class Rounding {
  TiesToEven, ToZero, Down, Up, TiesAwayFromZero
};

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
constexpr bool IsHostLittleEndian{false};
#elif __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
constexpr bool IsHostLittleEndian{true};
#else
#error host endianness is not known
#endif

// HostUnsignedInt<BITS> finds the smallest native unsigned integer type
// whose size is >= BITS.
template<bool LE8, bool LE16, bool LE32, bool LE64> struct SmallestUInt {};
template<> struct SmallestUInt<true, true, true, true> {
  using type = std::uint8_t;
};
template<> struct SmallestUInt<false, true, true, true> {
  using type = std::uint16_t;
};
template<> struct SmallestUInt<false, false, true, true> {
  using type = std::uint32_t;
};
template<> struct SmallestUInt<false, false, false, true> {
  using type = std::uint64_t;
};
template<int BITS>
using HostUnsignedInt =
    typename SmallestUInt<BITS <= 8, BITS <= 16, BITS <= 32, BITS <= 64>::type;

}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_COMMON_H_
