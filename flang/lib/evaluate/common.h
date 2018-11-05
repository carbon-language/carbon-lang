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

#include "../common/enum-set.h"
#include "../common/fortran.h"
#include "../common/idioms.h"
#include "../common/indirection.h"
#include "../parser/message.h"
#include <cinttypes>

namespace Fortran::evaluate {

using common::RelationalOperator;

// Integers are always ordered; reals may not be.
ENUM_CLASS(Ordering, Less, Equal, Greater)
ENUM_CLASS(Relation, Less, Equal, Greater, Unordered)

template<typename A> static constexpr Ordering Compare(const A &x, const A &y) {
  if (x < y) {
    return Ordering::Less;
  } else if (x > y) {
    return Ordering::Greater;
  } else {
    return Ordering::Equal;
  }
}

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

static constexpr bool Satisfies(RelationalOperator op, Ordering order) {
  switch (order) {
  case Ordering::Less:
    return op == RelationalOperator::LT || op == RelationalOperator::LE ||
        op == RelationalOperator::NE;
  case Ordering::Equal:
    return op == RelationalOperator::LE || op == RelationalOperator::EQ ||
        op == RelationalOperator::GE;
  case Ordering::Greater:
    return op == RelationalOperator::NE || op == RelationalOperator::GE ||
        op == RelationalOperator::GT;
  }
  return false;  // silence g++ warning
}

static constexpr bool Satisfies(RelationalOperator op, Relation relation) {
  switch (relation) {
  case Relation::Less: return Satisfies(op, Ordering::Less);
  case Relation::Equal: return Satisfies(op, Ordering::Equal);
  case Relation::Greater: return Satisfies(op, Ordering::Greater);
  case Relation::Unordered: return false;
  }
  return false;  // silence g++ warning
}

ENUM_CLASS(
    RealFlag, Overflow, DivideByZero, InvalidArgument, Underflow, Inexact)

using RealFlags = common::EnumSet<RealFlag, RealFlag_enumSize>;

template<typename A> struct ValueWithRealFlags {
  A AccumulateFlags(RealFlags &f) {
    f |= flags;
    return value;
  }
  A value;
  RealFlags flags;
};

ENUM_CLASS(Rounding, TiesToEven, ToZero, Down, Up, TiesAwayFromZero)

static constexpr Rounding defaultRounding{Rounding::TiesToEven};

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

// Many classes in this library follow a common paradigm.
// - There is no default constructor (Class() {}), usually to prevent the
//   need for std::monostate as a default constituent in a std::variant<>.
// - There are full copy and move semantics for construction and assignment.
// - Discriminated unions have a std::variant<> member "u" and support
//   explicit copy and move constructors.
#define DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(t) \
  t(const t &) = default; \
  t(t &&) = default; \
  t &operator=(const t &) = default; \
  t &operator=(t &&) = default;

#define CLASS_BOILERPLATE(t) \
  t() = delete; \
  DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(t)

#define EVALUATE_UNION_CLASS_BOILERPLATE(t) \
  CLASS_BOILERPLATE(t) \
  template<typename _A> explicit t(const _A &x) : u{x} {} \
  template<typename _A> \
  explicit t(std::enable_if_t<!std::is_reference_v<_A>, _A> &&x) \
    : u(std::move(x)) {}

// Force availability of copy construction and assignment
template<typename A> using CopyableIndirection = common::Indirection<A, true>;

// Forward definition of Expr<> so that it can be indirectly used in its own
// definition
template<typename A> class Expr;

struct FoldingContext {
  explicit FoldingContext(const parser::ContextualMessages &m,
      Rounding round = defaultRounding, bool flush = false)
    : messages{m}, rounding{round}, flushDenormalsToZero{flush} {}
  FoldingContext(const parser::ContextualMessages &m, const FoldingContext &c)
    : messages{m}, rounding{c.rounding}, flushDenormalsToZero{
                                             c.flushDenormalsToZero} {}

  // For narrowed contexts
  FoldingContext(const FoldingContext &c, const parser::ContextualMessages &m)
    : messages{m}, rounding{c.rounding}, flushDenormalsToZero{
                                             c.flushDenormalsToZero} {}

  parser::ContextualMessages messages;
  Rounding rounding{defaultRounding};
  bool flushDenormalsToZero{false};
  bool bigEndian{false};
};

void RealFlagWarnings(FoldingContext &, const RealFlags &, const char *op);
}
#endif  // FORTRAN_EVALUATE_COMMON_H_
