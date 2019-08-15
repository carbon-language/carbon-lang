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

#ifndef FORTRAN_EVALUATE_COMMON_H_
#define FORTRAN_EVALUATE_COMMON_H_

#include "intrinsics-library.h"
#include "../common/Fortran.h"
#include "../common/default-kinds.h"
#include "../common/enum-set.h"
#include "../common/idioms.h"
#include "../common/indirection.h"
#include "../common/restorer.h"
#include "../parser/char-block.h"
#include "../parser/message.h"
#include <cinttypes>
#include <map>

namespace Fortran::semantics {
class DerivedTypeSpec;
}

namespace Fortran::evaluate {

using common::ConstantSubscript;
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
  RealFlags flags{};
};

ENUM_CLASS(RoundingMode, TiesToEven, ToZero, Down, Up, TiesAwayFromZero)

struct Rounding {
  RoundingMode mode{RoundingMode::TiesToEven};
  // When set, emulate status flag behavior peculiar to x86
  // (viz., fail to set the Underflow flag when an inexact product of a
  // multiplication is rounded up to a normal number from a subnormal
  // in some rounding modes)
#if __x86_64__
  bool x86CompatibleBehavior{true};
#else
  bool x86CompatibleBehavior{false};
#endif
};

static constexpr Rounding defaultRounding;

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
//   explicit copy and move constructors as well as comparison for equality.
#define DECLARE_CONSTRUCTORS_AND_ASSIGNMENTS(t) \
  t(const t &); \
  t(t &&); \
  t &operator=(const t &); \
  t &operator=(t &&);
#define DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(t) \
  t(const t &) = default; \
  t(t &&) = default; \
  t &operator=(const t &) = default; \
  t &operator=(t &&) = default;
#define DEFINE_DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(t) \
  t::t(const t &) = default; \
  t::t(t &&) = default; \
  t &t::operator=(const t &) = default; \
  t &t::operator=(t &&) = default;
#define CONSTEXPR_CONSTRUCTORS_AND_ASSIGNMENTS(t) \
  constexpr t(const t &) = default; \
  constexpr t(t &&) = default; \
  constexpr t &operator=(const t &) = default; \
  constexpr t &operator=(t &&) = default;

#define CLASS_BOILERPLATE(t) \
  t() = delete; \
  DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(t)

#define EVALUATE_UNION_CLASS_BOILERPLATE(t) \
  CLASS_BOILERPLATE(t) \
  template<typename _A> explicit t(const _A &x) : u{x} {} \
  template<typename _A, typename = common::NoLvalue<_A>> \
  explicit t(_A &&x) : u(std::move(x)) {} \
  bool operator==(const t &that) const { return u == that.u; }

// Forward definition of Expr<> so that it can be indirectly used in its own
// definition
template<typename A> class Expr;

class FoldingContext {
public:
  explicit FoldingContext(const common::IntrinsicTypeDefaultKinds &d)
    : defaults_{d} {}
  FoldingContext(const parser::ContextualMessages &m,
      const common::IntrinsicTypeDefaultKinds &d,
      Rounding round = defaultRounding, bool flush = false)
    : messages_{m}, defaults_{d}, rounding_{round}, flushSubnormalsToZero_{
                                                        flush} {}
  FoldingContext(const FoldingContext &that)
    : messages_{that.messages_}, defaults_{that.defaults_},
      rounding_{that.rounding_},
      flushSubnormalsToZero_{that.flushSubnormalsToZero_},
      pdtInstance_{that.pdtInstance_}, impliedDos_{that.impliedDos_} {}
  FoldingContext(
      const FoldingContext &that, const parser::ContextualMessages &m)
    : messages_{m}, defaults_{that.defaults_}, rounding_{that.rounding_},
      flushSubnormalsToZero_{that.flushSubnormalsToZero_},
      pdtInstance_{that.pdtInstance_}, impliedDos_{that.impliedDos_} {}

  parser::ContextualMessages &messages() { return messages_; }
  const common::IntrinsicTypeDefaultKinds &defaults() { return defaults_; }
  Rounding rounding() const { return rounding_; }
  bool flushSubnormalsToZero() const { return flushSubnormalsToZero_; }
  bool bigEndian() const { return bigEndian_; }
  const semantics::DerivedTypeSpec *pdtInstance() const { return pdtInstance_; }
  HostIntrinsicProceduresLibrary &hostIntrinsicsLibrary() {
    return hostIntrinsicsLibrary_;
  }

  ConstantSubscript &StartImpliedDo(parser::CharBlock, ConstantSubscript = 1);
  std::optional<ConstantSubscript> GetImpliedDo(parser::CharBlock) const;
  void EndImpliedDo(parser::CharBlock);

  std::map<parser::CharBlock, ConstantSubscript> &impliedDos() {
    return impliedDos_;
  }

  common::Restorer<const semantics::DerivedTypeSpec *> WithPDTInstance(
      const semantics::DerivedTypeSpec &spec) {
    return common::ScopedSet(pdtInstance_, &spec);
  }

private:
  parser::ContextualMessages messages_;
  const common::IntrinsicTypeDefaultKinds &defaults_;
  Rounding rounding_{defaultRounding};
  bool flushSubnormalsToZero_{false};
  bool bigEndian_{false};
  const semantics::DerivedTypeSpec *pdtInstance_{nullptr};
  std::map<parser::CharBlock, ConstantSubscript> impliedDos_;
  HostIntrinsicProceduresLibrary hostIntrinsicsLibrary_;
};

void RealFlagWarnings(FoldingContext &, const RealFlags &, const char *op);
}
#endif  // FORTRAN_EVALUATE_COMMON_H_
