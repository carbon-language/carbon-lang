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

#ifndef FORTRAN_EVALUATE_TYPE_H_
#define FORTRAN_EVALUATE_TYPE_H_

// These definitions map Fortran's intrinsic types, characterized by byte
// sizes encoded in KIND type parameter values, to their value representation
// types in the evaluation library, which are parameterized in terms of
// total bit width and real precision.

#include "complex.h"
#include "integer.h"
#include "logical.h"
#include "real.h"
#include "../common/fortran.h"
#include "../common/idioms.h"
#include <cinttypes>
#include <optional>
#include <string>
#include <variant>

namespace Fortran::evaluate {

using common::TypeCategory;

template<TypeCategory C, int KIND> struct TypeBase {
  static constexpr TypeCategory category{C};
  static constexpr TypeCategory GetCategory() { return C; };
  static constexpr int kind{KIND};
  static constexpr bool hasLen{false};
  static std::string Dump() {
    return EnumToString(category) + '(' + std::to_string(kind) + ')';
  }
};

template<TypeCategory C, int KIND> struct Type;

template<int KIND>
struct Type<TypeCategory::Integer, KIND>
  : public TypeBase<TypeCategory::Integer, KIND> {
  using Scalar = value::Integer<8 * KIND>;
};

template<>
struct Type<TypeCategory::Real, 2> : public TypeBase<TypeCategory::Real, 2> {
  using Scalar =
      value::Real<typename Type<TypeCategory::Integer, 2>::Scalar, 11>;
};

template<>
struct Type<TypeCategory::Real, 4> : public TypeBase<TypeCategory::Real, 4> {
  using Scalar =
      value::Real<typename Type<TypeCategory::Integer, 4>::Scalar, 24>;
};

template<>
struct Type<TypeCategory::Real, 8> : public TypeBase<TypeCategory::Real, 8> {
  using Scalar =
      value::Real<typename Type<TypeCategory::Integer, 8>::Scalar, 53>;
};

template<>
struct Type<TypeCategory::Real, 10> : public TypeBase<TypeCategory::Real, 10> {
  using Scalar = value::Real<value::Integer<80>, 64, false>;
};

template<>
struct Type<TypeCategory::Real, 16> : public TypeBase<TypeCategory::Real, 16> {
  using Scalar = value::Real<value::Integer<128>, 112>;
};

// The KIND type parameter on COMPLEX is the kind of each of its components.
template<int KIND>
struct Type<TypeCategory::Complex, KIND>
  : public TypeBase<TypeCategory::Complex, KIND> {
  using Part = Type<TypeCategory::Real, KIND>;
  using Scalar = value::Complex<typename Part::Scalar>;
};

template<int KIND> struct Type<TypeCategory::Character, KIND> {
  static constexpr TypeCategory category{TypeCategory::Character};
  static constexpr int kind{KIND};
  static constexpr bool hasLen{true};
  using Scalar = std::string;
  static std::string Dump() {
    return EnumToString(category) + '(' + std::to_string(kind) + ')';
  }
};

template<int KIND>
struct Type<TypeCategory::Logical, KIND>
  : public TypeBase<TypeCategory::Logical, KIND> {
  using Scalar = value::Logical<8 * KIND>;
};

// Type functions
template<typename T> using Scalar = typename std::decay_t<T>::Scalar;

template<TypeCategory C, typename T>
using SameKind = Type<C, std::decay_t<T>::kind>;

// Convenience type aliases:
// Default REAL just simply has to be IEEE-754 single precision today.
// It occupies one numeric storage unit by definition.  The default INTEGER
// and default LOGICAL intrinsic types also have to occupy one numeric
// storage unit, so their kinds are also forced.  Default COMPLEX occupies
// two numeric storage units.
// TODO: Support compile-time options to default reals, ints, or both to KIND=8

using DefaultReal = Type<TypeCategory::Real, 4>;
using DefaultDoublePrecision = Type<TypeCategory::Real, 2 * DefaultReal::kind>;
using DefaultInteger = Type<TypeCategory::Integer, DefaultReal::kind>;
using IntrinsicTypeParameterType = DefaultInteger;
using DefaultComplex = SameKind<TypeCategory::Complex, DefaultReal>;
using DefaultLogical = Type<TypeCategory::Logical, DefaultInteger::kind>;
using DefaultCharacter = Type<TypeCategory::Character, 1>;

using SubscriptInteger = Type<TypeCategory::Integer, 8>;
using LogicalResult = Type<TypeCategory::Logical, 1>;

// These macros invoke other macros on each of the supported kinds of
// a given category.
// TODO larger CHARACTER kinds, incl. Kanji
#define COMMA ,
#define FOR_EACH_INTEGER_KIND(M, SEP) M(1) SEP M(2) SEP M(4) SEP M(8) SEP M(16)
#define FOR_EACH_REAL_KIND(M, SEP) M(2) SEP M(4) SEP M(8) SEP M(10) SEP M(16)
#define FOR_EACH_COMPLEX_KIND(M, SEP) M(2) SEP M(4) SEP M(8) SEP M(10) SEP M(16)
#define FOR_EACH_CHARACTER_KIND(M, SEP) M(1)
#define FOR_EACH_LOGICAL_KIND(M, SEP) M(1) SEP M(2) SEP M(4) SEP M(8)

#define FOR_EACH_CATEGORY(M) \
  M(Integer, INTEGER) \
  M(Real, REAL) M(Complex, COMPLEX) M(Character, CHARACTER) M(Logical, LOGICAL)

// These macros and template create instances of std::variant<> that can contain
// applications of some class template to all of the supported kinds of
// a category of intrinsic type.
template<TypeCategory CAT, template<int> class T> struct KindsVariant;
#define TKIND(K) T<K>
#define MAKE(Cat, CAT) \
  template<template<int> class T> struct KindsVariant<TypeCategory::Cat, T> { \
    using type = std::variant<FOR_EACH_##CAT##_KIND(TKIND, COMMA)>; \
  };
FOR_EACH_CATEGORY(MAKE)
#undef MAKE
#undef TKIND

// Map scalar value types back to their Fortran types.
// For every type T = Type<CAT, KIND>, TypeOfScalarValue<T>> == T.
// E.g., TypeOfScalarValue<Integer<32>> is Type<TypeCategory::Integer, 4>.
template<typename CONST> struct GetTypeOfScalarValue;
#define TOSV(cat, kind) \
  template<> \
  struct GetTypeOfScalarValue<Scalar<Type<TypeCategory::cat, kind>>> { \
    using type = Type<TypeCategory::cat, kind>; \
  };
#define M(k) TOSV(Integer, k)
FOR_EACH_INTEGER_KIND(M, )
#undef M
#define M(k) TOSV(Real, k)
FOR_EACH_REAL_KIND(M, )
#undef M
#define M(k) TOSV(Complex, k)
FOR_EACH_COMPLEX_KIND(M, )
#undef M
#define M(k) TOSV(Character, k)
FOR_EACH_CHARACTER_KIND(M, )
#undef M
#define M(k) TOSV(Logical, k)
FOR_EACH_LOGICAL_KIND(M, )
#undef M
#undef TOSV

template<typename CONST>
using ScalarValueType =
    typename GetTypeOfScalarValue<std::decay_t<CONST>>::type;

// Holds a scalar value of any kind within a particular intrinsic type
// category.
template<TypeCategory CAT> struct SomeKindScalar {
  static constexpr TypeCategory category{CAT};
  CLASS_BOILERPLATE(SomeKindScalar)

  template<int KIND> using KindScalar = Scalar<Type<CAT, KIND>>;
  template<typename A> SomeKindScalar(const A &x) : u{x} {}
  template<typename A>
  SomeKindScalar(std::enable_if_t<!std::is_reference_v<A>, A> &&x)
    : u{std::move(x)} {}

  std::optional<std::int64_t> ToInt64() const {
    if constexpr (category == TypeCategory::Integer) {
      return std::visit(
          [](const auto &x) { return std::make_optional(x.ToInt64()); }, u);
    }
    return std::nullopt;
  }

  std::optional<std::string> ToString() const {
    return common::GetIf<std::string>(u);
  }

  typename KindsVariant<CAT, KindScalar>::type u;
};

// Holds a scalar constant of any intrinsic category and size.
struct GenericScalar {
  CLASS_BOILERPLATE(GenericScalar)

  template<TypeCategory CAT, int KIND>
  GenericScalar(const Scalar<Type<CAT, KIND>> &x) : u{SomeKindScalar<CAT>{x}} {}
  template<TypeCategory CAT, int KIND>
  GenericScalar(Scalar<Type<CAT, KIND>> &&x)
    : u{SomeKindScalar<CAT>{std::move(x)}} {}

  template<typename A> GenericScalar(const A &x) : u{x} {}
  template<typename A>
  GenericScalar(std::enable_if_t<!std::is_reference_v<A>, A> &&x)
    : u{std::move(x)} {}

  std::optional<std::int64_t> ToInt64() const {
    if (const auto *j{std::get_if<SomeKindScalar<TypeCategory::Integer>>(&u)}) {
      return j->ToInt64();
    }
    return std::nullopt;
  }

  std::optional<std::string> ToString() const {
    if (const auto *c{
            std::get_if<SomeKindScalar<TypeCategory::Character>>(&u)}) {
      return c->ToString();
    }
    return std::nullopt;
  }

  std::variant<SomeKindScalar<TypeCategory::Integer>,
      SomeKindScalar<TypeCategory::Real>, SomeKindScalar<TypeCategory::Complex>,
      SomeKindScalar<TypeCategory::Character>,
      SomeKindScalar<TypeCategory::Logical>>
      u;
};

// Represents a type of any supported kind within a particular category.
template<TypeCategory CAT> struct SomeKind {
  static constexpr TypeCategory category{CAT};
  using Scalar = SomeKindScalar<CAT>;
};

using SomeInteger = SomeKind<TypeCategory::Integer>;
using SomeReal = SomeKind<TypeCategory::Real>;
using SomeComplex = SomeKind<TypeCategory::Complex>;
using SomeCharacter = SomeKind<TypeCategory::Character>;
using SomeLogical = SomeKind<TypeCategory::Logical>;

// Represents a completely generic type.
struct SomeType {
  using Scalar = GenericScalar;
};

}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_TYPE_H_
