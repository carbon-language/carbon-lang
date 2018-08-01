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
#include <string>
#include <variant>

namespace Fortran::evaluate {

using common::TypeCategory;

template<TypeCategory C, int KIND> struct TypeBase {
  static constexpr TypeCategory category{C};
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
  using Value = value::Integer<8 * KIND>;
};

template<>
struct Type<TypeCategory::Real, 2> : public TypeBase<TypeCategory::Real, 2> {
  using Value = value::Real<typename Type<TypeCategory::Integer, 2>::Value, 11>;
  using Complex = Type<TypeCategory::Complex, 2>;
};

template<>
struct Type<TypeCategory::Real, 4> : public TypeBase<TypeCategory::Real, 4> {
  using Value = value::Real<typename Type<TypeCategory::Integer, 4>::Value, 24>;
  using Complex = Type<TypeCategory::Complex, 2>;
};

template<>
struct Type<TypeCategory::Real, 8> : public TypeBase<TypeCategory::Real, 8> {
  using Value = value::Real<typename Type<TypeCategory::Integer, 8>::Value, 53>;
  using Complex = Type<TypeCategory::Complex, 2>;
};

template<>
struct Type<TypeCategory::Real, 10> : public TypeBase<TypeCategory::Real, 10> {
  using Value = value::Real<value::Integer<80>, 64, false>;
  using Complex = Type<TypeCategory::Complex, 2>;
};

template<>
struct Type<TypeCategory::Real, 16> : public TypeBase<TypeCategory::Real, 16> {
  using Value =
      value::Real<typename Type<TypeCategory::Integer, 16>::Value, 112>;
  using Complex = Type<TypeCategory::Complex, 2>;
};

// The KIND type parameter on COMPLEX is the kind of each of its components.
template<int KIND>
struct Type<TypeCategory::Complex, KIND>
  : public TypeBase<TypeCategory::Complex, KIND> {
  using Part = Type<TypeCategory::Real, KIND>;
  using Value = value::Complex<typename Part::Value>;
};

template<int KIND> struct Type<TypeCategory::Character, KIND> {
  static constexpr TypeCategory category{TypeCategory::Character};
  static constexpr int kind{KIND};
  static constexpr bool hasLen{true};
  using Value = std::string;
  static std::string Dump() {
    return EnumToString(category) + '(' + std::to_string(kind) + ')';
  }
};

template<int KIND>
struct Type<TypeCategory::Logical, KIND>
  : public TypeBase<TypeCategory::Logical, KIND> {
  using Value = value::Logical<8 * KIND>;
};

// Default REAL just simply has to be IEEE-754 single precision today.
// It occupies one numeric storage unit by definition.  The default INTEGER
// and default LOGICAL intrinsic types also have to occupy one numeric
// storage unit, so their kinds are also forced.  Default COMPLEX occupies
// two numeric storage units.
// TODO: Support a compile-time option to default everything to KIND=8

using DefaultReal = Type<TypeCategory::Real, 4>;
using DefaultDoublePrecision = Type<TypeCategory::Real, 2 * DefaultReal::kind>;
using DefaultInteger = Type<TypeCategory::Integer, DefaultReal::kind>;
using IntrinsicTypeParameterType = DefaultInteger;
using DefaultComplex = typename DefaultReal::Complex;
using DefaultLogical = Type<TypeCategory::Logical, DefaultInteger::kind>;
using DefaultCharacter = Type<TypeCategory::Character, 1>;

using SubscriptInteger = Type<TypeCategory::Integer, 8>;

// These macros invoke other macros on each of the supported kinds of
// a given category.
// TODO larger CHARACTER kinds, incl. Kanji
#define COMMA ,
#define FOR_EACH_INTEGER_KIND(M, SEP) M(1) SEP M(2) SEP M(4) SEP M(8) SEP M(16)
#define FOR_EACH_REAL_KIND(M, SEP) M(2) SEP M(4) SEP M(8) SEP M(10) SEP M(16)
#define FOR_EACH_COMPLEX_KIND(M, SEP) M(2) SEP M(4) SEP M(8) SEP M(10) SEP M(16)
#define FOR_EACH_CHARACTER_KIND(M, SEP) M(1)
#define FOR_EACH_LOGICAL_KIND(M, SEP) M(1) SEP M(2) SEP M(4) SEP M(8)

// These templates create instances of std::variant<> that can contain
// applications of some class template to all of the supported kinds of
// a category of intrinsic type.
#define TKIND(K) T<K>
template<TypeCategory CAT, template<int> class T> struct KindsVariant;
template<template<int> class T> struct KindsVariant<TypeCategory::Integer, T> {
  using type = std::variant<FOR_EACH_INTEGER_KIND(TKIND, COMMA)>;
};
template<template<int> class T> struct KindsVariant<TypeCategory::Real, T> {
  using type = std::variant<FOR_EACH_REAL_KIND(TKIND, COMMA)>;
};
template<template<int> class T> struct KindsVariant<TypeCategory::Complex, T> {
  using type = std::variant<FOR_EACH_COMPLEX_KIND(TKIND, COMMA)>;
};
template<template<int> class T>
struct KindsVariant<TypeCategory::Character, T> {
  using type = std::variant<FOR_EACH_CHARACTER_KIND(TKIND, COMMA)>;
};
template<template<int> class T> struct KindsVariant<TypeCategory::Logical, T> {
  using type = std::variant<FOR_EACH_LOGICAL_KIND(TKIND, COMMA)>;
};
#undef TKIND

// Holds a scalar constant of any kind within a particular intrinsic type
// category.
template<TypeCategory CAT> struct ScalarConstant {
  CLASS_BOILERPLATE(ScalarConstant)
  template<int KIND> using KindScalar = typename Type<CAT, KIND>::Value;
  template<typename A> ScalarConstant(const A &x) : u{x} {}
  template<typename A>
  ScalarConstant(std::enable_if_t<!std::is_reference_v<A>, A> &&x)
    : u{std::move(x)} {}
  typename KindsVariant<CAT, KindScalar>::type u;
};

// Holds a scalar constant of any intrinsic category and size.
struct GenericScalar {
  CLASS_BOILERPLATE(GenericScalar)
  template<TypeCategory CAT, int KIND>
  GenericScalar(const typename Type<CAT, KIND>::Value &x)
    : u{ScalarConstant<CAT>{x}} {}
  template<TypeCategory CAT, int KIND>
  GenericScalar(typename Type<CAT, KIND>::Value &&x)
    : u{ScalarConstant<CAT>{std::move(x)}} {}
  template<typename A> GenericScalar(const A &x) : u{x} {}
  template<typename A>
  GenericScalar(std::enable_if_t<!std::is_reference_v<A>, A> &&x)
    : u{std::move(x)} {}
  std::variant<ScalarConstant<TypeCategory::Integer>,
      ScalarConstant<TypeCategory::Real>, ScalarConstant<TypeCategory::Complex>,
      ScalarConstant<TypeCategory::Character>,
      ScalarConstant<TypeCategory::Logical>>
      u;
};

// Represents a type of any supported kind within a particular category.
template<TypeCategory CAT> struct AnyKindType {
  static constexpr TypeCategory category{CAT};
  using Value = ScalarConstant<CAT>;
};
}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_TYPE_H_
