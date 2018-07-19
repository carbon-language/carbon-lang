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
#include "../common/idioms.h"
#include <string>
#include <variant>

namespace Fortran::evaluate {

ENUM_CLASS(Category, Integer, Real, Complex, Logical, Character, Derived)

template<Category C, int KIND> struct TypeBase {
  static constexpr Category category{C};
  static constexpr int kind{KIND};
  static constexpr bool hasLen{false};
  static std::string Dump() {
    return EnumToString(category) + '(' + std::to_string(kind) + ')';
  }
};

template<Category C, int KIND> struct Type;

template<int KIND>
struct Type<Category::Integer, KIND>
  : public TypeBase<Category::Integer, KIND> {
  using Value = value::Integer<8 * KIND>;
};

template<> struct Type<Category::Real, 2> : public TypeBase<Category::Real, 2> {
  using Value = value::Real<typename Type<Category::Integer, 2>::Value, 11>;
  using Complex = Type<Category::Complex, 2>;
};

template<> struct Type<Category::Real, 4> : public TypeBase<Category::Real, 4> {
  using Value = value::Real<typename Type<Category::Integer, 4>::Value, 24>;
  using Complex = Type<Category::Complex, 2>;
};

template<> struct Type<Category::Real, 8> : public TypeBase<Category::Real, 8> {
  using Value = value::Real<typename Type<Category::Integer, 8>::Value, 53>;
  using Complex = Type<Category::Complex, 2>;
};

template<>
struct Type<Category::Real, 10> : public TypeBase<Category::Real, 10> {
  using Value = value::Real<value::Integer<80>, 64, false>;
  using Complex = Type<Category::Complex, 2>;
};

template<>
struct Type<Category::Real, 16> : public TypeBase<Category::Real, 16> {
  using Value = value::Real<typename Type<Category::Integer, 16>::Value, 112>;
  using Complex = Type<Category::Complex, 2>;
};

// The KIND type parameter on COMPLEX is the kind of each of its components.
template<int KIND>
struct Type<Category::Complex, KIND>
  : public TypeBase<Category::Complex, KIND> {
  using Part = Type<Category::Real, KIND>;
  using Value = value::Complex<typename Part::Value>;
};

template<int KIND>
struct Type<Category::Logical, KIND>
  : public TypeBase<Category::Logical, KIND> {
  using Value = value::Logical<8 * KIND>;
};

template<int KIND> struct Type<Category::Character, KIND> {
  static constexpr Category category{Category::Character};
  static constexpr int kind{KIND};
  static constexpr bool hasLen{true};
  using Value = std::string;
  static std::string Dump() {
    return EnumToString(category) + '(' + std::to_string(kind) + ')';
  }
};

// Default REAL just simply has to be IEEE-754 single precision today.
// It occupies one numeric storage unit by definition.  The default INTEGER
// and default LOGICAL intrinsic types also have to occupy one numeric
// storage unit, so their kinds are also forced.  Default COMPLEX occupies
// two numeric storage units.
// TODO: Support a compile-time option to default everything to KIND=8

using DefaultReal = Type<Category::Real, 4>;
using DefaultDoublePrecision = Type<Category::Real, 2 * DefaultReal::kind>;
using DefaultInteger = Type<Category::Integer, DefaultReal::kind>;
using IntrinsicTypeParameterType = DefaultInteger;
using DefaultComplex = typename DefaultReal::Complex;
using DefaultLogical = Type<Category::Logical, DefaultInteger::kind>;
using DefaultCharacter = Type<Category::Character, 1>;

using SubscriptInteger = Type<Category::Integer, 8>;

// These macros invoke other macros on each of the supported kinds of
// a given category.
#define COMMA ,
#define FOR_EACH_INTEGER_KIND(M,SEP) M(1) SEP M(2) SEP M(4) SEP M(8) SEP M(16)
#define FOR_EACH_REAL_KIND(M,SEP) M(2) SEP M(4) SEP M(8) SEP M(10) SEP M(16)
#define FOR_EACH_COMPLEX_KIND(M,SEP) FOR_EACH_REAL_KIND(M,SEP)
#define FOR_EACH_CHARACTER_KIND(M,SEP) M(1)
#define FOR_EACH_LOGICAL_KIND(M,SEP) M(1) SEP M(2) SEP M(4) SEP M(8)

// These templates create instances of std::variant<> that can contain
// applications of some class template to all of the supported kinds of
// a category of intrinsic type.
#define TKIND(K) T<K>
template<Category CAT, template<int> class T> struct KindsVariant;
template<template<int> class T> struct KindsVariant<Category::Integer, T> {
  using type = std::variant<FOR_EACH_INTEGER_KIND(TKIND,COMMA)>;
};
// TODO use FOR_EACH...
template<template<int> class T> struct KindsVariant<Category::Real, T> {
  using type = std::variant<T<2>, T<4>, T<8>, T<10>, T<16>>;
};
template<template<int> class T> struct KindsVariant<Category::Complex, T> {
  using type = typename KindsVariant<Category::Real, T>::type;
};
template<template<int> class T> struct KindsVariant<Category::Character, T> {
  using type = std::variant<T<1>>;  // TODO larger CHARACTER kinds, incl. Kanji
};
template<template<int> class T> struct KindsVariant<Category::Logical, T> {
  using type = std::variant<T<1>, T<2>, T<4>, T<8>>;
};
#undef TKIND
}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_TYPE_H_
