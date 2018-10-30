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
// total bit width and real precision.  Instances of these class templates
// are suitable for use as template parameters to instantiate other class
// templates, like expressions, over the supported types and kinds.

#include "common.h"
#include "complex.h"
#include "integer.h"
#include "logical.h"
#include "real.h"
#include "../common/fortran.h"
#include "../common/idioms.h"
#include "../common/template.h"
#include <cinttypes>
#include <optional>
#include <string>
#include <variant>

namespace Fortran::semantics {
class DerivedTypeSpec;
class Symbol;
}

namespace Fortran::evaluate {

using common::TypeCategory;

struct DynamicType {
  bool operator==(const DynamicType &that) const {
    return category == that.category && kind == that.kind &&
        derived == that.derived;
  }
  std::string Dump() const {
    return EnumToString(category) + '(' + std::to_string(kind) + ')';
  }

  DynamicType ResultTypeForMultiply(const DynamicType &) const;

  TypeCategory category;
  int kind{0};
  const semantics::DerivedTypeSpec *derived{nullptr};
};

std::optional<DynamicType> GetSymbolType(const semantics::Symbol &);

// Specific intrinsic types are represented by specializations of
// this class template Type<CATEGORY, KIND>.
template<TypeCategory CATEGORY, int KIND = 0> class Type;

template<TypeCategory CATEGORY, int KIND> struct TypeBase {
  // Only types that represent a known kind of one of the five intrinsic
  // data types will have set this flag to true.
  static constexpr bool isSpecificIntrinsicType{true};
  static constexpr DynamicType dynamicType{CATEGORY, KIND};
  static constexpr std::optional<DynamicType> GetType() {
    return {dynamicType};
  }
  static constexpr TypeCategory category{CATEGORY};
  static constexpr int kind{KIND};
  static std::string Dump() { return dynamicType.Dump(); }
};

template<int KIND>
class Type<TypeCategory::Integer, KIND>
  : public TypeBase<TypeCategory::Integer, KIND> {
public:
  using Scalar = value::Integer<8 * KIND>;
};

template<>
class Type<TypeCategory::Real, 2> : public TypeBase<TypeCategory::Real, 2> {
public:
  using Scalar =
      value::Real<typename Type<TypeCategory::Integer, 2>::Scalar, 11>;
};

template<>
class Type<TypeCategory::Real, 4> : public TypeBase<TypeCategory::Real, 4> {
public:
  using Scalar =
      value::Real<typename Type<TypeCategory::Integer, 4>::Scalar, 24>;
};

template<>
class Type<TypeCategory::Real, 8> : public TypeBase<TypeCategory::Real, 8> {
public:
  using Scalar =
      value::Real<typename Type<TypeCategory::Integer, 8>::Scalar, 53>;
};

template<>
class Type<TypeCategory::Real, 10> : public TypeBase<TypeCategory::Real, 10> {
public:
  using Scalar = value::Real<value::Integer<80>, 64, false>;
};

template<>
class Type<TypeCategory::Real, 16> : public TypeBase<TypeCategory::Real, 16> {
public:
  using Scalar = value::Real<value::Integer<128>, 112>;
};

// The KIND type parameter on COMPLEX is the kind of each of its components.
template<int KIND>
class Type<TypeCategory::Complex, KIND>
  : public TypeBase<TypeCategory::Complex, KIND> {
public:
  using Part = Type<TypeCategory::Real, KIND>;
  using Scalar = value::Complex<typename Part::Scalar>;
};

template<>
class Type<TypeCategory::Character, 1>
  : public TypeBase<TypeCategory::Character, 1> {
public:
  using Scalar = std::string;
};

template<>
class Type<TypeCategory::Character, 2>
  : public TypeBase<TypeCategory::Character, 2> {
public:
  using Scalar = std::u16string;
};

template<>
class Type<TypeCategory::Character, 4>
  : public TypeBase<TypeCategory::Character, 4> {
public:
  using Scalar = std::u32string;
};

template<int KIND>
class Type<TypeCategory::Logical, KIND>
  : public TypeBase<TypeCategory::Logical, KIND> {
public:
  using Scalar = value::Logical<8 * KIND>;
};

// Type functions

template<typename T> using Scalar = typename std::decay_t<T>::Scalar;

// Given a specific type, find the type of the same kind in another category.
template<TypeCategory CATEGORY, typename T>
using SameKind = Type<CATEGORY, std::decay_t<T>::kind>;

using SubscriptInteger = Type<TypeCategory::Integer, 8>;
using LogicalResult = Type<TypeCategory::Logical, 1>;
using LargestReal = Type<TypeCategory::Real, 16>;

// A predicate that is true when a kind value is a kind that could possibly
// be supported for an intrinsic type category on some target instruction
// set architecture.
static constexpr bool IsValidKindOfIntrinsicType(
    TypeCategory category, std::int64_t kind) {
  switch (category) {
  case TypeCategory::Integer:
    return kind == 1 || kind == 2 || kind == 4 || kind == 8 || kind == 16;
  case TypeCategory::Real:
  case TypeCategory::Complex:
    return kind == 2 || kind == 4 || kind == 8 || kind == 10 || kind == 16;
  case TypeCategory::Character:
    return kind == 1;  // TODO: || kind == 2 || kind == 4;
  case TypeCategory::Logical:
    return kind == 1 || kind == 2 || kind == 4 || kind == 8;
  default: return false;
  }
}

// For each intrinsic type category CAT, CategoryTypes<CAT> is an instantiation
// of std::tuple<Type<CAT, K>> that comprises every kind value K in that
// category that could possibly be supported on any target.
template<TypeCategory CATEGORY, int KIND>
using CategoryKindTuple =
    std::conditional_t<IsValidKindOfIntrinsicType(CATEGORY, KIND),
        std::tuple<Type<CATEGORY, KIND>>, std::tuple<>>;

template<TypeCategory CATEGORY, int... KINDS>
using CategoryTypesHelper =
    common::CombineTuples<CategoryKindTuple<CATEGORY, KINDS>...>;

template<TypeCategory CATEGORY>
using CategoryTypes = CategoryTypesHelper<CATEGORY, 1, 2, 4, 8, 10, 16, 32>;

using IntegerTypes = CategoryTypes<TypeCategory::Integer>;
using RealTypes = CategoryTypes<TypeCategory::Real>;
using ComplexTypes = CategoryTypes<TypeCategory::Complex>;
using CharacterTypes = CategoryTypes<TypeCategory::Character>;
using LogicalTypes = CategoryTypes<TypeCategory::Logical>;

using FloatingTypes = common::CombineTuples<RealTypes, ComplexTypes>;
using NumericTypes = common::CombineTuples<IntegerTypes, FloatingTypes>;
using RelationalTypes = common::CombineTuples<NumericTypes, CharacterTypes>;
using AllIntrinsicTypes = common::CombineTuples<RelationalTypes, LogicalTypes>;

// When Scalar<T> is S, then TypeOf<S> is T.
// TypeOf is implemented by scanning all supported types for a match
// with Type<T>::Scalar.
template<typename CONST> struct TypeOfHelper {
  template<typename T> struct Predicate {
    static constexpr bool value() {
      return std::is_same_v<std::decay_t<CONST>,
          std::decay_t<typename T::Scalar>>;
    }
  };
  static constexpr int index{
      common::SearchMembers<Predicate, AllIntrinsicTypes>};
  using type = std::conditional_t<index >= 0,
      std::tuple_element_t<index, AllIntrinsicTypes>, void>;
};

template<typename CONST> using TypeOf = typename TypeOfHelper<CONST>::type;

// Represents a type of any supported kind within a particular category.
template<TypeCategory CATEGORY> struct SomeKind {
  static constexpr bool isSpecificIntrinsicType{false};
  static constexpr TypeCategory category{CATEGORY};
};

template<> class SomeKind<TypeCategory::Derived> {
public:
  static constexpr bool isSpecificIntrinsicType{false};
  static constexpr TypeCategory category{TypeCategory::Derived};

  CLASS_BOILERPLATE(SomeKind)
  explicit SomeKind(const semantics::DerivedTypeSpec &s) : spec_{&s} {}

  std::optional<DynamicType> GetType() const {
    return {DynamicType{category, 0, spec_}};
  }
  const semantics::DerivedTypeSpec &spec() const { return *spec_; }
  std::string Dump() const;

private:
  const semantics::DerivedTypeSpec *spec_;
};

using SomeInteger = SomeKind<TypeCategory::Integer>;
using SomeReal = SomeKind<TypeCategory::Real>;
using SomeComplex = SomeKind<TypeCategory::Complex>;
using SomeCharacter = SomeKind<TypeCategory::Character>;
using SomeLogical = SomeKind<TypeCategory::Logical>;
using SomeDerived = SomeKind<TypeCategory::Derived>;

// Represents a completely generic intrinsic type.
using SomeCategory = std::tuple<SomeInteger, SomeReal, SomeComplex,
    SomeCharacter, SomeLogical, SomeDerived>;
struct SomeType {
  static constexpr bool isSpecificIntrinsicType{false};
};

// For "[extern] template class", &c. boilerplate
#define FOR_EACH_INTEGER_KIND(PREFIX) \
  PREFIX<Type<TypeCategory::Integer, 1>>; \
  PREFIX<Type<TypeCategory::Integer, 2>>; \
  PREFIX<Type<TypeCategory::Integer, 4>>; \
  PREFIX<Type<TypeCategory::Integer, 8>>; \
  PREFIX<Type<TypeCategory::Integer, 16>>;
#define FOR_EACH_REAL_KIND(PREFIX) \
  PREFIX<Type<TypeCategory::Real, 2>>; \
  PREFIX<Type<TypeCategory::Real, 4>>; \
  PREFIX<Type<TypeCategory::Real, 8>>; \
  PREFIX<Type<TypeCategory::Real, 10>>; \
  PREFIX<Type<TypeCategory::Real, 16>>;
#define FOR_EACH_COMPLEX_KIND(PREFIX) \
  PREFIX<Type<TypeCategory::Complex, 2>>; \
  PREFIX<Type<TypeCategory::Complex, 4>>; \
  PREFIX<Type<TypeCategory::Complex, 8>>; \
  PREFIX<Type<TypeCategory::Complex, 10>>; \
  PREFIX<Type<TypeCategory::Complex, 16>>;
#define FOR_EACH_CHARACTER_KIND(PREFIX) \
  PREFIX<Type<TypeCategory::Character, 1>>; \
  PREFIX<Type<TypeCategory::Character, 2>>; \
  PREFIX<Type<TypeCategory::Character, 4>>;
#define FOR_EACH_LOGICAL_KIND(PREFIX) \
  PREFIX<Type<TypeCategory::Logical, 1>>; \
  PREFIX<Type<TypeCategory::Logical, 2>>; \
  PREFIX<Type<TypeCategory::Logical, 4>>; \
  PREFIX<Type<TypeCategory::Logical, 8>>;
#define FOR_EACH_INTRINSIC_KIND(PREFIX) \
  FOR_EACH_INTEGER_KIND(PREFIX) \
  FOR_EACH_REAL_KIND(PREFIX) \
  FOR_EACH_COMPLEX_KIND(PREFIX) \
  FOR_EACH_CHARACTER_KIND(PREFIX) \
  FOR_EACH_LOGICAL_KIND(PREFIX)
#define FOR_EACH_SPECIFIC_TYPE(PREFIX) \
  FOR_EACH_INTRINSIC_KIND(PREFIX) \
  PREFIX<SomeDerived>;
#define FOR_EACH_CATEGORY_TYPE(PREFIX) \
  PREFIX<SomeInteger>; \
  PREFIX<SomeReal>; \
  PREFIX<SomeComplex>; \
  PREFIX<SomeCharacter>; \
  PREFIX<SomeLogical>; \
  PREFIX<SomeType>;
#define FOR_EACH_TYPE_AND_KIND(PREFIX) \
  FOR_EACH_SPECIFIC_TYPE(PREFIX) \
  FOR_EACH_CATEGORY_TYPE(PREFIX)

// Wraps a constant scalar value of a specific intrinsic type
// in a class with its resolved type.
// N.B. Array constants are represented as array constructors
// and derived type constants are structure constructors; generic
// constants are generic expressions wrapping these constants.
template<typename T> struct Constant {
  // TODO: static_assert(T::isSpecificIntrinsicType);
  using Result = T;
  using Value = Scalar<Result>;
  CLASS_BOILERPLATE(Constant)
  template<typename A> Constant(const A &x) : value{x} {}
  template<typename A>
  Constant(std::enable_if_t<!std::is_reference_v<A>, A> &&x)
    : value(std::move(x)) {}
  constexpr std::optional<DynamicType> GetType() const {
    if constexpr (Result::isSpecificIntrinsicType) {
      return Result::GetType();
    } else {
      return value.GetType();
    }
  }
  int Rank() const { return 0; }
  std::ostream &Dump(std::ostream &) const;
  Value value;
};
}
#endif  // FORTRAN_EVALUATE_TYPE_H_
