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
}  // namespace Fortran::semantics

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

  TypeCategory category;
  int kind{0};
  const semantics::DerivedTypeSpec *derived{nullptr};
};

// Specific intrinsic types are represented by specializations of
// this class template Type<CATEGORY, KIND>.
template<TypeCategory CATEGORY, int KIND = 0> class Type;

template<TypeCategory CATEGORY, int KIND> struct TypeBase {
  static constexpr bool isSpecificType{true};
  static constexpr DynamicType dynamicType{CATEGORY, KIND};
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

// Convenience type aliases:
// Default REAL just simply has to be IEEE-754 single precision today.
// It occupies one numeric storage unit by definition.  The default INTEGER
// and default LOGICAL intrinsic types also have to occupy one numeric
// storage unit, so their kinds are also forced.  Default COMPLEX must always
// comprise two default REAL components.
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
using LargestReal = Type<TypeCategory::Real, 16>;

// For each intrinsic type category CAT, CategoryTypes<CAT> is an instantiation
// of std::tuple<Type<CAT, K>> that comprises every kind value K in that
// category that could possibly be supported on any target.
template<TypeCategory CATEGORY, int... KINDS>
using CategoryTypesTuple = std::tuple<Type<CATEGORY, KINDS>...>;

template<TypeCategory CATEGORY> struct CategoryTypesHelper;
template<> struct CategoryTypesHelper<TypeCategory::Integer> {
  using type = CategoryTypesTuple<TypeCategory::Integer, 1, 2, 4, 8, 16>;
};
template<> struct CategoryTypesHelper<TypeCategory::Real> {
  using type = CategoryTypesTuple<TypeCategory::Real, 2, 4, 8, 10, 16>;
};
template<> struct CategoryTypesHelper<TypeCategory::Complex> {
  using type = CategoryTypesTuple<TypeCategory::Complex, 2, 4, 8, 10, 16>;
};
template<> struct CategoryTypesHelper<TypeCategory::Character> {
  using type = CategoryTypesTuple<TypeCategory::Character, 1>;  // TODO: 2 & 4
};
template<> struct CategoryTypesHelper<TypeCategory::Logical> {
  using type = CategoryTypesTuple<TypeCategory::Logical, 1, 2, 4, 8>;
};
template<> struct CategoryTypesHelper<TypeCategory::Derived> {
  using type = std::tuple<>;
};
template<TypeCategory CATEGORY>
using CategoryTypes = typename CategoryTypesHelper<CATEGORY>::type;

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

// A variant union that can hold a scalar constant of any type chosen from
// a set of types, which is passed in as a tuple of Type<> specializations.
template<typename TYPES> struct SomeScalar {
  using Types = TYPES;
  CLASS_BOILERPLATE(SomeScalar)

  template<typename A> SomeScalar(const A &x) : u{x} {}
  template<typename A>
  SomeScalar(std::enable_if_t<!std::is_reference_v<A>, A> &&x)
    : u{std::move(x)} {}

  std::optional<std::int64_t> ToInt64() const {
    return std::visit(
        [](const auto &x) -> std::optional<std::int64_t> {
          if constexpr (TypeOf<decltype(x)>::category ==
              TypeCategory::Integer) {
            return {x.ToInt64()};
          }
          return std::nullopt;
        },
        u);
  }

  std::optional<std::string> ToString() const {
    return std::visit(
        [](const auto &x) -> std::optional<std::string> {
          if constexpr (std::is_same_v<std::string,
                            std::decay_t<decltype(x)>>) {
            return {x};
          }
          return std::nullopt;
        },
        u);
  }

  std::optional<bool> IsTrue() const {
    return std::visit(
        [](const auto &x) -> std::optional<bool> {
          if constexpr (TypeOf<decltype(x)>::category ==
              TypeCategory::Logical) {
            return {x.IsTrue()};
          }
          return std::nullopt;
        },
        u);
  }

  common::MapTemplate<Scalar, Types> u;
};

template<TypeCategory CATEGORY>
using SomeKindScalar = SomeScalar<CategoryTypes<CATEGORY>>;
using GenericScalar = SomeScalar<AllIntrinsicTypes>;

// Represents a type of any supported kind within a particular category.
template<TypeCategory CATEGORY> struct SomeKind {
  static constexpr bool isSpecificType{false};
  static constexpr TypeCategory category{CATEGORY};
  using Scalar = SomeKindScalar<category>;
};

template<> class SomeKind<TypeCategory::Derived> {
public:
  static constexpr bool isSpecificType{true};
  static constexpr TypeCategory category{TypeCategory::Derived};
  using Scalar = void;

  CLASS_BOILERPLATE(SomeKind)
  explicit SomeKind(const semantics::DerivedTypeSpec &s) : spec_{&s} {}

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
  static constexpr bool isSpecificType{false};
  using Scalar = GenericScalar;
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

}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_TYPE_H_
