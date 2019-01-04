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
// total bit width and real precision.  Instances of the Type class template
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

// DynamicType is suitable for use as the result type for
// GetType() functions and member functions.
struct DynamicType {
  bool operator==(const DynamicType &that) const;
  std::string AsFortran() const;
  DynamicType ResultTypeForMultiply(const DynamicType &) const;

  TypeCategory category;
  int kind{0};  // set only for intrinsic types
  const semantics::DerivedTypeSpec *derived{nullptr};
  const semantics::Symbol *descriptor{nullptr};
};

// Result will be missing when a symbol is absent or
// has an erroneous type, e.g., REAL(KIND=666).
std::optional<DynamicType> GetSymbolType(const semantics::Symbol *);

// Specific intrinsic types are represented by specializations of
// this class template Type<CATEGORY, KIND>.
template<TypeCategory CATEGORY, int KIND = 0> class Type;

template<TypeCategory CATEGORY, int KIND = 0> struct TypeBase {
  static constexpr TypeCategory category{CATEGORY};
  static constexpr int kind{KIND};
  constexpr bool operator==(const TypeBase &) const { return true; }
  static constexpr DynamicType GetType() { return {category, kind}; }
  static std::string AsFortran() { return GetType().AsFortran(); }
};

template<int KIND>
class Type<TypeCategory::Integer, KIND>
  : public TypeBase<TypeCategory::Integer, KIND> {
public:
  using Scalar = value::Integer<8 * KIND>;
};

// REAL(KIND=2) is IEEE half-precision (16 bits)
template<>
class Type<TypeCategory::Real, 2> : public TypeBase<TypeCategory::Real, 2> {
public:
  using Scalar =
      value::Real<typename Type<TypeCategory::Integer, 2>::Scalar, 11>;
};

// REAL(KIND=3) identifies the "other" half-precision format, which is
// basically REAL(4) without its least-order 16 fraction bits.
template<>
class Type<TypeCategory::Real, 3> : public TypeBase<TypeCategory::Real, 3> {
public:
  using Scalar =
      value::Real<typename Type<TypeCategory::Integer, 2>::Scalar, 8>;
};

// REAL(KIND=4) is IEEE-754 single precision (32 bits)
template<>
class Type<TypeCategory::Real, 4> : public TypeBase<TypeCategory::Real, 4> {
public:
  using Scalar =
      value::Real<typename Type<TypeCategory::Integer, 4>::Scalar, 24>;
};

// REAL(KIND=8) is IEEE double precision (64 bits)
template<>
class Type<TypeCategory::Real, 8> : public TypeBase<TypeCategory::Real, 8> {
public:
  using Scalar =
      value::Real<typename Type<TypeCategory::Integer, 8>::Scalar, 53>;
};

// REAL(KIND=10) is x87 FPU extended precision (80 bits, all explicit)
template<>
class Type<TypeCategory::Real, 10> : public TypeBase<TypeCategory::Real, 10> {
public:
  using Scalar = value::Real<value::Integer<80>, 64, false>;
};

// REAL(KIND=16) is IEEE quad precision (128 bits)
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

// Many expressions, including subscripts, CHARACTER lengths, array bounds,
// and effective type parameter values, are of a maximal kind of INTEGER.
using IndirectSubscriptIntegerExpr =
    CopyableIndirection<Expr<SubscriptInteger>>;

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
    return kind == 2 || kind == 3 || kind == 4 || kind == 8 || kind == 10 ||
        kind == 16;
  case TypeCategory::Character: return kind == 1 || kind == 2 || kind == 4;
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
using CategoryTypes = CategoryTypesHelper<CATEGORY, 1, 2, 3, 4, 8, 10, 16, 32>;

using IntegerTypes = CategoryTypes<TypeCategory::Integer>;
using RealTypes = CategoryTypes<TypeCategory::Real>;
using ComplexTypes = CategoryTypes<TypeCategory::Complex>;
using CharacterTypes = CategoryTypes<TypeCategory::Character>;
using LogicalTypes = CategoryTypes<TypeCategory::Logical>;

using FloatingTypes = common::CombineTuples<RealTypes, ComplexTypes>;
using NumericTypes = common::CombineTuples<IntegerTypes, FloatingTypes>;
using RelationalTypes = common::CombineTuples<NumericTypes, CharacterTypes>;
using AllIntrinsicTypes = common::CombineTuples<RelationalTypes, LogicalTypes>;
using LengthlessIntrinsicTypes =
    common::CombineTuples<NumericTypes, LogicalTypes>;

// Predicate: does a type represent a specific intrinsic type?
template<typename T>
constexpr bool IsSpecificIntrinsicType{common::HasMember<T, AllIntrinsicTypes>};

// Predicate: is a type an intrinsic type that is completely characterized
// by its category and kind parameter value, or might it have a derived type
// &/or a length type parameter?
template<typename T>
constexpr bool IsLengthlessIntrinsicType{
    common::HasMember<T, LengthlessIntrinsicTypes>};

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
  static constexpr TypeCategory category{CATEGORY};
  constexpr bool operator==(const SomeKind &) const { return true; }
};

template<> class SomeKind<TypeCategory::Derived> {
public:
  static constexpr TypeCategory category{TypeCategory::Derived};

  CLASS_BOILERPLATE(SomeKind)
  explicit SomeKind(const semantics::DerivedTypeSpec &dts,
      const semantics::Symbol *sym = nullptr)
    : spec_{&dts}, descriptor_{sym} {}

  DynamicType GetType() const {
    return DynamicType{category, 0, spec_, descriptor_};
  }
  const semantics::DerivedTypeSpec &spec() const { return *spec_; }
  const semantics::Symbol *descriptor() const { return descriptor_; }
  bool operator==(const SomeKind &) const;
  std::string AsFortran() const;

private:
  const semantics::DerivedTypeSpec *spec_;
  const semantics::Symbol *descriptor_{nullptr};
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
struct SomeType {};

// For generating "[extern] template class", &c. boilerplate
#define EXPAND_FOR_EACH_INTEGER_KIND(M, P) \
  M(P, 1) M(P, 2) M(P, 4) M(P, 8) M(P, 16)
#define EXPAND_FOR_EACH_REAL_KIND(M, P) \
  M(P, 2) M(P, 3) M(P, 4) M(P, 8) M(P, 10) M(P, 16)
#define EXPAND_FOR_EACH_COMPLEX_KIND(M, P) EXPAND_FOR_EACH_REAL_KIND(M, P)
#define EXPAND_FOR_EACH_CHARACTER_KIND(M, P) M(P, 1) M(P, 2) M(P, 4)
#define EXPAND_FOR_EACH_LOGICAL_KIND(M, P) M(P, 1) M(P, 2) M(P, 4) M(P, 8)
#define TEMPLATE_INSTANTIATION(P, ARG) P<ARG>;

#define FOR_EACH_INTEGER_KIND_HELP(PREFIX, K) \
  PREFIX<Type<TypeCategory::Integer, K>>;
#define FOR_EACH_REAL_KIND_HELP(PREFIX, K) PREFIX<Type<TypeCategory::Real, K>>;
#define FOR_EACH_COMPLEX_KIND_HELP(PREFIX, K) \
  PREFIX<Type<TypeCategory::Complex, K>>;
#define FOR_EACH_CHARACTER_KIND_HELP(PREFIX, K) \
  PREFIX<Type<TypeCategory::Character, K>>;
#define FOR_EACH_LOGICAL_KIND_HELP(PREFIX, K) \
  PREFIX<Type<TypeCategory::Logical, K>>;

#define FOR_EACH_INTEGER_KIND(PREFIX) \
  EXPAND_FOR_EACH_INTEGER_KIND(FOR_EACH_INTEGER_KIND_HELP, PREFIX)
#define FOR_EACH_REAL_KIND(PREFIX) \
  EXPAND_FOR_EACH_REAL_KIND(FOR_EACH_REAL_KIND_HELP, PREFIX)
#define FOR_EACH_COMPLEX_KIND(PREFIX) \
  EXPAND_FOR_EACH_COMPLEX_KIND(FOR_EACH_COMPLEX_KIND_HELP, PREFIX)
#define FOR_EACH_CHARACTER_KIND(PREFIX) \
  EXPAND_FOR_EACH_CHARACTER_KIND(FOR_EACH_CHARACTER_KIND_HELP, PREFIX)
#define FOR_EACH_LOGICAL_KIND(PREFIX) \
  EXPAND_FOR_EACH_LOGICAL_KIND(FOR_EACH_LOGICAL_KIND_HELP, PREFIX)

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
  PREFIX<SomeDerived>; \
  PREFIX<SomeType>;
#define FOR_EACH_TYPE_AND_KIND(PREFIX) \
  FOR_EACH_INTRINSIC_KIND(PREFIX) \
  FOR_EACH_CATEGORY_TYPE(PREFIX)

// Wraps a constant scalar value of a specific intrinsic type
// in a class with its resolved type.
// N.B. Array constants are represented as array constructors
// and derived type constants are structure constructors; generic
// constants are generic expressions wrapping these constants.
template<typename T> struct Constant {
  static_assert(IsSpecificIntrinsicType<T>);
  using Result = T;
  using Value = Scalar<Result>;

  CLASS_BOILERPLATE(Constant)
  template<typename A> Constant(const A &x) : value{x} {}
  template<typename A>
  Constant(std::enable_if_t<!std::is_reference_v<A>, A> &&x)
    : value(std::move(x)) {}

  constexpr DynamicType GetType() const { return Result::GetType(); }
  int Rank() const { return 0; }
  bool operator==(const Constant &that) const { return value == that.value; }
  std::ostream &AsFortran(std::ostream &) const;

  Value value;
};
}
#endif  // FORTRAN_EVALUATE_TYPE_H_
