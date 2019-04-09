// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef FORTRAN_EVALUATE_CONSTANT_H_
#define FORTRAN_EVALUATE_CONSTANT_H_

#include "formatting.h"
#include "type.h"
#include <map>
#include <ostream>

namespace Fortran::evaluate {

// Wraps a constant value in a class templated by its resolved type.
// This Constant<> template class should be instantiated only for
// concrete intrinsic types and SomeDerived.  There is no instance
// Constant<Expr<SomeType>> since there is no way to constrain each
// element of its array to hold the same type.  To represent a generic
// constants, use a generic expression like Expr<SomeInteger> &
// Expr<SomeType>) to wrap the appropriate instantiation of Constant<>.

template<typename> class Constant;

// Constant<> is specialized for Character kinds and SomeDerived.
// The non-Character intrinsic types, and SomeDerived, share enough
// common behavior that they use this common base class.
template<typename RESULT, typename SCALAR = Scalar<RESULT>> class ConstantBase {
  static_assert(RESULT::category != TypeCategory::Character);

public:
  using Result = RESULT;
  using ScalarValue = SCALAR;

  template<typename A> ConstantBase(const A &x) : values_{x} {}
  template<typename A, typename = common::NoLvalue<A>>
  ConstantBase(A &&x) : values_{std::move(x)} {}
  ConstantBase(std::vector<ScalarValue> &&x, std::vector<std::int64_t> &&dims)
    : values_(std::move(x)), shape_(std::move(dims)) {}
  DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(ConstantBase)
  ~ConstantBase();

  int Rank() const { return static_cast<int>(shape_.size()); }
  bool operator==(const ConstantBase &that) const {
    return shape_ == that.shape_ && values_ == that.values_;
  }
  bool empty() const { return values_.empty(); }
  std::size_t size() const { return values_.size(); }
  const std::vector<ScalarValue> &values() const { return values_; }
  const std::vector<std::int64_t> &shape() const { return shape_; }

  ScalarValue operator*() const {
    CHECK(values_.size() == 1);
    return values_.at(0);
  }

  // Apply 1-based subscripts
  ScalarValue At(const std::vector<std::int64_t> &) const;
  ScalarValue At(std::vector<std::int64_t> &&) const;

  Constant<SubscriptInteger> SHAPE() const;
  std::ostream &AsFortran(std::ostream &) const;

protected:
  std::vector<ScalarValue> values_;
  std::vector<std::int64_t> shape_;

private:
  const Constant<Result> &AsConstant() const {
    return *static_cast<const Constant<Result> *>(this);
  }

  DynamicType GetType() const { return AsConstant().GetType(); }
};

template<typename T> class Constant : public ConstantBase<T> {
public:
  using Result = T;
  using ScalarValue = Scalar<Result>;
  using ConstantBase<Result, ScalarValue>::ConstantBase;
  CLASS_BOILERPLATE(Constant)
  static constexpr DynamicType GetType() { return Result::GetType(); }
};

template<int KIND> class Constant<Type<TypeCategory::Character, KIND>> {
public:
  using Result = Type<TypeCategory::Character, KIND>;
  using ScalarValue = Scalar<Result>;
  CLASS_BOILERPLATE(Constant)
  explicit Constant(const ScalarValue &);
  explicit Constant(ScalarValue &&);
  Constant(
      std::int64_t, std::vector<ScalarValue> &&, std::vector<std::int64_t> &&);
  ~Constant();

  int Rank() const { return static_cast<int>(shape_.size()); }
  bool operator==(const Constant &that) const {
    return shape_ == that.shape_ && values_ == that.values_;
  }
  bool empty() const;
  std::size_t size() const;
  const std::vector<std::int64_t> &shape() const { return shape_; }

  std::int64_t LEN() const { return length_; }

  ScalarValue operator*() const {
    CHECK(static_cast<std::int64_t>(values_.size()) == length_);
    return values_;
  }

  // Apply 1-based subscripts
  ScalarValue At(const std::vector<std::int64_t> &) const;

  Constant<SubscriptInteger> SHAPE() const;
  std::ostream &AsFortran(std::ostream &) const;
  static constexpr DynamicType GetType() { return Result::GetType(); }

private:
  ScalarValue values_;  // one contiguous string
  std::int64_t length_;
  std::vector<std::int64_t> shape_;
};

using StructureConstructorValues = std::map<const semantics::Symbol *,
    common::CopyableIndirection<Expr<SomeType>>>;

template<>
class Constant<SomeDerived>
  : public ConstantBase<SomeDerived, StructureConstructorValues> {
public:
  using Result = SomeDerived;
  using Base = ConstantBase<Result, StructureConstructorValues>;
  Constant(const StructureConstructor &);
  Constant(StructureConstructor &&);
  Constant(const semantics::DerivedTypeSpec &, std::vector<ScalarValue> &&,
      std::vector<std::int64_t> &&);
  Constant(const semantics::DerivedTypeSpec &,
      std::vector<StructureConstructor> &&, std::vector<std::int64_t> &&);
  CLASS_BOILERPLATE(Constant)

  const semantics::DerivedTypeSpec &derivedTypeSpec() const {
    return *derivedTypeSpec_;
  }

  DynamicType GetType() const { return DynamicType{derivedTypeSpec()}; }

private:
  const semantics::DerivedTypeSpec *derivedTypeSpec_;
};

FOR_EACH_LENGTHLESS_INTRINSIC_KIND(extern template class ConstantBase, )
extern template class ConstantBase<SomeDerived, StructureConstructorValues>;
FOR_EACH_INTRINSIC_KIND(extern template class Constant, )

#define INSTANTIATE_CONSTANT_TEMPLATES \
  FOR_EACH_LENGTHLESS_INTRINSIC_KIND(template class ConstantBase, ) \
  template class ConstantBase<SomeDerived, StructureConstructorValues>; \
  FOR_EACH_INTRINSIC_KIND(template class Constant, )
}
#endif  // FORTRAN_EVALUATE_CONSTANT_H_
