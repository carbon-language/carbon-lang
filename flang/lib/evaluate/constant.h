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

#include "type.h"
#include <ostream>

namespace Fortran::evaluate {

// Wraps a constant value in a class templated by its resolved type.
// N.B. Generic constants are represented by generic expressions
// (like Expr<SomeInteger> & Expr<SomeType>) wrapping the appropriate
// instantiations of Constant.

template<typename> class Constant;

template<typename RESULT> class ConstantBase {
public:
  using Result = RESULT;
  using Value = Scalar<Result>;

  template<typename A> ConstantBase(const A &x) : values_{x} {}
  template<typename A>
  ConstantBase(std::enable_if_t<!std::is_reference_v<A>, A> &&x)
    : values_{std::move(x)} {}
  ConstantBase(std::vector<Value> &&x, std::vector<std::int64_t> &&s)
    : values_(std::move(x)), shape_(std::move(s)) {}
  ~ConstantBase();

  int Rank() const { return static_cast<int>(shape_.size()); }
  bool operator==(const ConstantBase &that) const {
    return shape_ == that.shape_ && values_ == that.values_;
  }
  bool empty() const { return values_.empty(); }
  std::size_t size() const { return values_.size(); }
  const std::vector<std::int64_t> &shape() const { return shape_; }

  Value operator*() const {
    CHECK(values_.size() == 1);
    return values_.at(0);
  }

  // Apply 1-based subscripts
  Value At(const std::vector<std::int64_t> &) const;

  Constant<SubscriptInteger> SHAPE() const;
  std::ostream &AsFortran(std::ostream &) const;

protected:
  std::vector<Value> values_;
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
  using ConstantBase<Result>::ConstantBase;
  CLASS_BOILERPLATE(Constant)
  static constexpr DynamicType GetType() { return Result::GetType(); }
};

template<int KIND>
class Constant<Type<TypeCategory::Character, KIND>>
  : public ConstantBase<Type<TypeCategory::Character, KIND>> {
public:
  using Result = Type<TypeCategory::Character, KIND>;
  using ConstantBase<Result>::ConstantBase;
  CLASS_BOILERPLATE(Constant)
  static constexpr DynamicType GetType() { return Result::GetType(); }
  std::int64_t LEN() const {
    if (this->values_.empty()) {
      return 0;
    } else {
      return static_cast<std::int64_t>(this->values_.front().size());
    }
  }
  // TODO pmk: make CHARACTER values contiguous (they're strings now)
};

template<> class Constant<SomeDerived> : public ConstantBase<SomeDerived> {
public:
  using Result = SomeDerived;
  using Base = ConstantBase<Result>;
  template<typename A>
  Constant(const semantics::DerivedTypeSpec &spec, const A &x)
    : Base{x}, spec_{&spec} {}
  template<typename A>
  Constant(const semantics::DerivedTypeSpec &spec,
      std::enable_if_t<!std::is_reference_v<A>, A> &&x)
    : Base{std::move(x)}, spec_{&spec} {}
  Constant(const semantics::DerivedTypeSpec &, std::vector<Value> &&,
      std::vector<std::int64_t> &&);

  CLASS_BOILERPLATE(Constant)
  DynamicType GetType() const {
    return DynamicType{TypeCategory::Derived, 0, spec_};
  }

private:
  const semantics::DerivedTypeSpec *spec_;
};

FOR_EACH_SPECIFIC_TYPE(extern template class ConstantBase)
FOR_EACH_INTRINSIC_KIND(extern template class Constant)
}
#endif  // FORTRAN_EVALUATE_CONSTANT_H_
