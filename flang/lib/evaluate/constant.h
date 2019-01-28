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
template<typename T> class Constant {
  static_assert(std::is_same_v<T, SomeDerived> || IsSpecificIntrinsicType<T>);

public:
  using Result = T;
  using Value = Scalar<Result>;

  CLASS_BOILERPLATE(Constant)
  template<typename A> Constant(const A &x) : values_{x} {}
  template<typename A>
  Constant(std::enable_if_t<!std::is_reference_v<A>, A> &&x)
    : values_{std::move(x)} {}
  Constant(std::vector<Value> &&x, std::vector<std::int64_t> &&s)
    : values_(std::move(x)), shape_(std::move(s)) {}

  constexpr DynamicType GetType() const { return Result::GetType(); }
  int Rank() const { return static_cast<int>(shape_.size()); }
  bool operator==(const Constant &that) const {
    return shape_ == that.shape_ && values_ == that.values_;
  }
  std::size_t size() const { return values_.size(); }
  const std::vector<std::int64_t> &shape() const { return shape_; }
  std::int64_t LEN() const {
    if constexpr (T::category != TypeCategory::Character) {
      common::die("LEN() of non-character Constant");
    } else if (values_.empty()) {
      return 0;
    } else {
      return static_cast<std::int64_t>(values_[0].size());
    }
  }

  const Value &operator*() const {
    CHECK(values_.size() == 1);
    return values_.at(0);
  }

  const Value &At(const std::vector<std::int64_t> &index) {
    CHECK(index.size() == static_cast<std::size_t>(Rank()));
    std::int64_t stride{1}, offset{0};
    int dim{0};
    for (std::int64_t j : index) {
      std::int64_t bound{shape_[dim++]};
      CHECK(j >= 1 && j <= bound);
      offset += stride * (j - 1);
      stride *= bound;
    }
    return values_.at(offset);
  }

  Constant<SubscriptInteger> SHAPE() const;
  std::ostream &AsFortran(std::ostream &) const;

private:
  std::vector<Value> values_;
  std::vector<std::int64_t> shape_;
  // TODO pmk: make CHARACTER values contiguous
};

FOR_EACH_INTRINSIC_KIND(extern template class Constant)
}
#endif  // FORTRAN_EVALUATE_CONSTANT_H_
