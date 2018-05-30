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

#ifndef FORTRAN_EVALUATE_CONSTANT_H_
#define FORTRAN_EVALUATE_CONSTANT_H_

#include "type.h"
#include "../parser/idioms.h"
#include <cinttypes>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

namespace Fortran::evaluate {

enum class Error { None, Overflow, DivisionByZero, InvalidOperation };
enum class Relation { LessThan, Equal, GreaterThan, Unordered };

template<typename IntrinsicTypeClassification,
    IntrinsicTypeClassification CLASSIFICATION,
    IntrinsicType::KindLenCType KIND>
class ScalarConstant;

template<typename IntrinsicTypeClassification,
    IntrinsicTypeClassification CLASSIFICATION,
    IntrinsicType::KindLenCType KIND>
class ScalarConstantBase {
public:
  constexpr ScalarConstantBase() {}
  constexpr IntrinsicType Type() const { return {CLASSIFICATION, KIND}; }
  constexpr Error error() const { return error_; }
  constexpr bool AnyError() const { return error_ != Error::None; }

protected:
  constexpr void SetError(Error error) {
    if (error_ == Error::None) {
      error_ = error;
    }
  }

private:
  Error error_{Error::None};
};

// Integer scalar constants
template<IntrinsicType::KindLenCType KIND>
class ScalarConstant<IntrinsicType::Classification,
    IntrinsicType::Classification::Integer, KIND>
  : public ScalarConstantBase<IntrinsicType::Classification,
        IntrinsicType::Classification::Integer, KIND> {
private:
  static_assert(KIND == 1 || KIND == 2 || KIND == 4 || KIND == 8);
  using BaseType = ScalarConstantBase<IntrinsicType::Classification,
      IntrinsicType::Classification::Integer, KIND>;

public:
  using ValueCType = std::int64_t;

  constexpr ScalarConstant() {}
  constexpr ScalarConstant(ValueCType x) { Assign(x); }
  constexpr ScalarConstant(std::uint64_t x) {
    value_ = x;
    if (value_ < 0) {
      BaseType::SetError(Error::Overflow);
    } else {
      CheckForOverflow();
    }
  }
  constexpr ScalarConstant(const ScalarConstant &that) = default;
  constexpr ScalarConstant &operator=(const ScalarConstant &) = default;

  constexpr ValueCType value() const { return value_; }

  constexpr void Assign(ValueCType x) {
    value_ = x;
    CheckForOverflow();
  }
  ScalarConstant Negate() const;
  ScalarConstant Add(const ScalarConstant &) const;
  ScalarConstant Subtract(const ScalarConstant &) const;
  ScalarConstant Multiply(const ScalarConstant &) const;
  ScalarConstant Divide(const ScalarConstant &) const;

private:
  using BigIntType = __int128_t;
  constexpr ScalarConstant &Assign(BigIntType x) {
    value_ = x;
    if (value_ != x) {
      BaseType::SetError(Error::Overflow);
    } else {
      CheckForOverflow();
    }
    return *this;
  }

  constexpr void CheckForOverflow() {
    if (KIND < 8 && !BaseType::AnyError()) {
      ValueCType limit{static_cast<ValueCType>(1) << (8 * KIND)};
      if (value_ >= limit) {
        BaseType::SetError(Error::Overflow);
        value_ &= limit - 1;
      } else if (value_ < -limit) {
        BaseType::SetError(Error::Overflow);
        value_ &= limit + limit - 1;
        if (value_ >= limit) {
          value_ |= -limit;
        }
      }
    }
  }

  ValueCType value_{0};
};

template<IntrinsicType::KindLenCType KIND>
using ScalarIntegerConstant = ScalarConstant<IntrinsicType::Classification,
    IntrinsicType::Classification::Integer, KIND>;

extern template class ScalarConstant<IntrinsicType::Classification,
    IntrinsicType::Classification::Integer, 1>;
extern template class ScalarConstant<IntrinsicType::Classification,
    IntrinsicType::Classification::Integer, 2>;
extern template class ScalarConstant<IntrinsicType::Classification,
    IntrinsicType::Classification::Integer, 4>;
extern template class ScalarConstant<IntrinsicType::Classification,
    IntrinsicType::Classification::Integer, 8>;

}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_CONSTANT_H_
