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

#include "constant.h"
#include <cinttypes>
#include <limits>

namespace Fortran::evaluate {

template<IntrinsicType::KindLenCType KIND>
ScalarIntegerConstant<KIND> ScalarIntegerConstant<KIND>::Negate() const {
  ScalarIntegerConstant<KIND> result{*this};
  return result.Assign(-static_cast<BigIntType>(value_));
}

template<IntrinsicType::KindLenCType KIND>
ScalarIntegerConstant<KIND> ScalarIntegerConstant<KIND>::Add(const ScalarIntegerConstant<KIND> &that) const {
  ScalarIntegerConstant<KIND> result{*this};
  return result.Assign(static_cast<BigIntType>(value_) +
                       static_cast<BigIntType>(that.value_));
}

template<IntrinsicType::KindLenCType KIND>
ScalarIntegerConstant<KIND> ScalarIntegerConstant<KIND>::Subtract(const ScalarIntegerConstant<KIND> &that) const {
  ScalarIntegerConstant<KIND> result{*this};
  return result.Assign(static_cast<BigIntType>(value_) -
                       static_cast<BigIntType>(that.value_));
}

template<IntrinsicType::KindLenCType KIND>
ScalarIntegerConstant<KIND> ScalarIntegerConstant<KIND>::Multiply(const ScalarIntegerConstant<KIND> &that) const {
  ScalarIntegerConstant<KIND> result{*this};
  return result.Assign(static_cast<BigIntType>(value_) -
                       static_cast<BigIntType>(that.value_));
}

template<IntrinsicType::KindLenCType KIND>
ScalarIntegerConstant<KIND> ScalarIntegerConstant<KIND>::Divide(const ScalarIntegerConstant<KIND> &that) const {
  ScalarIntegerConstant<KIND> result{*this};
  if (that.value_ == 0) {
    result.SetError(Error::DivisionByZero);
    return result;
  } else {
    return result.Assign(static_cast<BigIntType>(value_) /
                         static_cast<BigIntType>(that.value_));
  }
}

template class ScalarConstant<IntrinsicType::Classification,
                   IntrinsicType::Classification::Integer, 1>;

}  // namespace Fortran::evaluate
