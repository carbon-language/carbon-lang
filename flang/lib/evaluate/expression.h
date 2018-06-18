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

#ifndef FORTRAN_EVALUATE_EXPRESSION_H_
#define FORTRAN_EVALUATE_EXPRESSION_H_

#include "type.h"
#include "../common/indirection.h"
#include <variant>

namespace Fortran::evaluate {

template<Classification C, int KIND> struct Expression;

template<int KIND> struct Expression<Classification::Integer> {
  static constexpr Classification classification{Classification::Integer};
  static constexpr int kind{KIND};
};

using<int KIND> IntegerExpression = Expression<Classification::Integer, KIND>;
using<int KIND> RealExpression = Expression<Classification::Real, KIND>;

}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_EXPRESSION_H_
