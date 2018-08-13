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

#ifndef FORTRAN_EVALUATE_TOOLS_H_
#define FORTRAN_EVALUATE_TOOLS_H_

#include "expression.h"
#include "../common/idioms.h"
#include "../parser/message.h"
#include <optional>
#include <utility>

namespace Fortran::evaluate {

// Convert the second argument to the same type and kind of the first.
SomeKindRealExpr ConvertToTypeOf(
    const SomeKindRealExpr &to, const SomeKindIntegerExpr &from);
SomeKindRealExpr ConvertToTypeOf(
    const SomeKindRealExpr &to, const SomeKindRealExpr &from);

// Ensure that both operands of an intrinsic REAL operation or CMPLX()
// are INTEGER or REAL, and convert them as necessary to the same REAL type.
using ConvertRealOperandsResult =
    std::optional<std::pair<SomeKindRealExpr, SomeKindRealExpr>>;
ConvertRealOperandsResult ConvertRealOperands(
    parser::ContextualMessages &, GenericExpr &&, GenericExpr &&);

}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_TOOLS_H_
