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

// Static expression checking

#ifndef FORTRAN_EVALUATE_CHECK_EXPRESSION_H_
#define FORTRAN_EVALUATE_CHECK_EXPRESSION_H_

#include "expression.h"
#include "type.h"
#include <optional>

namespace Fortran::parser {
class ContextualMessages;
}
namespace Fortran::semantics {
class Scope;
}

namespace Fortran::evaluate {

// Predicate: true when an expression is a constant expression (in the
// strict sense of the Fortran standard); it may not (yet) be a hard
// constant value.
template<typename A> bool IsConstantExpr(const A &);
extern template bool IsConstantExpr(const Expr<SomeType> &);

// Predicate: true when an expression is an object designator with
// constant addressing and no vector-valued subscript.
bool IsInitialDataTarget(const Expr<SomeType> &);

// Check whether an expression is a specification expression
// (10.1.11(2), C1010).  Constant expressions are always valid
// specification expressions.
template<typename A>
void CheckSpecificationExpr(
    const A &, parser::ContextualMessages &, const semantics::Scope &);
extern template void CheckSpecificationExpr(const Expr<SomeType> &x,
    parser::ContextualMessages &, const semantics::Scope &);
extern template void CheckSpecificationExpr(
    const std::optional<Expr<SomeInteger>> &x, parser::ContextualMessages &,
    const semantics::Scope &);
extern template void CheckSpecificationExpr(
    const std::optional<Expr<SubscriptInteger>> &x,
    parser::ContextualMessages &, const semantics::Scope &);
}
#endif
