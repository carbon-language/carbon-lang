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

#include "type.h"
#include "expression.h"
#include "../common/idioms.h"
#include <cinttypes>
#include <optional>
#include <variant>

namespace Fortran::evaluate {

std::optional<std::int64_t> GenericScalar::ToInt64() const {
  if (const auto *j{std::get_if<SomeKindScalar<TypeCategory::Integer>>(&u)}) {
    return std::visit(
        [](const auto &k) { return std::optional<std::int64_t>{k.ToInt64()}; },
        j->u);
  }
  return std::nullopt;
}

std::optional<std::string> GenericScalar::ToString() const {
  if (const auto *c{std::get_if<SomeKindScalar<TypeCategory::Character>>(&u)}) {
    if (const std::string * s{std::get_if<std::string>(&c->u)}) {
      return std::optional<std::string>{*s};
    }
  }
  return std::nullopt;
}

// TODO pmk: maybe transplant these templates to type.h/expression.h?

// There's some admittedly opaque type-fu going on below.
// Given a GenericScalar value, we want to be able to (re-)wrap it as
// a GenericExpr.  So we extract its value, then build up an expression
// around it.  The subtle magic is in the first template, whose result
// is a specific expression whose Fortran type category and kind are inferred
// from the type of the scalar constant.
template<typename A> Expr<ScalarValueType<A>> ScalarConstantToExpr(const A &x) {
  return {x};
}

template<typename A>
Expr<SomeKind<A::category>> ToSomeKindExpr(const Expr<A> &x) {
  return {x};
}

template<TypeCategory CAT>
Expr<SomeKind<CAT>> SomeKindScalarToExpr(const SomeKindScalar<CAT> &x) {
  return std::visit(
      [](const auto &c) { return ToSomeKindExpr(ScalarConstantToExpr(c)); },
      x.u);
}

GenericExpr GenericScalar::ToGenericExpr() const {
  return std::visit(
      [&](const auto &c) { return GenericExpr{SomeKindScalarToExpr(c)}; }, u);
}

}  // namespace Fortran::evaluate
