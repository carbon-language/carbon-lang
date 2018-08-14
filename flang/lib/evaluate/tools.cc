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

#include "tools.h"
#include "../parser/message.h"
#include <variant>

using namespace Fortran::parser::literals;

namespace Fortran::evaluate {

Expr<SomeReal> ConvertToTypeOf(
    const Expr<SomeReal> &to, const Expr<SomeInteger> &from) {
  return std::visit(
      [&](const auto &rk) { return Expr<SomeReal>{decltype(rk){to}}; }, to.u);
}

Expr<SomeReal> ConvertToTypeOf(
    const Expr<SomeReal> &to, const Expr<SomeReal> &from) {
  return std::visit(
      [&](const auto &rk) { return Expr<SomeReal>{decltype(rk){to}}; }, to.u);
}

std::optional<std::pair<Expr<SomeReal>, Expr<SomeReal>>> ConvertRealOperands(
    parser::ContextualMessages &messages, GenericExpr &&x, GenericExpr &&y) {
  return std::visit(
      common::visitors{[&](Expr<SomeInteger> &&ix, Expr<SomeInteger> &&iy) {
                         // Can happen in a CMPLX() constructor.  Per F'2018,
                         // both integer operands are converted to default REAL.
                         return std::optional{std::make_pair(
                             Expr<SomeReal>{Expr<DefaultReal>{std::move(ix)}},
                             Expr<SomeReal>{Expr<DefaultReal>{std::move(iy)}})};
                       },
          [&](Expr<SomeInteger> &&ix, Expr<SomeReal> &&ry) {
            auto rx{ConvertToTypeOf(ry, std::move(ix))};
            return std::optional{std::make_pair(std::move(rx), std::move(ry))};
          },
          [&](Expr<SomeReal> &&rx, Expr<SomeInteger> &&iy) {
            auto ry{ConvertToTypeOf(rx, std::move(iy))};
            return std::optional{std::make_pair(std::move(rx), std::move(ry))};
          },
          [&](Expr<SomeReal> &&rx, Expr<SomeReal> &&ry) {
            ConvertToSameKind(rx, ry);
            return std::optional{std::make_pair(std::move(rx), std::move(ry))};
          },
          [&](const auto &, const auto &)
              -> std::optional<std::pair<Expr<SomeReal>, Expr<SomeReal>>> {
            messages.Say("operands must be INTEGER or REAL"_err_en_US);
            return std::nullopt;
          }},
      std::move(x.u), std::move(y.u));
}

std::optional<std::pair<Expr<SomeReal>, Expr<SomeReal>>> ConvertRealOperands(
    parser::ContextualMessages &messages, std::optional<GenericExpr> &&x,
    std::optional<GenericExpr> &&y) {
  if (x.has_value() && y.has_value()) {
    return ConvertRealOperands(messages, std::move(*x), std::move(*y));
  }
  return std::nullopt;
}

Expr<SomeType> GenericScalarToExpr(const Scalar<SomeType> &x) {
  return std::visit(
      [&](const auto &c) { return ToGenericExpr(SomeKindScalarToExpr(c)); },
      x.u);
}
}  // namespace Fortran::evaluate
