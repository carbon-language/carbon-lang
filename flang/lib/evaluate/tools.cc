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

SomeKindRealExpr ConvertToTypeOf(
    const SomeKindRealExpr &to, const SomeKindIntegerExpr &from) {
  return std::visit(
      [&](const auto &rk) { return SomeKindRealExpr{decltype(rk){to}}; }, to.u);
}

SomeKindRealExpr ConvertToTypeOf(
    const SomeKindRealExpr &to, const SomeKindRealExpr &from) {
  return std::visit(
      [&](const auto &rk) { return SomeKindRealExpr{decltype(rk){to}}; }, to.u);
}

static void ConvertToSameRealKind(SomeKindRealExpr &x, SomeKindRealExpr &y) {
  std::visit(
      [&](auto &xk, auto &yk) {
        using xt = typename std::decay<decltype(xk)>::type;
        using yt = typename std::decay<decltype(yk)>::type;
        constexpr int kindDiff{xt::Result::kind - yt::Result::kind};
        if constexpr (kindDiff < 0) {
          x.u = yt{xk};
        } else if constexpr (kindDiff > 0) {
          y.u = xt{yk};
        }
      },
      x.u, y.u);
}

std::optional<std::pair<SomeKindRealExpr, SomeKindRealExpr>>
ConvertRealOperands(
    parser::ContextualMessages &messages, GenericExpr &&x, GenericExpr &&y) {
  return std::visit(
      common::visitors{
          [&](SomeKindIntegerExpr &&ix, SomeKindIntegerExpr &&iy) {
            // Can happen in a CMPLX() constructor.  Per F'2018, both integer
            // operands are converted to default REAL.
            return std::optional{std::make_pair(
                SomeKindRealExpr{Expr<DefaultReal>{std::move(ix)}},
                SomeKindRealExpr{Expr<DefaultReal>{std::move(iy)}})};
          },
          [&](SomeKindIntegerExpr &&ix, SomeKindRealExpr &&ry) {
            auto rx{ConvertToTypeOf(ry, std::move(ix))};
            return std::optional{std::make_pair(std::move(rx), std::move(ry))};
          },
          [&](SomeKindRealExpr &&rx, SomeKindIntegerExpr &&iy) {
            auto ry{ConvertToTypeOf(rx, std::move(iy))};
            return std::optional{std::make_pair(std::move(rx), std::move(ry))};
          },
          [&](SomeKindRealExpr &&rx, SomeKindRealExpr &&ry) {
            ConvertToSameRealKind(rx, ry);
            return std::optional{std::make_pair(std::move(rx), std::move(ry))};
          },
          [&](const auto &, const auto &)
              -> std::optional<std::pair<SomeKindRealExpr, SomeKindRealExpr>> {
            messages.Say("operands must be INTEGER or REAL"_err_en_US);
            return std::nullopt;
          }},
      std::move(x.u), std::move(y.u));
}

std::optional<std::pair<SomeKindRealExpr, SomeKindRealExpr>>
ConvertRealOperands(parser::ContextualMessages &messages,
    std::optional<GenericExpr> &&x, std::optional<GenericExpr> &&y) {
  if (x.has_value() && y.has_value()) {
    return ConvertRealOperands(messages, std::move(*x), std::move(*y));
  }
  return std::nullopt;
}
}  // namespace Fortran::evaluate
