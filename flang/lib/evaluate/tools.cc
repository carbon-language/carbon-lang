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
#include "../common/idioms.h"
#include "../parser/message.h"
#include <algorithm>
#include <variant>

using namespace Fortran::parser::literals;

namespace Fortran::evaluate {

ConvertRealOperandsResult ConvertRealOperands(
    parser::ContextualMessages &messages, Expr<SomeType> &&x,
    Expr<SomeType> &&y) {
  return std::visit(
      common::visitors{
          [&](Expr<SomeInteger> &&ix,
              Expr<SomeInteger> &&iy) -> ConvertRealOperandsResult {
            // Can happen in a CMPLX() constructor.  Per F'2018,
            // both integer operands are converted to default REAL.
            return std::optional{std::make_pair(
                ToCategoryExpr(ConvertToType<DefaultReal>(std::move(ix))),
                ToCategoryExpr(ConvertToType<DefaultReal>(std::move(iy))))};
          },
          [&](Expr<SomeInteger> &&ix,
              Expr<SomeReal> &&ry) -> ConvertRealOperandsResult {
            auto rx{ConvertTo(ry, std::move(ix))};
            return std::optional{std::make_pair(std::move(rx), std::move(ry))};
          },
          [&](Expr<SomeReal> &&rx,
              Expr<SomeInteger> &&iy) -> ConvertRealOperandsResult {
            auto ry{ConvertTo(rx, std::move(iy))};
            return std::optional{std::make_pair(std::move(rx), std::move(ry))};
          },
          [&](Expr<SomeReal> &&rx,
              Expr<SomeReal> &&ry) -> ConvertRealOperandsResult {
            ConvertToSameKind(rx, ry);
            return std::optional{std::make_pair(std::move(rx), std::move(ry))};
          },
          [&](auto &&, auto &&) -> ConvertRealOperandsResult {
            // TODO: allow BOZ here?
            messages.Say("operands must be INTEGER or REAL"_err_en_US);
            return std::nullopt;
          }},
      std::move(x.u), std::move(y.u));
}

ConvertRealOperandsResult ConvertRealOperands(
    parser::ContextualMessages &messages, std::optional<Expr<SomeType>> &&x,
    std::optional<Expr<SomeType>> &&y) {
  auto partial{[&](Expr<SomeType> &&x, Expr<SomeType> &&y) {
    return ConvertRealOperands(messages, std::move(x), std::move(y));
  }};
  using fType = ConvertRealOperandsResult(Expr<SomeType> &&, Expr<SomeType> &&);
  std::function<fType> f{partial};
  return common::JoinOptionals(
      common::MapOptional(f, std::move(x), std::move(y)));
}

template<template<typename> class OPR, TypeCategory CAT>
std::optional<Expr<SomeType>> PromoteAndCombine(
    Expr<SomeKind<CAT>> &&x, Expr<SomeKind<CAT>> &&y) {
  return {Expr<SomeType>{std::visit(
      [&](auto &&xk, auto &&yk) -> Expr<SomeKind<CAT>> {
        using xt = ResultType<decltype(xk)>;
        using yt = ResultType<decltype(yk)>;
        using ToType = Type<CAT, std::max(xt::kind, yt::kind)>;
        return {Expr<ToType>{OPR<ToType>{EnsureKind<ToType>(std::move(xk)),
            EnsureKind<ToType>(std::move(yk))}}};
      },
      std::move(x.u), std::move(y.u))}};
}

template<template<typename> class OPR>
std::optional<Expr<SomeType>> NumericOperation(
    parser::ContextualMessages &messages, Expr<SomeType> &&x,
    Expr<SomeType> &&y) {
  return std::visit(
      common::visitors{[](Expr<SomeInteger> &&ix, Expr<SomeInteger> &&iy) {
                         return PromoteAndCombine<OPR, TypeCategory::Integer>(
                             std::move(ix), std::move(iy));
                       },
          [](Expr<SomeReal> &&rx, Expr<SomeReal> &&ry) {
            return PromoteAndCombine<OPR, TypeCategory::Real>(
                std::move(rx), std::move(ry));
          },
          [](Expr<SomeReal> &&rx, Expr<SomeInteger> &&iy) {
            return std::optional{Expr<SomeType>{std::visit(
                [&](auto &&rxk) -> Expr<SomeReal> {
                  using kindEx = decltype(rxk);
                  using resultType = ResultType<kindEx>;
                  return {kindEx{OPR<resultType>{std::move(rxk),
                      ConvertToType<resultType>(std::move(iy))}}};
                },
                std::move(rx.u))}};
          },
          [](Expr<SomeInteger> &&ix, Expr<SomeReal> &&ry) {
            return std::optional{Expr<SomeType>{std::visit(
                [&](auto &&ryk) -> Expr<SomeReal> {
                  using kindEx = decltype(ryk);
                  using resultType = ResultType<kindEx>;
                  return {kindEx{
                      OPR<resultType>{ConvertToType<resultType>(std::move(ix)),
                          std::move(ryk)}}};
                },
                std::move(ry.u))}};
          },
          [](Expr<SomeComplex> &&zx, Expr<SomeComplex> &&zy) {
            return PromoteAndCombine<OPR, TypeCategory::Complex>(
                std::move(zx), std::move(zy));
          },
          // TODO pmk complex; Add/Sub different from Mult/Div
          [&](auto &&, auto &&) {
            messages.Say("non-numeric operands to numeric operation"_err_en_US);
            return std::optional<Expr<SomeType>>{std::nullopt};
          }},
      std::move(x.u), std::move(y.u));
}

template std::optional<Expr<SomeType>> NumericOperation<Add>(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&);

}  // namespace Fortran::evaluate
