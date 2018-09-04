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
      common::visitors{[&](Expr<SomeInteger> &&ix, Expr<SomeInteger> &&iy)
                           -> ConvertRealOperandsResult {
                         // Can happen in a CMPLX() constructor.  Per F'2018,
                         // both integer operands are converted to default REAL.
                         return {AsSameKindExprs<TypeCategory::Real>(
                             ConvertToType<DefaultReal>(std::move(ix)),
                             ConvertToType<DefaultReal>(std::move(iy)))};
                       },
          [&](Expr<SomeInteger> &&ix,
              Expr<SomeReal> &&ry) -> ConvertRealOperandsResult {
            return {AsSameKindExprs<TypeCategory::Real>(
                ConvertTo(ry, std::move(ix)), std::move(ry))};
          },
          [&](Expr<SomeReal> &&rx,
              Expr<SomeInteger> &&iy) -> ConvertRealOperandsResult {
            return {AsSameKindExprs<TypeCategory::Real>(
                std::move(rx), ConvertTo(rx, std::move(iy)))};
          },
          [&](Expr<SomeReal> &&rx,
              Expr<SomeReal> &&ry) -> ConvertRealOperandsResult {
            return {AsSameKindExprs<TypeCategory::Real>(
                std::move(rx), std::move(ry))};
          },
          [&](auto &&, auto &&) -> ConvertRealOperandsResult {
            // TODO: allow BOZ here?
            messages.Say("operands must be INTEGER or REAL"_err_en_US);
            return std::nullopt;
          }},
      std::move(x.u), std::move(y.u));
}

// A helper template for NumericOperation below.
template<TypeCategory CAT>
std::optional<Expr<SomeType>> Package(Expr<SomeKind<CAT>> &&catExpr) {
  return {AsGenericExpr(std::move(catExpr))};
}

// N.B. When a "typeless" BOZ literal constant appears as one (not both!) of
// the operands to a dyadic INTEGER or REAL operation, it assumes the type
// and kind of the other operand.
template<template<typename> class OPR>
std::optional<Expr<SomeType>> NumericOperation(
    parser::ContextualMessages &messages, Expr<SomeType> &&x,
    Expr<SomeType> &&y) {
  return std::visit(
      common::visitors{[](Expr<SomeInteger> &&ix, Expr<SomeInteger> &&iy) {
                         return Package(
                             PromoteAndCombine<OPR, TypeCategory::Integer>(
                                 std::move(ix), std::move(iy)));
                       },
          [](Expr<SomeReal> &&rx, Expr<SomeReal> &&ry) {
            return Package(PromoteAndCombine<OPR, TypeCategory::Real>(
                std::move(rx), std::move(ry)));
          },
          [](Expr<SomeReal> &&rx, Expr<SomeInteger> &&iy) {
            return Package(std::visit(
                [&](auto &&rxk) -> Expr<SomeReal> {
                  using resultType = ResultType<decltype(rxk)>;
                  return AsCategoryExpr(AsExpr(OPR<resultType>{std::move(rxk),
                      ConvertToType<resultType>(std::move(iy))}));
                },
                std::move(rx.u)));
          },
          [](Expr<SomeInteger> &&ix, Expr<SomeReal> &&ry) {
            return Package(std::visit(
                [&](auto &&ryk) -> Expr<SomeReal> {
                  using resultType = ResultType<decltype(ryk)>;
                  return AsCategoryExpr(AsExpr(OPR<resultType>{
                      ConvertToType<resultType>(std::move(ix)), std::move(ryk)}));
                },
                std::move(ry.u)));
          },
          [](Expr<SomeComplex> &&zx, Expr<SomeComplex> &&zy) {
            return Package(PromoteAndCombine<OPR, TypeCategory::Complex>(
                std::move(zx), std::move(zy)));
          },
          [&](BOZLiteralConstant &&bx, Expr<SomeInteger> &&iy) {
            return NumericOperation<OPR>(messages, ConvertTo(iy, std::move(bx)), std::move(y));
          },
          [&](BOZLiteralConstant &&bx, Expr<SomeReal> &&ry) {
            return NumericOperation<OPR>(messages, ConvertTo(ry, std::move(bx)), std::move(y));
          },
          [&](Expr<SomeInteger> &&ix, BOZLiteralConstant &&by) {
            return NumericOperation<OPR>(messages, std::move(x), ConvertTo(ix, std::move(by)));
          },
          [&](Expr<SomeReal> &&rx, BOZLiteralConstant &&by) {
            return NumericOperation<OPR>(messages, std::move(x), ConvertTo(rx, std::move(by)));
          },
          // TODO pmk mixed complex; Add/Sub different from Mult/Div
          [&](auto &&, auto &&) {
            messages.Say("non-numeric operands to numeric operation"_err_en_US);
            return std::optional<Expr<SomeType>>{std::nullopt};
          }},
      std::move(x.u), std::move(y.u));
}

template std::optional<Expr<SomeType>> NumericOperation<Add>(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&);
template std::optional<Expr<SomeType>> NumericOperation<Subtract>(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&);
template std::optional<Expr<SomeType>> NumericOperation<Multiply>(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&);
template std::optional<Expr<SomeType>> NumericOperation<Divide>(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&);

}  // namespace Fortran::evaluate
