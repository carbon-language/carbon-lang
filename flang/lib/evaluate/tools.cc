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

// Helpers for NumericOperation and its subroutines below.
static std::optional<Expr<SomeType>> NoExpr() { return std::nullopt; }

template<TypeCategory CAT>
std::optional<Expr<SomeType>> Package(Expr<SomeKind<CAT>> &&catExpr) {
  return {AsGenericExpr(std::move(catExpr))};
}
template<TypeCategory CAT>
std::optional<Expr<SomeType>> Package(
    std::optional<Expr<SomeKind<CAT>>> &&catExpr) {
  if (catExpr.has_value()) {
    return {AsGenericExpr(std::move(*catExpr))};
  }
  return NoExpr();
}

std::optional<Expr<SomeComplex>> ConstructComplex(
    parser::ContextualMessages &messages, Expr<SomeType> &&real,
    Expr<SomeType> &&imaginary) {
  if (auto converted{ConvertRealOperands(
          messages, std::move(real), std::move(imaginary))}) {
    return {std::visit(
        [](auto &&pair) {
          return MakeComplex(std::move(pair[0]), std::move(pair[1]));
        },
        std::move(*converted))};
  }
  return std::nullopt;
}

std::optional<Expr<SomeComplex>> ConstructComplex(
    parser::ContextualMessages &messages, std::optional<Expr<SomeType>> &&real,
    std::optional<Expr<SomeType>> &&imaginary) {
  if (auto parts{common::AllPresent(std::move(real), std::move(imaginary))}) {
    return ConstructComplex(messages, std::move(std::get<0>(*parts)),
        std::move(std::get<1>(*parts)));
  }
  return std::nullopt;
}

Expr<SomeReal> GetComplexPart(const Expr<SomeComplex> &z, bool isImaginary) {
  return std::visit(
      [&](const auto &zk) {
        static constexpr int kind{ResultType<decltype(zk)>::kind};
        return AsCategoryExpr(AsExpr(ComplexComponent<kind>{isImaginary, zk}));
      },
      z.u);
}

// Handle mixed COMPLEX+REAL (or INTEGER) operations in a smarter way
// than just converting the second operand to COMPLEX and performing the
// corresponding COMPLEX+COMPLEX operation.
template<template<typename> class OPR, TypeCategory RCAT>
std::optional<Expr<SomeType>> MixedComplexLeft(
    parser::ContextualMessages &messages, Expr<SomeComplex> &&zx,
    Expr<SomeKind<RCAT>> &&iry) {
  Expr<SomeReal> zr{GetComplexPart(zx, false)};
  Expr<SomeReal> zi{GetComplexPart(zx, true)};
  if constexpr (std::is_same_v<OPR<DefaultReal>, Add<DefaultReal>> ||
      std::is_same_v<OPR<DefaultReal>, Subtract<DefaultReal>>) {
    // (a,b) + x -> (a+x, b)
    // (a,b) - x -> (a-x, b)
    if (std::optional<Expr<SomeType>> rr{NumericOperation<OPR>(messages,
            AsGenericExpr(std::move(zr)), AsGenericExpr(std::move(iry)))}) {
      return Package(ConstructComplex(
          messages, std::move(*rr), AsGenericExpr(std::move(zi))));
    }
  } else if constexpr (std::is_same_v<OPR<DefaultReal>,
                           Multiply<DefaultReal>> ||
      std::is_same_v<OPR<DefaultReal>, Divide<DefaultReal>>) {
    // (a,b) * x -> (a*x, b*x)
    // (a,b) / x -> (a/x, b/x)
    auto copy{iry};
    auto rr{NumericOperation<Multiply>(
        messages, AsGenericExpr(std::move(zr)), AsGenericExpr(std::move(iry)))};
    auto ri{NumericOperation<Multiply>(messages, AsGenericExpr(std::move(zi)),
        AsGenericExpr(std::move(copy)))};
    if (auto parts{common::AllPresent(std::move(rr), std::move(ri))}) {
      return Package(ConstructComplex(messages, std::move(std::get<0>(*parts)),
          std::move(std::get<1>(*parts))));
    }
  } else {
    // (a,b) ? x -> (a,b) ? (x,0)
    Expr<SomeComplex> zy{ConvertTo(zx, std::move(iry))};
    return Package(PromoteAndCombine<OPR>(std::move(zx), std::move(zy)));
  }
  return NoExpr();
}

// Mixed COMPLEX operations with the COMPLEX operand on the right.
//  x + (a,b) -> (x+a, b)
//  x - (a,b) -> (x-a, -b)
//  x * (a,b) -> (x*a, x*b)
//  x / (a,b) -> (x,0) / (a,b)
template<template<typename> class OPR, TypeCategory LCAT>
std::optional<Expr<SomeType>> MixedComplexRight(
    parser::ContextualMessages &messages, Expr<SomeKind<LCAT>> &&irx,
    Expr<SomeComplex> &&zy) {
  if constexpr (std::is_same_v<OPR<DefaultReal>, Add<DefaultReal>> ||
      std::is_same_v<OPR<DefaultReal>, Multiply<DefaultReal>>) {
    // x + (a,b) -> (a,b) + x -> (a+x, b)
    // x * (a,b) -> (a,b) * x -> (a*x, b*x)
    return MixedComplexLeft<Add, LCAT>(messages, std::move(zy), std::move(irx));
  } else if constexpr (std::is_same_v<OPR<DefaultReal>,
                           Subtract<DefaultReal>>) {
    // x - (a,b) -> (x-a, -b)
    Expr<SomeReal> zr{GetComplexPart(zy, false)};
    Expr<SomeReal> zi{GetComplexPart(zy, true)};
    if (std::optional<Expr<SomeType>> rr{NumericOperation<Subtract>(messages,
            AsGenericExpr(std::move(irx)), AsGenericExpr(std::move(zr)))}) {
      return Package(ConstructComplex(
          messages, std::move(*rr), AsGenericExpr(-std::move(zi))));
    }
  } else {
    // x / (a,b) -> (x,0) / (a,b)    and any other operators that make it here
    Expr<SomeComplex> zx{ConvertTo(zy, std::move(irx))};
    return Package(PromoteAndCombine<OPR>(std::move(zx), std::move(zy)));
  }
  return NoExpr();
}

// N.B. When a "typeless" BOZ literal constant appears as one (not both!) of
// the operands to a dyadic operation, it assumes the type and kind of the
// other operand.
// TODO pmk: add Power, RealToIntPower, &c.
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
          // Mixed INTEGER/REAL operations
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
                  return AsCategoryExpr(AsExpr(
                      OPR<resultType>{ConvertToType<resultType>(std::move(ix)),
                          std::move(ryk)}));
                },
                std::move(ry.u)));
          },
          // Homogenous and mixed COMPLEX operations
          [](Expr<SomeComplex> &&zx, Expr<SomeComplex> &&zy) {
            return Package(PromoteAndCombine<OPR, TypeCategory::Complex>(
                std::move(zx), std::move(zy)));
          },
          [&](Expr<SomeComplex> &&zx, Expr<SomeInteger> &&zy) {
            return MixedComplexLeft<OPR>(
                messages, std::move(zx), std::move(zy));
          },
          [&](Expr<SomeComplex> &&zx, Expr<SomeReal> &&zy) {
            return MixedComplexLeft<OPR>(
                messages, std::move(zx), std::move(zy));
          },
          [&](Expr<SomeInteger> &&zx, Expr<SomeComplex> &&zy) {
            return MixedComplexRight<OPR>(
                messages, std::move(zx), std::move(zy));
          },
          [&](Expr<SomeReal> &&zx, Expr<SomeComplex> &&zy) {
            return MixedComplexRight<OPR>(
                messages, std::move(zx), std::move(zy));
          },
          // Operations with one typeless operand
          [&](BOZLiteralConstant &&bx, Expr<SomeInteger> &&iy) {
            return NumericOperation<OPR>(
                messages, ConvertTo(iy, std::move(bx)), std::move(y));
          },
          [&](BOZLiteralConstant &&bx, Expr<SomeReal> &&ry) {
            return NumericOperation<OPR>(
                messages, ConvertTo(ry, std::move(bx)), std::move(y));
          },
          [&](Expr<SomeInteger> &&ix, BOZLiteralConstant &&by) {
            return NumericOperation<OPR>(
                messages, std::move(x), ConvertTo(ix, std::move(by)));
          },
          [&](Expr<SomeReal> &&rx, BOZLiteralConstant &&by) {
            return NumericOperation<OPR>(
                messages, std::move(x), ConvertTo(rx, std::move(by)));
          },
          // Default case
          [&](auto &&, auto &&) {
            // TODO: defined operator
            messages.Say("non-numeric operands to numeric operation"_err_en_US);
            return NoExpr();
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

std::optional<Expr<SomeType>> Negation(
    parser::ContextualMessages &messages, Expr<SomeType> &&x) {
  return std::visit(
      common::visitors{[&](BOZLiteralConstant &&) {
                         messages.Say(
                             "BOZ literal cannot be negated"_err_en_US);
                         return NoExpr();
                       },
          [&](Expr<SomeInteger> &&x) { return Package(std::move(x)); },
          [&](Expr<SomeReal> &&x) { return Package(-std::move(x)); },
          [&](Expr<SomeComplex> &&x) { return Package(-std::move(x)); },
          [&](Expr<SomeCharacter> &&x) {
            // TODO: defined operator
            messages.Say("CHARACTER cannot be negated"_err_en_US);
            return NoExpr();
          },
          [&](Expr<SomeLogical> &&x) {
            // TODO: defined operator
            messages.Say("LOGICAL cannot be negated"_err_en_US);
            return NoExpr();
          }},
      std::move(x.u));
}

}  // namespace Fortran::evaluate
