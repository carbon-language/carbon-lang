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

// Convenience functions and operator overloadings for expression construction.
template<TypeCategory C, int K>
Expr<Type<C, K>> operator-(Expr<Type<C, K>> &&x) {
  return {Negate<Type<C, K>>{std::move(x)}};
}

template<TypeCategory C, int K>
Expr<Type<C, K>> operator+(Expr<Type<C, K>> &&x, Expr<Type<C, K>> &&y) {
  return {Add<Type<C, K>>{std::move(x), std::move(y)}};
}

template<TypeCategory C, int K>
Expr<Type<C, K>> operator-(Expr<Type<C, K>> &&x, Expr<Type<C, K>> &&y) {
  return {Subtract<Type<C, K>>{std::move(x), std::move(y)}};
}

template<TypeCategory C, int K>
Expr<Type<C, K>> operator*(Expr<Type<C, K>> &&x, Expr<Type<C, K>> &&y) {
  return {Multiply<Type<C, K>>{std::move(x), std::move(y)}};
}

template<TypeCategory C, int K>
Expr<Type<C, K>> operator/(Expr<Type<C, K>> &&x, Expr<Type<C, K>> &&y) {
  return {Divide<Type<C, K>>{std::move(x), std::move(y)}};
}

template<TypeCategory C> Expr<SomeKind<C>> operator-(Expr<SomeKind<C>> &&x) {
  return std::visit(
      [](auto &xk) { return Expr<SomeKind<C>>{-std::move(xk)}; }, x.u.u);
}

template<TypeCategory C>
Expr<SomeKind<C>> operator+(Expr<SomeKind<C>> &&x, Expr<SomeKind<C>> &&y) {
  return std::visit(
      [](auto &xk, auto &yk) {
        return Expr<SomeKind<C>>{std::move(xk) + std::move(yk)};
      },
      x.u.u, y.u.u);
}

template<TypeCategory C>
Expr<SomeKind<C>> operator-(Expr<SomeKind<C>> &&x, Expr<SomeKind<C>> &&y) {
  return std::visit(
      [](auto &xk, auto &yk) {
        return Expr<SomeKind<C>>{std::move(xk) - std::move(yk)};
      },
      x.u.u, y.u.u);
}

template<TypeCategory C>
Expr<SomeKind<C>> operator*(Expr<SomeKind<C>> &&x, Expr<SomeKind<C>> &&y) {
  return std::visit(
      [](auto &xk, auto &yk) {
        return Expr<SomeKind<C>>{std::move(xk) * std::move(yk)};
      },
      x.u.u, y.u.u);
}

template<TypeCategory C>
Expr<SomeKind<C>> operator/(Expr<SomeKind<C>> &&x, Expr<SomeKind<C>> &&y) {
  return std::visit(
      [](auto &xk, auto &yk) {
        return Expr<SomeKind<C>>{std::move(xk) / std::move(yk)};
      },
      x.u.u, y.u.u);
}

// Convert the second argument expression to an expression of the same type
// and kind as that of the first.
template<TypeCategory TC, typename F>
Expr<SomeKind<TC>> ConvertToTypeAndKindOf(
    const Expr<SomeKind<TC>> &to, Expr<F> &&from) {
  return std::visit(
      [&](const auto &tk) -> Expr<SomeKind<TC>> {
        using SpecificExpr = std::decay_t<decltype(tk)>;
        return {SpecificExpr{std::move(from)}};
      },
      to.u.u);
}

// Given two expressions of the same type category, convert one to the
// kind of the other in place if it has a smaller kind.
template<TypeCategory CAT>
void ConvertToSameKind(Expr<SomeKind<CAT>> &x, Expr<SomeKind<CAT>> &y) {
  std::visit(
      [&](auto &xk, auto &yk) {
        using xt = ResultType<decltype(xk)>;
        using yt = ResultType<decltype(yk)>;
        if constexpr (xt::kind < yt::kind) {
          x.u = Expr<yt>{xk};
        } else if constexpr (xt::kind > yt::kind) {
          y.u = Expr<xt>{yk};
        }
      },
      x.u.u, y.u.u);
}

// Ensure that both operands of an intrinsic REAL operation (or CMPLX()
// constructor) are INTEGER or REAL, then convert them as necessary to the
// same kind of REAL.
using ConvertRealOperandsResult =
    std::optional<std::pair<Expr<SomeReal>, Expr<SomeReal>>>;
ConvertRealOperandsResult ConvertRealOperands(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&);
ConvertRealOperandsResult ConvertRealOperands(parser::ContextualMessages &,
    std::optional<Expr<SomeType>> &&, std::optional<Expr<SomeType>> &&);

template<typename A> Expr<TypeOf<A>> ScalarConstantToExpr(const A &x) {
  static_assert(std::is_same_v<Scalar<TypeOf<A>>, std::decay_t<A>> ||
      !"TypeOf<> is broken");
  return {x};
}

template<TypeCategory CAT, int KIND>
Expr<SomeKind<CAT>> ToSomeKindExpr(Expr<Type<CAT, KIND>> &&x) {
  return {std::move(x)};
}

template<TypeCategory CAT>
Expr<SomeKind<CAT>> SomeKindScalarToExpr(const SomeKindScalar<CAT> &x) {
  return std::visit(
      [](const auto &c) { return ToSomeKindExpr(ScalarConstantToExpr(c)); },
      x.u.u);
}

Expr<SomeType> GenericScalarToExpr(const Scalar<SomeType> &);

template<TypeCategory CAT, int KIND>
Expr<SomeType> ToGenericExpr(Expr<Type<CAT, KIND>> &&x) {
  return Expr<SomeType>{Expr<SomeKind<CAT>>{std::move(x)}};
}

template<TypeCategory CAT>
Expr<SomeType> ToGenericExpr(Expr<SomeKind<CAT>> &&x) {
  return Expr<SomeType>{std::move(x)};
}

// Convert, if necessary, an expression to a specific kind in the same
// category.
template<typename TOTYPE>
Expr<TOTYPE> EnsureKind(Expr<SomeKind<TOTYPE::category>> &&x) {
  using ToType = TOTYPE;
  using FromGenericType = SomeKind<ToType::category>;
  if (auto *p{std::get_if<Expr<ToType>>(&x.u.u)}) {
    return std::move(*p);
  }
  if constexpr (ToType::category == TypeCategory::Complex) {
    return {std::visit(
        [](auto &z) -> ComplexConstructor<ToType::kind> {
          using FromType = ResultType<decltype(z)>;
          using FromPart = typename FromType::Part;
          using FromGeneric = SomeKind<TypeCategory::Real>;
          using ToPart = typename ToType::Part;
          Convert<ToPart, FromGeneric> re{Expr<FromGeneric>{
              Expr<FromPart>{ComplexComponent<FromType::kind>{false, z}}}};
          Convert<ToPart, FromGeneric> im{Expr<FromGeneric>{
              Expr<FromPart>{ComplexComponent<FromType::kind>{true, z}}}};
          return {std::move(re), std::move(im)};
        },
        x.u.u)};
  } else {
    return {Convert<ToType, FromGenericType>{std::move(x)}};
  }
}

template<template<typename> class OPR>
std::optional<Expr<SomeType>> NumericOperation(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&);
extern template std::optional<Expr<SomeType>> NumericOperation<Add>(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&);

}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_TOOLS_H_
