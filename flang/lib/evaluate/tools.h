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
#include <array>
#include <optional>
#include <utility>

namespace Fortran::evaluate {

// Generalizing packagers: these take operations and expressions of more
// specific types and wrap them in Expr<> containers of more abstract types.

template<typename A> Expr<ResultType<A>> AsExpr(A &&x) {
  return {std::move(x)};
}

template<TypeCategory CAT, int KIND>
Expr<SomeKind<CAT>> AsCategoryExpr(Expr<Type<CAT, KIND>> &&x) {
  return {std::move(x)};
}

template<TypeCategory CAT>
Expr<SomeKind<CAT>> AsCategoryExpr(SomeKindScalar<CAT> &&x) {
  return std::visit(
      [](auto &&scalar) {
        using Ty = TypeOf<std::decay_t<decltype(scalar)>>;
        return Expr<SomeKind<CAT>>{Expr<Ty>{Constant<Ty>{std::move(scalar)}}};
      },
      x.u);
}

template<typename A> Expr<SomeType> AsGenericExpr(A &&x) {
  return {std::move(x)};
}

template<TypeCategory CAT, int KIND>
Expr<SomeType> AsGenericExpr(Expr<Type<CAT, KIND>> &&x) {
  return {AsCategoryExpr(std::move(x))};
}

template<> inline Expr<SomeType> AsGenericExpr(Constant<SomeType> &&x) {
  return std::visit(
      [](auto &&scalar) {
        using Ty = TypeOf<std::decay_t<decltype(scalar)>>;
        return Expr<SomeType>{Expr<SomeKind<Ty::category>>{
            Expr<Ty>{Constant<Ty>{std::move(scalar)}}}};
      },
      x.value.u);
}

template<> inline Expr<SomeType> AsGenericExpr(GenericScalar &&x) {
  return std::visit(
      [](auto &&scalar) {
        using Ty = TypeOf<std::decay_t<decltype(scalar)>>;
        return Expr<SomeType>{Expr<SomeKind<Ty::category>>{
            Expr<Ty>{Constant<Ty>{std::move(scalar)}}}};
      },
      x.u);
}

Expr<SomeReal> GetComplexPart(
    const Expr<SomeComplex> &, bool isImaginary = false);

template<int KIND>
Expr<SomeComplex> MakeComplex(Expr<Type<TypeCategory::Real, KIND>> &&re,
    Expr<Type<TypeCategory::Real, KIND>> &&im) {
  return AsCategoryExpr(
      AsExpr(ComplexConstructor<KIND>{std::move(re), std::move(im)}));
}

// Creation of conversion expressions can be done to either a known
// specific intrinsic type with ConvertToType<T>(x) or by converting
// one arbitrary expression to the type of another with ConvertTo(to, from).

template<typename TO, TypeCategory FROMCAT>
Expr<TO> ConvertToType(Expr<SomeKind<FROMCAT>> &&x) {
  static_assert(TO::isSpecificType);
  if constexpr (FROMCAT != TO::category) {
    if constexpr (TO::category == TypeCategory::Complex) {
      using Part = typename TO::Part;
      Scalar<Part> zero;
      return {ComplexConstructor<TO::kind>{
          ConvertToType<Part>(std::move(x)), Expr<Part>{Constant<Part>{zero}}}};
    } else {
      return {Convert<TO, FROMCAT>{std::move(x)}};
    }
  } else {
    // Same type category
    if (auto already{common::GetIf<Expr<TO>>(x.u)}) {
      return std::move(*already);
    }
    if constexpr (TO::category == TypeCategory::Complex) {
      // Extract, convert, and recombine the components.
      return {std::visit(
          [](auto &z) -> ComplexConstructor<TO::kind> {
            using FromType = ResultType<decltype(z)>;
            using FromPart = typename FromType::Part;
            using FromGeneric = SomeKind<TypeCategory::Real>;
            using ToPart = typename TO::Part;
            Convert<ToPart, TypeCategory::Real> re{Expr<FromGeneric>{
                Expr<FromPart>{ComplexComponent<FromType::kind>{false, z}}}};
            Convert<ToPart, TypeCategory::Real> im{Expr<FromGeneric>{
                Expr<FromPart>{ComplexComponent<FromType::kind>{true, z}}}};
            return {std::move(re), std::move(im)};
          },
          x.u)};
    } else {
      return {Convert<TO, TO::category>{std::move(x)}};
    }
  }
}

template<typename TO> Expr<TO> ConvertToType(BOZLiteralConstant &&x) {
  static_assert(TO::isSpecificType);
  using Value = typename Constant<TO>::Value;
  if constexpr (TO::category == TypeCategory::Integer) {
    return Expr<TO>{Constant<TO>{Value::ConvertUnsigned(std::move(x)).value}};
  } else {
    static_assert(TO::category == TypeCategory::Real);
    using Word = typename Value::Word;
    return Expr<TO>{Constant<TO>{Word::ConvertUnsigned(std::move(x)).value}};
  }
}

template<TypeCategory TC, int TK, TypeCategory FC>
Expr<Type<TC, TK>> ConvertTo(
    const Expr<Type<TC, TK>> &, Expr<SomeKind<FC>> &&x) {
  return ConvertToType<Type<TC, TK>>(std::move(x));
}

template<TypeCategory TC, int TK, TypeCategory FC, int FK>
Expr<Type<TC, TK>> ConvertTo(
    const Expr<Type<TC, TK>> &, Expr<Type<FC, FK>> &&x) {
  return ConvertToType<Type<TC, TK>>(AsCategoryExpr(std::move(x)));
}

template<TypeCategory TC, TypeCategory FC>
Expr<SomeKind<TC>> ConvertTo(
    const Expr<SomeKind<TC>> &to, Expr<SomeKind<FC>> &&from) {
  return std::visit(
      [&](const auto &toKindExpr) {
        using KindExpr = std::decay_t<decltype(toKindExpr)>;
        return AsCategoryExpr(
            ConvertToType<ResultType<KindExpr>>(std::move(from)));
      },
      to.u);
}

template<TypeCategory TC, TypeCategory FC, int FK>
Expr<SomeKind<TC>> ConvertTo(
    const Expr<SomeKind<TC>> &to, Expr<Type<FC, FK>> &&from) {
  return ConvertTo(to, AsCategoryExpr(std::move(from)));
}

template<typename FT>
Expr<SomeType> ConvertTo(const Expr<SomeType> &to, Expr<FT> &&from) {
  return std::visit(
      [&](const auto &toCatExpr) {
        return AsGenericExpr(ConvertTo(toCatExpr, std::move(from)));
      },
      to.u);
}

template<TypeCategory CAT>
Expr<SomeKind<CAT>> ConvertTo(
    const Expr<SomeKind<CAT>> &to, BOZLiteralConstant &&from) {
  return std::visit(
      [&](const auto &tok) {
        using Ty = ResultType<decltype(tok)>;
        return AsCategoryExpr(ConvertToType<Ty>(std::move(from)));
      },
      to.u);
}

template<typename A, int N = 2> using SameExprs = std::array<Expr<A>, N>;

// Given a type category CAT, SameKindExprs<CAT, N> is a variant that
// holds an arrays of expressions of the same supported kind in that
// category.
template<int N = 2> struct SameKindExprsHelper {
  template<typename A> using SameExprs = std::array<Expr<A>, N>;
};
template<TypeCategory CAT, int N = 2>
using SameKindExprs =
    common::MapTemplate<SameKindExprsHelper<N>::template SameExprs,
        CategoryTypes<CAT>>;

// Given references to two expressions of arbitrary kind in the same type
// category, convert one to the kind of the other when it has the smaller kind,
// then return them in a type-safe package.
template<TypeCategory CAT>
SameKindExprs<CAT, 2> AsSameKindExprs(
    Expr<SomeKind<CAT>> &&x, Expr<SomeKind<CAT>> &&y) {
  return std::visit(
      [&](auto &&kx, auto &&ky) -> SameKindExprs<CAT, 2> {
        using XTy = ResultType<decltype(kx)>;
        using YTy = ResultType<decltype(ky)>;
        if constexpr (std::is_same_v<XTy, YTy>) {
          return {SameExprs<XTy>{std::move(kx), std::move(ky)}};
        } else if constexpr (XTy::kind < YTy::kind) {
          return {SameExprs<YTy>{ConvertTo(ky, std::move(kx)), std::move(ky)}};
        } else {
          return {SameExprs<XTy>{std::move(kx), ConvertTo(kx, std::move(ky))}};
        }
#if !__clang__ && 100 * __GNUC__ + __GNUC_MINOR__ == 801
        // Silence a bogus warning about a missing return with G++ 8.1.0.
        // Doesn't execute, but must be correctly typed.
        CHECK(!"can't happen");
        return {SameExprs<XTy>{std::move(kx), std::move(kx)}};
#endif
      },
      std::move(x.u), std::move(y.u));
}

// Ensure that both operands of an intrinsic REAL operation (or CMPLX()
// constructor) are INTEGER or REAL, then convert them as necessary to the
// same kind of REAL.
using ConvertRealOperandsResult =
    std::optional<SameKindExprs<TypeCategory::Real, 2>>;
ConvertRealOperandsResult ConvertRealOperands(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&);

// Per F'2018 R718, if both components are INTEGER, they are both converted
// to default REAL and the result is default COMPLEX.  Otherwise, the
// kind of the result is the kind of most precise REAL component, and the other
// component is converted if necessary to its type.
std::optional<Expr<SomeComplex>> ConstructComplex(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&);
std::optional<Expr<SomeComplex>> ConstructComplex(parser::ContextualMessages &,
    std::optional<Expr<SomeType>> &&, std::optional<Expr<SomeType>> &&);

template<typename A> Expr<TypeOf<A>> ScalarConstantToExpr(const A &x) {
  using Ty = TypeOf<A>;
  static_assert(
      std::is_same_v<Scalar<Ty>, std::decay_t<A>> || !"TypeOf<> is broken");
  return {Constant<Ty>{x}};
}

// Combine two expressions of the same specific numeric type with an operation
// to produce a new expression.  Implements piecewise addition and subtraction
// for COMPLEX.
template<template<typename> class OPR, typename SPECIFIC>
Expr<SPECIFIC> Combine(Expr<SPECIFIC> &&x, Expr<SPECIFIC> &&y) {
  static_assert(SPECIFIC::isSpecificType);
  if constexpr (SPECIFIC::category == TypeCategory::Complex &&
      (std::is_same_v<OPR<DefaultReal>, Add<DefaultReal>> ||
          std::is_same_v<OPR<DefaultReal>, Subtract<DefaultReal>>)) {
    static constexpr int kind{SPECIFIC::kind};
    using Part = Type<TypeCategory::Real, kind>;
    return AsExpr(
        ComplexConstructor<kind>{OPR<Part>{ComplexComponent<kind>{false, x},
                                     ComplexComponent<kind>{false, y}},
            OPR<Part>{ComplexComponent<kind>{true, x},
                ComplexComponent<kind>{true, y}}});
  } else {
    return AsExpr(OPR<SPECIFIC>{std::move(x), std::move(y)});
  }
}

// Given two expressions of arbitrary kind in the same intrinsic type
// category, convert one of them if necessary to the larger kind of the
// other, then combine the resulting homogenized operands with a given
// operation, returning a new expression in the same type category.
template<template<typename> class OPR, TypeCategory CAT>
Expr<SomeKind<CAT>> PromoteAndCombine(
    Expr<SomeKind<CAT>> &&x, Expr<SomeKind<CAT>> &&y) {
  return std::visit(
      [](auto &&xy) {
        using Ty = ResultType<decltype(xy[0])>;
        return AsCategoryExpr(
            Combine<OPR, Ty>(std::move(xy[0]), std::move(xy[1])));
      },
      AsSameKindExprs(std::move(x), std::move(y)));
}

// Given two expressions of arbitrary type, try to combine them with a
// binary numeric operation (e.g., Add), possibly with data type conversion of
// one of the operands to the type of the other.  Handles special cases with
// typeless literal operands and with REAL/COMPLEX exponentiation to INTEGER
// powers.
template<template<typename> class OPR>
std::optional<Expr<SomeType>> NumericOperation(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&);

extern template std::optional<Expr<SomeType>> NumericOperation<Power>(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&);
extern template std::optional<Expr<SomeType>> NumericOperation<Multiply>(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&);
extern template std::optional<Expr<SomeType>> NumericOperation<Divide>(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&);
extern template std::optional<Expr<SomeType>> NumericOperation<Add>(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&);
extern template std::optional<Expr<SomeType>> NumericOperation<Subtract>(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&);

std::optional<Expr<SomeType>> Negation(
    parser::ContextualMessages &, Expr<SomeType> &&);

// Given two expressions of arbitrary type, try to combine them with a
// relational operator (e.g., .LT.), possibly with data type conversion.
std::optional<Expr<LogicalResult>> Relate(parser::ContextualMessages &,
    RelationalOperator, Expr<SomeType> &&, Expr<SomeType> &&);

Expr<SomeLogical> LogicalNegation(Expr<SomeLogical> &&);
Expr<SomeLogical> BinaryLogicalOperation(
    LogicalOperator, Expr<SomeLogical> &&, Expr<SomeLogical> &&);

// Convenience functions and operator overloadings for expression construction.
// These interfaces are defined only for those situations that can never
// emit any message.  Use the more general templates (above) in other
// situations.

template<TypeCategory C, int K>
Expr<Type<C, K>> operator-(Expr<Type<C, K>> &&x) {
  return {Negate<Type<C, K>>{std::move(x)}};
}

template<int K>
Expr<Type<TypeCategory::Complex, K>> operator-(
    Expr<Type<TypeCategory::Complex, K>> &&x) {
  using Part = Type<TypeCategory::Real, K>;
  return {ComplexConstructor<K>{Negate<Part>{ComplexComponent<K>{false, x}},
      Negate<Part>{ComplexComponent<K>{true, x}}}};
}

template<TypeCategory C, int K>
Expr<Type<C, K>> operator+(Expr<Type<C, K>> &&x, Expr<Type<C, K>> &&y) {
  return {Combine<Add, Type<C, K>>(std::move(x), std::move(y))};
}

template<TypeCategory C, int K>
Expr<Type<C, K>> operator-(Expr<Type<C, K>> &&x, Expr<Type<C, K>> &&y) {
  return {Combine<Subtract, Type<C, K>>(std::move(x), std::move(y))};
}

template<TypeCategory C, int K>
Expr<Type<C, K>> operator*(Expr<Type<C, K>> &&x, Expr<Type<C, K>> &&y) {
  return {Combine<Multiply, Type<C, K>>(std::move(x), std::move(y))};
}

template<TypeCategory C, int K>
Expr<Type<C, K>> operator/(Expr<Type<C, K>> &&x, Expr<Type<C, K>> &&y) {
  return {Combine<Divide, Type<C, K>>(std::move(x), std::move(y))};
}

template<TypeCategory C> Expr<SomeKind<C>> operator-(Expr<SomeKind<C>> &&x) {
  return std::visit(
      [](auto &xk) { return Expr<SomeKind<C>>{-std::move(xk)}; }, x.u);
}

template<TypeCategory CAT>
Expr<SomeKind<CAT>> operator+(
    Expr<SomeKind<CAT>> &&x, Expr<SomeKind<CAT>> &&y) {
  return PromoteAndCombine<Add, CAT>(std::move(x), std::move(y));
}

template<TypeCategory CAT>
Expr<SomeKind<CAT>> operator-(
    Expr<SomeKind<CAT>> &&x, Expr<SomeKind<CAT>> &&y) {
  return PromoteAndCombine<Subtract, CAT>(std::move(x), std::move(y));
}

template<TypeCategory CAT>
Expr<SomeKind<CAT>> operator*(
    Expr<SomeKind<CAT>> &&x, Expr<SomeKind<CAT>> &&y) {
  return PromoteAndCombine<Multiply, CAT>(std::move(x), std::move(y));
}

template<TypeCategory CAT>
Expr<SomeKind<CAT>> operator/(
    Expr<SomeKind<CAT>> &&x, Expr<SomeKind<CAT>> &&y) {
  return PromoteAndCombine<Divide, CAT>(std::move(x), std::move(y));
}

}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_TOOLS_H_
