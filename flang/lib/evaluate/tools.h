// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

#include "constant.h"
#include "expression.h"
#include "traversal.h"
#include "../common/idioms.h"
#include "../common/unwrap.h"
#include "../parser/message.h"
#include "../semantics/attr.h"
#include "../semantics/symbol.h"
#include <array>
#include <optional>
#include <utility>

namespace Fortran::evaluate {

// Some expression predicates and extractors.

// When an Expr holds something that is a Variable (i.e., a Designator
// or pointer-valued FunctionRef), return a copy of its contents in
// a Variable.
template<typename A>
std::optional<Variable<A>> AsVariable(const Expr<A> &expr) {
  using Variant = decltype(Variable<A>::u);
  return std::visit(
      [](const auto &x) -> std::optional<Variable<A>> {
        if constexpr (common::HasMember<std::decay_t<decltype(x)>, Variant>) {
          return std::make_optional<Variable<A>>(x);
        }
        return std::nullopt;
      },
      expr.u);
}

template<typename A>
std::optional<Variable<A>> AsVariable(const std::optional<Expr<A>> &expr) {
  if (expr.has_value()) {
    return AsVariable(*expr);
  } else {
    return std::nullopt;
  }
}

// Predicate: true when an expression is a variable reference, not an
// operation.  Be advised: a call to a function that returns an object
// pointer is a "variable" in Fortran (it can be the left-hand side of
// an assignment).
struct IsVariableVisitor : public virtual VisitorBase<std::optional<bool>> {
  // std::optional<> is used because it is default-constructible.
  using Result = std::optional<bool>;
  explicit IsVariableVisitor(std::nullptr_t) {}
  void Handle(const StaticDataObject &) { Return(false); }
  void Handle(const Symbol &) { Return(true); }
  void Pre(const Component &) { Return(true); }
  void Pre(const ArrayRef &) { Return(true); }
  void Pre(const CoarrayRef &) { Return(true); }
  void Pre(const ComplexPart &) { Return(true); }
  void Handle(const ProcedureDesignator &);
  template<TypeCategory CAT, int KIND>
  void Pre(const Expr<Type<CAT, KIND>> &x) {
    if (!std::holds_alternative<Designator<Type<CAT, KIND>>>(x.u) &&
        !std::holds_alternative<FunctionRef<Type<CAT, KIND>>>(x.u)) {
      Return(false);
    }
  }
  void Pre(const Expr<SomeDerived> &x) {
    if (!std::holds_alternative<Designator<SomeDerived>>(x.u) &&
        !std::holds_alternative<FunctionRef<SomeDerived>>(x.u)) {
      Return(false);
    }
  }
  template<typename A> void Post(const A &) { Return(false); }
};

template<typename A> bool IsVariable(const A &x) {
  Visitor<IsVariableVisitor> visitor{nullptr};
  if (auto optional{visitor.Traverse(x)}) {
    return *optional;
  } else {
    return false;
  }
}

// Predicate: true when an expression is assumed-rank
bool IsAssumedRank(const semantics::Symbol &);
bool IsAssumedRank(const ActualArgument &);
template<typename A> bool IsAssumedRank(const A &) { return false; }
template<typename A> bool IsAssumedRank(const Designator<A> &designator) {
  if (const auto *symbol{
          std::get_if<const semantics::Symbol *>(&designator.u)}) {
    return IsAssumedRank(*symbol);
  } else {
    return false;
  }
}
template<typename T> bool IsAssumedRank(const Expr<T> &expr) {
  return std::visit([](const auto &x) { return IsAssumedRank(x); }, expr.u);
}
template<typename A> bool IsAssumedRank(const std::optional<A> &x) {
  return x.has_value() && IsAssumedRank(*x);
}

// Generalizing packagers: these take operations and expressions of more
// specific types and wrap them in Expr<> containers of more abstract types.

template<typename A> common::IfNoLvalue<Expr<ResultType<A>>, A> AsExpr(A &&x) {
  return Expr<ResultType<A>>{std::move(x)};
}

template<typename T> Expr<T> AsExpr(Expr<T> &&x) {
  static_assert(IsSpecificIntrinsicType<T>);
  return std::move(x);
}

template<TypeCategory CATEGORY>
Expr<SomeKind<CATEGORY>> AsCategoryExpr(Expr<SomeKind<CATEGORY>> &&x) {
  return std::move(x);
}

template<typename A>
common::IfNoLvalue<Expr<SomeType>, A> AsGenericExpr(A &&x) {
  if constexpr (common::HasMember<A, TypelessExpression>) {
    return Expr<SomeType>{std::move(x)};
  } else {
    return Expr<SomeType>{AsCategoryExpr(std::move(x))};
  }
}

template<typename A>
common::IfNoLvalue<Expr<SomeKind<ResultType<A>::category>>, A> AsCategoryExpr(
    A &&x) {
  return Expr<SomeKind<ResultType<A>::category>>{AsExpr(std::move(x))};
}

inline Expr<SomeType> AsGenericExpr(Expr<SomeType> &&x) { return std::move(x); }

Expr<SomeReal> GetComplexPart(
    const Expr<SomeComplex> &, bool isImaginary = false);

template<int KIND>
Expr<SomeComplex> MakeComplex(Expr<Type<TypeCategory::Real, KIND>> &&re,
    Expr<Type<TypeCategory::Real, KIND>> &&im) {
  return AsCategoryExpr(ComplexConstructor<KIND>{std::move(re), std::move(im)});
}

template<typename A> constexpr bool IsNumericCategoryExpr() {
  if constexpr (common::HasMember<A, TypelessExpression>) {
    return false;
  } else {
    return common::HasMember<ResultType<A>, NumericCategoryTypes>;
  }
}

// Specializing extractor.  If an Expr wraps some type of object, perhaps
// in several layers, return a pointer to it; otherwise null.  Also works
// with expressions contained in ActualArgument.
template<typename A, typename B>
auto UnwrapExpr(B &x) -> common::Constify<A, B> * {
  using Ty = std::decay_t<B>;
  if constexpr (std::is_same_v<A, Ty>) {
    return &x;
  } else if constexpr (std::is_same_v<Ty, ActualArgument>) {
    if (auto *expr{x.UnwrapExpr()}) {
      return UnwrapExpr<A>(*expr);
    }
  } else if constexpr (std::is_same_v<Ty, Expr<SomeType>>) {
    return std::visit([](auto &x) { return UnwrapExpr<A>(x); }, x.u);
  } else if constexpr (!common::HasMember<A, TypelessExpression>) {
    if constexpr (std::is_same_v<Ty, Expr<ResultType<A>>> ||
        std::is_same_v<Ty, Expr<SomeKind<ResultType<A>::category>>>) {
      return std::visit([](auto &x) { return UnwrapExpr<A>(x); }, x.u);
    }
  }
  return nullptr;
}

template<typename A, typename B>
const A *UnwrapExpr(const std::optional<B> &x) {
  if (x.has_value()) {
    return UnwrapExpr<A>(*x);
  } else {
    return nullptr;
  }
}

template<typename A, typename B> A *UnwrapExpr(std::optional<B> &x) {
  if (x.has_value()) {
    return UnwrapExpr<A>(*x);
  } else {
    return nullptr;
  }
}

// If an expression simply wraps a DataRef, extract and return it.
template<typename A>
common::IfNoLvalue<std::optional<DataRef>, A> ExtractDataRef(const A &) {
  return std::nullopt;  // default base casec
}
template<typename T>
std::optional<DataRef> ExtractDataRef(const Designator<T> &d) {
  return std::visit(
      [](const auto &x) -> std::optional<DataRef> {
        if constexpr (common::HasMember<decltype(x), decltype(DataRef::u)>) {
          return DataRef{x};
        }
        return std::nullopt;
      },
      d.u);
}
template<typename T>
std::optional<DataRef> ExtractDataRef(const Expr<T> &expr) {
  return std::visit([](const auto &x) { return ExtractDataRef(x); }, expr.u);
}
template<typename A>
std::optional<DataRef> ExtractDataRef(const std::optional<A> &x) {
  if (x.has_value()) {
    return ExtractDataRef(*x);
  } else {
    return std::nullopt;
  }
}

// If an expression is simply a whole symbol data designator,
// extract and return that symbol, else null.
template<typename A> const Symbol *UnwrapWholeSymbolDataRef(const A &x) {
  if (auto dataRef{ExtractDataRef(x)}) {
    if (const Symbol **p{std::get_if<const Symbol *>(&dataRef->u)}) {
      return *p;
    }
  }
  return nullptr;
}

// Creation of conversion expressions can be done to either a known
// specific intrinsic type with ConvertToType<T>(x) or by converting
// one arbitrary expression to the type of another with ConvertTo(to, from).

template<typename TO, TypeCategory FROMCAT>
Expr<TO> ConvertToType(Expr<SomeKind<FROMCAT>> &&x) {
  static_assert(IsSpecificIntrinsicType<TO>);
  if constexpr (FROMCAT != TO::category) {
    if constexpr (TO::category == TypeCategory::Complex) {
      using Part = typename TO::Part;
      Scalar<Part> zero;
      return Expr<TO>{ComplexConstructor<TO::kind>{
          ConvertToType<Part>(std::move(x)), Expr<Part>{Constant<Part>{zero}}}};
    } else if constexpr (FROMCAT == TypeCategory::Complex) {
      // Extract and convert the real component of a complex value
      return std::visit(
          [&](auto &&z) {
            using ZType = ResultType<decltype(z)>;
            using Part = typename ZType::Part;
            return ConvertToType<TO, TypeCategory::Real>(Expr<SomeReal>{
                Expr<Part>{ComplexComponent<Part::kind>{false, std::move(z)}}});
          },
          std::move(x.u));
    } else {
      return Expr<TO>{Convert<TO, FROMCAT>{std::move(x)}};
    }
  } else {
    // Same type category
    if (auto *already{std::get_if<Expr<TO>>(&x.u)}) {
      return std::move(*already);
    }
    if constexpr (TO::category == TypeCategory::Complex) {
      // Extract, convert, and recombine the components.
      return Expr<TO>{std::visit(
          [](auto &z) {
            using FromType = ResultType<decltype(z)>;
            using FromPart = typename FromType::Part;
            using FromGeneric = SomeKind<TypeCategory::Real>;
            using ToPart = typename TO::Part;
            Convert<ToPart, TypeCategory::Real> re{Expr<FromGeneric>{
                Expr<FromPart>{ComplexComponent<FromType::kind>{false, z}}}};
            Convert<ToPart, TypeCategory::Real> im{Expr<FromGeneric>{
                Expr<FromPart>{ComplexComponent<FromType::kind>{true, z}}}};
            return ComplexConstructor<TO::kind>{
                AsExpr(std::move(re)), AsExpr(std::move(im))};
          },
          x.u)};
    } else {
      return Expr<TO>{Convert<TO, TO::category>{std::move(x)}};
    }
  }
}

template<typename TO, TypeCategory FROMCAT, int FROMKIND>
Expr<TO> ConvertToType(Expr<Type<FROMCAT, FROMKIND>> &&x) {
  return ConvertToType<TO, FROMCAT>(Expr<SomeKind<FROMCAT>>{std::move(x)});
}

template<typename TO> Expr<TO> ConvertToType(BOZLiteralConstant &&x) {
  static_assert(IsSpecificIntrinsicType<TO>);
  if constexpr (TO::category == TypeCategory::Integer) {
    return Expr<TO>{
        Constant<TO>{Scalar<TO>::ConvertUnsigned(std::move(x)).value}};
  } else {
    static_assert(TO::category == TypeCategory::Real);
    using Word = typename Scalar<TO>::Word;
    return Expr<TO>{
        Constant<TO>{Scalar<TO>{Word::ConvertUnsigned(std::move(x)).value}}};
  }
}

// Conversions to dynamic types
std::optional<Expr<SomeType>> ConvertToType(
    const DynamicType &, Expr<SomeType> &&);
std::optional<Expr<SomeType>> ConvertToType(
    const DynamicType &, std::optional<Expr<SomeType>> &&);
std::optional<Expr<SomeType>> ConvertToType(
    const semantics::Symbol &, Expr<SomeType> &&);
std::optional<Expr<SomeType>> ConvertToType(
    const semantics::Symbol &, std::optional<Expr<SomeType>> &&);

// Conversions to the type of another expression
template<TypeCategory TC, int TK, typename FROM>
common::IfNoLvalue<Expr<Type<TC, TK>>, FROM> ConvertTo(
    const Expr<Type<TC, TK>> &, FROM &&x) {
  return ConvertToType<Type<TC, TK>>(std::move(x));
}

template<TypeCategory TC, typename FROM>
common::IfNoLvalue<Expr<SomeKind<TC>>, FROM> ConvertTo(
    const Expr<SomeKind<TC>> &to, FROM &&from) {
  return std::visit(
      [&](const auto &toKindExpr) {
        using KindExpr = std::decay_t<decltype(toKindExpr)>;
        return AsCategoryExpr(
            ConvertToType<ResultType<KindExpr>>(std::move(from)));
      },
      to.u);
}

template<typename FROM>
common::IfNoLvalue<Expr<SomeType>, FROM> ConvertTo(
    const Expr<SomeType> &to, FROM &&from) {
  return std::visit(
      [&](const auto &toCatExpr) {
        return AsGenericExpr(ConvertTo(toCatExpr, std::move(from)));
      },
      to.u);
}

// Convert an expression of some known category to a dynamically chosen
// kind of some category (usually but not necessarily distinct).
template<TypeCategory TOCAT, typename VALUE> struct ConvertToKindHelper {
  using Result = std::optional<Expr<SomeKind<TOCAT>>>;
  using Types = CategoryTypes<TOCAT>;
  ConvertToKindHelper(int k, VALUE &&x) : kind{k}, value{std::move(x)} {}
  template<typename T> Result Test() {
    if (kind == T::kind) {
      return std::make_optional(
          AsCategoryExpr(ConvertToType<T>(std::move(value))));
    }
    return std::nullopt;
  }
  int kind;
  VALUE value;
};

template<TypeCategory TOCAT, typename VALUE>
common::IfNoLvalue<Expr<SomeKind<TOCAT>>, VALUE> ConvertToKind(
    int kind, VALUE &&x) {
  return common::SearchTypes(
      ConvertToKindHelper<TOCAT, VALUE>{kind, std::move(x)})
      .value();
}

// Given a type category CAT, SameKindExprs<CAT, N> is a variant that
// holds an arrays of expressions of the same supported kind in that
// category.
template<typename A, int N = 2> using SameExprs = std::array<Expr<A>, N>;
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
ConvertRealOperandsResult ConvertRealOperands(parser::ContextualMessages &,
    Expr<SomeType> &&, Expr<SomeType> &&, int defaultRealKind);

// Per F'2018 R718, if both components are INTEGER, they are both converted
// to default REAL and the result is default COMPLEX.  Otherwise, the
// kind of the result is the kind of most precise REAL component, and the other
// component is converted if necessary to its type.
std::optional<Expr<SomeComplex>> ConstructComplex(parser::ContextualMessages &,
    Expr<SomeType> &&, Expr<SomeType> &&, int defaultRealKind);
std::optional<Expr<SomeComplex>> ConstructComplex(parser::ContextualMessages &,
    std::optional<Expr<SomeType>> &&, std::optional<Expr<SomeType>> &&,
    int defaultRealKind);

template<typename A> Expr<TypeOf<A>> ScalarConstantToExpr(const A &x) {
  using Ty = TypeOf<A>;
  static_assert(
      std::is_same_v<Scalar<Ty>, std::decay_t<A>> || !"TypeOf<> is broken");
  return Expr<TypeOf<A>>{Constant<Ty>{x}};
}

// Combine two expressions of the same specific numeric type with an operation
// to produce a new expression.  Implements piecewise addition and subtraction
// for COMPLEX.
template<template<typename> class OPR, typename SPECIFIC>
Expr<SPECIFIC> Combine(Expr<SPECIFIC> &&x, Expr<SPECIFIC> &&y) {
  static_assert(IsSpecificIntrinsicType<SPECIFIC>);
  if constexpr (SPECIFIC::category == TypeCategory::Complex &&
      (std::is_same_v<OPR<LargestReal>, Add<LargestReal>> ||
          std::is_same_v<OPR<LargestReal>, Subtract<LargestReal>>)) {
    static constexpr int kind{SPECIFIC::kind};
    using Part = Type<TypeCategory::Real, kind>;
    return AsExpr(ComplexConstructor<kind>{
        AsExpr(OPR<Part>{AsExpr(ComplexComponent<kind>{false, x}),
            AsExpr(ComplexComponent<kind>{false, y})}),
        AsExpr(OPR<Part>{AsExpr(ComplexComponent<kind>{true, x}),
            AsExpr(ComplexComponent<kind>{true, y})})});
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
std::optional<Expr<SomeType>> NumericOperation(parser::ContextualMessages &,
    Expr<SomeType> &&, Expr<SomeType> &&, int defaultRealKind);

extern template std::optional<Expr<SomeType>> NumericOperation<Power>(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&,
    int defaultRealKind);
extern template std::optional<Expr<SomeType>> NumericOperation<Multiply>(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&,
    int defaultRealKind);
extern template std::optional<Expr<SomeType>> NumericOperation<Divide>(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&,
    int defaultRealKind);
extern template std::optional<Expr<SomeType>> NumericOperation<Add>(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&,
    int defaultRealKind);
extern template std::optional<Expr<SomeType>> NumericOperation<Subtract>(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&,
    int defaultRealKind);

std::optional<Expr<SomeType>> Negation(
    parser::ContextualMessages &, Expr<SomeType> &&);

// Given two expressions of arbitrary type, try to combine them with a
// relational operator (e.g., .LT.), possibly with data type conversion.
std::optional<Expr<LogicalResult>> Relate(parser::ContextualMessages &,
    RelationalOperator, Expr<SomeType> &&, Expr<SomeType> &&);

template<int K>
Expr<Type<TypeCategory::Logical, K>> LogicalNegation(
    Expr<Type<TypeCategory::Logical, K>> &&x) {
  return AsExpr(Not<K>{std::move(x)});
}

Expr<SomeLogical> LogicalNegation(Expr<SomeLogical> &&);

template<int K>
Expr<Type<TypeCategory::Logical, K>> BinaryLogicalOperation(LogicalOperator opr,
    Expr<Type<TypeCategory::Logical, K>> &&x,
    Expr<Type<TypeCategory::Logical, K>> &&y) {
  return AsExpr(LogicalOperation<K>{opr, std::move(x), std::move(y)});
}

Expr<SomeLogical> BinaryLogicalOperation(
    LogicalOperator, Expr<SomeLogical> &&, Expr<SomeLogical> &&);

// Convenience functions and operator overloadings for expression construction.
// These interfaces are defined only for those situations that can never
// emit any message.  Use the more general templates (above) in other
// situations.

template<TypeCategory C, int K>
Expr<Type<C, K>> operator-(Expr<Type<C, K>> &&x) {
  return AsExpr(Negate<Type<C, K>>{std::move(x)});
}

template<int K>
Expr<Type<TypeCategory::Complex, K>> operator-(
    Expr<Type<TypeCategory::Complex, K>> &&x) {
  using Part = Type<TypeCategory::Real, K>;
  return AsExpr(ComplexConstructor<K>{
      AsExpr(Negate<Part>{AsExpr(ComplexComponent<K>{false, x})}),
      AsExpr(Negate<Part>{AsExpr(ComplexComponent<K>{true, x})})});
}

template<TypeCategory C, int K>
Expr<Type<C, K>> operator+(Expr<Type<C, K>> &&x, Expr<Type<C, K>> &&y) {
  return AsExpr(Combine<Add, Type<C, K>>(std::move(x), std::move(y)));
}

template<TypeCategory C, int K>
Expr<Type<C, K>> operator-(Expr<Type<C, K>> &&x, Expr<Type<C, K>> &&y) {
  return AsExpr(Combine<Subtract, Type<C, K>>(std::move(x), std::move(y)));
}

template<TypeCategory C, int K>
Expr<Type<C, K>> operator*(Expr<Type<C, K>> &&x, Expr<Type<C, K>> &&y) {
  return AsExpr(Combine<Multiply, Type<C, K>>(std::move(x), std::move(y)));
}

template<TypeCategory C, int K>
Expr<Type<C, K>> operator/(Expr<Type<C, K>> &&x, Expr<Type<C, K>> &&y) {
  return AsExpr(Combine<Divide, Type<C, K>>(std::move(x), std::move(y)));
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

// A utility for use with common::SearchTypes to create generic expressions
// when an intrinsic type category for (say) a variable is known
// but the kind parameter value is not.
template<TypeCategory CAT, template<typename> class TEMPLATE, typename VALUE>
struct TypeKindVisitor {
  using Result = std::optional<Expr<SomeType>>;
  using Types = CategoryTypes<CAT>;

  TypeKindVisitor(int k, VALUE &&x) : kind{k}, value{std::move(x)} {}
  TypeKindVisitor(int k, const VALUE &x) : kind{k}, value{x} {}

  template<typename T> Result Test() {
    if (kind == T::kind) {
      return AsGenericExpr(TEMPLATE<T>{std::move(value)});
    }
    return std::nullopt;
  }

  int kind;
  VALUE value;
};

// GetLastSymbol() returns the rightmost symbol in an object or procedure
// designator (which has perhaps been wrapped in an Expr<>), or a null pointer
// when none is found.
struct GetLastSymbolVisitor
  : public virtual VisitorBase<std::optional<const semantics::Symbol *>> {
  // std::optional<> is used because it is default-constructible.
  using Result = std::optional<const semantics::Symbol *>;
  explicit GetLastSymbolVisitor(std::nullptr_t) {}
  void Handle(const semantics::Symbol &x) { Return(&x); }
  void Handle(const Component &x) { Return(&x.GetLastSymbol()); }
  void Handle(const NamedEntity &x) { Return(&x.GetLastSymbol()); }
  void Handle(const ProcedureDesignator &x) { Return(x.GetSymbol()); }
  template<TypeCategory CAT, int KIND>
  void Pre(const Expr<Type<CAT, KIND>> &x) {
    if (!std::holds_alternative<Designator<Type<CAT, KIND>>>(x.u)) {
      Return(nullptr);
    }
  }
  void Pre(const Expr<SomeDerived> &x) {
    if (!std::holds_alternative<Designator<SomeDerived>>(x.u)) {
      Return(nullptr);
    }
  }
};

template<typename A> const semantics::Symbol *GetLastSymbol(const A &x) {
  Visitor<GetLastSymbolVisitor> visitor{nullptr};
  if (auto optional{visitor.Traverse(x)}) {
    return *optional;
  } else {
    return nullptr;
  }
}

// Convenience: If GetLastSymbol() succeeds on the argument, return its
// set of attributes, otherwise the empty set.
template<typename A> semantics::Attrs GetAttrs(const A &x) {
  if (const semantics::Symbol * symbol{GetLastSymbol(x)}) {
    return symbol->attrs();
  } else {
    return {};
  }
}

// GetBaseObject()
template<typename A> std::optional<BaseObject> GetBaseObject(const A &) {
  return std::nullopt;
}
template<typename T>
std::optional<BaseObject> GetBaseObject(const Designator<T> &x) {
  return x.GetBaseObject();
}
template<typename T> std::optional<BaseObject> GetBaseObject(const Expr<T> &x) {
  return std::visit([](const auto &y) { return GetBaseObject(y); }, x.u);
}
template<typename A>
std::optional<BaseObject> GetBaseObject(const std::optional<A> &x) {
  if (x.has_value()) {
    return GetBaseObject(*x);
  } else {
    return std::nullopt;
  }
}

// Predicate: IsAllocatableOrPointer()
template<typename A> bool IsAllocatableOrPointer(const A &x) {
  return GetAttrs(x).HasAny(
      semantics::Attrs{semantics::Attr::POINTER, semantics::Attr::ALLOCATABLE});
}

// Predicate: IsProcedurePointer()
template<typename A> bool IsProcedurePointer(const A &) { return false; }
inline bool IsProcedurePointer(const ProcedureDesignator &) { return true; }
inline bool IsProcedurePointer(const ProcedureRef &) { return true; }
inline bool IsProcedurePointer(const Expr<SomeType> &expr) {
  return std::visit(
      [](const auto &x) { return IsProcedurePointer(x); }, expr.u);
}
template<typename A> bool IsProcedurePointer(const std::optional<A> &x) {
  return x.has_value() && IsProcedurePointer(*x);
}

// GetLastTarget() returns the rightmost symbol in an object
// designator (which has perhaps been wrapped in an Expr<>) that has the
// POINTER or TARGET attribute, or a null pointer when none is found.
struct GetLastTargetVisitor
  : public virtual VisitorBase<std::optional<const semantics::Symbol *>> {
  // std::optional<> is used because it is default-constructible.
  using Result = std::optional<const semantics::Symbol *>;
  explicit GetLastTargetVisitor(std::nullptr_t);
  void Handle(const semantics::Symbol &);
  void Pre(const Component &);
};

template<typename A> const semantics::Symbol *GetLastTarget(const A &x) {
  Visitor<GetLastTargetVisitor> visitor{nullptr};
  if (auto optional{visitor.Traverse(x)}) {
    return *optional;
  } else {
    return nullptr;
  }
}

}
#endif  // FORTRAN_EVALUATE_TOOLS_H_
