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

#include "fold.h"
#include "character.h"
#include "characteristics.h"
#include "common.h"
#include "constant.h"
#include "expression.h"
#include "host.h"
#include "int-power.h"
#include "intrinsics-library-templates.h"
#include "shape.h"
#include "tools.h"
#include "traversal.h"
#include "type.h"
#include "../common/indirection.h"
#include "../common/template.h"
#include "../common/unwrap.h"
#include "../parser/message.h"
#include "../semantics/scope.h"
#include "../semantics/symbol.h"
#include <cmath>
#include <complex>
#include <cstdio>
#include <optional>
#include <sstream>
#include <type_traits>
#include <variant>

// Some environments, viz. clang on Darwin, allow the macro HUGE
// to leak out of <math.h> even when it is never directly included.
#undef HUGE

namespace Fortran::evaluate {

// FoldOperation() rewrites expression tree nodes.
// If there is any possibility that the rewritten node will
// not have the same representation type, the result of
// FoldOperation() will be packaged in an Expr<> of the same
// specific type.

// no-op base case
template<typename A>
common::IfNoLvalue<Expr<ResultType<A>>, A> FoldOperation(
    FoldingContext &, A &&x) {
  static_assert(!std::is_same_v<A, Expr<ResultType<A>>> &&
      "call Fold() instead for Expr<>");
  return Expr<ResultType<A>>{std::move(x)};
}

// Forward declarations of overloads, template instantiations, and template
// specializations of FoldOperation() to enable mutual recursion between them.
static Component FoldOperation(FoldingContext &, Component &&);
static NamedEntity FoldOperation(FoldingContext &, NamedEntity &&);
static Triplet FoldOperation(FoldingContext &, Triplet &&);
static Subscript FoldOperation(FoldingContext &, Subscript &&);
static ArrayRef FoldOperation(FoldingContext &, ArrayRef &&);
static CoarrayRef FoldOperation(FoldingContext &, CoarrayRef &&);
static DataRef FoldOperation(FoldingContext &, DataRef &&);
static Substring FoldOperation(FoldingContext &, Substring &&);
static ComplexPart FoldOperation(FoldingContext &, ComplexPart &&);
template<typename T>
Expr<T> FoldOperation(FoldingContext &context, FunctionRef<T> &&);
template<int KIND>
Expr<Type<TypeCategory::Integer, KIND>> FoldIntrinsicFunction(
    FoldingContext &context, FunctionRef<Type<TypeCategory::Integer, KIND>> &&);
template<int KIND>
Expr<Type<TypeCategory::Real, KIND>> FoldIntrinsicFunction(
    FoldingContext &context, FunctionRef<Type<TypeCategory::Real, KIND>> &&);
template<int KIND>
Expr<Type<TypeCategory::Complex, KIND>> FoldIntrinsicFunction(
    FoldingContext &context, FunctionRef<Type<TypeCategory::Complex, KIND>> &&);
template<int KIND>
Expr<Type<TypeCategory::Logical, KIND>> FoldIntrinsicFunction(
    FoldingContext &context, FunctionRef<Type<TypeCategory::Logical, KIND>> &&);
template<typename T> Expr<T> FoldOperation(FoldingContext &, Designator<T> &&);

template<int KIND>
Expr<Type<TypeCategory::Integer, KIND>> FoldOperation(
    FoldingContext &, TypeParamInquiry<KIND> &&);
static Expr<ImpliedDoIndex::Result> FoldOperation(
    FoldingContext &context, ImpliedDoIndex &&);
template<typename T>
Expr<T> FoldOperation(FoldingContext &, ArrayConstructor<T> &&);
static Expr<SomeDerived> FoldOperation(
    FoldingContext &, StructureConstructor &&);

// Overloads, instantiations, and specializations of FoldOperation().

Component FoldOperation(FoldingContext &context, Component &&component) {
  return {FoldOperation(context, std::move(component.base())),
      component.GetLastSymbol()};
}

NamedEntity FoldOperation(FoldingContext &context, NamedEntity &&x) {
  if (Component * c{x.UnwrapComponent()}) {
    return NamedEntity{FoldOperation(context, std::move(*c))};
  } else {
    return std::move(x);
  }
}

Triplet FoldOperation(FoldingContext &context, Triplet &&triplet) {
  MaybeExtentExpr lower{triplet.lower()};
  MaybeExtentExpr upper{triplet.upper()};
  return {Fold(context, std::move(lower)), Fold(context, std::move(upper)),
      Fold(context, triplet.stride())};
}

Subscript FoldOperation(FoldingContext &context, Subscript &&subscript) {
  return std::visit(
      common::visitors{
          [&](IndirectSubscriptIntegerExpr &&expr) {
            expr.value() = Fold(context, std::move(expr.value()));
            return Subscript(std::move(expr));
          },
          [&](Triplet &&triplet) {
            return Subscript(FoldOperation(context, std::move(triplet)));
          },
      },
      std::move(subscript.u));
}

ArrayRef FoldOperation(FoldingContext &context, ArrayRef &&arrayRef) {
  NamedEntity base{FoldOperation(context, std::move(arrayRef.base()))};
  for (Subscript &subscript : arrayRef.subscript()) {
    subscript = FoldOperation(context, std::move(subscript));
  }
  return ArrayRef{std::move(base), std::move(arrayRef.subscript())};
}

CoarrayRef FoldOperation(FoldingContext &context, CoarrayRef &&coarrayRef) {
  std::vector<Subscript> subscript;
  for (Subscript x : coarrayRef.subscript()) {
    subscript.emplace_back(FoldOperation(context, std::move(x)));
  }
  std::vector<Expr<SubscriptInteger>> cosubscript;
  for (Expr<SubscriptInteger> x : coarrayRef.cosubscript()) {
    cosubscript.emplace_back(Fold(context, std::move(x)));
  }
  CoarrayRef folded{std::move(coarrayRef.base()), std::move(subscript),
      std::move(cosubscript)};
  if (std::optional<Expr<SomeInteger>> stat{coarrayRef.stat()}) {
    folded.set_stat(Fold(context, std::move(*stat)));
  }
  if (std::optional<Expr<SomeInteger>> team{coarrayRef.team()}) {
    folded.set_team(
        Fold(context, std::move(*team)), coarrayRef.teamIsTeamNumber());
  }
  return folded;
}

DataRef FoldOperation(FoldingContext &context, DataRef &&dataRef) {
  return std::visit(
      common::visitors{
          [&](const Symbol *symbol) { return DataRef{*symbol}; },
          [&](auto &&x) {
            return DataRef{FoldOperation(context, std::move(x))};
          },
      },
      std::move(dataRef.u));
}

Substring FoldOperation(FoldingContext &context, Substring &&substring) {
  auto lower{Fold(context, substring.lower())};
  auto upper{Fold(context, substring.upper())};
  if (const DataRef * dataRef{substring.GetParentIf<DataRef>()}) {
    return Substring{FoldOperation(context, DataRef{*dataRef}),
        std::move(lower), std::move(upper)};
  } else {
    auto p{*substring.GetParentIf<StaticDataObject::Pointer>()};
    return Substring{std::move(p), std::move(lower), std::move(upper)};
  }
}

ComplexPart FoldOperation(FoldingContext &context, ComplexPart &&complexPart) {
  DataRef complex{complexPart.complex()};
  return ComplexPart{
      FoldOperation(context, std::move(complex)), complexPart.part()};
}

// helpers to fold intrinsic function references
// Define callable types used in a common utility that
// takes care of array and cast/conversion aspects for elemental intrinsics

template<typename TR, typename... TArgs>
using ScalarFunc = std::function<Scalar<TR>(const Scalar<TArgs> &...)>;
template<typename TR, typename... TArgs>
using ScalarFuncWithContext =
    std::function<Scalar<TR>(FoldingContext &, const Scalar<TArgs> &...)>;

// Apply type conversion and re-folding if necessary.
// This is where BOZ arguments are converted.
template<typename T>
static inline Constant<T> *FoldConvertedArg(
    FoldingContext &context, std::optional<ActualArgument> &arg) {
  if (arg.has_value()) {
    if (auto *expr{arg->UnwrapExpr()}) {
      if (UnwrapExpr<Expr<T>>(*expr) == nullptr) {
        if (auto converted{ConvertToType(T::GetType(), std::move(*expr))}) {
          *expr = Fold(context, std::move(*converted));
        }
      }
      return UnwrapConstantValue<T>(*expr);
    }
  }
  return nullptr;
}

template<template<typename, typename...> typename WrapperType, typename TR,
    typename... TA, std::size_t... I>
static inline Expr<TR> FoldElementalIntrinsicHelper(FoldingContext &context,
    FunctionRef<TR> &&funcRef, WrapperType<TR, TA...> func,
    std::index_sequence<I...>) {
  static_assert(
      (... && IsSpecificIntrinsicType<TA>));  // TODO derived types for MERGE?
  static_assert(sizeof...(TA) > 0);
  std::tuple<const Constant<TA> *...> args{
      FoldConvertedArg<TA>(context, funcRef.arguments()[I])...};
  if ((... && (std::get<I>(args) != nullptr))) {
    // Compute the shape of the result based on shapes of arguments
    ConstantSubscripts shape;
    int rank{0};
    const ConstantSubscripts *shapes[sizeof...(TA)]{
        &std::get<I>(args)->shape()...};
    const int ranks[sizeof...(TA)]{std::get<I>(args)->Rank()...};
    for (unsigned int i{0}; i < sizeof...(TA); ++i) {
      if (ranks[i] > 0) {
        if (rank == 0) {
          rank = ranks[i];
          shape = *shapes[i];
        } else {
          if (shape != *shapes[i]) {
            // TODO: Rank compatibility was already checked but it seems to be
            // the first place where the actual shapes are checked to be the
            // same. Shouldn't this be checked elsewhere so that this is also
            // checked for non constexpr call to elemental intrinsics function?
            context.messages().Say(
                "Arguments in elemental intrinsic function are not conformable"_err_en_US);
            return Expr<TR>{std::move(funcRef)};
          }
        }
      }
    }
    CHECK(rank == GetRank(shape));

    // Compute all the scalar values of the results
    std::vector<Scalar<TR>> results;
    if (TotalElementCount(shape) > 0) {
      ConstantBounds bounds{shape};
      ConstantSubscripts index(rank, 1);
      do {
        if constexpr (std::is_same_v<WrapperType<TR, TA...>,
                          ScalarFuncWithContext<TR, TA...>>) {
          results.emplace_back(func(context,
              (ranks[I] ? std::get<I>(args)->At(index)
                        : std::get<I>(args)->GetScalarValue().value())...));
        } else if constexpr (std::is_same_v<WrapperType<TR, TA...>,
                                 ScalarFunc<TR, TA...>>) {
          results.emplace_back(func(
              (ranks[I] ? std::get<I>(args)->At(index)
                        : std::get<I>(args)->GetScalarValue().value())...));
        }
      } while (bounds.IncrementSubscripts(index));
    }
    // Build and return constant result
    if constexpr (TR::category == TypeCategory::Character) {
      auto len{static_cast<ConstantSubscript>(
          results.size() ? results[0].length() : 0)};
      return Expr<TR>{Constant<TR>{len, std::move(results), std::move(shape)}};
    } else {
      return Expr<TR>{Constant<TR>{std::move(results), std::move(shape)}};
    }
  }
  return Expr<TR>{std::move(funcRef)};
}

template<typename TR, typename... TA>
Expr<TR> FoldElementalIntrinsic(FoldingContext &context,
    FunctionRef<TR> &&funcRef, ScalarFunc<TR, TA...> func) {
  return FoldElementalIntrinsicHelper<ScalarFunc, TR, TA...>(
      context, std::move(funcRef), func, std::index_sequence_for<TA...>{});
}
template<typename TR, typename... TA>
Expr<TR> FoldElementalIntrinsic(FoldingContext &context,
    FunctionRef<TR> &&funcRef, ScalarFuncWithContext<TR, TA...> func) {
  return FoldElementalIntrinsicHelper<ScalarFuncWithContext, TR, TA...>(
      context, std::move(funcRef), func, std::index_sequence_for<TA...>{});
}

static std::optional<std::int64_t> GetInt64Arg(
    const std::optional<ActualArgument> &arg) {
  if (const auto *intExpr{UnwrapExpr<Expr<SomeInteger>>(arg)}) {
    return ToInt64(*intExpr);
  } else {
    return std::nullopt;
  }
}

static std::optional<std::int64_t> GetInt64ArgOr(
    const std::optional<ActualArgument> &arg, std::int64_t defaultValue) {
  if (!arg.has_value()) {
    return defaultValue;
  } else if (const auto *intExpr{UnwrapExpr<Expr<SomeInteger>>(arg)}) {
    return ToInt64(*intExpr);
  } else {
    return std::nullopt;
  }
}

template<typename A, typename B>
std::optional<std::vector<A>> GetIntegerVector(const B &x) {
  static_assert(std::is_integral_v<A>);
  if (const auto *someInteger{UnwrapExpr<Expr<SomeInteger>>(x)}) {
    return std::visit(
        [](const auto &typedExpr) -> std::optional<std::vector<A>> {
          using T = ResultType<decltype(typedExpr)>;
          if (const auto *constant{UnwrapConstantValue<T>(typedExpr)}) {
            if (constant->Rank() == 1) {
              std::vector<A> result;
              for (const auto &value : constant->values()) {
                result.push_back(static_cast<A>(value.ToInt64()));
              }
              return result;
            }
          }
          return std::nullopt;
        },
        someInteger->u);
  }
  return std::nullopt;
}

// Transform an intrinsic function reference that contains user errors
// into an intrinsic with the same characteristic but the "invalid" name.
// This to prevent generating warnings over and over if the expression
// gets re-folded.
template<typename T> Expr<T> MakeInvalidIntrinsic(FunctionRef<T> &&funcRef) {
  SpecificIntrinsic invalid{std::get<SpecificIntrinsic>(funcRef.proc().u)};
  invalid.name = "invalid";
  return Expr<T>{FunctionRef<T>{
      ProcedureDesignator{std::move(invalid)}, std::move(funcRef.arguments())}};
}

template<typename T>
Expr<T> Reshape(FoldingContext &context, FunctionRef<T> &&funcRef) {
  auto args{funcRef.arguments()};
  CHECK(args.size() == 4);
  const auto *source{UnwrapConstantValue<T>(args[0])};
  const auto *pad{UnwrapConstantValue<T>(args[2])};
  std::optional<std::vector<ConstantSubscript>> shape{
      GetIntegerVector<ConstantSubscript>(args[1])};
  std::optional<std::vector<int>> order{GetIntegerVector<int>(args[3])};

  if (!source || !shape || (args[2].has_value() && !pad) ||
      (args[3].has_value() && !order)) {
    return Expr<T>{std::move(funcRef)};  // Non-constant arguments
  } else if (!IsValidShape(shape.value())) {
    context.messages().Say("Invalid SHAPE in RESHAPE"_en_US);
  } else {
    int rank{GetRank(shape.value())};
    std::size_t resultElements{TotalElementCount(shape.value())};
    std::optional<std::vector<int>> dimOrder;
    if (order.has_value()) {
      dimOrder = ValidateDimensionOrder(rank, *order);
    }
    std::vector<int> *dimOrderPtr{
        dimOrder.has_value() ? &dimOrder.value() : nullptr};
    if (order.has_value() && !dimOrder.has_value()) {
      context.messages().Say("Invalid ORDER in RESHAPE"_en_US);
    } else if (resultElements > source->size() && (!pad || pad->empty())) {
      context.messages().Say("Too few SOURCE elements in RESHAPE and PAD"
                             "is not present or has null size"_en_US);
    } else {
      Constant<T> result{!source->empty()
              ? source->Reshape(std::move(shape.value()))
              : pad->Reshape(std::move(shape.value()))};
      ConstantSubscripts subscripts{result.lbounds()};
      auto copied{
          result.CopyFrom(*source, source->size(), subscripts, dimOrderPtr)};
      if (copied < resultElements) {
        CHECK(pad);
        copied += result.CopyFrom(
            *pad, resultElements - copied, subscripts, dimOrderPtr);
      }
      CHECK(copied == resultElements);
      return Expr<T>{std::move(result)};
    }
  }
  // Invalid, prevent re-folding
  return MakeInvalidIntrinsic(std::move(funcRef));
}

template<typename T>
Expr<T> FoldOperation(FoldingContext &context, FunctionRef<T> &&funcRef) {
  ActualArguments &args{funcRef.arguments()};
  for (std::optional<ActualArgument> &arg : args) {
    if (auto *expr{UnwrapExpr<Expr<SomeType>>(arg)}) {
      *expr = Fold(context, std::move(*expr));
    }
  }
  if (auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)}) {
    const std::string name{intrinsic->name};
    if (name == "reshape") {
      return Reshape(context, std::move(funcRef));
    }
    // TODO: other type independent transformational
    if constexpr (!std::is_same_v<T, SomeDerived>) {
      return FoldIntrinsicFunction(context, std::move(funcRef));
    }
  }
  return Expr<T>{std::move(funcRef)};
}

template<int KIND>
Expr<Type<TypeCategory::Integer, KIND>> FoldIntrinsicFunction(
    FoldingContext &context,
    FunctionRef<Type<TypeCategory::Integer, KIND>> &&funcRef) {
  using T = Type<TypeCategory::Integer, KIND>;
  using Int4 = Type<TypeCategory::Integer, 4>;
  ActualArguments &args{funcRef.arguments()};
  auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)};
  CHECK(intrinsic);
  std::string name{intrinsic->name};
  if (name == "abs") {
    return FoldElementalIntrinsic<T, T>(context, std::move(funcRef),
        ScalarFunc<T, T>([&context](const Scalar<T> &i) -> Scalar<T> {
          typename Scalar<T>::ValueWithOverflow j{i.ABS()};
          if (j.overflow) {
            context.messages().Say(
                "abs(integer(kind=%d)) folding overflowed"_en_US, KIND);
          }
          return j.value;
        }));
  } else if (name == "bit_size") {
    return Expr<T>{Scalar<T>::bits};
  } else if (name == "dim") {
    return FoldElementalIntrinsic<T, T, T>(
        context, std::move(funcRef), &Scalar<T>::DIM);
  } else if (name == "dshiftl" || name == "dshiftr") {
    const auto fptr{
        name == "dshiftl" ? &Scalar<T>::DSHIFTL : &Scalar<T>::DSHIFTR};
    // Third argument can be of any kind. However, it must be smaller or equal
    // than BIT_SIZE. It can be converted to Int4 to simplify.
    return FoldElementalIntrinsic<T, T, T, Int4>(context, std::move(funcRef),
        ScalarFunc<T, T, T, Int4>(
            [&fptr](const Scalar<T> &i, const Scalar<T> &j,
                const Scalar<Int4> &shift) -> Scalar<T> {
              return std::invoke(fptr, i, j, static_cast<int>(shift.ToInt64()));
            }));
  } else if (name == "exponent") {
    if (auto *sx{UnwrapExpr<Expr<SomeReal>>(args[0])}) {
      return std::visit(
          [&funcRef, &context](const auto &x) -> Expr<T> {
            using TR = typename std::decay_t<decltype(x)>::Result;
            return FoldElementalIntrinsic<T, TR>(context, std::move(funcRef),
                &Scalar<TR>::template EXPONENT<Scalar<T>>);
          },
          sx->u);
    } else {
      common::die("exponent argument must be real");
    }
  } else if (name == "huge") {
    return Expr<T>{Scalar<T>::HUGE()};
  } else if (name == "iachar" || name == "ichar") {
    auto *someChar{UnwrapExpr<Expr<SomeCharacter>>(args[0])};
    CHECK(someChar != nullptr);
    if (auto len{ToInt64(someChar->LEN())}) {
      if (len.value() != 1) {
        // Do not die, this was not checked before
        context.messages().Say(
            "Character in intrinsic function %s must have length one"_en_US,
            name);
      } else {
        return std::visit(
            [&funcRef, &context](const auto &str) -> Expr<T> {
              using Char = typename std::decay_t<decltype(str)>::Result;
              return FoldElementalIntrinsic<T, Char>(context,
                  std::move(funcRef),
                  ScalarFunc<T, Char>([](const Scalar<Char> &c) {
                    return Scalar<T>{CharacterUtils<Char::kind>::ICHAR(c)};
                  }));
            },
            someChar->u);
      }
    }
  } else if (name == "iand" || name == "ior" || name == "ieor") {
    auto fptr{&Scalar<T>::IAND};
    if (name == "iand") {  // done in fptr declaration
    } else if (name == "ior") {
      fptr = &Scalar<T>::IOR;
    } else if (name == "ieor") {
      fptr = &Scalar<T>::IEOR;
    } else {
      common::die("missing case to fold intrinsic function %s", name.c_str());
    }
    return FoldElementalIntrinsic<T, T, T>(
        context, std::move(funcRef), ScalarFunc<T, T, T>(fptr));
  } else if (name == "ibclr" || name == "ibset" || name == "ishft" ||
      name == "shifta" || name == "shiftr" || name == "shiftl") {
    // Second argument can be of any kind. However, it must be smaller or
    // equal than BIT_SIZE. It can be converted to Int4 to simplify.
    auto fptr{&Scalar<T>::IBCLR};
    if (name == "ibclr") {  // done in fprt definition
    } else if (name == "ibset") {
      fptr = &Scalar<T>::IBSET;
    } else if (name == "ishft") {
      fptr = &Scalar<T>::ISHFT;
    } else if (name == "shifta") {
      fptr = &Scalar<T>::SHIFTA;
    } else if (name == "shiftr") {
      fptr = &Scalar<T>::SHIFTR;
    } else if (name == "shiftl") {
      fptr = &Scalar<T>::SHIFTL;
    } else {
      common::die("missing case to fold intrinsic function %s", name.c_str());
    }
    return FoldElementalIntrinsic<T, T, Int4>(context, std::move(funcRef),
        ScalarFunc<T, T, Int4>(
            [&fptr](const Scalar<T> &i, const Scalar<Int4> &pos) -> Scalar<T> {
              return std::invoke(fptr, i, static_cast<int>(pos.ToInt64()));
            }));
  } else if (name == "int") {
    if (auto *expr{args[0].value().UnwrapExpr()}) {
      return std::visit(
          [&](auto &&x) -> Expr<T> {
            using From = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<From, BOZLiteralConstant> ||
                IsNumericCategoryExpr<From>()) {
              return Fold(context, ConvertToType<T>(std::move(x)));
            }
            common::die("int() argument type not valid");
          },
          std::move(expr->u));
    }
  } else if (name == "kind") {
    if constexpr (common::HasMember<T, IntegerTypes>) {
      return Expr<T>{args[0].value().GetType()->kind()};
    } else {
      common::die("kind() result not integral");
    }
  } else if (name == "leadz" || name == "trailz" || name == "poppar" ||
      name == "popcnt") {
    if (auto *sn{UnwrapExpr<Expr<SomeInteger>>(args[0])}) {
      return std::visit(
          [&funcRef, &context, &name](const auto &n) -> Expr<T> {
            using TI = typename std::decay_t<decltype(n)>::Result;
            if (name == "poppar") {
              return FoldElementalIntrinsic<T, TI>(context, std::move(funcRef),
                  ScalarFunc<T, TI>([](const Scalar<TI> &i) -> Scalar<T> {
                    return Scalar<T>{i.POPPAR() ? 1 : 0};
                  }));
            }
            auto fptr{&Scalar<TI>::LEADZ};
            if (name == "leadz") {  // done in fprt definition
            } else if (name == "trailz") {
              fptr = &Scalar<TI>::TRAILZ;
            } else if (name == "popcnt") {
              fptr = &Scalar<TI>::POPCNT;
            } else {
              common::die(
                  "missing case to fold intrinsic function %s", name.c_str());
            }
            return FoldElementalIntrinsic<T, TI>(context, std::move(funcRef),
                ScalarFunc<T, TI>([&fptr](const Scalar<TI> &i) -> Scalar<T> {
                  return Scalar<T>{std::invoke(fptr, i)};
                }));
          },
          sn->u);
    } else {
      common::die("leadz argument must be integer");
    }
  } else if (name == "len") {
    if (auto *charExpr{UnwrapExpr<Expr<SomeCharacter>>(args[0])}) {
      return std::visit(
          [&](auto &kx) {
            if (auto len{kx.LEN()}) {
              return Fold(context, ConvertToType<T>(*std::move(len)));
            } else {
              return Expr<T>{std::move(funcRef)};
            }
          },
          charExpr->u);
    } else {
      common::die("len() argument must be of character type");
    }
  } else if (name == "maskl" || name == "maskr") {
    // Argument can be of any kind but value has to be smaller than BIT_SIZE.
    // It can be safely converted to Int4 to simplify.
    const auto fptr{name == "maskl" ? &Scalar<T>::MASKL : &Scalar<T>::MASKR};
    return FoldElementalIntrinsic<T, Int4>(context, std::move(funcRef),
        ScalarFunc<T, Int4>([&fptr](const Scalar<Int4> &places) -> Scalar<T> {
          return fptr(static_cast<int>(places.ToInt64()));
        }));
  } else if (name == "maxexponent") {
    if (auto *sx{UnwrapExpr<Expr<SomeReal>>(args[0])}) {
      return std::visit(
          [](const auto &x) {
            using TR = typename std::decay_t<decltype(x)>::Result;
            return Expr<T>{Scalar<TR>::MAXEXPONENT};
          },
          sx->u);
    }
  } else if (name == "merge_bits") {
    return FoldElementalIntrinsic<T, T, T, T>(
        context, std::move(funcRef), &Scalar<T>::MERGE_BITS);
  } else if (name == "minexponent") {
    if (auto *sx{UnwrapExpr<Expr<SomeReal>>(args[0])}) {
      return std::visit(
          [](const auto &x) {
            using TR = typename std::decay_t<decltype(x)>::Result;
            return Expr<T>{Scalar<TR>::MINEXPONENT};
          },
          sx->u);
    }
  } else if (name == "precision") {
    if (const auto *cx{UnwrapExpr<Expr<SomeReal>>(args[0])}) {
      return Expr<T>{std::visit(
          [](const auto &kx) {
            return Scalar<ResultType<decltype(kx)>>::PRECISION;
          },
          cx->u)};
    } else if (const auto *cx{UnwrapExpr<Expr<SomeComplex>>(args[0])}) {
      return Expr<T>{std::visit(
          [](const auto &kx) {
            return Scalar<typename ResultType<decltype(kx)>::Part>::PRECISION;
          },
          cx->u)};
    }
  } else if (name == "radix") {
    return Expr<T>{2};
  } else if (name == "range") {
    if (const auto *cx{UnwrapExpr<Expr<SomeInteger>>(args[0])}) {
      return Expr<T>{std::visit(
          [](const auto &kx) {
            return Scalar<ResultType<decltype(kx)>>::RANGE;
          },
          cx->u)};
    } else if (const auto *cx{UnwrapExpr<Expr<SomeReal>>(args[0])}) {
      return Expr<T>{std::visit(
          [](const auto &kx) {
            return Scalar<ResultType<decltype(kx)>>::RANGE;
          },
          cx->u)};
    } else if (const auto *cx{UnwrapExpr<Expr<SomeComplex>>(args[0])}) {
      return Expr<T>{std::visit(
          [](const auto &kx) {
            return Scalar<typename ResultType<decltype(kx)>::Part>::RANGE;
          },
          cx->u)};
    }
  } else if (name == "rank") {
    // TODO assumed-rank dummy argument
    return Expr<T>{args[0].value().Rank()};
  } else if (name == "selected_char_kind") {
    if (const auto *chCon{UnwrapExpr<Constant<TypeOf<std::string>>>(args[0])}) {
      if (std::optional<std::string> value{chCon->GetScalarValue()}) {
        if (*value == "default") {
          return Expr<T>{
              context.defaults().GetDefaultKind(TypeCategory::Character)};
        } else {
          return Expr<T>{SelectedCharKind(*value)};
        }
      }
    }
  } else if (name == "selected_int_kind") {
    if (auto p{GetInt64Arg(args[0])}) {
      return Expr<T>{SelectedIntKind(*p)};
    }
  } else if (name == "selected_real_kind") {
    if (auto p{GetInt64ArgOr(args[0], 0)}) {
      if (auto r{GetInt64ArgOr(args[1], 0)}) {
        if (auto radix{GetInt64ArgOr(args[2], 2)}) {
          return Expr<T>{SelectedRealKind(*p, *r, *radix)};
        }
      }
    }
  } else if (name == "shape") {
    if (auto shape{GetShape(context, args[0].value())}) {
      if (auto shapeExpr{AsExtentArrayExpr(*shape)}) {
        return Fold(context, ConvertToType<T>(std::move(*shapeExpr)));
      }
    }
  } else if (name == "size") {
    if (auto shape{GetShape(context, args[0].value())}) {
      if (auto &dimArg{args[1]}) {  // DIM= is present, get one extent
        if (auto dim{GetInt64Arg(args[1])}) {
          int rank = GetRank(*shape);
          if (*dim >= 1 && *dim <= rank) {
            if (auto &extent{shape->at(*dim - 1)}) {
              return Fold(context, ConvertToType<T>(std::move(*extent)));
            }
          } else {
            context.messages().Say(
                "size(array,dim=%jd) dimension is out of range for rank-%d array"_en_US,
                static_cast<std::intmax_t>(*dim), static_cast<int>(rank));
          }
        }
      } else if (auto extents{common::AllElementsPresent(std::move(*shape))}) {
        // DIM= is absent; compute PRODUCT(SHAPE())
        ExtentExpr product{1};
        for (auto &&extent : std::move(*extents)) {
          product = std::move(product) * std::move(extent);
        }
        return Expr<T>{ConvertToType<T>(Fold(context, std::move(product)))};
      }
    }
  }
  // TODO:
  // ceiling, count, cshift, dot_product, eoshift,
  // findloc, floor, iall, iany, iparity, ibits, image_status, index, ishftc,
  // lbound, len_trim, matmul, max, maxloc, maxval, merge, min,
  // minloc, minval, mod, modulo, nint, not, pack, product, reduce,
  // scan, sign, spread, sum, transfer, transpose, ubound, unpack, verify
  return Expr<T>{std::move(funcRef)};
}

template<int KIND>
Expr<Type<TypeCategory::Real, KIND>> ToReal(
    FoldingContext &context, Expr<SomeType> &&expr) {
  using Result = Type<TypeCategory::Real, KIND>;
  std::optional<Expr<Result>> result;
  std::visit(
      [&](auto &&x) {
        using From = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<From, BOZLiteralConstant>) {
          // Move the bits without any integer->real conversion
          From original{x};
          result = ConvertToType<Result>(std::move(x));
          const auto *constant{UnwrapExpr<Constant<Result>>(*result)};
          CHECK(constant != nullptr);
          Scalar<Result> real{constant->GetScalarValue().value()};
          From converted{From::ConvertUnsigned(real.RawBits()).value};
          if (!(original == converted)) {  // C1601
            context.messages().Say(
                "Nonzero bits truncated from BOZ literal constant in REAL intrinsic"_en_US);
          }
        } else if constexpr (IsNumericCategoryExpr<From>()) {
          result = Fold(context, ConvertToType<Result>(std::move(x)));
        } else {
          common::die("ToReal: bad argument expression");
        }
      },
      std::move(expr.u));
  return result.value();
}

template<int KIND>
Expr<Type<TypeCategory::Real, KIND>> FoldIntrinsicFunction(
    FoldingContext &context,
    FunctionRef<Type<TypeCategory::Real, KIND>> &&funcRef) {
  using T = Type<TypeCategory::Real, KIND>;
  using ComplexT = Type<TypeCategory::Complex, KIND>;
  ActualArguments &args{funcRef.arguments()};
  auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)};
  CHECK(intrinsic);
  std::string name{intrinsic->name};
  if (name == "acos" || name == "acosh" || name == "asin" || name == "asinh" ||
      (name == "atan" && args.size() == 1) || name == "atanh" ||
      name == "bessel_j0" || name == "bessel_j1" || name == "bessel_y0" ||
      name == "bessel_y1" || name == "cos" || name == "cosh" || name == "erf" ||
      name == "erfc" || name == "erfc_scaled" || name == "exp" ||
      name == "gamma" || name == "log" || name == "log10" ||
      name == "log_gamma" || name == "sin" || name == "sinh" ||
      name == "sqrt" || name == "tan" || name == "tanh") {
    CHECK(args.size() == 1);
    if (auto callable{context.hostIntrinsicsLibrary()
                          .GetHostProcedureWrapper<Scalar, T, T>(name)}) {
      return FoldElementalIntrinsic<T, T>(
          context, std::move(funcRef), *callable);
    } else {
      context.messages().Say(
          "%s(real(kind=%d)) cannot be folded on host"_en_US, name, KIND);
    }
  }
  if (name == "atan" || name == "atan2" || name == "hypot" || name == "mod") {
    std::string localName{name == "atan2" ? "atan" : name};
    CHECK(args.size() == 2);
    if (auto callable{
            context.hostIntrinsicsLibrary()
                .GetHostProcedureWrapper<Scalar, T, T, T>(localName)}) {
      return FoldElementalIntrinsic<T, T, T>(
          context, std::move(funcRef), *callable);
    } else {
      context.messages().Say(
          "%s(real(kind=%d), real(kind%d)) cannot be folded on host"_en_US,
          name, KIND, KIND);
    }
  } else if (name == "bessel_jn" || name == "bessel_yn") {
    if (args.size() == 2) {  // elemental
      // runtime functions use int arg
      using Int4 = Type<TypeCategory::Integer, 4>;
      if (auto callable{
              context.hostIntrinsicsLibrary()
                  .GetHostProcedureWrapper<Scalar, T, Int4, T>(name)}) {
        return FoldElementalIntrinsic<T, Int4, T>(
            context, std::move(funcRef), *callable);
      } else {
        context.messages().Say(
            "%s(integer(kind=4), real(kind=%d)) cannot be folded on host"_en_US,
            name, KIND);
      }
    }
  } else if (name == "abs") {
    // Argument can be complex or real
    if (auto *x{UnwrapExpr<Expr<SomeReal>>(args[0])}) {
      return FoldElementalIntrinsic<T, T>(
          context, std::move(funcRef), &Scalar<T>::ABS);
    } else if (auto *z{UnwrapExpr<Expr<SomeComplex>>(args[0])}) {
      if (auto callable{
              context.hostIntrinsicsLibrary()
                  .GetHostProcedureWrapper<Scalar, T, ComplexT>("abs")}) {
        return FoldElementalIntrinsic<T, ComplexT>(
            context, std::move(funcRef), *callable);
      } else {
        context.messages().Say(
            "abs(complex(kind=%d)) cannot be folded on host"_en_US, KIND);
      }
    } else {
      common::die(" unexpected argument type inside abs");
    }
  } else if (name == "aimag") {
    return FoldElementalIntrinsic<T, ComplexT>(
        context, std::move(funcRef), &Scalar<ComplexT>::AIMAG);
  } else if (name == "aint") {
    return FoldElementalIntrinsic<T, T>(context, std::move(funcRef),
        ScalarFunc<T, T>([&name, &context](const Scalar<T> &x) -> Scalar<T> {
          ValueWithRealFlags<Scalar<T>> y{x.AINT()};
          if (y.flags.test(RealFlag::Overflow)) {
            context.messages().Say("%s intrinsic folding overflow"_en_US, name);
          }
          return y.value;
        }));
  } else if (name == "dprod") {
    if (auto *x{UnwrapExpr<Expr<SomeReal>>(args[0])}) {
      if (auto *y{UnwrapExpr<Expr<SomeReal>>(args[1])}) {
        return Fold(context,
            Expr<T>{Multiply<T>{ConvertToType<T>(std::move(*x)),
                ConvertToType<T>(std::move(*y))}});
      }
    }
    common::die("Wrong argument type in dprod()");
  } else if (name == "epsilon") {
    return Expr<T>{Scalar<T>::EPSILON()};
  } else if (name == "huge") {
    return Expr<T>{Scalar<T>::HUGE()};
  } else if (name == "real") {
    if (auto *expr{args[0].value().UnwrapExpr()}) {
      return ToReal<KIND>(context, std::move(*expr));
    }
  } else if (name == "tiny") {
    return Expr<T>{Scalar<T>::TINY()};
  }
  // TODO: anint, cshift, dim, dot_product, eoshift, fraction, matmul,
  // max, maxval, merge, min, minval, modulo, nearest, norm2, pack, product,
  // reduce, rrspacing, scale, set_exponent, sign, spacing, spread,
  // sum, transfer, transpose, unpack, bessel_jn (transformational) and
  // bessel_yn (transformational)
  return Expr<T>{std::move(funcRef)};
}

template<int KIND>
Expr<Type<TypeCategory::Complex, KIND>> FoldIntrinsicFunction(
    FoldingContext &context,
    FunctionRef<Type<TypeCategory::Complex, KIND>> &&funcRef) {
  using T = Type<TypeCategory::Complex, KIND>;
  ActualArguments &args{funcRef.arguments()};
  auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)};
  CHECK(intrinsic);
  std::string name{intrinsic->name};
  if (name == "acos" || name == "acosh" || name == "asin" || name == "asinh" ||
      name == "atan" || name == "atanh" || name == "cos" || name == "cosh" ||
      name == "exp" || name == "log" || name == "sin" || name == "sinh" ||
      name == "sqrt" || name == "tan" || name == "tanh") {
    if (auto callable{context.hostIntrinsicsLibrary()
                          .GetHostProcedureWrapper<Scalar, T, T>(name)}) {
      return FoldElementalIntrinsic<T, T>(
          context, std::move(funcRef), *callable);
    } else {
      context.messages().Say(
          "%s(complex(kind=%d)) cannot be folded on host"_en_US, name, KIND);
    }
  } else if (name == "conjg") {
    return FoldElementalIntrinsic<T, T>(
        context, std::move(funcRef), &Scalar<T>::CONJG);
  } else if (name == "cmplx") {
    if (args.size() == 2) {
      if (auto *x{UnwrapExpr<Expr<SomeComplex>>(args[0])}) {
        return Fold(context, ConvertToType<T>(std::move(*x)));
      } else {
        common::die("x must be complex in cmplx(x[, kind])");
      }
    } else {
      CHECK(args.size() == 3);
      using Part = typename T::Part;
      Expr<SomeType> re{std::move(*args[0].value().UnwrapExpr())};
      Expr<SomeType> im{args[1].has_value()
              ? std::move(*args[1].value().UnwrapExpr())
              : AsGenericExpr(Constant<Part>{Scalar<Part>{}})};
      return Fold(context,
          Expr<T>{ComplexConstructor<KIND>{ToReal<KIND>(context, std::move(re)),
              ToReal<KIND>(context, std::move(im))}});
    }
  }
  // TODO: cshift, dot_product, eoshift, matmul, merge, pack, product,
  // reduce, spread, sum, transfer, transpose, unpack
  return Expr<T>{std::move(funcRef)};
}

template<int KIND>
Expr<Type<TypeCategory::Logical, KIND>> FoldIntrinsicFunction(
    FoldingContext &context,
    FunctionRef<Type<TypeCategory::Logical, KIND>> &&funcRef) {
  using T = Type<TypeCategory::Logical, KIND>;
  ActualArguments &args{funcRef.arguments()};
  auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)};
  CHECK(intrinsic);
  std::string name{intrinsic->name};
  if (name == "all") {
    if (!args[1].has_value()) {  // TODO: ALL(x,DIM=d)
      if (const auto *constant{UnwrapConstantValue<T>(args[0])}) {
        bool result{true};
        for (const auto &element : constant->values()) {
          if (!element.IsTrue()) {
            result = false;
            break;
          }
        }
        return Expr<T>{result};
      }
    }
  } else if (name == "any") {
    if (!args[1].has_value()) {  // TODO: ANY(x,DIM=d)
      if (const auto *constant{UnwrapConstantValue<T>(args[0])}) {
        bool result{false};
        for (const auto &element : constant->values()) {
          if (element.IsTrue()) {
            result = true;
            break;
          }
        }
        return Expr<T>{result};
      }
    }
  } else if (name == "bge" || name == "bgt" || name == "ble" || name == "blt") {
    using LargestInt = Type<TypeCategory::Integer, 16>;
    static_assert(std::is_same_v<Scalar<LargestInt>, BOZLiteralConstant>);
    // Arguments do not have to be of the same integer type. Convert all
    // arguments to the biggest integer type before comparing them to
    // simplify.
    for (int i{0}; i <= 1; ++i) {
      if (auto *x{UnwrapExpr<Expr<SomeInteger>>(args[i])}) {
        *args[i] = AsGenericExpr(
            Fold(context, ConvertToType<LargestInt>(std::move(*x))));
      } else if (auto *x{UnwrapExpr<BOZLiteralConstant>(args[i])}) {
        *args[i] = AsGenericExpr(Constant<LargestInt>{std::move(*x)});
      }
    }
    auto fptr{&Scalar<LargestInt>::BGE};
    if (name == "bge") {  // done in fptr declaration
    } else if (name == "bgt") {
      fptr = &Scalar<LargestInt>::BGT;
    } else if (name == "ble") {
      fptr = &Scalar<LargestInt>::BLE;
    } else if (name == "blt") {
      fptr = &Scalar<LargestInt>::BLT;
    } else {
      common::die("missing case to fold intrinsic function %s", name.c_str());
    }
    return FoldElementalIntrinsic<T, LargestInt, LargestInt>(context,
        std::move(funcRef),
        ScalarFunc<T, LargestInt, LargestInt>(
            [&fptr](const Scalar<LargestInt> &i, const Scalar<LargestInt> &j) {
              return Scalar<T>{std::invoke(fptr, i, j)};
            }));
  }
  // TODO: btest, cshift, dot_product, eoshift, is_iostat_end,
  // is_iostat_eor, lge, lgt, lle, llt, logical, matmul, merge, out_of_range,
  // pack, parity, reduce, spread, transfer, transpose, unpack
  return Expr<T>{std::move(funcRef)};
}

template<int KIND>
Expr<Type<TypeCategory::Character, KIND>> FoldIntrinsicFunction(
    FoldingContext &context,
    FunctionRef<Type<TypeCategory::Character, KIND>> &&funcRef) {
  using T = Type<TypeCategory::Character, KIND>;
  auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)};
  CHECK(intrinsic);
  std::string name{intrinsic->name};
  if (name == "achar" || name == "char") {
    using IntT = SubscriptInteger;
    return FoldElementalIntrinsic<T, IntT>(context, std::move(funcRef),
        ScalarFunc<T, IntT>([](const Scalar<IntT> &i) {
          return CharacterUtils<KIND>::CHAR(i.ToUInt64());
        }));
  } else if (name == "adjustl") {
    return FoldElementalIntrinsic<T, T>(
        context, std::move(funcRef), CharacterUtils<KIND>::ADJUSTL);
  } else if (name == "adjustr") {
    return FoldElementalIntrinsic<T, T>(
        context, std::move(funcRef), CharacterUtils<KIND>::ADJUSTR);
  } else if (name == "new_line") {
    return Expr<T>{Constant<T>{CharacterUtils<KIND>::NEW_LINE()}};
  }
  // TODO: cshift, eoshift, max, maxval, merge, min, minval, pack, reduce,
  // repeat, spread, transfer, transpose, trim, unpack
  return Expr<T>{std::move(funcRef)};
}

// Get the value of a PARAMETER
template<typename T>
std::optional<Expr<T>> GetParameterValue(
    FoldingContext &context, const Symbol &symbol) {
  if (symbol.attrs().test(semantics::Attr::PARAMETER)) {
    if (const auto *object{
            symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
      if (object->initWasValidated()) {
        const auto *constant{UnwrapConstantValue<T>(object->init())};
        CHECK(constant != nullptr);
        return Expr<T>{*constant};
      }
      if (const auto &init{object->init()}) {
        if (auto dyType{DynamicType::From(symbol)}) {
          semantics::ObjectEntityDetails *mutableObject{
              const_cast<semantics::ObjectEntityDetails *>(object)};
          auto converted{
              ConvertToType(*dyType, std::move(mutableObject->init().value()))};
          // Reset expression now to prevent infinite loops if the init
          // expression depends on symbol itself.
          mutableObject->set_init(std::nullopt);
          if (converted.has_value()) {
            *converted = Fold(context, std::move(*converted));
            auto *unwrapped{UnwrapExpr<Expr<T>>(*converted)};
            CHECK(unwrapped != nullptr);
            if (auto *constant{UnwrapConstantValue<T>(*unwrapped)}) {
              if (symbol.Rank() > 0) {
                if (constant->Rank() == 0) {
                  // scalar expansion
                  if (auto symShape{GetShape(context, symbol)}) {
                    if (auto extents{AsConstantExtents(*symShape)}) {
                      *constant = constant->Reshape(std::move(*extents));
                      CHECK(constant->Rank() == symbol.Rank());
                    }
                  }
                }
                if (constant->Rank() == symbol.Rank()) {
                  NamedEntity base{symbol};
                  if (auto lbounds{
                          AsConstantExtents(GetLowerBounds(context, base))}) {
                    constant->set_lbounds(*std::move(lbounds));
                  }
                }
              }
              mutableObject->set_init(AsGenericExpr(Expr<T>{*constant}));
              if (auto constShape{GetShape(context, *constant)}) {
                if (auto symShape{GetShape(context, symbol)}) {
                  if (CheckConformance(context.messages(), *constShape,
                          *symShape, "initialization expression",
                          "PARAMETER")) {
                    mutableObject->set_initWasValidated();
                    return std::move(*unwrapped);
                  }
                } else {
                  context.messages().Say(symbol.name(),
                      "Could not determine the shape of the PARAMETER"_err_en_US);
                }
              } else {
                context.messages().Say(symbol.name(),
                    "Could not determine the shape of the initialization expression"_err_en_US);
              }
              mutableObject->set_init(std::nullopt);
            } else {
              std::stringstream ss;
              unwrapped->AsFortran(ss);
              context.messages().Say(symbol.name(),
                  "Initialization expression for PARAMETER '%s' (%s) cannot be computed as a constant value"_err_en_US,
                  symbol.name(), ss.str());
            }
          } else {
            std::stringstream ss;
            init->AsFortran(ss);
            context.messages().Say(symbol.name(),
                "Initialization expression for PARAMETER '%s' (%s) cannot be converted to its type (%s)"_err_en_US,
                symbol.name(), ss.str(), dyType->AsFortran());
          }
        }
      }
    }
  }
  return std::nullopt;
}

template<typename T>
static std::optional<Constant<T>> GetFoldedParameterValue(
    FoldingContext &context, const Symbol &symbol) {
  if (auto value{GetParameterValue<T>(context, symbol)}) {
    Expr<T> folded{Fold(context, std::move(*value))};
    if (const Constant<T> *value{UnwrapConstantValue<T>(folded)}) {
      return *value;
    }
  }
  return std::nullopt;
}

// Apply subscripts to a constant array

static std::optional<Constant<SubscriptInteger>> GetConstantSubscript(
    FoldingContext &context, Subscript &ss, const NamedEntity &base, int dim) {
  ss = FoldOperation(context, std::move(ss));
  return std::visit(
      common::visitors{
          [](IndirectSubscriptIntegerExpr &expr)
              -> std::optional<Constant<SubscriptInteger>> {
            if (auto constant{
                    GetScalarConstantValue<SubscriptInteger>(expr.value())}) {
              return Constant<SubscriptInteger>{*constant};
            } else {
              return std::nullopt;
            }
          },
          [&](Triplet &triplet) -> std::optional<Constant<SubscriptInteger>> {
            auto lower{triplet.lower()}, upper{triplet.upper()};
            std::optional<ConstantSubscript> stride{ToInt64(triplet.stride())};
            if (!lower.has_value()) {
              lower = GetLowerBound(context, base, dim);
            }
            if (!upper.has_value()) {
              upper = GetUpperBound(context, GetLowerBound(context, base, dim),
                  GetExtent(context, base, dim));
            }
            auto lbi{ToInt64(lower)}, ubi{ToInt64(upper)};
            if (lbi.has_value() && ubi.has_value() && stride.has_value() &&
                *stride != 0) {
              std::vector<SubscriptInteger::Scalar> values;
              while ((*stride > 0 && *lbi <= *ubi) ||
                  (*stride < 0 && *lbi >= *ubi)) {
                values.emplace_back(*lbi);
                *lbi += *stride;
              }
              return Constant<SubscriptInteger>{std::move(values),
                  ConstantSubscripts{
                      static_cast<ConstantSubscript>(values.size())}};
            } else {
              return std::nullopt;
            }
          },
      },
      ss.u);
}

template<typename T>
std::optional<Constant<T>> ApplySubscripts(parser::ContextualMessages &messages,
    const Constant<T> &array,
    const std::vector<Constant<SubscriptInteger>> &subscripts) {
  const auto &shape{array.shape()};
  const auto &lbounds{array.lbounds()};
  int rank{GetRank(shape)};
  CHECK(rank == static_cast<int>(subscripts.size()));
  std::size_t elements{1};
  ConstantSubscripts resultShape;
  ConstantSubscripts ssLB;
  for (const auto &ss : subscripts) {
    CHECK(ss.Rank() <= 1);
    if (ss.Rank() == 1) {
      resultShape.push_back(static_cast<ConstantSubscript>(ss.size()));
      elements *= ss.size();
      ssLB.push_back(ss.lbounds().front());
    }
  }
  ConstantSubscripts ssAt(rank, 0), at(rank, 0), tmp(1, 0);
  std::vector<Scalar<T>> values;
  while (elements-- > 0) {
    bool increment{true};
    int k{0};
    for (int j{0}; j < rank; ++j) {
      if (subscripts[j].Rank() == 0) {
        at[j] = subscripts[j].GetScalarValue().value().ToInt64();
      } else {
        CHECK(k < GetRank(resultShape));
        tmp[0] = ssLB[j] + ssAt[j];
        at[j] = subscripts[j].At(tmp).ToInt64();
        if (increment) {
          if (++ssAt[j] == resultShape[k]) {
            ssAt[j] = 0;
          } else {
            increment = false;
          }
        }
        ++k;
      }
      if (at[j] < lbounds[j] || at[j] >= lbounds[j] + shape[j]) {
        messages.Say("Subscript value (%jd) is out of range on dimension %d "
                     "in reference to a constant array value"_err_en_US,
            static_cast<std::intmax_t>(at[j]), j + 1);
        return std::nullopt;
      }
    }
    values.emplace_back(array.At(at));
    CHECK(!increment || elements == 0);
    CHECK(k == GetRank(resultShape));
  }
  if constexpr (T::category == TypeCategory::Character) {
    return Constant<T>{array.LEN(), std::move(values), std::move(resultShape)};
  } else if constexpr (std::is_same_v<T, SomeDerived>) {
    return Constant<T>{array.result().derivedTypeSpec(), std::move(values),
        std::move(resultShape)};
  } else {
    return Constant<T>{std::move(values), std::move(resultShape)};
  }
}

// GetConstantComponent() is mutually recursive with FoldArrayRef().
template<typename T>
std::optional<Constant<T>> GetConstantComponent(FoldingContext &, Component &,
    const std::vector<Constant<SubscriptInteger>> * = nullptr);

template<typename T>
std::optional<Constant<T>> ApplyComponent(FoldingContext &context,
    Constant<SomeDerived> &&structures, const Symbol &component,
    const std::vector<Constant<SubscriptInteger>> *subscripts = nullptr) {
  if (auto scalar{structures.GetScalarValue()}) {
    if (auto *expr{scalar->Find(&component)}) {
      if (const Constant<T> *value{UnwrapConstantValue<T>(*expr)}) {
        if (subscripts == nullptr) {
          return std::move(*value);
        } else {
          return ApplySubscripts<T>(context.messages(), *value, *subscripts);
        }
      }
    }
  } else {
    // A(:)%scalar_component & A(:)%array_component(subscripts)
    std::unique_ptr<ArrayConstructor<T>> array;
    if (structures.empty()) {
      return std::nullopt;
    }
    ConstantSubscripts at{structures.lbounds()};
    do {
      StructureConstructor scalar{structures.At(at)};
      if (auto *expr{scalar.Find(&component)}) {
        if (const Constant<T> *value{UnwrapConstantValue<T>(*expr)}) {
          if (array.get() == nullptr) {
            // This technique ensures that character length or derived type
            // information is propagated to the array constructor.
            auto *typedExpr{UnwrapExpr<Expr<T>>(*expr)};
            CHECK(typedExpr != nullptr);
            array = std::make_unique<ArrayConstructor<T>>(*typedExpr);
          }
          if (subscripts != nullptr) {
            if (auto element{ApplySubscripts<T>(
                    context.messages(), *value, *subscripts)}) {
              CHECK(element->Rank() == 0);
              array->Push(Expr<T>{std::move(*element)});
            } else {
              return std::nullopt;
            }
          } else {
            CHECK(value->Rank() == 0);
            array->Push(Expr<T>{*value});
          }
        } else {
          return std::nullopt;
        }
      }
    } while (structures.IncrementSubscripts(at));
    // Fold the ArrayConstructor<> into a Constant<>.
    CHECK(array);
    Expr<T> result{Fold(context, Expr<T>{std::move(*array)})};
    if (auto *constant{UnwrapConstantValue<T>(result)}) {
      return constant->Reshape(common::Clone(structures.shape()));
    }
  }
  return std::nullopt;
}

template<typename T>
std::optional<Constant<T>> FoldArrayRef(
    FoldingContext &context, ArrayRef &aRef) {
  std::vector<Constant<SubscriptInteger>> subscripts;
  int dim{0};
  for (Subscript &ss : aRef.subscript()) {
    if (auto constant{GetConstantSubscript(context, ss, aRef.base(), dim++)}) {
      subscripts.emplace_back(std::move(*constant));
    } else {
      return std::nullopt;
    }
  }
  if (Component * component{aRef.base().UnwrapComponent()}) {
    return GetConstantComponent<T>(context, *component, &subscripts);
  } else if (std::optional<Constant<T>> array{GetFoldedParameterValue<T>(
                 context, aRef.base().GetLastSymbol())}) {
    return ApplySubscripts(context.messages(), *array, subscripts);
  } else {
    return std::nullopt;
  }
}

template<typename T>
std::optional<Constant<T>> GetConstantComponent(FoldingContext &context,
    Component &component,
    const std::vector<Constant<SubscriptInteger>> *subscripts) {
  if (std::optional<Constant<SomeDerived>> structures{std::visit(
          common::visitors{
              [&](const Symbol *symbol) {
                return GetFoldedParameterValue<SomeDerived>(context, *symbol);
              },
              [&](ArrayRef &aRef) {
                return FoldArrayRef<SomeDerived>(context, aRef);
              },
              [&](Component &base) {
                return GetConstantComponent<SomeDerived>(context, base);
              },
              [&](CoarrayRef &) -> std::optional<Constant<SomeDerived>> {
                return std::nullopt;
              },
          },
          component.base().u)}) {
    return ApplyComponent<T>(
        context, std::move(*structures), component.GetLastSymbol(), subscripts);
  } else {
    return std::nullopt;
  }
}

template<typename T>
Expr<T> FoldOperation(FoldingContext &context, Designator<T> &&designator) {
  if constexpr (T::category == TypeCategory::Character) {
    if (auto *substring{common::Unwrap<Substring>(designator.u)}) {
      if (std::optional<Expr<SomeCharacter>> folded{substring->Fold(context)}) {
        if (auto value{GetScalarConstantValue<T>(*folded)}) {
          return Expr<T>{*value};
        }
      }
      if (auto length{ToInt64(Fold(context, substring->LEN()))}) {
        if (*length == 0) {
          return Expr<T>{Constant<T>{Scalar<T>{}}};
        }
      }
    }
  }
  return std::visit(
      common::visitors{
          [&](const Symbol *symbol) {
            if (symbol != nullptr) {
              if (auto constant{GetFoldedParameterValue<T>(context, *symbol)}) {
                return Expr<T>{std::move(*constant)};
              }
            }
            return Expr<T>{std::move(designator)};
          },
          [&](ArrayRef &&aRef) {
            aRef = FoldOperation(context, std::move(aRef));
            if (auto c{FoldArrayRef<T>(context, aRef)}) {
              return Expr<T>{std::move(*c)};
            } else {
              return Expr<T>{Designator<T>{std::move(aRef)}};
            }
          },
          [&](Component &&component) {
            component = FoldOperation(context, std::move(component));
            if (auto c{GetConstantComponent<T>(context, component)}) {
              return Expr<T>{std::move(*c)};
            } else {
              return Expr<T>{Designator<T>{std::move(component)}};
            }
          },
          [&](auto &&x) {
            return Expr<T>{Designator<T>{FoldOperation(context, std::move(x))}};
          },
      },
      std::move(designator.u));
}

// Array constructor folding

Expr<ImpliedDoIndex::Result> FoldOperation(
    FoldingContext &context, ImpliedDoIndex &&iDo) {
  if (std::optional<ConstantSubscript> value{context.GetImpliedDo(iDo.name)}) {
    return Expr<ImpliedDoIndex::Result>{*value};
  } else {
    return Expr<ImpliedDoIndex::Result>{std::move(iDo)};
  }
}

template<typename T> class ArrayConstructorFolder {
public:
  explicit ArrayConstructorFolder(const FoldingContext &c) : context_{c} {}

  Expr<T> FoldArray(ArrayConstructor<T> &&array) {
    // Calls FoldArray(const ArrayConstructorValues<T> &) below
    if (FoldArray(array)) {
      auto n{static_cast<ConstantSubscript>(elements_.size())};
      if constexpr (std::is_same_v<T, SomeDerived>) {
        return Expr<T>{Constant<T>{array.GetType().GetDerivedTypeSpec(),
            std::move(elements_), ConstantSubscripts{n}}};
      } else if constexpr (T::category == TypeCategory::Character) {
        auto length{Fold(context_, common::Clone(array.LEN()))};
        if (std::optional<ConstantSubscript> lengthValue{ToInt64(length)}) {
          return Expr<T>{Constant<T>{
              *lengthValue, std::move(elements_), ConstantSubscripts{n}}};
        }
      } else {
        return Expr<T>{
            Constant<T>{std::move(elements_), ConstantSubscripts{n}}};
      }
    }
    return Expr<T>{std::move(array)};
  }

private:
  bool FoldArray(const common::CopyableIndirection<Expr<T>> &expr) {
    Expr<T> folded{Fold(context_, common::Clone(expr.value()))};
    if (const auto *c{UnwrapConstantValue<T>(folded)}) {
      // Copy elements in Fortran array element order
      ConstantSubscripts shape{c->shape()};
      int rank{c->Rank()};
      ConstantSubscripts index(GetRank(shape), 1);
      for (std::size_t n{c->size()}; n-- > 0;) {
        elements_.emplace_back(c->At(index));
        for (int d{0}; d < rank; ++d) {
          if (++index[d] <= shape[d]) {
            break;
          }
          index[d] = 1;
        }
      }
      return true;
    } else {
      return false;
    }
  }
  bool FoldArray(const ImpliedDo<T> &iDo) {
    Expr<SubscriptInteger> lower{
        Fold(context_, Expr<SubscriptInteger>{iDo.lower()})};
    Expr<SubscriptInteger> upper{
        Fold(context_, Expr<SubscriptInteger>{iDo.upper()})};
    Expr<SubscriptInteger> stride{
        Fold(context_, Expr<SubscriptInteger>{iDo.stride()})};
    std::optional<ConstantSubscript> start{ToInt64(lower)}, end{ToInt64(upper)},
        step{ToInt64(stride)};
    if (start.has_value() && end.has_value() && step.has_value()) {
      if (*step == 0) {
        return false;
      }
      bool result{true};
      ConstantSubscript &j{context_.StartImpliedDo(iDo.name(), *start)};
      if (*step > 0) {
        for (; j <= *end; j += *step) {
          result &= FoldArray(iDo.values());
        }
      } else {
        for (; j >= *end; j += *step) {
          result &= FoldArray(iDo.values());
        }
      }
      context_.EndImpliedDo(iDo.name());
      return result;
    } else {
      return false;
    }
  }
  bool FoldArray(const ArrayConstructorValue<T> &x) {
    return std::visit([&](const auto &y) { return FoldArray(y); }, x.u);
  }
  bool FoldArray(const ArrayConstructorValues<T> &xs) {
    for (const auto &x : xs) {
      if (!FoldArray(x)) {
        return false;
      }
    }
    return true;
  }

  FoldingContext context_;
  std::vector<Scalar<T>> elements_;
};

template<typename T>
Expr<T> FoldOperation(FoldingContext &context, ArrayConstructor<T> &&array) {
  return ArrayConstructorFolder<T>{context}.FoldArray(std::move(array));
}

Expr<SomeDerived> FoldOperation(
    FoldingContext &context, StructureConstructor &&structure) {
  StructureConstructor result{structure.derivedTypeSpec()};
  for (auto &&[symbol, value] : std::move(structure)) {
    result.Add(*symbol, Fold(context, std::move(value.value())));
  }
  return Expr<SomeDerived>{Constant<SomeDerived>{result}};
}

// Substitute a bare type parameter reference with its value if it has one now
template<int KIND>
Expr<Type<TypeCategory::Integer, KIND>> FoldOperation(
    FoldingContext &context, TypeParamInquiry<KIND> &&inquiry) {
  using IntKIND = Type<TypeCategory::Integer, KIND>;
  if (!inquiry.base().has_value()) {
    // A "bare" type parameter: replace with its value, if that's now known.
    if (const auto *pdt{context.pdtInstance()}) {
      if (const semantics::Scope * scope{context.pdtInstance()->scope()}) {
        auto iter{scope->find(inquiry.parameter().name())};
        if (iter != scope->end()) {
          const Symbol &symbol{*iter->second};
          const auto *details{symbol.detailsIf<semantics::TypeParamDetails>()};
          if (details && details->init().has_value()) {
            Expr<SomeInteger> expr{*details->init()};
            return Fold(context,
                Expr<IntKIND>{
                    Convert<IntKIND, TypeCategory::Integer>(std::move(expr))});
          }
        }
      }
      if (const auto *value{pdt->FindParameter(inquiry.parameter().name())}) {
        if (value->isExplicit()) {
          return Fold(context,
              Expr<IntKIND>{Convert<IntKIND, TypeCategory::Integer>(
                  Expr<SomeInteger>{value->GetExplicit().value()})});
        }
      }
    }
  }
  return Expr<IntKIND>{std::move(inquiry)};
}

// Array operation elemental application: When all operands to an operation
// are constant arrays, array constructors without any implied DO loops,
// &/or expanded scalars, pull the operation "into" the array result by
// applying it in an elementwise fashion.  For example, [A,1]+[B,2]
// is rewritten into [A+B,1+2] and then partially folded to [A+B,3].

// If possible, restructures an array expression into an array constructor
// that comprises a "flat" ArrayConstructorValues with no implied DO loops.
template<typename T>
bool ArrayConstructorIsFlat(const ArrayConstructorValues<T> &values) {
  for (const ArrayConstructorValue<T> &x : values) {
    if (!std::holds_alternative<Expr<T>>(x.u)) {
      return false;
    }
  }
  return true;
}

template<typename T>
std::optional<Expr<T>> AsFlatArrayConstructor(const Expr<T> &expr) {
  if (const auto *c{UnwrapConstantValue<T>(expr)}) {
    ArrayConstructor<T> result{expr};
    if (c->size() > 0) {
      ConstantSubscripts at{c->lbounds()};
      do {
        result.Push(Expr<T>{Constant<T>{c->At(at)}});
      } while (c->IncrementSubscripts(at));
    }
    return std::make_optional<Expr<T>>(std::move(result));
  } else if (const auto *a{UnwrapExpr<ArrayConstructor<T>>(expr)}) {
    if (ArrayConstructorIsFlat(*a)) {
      return std::make_optional<Expr<T>>(expr);
    }
  } else if (const auto *p{UnwrapExpr<Parentheses<T>>(expr)}) {
    return AsFlatArrayConstructor(Expr<T>{p->left()});
  }
  return std::nullopt;
}

template<TypeCategory CAT>
std::enable_if_t<CAT != TypeCategory::Derived,
    std::optional<Expr<SomeKind<CAT>>>>
AsFlatArrayConstructor(const Expr<SomeKind<CAT>> &expr) {
  return std::visit(
      [&](const auto &kindExpr) -> std::optional<Expr<SomeKind<CAT>>> {
        if (auto flattened{AsFlatArrayConstructor(kindExpr)}) {
          return Expr<SomeKind<CAT>>{std::move(*flattened)};
        } else {
          return std::nullopt;
        }
      },
      expr.u);
}

// FromArrayConstructor is a subroutine for MapOperation() below.
// Given a flat ArrayConstructor<T> and a shape, it wraps the array
// into an Expr<T>, folds it, and returns the resulting wrapped
// array constructor or constant array value.
template<typename T>
Expr<T> FromArrayConstructor(FoldingContext &context,
    ArrayConstructor<T> &&values, std::optional<ConstantSubscripts> &&shape) {
  Expr<T> result{Fold(context, Expr<T>{std::move(values)})};
  if (shape.has_value()) {
    if (auto *constant{UnwrapConstantValue<T>(result)}) {
      return Expr<T>{constant->Reshape(std::move(*shape))};
    }
  }
  return result;
}

// MapOperation is a utility for various specializations of ApplyElementwise()
// that follow.  Given one or two flat ArrayConstructor<OPERAND> (wrapped in an
// Expr<OPERAND>) for some specific operand type(s), apply a given function f
// to each of their corresponding elements to produce a flat
// ArrayConstructor<RESULT> (wrapped in an Expr<RESULT>).
// Preserves shape.

// Unary case
template<typename RESULT, typename OPERAND>
Expr<RESULT> MapOperation(FoldingContext &context,
    std::function<Expr<RESULT>(Expr<OPERAND> &&)> &&f, const Shape &shape,
    Expr<OPERAND> &&values) {
  ArrayConstructor<RESULT> result{values};
  if constexpr (common::HasMember<OPERAND, AllIntrinsicCategoryTypes>) {
    std::visit(
        [&](auto &&kindExpr) {
          using kindType = ResultType<decltype(kindExpr)>;
          auto &aConst{std::get<ArrayConstructor<kindType>>(kindExpr.u)};
          for (auto &acValue : aConst) {
            auto &scalar{std::get<Expr<kindType>>(acValue.u)};
            result.Push(Fold(context, f(Expr<OPERAND>{std::move(scalar)})));
          }
        },
        std::move(values.u));
  } else {
    auto &aConst{std::get<ArrayConstructor<OPERAND>>(values.u)};
    for (auto &acValue : aConst) {
      auto &scalar{std::get<Expr<OPERAND>>(acValue.u)};
      result.Push(Fold(context, f(std::move(scalar))));
    }
  }
  return FromArrayConstructor(
      context, std::move(result), AsConstantExtents(shape));
}

// array * array case
template<typename RESULT, typename LEFT, typename RIGHT>
Expr<RESULT> MapOperation(FoldingContext &context,
    std::function<Expr<RESULT>(Expr<LEFT> &&, Expr<RIGHT> &&)> &&f,
    const Shape &shape, Expr<LEFT> &&leftValues, Expr<RIGHT> &&rightValues) {
  ArrayConstructor<RESULT> result{leftValues};
  auto &leftArrConst{std::get<ArrayConstructor<LEFT>>(leftValues.u)};
  if constexpr (common::HasMember<RIGHT, AllIntrinsicCategoryTypes>) {
    std::visit(
        [&](auto &&kindExpr) {
          using kindType = ResultType<decltype(kindExpr)>;

          auto &rightArrConst{std::get<ArrayConstructor<kindType>>(kindExpr.u)};
          auto rightIter{rightArrConst.begin()};
          for (auto &leftValue : leftArrConst) {
            CHECK(rightIter != rightArrConst.end());
            auto &leftScalar{std::get<Expr<LEFT>>(leftValue.u)};
            auto &rightScalar{std::get<Expr<kindType>>(rightIter->u)};
            result.Push(Fold(context,
                f(std::move(leftScalar), Expr<RIGHT>{std::move(rightScalar)})));
            ++rightIter;
          }
        },
        std::move(rightValues.u));
  } else {
    auto &rightArrConst{std::get<ArrayConstructor<RIGHT>>(rightValues.u)};
    auto rightIter{rightArrConst.begin()};
    for (auto &leftValue : leftArrConst) {
      CHECK(rightIter != rightArrConst.end());
      auto &leftScalar{std::get<Expr<LEFT>>(leftValue.u)};
      auto &rightScalar{std::get<Expr<RIGHT>>(rightIter->u)};
      result.Push(
          Fold(context, f(std::move(leftScalar), std::move(rightScalar))));
      ++rightIter;
    }
  }
  return FromArrayConstructor(
      context, std::move(result), AsConstantExtents(shape));
}

// array * scalar case
template<typename RESULT, typename LEFT, typename RIGHT>
Expr<RESULT> MapOperation(FoldingContext &context,
    std::function<Expr<RESULT>(Expr<LEFT> &&, Expr<RIGHT> &&)> &&f,
    const Shape &shape, Expr<LEFT> &&leftValues,
    const Expr<RIGHT> &rightScalar) {
  ArrayConstructor<RESULT> result{leftValues};
  auto &leftArrConst{std::get<ArrayConstructor<LEFT>>(leftValues.u)};
  for (auto &leftValue : leftArrConst) {
    auto &leftScalar{std::get<Expr<LEFT>>(leftValue.u)};
    result.Push(
        Fold(context, f(std::move(leftScalar), Expr<RIGHT>{rightScalar})));
  }
  return FromArrayConstructor(
      context, std::move(result), AsConstantExtents(shape));
}

// scalar * array case
template<typename RESULT, typename LEFT, typename RIGHT>
Expr<RESULT> MapOperation(FoldingContext &context,
    std::function<Expr<RESULT>(Expr<LEFT> &&, Expr<RIGHT> &&)> &&f,
    const Shape &shape, const Expr<LEFT> &leftScalar,
    Expr<RIGHT> &&rightValues) {
  ArrayConstructor<RESULT> result{leftScalar};
  if constexpr (common::HasMember<RIGHT, AllIntrinsicCategoryTypes>) {
    std::visit(
        [&](auto &&kindExpr) {
          using kindType = ResultType<decltype(kindExpr)>;
          auto &rightArrConst{std::get<ArrayConstructor<kindType>>(kindExpr.u)};
          for (auto &rightValue : rightArrConst) {
            auto &rightScalar{std::get<Expr<kindType>>(rightValue.u)};
            result.Push(Fold(context,
                f(Expr<LEFT>{leftScalar},
                    Expr<RIGHT>{std::move(rightScalar)})));
          }
        },
        std::move(rightValues.u));
  } else {
    auto &rightArrConst{std::get<ArrayConstructor<RIGHT>>(rightValues.u)};
    for (auto &rightValue : rightArrConst) {
      auto &rightScalar{std::get<Expr<RIGHT>>(rightValue.u)};
      result.Push(
          Fold(context, f(Expr<LEFT>{leftScalar}, std::move(rightScalar))));
    }
  }
  return FromArrayConstructor(
      context, std::move(result), AsConstantExtents(shape));
}

// ApplyElementwise() recursively folds the operand expression(s) of an
// operation, then attempts to apply the operation to the (corresponding)
// scalar element(s) of those operands.  Returns std::nullopt for scalars
// or unlinearizable operands.
template<typename DERIVED, typename RESULT, typename OPERAND>
auto ApplyElementwise(FoldingContext &context,
    Operation<DERIVED, RESULT, OPERAND> &operation,
    std::function<Expr<RESULT>(Expr<OPERAND> &&)> &&f)
    -> std::optional<Expr<RESULT>> {
  auto &expr{operation.left()};
  expr = Fold(context, std::move(expr));
  if (expr.Rank() > 0) {
    if (std::optional<Shape> shape{GetShape(context, expr)}) {
      if (auto values{AsFlatArrayConstructor(expr)}) {
        return MapOperation(context, std::move(f), *shape, std::move(*values));
      }
    }
  }
  return std::nullopt;
}

template<typename DERIVED, typename RESULT, typename OPERAND>
auto ApplyElementwise(
    FoldingContext &context, Operation<DERIVED, RESULT, OPERAND> &operation)
    -> std::optional<Expr<RESULT>> {
  return ApplyElementwise(context, operation,
      std::function<Expr<RESULT>(Expr<OPERAND> &&)>{
          [](Expr<OPERAND> &&operand) {
            return Expr<RESULT>{DERIVED{std::move(operand)}};
          }});
}

// Predicate: is a scalar expression suitable for naive scalar expansion
// in the flattening of an array expression?
// TODO: capture such scalar expansions in temporaries, flatten everything
struct UnexpandabilityFindingVisitor : public virtual VisitorBase<bool> {
  using Result = bool;
  explicit UnexpandabilityFindingVisitor(int) { result() = false; }
  template<typename T> void Handle(FunctionRef<T> &) { Return(true); }
  template<typename T> void Handle(CoarrayRef &) { Return(true); }
};

template<typename T> bool IsExpandableScalar(const Expr<T> &expr) {
  return !Visitor<UnexpandabilityFindingVisitor>{0}.Traverse(expr);
}

template<typename DERIVED, typename RESULT, typename LEFT, typename RIGHT>
auto ApplyElementwise(FoldingContext &context,
    Operation<DERIVED, RESULT, LEFT, RIGHT> &operation,
    std::function<Expr<RESULT>(Expr<LEFT> &&, Expr<RIGHT> &&)> &&f)
    -> std::optional<Expr<RESULT>> {
  auto &leftExpr{operation.left()};
  leftExpr = Fold(context, std::move(leftExpr));
  auto &rightExpr{operation.right()};
  rightExpr = Fold(context, std::move(rightExpr));
  if (leftExpr.Rank() > 0) {
    if (std::optional<Shape> leftShape{GetShape(context, leftExpr)}) {
      if (auto left{AsFlatArrayConstructor(leftExpr)}) {
        if (rightExpr.Rank() > 0) {
          if (std::optional<Shape> rightShape{GetShape(context, rightExpr)}) {
            if (auto right{AsFlatArrayConstructor(rightExpr)}) {
              CheckConformance(context.messages(), *leftShape, *rightShape);
              return MapOperation(context, std::move(f), *leftShape,
                  std::move(*left), std::move(*right));
            }
          }
        } else if (IsExpandableScalar(rightExpr)) {
          return MapOperation(
              context, std::move(f), *leftShape, std::move(*left), rightExpr);
        }
      }
    }
  } else if (rightExpr.Rank() > 0 && IsExpandableScalar(leftExpr)) {
    if (std::optional<Shape> shape{GetShape(context, rightExpr)}) {
      if (auto right{AsFlatArrayConstructor(rightExpr)}) {
        return MapOperation(
            context, std::move(f), *shape, leftExpr, std::move(*right));
      }
    }
  }
  return std::nullopt;
}

template<typename DERIVED, typename RESULT, typename LEFT, typename RIGHT>
auto ApplyElementwise(
    FoldingContext &context, Operation<DERIVED, RESULT, LEFT, RIGHT> &operation)
    -> std::optional<Expr<RESULT>> {
  return ApplyElementwise(context, operation,
      std::function<Expr<RESULT>(Expr<LEFT> &&, Expr<RIGHT> &&)>{
          [](Expr<LEFT> &&left, Expr<RIGHT> &&right) {
            return Expr<RESULT>{DERIVED{std::move(left), std::move(right)}};
          }});
}

// Unary operations

template<typename TO, typename FROM>
common::IfNoLvalue<std::optional<TO>, FROM> ConvertString(FROM &&s) {
  if constexpr (std::is_same_v<TO, FROM>) {
    return std::make_optional<TO>(std::move(s));
  } else {
    // Fortran character conversion is well defined between distinct kinds
    // only when the actual characters are valid 7-bit ASCII.
    TO str;
    for (auto iter{s.cbegin()}; iter != s.cend(); ++iter) {
      if (static_cast<std::uint64_t>(*iter) > 127) {
        return std::nullopt;
      }
      str.push_back(*iter);
    }
    return std::make_optional<TO>(std::move(str));
  }
}

template<typename TO, TypeCategory FROMCAT>
Expr<TO> FoldOperation(
    FoldingContext &context, Convert<TO, FROMCAT> &&convert) {
  if (auto array{ApplyElementwise(context, convert)}) {
    return *array;
  }
  return std::visit(
      [&](auto &kindExpr) -> Expr<TO> {
        using Operand = ResultType<decltype(kindExpr)>;
        char buffer[64];
        if (auto value{GetScalarConstantValue<Operand>(kindExpr)}) {
          if constexpr (TO::category == TypeCategory::Integer) {
            if constexpr (Operand::category == TypeCategory::Integer) {
              auto converted{Scalar<TO>::ConvertSigned(*value)};
              if (converted.overflow) {
                context.messages().Say(
                    "INTEGER(%d) to INTEGER(%d) conversion overflowed"_en_US,
                    Operand::kind, TO::kind);
              }
              return ScalarConstantToExpr(std::move(converted.value));
            } else if constexpr (Operand::category == TypeCategory::Real) {
              auto converted{value->template ToInteger<Scalar<TO>>()};
              if (converted.flags.test(RealFlag::InvalidArgument)) {
                context.messages().Say(
                    "REAL(%d) to INTEGER(%d) conversion: invalid argument"_en_US,
                    Operand::kind, TO::kind);
              } else if (converted.flags.test(RealFlag::Overflow)) {
                context.messages().Say(
                    "REAL(%d) to INTEGER(%d) conversion overflowed"_en_US,
                    Operand::kind, TO::kind);
              }
              return ScalarConstantToExpr(std::move(converted.value));
            }
          } else if constexpr (TO::category == TypeCategory::Real) {
            if constexpr (Operand::category == TypeCategory::Integer) {
              auto converted{Scalar<TO>::FromInteger(*value)};
              if (!converted.flags.empty()) {
                std::snprintf(buffer, sizeof buffer,
                    "INTEGER(%d) to REAL(%d) conversion", Operand::kind,
                    TO::kind);
                RealFlagWarnings(context, converted.flags, buffer);
              }
              return ScalarConstantToExpr(std::move(converted.value));
            } else if constexpr (Operand::category == TypeCategory::Real) {
              auto converted{Scalar<TO>::Convert(*value)};
              if (!converted.flags.empty()) {
                std::snprintf(buffer, sizeof buffer,
                    "REAL(%d) to REAL(%d) conversion", Operand::kind, TO::kind);
                RealFlagWarnings(context, converted.flags, buffer);
              }
              if (context.flushSubnormalsToZero()) {
                converted.value = converted.value.FlushSubnormalToZero();
              }
              return ScalarConstantToExpr(std::move(converted.value));
            }
          } else if constexpr (TO::category == TypeCategory::Character &&
              Operand::category == TypeCategory::Character) {
            if (auto converted{ConvertString<Scalar<TO>>(std::move(*value))}) {
              return ScalarConstantToExpr(std::move(*converted));
            }
          } else if constexpr (TO::category == TypeCategory::Logical &&
              Operand::category == TypeCategory::Logical) {
            return Expr<TO>{value->IsTrue()};
          }
        }
        return Expr<TO>{std::move(convert)};
      },
      convert.left().u);
}

template<typename T>
Expr<T> FoldOperation(FoldingContext &context, Parentheses<T> &&x) {
  auto &operand{x.left()};
  operand = Fold(context, std::move(operand));
  if (auto value{GetScalarConstantValue<T>(operand)}) {
    // Preserve parentheses, even around constants.
    return Expr<T>{Parentheses<T>{Expr<T>{Constant<T>{*value}}}};
  }
  return Expr<T>{Parentheses<T>{std::move(operand)}};
}

template<typename T>
Expr<T> FoldOperation(FoldingContext &context, Negate<T> &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  auto &operand{x.left()};
  if (auto value{GetScalarConstantValue<T>(operand)}) {
    if constexpr (T::category == TypeCategory::Integer) {
      auto negated{value->Negate()};
      if (negated.overflow) {
        context.messages().Say(
            "INTEGER(%d) negation overflowed"_en_US, T::kind);
      }
      return Expr<T>{Constant<T>{std::move(negated.value)}};
    } else {
      // REAL & COMPLEX negation: no exceptions possible
      return Expr<T>{Constant<T>{value->Negate()}};
    }
  }
  return Expr<T>{std::move(x)};
}

template<int KIND>
Expr<Type<TypeCategory::Real, KIND>> FoldOperation(
    FoldingContext &context, ComplexComponent<KIND> &&x) {
  using Operand = Type<TypeCategory::Complex, KIND>;
  using Result = Type<TypeCategory::Real, KIND>;
  if (auto array{ApplyElementwise(context, x,
          std::function<Expr<Result>(Expr<Operand> &&)>{
              [=](Expr<Operand> &&operand) {
                return Expr<Result>{ComplexComponent<KIND>{
                    x.isImaginaryPart, std::move(operand)}};
              }})}) {
    return *array;
  }
  using Part = Type<TypeCategory::Real, KIND>;
  auto &operand{x.left()};
  if (auto value{GetScalarConstantValue<Operand>(operand)}) {
    if (x.isImaginaryPart) {
      return Expr<Part>{Constant<Part>{value->AIMAG()}};
    } else {
      return Expr<Part>{Constant<Part>{value->REAL()}};
    }
  }
  return Expr<Part>{std::move(x)};
}

template<int KIND>
Expr<Type<TypeCategory::Logical, KIND>> FoldOperation(
    FoldingContext &context, Not<KIND> &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  using Ty = Type<TypeCategory::Logical, KIND>;
  auto &operand{x.left()};
  if (auto value{GetScalarConstantValue<Ty>(operand)}) {
    return Expr<Ty>{Constant<Ty>{!value->IsTrue()}};
  }
  return Expr<Ty>{x};
}

// Binary (dyadic) operations

template<typename LEFT, typename RIGHT>
std::optional<std::pair<Scalar<LEFT>, Scalar<RIGHT>>> OperandsAreConstants(
    const Expr<LEFT> &x, const Expr<RIGHT> &y) {
  if (auto xvalue{GetScalarConstantValue<LEFT>(x)}) {
    if (auto yvalue{GetScalarConstantValue<RIGHT>(y)}) {
      return {std::make_pair(*xvalue, *yvalue)};
    }
  }
  return std::nullopt;
}

template<typename DERIVED, typename RESULT, typename LEFT, typename RIGHT>
std::optional<std::pair<Scalar<LEFT>, Scalar<RIGHT>>> OperandsAreConstants(
    const Operation<DERIVED, RESULT, LEFT, RIGHT> &operation) {
  return OperandsAreConstants(operation.left(), operation.right());
}

template<typename T>
Expr<T> FoldOperation(FoldingContext &context, Add<T> &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  if (auto folded{OperandsAreConstants(x)}) {
    if constexpr (T::category == TypeCategory::Integer) {
      auto sum{folded->first.AddSigned(folded->second)};
      if (sum.overflow) {
        context.messages().Say(
            "INTEGER(%d) addition overflowed"_en_US, T::kind);
      }
      return Expr<T>{Constant<T>{sum.value}};
    } else {
      auto sum{folded->first.Add(folded->second, context.rounding())};
      RealFlagWarnings(context, sum.flags, "addition");
      if (context.flushSubnormalsToZero()) {
        sum.value = sum.value.FlushSubnormalToZero();
      }
      return Expr<T>{Constant<T>{sum.value}};
    }
  }
  return Expr<T>{std::move(x)};
}

template<typename T>
Expr<T> FoldOperation(FoldingContext &context, Subtract<T> &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  if (auto folded{OperandsAreConstants(x)}) {
    if constexpr (T::category == TypeCategory::Integer) {
      auto difference{folded->first.SubtractSigned(folded->second)};
      if (difference.overflow) {
        context.messages().Say(
            "INTEGER(%d) subtraction overflowed"_en_US, T::kind);
      }
      return Expr<T>{Constant<T>{difference.value}};
    } else {
      auto difference{
          folded->first.Subtract(folded->second, context.rounding())};
      RealFlagWarnings(context, difference.flags, "subtraction");
      if (context.flushSubnormalsToZero()) {
        difference.value = difference.value.FlushSubnormalToZero();
      }
      return Expr<T>{Constant<T>{difference.value}};
    }
  }
  return Expr<T>{std::move(x)};
}

template<typename T>
Expr<T> FoldOperation(FoldingContext &context, Multiply<T> &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  if (auto folded{OperandsAreConstants(x)}) {
    if constexpr (T::category == TypeCategory::Integer) {
      auto product{folded->first.MultiplySigned(folded->second)};
      if (product.SignedMultiplicationOverflowed()) {
        context.messages().Say(
            "INTEGER(%d) multiplication overflowed"_en_US, T::kind);
      }
      return Expr<T>{Constant<T>{product.lower}};
    } else {
      auto product{folded->first.Multiply(folded->second, context.rounding())};
      RealFlagWarnings(context, product.flags, "multiplication");
      if (context.flushSubnormalsToZero()) {
        product.value = product.value.FlushSubnormalToZero();
      }
      return Expr<T>{Constant<T>{product.value}};
    }
  }
  return Expr<T>{std::move(x)};
}

template<typename T>
Expr<T> FoldOperation(FoldingContext &context, Divide<T> &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  if (auto folded{OperandsAreConstants(x)}) {
    if constexpr (T::category == TypeCategory::Integer) {
      auto quotAndRem{folded->first.DivideSigned(folded->second)};
      if (quotAndRem.divisionByZero) {
        context.messages().Say("INTEGER(%d) division by zero"_en_US, T::kind);
        return Expr<T>{std::move(x)};
      }
      if (quotAndRem.overflow) {
        context.messages().Say(
            "INTEGER(%d) division overflowed"_en_US, T::kind);
      }
      return Expr<T>{Constant<T>{quotAndRem.quotient}};
    } else {
      auto quotient{folded->first.Divide(folded->second, context.rounding())};
      RealFlagWarnings(context, quotient.flags, "division");
      if (context.flushSubnormalsToZero()) {
        quotient.value = quotient.value.FlushSubnormalToZero();
      }
      return Expr<T>{Constant<T>{quotient.value}};
    }
  }
  return Expr<T>{std::move(x)};
}

template<typename T>
Expr<T> FoldOperation(FoldingContext &context, Power<T> &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  if (auto folded{OperandsAreConstants(x)}) {
    if constexpr (T::category == TypeCategory::Integer) {
      auto power{folded->first.Power(folded->second)};
      if (power.divisionByZero) {
        context.messages().Say(
            "INTEGER(%d) zero to negative power"_en_US, T::kind);
      } else if (power.overflow) {
        context.messages().Say("INTEGER(%d) power overflowed"_en_US, T::kind);
      } else if (power.zeroToZero) {
        context.messages().Say(
            "INTEGER(%d) 0**0 is not defined"_en_US, T::kind);
      }
      return Expr<T>{Constant<T>{power.power}};
    } else {
      // TODO: real & complex power with non-integral exponent
    }
  }
  return Expr<T>{std::move(x)};
}

template<typename T>
Expr<T> FoldOperation(FoldingContext &context, RealToIntPower<T> &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  return std::visit(
      [&](auto &y) -> Expr<T> {
        if (auto folded{OperandsAreConstants(x.left(), y)}) {
          auto power{evaluate::IntPower(folded->first, folded->second)};
          RealFlagWarnings(context, power.flags, "power with INTEGER exponent");
          if (context.flushSubnormalsToZero()) {
            power.value = power.value.FlushSubnormalToZero();
          }
          return Expr<T>{Constant<T>{power.value}};
        } else {
          return Expr<T>{std::move(x)};
        }
      },
      x.right().u);
}

template<typename T>
Expr<T> FoldOperation(FoldingContext &context, Extremum<T> &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  if (auto folded{OperandsAreConstants(x)}) {
    if constexpr (T::category == TypeCategory::Integer) {
      if (folded->first.CompareSigned(folded->second) == x.ordering) {
        return Expr<T>{Constant<T>{folded->first}};
      }
    } else if constexpr (T::category == TypeCategory::Real) {
      if (folded->first.IsNotANumber() ||
          (folded->first.Compare(folded->second) == Relation::Less) ==
              (x.ordering == Ordering::Less)) {
        return Expr<T>{Constant<T>{folded->first}};
      }
    } else {
      if (x.ordering == Compare(folded->first, folded->second)) {
        return Expr<T>{Constant<T>{folded->first}};
      }
    }
    return Expr<T>{Constant<T>{folded->second}};
  }
  return Expr<T>{std::move(x)};
}

template<int KIND>
Expr<Type<TypeCategory::Complex, KIND>> FoldOperation(
    FoldingContext &context, ComplexConstructor<KIND> &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  using Result = Type<TypeCategory::Complex, KIND>;
  if (auto folded{OperandsAreConstants(x)}) {
    return Expr<Result>{
        Constant<Result>{Scalar<Result>{folded->first, folded->second}}};
  }
  return Expr<Result>{std::move(x)};
}

template<int KIND>
Expr<Type<TypeCategory::Character, KIND>> FoldOperation(
    FoldingContext &context, Concat<KIND> &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  using Result = Type<TypeCategory::Character, KIND>;
  if (auto folded{OperandsAreConstants(x)}) {
    return Expr<Result>{Constant<Result>{folded->first + folded->second}};
  }
  return Expr<Result>{std::move(x)};
}

template<int KIND>
Expr<Type<TypeCategory::Character, KIND>> FoldOperation(
    FoldingContext &context, SetLength<KIND> &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  using Result = Type<TypeCategory::Character, KIND>;
  if (auto folded{OperandsAreConstants(x)}) {
    auto oldLength{static_cast<ConstantSubscript>(folded->first.size())};
    auto newLength{folded->second.ToInt64()};
    if (newLength < oldLength) {
      folded->first.erase(newLength);
    } else {
      folded->first.append(newLength - oldLength, ' ');
    }
    CHECK(static_cast<ConstantSubscript>(folded->first.size()) == newLength);
    return Expr<Result>{Constant<Result>{std::move(folded->first)}};
  }
  return Expr<Result>{std::move(x)};
}

template<typename T>
Expr<LogicalResult> FoldOperation(
    FoldingContext &context, Relational<T> &&relation) {
  if (auto array{ApplyElementwise(context, relation,
          std::function<Expr<LogicalResult>(Expr<T> &&, Expr<T> &&)>{
              [=](Expr<T> &&x, Expr<T> &&y) {
                return Expr<LogicalResult>{Relational<SomeType>{
                    Relational<T>{relation.opr, std::move(x), std::move(y)}}};
              }})}) {
    return *array;
  }
  if (auto folded{OperandsAreConstants(relation)}) {
    bool result{};
    if constexpr (T::category == TypeCategory::Integer) {
      result =
          Satisfies(relation.opr, folded->first.CompareSigned(folded->second));
    } else if constexpr (T::category == TypeCategory::Real) {
      result = Satisfies(relation.opr, folded->first.Compare(folded->second));
    } else if constexpr (T::category == TypeCategory::Character) {
      result = Satisfies(relation.opr, Compare(folded->first, folded->second));
    } else {
      static_assert(T::category != TypeCategory::Complex &&
          T::category != TypeCategory::Logical);
    }
    return Expr<LogicalResult>{Constant<LogicalResult>{result}};
  }
  return Expr<LogicalResult>{Relational<SomeType>{std::move(relation)}};
}

inline Expr<LogicalResult> FoldOperation(
    FoldingContext &context, Relational<SomeType> &&relation) {
  return std::visit(
      [&](auto &&x) {
        return Expr<LogicalResult>{FoldOperation(context, std::move(x))};
      },
      std::move(relation.u));
}

template<int KIND>
Expr<Type<TypeCategory::Logical, KIND>> FoldOperation(
    FoldingContext &context, LogicalOperation<KIND> &&operation) {
  using LOGICAL = Type<TypeCategory::Logical, KIND>;
  if (auto array{ApplyElementwise(context, operation,
          std::function<Expr<LOGICAL>(Expr<LOGICAL> &&, Expr<LOGICAL> &&)>{
              [=](Expr<LOGICAL> &&x, Expr<LOGICAL> &&y) {
                return Expr<LOGICAL>{LogicalOperation<KIND>{
                    operation.logicalOperator, std::move(x), std::move(y)}};
              }})}) {
    return *array;
  }
  if (auto folded{OperandsAreConstants(operation)}) {
    bool xt{folded->first.IsTrue()}, yt{folded->second.IsTrue()}, result{};
    switch (operation.logicalOperator) {
    case LogicalOperator::And: result = xt && yt; break;
    case LogicalOperator::Or: result = xt || yt; break;
    case LogicalOperator::Eqv: result = xt == yt; break;
    case LogicalOperator::Neqv: result = xt != yt; break;
    }
    return Expr<LOGICAL>{Constant<LOGICAL>{result}};
  }
  return Expr<LOGICAL>{std::move(operation)};
}

// end per-operation folding functions

template<typename T>
Expr<T> ExpressionBase<T>::Rewrite(FoldingContext &context, Expr<T> &&expr) {
  return std::visit(
      [&](auto &&x) -> Expr<T> {
        if constexpr (IsSpecificIntrinsicType<T>) {
          return FoldOperation(context, std::move(x));
        } else if constexpr (std::is_same_v<T, SomeDerived>) {
          return FoldOperation(context, std::move(x));
        } else if constexpr (common::HasMember<decltype(x),
                                 TypelessExpression>) {
          return std::move(expr);
        } else {
          return Expr<T>{Fold(context, std::move(x))};
        }
      },
      std::move(expr.u));
}

FOR_EACH_TYPE_AND_KIND(template class ExpressionBase, )

// Constant expression predicate IsConstantExpr().
// This code determines whether an expression is a "constant expression"
// in the sense of section 10.1.12.  This is not the same thing as being
// able to fold it (yet) into a known constant value; specifically,
// the expression may reference derived type kind parameters whose values
// are not yet known.
class IsConstantExprVisitor : public virtual VisitorBase<bool> {
public:
  using Result = bool;
  explicit IsConstantExprVisitor(int) { result() = true; }

  template<int KIND> void Handle(const TypeParamInquiry<KIND> &inq) {
    Check(inq.parameter().template get<semantics::TypeParamDetails>().attr() ==
        common::TypeParamAttr::Kind);
  }
  void Handle(const semantics::Symbol &symbol) {
    Check(symbol.attrs().test(semantics::Attr::PARAMETER));
  }
  void Handle(const CoarrayRef &) { Return(false); }
  void Pre(const semantics::ParamValue &param) { Check(param.isExplicit()); }
  template<typename T> void Pre(const FunctionRef<T> &call) {
    if (const auto *intrinsic{std::get_if<SpecificIntrinsic>(&call.proc().u)}) {
      Check(intrinsic->name == "kind");
      // TODO: Obviously many other intrinsics can be allowed
    } else {
      Return(false);
    }
  }

  // Forbid integer divide by zero in constants.
  template<int KIND>
  void Handle(const Divide<Type<TypeCategory::Integer, KIND>> &division) {
    using T = Type<TypeCategory::Integer, KIND>;
    if (const auto divisor{GetScalarConstantValue<T>(division.right())}) {
      Check(!divisor->IsZero());
    }
  }

private:
  void Check(bool ok) {
    if (!ok) {
      Return(false);
    }
  }
};

template<typename A> bool IsConstantExpr(const A &x) {
  return Visitor<IsConstantExprVisitor>{0}.Traverse(x);
}

bool IsConstantExpr(const Expr<SomeType> &expr) {
  return Visitor<IsConstantExprVisitor>{0}.Traverse(expr);
}

std::optional<std::int64_t> ToInt64(const Expr<SomeInteger> &expr) {
  return std::visit(
      [](const auto &kindExpr) { return ToInt64(kindExpr); }, expr.u);
}

std::optional<std::int64_t> ToInt64(const Expr<SomeType> &expr) {
  if (const auto *intExpr{UnwrapExpr<Expr<SomeInteger>>(expr)}) {
    return ToInt64(*intExpr);
  } else {
    return std::nullopt;
  }
}

// Object pointer initialization checking predicate IsInitialDataTarget().
// This code determines whether an expression is allowable as the static
// data address used to initialize a pointer with "=> x".  See C765.
// The caller is responsible for checking the base object symbol's
// characteristics (TARGET, SAVE, &c.) since this code can't use GetUltimate().
template<typename A> bool IsInitialDataTarget(const A &) { return false; }
template<typename... A> bool IsInitialDataTarget(const std::variant<A...> &);
bool IsInitialDataTarget(const DataRef &);
template<typename T> bool IsInitialDataTarget(const Expr<T> &);
bool IsInitialDataTarget(const semantics::Symbol &s) { return true; }
bool IsInitialDataTarget(const semantics::Symbol *s) {
  return IsInitialDataTarget(*s);
}
bool IsInitialDataTarget(const Component &x) {
  return IsInitialDataTarget(x.base());
}
bool IsInitialDataTarget(const NamedEntity &x) {
  if (const Component * c{x.UnwrapComponent()}) {
    return IsInitialDataTarget(*c);
  } else {
    return IsInitialDataTarget(x.GetLastSymbol());
  }
}
bool IsInitialDataTarget(const Triplet &x) {
  if (auto lower{x.lower()}) {
    if (!IsConstantExpr(*lower)) {
      return false;
    }
  }
  if (auto upper{x.upper()}) {
    if (!IsConstantExpr(*upper)) {
      return false;
    }
  }
  return IsConstantExpr(x.stride());
}
bool IsInitialDataTarget(const Subscript &x) {
  return std::visit(
      common::visitors{
          [](const Triplet &t) { return IsInitialDataTarget(t); },
          [&](const auto &y) {
            return y.value().Rank() == 0 && IsConstantExpr(y.value());
          },
      },
      x.u);
}
bool IsInitialDataTarget(const ArrayRef &x) {
  for (const Subscript &ss : x.subscript()) {
    if (!IsInitialDataTarget(ss)) {
      return false;
    }
  }
  return IsInitialDataTarget(x.base());
}
bool IsInitialDataTarget(const DataRef &x) { return IsInitialDataTarget(x.u); }
bool IsInitialDataTarget(const Substring &x) {
  return IsConstantExpr(x.lower()) && IsConstantExpr(x.upper());
}
bool IsInitialDataTarget(const ComplexPart &x) {
  return IsInitialDataTarget(x.complex());
}
template<typename T> bool IsInitialDataTarget(const Designator<T> &x) {
  return IsInitialDataTarget(x.u);
}
bool IsInitialDataTarget(const NullPointer &) { return true; }
template<typename T> bool IsInitialDataTarget(const Expr<T> &x) {
  return IsInitialDataTarget(x.u);
}
template<typename... A> bool IsInitialDataTarget(const std::variant<A...> &u) {
  return std::visit([](const auto &x) { return IsInitialDataTarget(x); }, u);
}

bool IsInitialDataTarget(const Expr<SomeType> &x) {
  return IsInitialDataTarget(x.u);
}

}
