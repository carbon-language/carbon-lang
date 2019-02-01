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
#include "common.h"
#include "constant.h"
#include "expression.h"
#include "host.h"
#include "int-power.h"
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
#include <type_traits>
#include <variant>

namespace Fortran::evaluate {

// no-op base case
template<typename A>
Expr<ResultType<A>> FoldOperation(FoldingContext &, A &&x) {
  return Expr<ResultType<A>>{std::move(x)};
}

// Forward declarations of overloads, template instantiations, and template
// specializations of FoldOperation() to enable mutual recursion between them.
BaseObject FoldOperation(FoldingContext &, BaseObject &&);
Component FoldOperation(FoldingContext &, Component &&);
Triplet FoldOperation(FoldingContext &, Triplet &&);
Subscript FoldOperation(FoldingContext &, Subscript &&);
ArrayRef FoldOperation(FoldingContext &, ArrayRef &&);
CoarrayRef FoldOperation(FoldingContext &, CoarrayRef &&);
DataRef FoldOperation(FoldingContext &, DataRef &&);
Substring FoldOperation(FoldingContext &, Substring &&);
ComplexPart FoldOperation(FoldingContext &, ComplexPart &&);
template<int KIND>
Expr<Type<TypeCategory::Integer, KIND>> FoldOperation(FoldingContext &context,
    FunctionRef<Type<TypeCategory::Integer, KIND>> &&funcRef);
template<typename T> Expr<T> FoldOperation(FoldingContext &, Designator<T> &&);
template<int KIND>
Expr<Type<TypeCategory::Integer, KIND>> FoldOperation(
    FoldingContext &, TypeParamInquiry<KIND> &&);
template<typename T>
Expr<T> FoldOperation(FoldingContext &, ArrayConstructor<T> &&);
Expr<SomeDerived> FoldOperation(FoldingContext &, StructureConstructor &&);

// Overloads, instantiations, and specializations of FoldOperation().

BaseObject FoldOperation(FoldingContext &, BaseObject &&object) {
  return std::move(object);
}

Component FoldOperation(FoldingContext &context, Component &&component) {
  return {FoldOperation(context, std::move(component.base())),
      component.GetLastSymbol()};
}

Triplet FoldOperation(FoldingContext &context, Triplet &&triplet) {
  return {Fold(context, triplet.lower()), Fold(context, triplet.upper()),
      Fold(context, common::Clone(triplet.stride()))};
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
  for (Subscript &subscript : arrayRef.subscript()) {
    subscript = FoldOperation(context, std::move(subscript));
  }
  return std::visit(
      common::visitors{
          [&](const Symbol *symbol) {
            return ArrayRef{*symbol, std::move(arrayRef.subscript())};
          },
          [&](Component &&component) {
            return ArrayRef{FoldOperation(context, std::move(component)),
                std::move(arrayRef.subscript())};
          },
      },
      std::move(arrayRef.base()));
}

CoarrayRef FoldOperation(FoldingContext &context, CoarrayRef &&coarrayRef) {
  auto base{coarrayRef.base()};
  std::vector<Expr<SubscriptInteger>> subscript, cosubscript;
  for (Expr<SubscriptInteger> x : coarrayRef.subscript()) {
    subscript.emplace_back(Fold(context, std::move(x)));
  }
  for (Expr<SubscriptInteger> x : coarrayRef.cosubscript()) {
    cosubscript.emplace_back(Fold(context, std::move(x)));
  }
  CoarrayRef folded{
      std::move(base), std::move(subscript), std::move(cosubscript)};
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
  std::optional<Expr<SubscriptInteger>> lower{Fold(context, substring.lower())};
  std::optional<Expr<SubscriptInteger>> upper{Fold(context, substring.upper())};
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
namespace intrinsicHelper {
// helper to produce hash of intrinsic names based the first 3 letters. All
// intrinsic names are longer than 3 letters
static constexpr inline std::int32_t CommonHash(const char *s, std::size_t n) {
  if (n < 3) {
    return 0;
  }
  return (((static_cast<std::int32_t>(s[0]) << 8) + s[1]) << 8) + s[2];
}

static constexpr std::int32_t operator"" _hash(const char *s, std::size_t n) {
  return CommonHash(s, n);
}

static std::int32_t DynamicHash(const std::string &s) {
  return CommonHash(s.data(), s.size());
}

// Define function pointer and callable types used in a common utility that
// takes care of array and cast/conversion aspects for elemental intrinsics
// Note: math complex functions from <complex> are passing arg as const ref
template<typename TR, typename... TA> using FuncPointer = TR (*)(TA...);

template<typename TR, typename... TA>
using HostFuncPointer = FuncPointer<Host::HostType<TR>,
    std::conditional_t<TA::category == TypeCategory::Complex,
        const Host::HostType<TA> &, Host::HostType<TA>>...>;

template<typename TR, typename... TArgs>
using ScalarFunc = std::function<Scalar<TR>(const Scalar<TArgs> &...)>;

// Helper that build std::function operating on Scalar types from host runtime
// function. There is version that only works if the scalar has a matching host
// type and one that allow conversions of scalar types toward "bigger" host
// types. By "bigger", it is meant that all the scalar types can be converted to
// and from this host type without any precision loss. The purpose of this is
// mainly to allow folding of 16 bits float intrinsic function with the host
// runtime for 32bit floats when it is acceptable (e.g acos).
template<typename TR, typename... TA>
static constexpr inline ScalarFunc<TR, TA...> HostFuncWrap(
    HostFuncPointer<TR, TA...> func) {
  return [=](const Scalar<TA> &... x) -> Scalar<TR> {
    // TODO fp-exception
    return Host::CastHostToFortran<TR>(func(Host::CastFortranToHost<TA>(x)...));
  };
}

// A utility that applies a scalar function over arrays or scalar for elemental
// intrinsics.
template<typename TR, typename... TA, std::size_t... I>
static inline Expr<TR> FoldElementalIntrinsicHelper(FunctionRef<TR> &&funcRef,
    ScalarFunc<TR, TA...> scalarFunc, std::index_sequence<I...>) {
  static_assert(
      (... && IsSpecificIntrinsicType<TA>));  // TODO derived types for MERGE?
  std::tuple<const std::optional<Scalar<TA>>...> scalars{
      GetScalarConstantValue<TA>(*funcRef.arguments()[I]->value)...};
  if ((... && std::get<I>(scalars).has_value())) {
    return Expr<TR>{Constant<TR>{scalarFunc(*std::get<I>(scalars)...)}};
  }
  // TODO: handle arrays when Constant<T> can represent them
  return Expr<TR>{std::move(funcRef)};
}

template<typename TR, typename... TA>
static Expr<TR> FoldElementalIntrinsic(
    FunctionRef<TR> &&funcRef, ScalarFunc<TR, TA...> scalarFunc) {
  return FoldElementalIntrinsicHelper<TR, TA...>(
      std::move(funcRef), scalarFunc, std::index_sequence_for<TA...>{});
}
}

template<int KIND>
Expr<Type<TypeCategory::Integer, KIND>> FoldOperation(FoldingContext &context,
    FunctionRef<Type<TypeCategory::Integer, KIND>> &&funcRef) {
  using namespace intrinsicHelper;
  using T = Type<TypeCategory::Integer, KIND>;
  for (std::optional<ActualArgument> &arg : funcRef.arguments()) {
    if (arg.has_value()) {
      arg.value().value() =
          FoldOperation(context, std::move(arg.value().value()));
    }
  }
  if (auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)}) {
    std::string name{intrinsic->name};
    switch (DynamicHash(name)) {
    case "kin"_hash:
      if (name == "kind") {
        if constexpr (common::HasMember<T, IntegerTypes>) {
          return Expr<T>{funcRef.arguments()[0]->value()->GetType()->kind};
        } else {
          common::die("kind() result not integral");
        }
      }
      break;
    case "len"_hash:
      if (name == "len") {
        if constexpr (std::is_same_v<T, SubscriptInteger>) {
          if (auto *charExpr{UnwrapExpr<Expr<SomeCharacter>>(
                  *funcRef.arguments()[0]->value())}) {
            return std::visit([](auto &kx) { return kx.LEN(); }, charExpr->u);
          }
        } else {
          common::die("len() result not SubscriptInteger");
        }
      }
      break;
    case "ian"_hash:
      if (name == "iand") {
        if (auto *x{std::get_if<BOZLiteralConstant>(
                &funcRef.arguments()[0]->value->u)}) {
          *funcRef.arguments()[0]->value =
              Fold(context, ConvertToType<T>(std::move(*x)));
        }
        if (auto *x{std::get_if<BOZLiteralConstant>(
                &funcRef.arguments()[1]->value->u)}) {
          *funcRef.arguments()[1]->value =
              Fold(context, ConvertToType<T>(std::move(*x)));
        }
        return FoldElementalIntrinsic<T, T, T>(
            std::move(funcRef), ScalarFunc<T, T, T>(&Scalar<T>::IAND));
      }
      break;
    case "int"_hash:
      if (name == "int") {
        return std::visit(
            [&](auto &&x) -> Expr<T> {
              using From = std::decay_t<decltype(x)>;
              if constexpr (std::is_same_v<From, BOZLiteralConstant> ||
                  std::is_same_v<From, Expr<SomeReal>> ||
                  std::is_same_v<From, Expr<SomeInteger>> ||
                  std::is_same_v<From, Expr<SomeComplex>>) {
                return Fold(context, ConvertToType<T>(std::move(x)));
              } else {
                common::die("int() argument type not valid");
                return Expr<T>{std::move(funcRef)};  // unreachable
              }
            },
            std::move(funcRef.arguments()[0]->value->u));
      }
      break;
    default:
      // TODO: many more intrinsic functions
      break;
    }
  }
  return Expr<T>{std::move(funcRef)};
}

template<int KIND>
Expr<Type<TypeCategory::Real, KIND>> FoldOperation(FoldingContext &context,
    FunctionRef<Type<TypeCategory::Real, KIND>> &&funcRef) {
  using namespace intrinsicHelper;
  using T = Type<TypeCategory::Real, KIND>;
  for (std::optional<ActualArgument> &arg : funcRef.arguments()) {
    if (arg.has_value()) {
      *arg->value = FoldOperation(context, std::move(*arg->value));
    }
  }
  if (auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)}) {
    std::string name{intrinsic->name};
    switch (DynamicHash(name)) {
    case "aco"_hash:
      if (name == "acos") {
        if constexpr (Host::HostTypeExists<T>()) {
          return FoldElementalIntrinsic<T, T>(std::move(funcRef),
              HostFuncWrap<T, T>(HostFuncPointer<T, T>{std::acos}));
        } else {
          context.messages().Say(
              "acos(real(kind=%d)) cannot be folded on host"_en_US, KIND);
        }
      } else if (name == "acosh") {
        if constexpr (Host::HostTypeExists<T>()) {
          return FoldElementalIntrinsic<T, T>(std::move(funcRef),
              HostFuncWrap<T, T>(HostFuncPointer<T, T>{std::acosh}));
        } else {
          context.messages().Say(
              "acosh(real(kind=%d)) cannot be folded on host"_en_US, KIND);
        }
      }
    case "bes"_hash:
      if (name == "bessel_jn" || name == "bessel_yn") {
        if (funcRef.arguments().size() == 2) {  // elemental
          if constexpr (Host::HostTypeExists<T>()) {
            // TODO mapping to <cmath> function to be tested.<cmath> func takes
            // real arg for n
            if (auto *n{std::get_if<Expr<SomeInteger>>(
                    &funcRef.arguments()[0]->value->u)}) {
              *funcRef.arguments()[0]->value =
                  Fold(context, ConvertToType<T>(std::move(*n)));
            }
            auto hostFunc{name == "bessel_jn"
                    ? HostFuncPointer<T, T, T>{std::cyl_bessel_j}
                    : HostFuncPointer<T, T, T>{std::cyl_neumann}};
            return FoldElementalIntrinsic<T, T, T>(
                std::move(funcRef), HostFuncWrap<T, T, T>(hostFunc));
          }
        }
      }
      break;
    case "dpr"_hash:
      if (name == "dprod") {
        if (auto *x{std::get_if<Expr<SomeReal>>(
                &funcRef.arguments()[0]->value->u)}) {
          if (auto *y{std::get_if<Expr<SomeReal>>(
                  &funcRef.arguments()[1]->value->u)}) {
            return Fold(context,
                Expr<T>{Multiply<T>{ConvertToType<T>(std::move(*x)),
                    ConvertToType<T>(std::move(*y))}});
          }
        }
        common::die("Wrong argument type in dprod()");
        break;
      }
      break;
    case "rea"_hash:
      if (name == "real") {
        return std::visit(
            [&](auto &&x) -> Expr<T> {
              using From = std::decay_t<decltype(x)>;
              if constexpr (std::is_same_v<From, BOZLiteralConstant>) {
                typename T::Scalar::Word::ValueWithOverflow result{
                    T::Scalar::Word::ConvertUnsigned(x)};
                if (result.overflow) {  // C1601
                  context.messages().Say(
                      "Non null truncated bits of boz literal constant in REAL intrinsic"_en_US);
                }
                return Expr<T>{Constant<T>{Scalar<T>(std::move(result.value))}};
              } else if constexpr (std::is_same_v<From, Expr<SomeReal>> ||
                  std::is_same_v<From, Expr<SomeInteger>> ||
                  std::is_same_v<From, Expr<SomeComplex>>) {
                return Fold(context, ConvertToType<T>(std::move(x)));
              } else {
                common::die("real() argument type not valid");
                return Expr<T>{std::move(funcRef)};  // unreachable
              }
            },
            std::move(funcRef.arguments()[0]->value->u));
      }
      break;
    default:
      // TODO: many more intrinsic functions
      break;
    }
  }
  return Expr<T>{std::move(funcRef)};
}

template<int KIND>
Expr<Type<TypeCategory::Complex, KIND>> FoldOperation(FoldingContext &context,
    FunctionRef<Type<TypeCategory::Complex, KIND>> &&funcRef) {
  using namespace intrinsicHelper;
  using T = Type<TypeCategory::Complex, KIND>;
  for (std::optional<ActualArgument> &arg : funcRef.arguments()) {
    if (arg.has_value()) {
      *arg->value = FoldOperation(context, std::move(*arg->value));
    }
  }
  if (auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)}) {
    std::string name{intrinsic->name};
    switch (DynamicHash(name)) {
    case "aco"_hash:
      if (name == "acos") {
        if constexpr (Host::HostTypeExists<T>()) {
          return FoldElementalIntrinsic<T, T>(std::move(funcRef),
              HostFuncWrap<T, T>(HostFuncPointer<T, T>{std::acos}));
        } else {
          context.messages().Say(
              "acos(complex(kind=%d)) cannot be folded on host"_en_US, KIND);
        }
      } else if (name == "acosh") {
        if constexpr (Host::HostTypeExists<T>()) {
          return FoldElementalIntrinsic<T, T>(std::move(funcRef),
              HostFuncWrap<T, T>(HostFuncPointer<T, T>{std::acosh}));
        } else {
          context.messages().Say(
              "acosh(complex(kind=%d)) cannot be folded on host"_en_US, KIND);
        }
      }
    case "cmp"_hash:
      if (name == "cmplx") {
        if (funcRef.arguments().size() == 2) {
          if (auto *x{std::get_if<Expr<SomeComplex>>(
                  &funcRef.arguments()[0]->value->u)}) {
            return Fold(context, ConvertToType<T>(std::move(*x)));
          } else {
            common::die("x must be complex in cmplx(x[, kind])");
          }
        } else {
          CHECK(funcRef.arguments().size() == 3);
          using Part = typename T::Part;
          Expr<SomeType> im{funcRef.arguments()[1].has_value()
                  ? std::move(*funcRef.arguments()[1]->value)
                  : AsGenericExpr(Constant<Part>{Scalar<Part>{}})};
          Expr<SomeType> re{std::move(*funcRef.arguments()[0]->value)};
          int reRank{re.Rank()};
          int imRank{im.Rank()};
          semantics::Attrs attrs;
          attrs.set(semantics::Attr::ELEMENTAL);
          auto reReal{
              FunctionRef<Part>{ProcedureDesignator{SpecificIntrinsic{
                                    "real", Part::GetType(), reRank, attrs}},
                  ActualArguments{ActualArgument{std::move(re)}}}};
          auto imReal{
              FunctionRef<Part>{ProcedureDesignator{SpecificIntrinsic{
                                    "real", Part::GetType(), imRank, attrs}},
                  ActualArguments{ActualArgument{std::move(im)}}}};
          return Fold(context,
              Expr<T>{ComplexConstructor<T::kind>{Expr<Part>{std::move(reReal)},
                  Expr<Part>{std::move(imReal)}}});
        }
      }
      break;
    default:
      // TODO: many more intrinsic functions
      break;
    }
  }
  return Expr<T>{std::move(funcRef)};
}

template<int KIND>
Expr<Type<TypeCategory::Logical, KIND>> FoldOperation(FoldingContext &context,
    FunctionRef<Type<TypeCategory::Logical, KIND>> &&funcRef) {
  using namespace intrinsicHelper;
  using T = Type<TypeCategory::Logical, KIND>;
  for (std::optional<ActualArgument> &arg : funcRef.arguments()) {
    if (arg.has_value()) {
      *arg->value = FoldOperation(context, std::move(*arg->value));
    }
  }
  if (auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)}) {
    std::string name{intrinsic->name};
    switch (DynamicHash(name)) {
    case "bge"_hash:
      if (name == "bge") {
        using LargestInt = Type<TypeCategory::Integer, 16>;
        static_assert(std::is_same_v<Scalar<LargestInt>, BOZLiteralConstant>);
        if (auto *x{std::get_if<Expr<SomeInteger>>(
                &funcRef.arguments()[0]->value->u)}) {
          *funcRef.arguments()[0]->value =
              Fold(context, ConvertToType<LargestInt>(std::move(*x)));
        } else if (auto *x{std::get_if<BOZLiteralConstant>(
                       &funcRef.arguments()[0]->value->u)}) {
          *funcRef.arguments()[0]->value =
              AsGenericExpr(Constant<LargestInt>{std::move(*x)});
        }
        if (auto *x{std::get_if<Expr<SomeInteger>>(
                &funcRef.arguments()[1]->value->u)}) {
          *funcRef.arguments()[1]->value =
              Fold(context, ConvertToType<LargestInt>(std::move(*x)));
        } else if (auto *x{std::get_if<BOZLiteralConstant>(
                       &funcRef.arguments()[1]->value->u)}) {
          *funcRef.arguments()[1]->value =
              AsGenericExpr(Constant<LargestInt>{std::move(*x)});
        }
        return FoldElementalIntrinsic<T, LargestInt, LargestInt>(
            std::move(funcRef),
            ScalarFunc<T, LargestInt, LargestInt>(
                [](const Scalar<LargestInt> &i, const Scalar<LargestInt> &j) {
                  return Scalar<T>{i.BGE(j)};
                }));
      }
      break;
    default:
      // TODO: many more intrinsic functions
      break;
    }
  }
  return Expr<T>{std::move(funcRef)};
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
          [&](const Symbol *symbol) { return Expr<T>{std::move(designator)}; },
          [&](auto &&x) {
            return Expr<T>{Designator<T>{FoldOperation(context, std::move(x))}};
          },
      },
      std::move(designator.u));
}

// Array constructor folding

Expr<ImpliedDoIndex::Result> FoldOperation(
    FoldingContext &context, ImpliedDoIndex &&iDo) {
  if (std::optional<std::int64_t> value{context.GetImpliedDo(iDo.name)}) {
    return Expr<ImpliedDoIndex::Result>{*value};
  } else {
    return Expr<ImpliedDoIndex::Result>{std::move(iDo)};
  }
}

template<typename T> class ArrayConstructorFolder {
public:
  explicit ArrayConstructorFolder(const FoldingContext &c) : context_{c} {}

  Expr<T> FoldArray(ArrayConstructor<T> &&array) {
    if (FoldArray(array)) {
      auto n{static_cast<std::int64_t>(elements_.size())};
      if constexpr (std::is_same_v<T, SomeDerived>) {
        return Expr<T>{Constant<T>{array.derivedTypeSpec(),
            std::move(elements_), std::vector<std::int64_t>{n}}};
      } else if constexpr (T::category == TypeCategory::Character) {
        auto length{Fold(context_, common::Clone(array.LEN()))};
        if (std::optional<std::int64_t> lengthValue{ToInt64(length)}) {
          return Expr<T>{Constant<T>{*lengthValue, std::move(elements_),
              std::vector<std::int64_t>{n}}};
        }
      } else {
        return Expr<T>{
            Constant<T>{std::move(elements_), std::vector<std::int64_t>{n}}};
      }
    }
    return Expr<T>{std::move(array)};
  }

private:
  bool FoldArray(const common::CopyableIndirection<Expr<T>> &expr) {
    Expr<T> folded{Fold(context_, common::Clone(expr.value()))};
    if (auto *c{UnwrapExpr<Constant<T>>(folded)}) {
      // Copy elements in Fortran array element order
      std::vector<std::int64_t> shape{c->shape()};
      int rank{c->Rank()};
      std::vector<std::int64_t> index(shape.size(), 1);
      for (std::size_t n{c->size()}; n-- > 0;) {
        if constexpr (std::is_same_v<T, SomeDerived>) {
          elements_.emplace_back(c->derivedTypeSpec(), c->At(index));
        } else {
          elements_.emplace_back(c->At(index));
        }
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
    std::optional<std::int64_t> start{ToInt64(lower)}, end{ToInt64(upper)},
        step{ToInt64(stride)};
    if (start.has_value() && end.has_value() && step.has_value()) {
      bool result{true};
      for (std::int64_t &j{context_.StartImpliedDo(iDo.name(), *start)};
           j <= *end; j += *step) {
        result &= FoldArray(iDo.values());
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
    for (const auto &x : xs.values()) {
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
  ArrayConstructorFolder<T> folder{context};
  Expr<T> result{folder.FoldArray(std::move(array))};
  return result;
}

Expr<SomeDerived> FoldOperation(
    FoldingContext &context, StructureConstructor &&structure) {
  StructureConstructor result{structure.derivedTypeSpec()};
  for (auto &&[symbol, value] : std::move(structure.values())) {
    result.Add(*symbol, Fold(context, std::move(value.value())));
  }
  return Expr<SomeDerived>{Constant<SomeDerived>{result}};
}

// Substitute a bare type parameter reference with its value if it has one now
template<int KIND>
Expr<Type<TypeCategory::Integer, KIND>> FoldOperation(
    FoldingContext &context, TypeParamInquiry<KIND> &&inquiry) {
  using IntKIND = Type<TypeCategory::Integer, KIND>;
  if (Component * component{common::Unwrap<Component>(inquiry.base())}) {
    return Expr<IntKIND>{TypeParamInquiry<KIND>{
        FoldOperation(context, std::move(*component)), inquiry.parameter()}};
  }
  if (context.pdtInstance() != nullptr &&
      std::get<const Symbol *>(inquiry.base()) == nullptr) {
    // "bare" type parameter: replace with actual value
    const semantics::Scope *scope{context.pdtInstance()->scope()};
    CHECK(scope != nullptr);
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
    } else if (const auto *value{context.pdtInstance()->FindParameter(
                   inquiry.parameter().name())}) {
      // Parameter of a parent derived type; these are saved in the spec.
      if (value->isExplicit()) {
        return Fold(context,
            Expr<IntKIND>{Convert<IntKIND, TypeCategory::Integer>(
                Expr<SomeInteger>{value->GetExplicit().value()})});
      }
    }
  }
  return Expr<IntKIND>{std::move(inquiry)};
}

// Unary operations

template<typename TO, typename FROM> std::optional<TO> ConvertString(FROM &&s) {
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
  return std::visit(
      [&](auto &kindExpr) -> Expr<TO> {
        kindExpr = Fold(context, std::move(kindExpr));
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
  return Expr<T>{std::move(x)};
}

template<typename T>
Expr<T> FoldOperation(FoldingContext &context, Negate<T> &&x) {
  auto &operand{x.left()};
  operand = Fold(context, std::move(operand));
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
  using Part = Type<TypeCategory::Real, KIND>;
  auto &operand{x.left()};
  operand = Fold(context, std::move(operand));
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
  using Ty = Type<TypeCategory::Logical, KIND>;
  auto &operand{x.left()};
  operand = Fold(context, std::move(operand));
  if (auto value{GetScalarConstantValue<Ty>(operand)}) {
    return Expr<Ty>{Constant<Ty>{!value->IsTrue()}};
  }
  return Expr<Ty>{x};
}

// Binary (dyadic) operations

template<typename T1, typename T2>
std::optional<std::pair<Scalar<T1>, Scalar<T2>>> FoldOperands(
    FoldingContext &context, Expr<T1> &x, Expr<T2> &y) {
  x = Fold(context, std::move(x));  // use of std::move() on &x is intentional
  y = Fold(context, std::move(y));
  if (auto xvalue{GetScalarConstantValue<T1>(x)}) {
    if (auto yvalue{GetScalarConstantValue<T2>(y)}) {
      return {std::make_pair(*xvalue, *yvalue)};
    }
  }
  return std::nullopt;
}

template<typename T>
Expr<T> FoldOperation(FoldingContext &context, Add<T> &&x) {
  if (auto folded{FoldOperands(context, x.left(), x.right())}) {
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
  if (auto folded{FoldOperands(context, x.left(), x.right())}) {
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
  if (auto folded{FoldOperands(context, x.left(), x.right())}) {
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
  if (auto folded{FoldOperands(context, x.left(), x.right())}) {
    if constexpr (T::category == TypeCategory::Integer) {
      auto quotAndRem{folded->first.DivideSigned(folded->second)};
      if (quotAndRem.divisionByZero) {
        context.messages().Say("INTEGER(%d) division by zero"_en_US, T::kind);
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
  if (auto folded{FoldOperands(context, x.left(), x.right())}) {
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
  return std::visit(
      [&](auto &y) -> Expr<T> {
        if (auto folded{FoldOperands(context, x.left(), y)}) {
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
  if (auto folded{FoldOperands(context, x.left(), x.right())}) {
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
  using Result = Type<TypeCategory::Complex, KIND>;
  if (auto folded{FoldOperands(context, x.left(), x.right())}) {
    return Expr<Result>{
        Constant<Result>{Scalar<Result>{folded->first, folded->second}}};
  }
  return Expr<Result>{std::move(x)};
}

template<int KIND>
Expr<Type<TypeCategory::Character, KIND>> FoldOperation(
    FoldingContext &context, Concat<KIND> &&x) {
  using Result = Type<TypeCategory::Character, KIND>;
  if (auto folded{FoldOperands(context, x.left(), x.right())}) {
    return Expr<Result>{Constant<Result>{folded->first + folded->second}};
  }
  return Expr<Result>{std::move(x)};
}

template<int KIND>
Expr<Type<TypeCategory::Character, KIND>> FoldOperation(
    FoldingContext &context, SetLength<KIND> &&x) {
  using Result = Type<TypeCategory::Character, KIND>;
  if (auto folded{FoldOperands(context, x.left(), x.right())}) {
    auto oldLength{static_cast<std::int64_t>(folded->first.size())};
    auto newLength{folded->second.ToInt64()};
    if (newLength < oldLength) {
      folded->first.erase(newLength);
    } else {
      folded->first.append(newLength - oldLength, ' ');
    }
    CHECK(static_cast<std::int64_t>(folded->first.size()) == newLength);
    return Expr<Result>{Constant<Result>{std::move(folded->first)}};
  }
  return Expr<Result>{std::move(x)};
}

template<typename T>
Expr<LogicalResult> FoldOperation(
    FoldingContext &context, Relational<T> &&relation) {
  if (auto folded{FoldOperands(context, relation.left(), relation.right())}) {
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
    FoldingContext &context, LogicalOperation<KIND> &&x) {
  using LOGICAL = Type<TypeCategory::Logical, KIND>;
  if (auto folded{FoldOperands(context, x.left(), x.right())}) {
    bool xt{folded->first.IsTrue()}, yt{folded->second.IsTrue()}, result{};
    switch (x.logicalOperator) {
    case LogicalOperator::And: result = xt && yt; break;
    case LogicalOperator::Or: result = xt || yt; break;
    case LogicalOperator::Eqv: result = xt == yt; break;
    case LogicalOperator::Neqv: result = xt != yt; break;
    }
    return Expr<LOGICAL>{Constant<LOGICAL>{result}};
  }
  return Expr<LOGICAL>{std::move(x)};
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
        } else {
          using Ty = std::decay_t<decltype(x)>;
          if constexpr (std::is_same_v<Ty, BOZLiteralConstant> ||
              std::is_same_v<Ty, NullPointer>) {
            return std::move(expr);
          } else {
            return Expr<T>{Fold(context, std::move(x))};
          }
        }
      },
      std::move(expr.u));
}

FOR_EACH_TYPE_AND_KIND(template class ExpressionBase)

// Constant expression predicate IsConstantExpr().
// This code determines whether an expression is a "constant expression"
// in the sense of section 10.1.12.  This is not the same thing as being
// able to fold it (yet) into a known constant value; specifically,
// the expression may reference derived type kind parameters whose values
// are not yet known.

class IsConstantExprVisitor : public virtual VisitorBase<bool> {
public:
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

private:
  void Check(bool ok) {
    if (!ok) {
      Return(false);
    }
  }
};

bool IsConstantExpr(const Expr<SomeType> &expr) {
  return Visitor<bool, IsConstantExprVisitor>{0}.Traverse(expr);
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
}
