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
#include "expression.h"
#include "int-power.h"
#include "tools.h"
#include "type.h"
#include "../common/indirection.h"
#include "../common/template.h"
#include "../common/unwrap.h"
#include "../parser/message.h"
#include "../semantics/scope.h"
#include "../semantics/symbol.h"
#include <cstdio>
#include <optional>
#include <set>
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
template<typename T> Expr<T> FoldOperation(FoldingContext &, FunctionRef<T> &&);
template<typename T> Expr<T> FoldOperation(FoldingContext &, Designator<T> &&);
template<int KIND>
Expr<Type<TypeCategory::Integer, KIND>> FoldOperation(
    FoldingContext &, TypeParamInquiry<KIND> &&);

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
      Fold(context, Expr<SubscriptInteger>{triplet.stride()})};
}

Subscript FoldOperation(FoldingContext &context, Subscript &&subscript) {
  return std::visit(
      common::visitors{
          [&](IndirectSubscriptIntegerExpr &&expr) {
            *expr = Fold(context, std::move(*expr));
            return Subscript(std::move(expr));
          },
          [&](Triplet &&triplet) {
            return Subscript(FoldOperation(context, std::move(triplet)));
          },
      },
      std::move(subscript.u));
}

ArrayRef FoldOperation(FoldingContext &context, ArrayRef &&arrayRef) {
  for (Subscript &subscript : arrayRef.subscript) {
    subscript = FoldOperation(context, std::move(subscript));
  }
  return std::visit(
      common::visitors{
          [&](const Symbol *symbol) {
            return ArrayRef{*symbol, std::move(arrayRef.subscript)};
          },
          [&](Component &&component) {
            return ArrayRef{FoldOperation(context, std::move(component)),
                std::move(arrayRef.subscript)};
          },
      },
      std::move(arrayRef.u));
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

template<typename T>
Expr<T> FoldOperation(FoldingContext &context, FunctionRef<T> &&funcRef) {
  ActualArguments args{std::move(funcRef.arguments())};
  for (std::optional<ActualArgument> &arg : args) {
    if (arg.has_value()) {
      *arg->value = FoldOperation(context, std::move(*arg->value));
    }
  }
  if (auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)}) {
    std::string name{intrinsic->name};
    if (name == "kind") {
      if constexpr (common::HasMember<T, IntegerTypes>) {
        return Expr<T>{args[0]->value->GetType()->kind};
      } else {
        common::die("kind() result not integral");
      }
    } else if (name == "len") {
      if constexpr (std::is_same_v<T, SubscriptInteger>) {
        if (auto *charExpr{UnwrapExpr<Expr<SomeCharacter>>(*args[0]->value)}) {
          return std::visit([](auto &kx) { return kx.LEN(); }, charExpr->u);
        }
      } else {
        common::die("len() result not SubscriptInteger");
      }
    } else {
      // TODO: many more intrinsic functions
    }
  }
  return Expr<T>{FunctionRef<T>{std::move(funcRef.proc()), std::move(args)}};
}

template<typename T>
Expr<T> FoldOperation(FoldingContext &context, Designator<T> &&designator) {
  if constexpr (T::category == TypeCategory::Character) {
    if (auto *substring{common::Unwrap<Substring>(designator.u)}) {
      if (std::optional<Expr<SomeCharacter>> folded{substring->Fold(context)}) {
        if (const auto *value{GetScalarConstantValue<T>(*folded)}) {
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

// Substitute a bare type parameter reference with its value if it has one now
template<int KIND>
Expr<Type<TypeCategory::Integer, KIND>> FoldOperation(
    FoldingContext &context, TypeParamInquiry<KIND> &&inquiry) {
  using IntKIND = Type<TypeCategory::Integer, KIND>;
  if (Component * component{common::Unwrap<Component>(inquiry.u)}) {
    return Expr<IntKIND>{TypeParamInquiry<KIND>{
        FoldOperation(context, std::move(*component)), *inquiry.parameter}};
  }
  if (context.pdtInstance != nullptr &&
      std::get<const Symbol *>(inquiry.u) == nullptr) {
    // "bare" type parameter: replace with actual value
    const semantics::Scope *scope{context.pdtInstance->scope()};
    CHECK(scope != nullptr);
    auto iter{scope->find(inquiry.parameter->name())};
    if (iter != scope->end()) {
      const Symbol &symbol{*iter->second};
      const auto *details{symbol.detailsIf<semantics::TypeParamDetails>()};
      CHECK(details != nullptr);
      CHECK(details->init().has_value());
      Expr<SomeInteger> expr{*details->init()};
      return Fold(context,
          Expr<IntKIND>{
              Convert<IntKIND, TypeCategory::Integer>(std::move(expr))});
    } else {
      // Parameter of a parent derived type; these are saved in the spec.
      const auto *value{
          context.pdtInstance->FindParameter(inquiry.parameter->name())};
      CHECK(value != nullptr);
      CHECK(value->isExplicit());
      return Fold(context,
          Expr<IntKIND>{Convert<IntKIND, TypeCategory::Integer>(
              value->GetExplicit().value())});
    }
  }
  return Expr<IntKIND>{std::move(inquiry)};
}

// Unary operations

template<typename TO, TypeCategory FROMCAT>
Expr<TO> FoldOperation(
    FoldingContext &context, Convert<TO, FROMCAT> &&convert) {
  return std::visit(
      [&](auto &kindExpr) -> Expr<TO> {
        kindExpr = Fold(context, std::move(kindExpr));
        using Operand = ResultType<decltype(kindExpr)>;
        char buffer[64];
        if (const auto *value{GetScalarConstantValue<Operand>(kindExpr)}) {
          if constexpr (TO::category == TypeCategory::Integer) {
            if constexpr (Operand::category == TypeCategory::Integer) {
              auto converted{Scalar<TO>::ConvertSigned(*value)};
              if (converted.overflow) {
                context.messages.Say(
                    "INTEGER(%d) to INTEGER(%d) conversion overflowed"_en_US,
                    Operand::kind, TO::kind);
              }
              return Expr<TO>{Constant<TO>{std::move(converted.value)}};
            } else if constexpr (Operand::category == TypeCategory::Real) {
              auto converted{value->template ToInteger<Scalar<TO>>()};
              if (converted.flags.test(RealFlag::InvalidArgument)) {
                context.messages.Say(
                    "REAL(%d) to INTEGER(%d) conversion: invalid argument"_en_US,
                    Operand::kind, TO::kind);
              } else if (converted.flags.test(RealFlag::Overflow)) {
                context.messages.Say(
                    "REAL(%d) to INTEGER(%d) conversion overflowed"_en_US,
                    Operand::kind, TO::kind);
              }
              return Expr<TO>{Constant<TO>{std::move(converted.value)}};
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
              return Expr<TO>{Constant<TO>{std::move(converted.value)}};
            } else if constexpr (Operand::category == TypeCategory::Real) {
              auto converted{Scalar<TO>::Convert(*value)};
              if (!converted.flags.empty()) {
                std::snprintf(buffer, sizeof buffer,
                    "REAL(%d) to REAL(%d) conversion", Operand::kind, TO::kind);
                RealFlagWarnings(context, converted.flags, buffer);
              }
              if (context.flushSubnormalsToZero) {
                converted.value = converted.value.FlushSubnormalToZero();
              }
              return Expr<TO>{Constant<TO>{std::move(converted.value)}};
            }
          } else if constexpr (TO::category == TypeCategory::Logical &&
              Operand::category == TypeCategory::Logical) {
            return Expr<TO>{Constant<TO>{value->IsTrue()}};
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
  if (const auto *value{GetScalarConstantValue<T>(operand)}) {
    // Preserve parentheses, even around constants.
    return Expr<T>{Parentheses<T>{Expr<T>{Constant<T>{*value}}}};
  }
  return Expr<T>{std::move(x)};
}

template<typename T>
Expr<T> FoldOperation(FoldingContext &context, Negate<T> &&x) {
  auto &operand{x.left()};
  operand = Fold(context, std::move(operand));
  if (const auto *value{GetScalarConstantValue<T>(operand)}) {
    if constexpr (T::category == TypeCategory::Integer) {
      auto negated{value->Negate()};
      if (negated.overflow) {
        context.messages.Say("INTEGER(%d) negation overflowed"_en_US, T::kind);
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
  if (const auto *value{GetScalarConstantValue<Operand>(operand)}) {
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
  if (const auto *value{GetScalarConstantValue<Ty>(operand)}) {
    return Expr<Ty>{Constant<Ty>{value->IsTrue()}};
  }
  return Expr<Ty>{x};
}

// Binary (dyadic) operations

template<typename T1, typename T2>
std::optional<std::pair<Scalar<T1>, Scalar<T2>>> FoldOperands(
    FoldingContext &context, Expr<T1> &x, Expr<T2> &y) {
  x = Fold(context, std::move(x));  // use of std::move() on &x is intentional
  y = Fold(context, std::move(y));
  if (const auto *xvalue{GetScalarConstantValue<T1>(x)}) {
    if (const auto *yvalue{GetScalarConstantValue<T2>(y)}) {
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
        context.messages.Say("INTEGER(%d) addition overflowed"_en_US, T::kind);
      }
      return Expr<T>{Constant<T>{sum.value}};
    } else {
      auto sum{folded->first.Add(folded->second, context.rounding)};
      RealFlagWarnings(context, sum.flags, "addition");
      if (context.flushSubnormalsToZero) {
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
        context.messages.Say(
            "INTEGER(%d) subtraction overflowed"_en_US, T::kind);
      }
      return Expr<T>{Constant<T>{difference.value}};
    } else {
      auto difference{folded->first.Subtract(folded->second, context.rounding)};
      RealFlagWarnings(context, difference.flags, "subtraction");
      if (context.flushSubnormalsToZero) {
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
        context.messages.Say(
            "INTEGER(%d) multiplication overflowed"_en_US, T::kind);
      }
      return Expr<T>{Constant<T>{product.lower}};
    } else {
      auto product{folded->first.Multiply(folded->second, context.rounding)};
      RealFlagWarnings(context, product.flags, "multiplication");
      if (context.flushSubnormalsToZero) {
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
        context.messages.Say("INTEGER(%d) division by zero"_en_US, T::kind);
      }
      if (quotAndRem.overflow) {
        context.messages.Say("INTEGER(%d) division overflowed"_en_US, T::kind);
      }
      return Expr<T>{Constant<T>{quotAndRem.quotient}};
    } else {
      auto quotient{folded->first.Divide(folded->second, context.rounding)};
      RealFlagWarnings(context, quotient.flags, "division");
      if (context.flushSubnormalsToZero) {
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
        context.messages.Say(
            "INTEGER(%d) zero to negative power"_en_US, T::kind);
      } else if (power.overflow) {
        context.messages.Say("INTEGER(%d) power overflowed"_en_US, T::kind);
      } else if (power.zeroToZero) {
        context.messages.Say("INTEGER(%d) 0**0 is not defined"_en_US, T::kind);
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
          if (context.flushSubnormalsToZero) {
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
        } else if constexpr (std::is_same_v<BOZLiteralConstant,
                                 std::decay_t<decltype(x)>>) {
          return std::move(expr);
        } else {
          return Expr<T>{Fold(context, std::move(x))};
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
//
// The implementation uses mutually recursive helper function overloadings and
// templates.

struct ConstExprContext {
  std::set<parser::CharBlock> constantNames;
};

// Base cases
bool IsConstExpr(ConstExprContext &, const BOZLiteralConstant &) {
  return true;
}
template<typename A> bool IsConstExpr(ConstExprContext &, const Constant<A> &) {
  return true;
}
bool IsConstExpr(ConstExprContext &, const StaticDataObject::Pointer) {
  return true;
}
template<int KIND>
bool IsConstExpr(ConstExprContext &, const TypeParamInquiry<KIND> &inquiry) {
  return inquiry.parameter->template get<semantics::TypeParamDetails>()
             .attr() == common::TypeParamAttr::Kind;
}
bool IsConstExpr(ConstExprContext &, const Symbol *symbol) {
  return symbol->attrs().test(semantics::Attr::PARAMETER);
}
bool IsConstExpr(ConstExprContext &, const CoarrayRef &) { return false; }
bool IsConstExpr(ConstExprContext &, const ImpliedDoIndex &) {
  return true;  // only tested when bounds are constant
}

// Prototypes for mutual recursion
template<typename D, typename R, typename O1>
bool IsConstExpr(ConstExprContext &, const Operation<D, R, O1> &);
template<typename D, typename R, typename O1, typename O2>
bool IsConstExpr(ConstExprContext &, const Operation<D, R, O1, O2> &);
template<typename V> bool IsConstExpr(ConstExprContext &, const ImpliedDo<V> &);
template<typename A>
bool IsConstExpr(ConstExprContext &, const ArrayConstructorValue<A> &);
template<typename A>
bool IsConstExpr(ConstExprContext &, const ArrayConstructorValues<A> &);
template<typename A>
bool IsConstExpr(ConstExprContext &, const ArrayConstructor<A> &);
bool IsConstExpr(ConstExprContext &, const BaseObject &);
bool IsConstExpr(ConstExprContext &, const Component &);
bool IsConstExpr(ConstExprContext &, const Triplet &);
bool IsConstExpr(ConstExprContext &, const Subscript &);
bool IsConstExpr(ConstExprContext &, const ArrayRef &);
bool IsConstExpr(ConstExprContext &, const DataRef &);
bool IsConstExpr(ConstExprContext &, const Substring &);
bool IsConstExpr(ConstExprContext &, const ComplexPart &);
template<typename A>
bool IsConstExpr(ConstExprContext &, const Designator<A> &);
bool IsConstExpr(ConstExprContext &, const ActualArgument &);
template<typename A>
bool IsConstExpr(ConstExprContext &, const FunctionRef<A> &);
template<typename A> bool IsConstExpr(ConstExprContext &, const Expr<A> &);
template<typename A>
bool IsConstExpr(ConstExprContext &, const CopyableIndirection<A> &);
template<typename A>
bool IsConstExpr(ConstExprContext &, const std::optional<A> &);
template<typename A>
bool IsConstExpr(ConstExprContext &, const std::vector<A> &);
template<typename... As>
bool IsConstExpr(ConstExprContext &, const std::variant<As...> &);
bool IsConstExpr(ConstExprContext &, const Relational<SomeType> &);

template<typename D, typename R, typename O1>
bool IsConstExpr(
    ConstExprContext &context, const Operation<D, R, O1> &operation) {
  return IsConstExpr(context, operation.left());
}
template<typename D, typename R, typename O1, typename O2>
bool IsConstExpr(
    ConstExprContext &context, const Operation<D, R, O1, O2> &operation) {
  return IsConstExpr(context, operation.left()) &&
      IsConstExpr(context, operation.right());
}
template<typename V>
bool IsConstExpr(ConstExprContext &context, const ImpliedDo<V> &impliedDo) {
  if (!IsConstExpr(context, impliedDo.lower) ||
      !IsConstExpr(context, impliedDo.upper) ||
      !IsConstExpr(context, impliedDo.stride)) {
    return false;
  }
  ConstExprContext newContext{context};
  newContext.constantNames.insert(impliedDo.controlVariableName);
  return IsConstExpr(newContext, impliedDo.values);
}
template<typename A>
bool IsConstExpr(
    ConstExprContext &context, const ArrayConstructorValue<A> &value) {
  return IsConstExpr(context, value.u);
}
template<typename A>
bool IsConstExpr(
    ConstExprContext &context, const ArrayConstructorValues<A> &values) {
  return IsConstExpr(context, values.values);
}
template<typename A>
bool IsConstExpr(ConstExprContext &context, const ArrayConstructor<A> &array) {
  return IsConstExpr(context, array.values);
}
bool IsConstExpr(ConstExprContext &context, const BaseObject &base) {
  return IsConstExpr(context, base.u);
}
bool IsConstExpr(ConstExprContext &context, const Component &component) {
  return IsConstExpr(context, component.base());
}
bool IsConstExpr(ConstExprContext &context, const Triplet &triplet) {
  return IsConstExpr(context, triplet.lower()) &&
      IsConstExpr(context, triplet.upper()) &&
      IsConstExpr(context, triplet.stride());
}
bool IsConstExpr(ConstExprContext &context, const Subscript &subscript) {
  return IsConstExpr(context, subscript.u);
}
bool IsConstExpr(ConstExprContext &context, const ArrayRef &arrayRef) {
  return IsConstExpr(context, arrayRef.u) &&
      IsConstExpr(context, arrayRef.subscript);
}
bool IsConstExpr(ConstExprContext &context, const DataRef &dataRef) {
  return IsConstExpr(context, dataRef.u);
}
bool IsConstExpr(ConstExprContext &context, const Substring &substring) {
  if (const auto *dataRef{substring.GetParentIf<DataRef>()}) {
    if (!IsConstExpr(context, *dataRef)) {
      return false;
    }
  }
  return IsConstExpr(context, substring.lower()) &&
      IsConstExpr(context, substring.upper());
}
bool IsConstExpr(ConstExprContext &context, const ComplexPart &complexPart) {
  return IsConstExpr(context, complexPart.complex());
}
template<typename A>
bool IsConstExpr(ConstExprContext &context, const Designator<A> &designator) {
  return IsConstExpr(context, designator.u);
}
bool IsConstExpr(ConstExprContext &context, const ActualArgument &arg) {
  return IsConstExpr(context, *arg.value);
}
template<typename A>
bool IsConstExpr(ConstExprContext &context, const FunctionRef<A> &funcRef) {
  if (const auto *intrinsic{
          std::get_if<SpecificIntrinsic>(&funcRef.proc().u)}) {
    if (intrinsic->name == "kind") {
      return true;
    }
    // TODO: This is a placeholder with obvious false positives
    return IsConstExpr(context, funcRef.arguments());
  }
  return false;
}
template<typename A>
bool IsConstExpr(ConstExprContext &context, const Expr<A> &expr) {
  return IsConstExpr(context, expr.u);
}
template<typename A>
bool IsConstExpr(ConstExprContext &context, const CopyableIndirection<A> &x) {
  return IsConstExpr(context, *x);
}
template<typename A>
bool IsConstExpr(ConstExprContext &context, const std::optional<A> &maybe) {
  return !maybe.has_value() || IsConstExpr(context, *maybe);
}
template<typename A>
bool IsConstExpr(ConstExprContext &context, const std::vector<A> &v) {
  for (const auto &x : v) {
    if (!IsConstExpr(context, x)) {
      return false;
    }
  }
  return true;
}
template<typename... As>
bool IsConstExpr(ConstExprContext &context, const std::variant<As...> &u) {
  return std::visit([&](const auto &x) { return IsConstExpr(context, x); }, u);
}
bool IsConstExpr(ConstExprContext &context, const Relational<SomeType> &rel) {
  return IsConstExpr(context, rel.u);
}

bool IsConstantExpr(const Expr<SomeType> &expr) {
  ConstExprContext context;
  return IsConstExpr(context, expr);
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
