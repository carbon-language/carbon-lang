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

#ifndef FORTRAN_EVALUATE_FOLD_H_
#define FORTRAN_EVALUATE_FOLD_H_

#include "common.h"
#include "expression.h"
#include "int-power.h"
#include "tools.h"
#include "type.h"
#include "../common/indirection.h"
#include "../parser/message.h"
#include <cstdio>
#include <optional>
#include <type_traits>
#include <variant>

namespace Fortran::evaluate {

using namespace Fortran::parser::literals;

// The result of Fold() is always packaged as an Expr<>.
// This allows Fold() to replace an operation with a constant or
// a canonicalized expression.
// When the operand is an Expr<A>, the result has the same type.

// Base cases
template<typename A> Expr<ResultType<A>> Fold(FoldingContext &, A &&x) {
  return Expr<ResultType<A>>{std::move(x)};
}

template<typename A> Expr<A> Fold(FoldingContext &context, Expr<A> &&expr) {
  static_assert(A::isSpecificIntrinsicType);
  return std::visit([&](auto &&x) -> Expr<A> { return Fold(context, std::move(x)); }, std::move(expr.u));
}

template<TypeCategory CAT>
Expr<SomeKind<CAT>>
Fold(FoldingContext &context, Expr<SomeKind<CAT>> &&expr) {
  return std::visit([&](auto &&x) -> Expr<SomeKind<CAT>> {
    if constexpr (CAT == TypeCategory::Derived) {
      return Fold(context, std::move(x));
    } else {
      return Expr<SomeKind<CAT>>{Fold(context, std::move(x))};
    }
  }, std::move(expr.u));
}

template<> inline Expr<SomeType> Fold(FoldingContext &context, Expr<SomeType> &&expr) {
  return std::visit([&](auto &&x) -> Expr<SomeType> {
    if constexpr (std::is_same_v<std::decay_t<decltype(x)>, BOZLiteralConstant>) {
      return std::move(expr);
    } else {
      return Expr<SomeType>{Fold(context, std::move(x))};
    }
  }, std::move(expr.u));
}

// Unary operations

template<typename TO, TypeCategory FROMCAT> Expr<TO> Fold(FoldingContext &context, Convert<TO, FROMCAT> &&convert) {
  return std::visit([&](auto &kindExpr) -> Expr<TO> {
    kindExpr = Fold(context, std::move(kindExpr));
    using Operand = ResultType<decltype(kindExpr)>;
    if (const auto *c{std::get_if<Constant<Operand>>(&kindExpr.u)}) {
      if constexpr (TO::category == TypeCategory::Integer) {
        if constexpr (Operand::category == TypeCategory::Integer) {
          auto converted{Scalar<TO>::ConvertSigned(c->value)};
          if (converted.overflow) {
            context.messages.Say("INTEGER(%d) to INTEGER(%d) conversion overflowed"_en_US, Operand::kind, TO::kind);
          }
          return Expr<TO>{Constant<TO>{std::move(converted.value)}};
        } else if constexpr (Operand::category == TypeCategory::Real) {
          auto converted{c->value.template ToInteger<Scalar<TO>>()};
          if (converted.flags.test(RealFlag::InvalidArgument)) {
            context.messages.Say(
                "REAL(%d) to INTEGER(%d) conversion: invalid argument"_en_US, Operand::kind, TO::kind);
          } else if (converted.flags.test(RealFlag::Overflow)) {
            context.messages.Say(
                "REAL(%d) to INTEGER(%d) conversion overflowed"_en_US, Operand::kind, TO::kind);
          }
          return Expr<TO>{Constant<TO>{std::move(converted.value)}};
        }
      } else if constexpr (TO::category == TypeCategory::Real) {
        if constexpr (Operand::category == TypeCategory::Integer) {
          auto converted{Scalar<TO>::FromInteger(c->value)};
          if (!converted.flags.empty()) {
            char buffer[64];
            std::snprintf(buffer, sizeof buffer, "INTEGER(%d) to REAL(%d) conversion", Operand::kind, TO::kind);
            RealFlagWarnings(context, converted.flags, buffer);
          }
          return Expr<TO>{Constant<TO>{std::move(converted.value)}};
        } else if constexpr (Operand::category == TypeCategory::Real) {
          auto converted{Scalar<TO>::Convert(c->value)};
          if (!converted.flags.empty()) {
            char buffer[64];
            std::snprintf(buffer, sizeof buffer, "REAL(%d) to REAL(%d) conversion", Operand::kind, TO::kind);
            RealFlagWarnings(context, converted.flags, buffer);
          }
          return Expr<TO>{Constant<TO>{std::move(converted.value)}};
        }
      } else if constexpr (TO::category == TypeCategory::Logical &&
                           Operand::category == TypeCategory::Logical) {
        return Expr<TO>{Constant<TO>{c->value.IsTrue()}};
      }
    }
    return Expr<TO>{std::move(convert)};
  }, convert.left().u);
}

template<typename T> Expr<T> Fold(FoldingContext &context, Negate<T> &&x) {
  auto &operand{x.left()};
  operand = Fold(context, std::move(operand));
  if (const auto *c{std::get_if<Constant<T>>(&operand.u)}) {
    if constexpr (T::category == TypeCategory::Integer) {
      auto negated{c->value.Negate()};
      if (negated.overflow) {
        context.messages.Say("INTEGER(%d) negation overflowed"_en_US, T::kind);
      }
      return Expr<T>{Constant<T>{std::move(negated.value)}};
    } else {
      return Expr<T>{Constant<T>{c->value.Negate()}};  // REAL & COMPLEX negation: no exceptions possible
    }
  }
  return Expr<T>{std::move(x)};
}

template<int KIND> Expr<Type<TypeCategory::Real, KIND>> Fold(FoldingContext &context, ComplexComponent<KIND> &&x) {
  using Operand = Type<TypeCategory::Complex, KIND>;
  using Part = Type<TypeCategory::Real, KIND>;
  auto &operand{x.left()};
  operand = Fold(context, std::move(operand));
  if (const auto *z{std::get_if<Constant<Operand>>(&operand.u)}) {
    if (x.isImaginaryPart) {
      return Expr<Part>{Constant<Part>{z->value.AIMAG()}};
    } else {
      return Expr<Part>{Constant<Part>{z->value.REAL()}};
    }
  }
  return Expr<Part>{std::move(x)};
}

template<int KIND> Expr<Type<TypeCategory::Logical, KIND>> Fold(FoldingContext &context, Not<KIND> &&x) {
  using Ty = Type<TypeCategory::Logical, KIND>;
  auto &operand{x.left()};
  operand = Fold(context, std::move(operand));
  if (const auto *c{std::get_if<Constant<Ty>>(&operand.u)}) {
    return Expr<Ty>{Constant<Ty>{c->value.IsTrue()}};
  }
  return Expr<Ty>{x};
}

// Binary (dyadic) operations

template<typename T1, typename T2> std::optional<std::pair<Scalar<T1>, Scalar<T2>>>
FoldOperands(FoldingContext &context, Expr<T1> &x, Expr<T2> &y) {
  x = Fold(context, std::move(x));
  y = Fold(context, std::move(y));
  if (const auto *xc{std::get_if<Constant<T1>>(&x.u)}) {
    if (const auto *yc{std::get_if<Constant<T2>>(&y.u)}) {
      return {std::make_pair(xc->value, yc->value)};
    }
  }
  return std::nullopt;
}

template<typename T> Expr<T> Fold(FoldingContext &context, Add<T> &&x) {
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
      return Expr<T>{Constant<T>{sum.value}};
    }
  }
  return Expr<T>{std::move(x)};
}

template<typename T> Expr<T> Fold(FoldingContext &context, Subtract<T> &&x) {
  if (auto folded{FoldOperands(context, x.left(), x.right())}) {
    if constexpr (T::category == TypeCategory::Integer) {
      auto difference{folded->first.SubtractSigned(folded->second)};
      if (difference.overflow) {
        context.messages.Say("INTEGER(%d) subtraction overflowed"_en_US, T::kind);
      }
      return Expr<T>{Constant<T>{difference.value}};
    } else {
      auto difference{folded->first.Subtract(folded->second, context.rounding)};
      RealFlagWarnings(context, difference.flags, "subtraction");
      return Expr<T>{Constant<T>{difference.value}};
    }
  }
  return Expr<T>{std::move(x)};
}

template<typename T> Expr<T> Fold(FoldingContext &context, Multiply<T> &&x) {
  if (auto folded{FoldOperands(context, x.left(), x.right())}) {
    if constexpr (T::category == TypeCategory::Integer) {
      auto product{folded->first.MultiplySigned(folded->second)};
      if (product.SignedMultiplicationOverflowed()) {
        context.messages.Say("INTEGER(%d) multiplication overflowed"_en_US, T::kind);
      }
      return Expr<T>{Constant<T>{product.lower}};
    } else {
      auto product{folded->first.Multiply(folded->second, context.rounding)};
      RealFlagWarnings(context, product.flags, "multiplication");
      return Expr<T>{Constant<T>{product.value}};
    }
  }
  return Expr<T>{std::move(x)};
}

template<typename T> Expr<T> Fold(FoldingContext &context, Divide<T> &&x) {
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
      return Expr<T>{Constant<T>{quotient.value}};
    }
  }
  return Expr<T>{std::move(x)};
}

template<typename T> Expr<T> Fold(FoldingContext &context, Power<T> &&x) {
  if (auto folded{FoldOperands(context, x.left(), x.right())}) {
    if constexpr (T::category == TypeCategory::Integer) {
      auto power{folded->first.Power(folded->second)};
      if (power.divisionByZero) {
        context.messages.Say("INTEGER(%d) zero to negative power"_en_US, T::kind);
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

template<typename T> Expr<T> Fold(FoldingContext &context, RealToIntPower<T> &&x) {
  return std::visit([&](auto &y) -> Expr<T> {
    if (auto folded{FoldOperands(context, x.left(), y)}) {
      auto power{evaluate::IntPower(folded->first, folded->second)};
      RealFlagWarnings(context, power.flags, "power with INTEGER exponent");
      return Expr<T>{Constant<T>{power.value}};
    } else {
      return Expr<T>{std::move(x)};
    }
  }, x.right().u);
}

template<typename T> Expr<T> Fold(FoldingContext &context, Extremum<T> &&x) {
  if (auto folded{FoldOperands(context, x.left(), x.right())}) {
    if constexpr (T::category == TypeCategory::Integer) {
      if (folded->first.CompareSigned(folded->second) == x.ordering) {
        return Expr<T>{Constant<T>{folded->first}};
      }
    } else if constexpr (T::category == TypeCategory::Real) {
      if (folded->first.IsNotANumber() || (folded->first.Compare(folded->second) == Relation::Less) == (x.ordering == Ordering::Less)) {
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

}
#endif  // FORTRAN_EVALUATE_FOLD_H_
