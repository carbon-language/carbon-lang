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

#include "fold.h"
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

// no-op base case
template<typename A>
Expr<ResultType<A>> FoldOperation(FoldingContext &, A &&x) {
  return Expr<ResultType<A>>{std::move(x)};
}

// Designators
// At the moment, only substrings fold.
// TODO: Parameters, KIND type parameters
template<int KIND>
Expr<Type<TypeCategory::Character, KIND>> FoldOperation(FoldingContext &context,
    Designator<Type<TypeCategory::Character, KIND>> &&designator) {
  using CHAR = Type<TypeCategory::Character, KIND>;
  if (auto *substring{std::get_if<Substring>(&designator.u)}) {
    if (auto folded{substring->Fold(context)}) {
      if (auto *string{std::get_if<Scalar<CHAR>>(&*folded)}) {
        return Expr<CHAR>{Constant<CHAR>{std::move(*string)}};
      }
      // A zero-length substring of an arbitrary data reference can
      // be folded, but the C++ string type of the empty value will be
      // std::string and that may not be right for multi-byte CHARACTER
      // kinds.
      if (auto length{ToInt64(Fold(context, substring->LEN()))}) {
        if (*length == 0) {
          return Expr<CHAR>{Constant<CHAR>{Scalar<CHAR>{}}};
        }
      }
    }
  }
  return Expr<CHAR>{std::move(designator)};
}

// TODO: Fold/rewrite intrinsic function references

// Unary operations

template<typename TO, TypeCategory FROMCAT>
Expr<TO> FoldOperation(
    FoldingContext &context, Convert<TO, FROMCAT> &&convert) {
  return std::visit(
      [&](auto &kindExpr) -> Expr<TO> {
        kindExpr = Fold(context, std::move(kindExpr));
        using Operand = ResultType<decltype(kindExpr)>;
        char buffer[64];
        if (auto c{GetScalarConstantValue(kindExpr)}) {
          if constexpr (TO::category == TypeCategory::Integer) {
            if constexpr (Operand::category == TypeCategory::Integer) {
              auto converted{Scalar<TO>::ConvertSigned(c->value)};
              if (converted.overflow) {
                context.messages.Say(
                    "INTEGER(%d) to INTEGER(%d) conversion overflowed"_en_US,
                    Operand::kind, TO::kind);
              }
              return Expr<TO>{Constant<TO>{std::move(converted.value)}};
            } else if constexpr (Operand::category == TypeCategory::Real) {
              auto converted{c->value.template ToInteger<Scalar<TO>>()};
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
              auto converted{Scalar<TO>::FromInteger(c->value)};
              if (!converted.flags.empty()) {
                std::snprintf(buffer, sizeof buffer,
                    "INTEGER(%d) to REAL(%d) conversion", Operand::kind,
                    TO::kind);
                RealFlagWarnings(context, converted.flags, buffer);
              }
              return Expr<TO>{Constant<TO>{std::move(converted.value)}};
            } else if constexpr (Operand::category == TypeCategory::Real) {
              auto converted{Scalar<TO>::Convert(c->value)};
              if (!converted.flags.empty()) {
                std::snprintf(buffer, sizeof buffer,
                    "REAL(%d) to REAL(%d) conversion", Operand::kind, TO::kind);
                RealFlagWarnings(context, converted.flags, buffer);
              }
              if (context.flushDenormalsToZero) {
                converted.value = converted.value.FlushDenormalToZero();
              }
              return Expr<TO>{Constant<TO>{std::move(converted.value)}};
            }
          } else if constexpr (TO::category == TypeCategory::Logical &&
              Operand::category == TypeCategory::Logical) {
            return Expr<TO>{Constant<TO>{c->value.IsTrue()}};
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
  if (auto c{GetScalarConstantValue(operand)}) {
    // Preserve parentheses, even around constants.
    return Expr<T>{Parentheses<T>{Expr<T>{Constant<T>{std::move(c->value)}}}};
  }
  return Expr<T>{std::move(x)};
}

template<typename T>
Expr<T> FoldOperation(FoldingContext &context, Negate<T> &&x) {
  auto &operand{x.left()};
  operand = Fold(context, std::move(operand));
  if (auto c{GetScalarConstantValue(operand)}) {
    if constexpr (T::category == TypeCategory::Integer) {
      auto negated{c->value.Negate()};
      if (negated.overflow) {
        context.messages.Say("INTEGER(%d) negation overflowed"_en_US, T::kind);
      }
      return Expr<T>{Constant<T>{std::move(negated.value)}};
    } else {
      // REAL & COMPLEX negation: no exceptions possible
      return Expr<T>{Constant<T>{c->value.Negate()}};
    }
  }
  return Expr<T>{std::move(x)};
}

template<int KIND>
Expr<Type<TypeCategory::Real, KIND>> FoldOperation(
    FoldingContext &context, ComplexComponent<KIND> &&x) {
  using Part = Type<TypeCategory::Real, KIND>;
  auto &operand{x.left()};
  operand = Fold(context, std::move(operand));
  if (auto z{GetScalarConstantValue(operand)}) {
    if (x.isImaginaryPart) {
      return Expr<Part>{Constant<Part>{z->value.AIMAG()}};
    } else {
      return Expr<Part>{Constant<Part>{z->value.REAL()}};
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
  if (auto c{GetScalarConstantValue(operand)}) {
    return Expr<Ty>{Constant<Ty>{c->value.IsTrue()}};
  }
  return Expr<Ty>{x};
}

// Binary (dyadic) operations

template<typename T1, typename T2>
std::optional<std::pair<Scalar<T1>, Scalar<T2>>> FoldOperands(
    FoldingContext &context, Expr<T1> &x, Expr<T2> &y) {
  x = Fold(context, std::move(x));
  y = Fold(context, std::move(y));
  if (auto xc{GetScalarConstantValue(x)}) {
    if (auto yc{GetScalarConstantValue(y)}) {
      return {std::make_pair(xc->value, yc->value)};
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
      if (context.flushDenormalsToZero) {
        sum.value = sum.value.FlushDenormalToZero();
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
      if (context.flushDenormalsToZero) {
        difference.value = difference.value.FlushDenormalToZero();
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
      if (context.flushDenormalsToZero) {
        product.value = product.value.FlushDenormalToZero();
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
      if (context.flushDenormalsToZero) {
        quotient.value = quotient.value.FlushDenormalToZero();
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
          if (context.flushDenormalsToZero) {
            power.value = power.value.FlushDenormalToZero();
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
  using COMPLEX = Type<TypeCategory::Complex, KIND>;
  if (auto folded{FoldOperands(context, x.left(), x.right())}) {
    return Expr<COMPLEX>{
        Constant<COMPLEX>{Scalar<COMPLEX>{folded->first, folded->second}}};
  }
  return Expr<COMPLEX>{std::move(x)};
}

template<int KIND>
Expr<Type<TypeCategory::Character, KIND>> FoldOperation(
    FoldingContext &context, Concat<KIND> &&x) {
  using CHAR = Type<TypeCategory::Character, KIND>;
  if (auto folded{FoldOperands(context, x.left(), x.right())}) {
    return Expr<CHAR>{Constant<CHAR>{folded->first + folded->second}};
  }
  return Expr<CHAR>{std::move(x)};
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

template<>
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
        if constexpr (T::isSpecificIntrinsicType) {
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

template<typename T>
std::optional<Constant<T>>
GetScalarConstantValueHelper<T>::GetScalarConstantValue(const Expr<T> &expr) {
  if (const auto *c{std::get_if<Constant<T>>(&expr.u)}) {
    return {*c};
  } else if (const auto *p{std::get_if<Parentheses<T>>(&expr.u)}) {
    return GetScalarConstantValue(p->left());
  } else {
    return std::nullopt;
  }
}

FOR_EACH_INTRINSIC_KIND(template struct GetScalarConstantValueHelper)
}
