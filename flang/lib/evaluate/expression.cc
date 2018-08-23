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

#include "expression.h"
#include "common.h"
#include "int-power.h"
#include "tools.h"
#include "variable.h"
#include "../common/idioms.h"
#include "../parser/characters.h"
#include "../parser/message.h"
#include <ostream>
#include <string>
#include <type_traits>

using namespace Fortran::parser::literals;

namespace Fortran::evaluate {

// Fold

template<typename D, typename R, typename... O>
auto Operation<D, R, O...>::Fold(FoldingContext &context)
    -> std::optional<Scalar<Result>> {
  auto c0{operand<0>().Fold(context)};
  if constexpr (operands() == 1) {
    if (c0.has_value()) {
      return derived().FoldScalar(context, *c0);
    }
  } else {
    auto c1{operand<1>().Fold(context)};
    if (c0.has_value() && c1.has_value()) {
      return derived().FoldScalar(context, *c0, *c1);
    }
  }
  return std::nullopt;
}

template<int KIND>
auto Expr<Type<TypeCategory::Integer, KIND>>::Fold(FoldingContext &context)
    -> std::optional<Scalar<Result>> {
  if (auto c{ScalarValue()}) {
    return c;
  }
  return std::visit(
      [&](auto &x) -> std::optional<Scalar<Result>> {
        using Ty = std::decay_t<decltype(x)>;
        if constexpr (evaluate::FoldableTrait<Ty>) {
          if (auto c{x.Fold(context)}) {
            if constexpr (std::is_same_v<Ty, Parentheses<Result>>) {
              // Preserve parentheses around constants.
              u_ = Parentheses<Result>{Expr{*c}};
            } else {
              u_ = *c;
            }
            return c;
          }
        }
        return std::nullopt;
      },
      u_);
}

template<int KIND>
auto Expr<Type<TypeCategory::Real, KIND>>::Fold(FoldingContext &context)
    -> std::optional<Scalar<Result>> {
  if (auto c{ScalarValue()}) {
    return c;
  }
  return std::visit(
      [&](auto &x) -> std::optional<Scalar<Result>> {
        using Ty = std::decay_t<decltype(x)>;
        if constexpr (evaluate::FoldableTrait<Ty>) {
          if (auto c{x.Fold(context)}) {
            if (context.flushDenormalsToZero) {
              *c = c->FlushDenormalToZero();
            }
            if constexpr (std::is_same_v<Ty, Parentheses<Result>>) {
              // Preserve parentheses around constants.
              u_ = Parentheses<Result>{Expr{*c}};
            } else {
              u_ = *c;
            }
            return c;
          }
        }
        return std::nullopt;
      },
      u_);
}

template<int KIND>
auto Expr<Type<TypeCategory::Complex, KIND>>::Fold(FoldingContext &context)
    -> std::optional<Scalar<Result>> {
  if (auto c{ScalarValue()}) {
    return c;
  }
  return std::visit(
      [&](auto &x) -> std::optional<Scalar<Result>> {
        using Ty = std::decay_t<decltype(x)>;
        if constexpr (evaluate::FoldableTrait<Ty>) {
          if (auto c{x.Fold(context)}) {
            if (context.flushDenormalsToZero) {
              *c = c->FlushDenormalToZero();
            }
            if constexpr (std::is_same_v<Ty, Parentheses<Result>>) {
              // Preserve parentheses around constants.
              u_ = Parentheses<Result>{Expr{*c}};
            } else {
              u_ = *c;
            }
            return c;
          }
        }
        return std::nullopt;
      },
      u_);
}

template<int KIND>
auto Expr<Type<TypeCategory::Character, KIND>>::Fold(FoldingContext &context)
    -> std::optional<Scalar<Result>> {
  if (auto c{ScalarValue()}) {
    return c;
  }
  return std::visit(
      [&](auto &x) -> std::optional<Scalar<Result>> {
        using Ty = std::decay_t<decltype(x)>;
        if constexpr (evaluate::FoldableTrait<Ty>) {
          if (auto c{x.Fold(context)}) {
            u_ = *c;
            return c;
          }
        }
        return std::nullopt;
      },
      u_);
}

template<int KIND>
auto Expr<Type<TypeCategory::Logical, KIND>>::Fold(FoldingContext &context)
    -> std::optional<Scalar<Result>> {
  if (auto c{ScalarValue()}) {
    return c;
  }
  return std::visit(
      [&](auto &x) -> std::optional<Scalar<Result>> {
        using Ty = std::decay_t<decltype(x)>;
        if constexpr (evaluate::FoldableTrait<Ty>) {
          if (auto c{x.Fold(context)}) {
            Scalar<Result> result{c->IsTrue()};
            u_ = result;
            return {result};
          }
        }
        return std::nullopt;
      },
      u_);
}

template<TypeCategory CAT>
auto Expr<SomeKind<CAT>>::Fold(FoldingContext &context)
    -> std::optional<Scalar<Result>> {
  return std::visit(
      [&](auto &x) -> std::optional<Scalar<Result>> {
        if (auto c{x.Fold(context)}) {
          return {Scalar<Result>{std::move(*c)}};
        }
        return std::nullopt;
      },
      u.u);
}

auto Expr<SomeType>::Fold(FoldingContext &context)
    -> std::optional<Scalar<Result>> {
  return std::visit(
      common::visitors{
          [](BOZLiteralConstant &) -> std::optional<Scalar<Result>> {
            return std::nullopt;
          },
          [&](auto &x) -> std::optional<Scalar<Result>> {
            if (auto c{x.Fold(context)}) {
              return {common::MoveVariant<Scalar<Result>>(std::move(c->u))};
            }
            return std::nullopt;
          }},
      u);
}

// FoldScalar

template<typename TO, typename FROM>
auto Convert<TO, FROM>::FoldScalar(FoldingContext &context,
    const Scalar<Operand> &c) -> std::optional<Scalar<Result>> {
  if constexpr (std::is_same_v<Result, Operand>) {
    return {c};
  } else if constexpr (std::is_same_v<Result, SomeType>) {
    using Generic = SomeKind<Operand::category>;
    if constexpr (std::is_same_v<Operand, Generic>) {
      return {Scalar<Result>{c}};
    } else {
      return {Scalar<Result>{Generic{c}}};
    }
  } else if constexpr (std::is_same_v<Operand, SomeType>) {
    return std::visit(
        [&](const auto &x) -> std::optional<Scalar<Result>> {
          using Ty = std::decay_t<decltype(x)>;
          return Convert<Result, Ty>::FoldScalar(context, x);
        },
        c.u.u);
  } else if constexpr (std::is_same_v<Result, SomeKind<Result::category>>) {
    if constexpr (Result::category == Operand::category) {
      return {Scalar<Result>{c}};
    }
  } else if constexpr (std::is_same_v<Operand, SomeKind<Operand::category>>) {
    return std::visit(
        [&](const auto &x) -> std::optional<Scalar<Result>> {
          using Ty = TypeOf<std::decay_t<decltype(x)>>;
          return Convert<Result, Ty>::FoldScalar(context, x);
        },
        c.u);
  } else if constexpr (Result::category == TypeCategory::Integer) {
    if constexpr (Operand::category == TypeCategory::Integer) {
      auto converted{Scalar<Result>::ConvertSigned(c)};
      if (converted.overflow) {
        context.messages.Say("INTEGER to INTEGER conversion overflowed"_en_US);
      } else {
        return {std::move(converted.value)};
      }
    } else if constexpr (Operand::category == TypeCategory::Real) {
      auto converted{c.template ToInteger<Scalar<Result>>()};
      if (converted.flags.test(RealFlag::InvalidArgument)) {
        context.messages.Say(
            "REAL to INTEGER conversion: invalid argument"_en_US);
      } else if (converted.flags.test(RealFlag::Overflow)) {
        context.messages.Say("REAL to INTEGER conversion overflowed"_en_US);
      } else {
        return {std::move(converted.value)};
      }
    }
  } else if constexpr (Result::category == TypeCategory::Real) {
    if constexpr (Operand::category == TypeCategory::Integer) {
      auto converted{Scalar<Result>::FromInteger(c)};
      RealFlagWarnings(context, converted.flags, "INTEGER to REAL conversion");
      return {std::move(converted.value)};
    } else if constexpr (Operand::category == TypeCategory::Real) {
      auto converted{Scalar<Result>::Convert(c)};
      RealFlagWarnings(context, converted.flags, "REAL to REAL conversion");
      return {std::move(converted.value)};
    }
  }
  return std::nullopt;
}

template<typename A>
auto Negate<A>::FoldScalar(FoldingContext &context, const Scalar<Operand> &c)
    -> std::optional<Scalar<Result>> {
  if constexpr (Result::category == TypeCategory::Integer) {
    auto negated{c.Negate()};
    if (negated.overflow) {
      context.messages.Say("INTEGER negation overflowed"_en_US);
    } else {
      return {std::move(negated.value)};
    }
  } else {
    return {c.Negate()};  // REAL & COMPLEX: no exceptions possible
  }
  return std::nullopt;
}

template<int KIND>
auto ComplexComponent<KIND>::FoldScalar(FoldingContext &context,
    const Scalar<Operand> &z) const -> std::optional<Scalar<Result>> {
  return {isImaginaryPart ? z.AIMAG() : z.REAL()};
}

template<int KIND>
auto Not<KIND>::FoldScalar(FoldingContext &context, const Scalar<Operand> &x)
    -> std::optional<Scalar<Result>> {
  return {Scalar<Result>{!x.IsTrue()}};
}

template<typename A>
auto Add<A>::FoldScalar(FoldingContext &context, const Scalar<Operand> &x,
    const Scalar<Operand> &y) -> std::optional<Scalar<Result>> {
  if constexpr (Result::category == TypeCategory::Integer) {
    auto sum{x.AddSigned(y)};
    if (sum.overflow) {
      context.messages.Say(
          "INTEGER(KIND=%d) addition overflowed"_en_US, Result::kind);
      return std::nullopt;
    }
    return {std::move(sum.value)};
  } else {
    auto sum{x.Add(y, context.rounding)};
    RealFlagWarnings(context, sum.flags, "addition");
    return {std::move(sum.value)};
  }
}

template<typename A>
auto Subtract<A>::FoldScalar(FoldingContext &context, const Scalar<Operand> &x,
    const Scalar<Operand> &y) -> std::optional<Scalar<Result>> {
  if constexpr (Result::category == TypeCategory::Integer) {
    auto diff{x.SubtractSigned(y)};
    if (diff.overflow) {
      context.messages.Say(
          "INTEGER(KIND=%d) subtraction overflowed"_en_US, Result::kind);
      return std::nullopt;
    }
    return {std::move(diff.value)};
  } else {
    auto difference{x.Subtract(y, context.rounding)};
    RealFlagWarnings(context, difference.flags, "subtraction");
    return {std::move(difference.value)};
  }
}

template<typename A>
auto Multiply<A>::FoldScalar(FoldingContext &context, const Scalar<Operand> &x,
    const Scalar<Operand> &y) -> std::optional<Scalar<Result>> {
  if constexpr (Result::category == TypeCategory::Integer) {
    auto product{x.MultiplySigned(y)};
    if (product.SignedMultiplicationOverflowed()) {
      context.messages.Say(
          "INTEGER(KIND=%d) multiplication overflowed"_en_US, Result::kind);
      return std::nullopt;
    }
    return {std::move(product.lower)};
  } else {
    auto product{x.Multiply(y, context.rounding)};
    RealFlagWarnings(context, product.flags, "multiplication");
    return {std::move(product.value)};
  }
}

template<typename A>
auto Divide<A>::FoldScalar(FoldingContext &context, const Scalar<Operand> &x,
    const Scalar<Operand> &y) -> std::optional<Scalar<Result>> {
  if constexpr (Result::category == TypeCategory::Integer) {
    auto qr{x.DivideSigned(y)};
    if (qr.divisionByZero) {
      context.messages.Say("INTEGER division by zero"_en_US);
      return std::nullopt;
    }
    if (qr.overflow) {
      context.messages.Say(
          "INTEGER(KIND=%d) division overflowed"_en_US, Result::kind);
      return std::nullopt;
    }
    return {std::move(qr.quotient)};
  } else {
    auto quotient{x.Divide(y, context.rounding)};
    RealFlagWarnings(context, quotient.flags, "division");
    return {std::move(quotient.value)};
  }
}

template<typename A>
auto Power<A>::FoldScalar(FoldingContext &context, const Scalar<Operand> &x,
    const Scalar<Operand> &y) -> std::optional<Scalar<Result>> {
  if constexpr (Result::category == TypeCategory::Integer) {
    typename Scalar<Result>::PowerWithErrors power{x.Power(y)};
    if (power.divisionByZero) {
      context.messages.Say("zero to negative power"_en_US);
    } else if (power.overflow) {
      context.messages.Say(
          "INTEGER(KIND=%d) power overflowed"_en_US, Result::kind);
    } else if (power.zeroToZero) {
      context.messages.Say("INTEGER 0**0 is not defined"_en_US);
    } else {
      return {std::move(power.power)};
    }
  } else {
    // TODO: real and complex exponentiation to non-integer powers
  }
  return std::nullopt;
}

template<typename A, typename B>
auto RealToIntPower<A, B>::FoldScalar(FoldingContext &context,
    const Scalar<Operand> &x, const Scalar<ExponentOperand> &y)
    -> std::optional<Scalar<Result>> {
  return std::visit(
      [&](const auto &pow) -> std::optional<Scalar<Result>> {
        auto power{evaluate::IntPower(x, pow)};
        RealFlagWarnings(context, power.flags, "raising to INTEGER power");
        return {std::move(power.value)};
      },
      y.u);
}

template<typename A>
auto Extremum<A>::FoldScalar(FoldingContext &context, const Scalar<Operand> &x,
    const Scalar<Operand> &y) const -> std::optional<Scalar<Result>> {
  if constexpr (Operand::category == TypeCategory::Integer) {
    if (ordering == x.CompareSigned(y)) {
      return {x};
    }
  } else if constexpr (Operand::category == TypeCategory::Real) {
    if (x.IsNotANumber() ||
        (x.Compare(y) == Relation::Less) == (ordering == Ordering::Less)) {
      return {x};
    }
  } else {
    if (ordering == Compare(x, y)) {
      return {x};
    }
  }
  return {y};
}

template<int KIND>
auto ComplexConstructor<KIND>::FoldScalar(
    FoldingContext &context, const Scalar<Operand> &x, const Scalar<Operand> &y)
    -> std::optional<Scalar<Result>> {
  return {Scalar<Result>{x, y}};
}

template<int KIND>
auto Concat<KIND>::FoldScalar(FoldingContext &context, const Scalar<Operand> &x,
    const Scalar<Operand> &y) -> std::optional<Scalar<Result>> {
  if constexpr (KIND == 1) {
    return {x + y};
  }
  return std::nullopt;
}

template<typename A>
auto Relational<A>::FoldScalar(FoldingContext &c, const Scalar<Operand> &a,
    const Scalar<Operand> &b) -> std::optional<Scalar<Result>> {
  if constexpr (A::category == TypeCategory::Integer) {
    switch (a.CompareSigned(b)) {
    case Ordering::Less:
      return {opr == RelationalOperator::LE || opr == RelationalOperator::LE ||
          opr == RelationalOperator::NE};
    case Ordering::Equal:
      return {opr == RelationalOperator::LE || opr == RelationalOperator::EQ ||
          opr == RelationalOperator::GE};
    case Ordering::Greater:
      return {opr == RelationalOperator::NE || opr == RelationalOperator::GE ||
          opr == RelationalOperator::GT};
    }
  }
  if constexpr (A::category == TypeCategory::Real) {
    switch (a.Compare(b)) {
    case Relation::Less:
      return {opr == RelationalOperator::LE || opr == RelationalOperator::LE ||
          opr == RelationalOperator::NE};
    case Relation::Equal:
      return {opr == RelationalOperator::LE || opr == RelationalOperator::EQ ||
          opr == RelationalOperator::GE};
    case Relation::Greater:
      return {opr == RelationalOperator::NE || opr == RelationalOperator::GE ||
          opr == RelationalOperator::GT};
    case Relation::Unordered: return std::nullopt;
    }
  }
  if constexpr (A::category == TypeCategory::Complex) {
    bool eqOk{opr == RelationalOperator::LE || opr == RelationalOperator::EQ ||
        opr == RelationalOperator::GE};
    return {eqOk == a.Equals(b)};
  }
  if constexpr (A::category == TypeCategory::Character) {
    switch (Compare(a, b)) {
    case Ordering::Less:
      return {opr == RelationalOperator::LE || opr == RelationalOperator::LE ||
          opr == RelationalOperator::NE};
    case Ordering::Equal:
      return {opr == RelationalOperator::LE || opr == RelationalOperator::EQ ||
          opr == RelationalOperator::GE};
    case Ordering::Greater:
      return {opr == RelationalOperator::NE || opr == RelationalOperator::GE ||
          opr == RelationalOperator::GT};
    }
  }
  return std::nullopt;
}

template<int KIND>
auto LogicalOperation<KIND>::FoldScalar(FoldingContext &context,
    const Scalar<Operand> &x, const Scalar<Operand> &y) const
    -> std::optional<Scalar<Result>> {
  bool xt{x.IsTrue()}, yt{y.IsTrue()};
  switch (logicalOperator) {
  case LogicalOperator::And: return {Scalar<Result>{xt && yt}};
  case LogicalOperator::Or: return {Scalar<Result>{xt || yt}};
  case LogicalOperator::Eqv: return {Scalar<Result>{xt == yt}};
  case LogicalOperator::Neqv: return {Scalar<Result>{xt != yt}};
  }
  return std::nullopt;
}

// Dump

template<typename D, typename R, typename... O>
std::ostream &Operation<D, R, O...>::Dump(std::ostream &o) const {
  operand<0>().Dump(o << derived().prefix());
  if constexpr (operands() > 1) {
    operand<1>().Dump(o << derived().infix());
  }
  return o << derived().suffix();
}

template<typename A> std::string Relational<A>::infix() const {
  return "."s + EnumToString(opr) + '.';
}

template<int KIND> const char *LogicalOperation<KIND>::infix() const {
  const char *result{nullptr};
  switch (logicalOperator) {
  case LogicalOperator::And: result = ".AND."; break;
  case LogicalOperator::Or: result = ".OR."; break;
  case LogicalOperator::Eqv: result = ".EQV."; break;
  case LogicalOperator::Neqv: result = ".NEQV."; break;
  }
  return result;
}

template<typename... A>
std::ostream &DumpExpr(std::ostream &o, const std::variant<A...> &u) {
  std::visit(common::visitors{[&](const BOZLiteralConstant &x) {
                                o << "Z'" << x.Hexadecimal() << "'";
                              },
                 [&](const auto &x) { x.Dump(o); }},
      u);
  return o;
}

template<TypeCategory CAT>
std::ostream &Expr<SomeKind<CAT>>::Dump(std::ostream &o) const {
  return DumpExpr(o, u.u);
}

std::ostream &AnyRelational::Dump(std::ostream &o) const {
  return DumpExpr(o, u);
}

std::ostream &Expr<SomeType>::Dump(std::ostream &o) const {
  return DumpExpr(o, u);
}

template<int KIND>
std::ostream &Expr<Type<TypeCategory::Integer, KIND>>::Dump(
    std::ostream &o) const {
  std::visit(common::visitors{[&](const Scalar<Result> &n) {
                                o << n.SignedDecimal() << '_' << KIND;
                              },
                 [&](const CopyableIndirection<DataRef> &d) { d->Dump(o); },
                 [&](const CopyableIndirection<FunctionRef> &d) { d->Dump(o); },
                 [&](const auto &x) { x.Dump(o); }},
      u_);
  return o;
}

template<int KIND>
std::ostream &Expr<Type<TypeCategory::Real, KIND>>::Dump(
    std::ostream &o) const {
  std::visit(common::visitors{[&](const Scalar<Result> &n) {
                                o << n.DumpHexadecimal();
                              },
                 [&](const CopyableIndirection<DataRef> &d) { d->Dump(o); },
                 [&](const CopyableIndirection<ComplexPart> &d) { d->Dump(o); },
                 [&](const CopyableIndirection<FunctionRef> &d) { d->Dump(o); },
                 [&](const auto &x) { x.Dump(o); }},
      u_);
  return o;
}

template<int KIND>
std::ostream &Expr<Type<TypeCategory::Complex, KIND>>::Dump(
    std::ostream &o) const {
  std::visit(common::visitors{[&](const Scalar<Result> &n) {
                                o << n.DumpHexadecimal();
                              },
                 [&](const CopyableIndirection<DataRef> &d) { d->Dump(o); },
                 [&](const CopyableIndirection<FunctionRef> &d) { d->Dump(o); },
                 [&](const auto &x) { x.Dump(o); }},
      u_);
  return o;
}

template<int KIND>
std::ostream &Expr<Type<TypeCategory::Character, KIND>>::Dump(
    std::ostream &o) const {
  std::visit(common::visitors{[&](const Scalar<Result> &s) {
                                o << KIND << '_'
                                  << parser::QuoteCharacterLiteral(s);
                              },
                 //          [&](const Parentheses<Result> &p) { p.Dump(o); },
                 [&](const Concat<KIND> &c) { c.Dump(o); },
                 [&](const Extremum<Result> &mm) { mm.Dump(o); },
                 [&](const auto &ind) { ind->Dump(o); }},
      u_);
  return o;
}

template<int KIND>
std::ostream &Expr<Type<TypeCategory::Logical, KIND>>::Dump(
    std::ostream &o) const {
  std::visit(common::visitors{[&](const Scalar<Result> &tf) {
                                o << (tf.IsTrue() ? ".TRUE." : ".FALSE.") << '_'
                                  << KIND;
                              },
                 [&](const CopyableIndirection<DataRef> &d) { d->Dump(o); },
                 [&](const CopyableIndirection<FunctionRef> &d) { d->Dump(o); },
                 [&](const auto &x) { x.Dump(o); }},
      u_);
  return o;
}

// LEN()
template<int KIND>
Expr<SubscriptInteger> Expr<Type<TypeCategory::Character, KIND>>::LEN() const {
  return std::visit(
      common::visitors{[](const Scalar<Result> &c) {
                         // std::string::size_type isn't convertible to uint64_t
                         // on Darwin
                         return Expr<SubscriptInteger>{
                             static_cast<std::uint64_t>(c.size())};
                       },
          [](const Concat<KIND> &c) {
            return c.template operand<0>().LEN() +
                c.template operand<1>().LEN();
          },
          [](const Extremum<Result> &c) {
            return Expr<SubscriptInteger>{Extremum<SubscriptInteger>{
                c.template operand<0>().LEN(), c.template operand<1>().LEN()}};
          },
          [](const CopyableIndirection<DataRef> &dr) { return dr->LEN(); },
          [](const CopyableIndirection<Substring> &ss) { return ss->LEN(); },
          [](const CopyableIndirection<FunctionRef> &fr) {
            return fr->proc().LEN();
          }},
      u_);
}

// ScalarValue

template<TypeCategory CAT>
auto Expr<SomeKind<CAT>>::ScalarValue() const -> std::optional<Scalar<Result>> {
  return std::visit(
      [](const auto &x) -> std::optional<Scalar<Result>> {
        if (auto c{x.ScalarValue()}) {
          return {Scalar<Result>{std::move(*c)}};
        }
        return std::nullopt;
      },
      u.u);
}

auto Expr<SomeType>::ScalarValue() const -> std::optional<Scalar<Result>> {
  return std::visit(
      common::visitors{
          [](const BOZLiteralConstant &) -> std::optional<Scalar<Result>> {
            return std::nullopt;
          },
          [](const auto &x) -> std::optional<Scalar<Result>> {
            if (auto c{x.ScalarValue()}) {
              return {common::MoveVariant<Scalar<Result>>(std::move(c->u))};
            }
            return std::nullopt;
          }},
      u);
}

// Rank

template<TypeCategory CAT> int Expr<SomeKind<CAT>>::Rank() const {
  return std::visit([](const auto &x) { return x.Rank(); }, u.u);
}

int Expr<SomeType>::Rank() const {
  // Written thus, instead of common::visitors, to dodge a bug in G++ 7.2.
  return std::visit(
      [](const auto &x) {
        if constexpr (std::is_same_v<std::decay_t<decltype(x)>,
                          BOZLiteralConstant>) {
          return 1;
        } else {
          return x.Rank();
        }
      },
      u);
}

// Template instantiations

template class Expr<Type<TypeCategory::Integer, 1>>;
template class Expr<Type<TypeCategory::Integer, 2>>;
template class Expr<Type<TypeCategory::Integer, 4>>;
template class Expr<Type<TypeCategory::Integer, 8>>;
template class Expr<Type<TypeCategory::Integer, 16>>;
template class Expr<Type<TypeCategory::Real, 2>>;
template class Expr<Type<TypeCategory::Real, 4>>;
template class Expr<Type<TypeCategory::Real, 8>>;
template class Expr<Type<TypeCategory::Real, 10>>;
template class Expr<Type<TypeCategory::Real, 16>>;
template class Expr<Type<TypeCategory::Complex, 2>>;
template class Expr<Type<TypeCategory::Complex, 4>>;
template class Expr<Type<TypeCategory::Complex, 8>>;
template class Expr<Type<TypeCategory::Complex, 10>>;
template class Expr<Type<TypeCategory::Complex, 16>>;
template class Expr<Type<TypeCategory::Character, 1>>;  // TODO others

template struct Relational<Type<TypeCategory::Integer, 1>>;
template struct Relational<Type<TypeCategory::Integer, 2>>;
template struct Relational<Type<TypeCategory::Integer, 4>>;
template struct Relational<Type<TypeCategory::Integer, 8>>;
template struct Relational<Type<TypeCategory::Integer, 16>>;
template struct Relational<Type<TypeCategory::Real, 2>>;
template struct Relational<Type<TypeCategory::Real, 4>>;
template struct Relational<Type<TypeCategory::Real, 8>>;
template struct Relational<Type<TypeCategory::Real, 10>>;
template struct Relational<Type<TypeCategory::Real, 16>>;
template struct Relational<Type<TypeCategory::Complex, 2>>;
template struct Relational<Type<TypeCategory::Complex, 4>>;
template struct Relational<Type<TypeCategory::Complex, 8>>;
template struct Relational<Type<TypeCategory::Complex, 10>>;
template struct Relational<Type<TypeCategory::Complex, 16>>;
template struct Relational<Type<TypeCategory::Character, 1>>;  // TODO others

template class Expr<Type<TypeCategory::Logical, 1>>;
template class Expr<Type<TypeCategory::Logical, 2>>;
template class Expr<Type<TypeCategory::Logical, 4>>;
template class Expr<Type<TypeCategory::Logical, 8>>;

template class Expr<SomeInteger>;
template class Expr<SomeReal>;
template class Expr<SomeComplex>;
template class Expr<SomeCharacter>;
template class Expr<SomeLogical>;

template class Expr<SomeType>;
}  // namespace Fortran::evaluate
