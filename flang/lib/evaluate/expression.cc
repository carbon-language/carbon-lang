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
#include "variable.h"
#include "../common/idioms.h"
#include "../parser/characters.h"
#include "../parser/message.h"
#include <ostream>
#include <string>
#include <type_traits>

using namespace Fortran::parser::literals;

namespace Fortran::evaluate {

// Folding

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
        c.u);
  } else if constexpr (std::is_same_v<Result, SomeKind<Result::category>>) {
    if constexpr (Result::category == Operand::category) {
      return {Scalar<Result>{c}};
    }
  } else if constexpr (std::is_same_v<Operand, SomeKind<Operand::category>>) {
    return std::visit(
        [&](const auto &x) -> std::optional<Scalar<Result>> {
          using Ty = ScalarValueType<std::decay_t<decltype(x)>>;
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
  return {isRealPart ? z.REAL() : z.AIMAG()};
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
      context.messages.Say("INTEGER addition overflowed"_en_US);
    } else {
      return {std::move(sum.value)};
    }
  } else {
    auto sum{x.Add(y, context.rounding)};
    RealFlagWarnings(context, sum.flags, "addition");
    return {std::move(sum.value)};
  }
  return std::nullopt;
}

// Dumping

template<typename D, typename R, typename... O>
std::ostream &Operation<D, R, O...>::Dump(std::ostream &o) const {
  operand<0>().Dump(o << derived().prefix());
  if constexpr (operands() > 1) {
    operand<1>().Dump(o << derived().infix());
  }
  return o << derived().suffix();
}

template<typename A> std::string Comparison<A>::infix() const {
  return "."s + EnumToString(opr) + '.';
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
  return DumpExpr(o, u);
}

template<TypeCategory CAT>
std::ostream &CategoryComparison<CAT>::Dump(std::ostream &o) const {
  return DumpExpr(o, u);
}

std::ostream &Expr<SomeType>::Dump(std::ostream &o) const {
  return DumpExpr(o, u);
}

template<typename CRTP, typename RESULT, typename A, typename B>
std::ostream &Binary<CRTP, RESULT, A, B>::Dump(
    std::ostream &o, const char *opr, const char *before) const {
  return right().Dump(left().Dump(o << before) << opr) << ')';
}

template<int KIND>
std::ostream &Expr<Type<TypeCategory::Integer, KIND>>::Dump(
    std::ostream &o) const {
  std::visit(common::visitors{[&](const Scalar<Result> &n) {
                                o << n.SignedDecimal() << '_' << KIND;
                              },
                 [&](const CopyableIndirection<DataRef> &d) { d->Dump(o); },
                 [&](const CopyableIndirection<FunctionRef> &d) { d->Dump(o); },
                 [&](const Subtract &s) { s.Dump(o, "-"); },
                 [&](const Multiply &m) { m.Dump(o, "*"); },
                 [&](const Divide &d) { d.Dump(o, "/"); },
                 [&](const Power &p) { p.Dump(o, "**"); },
                 [&](const Max &m) { m.Dump(o, ",", "MAX("); },
                 [&](const Min &m) { m.Dump(o, ",", "MIN("); },
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
                 [&](const Subtract &s) { s.Dump(o, "-"); },
                 [&](const Multiply &m) { m.Dump(o, "*"); },
                 [&](const Divide &d) { d.Dump(o, "/"); },
                 [&](const Power &p) { p.Dump(o, "**"); },
                 [&](const IntPower &p) { p.Dump(o, "**"); },
                 [&](const Max &m) { m.Dump(o, ",", "MAX("); },
                 [&](const Min &m) { m.Dump(o, ",", "MIN("); },
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
                 [&](const Subtract &s) { s.Dump(o, "-"); },
                 [&](const Multiply &m) { m.Dump(o, "*"); },
                 [&](const Divide &d) { d.Dump(o, "/"); },
                 [&](const Power &p) { p.Dump(o, "**"); },
                 [&](const IntPower &p) { p.Dump(o, "**"); },
                 [&](const CMPLX &c) { c.Dump(o, ","); },
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
                 [&](const Concat &concat) { concat.Dump(o, "//"); },
                 [&](const Max &m) { m.Dump(o, ",", "MAX("); },
                 [&](const Min &m) { m.Dump(o, ",", "MIN("); },
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
                 //          [&](const Parentheses<Result> &p) { p.Dump(o); },
                 [&](const Not<KIND> &n) { n.Dump(o); },
                 [&](const And &a) { a.Dump(o, ".AND."); },
                 [&](const Or &a) { a.Dump(o, ".OR."); },
                 [&](const Eqv &a) { a.Dump(o, ".EQV."); },
                 [&](const Neqv &a) { a.Dump(o, ".NEQV."); },
                 [&](const auto &comparison) { comparison.Dump(o); }},
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
          [](const Concat &c) { return c.left().LEN() + c.right().LEN(); },
          [](const Max &c) {
            return Expr<SubscriptInteger>{
                Expr<SubscriptInteger>::Max{c.left().LEN(), c.right().LEN()}};
          },
          [](const Min &c) {
            return Expr<SubscriptInteger>{
                Expr<SubscriptInteger>::Max{c.left().LEN(), c.right().LEN()}};
          },
          [](const CopyableIndirection<DataRef> &dr) { return dr->LEN(); },
          [](const CopyableIndirection<Substring> &ss) { return ss->LEN(); },
          [](const CopyableIndirection<FunctionRef> &fr) {
            return fr->proc().LEN();
          }},
      u_);
}

// Rank
template<typename CRTP, typename RESULT, typename A, typename B>
int Binary<CRTP, RESULT, A, B>::Rank() const {
  int lrank{left_.Rank()};
  if (lrank > 0) {
    return lrank;
  }
  return right_.Rank();
}

// Folding
template<typename CRTP, typename RESULT, typename A, typename B>
auto Binary<CRTP, RESULT, A, B>::Fold(FoldingContext &context)
    -> std::optional<Scalar<Result>> {
  std::optional<Scalar<Left>> lc{left_->Fold(context)};
  std::optional<Scalar<Right>> rc{right_->Fold(context)};
  if (lc.has_value() && rc.has_value()) {
    return static_cast<CRTP *>(this)->FoldScalar(context, *lc, *rc);
  }
  return std::nullopt;
}

template<int KIND>
auto Expr<Type<TypeCategory::Integer, KIND>>::Subtract::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a, const Scalar<Result> &b)
    -> std::optional<Scalar<Result>> {
  auto diff{a.SubtractSigned(b)};
  if (diff.overflow) {
    context.messages.Say("integer subtraction overflowed"_en_US);
    return std::nullopt;
  }
  return {std::move(diff.value)};
}

template<int KIND>
auto Expr<Type<TypeCategory::Integer, KIND>>::Multiply::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a, const Scalar<Result> &b)
    -> std::optional<Scalar<Result>> {
  auto product{a.MultiplySigned(b)};
  if (product.SignedMultiplicationOverflowed()) {
    context.messages.Say("integer multiplication overflowed"_en_US);
    return std::nullopt;
  }
  return {std::move(product.lower)};
}

template<int KIND>
auto Expr<Type<TypeCategory::Integer, KIND>>::Divide::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a, const Scalar<Result> &b)
    -> std::optional<Scalar<Result>> {
  auto qr{a.DivideSigned(b)};
  if (qr.divisionByZero) {
    context.messages.Say("integer division by zero"_en_US);
    return std::nullopt;
  }
  if (qr.overflow) {
    context.messages.Say("integer division overflowed"_en_US);
    return std::nullopt;
  }
  return {std::move(qr.quotient)};
}

template<int KIND>
auto Expr<Type<TypeCategory::Integer, KIND>>::Power::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a, const Scalar<Result> &b)
    -> std::optional<Scalar<Result>> {
  typename Scalar<Result>::PowerWithErrors power{a.Power(b)};
  if (power.divisionByZero) {
    context.messages.Say("zero to negative power"_en_US);
    return std::nullopt;
  }
  if (power.overflow) {
    context.messages.Say("integer power overflowed"_en_US);
    return std::nullopt;
  }
  if (power.zeroToZero) {
    context.messages.Say("integer 0**0"_en_US);
    return std::nullopt;
  }
  return {std::move(power.power)};
}

template<int KIND>
auto Expr<Type<TypeCategory::Integer, KIND>>::Max::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a, const Scalar<Result> &b)
    -> std::optional<Scalar<Result>> {
  if (a.CompareSigned(b) == Ordering::Greater) {
    return {a};
  }
  return {b};
}

template<int KIND>
auto Expr<Type<TypeCategory::Integer, KIND>>::Min::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a, const Scalar<Result> &b)
    -> std::optional<Scalar<Result>> {
  if (a.CompareSigned(b) == Ordering::Less) {
    return {a};
  }
  return {b};
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
auto Expr<Type<TypeCategory::Real, KIND>>::Subtract::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a, const Scalar<Result> &b)
    -> std::optional<Scalar<Result>> {
  auto difference{a.Subtract(b, context.rounding)};
  RealFlagWarnings(context, difference.flags, "real subtraction");
  return {std::move(difference.value)};
}

template<int KIND>
auto Expr<Type<TypeCategory::Real, KIND>>::Multiply::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a, const Scalar<Result> &b)
    -> std::optional<Scalar<Result>> {
  auto product{a.Multiply(b, context.rounding)};
  RealFlagWarnings(context, product.flags, "real multiplication");
  return {std::move(product.value)};
}

template<int KIND>
auto Expr<Type<TypeCategory::Real, KIND>>::Divide::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a, const Scalar<Result> &b)
    -> std::optional<Scalar<Result>> {
  auto quotient{a.Divide(b, context.rounding)};
  RealFlagWarnings(context, quotient.flags, "real division");
  return {std::move(quotient.value)};
}

template<int KIND>
auto Expr<Type<TypeCategory::Real, KIND>>::Power::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a, const Scalar<Result> &b)
    -> std::optional<Scalar<Result>> {
  return std::nullopt;  // TODO
}

template<int KIND>
auto Expr<Type<TypeCategory::Real, KIND>>::IntPower::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a,
    const SomeKindScalar<TypeCategory::Integer> &b)
    -> std::optional<Scalar<Result>> {
  return std::visit(
      [&](const auto &pow) -> std::optional<Scalar<Result>> {
        auto power{evaluate::IntPower(a, pow)};
        RealFlagWarnings(context, power.flags, "raising to integer power");
        return {std::move(power.value)};
      },
      b.u);
}

template<int KIND>
auto Expr<Type<TypeCategory::Real, KIND>>::Max::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a, const Scalar<Result> &b)
    -> std::optional<Scalar<Result>> {
  if (b.IsNotANumber() || a.Compare(b) == Relation::Less) {
    return {b};
  }
  return {a};
}

template<int KIND>
auto Expr<Type<TypeCategory::Real, KIND>>::Min::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a, const Scalar<Result> &b)
    -> std::optional<Scalar<Result>> {
  if (b.IsNotANumber() || a.Compare(b) == Relation::Greater) {
    return {b};
  }
  return {a};
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
auto Expr<Type<TypeCategory::Complex, KIND>>::Subtract::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a, const Scalar<Result> &b)
    -> std::optional<Scalar<Result>> {
  auto difference{a.Subtract(b, context.rounding)};
  RealFlagWarnings(context, difference.flags, "complex subtraction");
  return {std::move(difference.value)};
}

template<int KIND>
auto Expr<Type<TypeCategory::Complex, KIND>>::Multiply::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a, const Scalar<Result> &b)
    -> std::optional<Scalar<Result>> {
  auto product{a.Multiply(b, context.rounding)};
  RealFlagWarnings(context, product.flags, "complex multiplication");
  return {std::move(product.value)};
}

template<int KIND>
auto Expr<Type<TypeCategory::Complex, KIND>>::Divide::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a, const Scalar<Result> &b)
    -> std::optional<Scalar<Result>> {
  auto quotient{a.Divide(b, context.rounding)};
  RealFlagWarnings(context, quotient.flags, "complex  division");
  return {std::move(quotient.value)};
}

template<int KIND>
auto Expr<Type<TypeCategory::Complex, KIND>>::Power::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a, const Scalar<Result> &b)
    -> std::optional<Scalar<Result>> {
  return std::nullopt;  // TODO
}

template<int KIND>
auto Expr<Type<TypeCategory::Complex, KIND>>::IntPower::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a,
    const SomeKindScalar<TypeCategory::Integer> &b)
    -> std::optional<Scalar<Result>> {
  return std::visit(
      [&](const auto &pow) -> std::optional<Scalar<Result>> {
        auto power{evaluate::IntPower(a, pow)};
        RealFlagWarnings(context, power.flags, "raising to integer power");
        return {std::move(power.value)};
      },
      b.u);
}

template<int KIND>
auto Expr<Type<TypeCategory::Complex, KIND>>::CMPLX::FoldScalar(
    FoldingContext &context,
    const Scalar<SameKind<TypeCategory::Real, Result>> &a,
    const Scalar<SameKind<TypeCategory::Real, Result>> &b)
    -> std::optional<Scalar<Result>> {
  return {Scalar<Result>{a, b}};
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
auto Expr<Type<TypeCategory::Character, KIND>>::Concat::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a, const Scalar<Result> &b)
    -> std::optional<Scalar<Result>> {
  if constexpr (KIND == 1) {
    return {a + b};
  }
  return std::nullopt;
}

template<int KIND>
auto Expr<Type<TypeCategory::Character, KIND>>::Max::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a, const Scalar<Result> &b)
    -> std::optional<Scalar<Result>> {
  if (Compare(a, b) == Ordering::Less) {
    return {b};
  }
  return {a};
}

template<int KIND>
auto Expr<Type<TypeCategory::Character, KIND>>::Min::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a, const Scalar<Result> &b)
    -> std::optional<Scalar<Result>> {
  if (Compare(a, b) == Ordering::Greater) {
    return {b};
  }
  return {a};
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

template<typename A>
auto Comparison<A>::FoldScalar(FoldingContext &c, const Scalar<Operand> &a,
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
auto Expr<Type<TypeCategory::Logical, KIND>>::And::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a, const Scalar<Result> &b)
    -> std::optional<Scalar<Result>> {
  return {Scalar<Result>{a.IsTrue() && b.IsTrue()}};
}

template<int KIND>
auto Expr<Type<TypeCategory::Logical, KIND>>::Or::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a, const Scalar<Result> &b)
    -> std::optional<Scalar<Result>> {
  return {Scalar<Result>{a.IsTrue() || b.IsTrue()}};
}

template<int KIND>
auto Expr<Type<TypeCategory::Logical, KIND>>::Eqv::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a, const Scalar<Result> &b)
    -> std::optional<Scalar<Result>> {
  return {Scalar<Result>{a.IsTrue() == b.IsTrue()}};
}

template<int KIND>
auto Expr<Type<TypeCategory::Logical, KIND>>::Neqv::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a, const Scalar<Result> &b)
    -> std::optional<Scalar<Result>> {
  return {Scalar<Result>{a.IsTrue() != b.IsTrue()}};
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
            u_ = *c;
            return c;
          }
        }
        return std::nullopt;
      },
      u_);
}

auto Expr<SomeType>::ScalarValue() const -> std::optional<Scalar<Result>> {
  return std::visit(
      common::visitors{
          [](const BOZLiteralConstant &) -> std::optional<Scalar<Result>> {
            return std::nullopt;
          },
          [](const auto &x) -> std::optional<Scalar<Result>> {
            if (auto c{x.ScalarValue()}) {
              return {Scalar<Result>{std::move(*c)}};
            }
            return std::nullopt;
          }},
      u);
}

template<TypeCategory CAT>
auto Expr<SomeKind<CAT>>::ScalarValue() const -> std::optional<Scalar<Result>> {
  return std::visit(
      [](const auto &x) -> std::optional<Scalar<Result>> {
        if (auto c{x.ScalarValue()}) {
          return {Scalar<Result>{std::move(*c)}};
        }
        return std::nullopt;
      },
      u);
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
      u);
}

template<TypeCategory CAT> int Expr<SomeKind<CAT>>::Rank() const {
  return std::visit([](const auto &x) { return x.Rank(); }, u);
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
              return {Scalar<Result>{std::move(*c)}};
            }
            return std::nullopt;
          }},
      u);
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

template struct Comparison<Type<TypeCategory::Integer, 1>>;
template struct Comparison<Type<TypeCategory::Integer, 2>>;
template struct Comparison<Type<TypeCategory::Integer, 4>>;
template struct Comparison<Type<TypeCategory::Integer, 8>>;
template struct Comparison<Type<TypeCategory::Integer, 16>>;
template struct Comparison<Type<TypeCategory::Real, 2>>;
template struct Comparison<Type<TypeCategory::Real, 4>>;
template struct Comparison<Type<TypeCategory::Real, 8>>;
template struct Comparison<Type<TypeCategory::Real, 10>>;
template struct Comparison<Type<TypeCategory::Real, 16>>;
template struct Comparison<Type<TypeCategory::Complex, 2>>;
template struct Comparison<Type<TypeCategory::Complex, 4>>;
template struct Comparison<Type<TypeCategory::Complex, 8>>;
template struct Comparison<Type<TypeCategory::Complex, 10>>;
template struct Comparison<Type<TypeCategory::Complex, 16>>;
template struct Comparison<Type<TypeCategory::Character, 1>>;  // TODO others

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
