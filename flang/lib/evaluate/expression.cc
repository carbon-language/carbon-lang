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

// Dumping
template<typename... A>
std::ostream &DumpExprWithType(std::ostream &o, const std::variant<A...> &u) {
  std::visit(
      [&](const auto &x) {
        using Ty = typename std::remove_reference_t<decltype(x)>::Result;
        x.Dump(o << '(' << Ty::Dump() << "::") << ')';
      },
      u);
  return o;
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

template<typename CRTP, typename RESULT, typename A>
std::ostream &Unary<CRTP, RESULT, A>::Dump(
    std::ostream &o, const char *opr) const {
  return operand().Dump(o << opr) << ')';
}

template<typename CRTP, typename RESULT, typename A, typename B>
std::ostream &Binary<CRTP, RESULT, A, B>::Dump(
    std::ostream &o, const char *opr, const char *before) const {
  return right().Dump(left().Dump(o << before) << opr) << ')';
}

template<int KIND>
std::ostream &Expr<Type<TypeCategory::Integer, KIND>>::Dump(
    std::ostream &o) const {
  std::visit(
      common::visitors{[&](const Scalar<Result> &n) { o << n.SignedDecimal(); },
          [&](const CopyableIndirection<DataRef> &d) { d->Dump(o); },
          [&](const CopyableIndirection<FunctionRef> &d) { d->Dump(o); },
          [&](const Parentheses &p) { p.Dump(o, "("); },
          [&](const Negate &n) { n.Dump(o, "(-"); },
          [&](const Add &a) { a.Dump(o, "+"); },
          [&](const Subtract &s) { s.Dump(o, "-"); },
          [&](const Multiply &m) { m.Dump(o, "*"); },
          [&](const Divide &d) { d.Dump(o, "/"); },
          [&](const Power &p) { p.Dump(o, "**"); },
          [&](const Max &m) { m.Dump(o, ",", "MAX("); },
          [&](const Min &m) { m.Dump(o, ",", "MIN("); },
          [&](const auto &convert) {
            DumpExprWithType(o, convert.operand().u);
          }},
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
                 [&](const Parentheses &p) { p.Dump(o, "("); },
                 [&](const Negate &n) { n.Dump(o, "(-"); },
                 [&](const Add &a) { a.Dump(o, "+"); },
                 [&](const Subtract &s) { s.Dump(o, "-"); },
                 [&](const Multiply &m) { m.Dump(o, "*"); },
                 [&](const Divide &d) { d.Dump(o, "/"); },
                 [&](const Power &p) { p.Dump(o, "**"); },
                 [&](const IntPower &p) { p.Dump(o, "**"); },
                 [&](const Max &m) { m.Dump(o, ",", "MAX("); },
                 [&](const Min &m) { m.Dump(o, ",", "MIN("); },
                 [&](const RealPart &z) { z.Dump(o, "REAL("); },
                 [&](const AIMAG &p) { p.Dump(o, "AIMAG("); },
                 [&](const auto &convert) {
                   DumpExprWithType(o, convert.operand().u);
                 }},
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
                 [&](const Parentheses &p) { p.Dump(o, "("); },
                 [&](const Negate &n) { n.Dump(o, "(-"); },
                 [&](const Add &a) { a.Dump(o, "+"); },
                 [&](const Subtract &s) { s.Dump(o, "-"); },
                 [&](const Multiply &m) { m.Dump(o, "*"); },
                 [&](const Divide &d) { d.Dump(o, "/"); },
                 [&](const Power &p) { p.Dump(o, "**"); },
                 [&](const IntPower &p) { p.Dump(o, "**"); },
                 [&](const CMPLX &c) { c.Dump(o, ","); }},
      u_);
  return o;
}

template<int KIND>
std::ostream &Expr<Type<TypeCategory::Character, KIND>>::Dump(
    std::ostream &o) const {
  std::visit(common::visitors{[&](const Scalar<Result> &s) {
                                o << parser::QuoteCharacterLiteral(s);
                              },
                 [&](const Concat &concat) { concat.Dump(o, "//"); },
                 [&](const Max &m) { m.Dump(o, ",", "MAX("); },
                 [&](const Min &m) { m.Dump(o, ",", "MIN("); },
                 [&](const auto &ind) { ind->Dump(o); }},
      u_);
  return o;
}

template<typename A> std::ostream &Comparison<A>::Dump(std::ostream &o) const {
  o << '(' << A::Dump() << "::";
  this->left().Dump(o);
  o << '.' << EnumToString(this->opr) << '.';
  return this->right().Dump(o) << ')';
}

template<int KIND>
std::ostream &Expr<Type<TypeCategory::Logical, KIND>>::Dump(
    std::ostream &o) const {
  std::visit(common::visitors{[&](const Scalar<Result> &tf) {
                                o << (tf.IsTrue() ? ".TRUE." : ".FALSE.");
                              },
                 [&](const CopyableIndirection<DataRef> &d) { d->Dump(o); },
                 [&](const CopyableIndirection<FunctionRef> &d) { d->Dump(o); },
                 [&](const Not &n) { n.Dump(o, "(.NOT."); },
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
template<typename CRTP, typename RESULT, typename A>
auto Unary<CRTP, RESULT, A>::Fold(FoldingContext &context)
    -> std::optional<Scalar<Result>> {
  if (std::optional<Scalar<Operand>> c{operand_->Fold(context)}) {
    return static_cast<CRTP *>(this)->FoldScalar(context, *c);
  }
  return std::nullopt;
}

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
auto Expr<Type<TypeCategory::Integer, KIND>>::ConvertInteger::FoldScalar(
    FoldingContext &context, const SomeKindScalar<TypeCategory::Integer> &c)
    -> std::optional<Scalar<Result>> {
  return std::visit(
      [&](auto &x) -> std::optional<Scalar<Result>> {
        auto converted{Scalar<Result>::ConvertSigned(x)};
        if (converted.overflow) {
          context.messages.Say("integer conversion overflowed"_en_US);
          return std::nullopt;
        }
        return {std::move(converted.value)};
      },
      c.u);
}

template<int KIND>
auto Expr<Type<TypeCategory::Integer, KIND>>::ConvertReal::FoldScalar(
    FoldingContext &context, const SomeKindScalar<TypeCategory::Real> &c)
    -> std::optional<Scalar<Result>> {
  return std::visit(
      [&](auto &x) -> std::optional<Scalar<Result>> {
        auto converted{x.template ToInteger<Scalar<Result>>()};
        if (converted.flags.test(RealFlag::Overflow)) {
          context.messages.Say("real->integer conversion overflowed"_en_US);
          return std::nullopt;
        }
        if (converted.flags.test(RealFlag::InvalidArgument)) {
          context.messages.Say(
              "real->integer conversion: invalid argument"_en_US);
          return std::nullopt;
        }
        return {std::move(converted.value)};
      },
      c.u);
}

template<int KIND>
auto Expr<Type<TypeCategory::Integer, KIND>>::Negate::FoldScalar(
    FoldingContext &context, const Scalar<Result> &c)
    -> std::optional<Scalar<Result>> {
  auto negated{c.Negate()};
  if (negated.overflow) {
    context.messages.Say("integer negation overflowed"_en_US);
    return std::nullopt;
  }
  return {std::move(negated.value)};
}

template<int KIND>
auto Expr<Type<TypeCategory::Integer, KIND>>::Add::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a, const Scalar<Result> &b)
    -> std::optional<Scalar<Result>> {
  auto sum{a.AddSigned(b)};
  if (sum.overflow) {
    context.messages.Say("integer addition overflowed"_en_US);
    return std::nullopt;
  }
  return {std::move(sum.value)};
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
  return std::visit(
      [&](auto &x) -> std::optional<Scalar<Result>> {
        using Ty = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<Ty, Scalar<Result>>) {
          return {x};
        }
        if constexpr (evaluate::FoldableTrait<Ty>) {
          auto c{x.Fold(context)};
          if (c.has_value()) {
            u_ = *c;
            return c;
          }
        }
        return std::nullopt;
      },
      u_);
}

template<int KIND>
auto Expr<Type<TypeCategory::Real, KIND>>::ConvertInteger::FoldScalar(
    FoldingContext &context, const SomeKindScalar<TypeCategory::Integer> &c)
    -> std::optional<Scalar<Result>> {
  return std::visit(
      [&](auto &x) -> std::optional<Scalar<Result>> {
        auto converted{Scalar<Result>::FromInteger(x)};
        RealFlagWarnings(context, converted.flags, "integer->real conversion");
        return {std::move(converted.value)};
      },
      c.u);
}

template<int KIND>
auto Expr<Type<TypeCategory::Real, KIND>>::ConvertReal::FoldScalar(
    FoldingContext &context, const SomeKindScalar<TypeCategory::Real> &c)
    -> std::optional<Scalar<Result>> {
  return std::visit(
      [&](auto &x) -> std::optional<Scalar<Result>> {
        auto converted{Scalar<Result>::Convert(x)};
        RealFlagWarnings(context, converted.flags, "real conversion");
        return {std::move(converted.value)};
      },
      c.u);
}

template<int KIND>
auto Expr<Type<TypeCategory::Real, KIND>>::Negate::FoldScalar(
    FoldingContext &context, const Scalar<Result> &c)
    -> std::optional<Scalar<Result>> {
  return {c.Negate()};
}

template<int KIND>
auto Expr<Type<TypeCategory::Real, KIND>>::Add::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a, const Scalar<Result> &b)
    -> std::optional<Scalar<Result>> {
  auto sum{a.Add(b, context.rounding)};
  RealFlagWarnings(context, sum.flags, "real addition");
  return {std::move(sum.value)};
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
auto Expr<Type<TypeCategory::Real, KIND>>::RealPart::FoldScalar(
    FoldingContext &context,
    const Scalar<SameKind<TypeCategory::Complex, Result>> &z)
    -> std::optional<Scalar<Result>> {
  return {z.REAL()};
}

template<int KIND>
auto Expr<Type<TypeCategory::Real, KIND>>::AIMAG::FoldScalar(
    FoldingContext &context,
    const Scalar<SameKind<TypeCategory::Complex, Result>> &z)
    -> std::optional<Scalar<Result>> {
  return {z.AIMAG()};
}

template<int KIND>
auto Expr<Type<TypeCategory::Real, KIND>>::Fold(FoldingContext &context)
    -> std::optional<Scalar<Result>> {
  return std::visit(
      [&](auto &x) -> std::optional<Scalar<Result>> {
        using Ty = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<Ty, Scalar<Result>>) {
          return {x};
        }
        if constexpr (evaluate::FoldableTrait<Ty>) {
          auto c{x.Fold(context)};
          if (c.has_value()) {
            if (context.flushDenormalsToZero) {
              *c = c->FlushDenormalToZero();
            }
            u_ = *c;
            return c;
          }
        }
        return std::nullopt;
      },
      u_);
}

template<int KIND>
auto Expr<Type<TypeCategory::Complex, KIND>>::Negate::FoldScalar(
    FoldingContext &context, const Scalar<Result> &c)
    -> std::optional<Scalar<Result>> {
  return {c.Negate()};
}

template<int KIND>
auto Expr<Type<TypeCategory::Complex, KIND>>::Add::FoldScalar(
    FoldingContext &context, const Scalar<Result> &a, const Scalar<Result> &b)
    -> std::optional<Scalar<Result>> {
  auto sum{a.Add(b, context.rounding)};
  RealFlagWarnings(context, sum.flags, "complex addition");
  return {std::move(sum.value)};
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
  return std::visit(
      [&](auto &x) -> std::optional<Scalar<Result>> {
        using Ty = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<Ty, Scalar<Result>>) {
          return {x};
        }
        if constexpr (evaluate::FoldableTrait<Ty>) {
          auto c{x.Fold(context)};
          if (c.has_value()) {
            if (context.flushDenormalsToZero) {
              *c = c->FlushDenormalToZero();
            }
            u_ = *c;
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
  return std::visit(
      [&](auto &x) -> std::optional<Scalar<Result>> {
        using Ty = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<Ty, Scalar<Result>>) {
          return {x};
        }
        if constexpr (evaluate::FoldableTrait<Ty>) {
          auto c{x.Fold(context)};
          if (c.has_value()) {
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
auto Expr<Type<TypeCategory::Logical, KIND>>::Not::FoldScalar(
    FoldingContext &context, const Scalar<Result> &x)
    -> std::optional<Scalar<Result>> {
  return {Scalar<Result>{!x.IsTrue()}};
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
  return std::visit(
      [&](auto &x) -> std::optional<Scalar<Result>> {
        using Ty = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<Ty, Scalar<Result>>) {
          return {x};
        }
        if constexpr (evaluate::FoldableTrait<Ty>) {
          std::optional<Scalar<Result>> c{x.Fold(context)};
          if (c.has_value()) {
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

template class Expr<SomeType>;
template class Expr<SomeKind<TypeCategory::Integer>>;
template class Expr<SomeKind<TypeCategory::Real>>;
template class Expr<SomeKind<TypeCategory::Complex>>;
template class Expr<SomeKind<TypeCategory::Character>>;
template class Expr<SomeKind<TypeCategory::Logical>>;

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
template class Expr<Type<TypeCategory::Character, 1>>;
template class Expr<Type<TypeCategory::Logical, 1>>;
template class Expr<Type<TypeCategory::Logical, 2>>;
template class Expr<Type<TypeCategory::Logical, 4>>;
template class Expr<Type<TypeCategory::Logical, 8>>;

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
template struct Comparison<Type<TypeCategory::Character, 1>>;
}  // namespace Fortran::evaluate
