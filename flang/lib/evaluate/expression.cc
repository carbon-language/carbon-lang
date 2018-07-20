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
#include "variable.h"
#include "../common/idioms.h"
#include "../parser/characters.h"
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
  std::visit([&](const auto &x) { x.Dump(o); }, u);
  return o;
}

template<Category CAT>
std::ostream &CategoryExpr<CAT>::Dump(std::ostream &o) const {
  return DumpExpr(o, u);
}

template<Category CAT>
std::ostream &CategoryComparison<CAT>::Dump(std::ostream &o) const {
  return DumpExpr(o, u);
}

std::ostream &GenericExpr::Dump(std::ostream &o) const {
  return DumpExpr(o, u);
}

template<typename CRTP, typename RESULT, typename A, typename ASCALAR>
std::ostream &Unary<CRTP, RESULT, A, ASCALAR>::Dump(
    std::ostream &o, const char *opr) const {
  return operand().Dump(o << opr) << ')';
}

template<typename CRTP, typename RESULT, typename A, typename B,
    typename ASCALAR, typename BSCALAR>
std::ostream &Binary<CRTP, RESULT, A, B, ASCALAR, BSCALAR>::Dump(
    std::ostream &o, const char *opr, const char *before) const {
  return right().Dump(left().Dump(o << before) << opr) << ')';
}

template<int KIND>
std::ostream &IntegerExpr<KIND>::Dump(std::ostream &o) const {
  std::visit(common::visitors{[&](const Scalar &n) { o << n.SignedDecimal(); },
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

template<int KIND> std::ostream &RealExpr<KIND>::Dump(std::ostream &o) const {
  std::visit(
      common::visitors{[&](const Scalar &n) { o << n.DumpHexadecimal(); },
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
std::ostream &ComplexExpr<KIND>::Dump(std::ostream &o) const {
  std::visit(
      common::visitors{[&](const Scalar &n) { o << n.DumpHexadecimal(); },
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
std::ostream &CharacterExpr<KIND>::Dump(std::ostream &o) const {
  std::visit(common::visitors{[&](const Scalar &s) {
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
  using Ty = typename A::Result;
  o << '(' << Ty::Dump() << "::";
  this->left().Dump(o);  // TODO: is this-> still needed?  Also below.
  o << '.' << EnumToString(this->opr) << '.';
  return this->right().Dump(o) << ')';
}

std::ostream &LogicalExpr::Dump(std::ostream &o) const {
  std::visit(
      common::visitors{[&](const bool &tf) { o << (tf ? ".T." : ".F."); },
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
template<int KIND> SubscriptIntegerExpr CharacterExpr<KIND>::LEN() const {
  return std::visit(
      common::visitors{
          [](const Scalar &c) { return SubscriptIntegerExpr{c.size()}; },
          [](const Concat &c) { return c.left().LEN() + c.right().LEN(); },
          [](const Max &c) {
            return SubscriptIntegerExpr{
                SubscriptIntegerExpr::Max{c.left().LEN(), c.right().LEN()}};
          },
          [](const Min &c) {
            return SubscriptIntegerExpr{
                SubscriptIntegerExpr::Max{c.left().LEN(), c.right().LEN()}};
          },
          [](const CopyableIndirection<DataRef> &dr) { return dr->LEN(); },
          [](const CopyableIndirection<Substring> &ss) { return ss->LEN(); },
          [](const CopyableIndirection<FunctionRef> &fr) {
            return fr->proc().LEN();
          }},
      u_);
}

// Rank
template<typename CRTP, typename RESULT, typename A, typename B,
    typename ASCALAR, typename BSCALAR>
int Binary<CRTP, RESULT, A, B, ASCALAR, BSCALAR>::Rank() const {
  int lrank{left_.Rank()};
  if (lrank > 0) {
    return lrank;
  }
  return right_.Rank();
}

// Folding
template<typename CRTP, typename RESULT, typename A, typename ASCALAR>
auto Unary<CRTP, RESULT, A, ASCALAR>::Fold(FoldingContext &context)
    -> std::optional<Scalar> {
  if (std::optional<OperandScalar> c{operand_->Fold(context)}) {
    return static_cast<CRTP *>(this)->FoldScalar(context, *c);
  }
  return {};
}

template<typename CRTP, typename RESULT, typename A, typename B,
    typename ASCALAR, typename BSCALAR>
auto Binary<CRTP, RESULT, A, B, ASCALAR, BSCALAR>::Fold(FoldingContext &context)
    -> std::optional<Scalar> {
  std::optional<LeftScalar> lc{left_->Fold(context)};
  std::optional<RightScalar> rc{right_->Fold(context)};
  if (lc.has_value() && rc.has_value()) {
    return static_cast<CRTP *>(this)->FoldScalar(context, *lc, *rc);
  }
  return {};
}

template<int KIND>
auto IntegerExpr<KIND>::ConvertInteger::FoldScalar(FoldingContext &context,
    const CategoryScalar<Category::Integer> &c) -> std::optional<Scalar> {
  return std::visit(
      [&](auto &x) -> std::optional<Scalar> {
        auto converted{Scalar::ConvertSigned(x)};
        if (converted.overflow && context.messages != nullptr) {
          context.messages->Say(
              context.at, "integer conversion overflowed"_en_US);
          return {};
        }
        return {std::move(converted.value)};
      },
      c.u);
}

template<int KIND>
auto IntegerExpr<KIND>::Negate::FoldScalar(
    FoldingContext &context, const Scalar &c) -> std::optional<Scalar> {
  auto negated{c.Negate()};
  if (negated.overflow && context.messages != nullptr) {
    context.messages->Say(context.at, "integer negation overflowed"_en_US);
    return {};
  }
  return {std::move(negated.value)};
}

template<int KIND>
auto IntegerExpr<KIND>::Add::FoldScalar(FoldingContext &context,
    const Scalar &a, const Scalar &b) -> std::optional<Scalar> {
  auto sum{a.AddSigned(b)};
  if (sum.overflow && context.messages != nullptr) {
    context.messages->Say(context.at, "integer addition overflowed"_en_US);
    return {};
  }
  return {std::move(sum.value)};
}

template<int KIND>
auto IntegerExpr<KIND>::Subtract::FoldScalar(FoldingContext &context,
    const Scalar &a, const Scalar &b) -> std::optional<Scalar> {
  auto diff{a.SubtractSigned(b)};
  if (diff.overflow && context.messages != nullptr) {
    context.messages->Say(context.at, "integer subtraction overflowed"_en_US);
    return {};
  }
  return {std::move(diff.value)};
}

template<int KIND>
auto IntegerExpr<KIND>::Multiply::FoldScalar(FoldingContext &context,
    const Scalar &a, const Scalar &b) -> std::optional<Scalar> {
  auto product{a.MultiplySigned(b)};
  if (product.SignedMultiplicationOverflowed() && context.messages != nullptr) {
    context.messages->Say(
        context.at, "integer multiplication overflowed"_en_US);
    return {};
  }
  return {std::move(product.lower)};
}

template<int KIND>
auto IntegerExpr<KIND>::Divide::FoldScalar(FoldingContext &context,
    const Scalar &a, const Scalar &b) -> std::optional<Scalar> {
  auto qr{a.DivideSigned(b)};
  if (context.messages != nullptr) {
    if (qr.divisionByZero) {
      context.messages->Say(context.at, "integer division by zero"_en_US);
      return {};
    }
    if (qr.overflow) {
      context.messages->Say(context.at, "integer division overflowed"_en_US);
      return {};
    }
  }
  return {std::move(qr.quotient)};
}

template<int KIND>
auto IntegerExpr<KIND>::Power::FoldScalar(FoldingContext &context,
    const Scalar &a, const Scalar &b) -> std::optional<Scalar> {
  typename Scalar::PowerWithErrors power{a.Power(b)};
  if (context.messages != nullptr) {
    if (power.divisionByZero) {
      context.messages->Say(context.at, "zero to negative power"_en_US);
      return {};
    }
    if (power.overflow) {
      context.messages->Say(context.at, "integer power overflowed"_en_US);
      return {};
    }
    if (power.zeroToZero) {
      context.messages->Say(context.at, "integer 0**0"_en_US);
      return {};
    }
  }
  return {std::move(power.power)};
}

template<int KIND>
auto IntegerExpr<KIND>::Max::FoldScalar(FoldingContext &context,
    const Scalar &a, const Scalar &b) -> std::optional<Scalar> {
  if (a.CompareSigned(b) == Ordering::Greater) {
    return {a};
  }
  return {b};
}

template<int KIND>
auto IntegerExpr<KIND>::Min::FoldScalar(FoldingContext &context,
    const Scalar &a, const Scalar &b) -> std::optional<Scalar> {
  if (a.CompareSigned(b) == Ordering::Less) {
    return {a};
  }
  return {b};
}

template<int KIND>
auto IntegerExpr<KIND>::Fold(FoldingContext &context) -> std::optional<Scalar> {
  return std::visit(
      [&](auto &x) -> std::optional<Scalar> {
        using Ty = typename std::decay<decltype(x)>::type;
        if constexpr (std::is_same_v<Ty, Scalar>) {
          return {x};
        }
        if constexpr (evaluate::FoldableTrait<Ty>) {
          auto c{x.Fold(context)};
          if (c.has_value()) {
            u_ = *c;
            return c;
          }
        }
        return {};
      },
      u_);
}

template<int KIND>
auto RealExpr<KIND>::Fold(FoldingContext &context) -> std::optional<Scalar> {
  return {};  // TODO
}

template<int KIND>
auto ComplexExpr<KIND>::Fold(FoldingContext &context) -> std::optional<Scalar> {
  return {};  // TODO
}

template<int KIND>
auto CharacterExpr<KIND>::Fold(FoldingContext &context)
    -> std::optional<Scalar> {
  return {};  // TODO
}

std::optional<bool> LogicalExpr::Fold(FoldingContext &context) {
  return {};  // TODO and comparisons too
}

std::optional<GenericScalar> GenericExpr::ScalarValue() const {
  return std::visit(
      [](const auto &x) -> std::optional<GenericScalar> {
        if (auto c{x.ScalarValue()}) {
          return {GenericScalar{std::move(*c)}};
        }
        return {};
      },
      u);
}

template<Category CAT>
auto CategoryExpr<CAT>::ScalarValue() const -> std::optional<Scalar> {
  return std::visit(
      [](const auto &x) -> std::optional<Scalar> {
        if (auto c{x.ScalarValue()}) {
          return {Scalar{std::move(*c)}};
        }
        return {};
      },
      u);
}

template<Category CAT>
auto CategoryExpr<CAT>::Fold(FoldingContext &context) -> std::optional<Scalar> {
  return std::visit(
      [&](auto &x) -> std::optional<Scalar> {
        if (auto c{x.Fold(context)}) {
          return {Scalar{std::move(*c)}};
        }
        return {};
      },
      u);
}

std::optional<GenericScalar> GenericExpr::Fold(FoldingContext &context) {
  return std::visit(
      [&](auto &x) -> std::optional<GenericScalar> {
        if (auto c{x.Fold(context)}) {
          return {GenericScalar{std::move(*c)}};
        }
        return {};
      },
      u);
}

template struct CategoryExpr<Category::Integer>;
template struct CategoryExpr<Category::Real>;
template struct CategoryExpr<Category::Complex>;
template struct CategoryExpr<Category::Character>;

template class Expr<Category::Integer, 1>;
template class Expr<Category::Integer, 2>;
template class Expr<Category::Integer, 4>;
template class Expr<Category::Integer, 8>;
template class Expr<Category::Integer, 16>;
template class Expr<Category::Real, 2>;
template class Expr<Category::Real, 4>;
template class Expr<Category::Real, 8>;
template class Expr<Category::Real, 10>;
template class Expr<Category::Real, 16>;
template class Expr<Category::Complex, 2>;
template class Expr<Category::Complex, 4>;
template class Expr<Category::Complex, 8>;
template class Expr<Category::Complex, 10>;
template class Expr<Category::Complex, 16>;
template class Expr<Category::Character, 1>;
template class Expr<Category::Logical, 1>;

template struct Comparison<IntegerExpr<1>>;
template struct Comparison<IntegerExpr<2>>;
template struct Comparison<IntegerExpr<4>>;
template struct Comparison<IntegerExpr<8>>;
template struct Comparison<IntegerExpr<16>>;
template struct Comparison<RealExpr<2>>;
template struct Comparison<RealExpr<4>>;
template struct Comparison<RealExpr<8>>;
template struct Comparison<RealExpr<10>>;
template struct Comparison<RealExpr<16>>;
template struct Comparison<ComplexExpr<2>>;
template struct Comparison<ComplexExpr<4>>;
template struct Comparison<ComplexExpr<8>>;
template struct Comparison<ComplexExpr<10>>;
template struct Comparison<ComplexExpr<16>>;
template struct Comparison<CharacterExpr<1>>;
}  // namespace Fortran::evaluate
