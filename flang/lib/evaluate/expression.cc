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

template<typename A, typename CONST>
std::ostream &Unary<A, CONST>::Dump(std::ostream &o, const char *opr) const {
  return operand().Dump(o << opr) << ')';
}

template<typename A, typename B, typename CONST>
std::ostream &Binary<A, B, CONST>::Dump(
    std::ostream &o, const char *opr, const char *before) const {
  return right().Dump(left().Dump(o << before) << opr) << ')';
}

template<int KIND>
std::ostream &IntegerExpr<KIND>::Dump(std::ostream &o) const {
  std::visit(
      common::visitors{[&](const Constant &n) { o << n.SignedDecimal(); },
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
      common::visitors{[&](const Constant &n) { o << n.DumpHexadecimal(); },
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
      common::visitors{[&](const Constant &n) { o << n.DumpHexadecimal(); },
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
  std::visit(common::visitors{[&](const Constant &s) {
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

template<int KIND> SubscriptIntegerExpr CharacterExpr<KIND>::LEN() const {
  return std::visit(
      common::visitors{
          [](const Constant &c) { return SubscriptIntegerExpr{c.size()}; },
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

template<typename A, typename CONST>
std::optional<CONST> Unary<A, CONST>::Fold(FoldingContext &context) {
  operand_->Fold(context);
  return {};
}

template<typename A, typename B, typename CONST>
std::optional<CONST> Binary<A, B, CONST>::Fold(FoldingContext &context) {
  left_->Fold(context);
  right_->Fold(context);
  return {};
}

template<int KIND>
std::optional<typename IntegerExpr<KIND>::Constant>
IntegerExpr<KIND>::ConstantValue() const {
  if (auto c{std::get_if<Constant>(&u_)}) {
    return {*c};
  }
  return {};
}

template<int KIND>
std::optional<typename IntegerExpr<KIND>::Constant>
IntegerExpr<KIND>::ConvertInteger::Fold(FoldingContext &context) {
  return std::visit(
      [&](auto &x) -> std::optional<typename IntegerExpr<KIND>::Constant> {
        if (auto c{x.Fold(context)}) {
          auto converted{Constant::ConvertSigned(*c)};
          if (converted.overflow && context.messages != nullptr) {
            context.messages->Say(
                context.at, "integer conversion overflowed"_en_US);
          }
          return {std::move(converted.value)};
        }
        // g++ 8.1.0 choked on the legal "return {};" that should be here,
        // saying that it may be used uninitialized.
        std::optional<typename IntegerExpr<KIND>::Constant> result;
        return std::move(result);
      },
      this->operand().u);
}

template<int KIND>
std::optional<typename IntegerExpr<KIND>::Constant>
IntegerExpr<KIND>::Negate::Fold(FoldingContext &context) {
  if (auto c{this->operand().Fold(context)}) {
    auto negated{c->Negate()};
    if (negated.overflow && context.messages != nullptr) {
      context.messages->Say(context.at, "integer negation overflowed"_en_US);
    }
    return {std::move(negated.value)};
  }
  return {};
}

template<int KIND>
std::optional<typename IntegerExpr<KIND>::Constant>
IntegerExpr<KIND>::Add::Fold(FoldingContext &context) {
  auto lc{this->left().Fold(context)};
  auto rc{this->right().Fold(context)};
  if (lc && rc) {
    auto sum{lc->AddSigned(*rc)};
    if (sum.overflow && context.messages != nullptr) {
      context.messages->Say(context.at, "integer addition overflowed"_en_US);
    }
    return {std::move(sum.value)};
  }
  return {};
}

template<int KIND>
std::optional<typename IntegerExpr<KIND>::Constant>
IntegerExpr<KIND>::Subtract::Fold(FoldingContext &context) {
  auto lc{this->left().Fold(context)};
  auto rc{this->right().Fold(context)};
  if (lc && rc) {
    auto diff{lc->SubtractSigned(*rc)};
    if (diff.overflow && context.messages != nullptr) {
      context.messages->Say(context.at, "integer subtraction overflowed"_en_US);
    }
    return {std::move(diff.value)};
  }
  return {};
}

template<int KIND>
std::optional<typename IntegerExpr<KIND>::Constant>
IntegerExpr<KIND>::Multiply::Fold(FoldingContext &context) {
  auto lc{this->left().Fold(context)};
  auto rc{this->right().Fold(context)};
  if (lc && rc) {
    auto product{lc->MultiplySigned(*rc)};
    if (product.SignedMultiplicationOverflowed() &&
        context.messages != nullptr) {
      context.messages->Say(
          context.at, "integer multiplication overflowed"_en_US);
    }
    return {std::move(product.lower)};
  }
  return {};
}

template<int KIND>
std::optional<typename IntegerExpr<KIND>::Constant>
IntegerExpr<KIND>::Divide::Fold(FoldingContext &context) {
  auto lc{this->left().Fold(context)};
  auto rc{this->right().Fold(context)};
  if (lc && rc) {
    auto qr{lc->DivideSigned(*rc)};
    if (context.messages != nullptr) {
      if (qr.divisionByZero) {
        context.messages->Say(context.at, "integer division by zero"_en_US);
      } else if (qr.overflow) {
        context.messages->Say(context.at, "integer division overflowed"_en_US);
      }
    }
    return {std::move(qr.quotient)};
  }
  return {};
}

template<int KIND>
std::optional<typename IntegerExpr<KIND>::Constant>
IntegerExpr<KIND>::Power::Fold(FoldingContext &context) {
  auto lc{this->left().Fold(context)};
  auto rc{this->right().Fold(context)};
  if (lc && rc) {
    typename Constant::PowerWithErrors power{lc->Power(*rc)};
    if (context.messages != nullptr) {
      if (power.divisionByZero) {
        context.messages->Say(context.at, "zero to negative power"_en_US);
      } else if (power.overflow) {
        context.messages->Say(context.at, "integer power overflowed"_en_US);
      } else if (power.zeroToZero) {
        context.messages->Say(context.at, "integer 0**0"_en_US);
      }
    }
    return {std::move(power.power)};
  }
  return {};
}

template<int KIND>
std::optional<typename IntegerExpr<KIND>::Constant>
IntegerExpr<KIND>::Max::Fold(FoldingContext &context) {
  auto lc{this->left().Fold(context)};
  auto rc{this->right().Fold(context)};
  if (lc && rc) {
    if (lc->CompareSigned(*rc) == Ordering::Greater) {
      return lc;
    }
    return rc;
  }
  return {};
}

template<int KIND>
std::optional<typename IntegerExpr<KIND>::Constant>
IntegerExpr<KIND>::Min::Fold(FoldingContext &context) {
  auto lc{this->left().Fold(context)};
  auto rc{this->right().Fold(context)};
  if (lc && rc) {
    if (lc->CompareSigned(*rc) == Ordering::Less) {
      return lc;
    }
    return rc;
  }
  return {};
}

template<int KIND>
std::optional<typename IntegerExpr<KIND>::Constant> IntegerExpr<KIND>::Fold(
    FoldingContext &context) {
  return std::visit(
      [&](auto &x) -> std::optional<Constant> {
        using Ty = typename std::decay<decltype(x)>::type;
        if constexpr (std::is_same_v<Ty, Constant>) {
          return {x};
        }
        if constexpr (std::is_base_of_v<Un, Ty> || std::is_base_of_v<Bin, Ty>) {
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
