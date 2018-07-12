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

template<typename A>
std::ostream &Unary<A>::Dump(std::ostream &o, const char *opr) const {
  return operand().Dump(o << opr) << ')';
}

template<typename A, typename B>
std::ostream &Binary<A, B>::Dump(
    std::ostream &o, const char *opr, const char *before) const {
  return right().Dump(left().Dump(o << before) << opr) << ')';
}

template<int KIND>
std::ostream &Expr<Category::Integer, KIND>::Dump(std::ostream &o) const {
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

template<int KIND>
std::ostream &Expr<Category::Real, KIND>::Dump(std::ostream &o) const {
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
std::ostream &Expr<Category::Complex, KIND>::Dump(std::ostream &o) const {
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
std::ostream &Expr<Category::Character, KIND>::Dump(std::ostream &o) const {
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

std::ostream &Expr<Category::Logical, 1>::Dump(std::ostream &o) const {
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

template<int KIND>
SubscriptIntegerExpr Expr<Category::Character, KIND>::LEN() const {
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

template<int KIND>
std::optional<typename Expr<Category::Integer, KIND>::Constant>
Expr<Category::Integer, KIND>::ConstantValue() const {
  if (auto c{std::get_if<Constant>(&u_)}) {
    return {*c};
  }
  return {};
}

template<int KIND>
void Expr<Category::Integer, KIND>::Fold(FoldingContext &context) {
  std::visit(common::visitors{[&](Parentheses &p) {
                                p.operand().Fold(context);
                                if (auto c{p.operand().ConstantValue()}) {
                                  u_ = std::move(*c);
                                }
                              },
                 [&](Negate &n) {
                   n.operand().Fold(context);
                   if (auto c{n.operand().ConstantValue()}) {
                     auto negated{c->Negate()};
                     if (negated.overflow && context.messages != nullptr) {
                       context.messages->Say(
                           context.at, "integer negation overflowed"_en_US);
                     }
                     u_ = std::move(negated.value);
                   }
                 },
                 [&](Add &a) {
                   a.left().Fold(context);
                   a.right().Fold(context);
                   if (auto xc{a.left().ConstantValue()}) {
                     if (auto yc{a.right().ConstantValue()}) {
                       auto sum{xc->AddSigned(*yc)};
                       if (sum.overflow && context.messages != nullptr) {
                         context.messages->Say(
                             context.at, "integer addition overflowed"_en_US);
                       }
                       u_ = std::move(sum.value);
                     }
                   }
                 },
                 [&](Multiply &a) {
                   a.left().Fold(context);
                   a.right().Fold(context);
                   if (auto xc{a.left().ConstantValue()}) {
                     if (auto yc{a.right().ConstantValue()}) {
                       auto product{xc->MultiplySigned(*yc)};
                       if (product.SignedMultiplicationOverflowed() &&
                           context.messages != nullptr) {
                         context.messages->Say(context.at,
                             "integer multiplication overflowed"_en_US);
                       }
                       u_ = std::move(product.lower);
                     }
                   }
                 },
                 [&](Bin &b) {
                   b.left().Fold(context);
                   b.right().Fold(context);
                 },
                 [&](const auto &) {  // TODO: more
                 }},
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
