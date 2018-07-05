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
#include "../common/idioms.h"
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
std::ostream &AnyKindExpr<CAT>::Dump(std::ostream &o) const {
  return DumpExpr(o, u);
}

template<Category CAT>
std::ostream &AnyKindComparison<CAT>::Dump(std::ostream &o) const {
  return DumpExpr(o, u);
}

std::ostream &AnyExpr::Dump(std::ostream &o) const { return DumpExpr(o, u); }

template<typename A>
std::ostream &Unary<A>::Dump(std::ostream &o, const char *opr) const {
  return x->Dump(o << opr) << ')';
}

template<typename A, typename B>
std::ostream &Binary<A, B>::Dump(std::ostream &o, const char *opr) const {
  return y->Dump(x->Dump(o << '(') << opr) << ')';
}

template<int KIND>
std::ostream &Expr<Category::Integer, KIND>::Dump(std::ostream &o) const {
  std::visit(
      common::visitors{[&](const Constant &n) { o << n.SignedDecimal(); },
          [&](const Parentheses &p) { p.Dump(o, "("); },
          [&](const Negate &n) { n.Dump(o, "(-"); },
          [&](const Add &a) { a.Dump(o, "+"); },
          [&](const Subtract &s) { s.Dump(o, "-"); },
          [&](const Multiply &m) { m.Dump(o, "*"); },
          [&](const Divide &d) { d.Dump(o, "/"); },
          [&](const Power &p) { p.Dump(o, "**"); },
          [&](const auto &convert) { DumpExprWithType(o, convert.x->u); }},
      u);
  return o;
}

template<int KIND>
std::ostream &Expr<Category::Real, KIND>::Dump(std::ostream &o) const {
  std::visit(
      common::visitors{[&](const Constant &n) { o << n.DumpHexadecimal(); },
          [&](const Parentheses &p) { p.Dump(o, "("); },
          [&](const Negate &n) { n.Dump(o, "(-"); },
          [&](const Add &a) { a.Dump(o, "+"); },
          [&](const Subtract &s) { s.Dump(o, "-"); },
          [&](const Multiply &m) { m.Dump(o, "*"); },
          [&](const Divide &d) { d.Dump(o, "/"); },
          [&](const Power &p) { p.Dump(o, "**"); },
          [&](const IntPower &p) { p.Dump(o, "**"); },
          [&](const RealPart &z) { z.Dump(o, "REAL("); },
          [&](const AIMAG &p) { p.Dump(o, "AIMAG("); },
          [&](const auto &convert) { DumpExprWithType(o, convert.x->u); }},
      u);
  return o;
}

template<int KIND>
std::ostream &Expr<Category::Complex, KIND>::Dump(std::ostream &o) const {
  std::visit(
      common::visitors{[&](const Constant &n) { o << n.DumpHexadecimal(); },
          [&](const Parentheses &p) { p.Dump(o, "("); },
          [&](const Negate &n) { n.Dump(o, "(-"); },
          [&](const Add &a) { a.Dump(o, "+"); },
          [&](const Subtract &s) { s.Dump(o, "-"); },
          [&](const Multiply &m) { m.Dump(o, "*"); },
          [&](const Divide &d) { d.Dump(o, "/"); },
          [&](const Power &p) { p.Dump(o, "**"); },
          [&](const IntPower &p) { p.Dump(o, "**"); },
          [&](const CMPLX &c) { c.Dump(o, ","); }},
      u);
  return o;
}

template<int KIND>
std::ostream &Expr<Category::Character, KIND>::Dump(std::ostream &o) const {
  std::visit(common::visitors{[&](const Constant &s) { o << '"' << s << '"'; },
                 [&](const Concat &c) { c.y->Dump(c.x->Dump(o) << "//"); }},
      u);
  return o;
}

template<typename A> std::ostream &Comparison<A>::Dump(std::ostream &o) const {
  using Ty = typename A::Result;
  o << '(' << Ty::Dump() << "::";
  this->x->Dump(o);
  o << '.' << EnumToString(this->opr) << '.';
  return this->y->Dump(o) << ')';
}

std::ostream &Expr<Category::Logical, 1>::Dump(std::ostream &o) const {
  std::visit(
      common::visitors{[&](const bool &tf) { o << (tf ? ".T." : ".F."); },
          [&](const Not &n) { n.Dump(o, "(.NOT."); },
          [&](const And &a) { a.Dump(o, ".AND."); },
          [&](const Or &a) { a.Dump(o, ".OR."); },
          [&](const Eqv &a) { a.Dump(o, ".EQV."); },
          [&](const Neqv &a) { a.Dump(o, ".NEQV."); },
          [&](const auto &comparison) { comparison.Dump(o); }},
      u);
  return o;
}

template<int KIND>
void Expr<Category::Integer, KIND>::Fold(FoldingContext &context) {
  std::visit(common::visitors{[&](const Parentheses &p) {
                                p.x->Fold(context);
                                if (auto c{std::get_if<Constant>(&p.x->u)}) {
                                  u = std::move(*c);
                                }
                              },
                 [&](const Negate &n) {
                   n.x->Fold(context);
                   if (auto c{std::get_if<Constant>(&n.x->u)}) {
                     auto negated{c->Negate()};
                     if (negated.overflow && context.messages != nullptr) {
                       context.messages->Say(
                           context.at, "integer negation overflowed"_en_US);
                     }
                     u = std::move(negated.value);
                   }
                 },
                 [&](const Add &a) {
                   a.x->Fold(context);
                   a.y->Fold(context);
                   if (auto xc{std::get_if<Constant>(&a.x->u)}) {
                     if (auto yc{std::get_if<Constant>(&a.y->u)}) {
                       auto sum{xc->AddSigned(*yc)};
                       if (sum.overflow && context.messages != nullptr) {
                         context.messages->Say(
                             context.at, "integer addition overflowed"_en_US);
                       }
                       u = std::move(sum.value);
                     }
                   }
                 },
                 [&](const Multiply &a) {
                   a.x->Fold(context);
                   a.y->Fold(context);
                   if (auto xc{std::get_if<Constant>(&a.x->u)}) {
                     if (auto yc{std::get_if<Constant>(&a.y->u)}) {
                       auto product{xc->MultiplySigned(*yc)};
                       if (product.SignedMultiplicationOverflowed() &&
                           context.messages != nullptr) {
                         context.messages->Say(context.at,
                             "integer multiplication overflowed"_en_US);
                       }
                       u = std::move(product.lower);
                     }
                   }
                 },
                 [&](const Bin &b) {
                   b.x->Fold(context);
                   b.y->Fold(context);
                 },
                 [&](const auto &) {  // TODO: more
                 }},
      u);
}

template<int KIND>
typename CharacterExpr<KIND>::LengthExpr CharacterExpr<KIND>::LEN() const {
  return std::visit(
      common::visitors{
          [](const std::string &str) { return LengthExpr{str.size()}; },
          [](const Concat &c) {
            return LengthExpr{LengthExpr::Add{c.x->LEN(), c.y->LEN()}};
          }},
      u);
}

template struct Expr<Category::Integer, 1>;
template struct Expr<Category::Integer, 2>;
template struct Expr<Category::Integer, 4>;
template struct Expr<Category::Integer, 8>;
template struct Expr<Category::Integer, 16>;
template struct Expr<Category::Real, 2>;
template struct Expr<Category::Real, 4>;
template struct Expr<Category::Real, 8>;
template struct Expr<Category::Real, 10>;
template struct Expr<Category::Real, 16>;
template struct Expr<Category::Complex, 2>;
template struct Expr<Category::Complex, 4>;
template struct Expr<Category::Complex, 8>;
template struct Expr<Category::Complex, 10>;
template struct Expr<Category::Complex, 16>;
template struct Expr<Category::Character, 1>;
template struct Expr<Category::Logical, 1>;

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
