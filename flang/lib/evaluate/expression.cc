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
        x.Dump(o << '(' << Ty::Dump() << ' ') << ')';
      },
      u);
  return o;
}

std::ostream &AnyIntegerExpr::Dump(std::ostream &o) const {
  return DumpExprWithType(o, u);
}

std::ostream &AnyRealExpr::Dump(std::ostream &o) const {
  return DumpExprWithType(o, u);
}

template<typename... A>
std::ostream &DumpExpr(std::ostream &o, const std::variant<A...> &u) {
  std::visit([&](const auto &x) { x.Dump(o); }, u);
  return o;
}

std::ostream &AnyCharacterExpr::Dump(std::ostream &o) const {
  return DumpExpr(o, u);
}

std::ostream &AnyComplexExpr::Dump(std::ostream &o) const {
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
std::ostream &IntegerExpr<KIND>::Dump(std::ostream &o) const {
  std::visit(
      common::visitors{[&](const Constant &n) { o << n.SignedDecimal(); },
          [&](const Convert<AnyIntegerExpr> &j) { j.x->Dump(o); },
          [&](const Convert<AnyRealExpr> &a) { a.x->Dump(o); },
          [&](const Parentheses &p) { p.Dump(o, "("); },
          [&](const Negate &n) { n.Dump(o, "(-"); },
          [&](const Add &a) { a.Dump(o, "+"); },
          [&](const Subtract &s) { s.Dump(o, "-"); },
          [&](const Multiply &m) { m.Dump(o, "*"); },
          [&](const Divide &d) { d.Dump(o, "/"); },
          [&](const Power &p) { p.Dump(o, "**"); }},
      u);
  return o;
}

template<int KIND> std::ostream &RealExpr<KIND>::Dump(std::ostream &o) const {
  std::visit(
      common::visitors{[&](const Constant &n) { o << n.DumpHexadecimal(); },
          [&](const Convert<AnyIntegerExpr> &j) { j.x->Dump(o); },
          [&](const Convert<AnyRealExpr> &a) { a.x->Dump(o); },
          [&](const Parentheses &p) { p.Dump(o, "("); },
          [&](const Negate &n) { n.Dump(o, "(-"); },
          [&](const Add &a) { a.Dump(o, "+"); },
          [&](const Subtract &s) { s.Dump(o, "-"); },
          [&](const Multiply &m) { m.Dump(o, "*"); },
          [&](const Divide &d) { d.Dump(o, "/"); },
          [&](const Power &p) { p.Dump(o, "**"); },
          [&](const IntPower &p) { p.Dump(o, "**"); },
          [&](const RealPart &z) { z.Dump(o, "REAL("); },
          [&](const AIMAG &p) { p.Dump(o, "AIMAG("); }},
      u);
  return o;
}

template<int KIND>
std::ostream &ComplexExpr<KIND>::Dump(std::ostream &o) const {
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
std::ostream &CharacterExpr<KIND>::Dump(std::ostream &o) const {
  std::visit(common::visitors{[&](const Constant &s) { o << '"' << s << '"'; },
                 [&](const Concat &c) { c.y->Dump(c.x->Dump(o) << "//"); }},
      u);
  return o;
}

template<typename T> std::ostream &Comparison<T>::Dump(std::ostream &o) const {
  std::visit(common::visitors{[&](const LT &c) { c.Dump(o, ".LT."); },
                 [&](const LE &c) { c.Dump(o, ".LE."); },
                 [&](const EQ &c) { c.Dump(o, ".EQ."); },
                 [&](const NE &c) { c.Dump(o, ".NE."); },
                 [&](const GE &c) { c.Dump(o, ".GE."); },
                 [&](const GT &c) { c.Dump(o, ".GT."); }},
      u);
  return o;
}

template<int KIND>
std::ostream &Comparison<ComplexExpr<KIND>>::Dump(std::ostream &o) const {
  std::visit(common::visitors{[&](const EQ &c) { c.Dump(o, ".EQ."); },
                 [&](const NE &c) { c.Dump(o, ".NE."); }},
      u);
  return o;
}

std::ostream &IntegerComparison::Dump(std::ostream &o) const {
  std::visit([&](const auto &c) { c.Dump(o); }, u);
  return o;
}

std::ostream &RealComparison::Dump(std::ostream &o) const {
  std::visit([&](const auto &c) { c.Dump(o); }, u);
  return o;
}

std::ostream &ComplexComparison::Dump(std::ostream &o) const {
  std::visit([&](const auto &c) { c.Dump(o); }, u);
  return o;
}

std::ostream &CharacterComparison::Dump(std::ostream &o) const {
  std::visit([&](const auto &c) { c.Dump(o); }, u);
  return o;
}

std::ostream &LogicalExpr::Dump(std::ostream &o) const {
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

template<int KIND> void IntegerExpr<KIND>::Fold(FoldingContext &context) {
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

template struct IntegerExpr<1>;
template struct IntegerExpr<2>;
template struct IntegerExpr<4>;
template struct IntegerExpr<8>;
template struct IntegerExpr<16>;
template struct RealExpr<2>;
template struct RealExpr<4>;
template struct RealExpr<8>;
template struct RealExpr<10>;
template struct RealExpr<16>;
template struct ComplexExpr<2>;
template struct ComplexExpr<4>;
template struct ComplexExpr<8>;
template struct ComplexExpr<10>;
template struct ComplexExpr<16>;
template struct CharacterExpr<1>;
}  // namespace Fortran::evaluate
