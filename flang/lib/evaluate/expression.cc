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

namespace Fortran::evaluate {

template<typename A>
std::ostream &DumpExprWithType(std::ostream &o, const A &x) {
  using Ty = typename A::Result;
  return x.Dump(o << '(' << Ty::Dump() << ' ') << ')';
}

std::ostream &AnyIntegerExpr::Dump(std::ostream &o) const {
  std::visit([&](const auto &x) { DumpExprWithType(o, x); }, u);
  return o;
}

std::ostream &AnyRealExpr::Dump(std::ostream &o) const {
  std::visit([&](const auto &x) { DumpExprWithType(o, x); }, u);
  return o;
}

std::ostream &AnyCharacterExpr::Dump(std::ostream &o) const {
  std::visit([&](const auto &x) { x.Dump(o); }, u);
  return o;
}

std::ostream &AnyIntegerOrRealExpr::Dump(std::ostream &o) const {
  std::visit([&](const auto &x) { x.Dump(o); }, u);
  return o;
}

template<int KIND>
std::ostream &IntegerExpr<KIND>::Dump(std::ostream &o) const {
  std::visit(
      common::visitors{[&](const Constant &n) { o << n.SignedDecimal(); },
          [&](const Convert &c) { c.x->Dump(o); },
          [&](const Parentheses &p) { p.x->Dump(o << '(') << ')'; },
          [&](const Negate &n) { n.x->Dump(o << "(-") << ')'; },
          [&](const Add &a) { a.y->Dump(a.x->Dump(o << '(') << '+') << ')'; },
          [&](const Subtract &s) {
            s.y->Dump(s.x->Dump(o << '(') << '-') << ')';
          },
          [&](const Multiply &m) {
            m.y->Dump(m.x->Dump(o << '(') << '*') << ')';
          },
          [&](const Divide &d) {
            d.y->Dump(d.x->Dump(o << '(') << '/') << ')';
          },
          [&](const Power &p) {
            p.y->Dump(p.x->Dump(o << '(') << "**") << ')';
          }},
      u);
  return o;
}

template<int KIND> std::ostream &RealExpr<KIND>::Dump(std::ostream &o) const {
  std::visit(
      common::visitors{[&](const Constant &n) { o << n.DumpHexadecimal(); },
          [&](const Convert &c) { c.x->Dump(o); },
          [&](const Parentheses &p) { p.x->Dump(o << '(') << ')'; },
          [&](const Negate &n) { n.x->Dump(o << "(-") << ')'; },
          [&](const Add &a) { a.y->Dump(a.x->Dump(o << '(') << '+') << ')'; },
          [&](const Subtract &s) {
            s.y->Dump(s.x->Dump(o << '(') << '-') << ')';
          },
          [&](const Multiply &m) {
            m.y->Dump(m.x->Dump(o << '(') << '*') << ')';
          },
          [&](const Divide &d) {
            d.y->Dump(d.x->Dump(o << '(') << '/') << ')';
          },
          [&](const Power &p) {
            p.y->Dump(p.x->Dump(o << '(') << "**") << ')';
          },
          [&](const IntPower &p) {
            p.y->Dump(p.x->Dump(o << '(') << "**") << ')';
          },
          [&](const RealPart &z) { z.z->Dump(o << "REAL(") << ')'; },
          [&](const AIMAG &z) { z.z->Dump(o << "AIMAG(") << ')'; }},
      u);
  return o;
}

template<int KIND>
std::ostream &ComplexExpr<KIND>::Dump(std::ostream &o) const {
  std::visit(
      common::visitors{[&](const Constant &n) { o << n.DumpHexadecimal(); },
          [&](const Parentheses &p) { p.x->Dump(o << '(') << ')'; },
          [&](const Negate &n) { n.x->Dump(o << "(-") << ')'; },
          [&](const Add &a) { a.y->Dump(a.x->Dump(o << '(') << '+') << ')'; },
          [&](const Subtract &s) {
            s.y->Dump(s.x->Dump(o << '(') << '-') << ')';
          },
          [&](const Multiply &m) {
            m.y->Dump(m.x->Dump(o << '(') << '*') << ')';
          },
          [&](const Divide &d) {
            d.y->Dump(d.x->Dump(o << '(') << '/') << ')';
          },
          [&](const Power &p) {
            p.y->Dump(p.x->Dump(o << '(') << "**") << ')';
          },
          [&](const IntPower &p) {
            p.y->Dump(p.x->Dump(o << '(') << "**") << ')';
          },
          [&](const CMPLX &c) {
            c.im->Dump(c.re->Dump(o << '(') << ',') << ')';
          }},
      u);
  return o;
}

template<int KIND>
typename CharacterExpr<KIND>::Length CharacterExpr<KIND>::LEN() const {
  return std::visit(
      common::visitors{
          [](const std::string &str) { return Length{str.size()}; },
          [](const Concat &c) { return Length{c.x->LEN() + c.y->LEN()}; }},
      u);
}

template<int KIND>
std::ostream &CharacterExpr<KIND>::Dump(std::ostream &o) const {
  std::visit(common::visitors{[&](const Constant &s) { o << '"' << s << '"'; },
                 [&](const Concat &c) { c.y->Dump(c.x->Dump(o) << "//"); }},
      u);
  return o;
}

template<typename T> std::ostream &Comparison<T>::Dump(std::ostream &o) const {
  std::visit(
      common::visitors{
          [&](const LT &c) { c.y->Dump(c.x->Dump(o << '(') << ".LT.") << ')'; },
          [&](const LE &c) { c.y->Dump(c.x->Dump(o << '(') << ".LE.") << ')'; },
          [&](const EQ &c) { c.y->Dump(c.x->Dump(o << '(') << ".EQ.") << ')'; },
          [&](const NE &c) { c.y->Dump(c.x->Dump(o << '(') << ".NE.") << ')'; },
          [&](const GE &c) { c.y->Dump(c.x->Dump(o << '(') << ".GE.") << ')'; },
          [&](const GT &c) {
            c.y->Dump(c.x->Dump(o << '(') << ".GT.") << ')';
          }},
      u);
  return o;
}

template<int KIND>
std::ostream &Comparison<ComplexExpr<KIND>>::Dump(std::ostream &o) const {
  std::visit(common::visitors{[&](const EQ &c) {
                                c.y->Dump(c.x->Dump(o << '(') << ".EQ.") << ')';
                              },
                 [&](const NE &c) {
                   c.y->Dump(c.x->Dump(o << '(') << ".NE.") << ')';
                 }},
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
          [&](const Not &n) { n.x->Dump(o << "(.NOT.") << ')'; },
          [&](const And &a) {
            a.y->Dump(a.x->Dump(o << '(') << ".AND.") << ')';
          },
          [&](const Or &a) { a.y->Dump(a.x->Dump(o << '(') << ".OR.") << ')'; },
          [&](const Eqv &a) {
            a.y->Dump(a.x->Dump(o << '(') << ".EQV.") << ')';
          },
          [&](const Neqv &a) {
            a.y->Dump(a.x->Dump(o << '(') << ".NEQV.") << ')';
          },
          [&](const auto &comparison) { comparison.Dump(o); }},
      u);
  return o;
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
