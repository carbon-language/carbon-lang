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

std::ostream &IntegerOperand::Dump(std::ostream &o) const {
  std::visit([&](const auto &x) { DumpExprWithType(o, *x); }, u);
  return o;
}

std::ostream &RealOperand::Dump(std::ostream &o) const {
  std::visit([&](const auto &x) { DumpExprWithType(o, *x); }, u);
  return o;
}

std::ostream &CharacterOperand::Dump(std::ostream &o) const {
  std::visit([&](const auto &x) { x->Dump(o); }, u);
  return o;
}

std::ostream &ConversionOperand::Dump(std::ostream &o) const {
  std::visit([&](const auto &x) { x.Dump(o); }, u);
  return o;
}

template<int KIND> std::ostream &IntExpr<KIND>::Dump(std::ostream &o) const {
  std::visit(
      common::visitors{[&](const Constant &n) { o << n.SignedDecimal(); },
          [&](const Convert &c) { c.x.Dump(o); },
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
          [&](const Convert &c) { c.x.Dump(o); },
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
            p.y.Dump(p.x->Dump(o << '(') << "**") << ')';
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
            p.y.Dump(p.x->Dump(o << '(') << "**") << ')';
          },
          [&](const CMPLX &c) {
            c.im->Dump(c.re->Dump(o << '(') << ',') << ')';
          }},
      u);
  return o;
}

template<int KIND> typename CharExpr<KIND>::Length CharExpr<KIND>::LEN() const {
  return std::visit(
      common::visitors{
          [](const std::string &str) { return Length{str.size()}; },
          [](const Concat &c) { return Length{c.x->LEN() + c.y->LEN()}; }},
      u);
}

template<int KIND> std::ostream &CharExpr<KIND>::Dump(std::ostream &o) const {
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
std::ostream &Comparison<Type<Category::Complex, KIND>>::Dump(
    std::ostream &o) const {
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

template struct Expression<Type<Category::Integer, 1>>;
template struct Expression<Type<Category::Integer, 2>>;
template struct Expression<Type<Category::Integer, 4>>;
template struct Expression<Type<Category::Integer, 8>>;
template struct Expression<Type<Category::Integer, 16>>;
template struct Expression<Type<Category::Real, 2>>;
template struct Expression<Type<Category::Real, 4>>;
template struct Expression<Type<Category::Real, 8>>;
template struct Expression<Type<Category::Real, 10>>;
template struct Expression<Type<Category::Real, 16>>;
template struct Expression<Type<Category::Complex, 2>>;
template struct Expression<Type<Category::Complex, 4>>;
template struct Expression<Type<Category::Complex, 8>>;
template struct Expression<Type<Category::Complex, 10>>;
template struct Expression<Type<Category::Complex, 16>>;
template struct Expression<Type<Category::Logical, 1>>;
template struct Expression<Type<Category::Character, 1>>;
}  // namespace Fortran::evaluate
