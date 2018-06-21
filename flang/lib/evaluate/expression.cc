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

template<int KIND> std::ostream &IntExpr<KIND>::Dump(std::ostream &o) const {
  std::visit(
      common::visitors{[&](const Constant &n) { o << n.SignedDecimal(); },
          [&](const Convert &c) {
            o << "convert(";
            std::visit(
                [&](const auto &x) {  // x is IntegerOperand or RealOperand
                  std::visit(
                      [&](const auto &y) {  // y is ExprOperand<Type<C,K>>
                        using Expr =
                            typename std::remove_reference<decltype(*y)>::type;
                        using Ty = typename Expr::Result;
                        y->Dump(o << Ty::Dump() << ' ');
                      },
                      x.u);
                },
                c.u);
            o << ')';
          },
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
          [&](const auto &) { o << "TODO"; }},
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

// TODO dump logical and comparison

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
