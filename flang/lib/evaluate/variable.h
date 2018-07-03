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

#ifndef FORTRAN_EVALUATE_VARIABLE_H_
#define FORTRAN_EVALUATE_VARIABLE_H_

#include "expression.h"
#include "traverse.h"
#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace Fortran::evaluate {

struct DataRef;
struct Variable;
struct ActualArg;
struct Label;  // TODO

using semantics::Symbol;

struct Component {
  Component(const Symbol &c, std::unique_ptr<DataRef> &&b)
    : sym{c}, base{std::move(b)} {}
  const Symbol &sym;
  template<typename V> void DefaultTraverse(V &v) { v(base); }
  std::unique_ptr<DataRef> base;
};

using SubscriptExpr = DefaultIntExpr;

struct Triplet {
  Triplet(std::optional<SubscriptExpr> &&l, std::optional<SubscriptExpr> &&u,
      std::optional<SubscriptExpr> &&s)
    : lower{std::move(l)}, upper{std::move(u)}, stride{std::move(s)} {}
  template<typename V> void DefaultTraverse(V &v) {
    v(lower);
    v(upper);
    v(stride);
  }
  std::optional<SubscriptExpr> lower, upper, stride;
};

struct Subscript {
  Subscript() = delete;
  explicit Subscript(SubscriptExpr &&s) : u{std::move(s)} {}
  explicit Subscript(Triplet &&t) : u{std::move(t)} {}
  template<typename V> void DefaultTraverse(V &v) { v(u); }
  std::variant<SubscriptExpr, Triplet> u;
};

struct ArrayRef {
  ArrayRef() = delete;
  ArrayRef(const Symbol &n, std::vector<Subscript> &&s)
    : u{n}, subscript{std::move(ss)} {}
  ArrayRef(Component &&c, std::vector<Subscript> &&s)
    : u{std::move(c)}, subscript{std::move(ss)} {}
  template<typename V> void DefaultTraverse(V &v) {
    v(u);
    v(subscript);
  }
  std::variant<const Symbol &, Component> u;
  std::vector<Subscript> subscript;
};

struct CoarrayRef {
  CoarrayRef() = delete;
  CoarrayRef(const Symbol &n, std::vector<SubscriptExpr> &&s)
    : u{n}, cosubscript{std::move(s)} {}
  CoarrayRef(Component &&c, std::vector<SubscriptExpr> &&s)
    : u{std::move(c)}, cosubscript{std::move(s)} {}
  CoarrayRef(ArrayRef &&a, std::vector<SubscriptExpr> &&s)
    : u{std::move(a)}, cosubscript{std::move(s)} {}
  template<typename V> void DefaultTraverse(V &v) {
    v(u);
    v(cosubscript);
    v(stat);
    v(team);
    v(teamNumber);
  }
  std::variant<const Symbol &, Component, ArrayRef> u;
  std::vector<SubscriptExpr> cosubscript;
  std::unique_ptr<Variable> stat, team, teamNumber;  // nullable
};

struct DataRef {
  DataRef() = delete;
  explicit DataRef(const Symbol &n) : u{n} {}
  explicit DataRef(Component &&c) : u{std::move(c)} {}
  explicit DataRef(ArrayRef &&a) : u{std::move(a)} {}
  explicit DataRef(CoarrayRef &&c) : u{std::move(c)} {}
  template<typename V> void DefaultTraverse(V &v) { v(u); }
  std::variant<const Symbol &, Component, ArrayRef, CoarrayRef> u;
};

struct Substring {
  Substring() = delete;
  Substring(DataRef &&d, std::optional<SubscriptExpr> &&f,
      std::optional<SubscriptExpr> &&l)
    : u{std::move(d)}, first{std::move(f)}, last{std::move(l)} {}
  Substring(std::string &&s, std::optional<SubscriptExpr> &&f,
      std::optional<SubscriptExpr> &&l)
    : u{std::move(s)}, first{std::move(f)}, last{std::move(l)} {}
  template<typename V> void DefaultTraverse(V &v) {
    v(u);
    v(first);
    v(last);
  }
  std::variant<DataRef, std::string> u;
  std::optional<SubscriptExpr> first, last;
};

struct ComplexPart {
  enum class Part { RE, IM };
  ComplexPart(DataRef &&z, Part p) : complex{std::move(z)}, part{p} {}
  template<typename V> void DefaultTraverse(V &v) { v(complex); }
  DataRef complex;
  Part part;
};

struct Designator {
  Designator() = delete;
  explicit Designator(DataRef &&d) : u{std::move(d)} {}
  explicit Designator(Substring &&s) : u{std::move(s)} {}
  explicit Designator(ComplexPart &&c) : u{std::move(c)} {}
  template<typename V> void DefaultTraverse(V &v) { v(u); }
  std::variant<DataRef, Substring, ComplexPart> u;
};

struct ProcedureDesignator {
  ProcedureDesignator() = delete;
  ProcedureDesignator(std::unique_ptr<Variable> &&v, const Symbol &n)
    : u{std::move(v)}, sym{n} {}
  ProcedureDesignator(DataRef &&d, const Symbol &n) : u{std::move(d)}, sym{n} {}
  template<typename V> void DefaultTraverse(V &v) { v(u); }
  std::variant<std::unique_ptr<Variable>, DataRef> u;
  const Symbol &sym;
};

struct ProcedureRef {
  ProcedureRef() = delete;
  ProcedureRef(
      ProcedureDesignator &&p, std::vector<std::unique_ptr<ActualArg>> &&a)
    : proc{std::move(p)}, arg{std::move(a)} {}
  template<typename V> void DefaultTraverse(V &v) {
    v(proc);
    v(arg);
  }
  ProcedureDesignator proc;
  std::vector<std::unique_ptr<ActualArg>> arg;  // nullable
};

struct Variable {
  Variable() = delete;
  explicit Variable(Designator &&d) : u{std::move(u)} {}
  explicit Variable(ProcedureRef &&p) : u{std::move(p)} {}
  template<typename V> void DefaultTraverse(V &v) { v(u); }
  std::variant<Designator, ProcedureRef> u;
};

struct ActualArg {
  ActualArg() = delete;
  explicit ActualArg(AnyExpr &&x) : u{std::move(x)} {}
  explicit ActualArg(Variable &&x) : u{std::move(x)} {}
  explicit ActualArg(const Label &l) : u{l} {}
  template<typename V> void DefaultTraverse(V &v) { v(u); }
  std::variant<AnyExpr, Variable, const Label &> u;
};
}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_VARIABLE_H_
