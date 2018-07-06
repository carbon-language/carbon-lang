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

#include "common.h"
#include "expression-forward.h"
#include "../common/idioms.h"
#include "../common/indirection.h"
#include "../semantics/symbol.h"
#include <optional>
#include <ostream>
#include <variant>
#include <vector>

namespace Fortran::evaluate {

using semantics::Symbol;

// TODO: Reference sections in the Standard

struct DataRef;
struct Component {
  CLASS_BOILERPLATE(Component)
  Component(const DataRef &b, const Symbol &c) : base{b}, sym{&c} {}
  Component(common::Indirection<DataRef> &&b, const Symbol &c)
    : base{std::move(b)}, sym{&c} {}
  common::Indirection<DataRef> base;
  const Symbol *sym;
};

using SubscriptExpr = common::Indirection<DefaultIntegerExpr>;

struct Triplet {
  CLASS_BOILERPLATE(Triplet)
  Triplet(std::optional<SubscriptExpr> &&, std::optional<SubscriptExpr> &&,
      std::optional<SubscriptExpr> &&);
  std::optional<SubscriptExpr> lower, upper, stride;
};

struct Subscript {
  CLASS_BOILERPLATE(Subscript)
  explicit Subscript(const SubscriptExpr &s) : u{s} {}
  explicit Subscript(SubscriptExpr &&s) : u{std::move(s)} {}
  explicit Subscript(const Triplet &t) : u{t} {}
  explicit Subscript(Triplet &&t) : u{std::move(t)} {}
  std::variant<SubscriptExpr, Triplet> u;
};

struct ArrayRef {
  CLASS_BOILERPLATE(ArrayRef)
  ArrayRef(const Symbol &n, std::vector<Subscript> &&ss)
    : u{&n}, subscript(std::move(ss)) {}
  ArrayRef(Component &&c, std::vector<Subscript> &&ss)
    : u{std::move(c)}, subscript(std::move(ss)) {}
  std::variant<const Symbol *, Component> u;
  std::vector<Subscript> subscript;
};

struct Variable;
struct CoarrayRef {
  CLASS_BOILERPLATE(CoarrayRef)
  CoarrayRef(const Symbol &n, std::vector<SubscriptExpr> &&s)
    : u{&n}, cosubscript(std::move(s)) {}
  CoarrayRef(Component &&c, std::vector<SubscriptExpr> &&s)
    : u{std::move(c)}, cosubscript(std::move(s)) {}
  CoarrayRef(ArrayRef &&a, std::vector<SubscriptExpr> &&s)
    : u{std::move(a)}, cosubscript(std::move(s)) {}
  std::variant<const Symbol *, Component, ArrayRef> u;
  std::vector<SubscriptExpr> cosubscript;
  std::optional<common::Indirection<Variable>> stat, team, teamNumber;
};

struct DataRef {
  CLASS_BOILERPLATE(DataRef)
  explicit DataRef(const Symbol &n) : u{&n} {}
  explicit DataRef(Component &&c) : u{std::move(c)} {}
  explicit DataRef(ArrayRef &&a) : u{std::move(a)} {}
  explicit DataRef(CoarrayRef &&c) : u{std::move(c)} {}
  std::variant<const Symbol *, Component, ArrayRef, CoarrayRef> u;
};

struct Substring {
  CLASS_BOILERPLATE(Substring)
  Substring(DataRef &&d, std::optional<SubscriptExpr> &&f,
      std::optional<SubscriptExpr> &&l)
    : u{std::move(d)}, first{std::move(f)}, last{std::move(l)} {}
  Substring(std::string &&s, std::optional<SubscriptExpr> &&f,
      std::optional<SubscriptExpr> &&l)
    : u{std::move(s)}, first{std::move(f)}, last{std::move(l)} {}
  std::variant<DataRef, std::string> u;
  std::optional<SubscriptExpr> first, last;
};

struct ComplexPart {
  ENUM_CLASS(Part, RE, IM)
  CLASS_BOILERPLATE(ComplexPart)
  ComplexPart(DataRef &&z, Part p) : complex{std::move(z)}, part{p} {}
  DataRef complex;
  Part part;
};

struct Designator {
  CLASS_BOILERPLATE(Designator)
  explicit Designator(DataRef &&d) : u{std::move(d)} {}
  explicit Designator(Substring &&s) : u{std::move(s)} {}
  explicit Designator(ComplexPart &&c) : u{std::move(c)} {}
  std::variant<DataRef, Substring, ComplexPart> u;
};

struct ProcedureDesignator {
  CLASS_BOILERPLATE(ProcedureDesignator)
  explicit ProcedureDesignator(const Symbol &n) : u{&n} {}
  explicit ProcedureDesignator(const Component &c) : u{c} {}
  explicit ProcedureDesignator(Component &&c) : u{std::move(c)} {}
  std::variant<const Symbol *, Component> u;
};

template<typename ARG> struct ProcedureRef {
  using ArgumentType = common::Indirection<ARG>;
  CLASS_BOILERPLATE(ProcedureRef)
  ProcedureRef(ProcedureDesignator &&p, std::vector<ArgumentType> &&a)
    : proc{std::move(p)}, argument(std::move(a)) {}
  ProcedureDesignator proc;
  std::vector<ArgumentType> argument;
};

struct ActualFunctionArg;
using FunctionRef = ProcedureRef<ActualFunctionArg>;

struct Variable {
  CLASS_BOILERPLATE(Variable)
  explicit Variable(Designator &&d) : u{std::move(d)} {}
  explicit Variable(FunctionRef &&p) : u{std::move(p)} {}
  std::variant<Designator, FunctionRef> u;
};

struct ActualFunctionArg {
  CLASS_BOILERPLATE(ActualFunctionArg)
  explicit ActualFunctionArg(GenericExpr &&x) : u{std::move(x)} {}
  explicit ActualFunctionArg(Variable &&x) : u{std::move(x)} {}
  std::variant<common::Indirection<GenericExpr>, Variable> u;
};

struct Label {  // TODO: this is a placeholder
  CLASS_BOILERPLATE(Label)
  explicit Label(int lab) : label{lab} {}
  int label;
};

struct ActualSubroutineArg {
  CLASS_BOILERPLATE(ActualSubroutineArg)
  explicit ActualSubroutineArg(GenericExpr &&x) : u{std::move(x)} {}
  explicit ActualSubroutineArg(Variable &&x) : u{std::move(x)} {}
  explicit ActualSubroutineArg(const Label &l) : u{&l} {}
  std::variant<common::Indirection<GenericExpr>, Variable, const Label *> u;
};

using SubroutineRef = ProcedureRef<ActualSubroutineArg>;

}  // namespace Fortran::evaluate

// This inclusion must follow the definitions in this header due to
// mutual references.
#include "expression.h"

#endif  // FORTRAN_EVALUATE_VARIABLE_H_
