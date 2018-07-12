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

// Defines data structures to represent data access and function calls
// for use in expressions and assignment statements.  Both copy and move
// semantics are supported.  The representation adheres closely to the
// Fortran 2018 language standard (q.v.) and uses strong typing to ensure
// that only admissable combinations can be constructed.

#include "common.h"
#include "expression-forward.h"
#include "intrinsics.h"
#include "../common/idioms.h"
#include "../semantics/symbol.h"
#include <optional>
#include <ostream>
#include <variant>
#include <vector>

namespace Fortran::evaluate {

using semantics::Symbol;

// Forward declarations
struct DataRef;
struct Variable;
struct ActualFunctionArg;

// Subscript and cosubscript expressions are of a kind that matches the
// address size, at least at the top level.
using SubscriptIntegerExpr = IntegerExpr<SubscriptInteger::kind>;
using IndirectSubscriptIntegerExpr = CopyableIndirection<SubscriptIntegerExpr>;

// R913 structure-component & C920: Defined to be a multi-part
// data-ref whose last part has no subscripts (or image-selector, although
// that isn't explicit in the document).  Pointer and allocatable components
// are not explicitly indirected in this representation.
// Complex components (%RE, %IM) are isolated below in ComplexPart.
class Component {
public:
  CLASS_BOILERPLATE(Component)
  Component(const DataRef &b, const Symbol &c) : base_{b}, symbol_{&c} {}
  Component(DataRef &&b, const Symbol &c) : base_{std::move(b)}, symbol_{&c} {}
  Component(CopyableIndirection<DataRef> &&b, const Symbol &c)
    : base_{std::move(b)}, symbol_{&c} {}
  const DataRef &base() const { return *base_; }
  DataRef &base() { return *base_; }
  const Symbol &symbol() const { return *symbol_; }
  SubscriptIntegerExpr LEN() const;
private:
  CopyableIndirection<DataRef> base_;
  const Symbol *symbol_;
};

// TODO pmk continue data hiding from here...

// R921 subscript-triplet
struct Triplet {
  CLASS_BOILERPLATE(Triplet)
  Triplet(std::optional<SubscriptIntegerExpr> &&,
      std::optional<SubscriptIntegerExpr> &&,
      std::optional<SubscriptIntegerExpr> &&);
  std::optional<IndirectSubscriptIntegerExpr> lower, upper, stride;
};

// R919 subscript when rank 0, R923 vector-subscript when rank 1
struct Subscript {
  CLASS_BOILERPLATE(Subscript)
  explicit Subscript(const SubscriptIntegerExpr &s)
    : u{IndirectSubscriptIntegerExpr::Make(s)} {}
  explicit Subscript(SubscriptIntegerExpr &&s)
    : u{IndirectSubscriptIntegerExpr::Make(std::move(s))} {}
  explicit Subscript(const Triplet &t) : u{t} {}
  explicit Subscript(Triplet &&t) : u{std::move(t)} {}
  std::variant<IndirectSubscriptIntegerExpr, Triplet> u;
};

// R917 array-element, R918 array-section; however, the case of an
// array-section that is a complex-part-designator is represented here
// as a ComplexPart instead.  C919 & C925 require that at most one set of
// subscripts have rank greater than 0, but that is not explicit in
// these types.
struct ArrayRef {
  CLASS_BOILERPLATE(ArrayRef)
  ArrayRef(const Symbol &n, std::vector<Subscript> &&ss)
    : u{&n}, subscript(std::move(ss)) {}
  ArrayRef(Component &&c, std::vector<Subscript> &&ss)
    : u{std::move(c)}, subscript(std::move(ss)) {}
  SubscriptIntegerExpr LEN() const;
  std::variant<const Symbol *, Component> u;
  std::vector<Subscript> subscript;
};

// R914 coindexed-named-object
// R924 image-selector, R926 image-selector-spec.
// C824 severely limits the usage of derived types with coarray ultimate
// components: they can't be pointers, allocatables, arrays, coarrays, or
// function results.  They can be components of other derived types.
// C930 precludes having both TEAM= and TEAM_NUMBER=.
// TODO C931 prohibits the use of a coindexed object as a stat-variable.
struct CoarrayRef {
  CLASS_BOILERPLATE(CoarrayRef)
  CoarrayRef(std::vector<const Symbol *> &&,
      std::vector<SubscriptIntegerExpr> &&,
      std::vector<SubscriptIntegerExpr> &&);  // TODO: stat & team?
  SubscriptIntegerExpr LEN() const;
  std::vector<const Symbol *> base;
  std::vector<SubscriptIntegerExpr> subscript, cosubscript;
  std::optional<CopyableIndirection<Variable>> stat, team;
  bool teamIsTeamNumber{false};  // false: TEAM=, true: TEAM_NUMBER=
};

// R911 data-ref is defined syntactically as a series of part-refs, which
// would be far too expressive if the constraints were ignored.  Here, the
// possible outcomes are spelled out.  Note that a data-ref cannot include
// a terminal substring range or complex component designator; use
// R901 designator for that.
struct DataRef {
  CLASS_BOILERPLATE(DataRef)
  explicit DataRef(const Symbol &n) : u{&n} {}
  explicit DataRef(Component &&c) : u{std::move(c)} {}
  explicit DataRef(ArrayRef &&a) : u{std::move(a)} {}
  explicit DataRef(CoarrayRef &&a) : u{std::move(a)} {}
  SubscriptIntegerExpr LEN() const;
  std::variant<const Symbol *, Component, ArrayRef, CoarrayRef> u;
};

// R908 substring, R909 parent-string, R910 substring-range.
// The base object of a substring can be a literal.
// In the F2018 standard, substrings of array sections are parsed as
// variants of sections instead.
struct Substring {
  CLASS_BOILERPLATE(Substring)
  Substring(DataRef &&, std::optional<SubscriptIntegerExpr> &&,
      std::optional<SubscriptIntegerExpr> &&);
  Substring(std::string &&, std::optional<SubscriptIntegerExpr> &&,
      std::optional<SubscriptIntegerExpr> &&);

  SubscriptIntegerExpr First() const;
  SubscriptIntegerExpr Last() const;
  SubscriptIntegerExpr LEN() const;

  std::variant<DataRef, std::string> u;
  std::optional<IndirectSubscriptIntegerExpr> first, last;
};

// R915 complex-part-designator
// In the F2018 standard, complex parts of array sections are parsed as
// variants of sections instead.
struct ComplexPart {
  ENUM_CLASS(Part, RE, IM)
  CLASS_BOILERPLATE(ComplexPart)
  ComplexPart(DataRef &&z, Part p) : complex{std::move(z)}, part{p} {}
  DataRef complex;
  Part part;
};

// R901 designator is the most general data reference object, apart from
// calls to pointer-valued functions.
struct Designator {
  CLASS_BOILERPLATE(Designator)
  explicit Designator(DataRef &&d) : u{std::move(d)} {}
  explicit Designator(Substring &&s) : u{std::move(s)} {}
  explicit Designator(ComplexPart &&c) : u{std::move(c)} {}
  std::variant<DataRef, Substring, ComplexPart> u;
};

struct ProcedureDesignator {
  CLASS_BOILERPLATE(ProcedureDesignator)
  explicit ProcedureDesignator(IntrinsicProcedure p) : u{p} {}
  explicit ProcedureDesignator(const Symbol &n) : u{&n} {}
  explicit ProcedureDesignator(const Component &c) : u{c} {}
  explicit ProcedureDesignator(Component &&c) : u{std::move(c)} {}
  SubscriptIntegerExpr LEN() const;
  std::variant<IntrinsicProcedure, const Symbol *, Component> u;
};

template<typename ARG> struct ProcedureRef {
  using ArgumentType = CopyableIndirection<ARG>;
  CLASS_BOILERPLATE(ProcedureRef)
  ProcedureRef(ProcedureDesignator &&p, std::vector<ArgumentType> &&a)
    : proc{std::move(p)}, argument(std::move(a)) {}
  ProcedureDesignator proc;
  std::vector<ArgumentType> argument;
};

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
  std::variant<CopyableIndirection<GenericExpr>, Variable> u;
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
  std::variant<CopyableIndirection<GenericExpr>, Variable, const Label *> u;
};

using SubroutineRef = ProcedureRef<ActualSubroutineArg>;

}  // namespace Fortran::evaluate

// This inclusion must follow the definitions in this header due to
// mutual references.
#include "expression.h"

#endif  // FORTRAN_EVALUATE_VARIABLE_H_
