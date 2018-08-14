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
class DataRef;
class Variable;
class ActualFunctionArg;

// Subscript and cosubscript expressions are of a kind that matches the
// address size, at least at the top level.
using IndirectSubscriptIntegerExpr =
    CopyableIndirection<Expr<SubscriptInteger>>;

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
  Expr<SubscriptInteger> LEN() const;

private:
  CopyableIndirection<DataRef> base_;
  const Symbol *symbol_;
};

// R921 subscript-triplet
class Triplet {
public:
  CLASS_BOILERPLATE(Triplet)
  Triplet(std::optional<Expr<SubscriptInteger>> &&,
      std::optional<Expr<SubscriptInteger>> &&,
      std::optional<Expr<SubscriptInteger>> &&);
  std::optional<Expr<SubscriptInteger>> lower() const;
  std::optional<Expr<SubscriptInteger>> upper() const;
  std::optional<Expr<SubscriptInteger>> stride() const;

private:
  std::optional<IndirectSubscriptIntegerExpr> lower_, upper_, stride_;
};

// R919 subscript when rank 0, R923 vector-subscript when rank 1
class Subscript {
public:
  CLASS_BOILERPLATE(Subscript)
  explicit Subscript(const Expr<SubscriptInteger> &s)
    : u_{IndirectSubscriptIntegerExpr::Make(s)} {}
  explicit Subscript(Expr<SubscriptInteger> &&s)
    : u_{IndirectSubscriptIntegerExpr::Make(std::move(s))} {}
  explicit Subscript(const Triplet &t) : u_{t} {}
  explicit Subscript(Triplet &&t) : u_{std::move(t)} {}

private:
  std::variant<IndirectSubscriptIntegerExpr, Triplet> u_;
};

// R917 array-element, R918 array-section; however, the case of an
// array-section that is a complex-part-designator is represented here
// as a ComplexPart instead.  C919 & C925 require that at most one set of
// subscripts have rank greater than 0, but that is not explicit in
// these types.
class ArrayRef {
public:
  CLASS_BOILERPLATE(ArrayRef)
  ArrayRef(const Symbol &n, std::vector<Subscript> &&ss)
    : u_{&n}, subscript_(std::move(ss)) {}
  ArrayRef(Component &&c, std::vector<Subscript> &&ss)
    : u_{std::move(c)}, subscript_(std::move(ss)) {}
  Expr<SubscriptInteger> LEN() const;

private:
  std::variant<const Symbol *, Component> u_;
  std::vector<Subscript> subscript_;
};

// R914 coindexed-named-object
// R924 image-selector, R926 image-selector-spec.
// C824 severely limits the usage of derived types with coarray ultimate
// components: they can't be pointers, allocatables, arrays, coarrays, or
// function results.  They can be components of other derived types.
// C930 precludes having both TEAM= and TEAM_NUMBER=.
// TODO C931 prohibits the use of a coindexed object as a stat-variable.
class CoarrayRef {
public:
  CLASS_BOILERPLATE(CoarrayRef)
  CoarrayRef(std::vector<const Symbol *> &&,
      std::vector<Expr<SubscriptInteger>> &&,
      std::vector<Expr<SubscriptInteger>> &&);  // TODO: stat & team?
  CoarrayRef &setStat(Variable &&);
  CoarrayRef &setTeam(Variable &&, bool isTeamNumber = false);
  Expr<SubscriptInteger> LEN() const;

private:
  std::vector<const Symbol *> base_;
  std::vector<Expr<SubscriptInteger>> subscript_, cosubscript_;
  std::optional<CopyableIndirection<Variable>> stat_, team_;
  bool teamIsTeamNumber_{false};  // false: TEAM=, true: TEAM_NUMBER=
};

// R911 data-ref is defined syntactically as a series of part-refs, which
// would be far too expressive if the constraints were ignored.  Here, the
// possible outcomes are spelled out.  Note that a data-ref cannot include
// a terminal substring range or complex component designator; use
// R901 designator for that.
class DataRef {
public:
  CLASS_BOILERPLATE(DataRef)
  explicit DataRef(const Symbol &n) : u_{&n} {}
  explicit DataRef(Component &&c) : u_{std::move(c)} {}
  explicit DataRef(ArrayRef &&a) : u_{std::move(a)} {}
  explicit DataRef(CoarrayRef &&a) : u_{std::move(a)} {}
  Expr<SubscriptInteger> LEN() const;

private:
  std::variant<const Symbol *, Component, ArrayRef, CoarrayRef> u_;
};

// R908 substring, R909 parent-string, R910 substring-range.
// The base object of a substring can be a literal.
// In the F2018 standard, substrings of array sections are parsed as
// variants of sections instead.
class Substring {
public:
  using FoldableTrait = std::true_type;
  CLASS_BOILERPLATE(Substring)
  Substring(DataRef &&, std::optional<Expr<SubscriptInteger>> &&,
      std::optional<Expr<SubscriptInteger>> &&);
  Substring(std::string &&, std::optional<Expr<SubscriptInteger>> &&,
      std::optional<Expr<SubscriptInteger>> &&);

  Expr<SubscriptInteger> first() const;
  Expr<SubscriptInteger> last() const;
  Expr<SubscriptInteger> LEN() const;
  std::optional<std::string> Fold(FoldingContext &);

private:
  std::variant<DataRef, std::string> u_;
  std::optional<IndirectSubscriptIntegerExpr> first_, last_;
};

// R915 complex-part-designator
// In the F2018 standard, complex parts of array sections are parsed as
// variants of sections instead.
class ComplexPart {
public:
  ENUM_CLASS(Part, RE, IM)
  CLASS_BOILERPLATE(ComplexPart)
  ComplexPart(DataRef &&z, Part p) : complex_{std::move(z)}, part_{p} {}
  const DataRef &complex() const { return complex_; }
  Part part() const { return part_; }

private:
  DataRef complex_;
  Part part_;
};

// R901 designator is the most general data reference object, apart from
// calls to pointer-valued functions.
class Designator {
public:
  CLASS_BOILERPLATE(Designator)
  explicit Designator(DataRef &&d) : u_{std::move(d)} {}
  explicit Designator(Substring &&s) : u_{std::move(s)} {}
  explicit Designator(ComplexPart &&c) : u_{std::move(c)} {}

private:
  std::variant<DataRef, Substring, ComplexPart> u_;
};

class ProcedureDesignator {
public:
  CLASS_BOILERPLATE(ProcedureDesignator)
  explicit ProcedureDesignator(IntrinsicProcedure p) : u_{p} {}
  explicit ProcedureDesignator(const Symbol &n) : u_{&n} {}
  explicit ProcedureDesignator(const Component &c) : u_{c} {}
  explicit ProcedureDesignator(Component &&c) : u_{std::move(c)} {}
  Expr<SubscriptInteger> LEN() const;

private:
  std::variant<IntrinsicProcedure, const Symbol *, Component> u_;
};

template<typename ARG> class ProcedureRef {
public:
  using ArgumentType = CopyableIndirection<ARG>;
  CLASS_BOILERPLATE(ProcedureRef)
  ProcedureRef(ProcedureDesignator &&p, std::vector<ArgumentType> &&a)
    : proc_{std::move(p)}, argument_(std::move(a)) {}
  const ProcedureDesignator &proc() const { return proc_; }
  const std::vector<ArgumentType> &argument() const { return argument_; }

private:
  ProcedureDesignator proc_;
  std::vector<ArgumentType> argument_;
};

using FunctionRef = ProcedureRef<ActualFunctionArg>;

class Variable {
public:
  CLASS_BOILERPLATE(Variable)
  explicit Variable(Designator &&d) : u_{std::move(d)} {}
  explicit Variable(FunctionRef &&p) : u_{std::move(p)} {}

private:
  std::variant<Designator, FunctionRef> u_;
};

class ActualFunctionArg {
public:
  CLASS_BOILERPLATE(ActualFunctionArg)
  explicit ActualFunctionArg(Expr<SomeType> &&x) : u_{std::move(x)} {}
  explicit ActualFunctionArg(Variable &&x) : u_{std::move(x)} {}

private:
  std::variant<CopyableIndirection<Expr<SomeType>>, Variable> u_;
};

struct Label {  // TODO: this is a placeholder
  CLASS_BOILERPLATE(Label)
  explicit Label(int lab) : label{lab} {}
  int label;
};

class ActualSubroutineArg {
public:
  CLASS_BOILERPLATE(ActualSubroutineArg)
  explicit ActualSubroutineArg(Expr<SomeType> &&x) : u_{std::move(x)} {}
  explicit ActualSubroutineArg(Variable &&x) : u_{std::move(x)} {}
  explicit ActualSubroutineArg(const Label &l) : u_{&l} {}

private:
  std::variant<CopyableIndirection<Expr<SomeType>>, Variable, const Label *> u_;
};

using SubroutineRef = ProcedureRef<ActualSubroutineArg>;

}  // namespace Fortran::evaluate

// This inclusion must follow the definitions in this header due to
// mutual references.
#include "expression.h"

#endif  // FORTRAN_EVALUATE_VARIABLE_H_
