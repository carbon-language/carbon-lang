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
#include "intrinsics.h"
#include "type.h"
#include "../common/idioms.h"
#include "../lib/common/template.h"
#include "../semantics/symbol.h"
#include <optional>
#include <ostream>
#include <variant>
#include <vector>

namespace Fortran::evaluate {

using semantics::Symbol;

// Forward declarations
struct DataRef;
template<typename A> struct Variable;

// Subscript and cosubscript expressions are of a kind that matches the
// address size, at least at the top level.
using IndirectSubscriptIntegerExpr =
    CopyableIndirection<Expr<SubscriptInteger>>;

// R913 structure-component & C920: Defined to be a multi-part
// data-ref whose last part has no subscripts (or image-selector, although
// that isn't explicit in the document).  Pointer and allocatable components
// are not explicitly indirected in this representation (TODO: yet?)
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
  int Rank() const;
  const Symbol *GetSymbol(bool first) const;
  Expr<SubscriptInteger> LEN() const;
  std::ostream &Dump(std::ostream &) const;

private:
  CopyableIndirection<DataRef> base_;
  const Symbol *symbol_;
};

// R921 subscript-triplet
class Triplet {
public:
  Triplet() {}
  DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(Triplet)
  Triplet(std::optional<Expr<SubscriptInteger>> &&,
      std::optional<Expr<SubscriptInteger>> &&,
      std::optional<Expr<SubscriptInteger>> &&);
  std::optional<Expr<SubscriptInteger>> lower() const;
  std::optional<Expr<SubscriptInteger>> upper() const;
  std::optional<Expr<SubscriptInteger>> stride() const;
  std::ostream &Dump(std::ostream &) const;

private:
  std::optional<IndirectSubscriptIntegerExpr> lower_, upper_, stride_;
};

// R919 subscript when rank 0, R923 vector-subscript when rank 1
struct Subscript {
  EVALUATE_UNION_CLASS_BOILERPLATE(Subscript)
  explicit Subscript(Expr<SubscriptInteger> &&s)
    : u{IndirectSubscriptIntegerExpr::Make(std::move(s))} {}
  int Rank() const;
  std::ostream &Dump(std::ostream &) const;
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

  int Rank() const;
  const Symbol *GetSymbol(bool first) const;
  Expr<SubscriptInteger> LEN() const;
  std::ostream &Dump(std::ostream &) const;

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
class CoarrayRef {
public:
  CLASS_BOILERPLATE(CoarrayRef)
  CoarrayRef(std::vector<const Symbol *> &&,
      std::vector<Expr<SubscriptInteger>> &&,
      std::vector<Expr<SubscriptInteger>> &&);  // TODO: stat & team?
  CoarrayRef &set_stat(Variable<DefaultInteger> &&);
  CoarrayRef &set_team(Variable<DefaultInteger> &&, bool isTeamNumber = false);

  int Rank() const;
  const Symbol *GetSymbol(bool first) const {
    if (first) {
      return base_.front();
    } else {
      return base_.back();
    }
  }
  Expr<SubscriptInteger> LEN() const;
  std::ostream &Dump(std::ostream &) const;

private:
  std::vector<const Symbol *> base_;
  std::vector<Expr<SubscriptInteger>> subscript_, cosubscript_;
  std::optional<CopyableIndirection<Variable<DefaultInteger>>> stat_, team_;
  bool teamIsTeamNumber_{false};  // false: TEAM=, true: TEAM_NUMBER=
};

// R911 data-ref is defined syntactically as a series of part-refs, which
// would be far too expressive if the constraints were ignored.  Here, the
// possible outcomes are spelled out.  Note that a data-ref cannot include
// a terminal substring range or complex component designator; use
// R901 designator for that.
struct DataRef {
  EVALUATE_UNION_CLASS_BOILERPLATE(DataRef)
  explicit DataRef(const Symbol &n) : u{&n} {}

  int Rank() const;
  const Symbol *GetSymbol(bool first) const;
  Expr<SubscriptInteger> LEN() const;
  std::ostream &Dump(std::ostream &) const;

  std::variant<const Symbol *, Component, ArrayRef, CoarrayRef> u;
};

// R908 substring, R909 parent-string, R910 substring-range.
// The base object of a substring can be a literal.
// In the F2018 standard, substrings of array sections are parsed as
// variants of sections instead.
class Substring {
public:
  using IsFoldableTrait = std::true_type;
  CLASS_BOILERPLATE(Substring)
  Substring(DataRef &&, std::optional<Expr<SubscriptInteger>> &&,
      std::optional<Expr<SubscriptInteger>> &&);
  Substring(std::string &&, std::optional<Expr<SubscriptInteger>> &&,
      std::optional<Expr<SubscriptInteger>> &&);

  Expr<SubscriptInteger> first() const;
  Expr<SubscriptInteger> last() const;
  int Rank() const;
  const Symbol *GetSymbol(bool first) const;
  Expr<SubscriptInteger> LEN() const;
  std::optional<std::string> Fold(FoldingContext &);
  std::ostream &Dump(std::ostream &) const;

private:
  // TODO: character kinds > 1
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
  int Rank() const;
  const Symbol *GetSymbol(bool first) const {
    return complex_.GetSymbol(first);
  }
  std::ostream &Dump(std::ostream &) const;

private:
  DataRef complex_;
  Part part_;
};

// R901 designator is the most general data reference object, apart from
// calls to pointer-valued functions.  Its variant holds everything that
// a DataRef can, and possibly either a substring reference or a complex
// part (%RE/%IM) reference.
template<typename A> class Designator {
  using DataRefs = decltype(DataRef::u);
  using MaybeSubstring =
      std::conditional_t<A::category == TypeCategory::Character,
          std::variant<Substring>, std::variant<>>;
  using MaybeComplexPart = std::conditional_t<A::category == TypeCategory::Real,
      std::variant<ComplexPart>, std::variant<>>;
  using Variant =
      common::CombineVariants<DataRefs, MaybeSubstring, MaybeComplexPart>;

public:
  using Result = A;
  static_assert(Result::isSpecificType);
  EVALUATE_UNION_CLASS_BOILERPLATE(Designator)
  Designator(const DataRef &that) : u{common::MoveVariant<Variant>(that.u)} {}
  Designator(DataRef &&that)
    : u{common::MoveVariant<Variant>(std::move(that.u))} {}

  std::optional<DynamicType> GetType() const {
    if constexpr (std::is_same_v<Result, SomeDerived>) {
      if (const Symbol * sym{GetSymbol(false)}) {
        return GetSymbolType(*sym);
      } else {
        return std::nullopt;
      }
    } else {
      return Result::GetType();
    }
  }

  int Rank() const {
    return std::visit(
        common::visitors{[](const Symbol *sym) { return sym->Rank(); },
            [](const auto &x) { return x.Rank(); }},
        u);
  }

  const Symbol *GetSymbol(bool first) const {
    return std::visit(common::visitors{[](const Symbol *sym) { return sym; },
                          [=](const auto &x) { return x.GetSymbol(first); }},
        u);
  }

  Expr<SubscriptInteger> LEN() const;

  std::ostream &Dump(std::ostream &o) const {
    std::visit(common::visitors{[&](const Symbol *sym) {
                                  o << sym->name().ToString();
                                },
                   [&](const auto &x) { x.Dump(o); }},
        u);
    return o;
  }

  Variant u;
};

// TODO pmk: move more of these into call.h/cc...
struct ProcedureDesignator {
  EVALUATE_UNION_CLASS_BOILERPLATE(ProcedureDesignator)
  explicit ProcedureDesignator(IntrinsicProcedure p) : u{p} {}
  explicit ProcedureDesignator(const Symbol &n) : u{&n} {}
  Expr<SubscriptInteger> LEN() const;
  int Rank() const;
  const Symbol *GetSymbol() const;
  std::ostream &Dump(std::ostream &) const;

  std::variant<IntrinsicProcedure, const Symbol *, Component> u;
};

class UntypedFunctionRef {
public:
  CLASS_BOILERPLATE(UntypedFunctionRef)
  UntypedFunctionRef(ProcedureDesignator &&p, Arguments &&a, int r)
    : proc_{std::move(p)}, arguments_(std::move(a)), rank_{r} {}
  UntypedFunctionRef(ProcedureDesignator &&p, Arguments &&a)
    : proc_{std::move(p)}, arguments_(std::move(a)) {}

  const ProcedureDesignator &proc() const { return proc_; }
  const Arguments &arguments() const { return arguments_; }

  Expr<SubscriptInteger> LEN() const;
  int Rank() const { return rank_; }
  std::ostream &Dump(std::ostream &) const;

protected:
  ProcedureDesignator proc_;
  Arguments arguments_;
  int rank_{proc_.Rank()};
};

template<typename A> struct FunctionRef : public UntypedFunctionRef {
  using Result = A;
  static_assert(Result::isSpecificType);
  // Subtlety: There is a distinction that must be maintained here between an
  // actual argument expression that *is* a variable and one that is not,
  // e.g. between X and (X).  The parser attempts to parse each argument
  // first as a variable, then as an expression, and the distinction appears
  // in the parse tree.
  CLASS_BOILERPLATE(FunctionRef)
  FunctionRef(UntypedFunctionRef &&ufr) : UntypedFunctionRef{std::move(ufr)} {}
  FunctionRef(ProcedureDesignator &&p, Arguments &&a, int rank = 0)
    : UntypedFunctionRef{std::move(p), std::move(a), rank} {}
  std::optional<DynamicType> GetType() const {
    if constexpr (std::is_same_v<Result, SomeDerived>) {
      if (const Symbol * symbol{proc_.GetSymbol()}) {
        return GetSymbolType(*symbol);
      }
    } else {
      return Result::GetType();
    }
    return std::nullopt;
  }
};

template<typename A> struct Variable {
  using Result = A;
  static_assert(Result::isSpecificType);
  EVALUATE_UNION_CLASS_BOILERPLATE(Variable)
  std::optional<DynamicType> GetType() const {
    return std::visit([](const auto &x) { return x.GetType(); }, u);
  }
  int Rank() const {
    return std::visit([](const auto &x) { return x.Rank(); }, u);
  }
  std::ostream &Dump(std::ostream &o) const {
    std::visit([&](const auto &x) { x.Dump(o); }, u);
    return o;
  }
  std::variant<Designator<Result>, FunctionRef<Result>> u;
};

struct Label {  // TODO: this is a placeholder
  CLASS_BOILERPLATE(Label)
  explicit Label(int lab) : label{lab} {}
  int label;
  std::ostream &Dump(std::ostream &) const;
};

class SubroutineCall {
public:
  CLASS_BOILERPLATE(SubroutineCall)
  SubroutineCall(ProcedureDesignator &&p, Arguments &&a)
    : proc_{std::move(p)}, arguments_(std::move(a)) {}
  const ProcedureDesignator &proc() const { return proc_; }
  const Arguments &arguments() const { return arguments_; }
  int Rank() const { return 0; }  // TODO: elemental subroutine representation
  std::ostream &Dump(std::ostream &) const;

private:
  ProcedureDesignator proc_;
  Arguments arguments_;
};

FOR_EACH_CHARACTER_KIND(extern template class Designator)

}  // namespace Fortran::evaluate

#endif  // FORTRAN_EVALUATE_VARIABLE_H_
