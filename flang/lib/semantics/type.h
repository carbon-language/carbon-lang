// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef FORTRAN_SEMANTICS_TYPE_H_
#define FORTRAN_SEMANTICS_TYPE_H_

#include "attr.h"
#include "../common/fortran.h"
#include "../common/idioms.h"
#include "../common/indirection.h"
#include "../evaluate/expression.h"
#include "../parser/char-block.h"
#include <list>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <variant>

namespace Fortran::parser {
struct Expr;
}

namespace Fortran::semantics {

class Scope;
class Symbol;
class SemanticsContext;
class ExprResolver;

/// A SourceName is a name in the cooked character stream,
/// i.e. a range of lower-case characters with provenance.
using SourceName = parser::CharBlock;
using TypeCategory = common::TypeCategory;
using SomeExpr = evaluate::Expr<evaluate::SomeType>;
using MaybeExpr = std::optional<SomeExpr>;
using SomeIntExpr = evaluate::Expr<evaluate::SomeInteger>;
using MaybeIntExpr = std::optional<SomeIntExpr>;
using SubscriptIntExpr = evaluate::Expr<evaluate::SubscriptInteger>;
using MaybeSubscriptIntExpr = std::optional<SubscriptIntExpr>;

// An array spec bound: an explicit integer expression or ASSUMED or DEFERRED
class Bound {
public:
  static Bound Assumed() { return Bound(Category::Assumed); }
  static Bound Deferred() { return Bound(Category::Deferred); }
  explicit Bound(MaybeSubscriptIntExpr &&expr) : expr_{std::move(expr)} {}
  explicit Bound(int bound);
  Bound(const Bound &) = default;
  Bound(Bound &&) = default;
  Bound &operator=(const Bound &) = default;
  Bound &operator=(Bound &&) = default;
  bool isExplicit() const { return category_ == Category::Explicit; }
  bool isAssumed() const { return category_ == Category::Assumed; }
  bool isDeferred() const { return category_ == Category::Deferred; }
  MaybeSubscriptIntExpr &GetExplicit() { return expr_; }
  const MaybeSubscriptIntExpr &GetExplicit() const { return expr_; }
  void SetExplicit(MaybeSubscriptIntExpr &&expr) {
    CHECK(isExplicit());
    expr_ = std::move(expr);
  }

private:
  enum class Category { Explicit, Deferred, Assumed };
  Bound(Category category) : category_{category} {}
  Bound(Category category, MaybeSubscriptIntExpr &&expr)
    : category_{category}, expr_{std::move(expr)} {}
  Category category_{Category::Explicit};
  MaybeSubscriptIntExpr expr_;
  friend std::ostream &operator<<(std::ostream &, const Bound &);
};

// A type parameter value: integer expression or assumed or deferred.
class ParamValue {
public:
  static ParamValue Assumed() { return Category::Assumed; }
  static ParamValue Deferred() { return Category::Deferred; }
  explicit ParamValue(MaybeIntExpr &&expr);
  explicit ParamValue(std::int64_t);
  bool isExplicit() const { return category_ == Category::Explicit; }
  bool isAssumed() const { return category_ == Category::Assumed; }
  bool isDeferred() const { return category_ == Category::Deferred; }
  const MaybeIntExpr &GetExplicit() const { return expr_; }

private:
  enum class Category { Explicit, Deferred, Assumed };
  ParamValue(Category category) : category_{category} {}
  Category category_{Category::Explicit};
  MaybeIntExpr expr_;
  friend std::ostream &operator<<(std::ostream &, const ParamValue &);
};

class IntrinsicTypeSpec {
public:
  TypeCategory category() const { return category_; }
  int kind() const { return kind_; }
  bool operator==(const IntrinsicTypeSpec &x) const {
    return category_ == x.category_ && kind_ == x.kind_;
  }
  bool operator!=(const IntrinsicTypeSpec &x) const { return !operator==(x); }

protected:
  IntrinsicTypeSpec(TypeCategory, int kind);

private:
  TypeCategory category_;
  int kind_;
  friend std::ostream &operator<<(std::ostream &os, const IntrinsicTypeSpec &x);
};

class NumericTypeSpec : public IntrinsicTypeSpec {
public:
  NumericTypeSpec(TypeCategory category, int kind)
    : IntrinsicTypeSpec(category, kind) {
    CHECK(category == TypeCategory::Integer || category == TypeCategory::Real ||
        category == TypeCategory::Complex);
  }
};

class LogicalTypeSpec : public IntrinsicTypeSpec {
public:
  LogicalTypeSpec(int kind) : IntrinsicTypeSpec(TypeCategory::Logical, kind) {}
};

class CharacterTypeSpec : public IntrinsicTypeSpec {
public:
  CharacterTypeSpec(ParamValue &&length, int kind)
    : IntrinsicTypeSpec(TypeCategory::Character, kind), length_{std::move(
                                                            length)} {}
  const ParamValue length() const { return length_; }

private:
  ParamValue length_;
  friend std::ostream &operator<<(std::ostream &os, const CharacterTypeSpec &x);
};

class ShapeSpec {
public:
  // lb:ub
  static ShapeSpec MakeExplicit(Bound &&lb, Bound &&ub) {
    return ShapeSpec(std::move(lb), std::move(ub));
  }
  // 1:ub
  static const ShapeSpec MakeExplicit(Bound &&ub) {
    return MakeExplicit(Bound{1}, std::move(ub));
  }
  // 1:
  static ShapeSpec MakeAssumed() {
    return ShapeSpec(Bound{1}, Bound::Deferred());
  }
  // lb:
  static ShapeSpec MakeAssumed(Bound &&lb) {
    return ShapeSpec(std::move(lb), Bound::Deferred());
  }
  // :
  static ShapeSpec MakeDeferred() {
    return ShapeSpec(Bound::Deferred(), Bound::Deferred());
  }
  // 1:*
  static ShapeSpec MakeImplied() {
    return ShapeSpec(Bound{1}, Bound::Assumed());
  }
  // lb:*
  static ShapeSpec MakeImplied(Bound &&lb) {
    return ShapeSpec(std::move(lb), Bound::Assumed());
  }
  // ..
  static ShapeSpec MakeAssumedRank() {
    return ShapeSpec(Bound::Assumed(), Bound::Assumed());
  }

  ShapeSpec(const ShapeSpec &) = default;
  ShapeSpec(ShapeSpec &&) = default;
  ShapeSpec &operator=(const ShapeSpec &) = default;
  ShapeSpec &operator=(ShapeSpec &&) = default;

  bool isExplicit() const { return ub_.isExplicit(); }
  bool isDeferred() const { return lb_.isDeferred(); }
  Bound &lbound() { return lb_; }
  const Bound &lbound() const { return lb_; }
  Bound &ubound() { return ub_; }
  const Bound &ubound() const { return ub_; }

private:
  ShapeSpec(Bound &&lb, Bound &&ub) : lb_{std::move(lb)}, ub_{std::move(ub)} {}
  Bound lb_;
  Bound ub_;
  friend ExprResolver;
  friend std::ostream &operator<<(std::ostream &, const ShapeSpec &);
};

using ArraySpec = std::list<ShapeSpec>;

class DerivedTypeSpec {
public:
  using listType = std::list<std::pair<std::optional<SourceName>, ParamValue>>;
  DerivedTypeSpec &operator=(const DerivedTypeSpec &) = delete;
  explicit DerivedTypeSpec(const Symbol &symbol) : typeSymbol_{symbol} {}
  DerivedTypeSpec() = delete;
  const Symbol &typeSymbol() const { return typeSymbol_; }
  const Scope *scope() const { return scope_; }
  void set_scope(const Scope &);
  listType &paramValues() { return paramValues_; }
  const listType &paramValues() const { return paramValues_; }
  void AddParamValue(ParamValue &&);
  void AddParamValue(const SourceName &, ParamValue &&);

private:
  const Symbol &typeSymbol_;
  const Scope *scope_{nullptr};
  listType paramValues_;
  friend std::ostream &operator<<(std::ostream &, const DerivedTypeSpec &);
};

class DeclTypeSpec {
public:
  enum Category {
    Numeric,
    Logical,
    Character,
    TypeDerived,
    ClassDerived,
    TypeStar,
    ClassStar
  };

  // intrinsic-type-spec or TYPE(intrinsic-type-spec), not character
  DeclTypeSpec(const NumericTypeSpec &);
  DeclTypeSpec(const LogicalTypeSpec &);
  // character
  DeclTypeSpec(CharacterTypeSpec &);
  // TYPE(derived-type-spec) or CLASS(derived-type-spec)
  DeclTypeSpec(Category, const DerivedTypeSpec &);
  // TYPE(*) or CLASS(*)
  DeclTypeSpec(Category);
  DeclTypeSpec() = delete;

  bool operator==(const DeclTypeSpec &) const;
  bool operator!=(const DeclTypeSpec &that) const { return !operator==(that); }

  Category category() const { return category_; }
  bool IsNumeric(TypeCategory) const;
  const IntrinsicTypeSpec *AsIntrinsic() const;
  const DerivedTypeSpec *AsDerived() const;
  const NumericTypeSpec &numericTypeSpec() const;
  const LogicalTypeSpec &logicalTypeSpec() const;
  const CharacterTypeSpec &characterTypeSpec() const;
  const DerivedTypeSpec &derivedTypeSpec() const;
  void set_category(Category category) { category_ = category; }

private:
  Category category_;
  union TypeSpec {
    TypeSpec() : derived{nullptr} {}
    TypeSpec(NumericTypeSpec numeric) : numeric{numeric} {}
    TypeSpec(LogicalTypeSpec logical) : logical{logical} {}
    TypeSpec(const CharacterTypeSpec *character) : character{character} {}
    TypeSpec(const DerivedTypeSpec *derived) : derived{derived} {}
    NumericTypeSpec numeric;
    LogicalTypeSpec logical;
    const CharacterTypeSpec *character;
    const DerivedTypeSpec *derived;
  } typeSpec_;
};
std::ostream &operator<<(std::ostream &, const DeclTypeSpec &);

// This represents a proc-interface in the declaration of a procedure or
// procedure component. It comprises a symbol (representing the specific
// interface), a decl-type-spec (representing the function return type),
// or neither.
class ProcInterface {
public:
  const Symbol *symbol() const { return symbol_; }
  const DeclTypeSpec *type() const { return type_; }
  void set_symbol(const Symbol &symbol);
  void set_type(const DeclTypeSpec &type);

private:
  const Symbol *symbol_{nullptr};
  const DeclTypeSpec *type_{nullptr};
};
}

#endif  // FORTRAN_SEMANTICS_TYPE_H_
