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
#include "../common/Fortran.h"
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
using KindExpr = SubscriptIntExpr;

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
  static ParamValue Assumed(common::TypeParamAttr attr) {
    return ParamValue{Category::Assumed, attr};
  }
  static ParamValue Deferred(common::TypeParamAttr attr) {
    return ParamValue{Category::Deferred, attr};
  }
  ParamValue(const ParamValue &) = default;
  explicit ParamValue(MaybeIntExpr &&, common::TypeParamAttr);
  explicit ParamValue(SomeIntExpr &&, common::TypeParamAttr attr);
  explicit ParamValue(common::ConstantSubscript, common::TypeParamAttr attr);
  bool isExplicit() const { return category_ == Category::Explicit; }
  bool isAssumed() const { return category_ == Category::Assumed; }
  bool isDeferred() const { return category_ == Category::Deferred; }
  const MaybeIntExpr &GetExplicit() const { return expr_; }
  void SetExplicit(SomeIntExpr &&);
  bool isKind() const { return attr_ == common::TypeParamAttr::Kind; }
  bool isLen() const { return attr_ == common::TypeParamAttr::Len; }
  void set_attr(common::TypeParamAttr attr) { attr_ = attr; }
  bool operator==(const ParamValue &that) const {
    return category_ == that.category_ && expr_ == that.expr_;
  }
  std::string AsFortran() const;

private:
  enum class Category { Explicit, Deferred, Assumed };
  ParamValue(Category category, common::TypeParamAttr attr)
    : category_{category}, attr_{attr} {}
  Category category_{Category::Explicit};
  common::TypeParamAttr attr_{common::TypeParamAttr::Kind};
  MaybeIntExpr expr_;
  friend std::ostream &operator<<(std::ostream &, const ParamValue &);
};

class IntrinsicTypeSpec {
public:
  TypeCategory category() const { return category_; }
  const KindExpr &kind() const { return kind_; }
  bool operator==(const IntrinsicTypeSpec &x) const {
    return category_ == x.category_ && kind_ == x.kind_;
  }
  bool operator!=(const IntrinsicTypeSpec &x) const { return !operator==(x); }
  std::string AsFortran() const;

protected:
  IntrinsicTypeSpec(TypeCategory, KindExpr &&);

private:
  TypeCategory category_;
  KindExpr kind_;
  friend std::ostream &operator<<(std::ostream &os, const IntrinsicTypeSpec &x);
};

class NumericTypeSpec : public IntrinsicTypeSpec {
public:
  NumericTypeSpec(TypeCategory category, KindExpr &&kind)
    : IntrinsicTypeSpec(category, std::move(kind)) {
    CHECK(common::IsNumericTypeCategory(category));
  }
};

class LogicalTypeSpec : public IntrinsicTypeSpec {
public:
  explicit LogicalTypeSpec(KindExpr &&kind)
    : IntrinsicTypeSpec(TypeCategory::Logical, std::move(kind)) {}
};

class CharacterTypeSpec : public IntrinsicTypeSpec {
public:
  CharacterTypeSpec(ParamValue &&length, KindExpr &&kind)
    : IntrinsicTypeSpec(TypeCategory::Character, std::move(kind)),
      length_{std::move(length)} {}
  const ParamValue &length() const { return length_; }
  std::string AsFortran() const;

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
bool IsExplicit(const ArraySpec &);

class DerivedTypeSpec {
public:
  explicit DerivedTypeSpec(const Symbol &symbol) : typeSymbol_{symbol} {}
  DerivedTypeSpec(const DerivedTypeSpec &);
  DerivedTypeSpec(DerivedTypeSpec &&);

  const Symbol &typeSymbol() const { return typeSymbol_; }
  const Scope *scope() const { return scope_; }
  void set_scope(const Scope &);
  void ReplaceScope(const Scope &);
  const std::map<SourceName, ParamValue> &parameters() const {
    return parameters_;
  }

  bool HasActualParameters() const { return !parameters_.empty(); }
  ParamValue &AddParamValue(SourceName, ParamValue &&);
  ParamValue *FindParameter(SourceName);
  const ParamValue *FindParameter(SourceName target) const {
    auto iter{parameters_.find(target)};
    if (iter != parameters_.end()) {
      return &iter->second;
    } else {
      return nullptr;
    }
  }
  bool operator==(const DerivedTypeSpec &that) const {
    return &typeSymbol_ == &that.typeSymbol_ && parameters_ == that.parameters_;
  }
  std::string AsFortran() const;

private:
  const Symbol &typeSymbol_;
  const Scope *scope_{nullptr};  // same as typeSymbol_.scope() unless PDT
  std::map<SourceName, ParamValue> parameters_;
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
  DeclTypeSpec(NumericTypeSpec &&);
  DeclTypeSpec(LogicalTypeSpec &&);
  // character
  DeclTypeSpec(const CharacterTypeSpec &);
  DeclTypeSpec(CharacterTypeSpec &&);
  // TYPE(derived-type-spec) or CLASS(derived-type-spec)
  DeclTypeSpec(Category, const DerivedTypeSpec &);
  DeclTypeSpec(Category, DerivedTypeSpec &&);
  // TYPE(*) or CLASS(*)
  DeclTypeSpec(Category);

  bool operator==(const DeclTypeSpec &) const;
  bool operator!=(const DeclTypeSpec &that) const { return !operator==(that); }

  Category category() const { return category_; }
  void set_category(Category category) { category_ = category; }
  bool IsPolymorphic() const {
    return category_ == ClassDerived || IsUnlimitedPolymorphic();
  }
  bool IsUnlimitedPolymorphic() const {
    return category_ == TypeStar || category_ == ClassStar;
  }
  bool IsNumeric(TypeCategory) const;
  const NumericTypeSpec &numericTypeSpec() const;
  const LogicalTypeSpec &logicalTypeSpec() const;
  const CharacterTypeSpec &characterTypeSpec() const {
    CHECK(category_ == Character);
    return std::get<CharacterTypeSpec>(typeSpec_);
  }
  const DerivedTypeSpec &derivedTypeSpec() const {
    CHECK(category_ == TypeDerived || category_ == ClassDerived);
    return std::get<DerivedTypeSpec>(typeSpec_);
  }
  DerivedTypeSpec &derivedTypeSpec() {
    CHECK(category_ == TypeDerived || category_ == ClassDerived);
    return std::get<DerivedTypeSpec>(typeSpec_);
  }

  IntrinsicTypeSpec *AsIntrinsic();
  const IntrinsicTypeSpec *AsIntrinsic() const {
    switch (category_) {
    case Numeric: return &std::get<NumericTypeSpec>(typeSpec_);
    case Logical: return &std::get<LogicalTypeSpec>(typeSpec_);
    case Character: return &std::get<CharacterTypeSpec>(typeSpec_);
    default: return nullptr;
    }
  }

  const DerivedTypeSpec *AsDerived() const {
    switch (category_) {
    case TypeDerived:
    case ClassDerived: return &std::get<DerivedTypeSpec>(typeSpec_);
    default: return nullptr;
    }
  }

  std::string AsFortran() const;

private:
  Category category_;
  std::variant<std::monostate, NumericTypeSpec, LogicalTypeSpec,
      CharacterTypeSpec, DerivedTypeSpec>
      typeSpec_;
};
std::ostream &operator<<(std::ostream &, const DeclTypeSpec &);

// This represents a proc-interface in the declaration of a procedure or
// procedure component. It comprises a symbol that represents the specific
// interface or a decl-type-spec that represents the function return type.
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
