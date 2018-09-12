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

#ifndef FORTRAN_SEMANTICS_TYPE_H_
#define FORTRAN_SEMANTICS_TYPE_H_

#include "attr.h"
#include "../common/fortran.h"
#include "../common/idioms.h"
#include "../parser/char-block.h"
#include <list>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>

namespace Fortran::semantics {

class Scope;
class Symbol;

/// A SourceName is a name in the cooked character stream,
/// i.e. a range of lower-case characters with provenance.
using SourceName = parser::CharBlock;

using TypeCategory = common::TypeCategory;

// TODO
class IntExpr {
public:
  static IntExpr MakeConst(std::uint64_t value) {
    return IntExpr();  // TODO
  }
  IntExpr() {}
  virtual ~IntExpr();
  virtual std::ostream &Output(std::ostream &o) const { return o << "IntExpr"; }
};

// TODO
class IntConst {
public:
  static const IntConst &Make(std::uint64_t value);
  bool operator==(const IntConst &x) const { return value_ == x.value_; }
  bool operator!=(const IntConst &x) const { return !operator==(x); }
  bool operator<(const IntConst &x) const { return value_ < x.value_; }
  std::uint64_t value() const { return value_; }
  std::ostream &Output(std::ostream &o) const { return o << this->value_; }

private:
  static std::unordered_map<std::uint64_t, IntConst> cache;
  IntConst(std::uint64_t value) : value_{value} {}
  const std::uint64_t value_;
  friend std::ostream &operator<<(std::ostream &, const IntConst &);
};

// An array spec bound: an explicit integer expression or ASSUMED or DEFERRED
class Bound {
public:
  static const Bound ASSUMED;
  static const Bound DEFERRED;
  Bound(const IntExpr &expr) : category_{Explicit}, expr_{expr} {}
  bool isExplicit() const { return category_ == Explicit; }
  bool isAssumed() const { return category_ == Assumed; }
  bool isDeferred() const { return category_ == Deferred; }
  const IntExpr &getExplicit() const {
    CHECK(isExplicit());
    return *expr_;
  }

private:
  enum Category { Explicit, Deferred, Assumed };
  Bound(Category category) : category_{category}, expr_{std::nullopt} {}
  Category category_;
  std::optional<IntExpr> expr_;
  friend std::ostream &operator<<(std::ostream &, const Bound &);
};

class IntrinsicTypeSpec {
public:
  IntrinsicTypeSpec(TypeCategory, int kind = 0);
  const TypeCategory category() const { return category_; }
  const int kind() const { return kind_; }
  bool operator==(const IntrinsicTypeSpec &x) const {
    return category_ == x.category_ && kind_ == x.kind_;
  }
  bool operator!=(const IntrinsicTypeSpec &x) const { return !operator==(x); }

  static int GetDefaultKind(TypeCategory category);

private:
  TypeCategory category_;
  int kind_;
  friend std::ostream &operator<<(std::ostream &os, const IntrinsicTypeSpec &x);
  // TODO: Character and len
};

class ShapeSpec {
public:
  // lb:ub
  static ShapeSpec MakeExplicit(const Bound &lb, const Bound &ub) {
    return ShapeSpec(lb, ub);
  }
  // 1:ub
  static const ShapeSpec MakeExplicit(const Bound &ub) {
    return MakeExplicit(IntExpr::MakeConst(1), ub);
  }
  // 1: or lb:
  static ShapeSpec MakeAssumed(const Bound &lb = IntExpr::MakeConst(1)) {
    return ShapeSpec(lb, Bound::DEFERRED);
  }
  // :
  static ShapeSpec MakeDeferred() {
    return ShapeSpec(Bound::DEFERRED, Bound::DEFERRED);
  }
  // 1:* or lb:*
  static ShapeSpec MakeImplied(const Bound &lb = IntExpr::MakeConst(1)) {
    return ShapeSpec(lb, Bound::ASSUMED);
  }
  // ..
  static ShapeSpec MakeAssumedRank() {
    return ShapeSpec(Bound::ASSUMED, Bound::ASSUMED);
  }

  bool isExplicit() const { return ub_.isExplicit(); }
  bool isDeferred() const { return lb_.isDeferred(); }

  const Bound &lbound() const { return lb_; }
  const Bound &ubound() const { return ub_; }

private:
  ShapeSpec(const Bound &lb, const Bound &ub) : lb_{lb}, ub_{ub} {}
  Bound lb_;
  Bound ub_;
  friend std::ostream &operator<<(std::ostream &, const ShapeSpec &);
};

using ArraySpec = std::list<ShapeSpec>;

class GenericSpec {
public:
  enum Kind {
    GENERIC_NAME,
    OP_DEFINED,
    ASSIGNMENT,
    READ_FORMATTED,
    READ_UNFORMATTED,
    WRITE_FORMATTED,
    WRITE_UNFORMATTED,
    OP_ADD,
    OP_AND,
    OP_CONCAT,
    OP_DIVIDE,
    OP_EQ,
    OP_EQV,
    OP_GE,
    OP_GT,
    OP_LE,
    OP_LT,
    OP_MULTIPLY,
    OP_NE,
    OP_NEQV,
    OP_NOT,
    OP_OR,
    OP_POWER,
    OP_SUBTRACT,
    OP_XOR,
  };
  static GenericSpec IntrinsicOp(Kind kind) {
    return GenericSpec(kind, nullptr);
  }
  static GenericSpec DefinedOp(const SourceName &name) {
    return GenericSpec(OP_DEFINED, &name);
  }
  static GenericSpec GenericName(const SourceName &name) {
    return GenericSpec(GENERIC_NAME, &name);
  }

  const Kind kind() const { return kind_; }
  const SourceName &genericName() const {
    CHECK(kind_ == GENERIC_NAME);
    return *name_;
  }
  const SourceName &definedOp() const {
    CHECK(kind_ == OP_DEFINED);
    return *name_;
  }

private:
  GenericSpec(Kind kind, const SourceName *name) : kind_{kind}, name_{name} {}
  const Kind kind_;
  const SourceName *const name_;  // only for GENERIC_NAME & OP_DEFINED
  friend std::ostream &operator<<(std::ostream &, const GenericSpec &);
};

// The value of a len type parameter
using LenParamValue = Bound;

using ParamValue = LenParamValue;

class DerivedTypeSpec {
public:
  explicit DerivedTypeSpec(const SourceName &name) : name_{&name} {}
  DerivedTypeSpec() = delete;
  const SourceName &name() const { return *name_; }
  const Scope *scope() const { return scope_; }
  void set_scope(const Scope &);

private:
  const SourceName *name_;
  const Scope *scope_{nullptr};
  std::list<std::pair<std::optional<SourceName>, ParamValue>> paramValues_;
  friend std::ostream &operator<<(std::ostream &, const DerivedTypeSpec &);
};

class DeclTypeSpec {
public:
  enum Category { Intrinsic, TypeDerived, ClassDerived, TypeStar, ClassStar };

  // intrinsic-type-spec or TYPE(intrinsic-type-spec)
  DeclTypeSpec(const IntrinsicTypeSpec &);
  // TYPE(derived-type-spec) or CLASS(derived-type-spec)
  DeclTypeSpec(Category, DerivedTypeSpec &);
  // TYPE(*) or CLASS(*)
  DeclTypeSpec(Category);
  DeclTypeSpec() = delete;

  bool operator==(const DeclTypeSpec &) const;
  bool operator!=(const DeclTypeSpec &that) const { return !operator==(that); }

  Category category() const { return category_; }
  const IntrinsicTypeSpec &intrinsicTypeSpec() const;
  DerivedTypeSpec &derivedTypeSpec();
  const DerivedTypeSpec &derivedTypeSpec() const;

private:
  Category category_;
  union TypeSpec {
    TypeSpec() : derived{nullptr} {}
    TypeSpec(IntrinsicTypeSpec intrinsic) : intrinsic{intrinsic} {}
    TypeSpec(DerivedTypeSpec *derived) : derived{derived} {}
    IntrinsicTypeSpec intrinsic;
    DerivedTypeSpec *derived;
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
  const DeclTypeSpec *type() const { return type_ ? &*type_ : nullptr; }
  void set_symbol(const Symbol &symbol);
  void set_type(const DeclTypeSpec &type);

private:
  const Symbol *symbol_{nullptr};
  std::optional<DeclTypeSpec> type_;
};

}  // namespace Fortran::semantics

#endif  // FORTRAN_SEMANTICS_TYPE_H_
