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
#include "../common/idioms.h"
#include "../parser/char-block.h"
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>

/*

Type specs are represented by a class hierarchy rooted at TypeSpec. Only the
leaves are concrete types:
  TypeSpec
    IntrinsicTypeSpec
      CharacterTypeSpec
      LogicalTypeSpec
      NumericTypeSpec
        IntegerTypeSpec
        RealTypeSpec
        ComplexTypeSpec
    DerivedTypeSpec

TypeSpec classes are immutable. For intrinsic types (except character) there
is a limited number of instances -- one for each kind.

A DerivedTypeSpec is based on a DerivedTypeDef (from a derived type statement)
with kind and len parameter values provided.

*/

namespace Fortran::semantics {

using Name = std::string;

/// A SourceName is a name in the cooked character stream,
/// i.e. a range of lower-case characters with provenance.
using SourceName = parser::CharBlock;

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

// The value of a kind type parameter
class KindParamValue {
public:
  KindParamValue(int value = 0) : KindParamValue(IntConst::Make(value)) {}
  KindParamValue(const IntConst &value) : value_{value} {}
  bool operator==(const KindParamValue &x) const { return value_ == x.value_; }
  bool operator!=(const KindParamValue &x) const { return !operator==(x); }
  bool operator<(const KindParamValue &x) const { return value_ < x.value_; }
  const IntConst &value() const { return value_; }

private:
  const IntConst &value_;
  friend std::ostream &operator<<(std::ostream &, const KindParamValue &);
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

// The value of a len type parameter
using LenParamValue = Bound;

class IntrinsicTypeSpec;
class DerivedTypeSpec;

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

  bool operator==(const DeclTypeSpec &that) const {
    if (category_ != that.category_) {
      return false;
    }
    switch (category_) {
    case Intrinsic: return typeSpec_.intrinsic == that.typeSpec_.intrinsic;
    case TypeDerived:
    case ClassDerived: return typeSpec_.derived == that.typeSpec_.derived;
    default: return true;
    }
  }
  bool operator!=(const DeclTypeSpec &that) const { return !operator==(that); }

  Category category() const { return category_; }
  const IntrinsicTypeSpec &intrinsicTypeSpec() const;
  DerivedTypeSpec &derivedTypeSpec();
  const DerivedTypeSpec &derivedTypeSpec() const;

private:
  Category category_;
  union {
    const IntrinsicTypeSpec *intrinsic;
    DerivedTypeSpec *derived;
  } typeSpec_;
};
std::ostream &operator<<(std::ostream &, const DeclTypeSpec &);

// Root of the *TypeSpec hierarchy
class TypeSpec {
public:
  virtual std::ostream &Output(std::ostream &o) const = 0;
};

class IntrinsicTypeSpec : public TypeSpec {
public:
  const KindParamValue &kind() const { return kind_; }

protected:
  IntrinsicTypeSpec(KindParamValue kind) : kind_{kind} {}
  const KindParamValue kind_;
};

class NumericTypeSpec : public IntrinsicTypeSpec {
protected:
  NumericTypeSpec(KindParamValue kind) : IntrinsicTypeSpec(kind) {}
};

namespace {

// Helper to cache mapping of kind to TypeSpec
template<typename T> class KindedTypeHelper {
public:
  std::map<KindParamValue, T> cache;
  KindedTypeHelper(Name name, KindParamValue defaultValue)
    : name_{name}, defaultValue_{defaultValue} {}
  const T &Make() { return Make(defaultValue_); }
  const T &Make(KindParamValue kind) {
    auto it{cache.find(kind)};
    if (it == cache.end()) {
      it = cache.insert(std::make_pair(kind, T{kind})).first;
    }
    return it->second;
  }
  std::ostream &Output(std::ostream &o, const T &x) {
    o << name_;
    if (x.kind_ != defaultValue_) o << '(' << x.kind_ << ')';
    return o;
  }

private:
  const Name name_;
  const KindParamValue defaultValue_;
};

}  // namespace

// One unique instance of LogicalTypeSpec for each kind.
class LogicalTypeSpec : public IntrinsicTypeSpec {
public:
  static const LogicalTypeSpec &Make();
  static const LogicalTypeSpec &Make(KindParamValue kind);
  std::ostream &Output(std::ostream &o) const override { return o << *this; }

private:
  friend class KindedTypeHelper<LogicalTypeSpec>;
  static KindedTypeHelper<LogicalTypeSpec> helper;
  LogicalTypeSpec(KindParamValue kind) : IntrinsicTypeSpec(kind) {}
  friend std::ostream &operator<<(std::ostream &o, const LogicalTypeSpec &x);
};

// One unique instance of IntegerTypeSpec for each kind.
class IntegerTypeSpec : public NumericTypeSpec {
public:
  static const IntegerTypeSpec &Make();
  static const IntegerTypeSpec &Make(KindParamValue kind);
  std::ostream &Output(std::ostream &o) const override { return o << *this; }

private:
  friend class KindedTypeHelper<IntegerTypeSpec>;
  static KindedTypeHelper<IntegerTypeSpec> helper;
  IntegerTypeSpec(KindParamValue kind) : NumericTypeSpec(kind) {}
  friend std::ostream &operator<<(std::ostream &o, const IntegerTypeSpec &x);
};

// One unique instance of RealTypeSpec for each kind.
class RealTypeSpec : public NumericTypeSpec {
public:
  static const RealTypeSpec &Make();
  static const RealTypeSpec &Make(KindParamValue kind);
  std::ostream &Output(std::ostream &o) const override { return o << *this; }

private:
  friend class KindedTypeHelper<RealTypeSpec>;
  static KindedTypeHelper<RealTypeSpec> helper;
  RealTypeSpec(KindParamValue kind) : NumericTypeSpec(kind) {}
  friend std::ostream &operator<<(std::ostream &o, const RealTypeSpec &x);
};

// One unique instance of ComplexTypeSpec for each kind.
class ComplexTypeSpec : public NumericTypeSpec {
public:
  static const ComplexTypeSpec &Make();
  static const ComplexTypeSpec &Make(KindParamValue kind);
  std::ostream &Output(std::ostream &o) const override { return o << *this; }

private:
  friend class KindedTypeHelper<ComplexTypeSpec>;
  static KindedTypeHelper<ComplexTypeSpec> helper;
  ComplexTypeSpec(KindParamValue kind) : NumericTypeSpec(kind) {}
  friend std::ostream &operator<<(std::ostream &o, const ComplexTypeSpec &x);
};

class CharacterTypeSpec : public IntrinsicTypeSpec {
public:
  static const int DefaultKind = 0;
  CharacterTypeSpec(LenParamValue len, KindParamValue kind = DefaultKind)
    : IntrinsicTypeSpec{kind}, len_{len} {}
  const LenParamValue &len() const { return len_; }
  std::ostream &Output(std::ostream &o) const override { return o << *this; }

private:
  const LenParamValue len_;
  friend std::ostream &operator<<(std::ostream &, const CharacterTypeSpec &);
};

// Definition of a type parameter
class TypeParamDef {
public:
  TypeParamDef(const Name &name, const IntegerTypeSpec &type,
      const std::optional<IntConst> &defaultValue = {})
    : name_{name}, type_{type}, defaultValue_{defaultValue} {};
  const Name &name() const { return name_; }
  const IntegerTypeSpec &type() const { return type_; }
  const std::optional<IntConst> &defaultValue() const { return defaultValue_; }

private:
  const Name name_;
  const IntegerTypeSpec type_;
  const std::optional<IntConst> defaultValue_;
};

using TypeParamDefs = std::list<TypeParamDef>;

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

class DataComponentDef {
public:
  // TODO: character-length - should be in DeclTypeSpec (overrides what is
  // there)
  // TODO: coarray-spec
  // TODO: component-initialization
  DataComponentDef(
      const DeclTypeSpec &type, const SourceName &name, const Attrs &attrs)
    : DataComponentDef(type, name, attrs, ArraySpec{}) {}
  DataComponentDef(const DeclTypeSpec &type, const SourceName &name,
      const Attrs &attrs, const ArraySpec &arraySpec);

  const DeclTypeSpec &type() const { return type_; }
  const SourceName &name() const { return name_; }
  const Attrs &attrs() const { return attrs_; }
  const ArraySpec &shape() const { return arraySpec_; }

private:
  const DeclTypeSpec type_;
  const SourceName name_;
  const Attrs attrs_;
  const ArraySpec arraySpec_;
  friend std::ostream &operator<<(std::ostream &, const DataComponentDef &);
};

class Scope;
class Symbol;

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

class ProcDecl {
public:
  ProcDecl(const ProcDecl &decl) = default;
  ProcDecl(const SourceName &name) : name_{name} {}
  // TODO: proc-pointer-init
  const SourceName &name() const { return name_; }

private:
  const SourceName name_;
  friend std::ostream &operator<<(std::ostream &, const ProcDecl &);
};

class ProcComponentDef {
public:
  ProcComponentDef(
      const ProcDecl &decl, Attrs attrs, const ProcInterface &interface);

  const ProcDecl &decl() const { return decl_; }
  const Attrs &attrs() const { return attrs_; }
  const ProcInterface &interface() const { return interface_; }

private:
  const ProcDecl decl_;
  const Attrs attrs_;
  const ProcInterface interface_;
  friend std::ostream &operator<<(std::ostream &, const ProcComponentDef &);
};

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

class TypeBoundGeneric {
public:
  TypeBoundGeneric(const SourceName &name, const Attrs &attrs,
      const GenericSpec &genericSpec)
    : name_{name}, attrs_{attrs}, genericSpec_{genericSpec} {
    attrs_.CheckValid({Attr::PUBLIC, Attr::PRIVATE});
  }

private:
  const SourceName name_;
  const Attrs attrs_;
  const GenericSpec genericSpec_;
  friend std::ostream &operator<<(std::ostream &, const TypeBoundGeneric &);
};

class TypeBoundProc {
public:
  TypeBoundProc(const SourceName &interface, const Attrs &attrs,
      const SourceName &binding)
    : TypeBoundProc(interface, attrs, binding, binding) {
    if (!attrs_.test(Attr::DEFERRED)) {
      common::die(
          "DEFERRED attribute is required if interface name is specified");
    }
  }
  TypeBoundProc(const Attrs &attrs, const SourceName &binding,
      const std::optional<SourceName> &procedure)
    : TypeBoundProc({}, attrs, binding, procedure ? *procedure : binding) {
    if (attrs_.test(Attr::DEFERRED)) {
      common::die("DEFERRED attribute is only allowed with interface name");
    }
  }

private:
  TypeBoundProc(const std::optional<SourceName> &interface, const Attrs &attrs,
      const SourceName &binding, const SourceName &procedure)
    : interface_{interface}, attrs_{attrs}, binding_{binding}, procedure_{
                                                                   procedure} {
    attrs_.CheckValid({Attr::PUBLIC, Attr::PRIVATE, Attr::NOPASS, Attr::PASS,
        Attr::DEFERRED, Attr::NON_OVERRIDABLE});
  }
  const std::optional<SourceName> interface_;
  const Attrs attrs_;
  const SourceName binding_;
  const SourceName procedure_;
  friend std::ostream &operator<<(std::ostream &, const TypeBoundProc &);
};

// Definition of a derived type
class DerivedTypeDef {
public:
  const SourceName &name() const { return *data_.name; }
  const SourceName *extends() const { return data_.extends; }
  const Attrs &attrs() const { return data_.attrs; }
  const TypeParamDefs &lenParams() const { return data_.lenParams; }
  const TypeParamDefs &kindParams() const { return data_.kindParams; }
  const std::list<DataComponentDef> &dataComponents() const {
    return data_.dataComps;
  }
  const std::list<ProcComponentDef> &procComponents() const {
    return data_.procComps;
  }
  const std::list<TypeBoundProc> &typeBoundProcs() const {
    return data_.typeBoundProcs;
  }
  const std::list<TypeBoundGeneric> &typeBoundGenerics() const {
    return data_.typeBoundGenerics;
  }
  const std::list<SourceName> finalProcs() const { return data_.finalProcs; }

  struct Data {
    const SourceName *name{nullptr};
    const SourceName *extends{nullptr};
    Attrs attrs;
    bool Private{false};
    bool sequence{false};
    TypeParamDefs lenParams;
    TypeParamDefs kindParams;
    std::list<DataComponentDef> dataComps;
    std::list<ProcComponentDef> procComps;
    bool bindingPrivate{false};
    std::list<TypeBoundProc> typeBoundProcs;
    std::list<TypeBoundGeneric> typeBoundGenerics;
    std::list<SourceName> finalProcs;
    bool hasTbpPart() const {
      return !finalProcs.empty() || !typeBoundProcs.empty() ||
          !typeBoundGenerics.empty();
    }
  };
  explicit DerivedTypeDef(const Data &x);

private:
  const Data data_;
  // TODO: type-bound procedures
  friend std::ostream &operator<<(std::ostream &, const DerivedTypeDef &);
};

using ParamValue = LenParamValue;

class DerivedTypeSpec : public TypeSpec {
public:
  std::ostream &Output(std::ostream &o) const override { return o << *this; }
  explicit DerivedTypeSpec(const SourceName &name) : name_{&name} {}
  DerivedTypeSpec() = delete;
  virtual ~DerivedTypeSpec();
  const SourceName &name() const { return *name_; }
  const Scope *scope() const { return scope_; }
  void set_scope(const Scope &);

private:
  const SourceName *name_;
  const Scope *scope_{nullptr};
  std::list<std::pair<std::optional<SourceName>, ParamValue>> paramValues_;
  friend std::ostream &operator<<(std::ostream &, const DerivedTypeSpec &);
};

}  // namespace Fortran::semantics

#endif  // FORTRAN_SEMANTICS_TYPE_H_
