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
#include "../parser/idioms.h"
#include "../parser/parse-tree.h"
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

TypeSpec classes are immutable. For instrinsic types (except character) there
is a limited number of instances -- one for each kind.

A DerivedTypeSpec is based on a DerivedTypeDef (from a derived type statement)
with kind and len parameter values provided.

*/

namespace Fortran::semantics {

using Name = std::string;

// TODO
class IntExpr {
public:
  static IntExpr MakeConst(std::uint64_t value) {
    return IntExpr();  // TODO
  }
  IntExpr() {}
  IntExpr(const parser::ScalarIntExpr &) { /*TODO*/
  }
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
  KindParamValue(int value = 0) : value_{IntConst::Make(value)} {}
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
  const Category category_;
  const std::optional<IntExpr> expr_;
  friend std::ostream &operator<<(std::ostream &, const Bound &);
};

// The value of a len type parameter
using LenParamValue = Bound;

class IntrinsicTypeSpec;
class DerivedTypeSpec;
class DeclTypeSpec {
public:
  // intrinsic-type-spec or TYPE(intrinsic-type-spec)
  static DeclTypeSpec MakeIntrinsic(const IntrinsicTypeSpec &typeSpec) {
    return DeclTypeSpec{typeSpec};
  }
  // TYPE(derived-type-spec)
  static DeclTypeSpec MakeTypeDerivedType(
      std::unique_ptr<DerivedTypeSpec> &&typeSpec) {
    return DeclTypeSpec{TypeDerived, std::move(typeSpec)};
  }
  // CLASS(derived-type-spec)
  static DeclTypeSpec MakeClassDerivedType(
      std::unique_ptr<DerivedTypeSpec> &&typeSpec) {
    return DeclTypeSpec{ClassDerived, std::move(typeSpec)};
  }
  // TYPE(*)
  static DeclTypeSpec MakeTypeStar() { return DeclTypeSpec{TypeStar}; }
  // CLASS(*)
  static DeclTypeSpec MakeClassStar() { return DeclTypeSpec{ClassStar}; }

  DeclTypeSpec(const DeclTypeSpec &);
  DeclTypeSpec &operator=(const DeclTypeSpec &);

  enum Category { Intrinsic, TypeDerived, ClassDerived, TypeStar, ClassStar };
  Category category() const { return category_; }
  const IntrinsicTypeSpec &intrinsicTypeSpec() const {
    return *intrinsicTypeSpec_;
  }
  const DerivedTypeSpec &derivedTypeSpec() const { return *derivedTypeSpec_; }

private:
  DeclTypeSpec(Category category) : category_{category} {
    CHECK(category == TypeStar || category == ClassStar);
  }
  DeclTypeSpec(Category category, std::unique_ptr<DerivedTypeSpec> &&typeSpec);
  DeclTypeSpec(const IntrinsicTypeSpec &intrinsicTypeSpec)
    : category_{Intrinsic}, intrinsicTypeSpec_{&intrinsicTypeSpec} {
    // All instances of IntrinsicTypeSpec live in caches and are never deleted,
    // so the pointer to intrinsicTypeSpec will always be valid.
  }

  Category category_;
  const IntrinsicTypeSpec *intrinsicTypeSpec_{nullptr};
  std::unique_ptr<DerivedTypeSpec> derivedTypeSpec_;
  friend std::ostream &operator<<(std::ostream &, const DeclTypeSpec &);
};

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
    auto it = cache.find(kind);
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
  const Bound lb_;
  const Bound ub_;
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
      const DeclTypeSpec &type, const Name &name, const Attrs &attrs)
    : DataComponentDef(type, name, attrs, ArraySpec{}) {}
  DataComponentDef(const DeclTypeSpec &type, const Name &name,
      const Attrs &attrs, const ArraySpec &arraySpec);

  const DeclTypeSpec &type() const { return type_; }
  const Name &name() const { return name_; }
  const Attrs &attrs() const { return attrs_; }
  const ArraySpec &shape() const { return arraySpec_; }

private:
  const DeclTypeSpec type_;
  const Name name_;
  const Attrs attrs_;
  const ArraySpec arraySpec_;
  friend std::ostream &operator<<(std::ostream &, const DataComponentDef &);
};

class ProcDecl {
public:
  ProcDecl(const Name &name) : name_{name} {}
  // TODO: proc-pointer-init
  const Name &name() const { return name_; }

private:
  const Name name_;
  friend std::ostream &operator<<(std::ostream &, const ProcDecl &);
};

class ProcComponentDef {
public:
  ProcComponentDef(ProcDecl decl, Attrs attrs)
    : ProcComponentDef(decl, attrs, std::nullopt, std::nullopt) {}
  ProcComponentDef(ProcDecl decl, Attrs attrs, const Name &interfaceName)
    : ProcComponentDef(decl, attrs, interfaceName, std::nullopt) {}
  ProcComponentDef(ProcDecl decl, Attrs attrs, const DeclTypeSpec &typeSpec)
    : ProcComponentDef(decl, attrs, std::nullopt, typeSpec) {}

  const ProcDecl &decl() const { return decl_; }
  const Attrs &attrs() const { return attrs_; }
  const std::optional<Name> &interfaceName() const { return interfaceName_; }
  const std::optional<DeclTypeSpec> &typeSpec() const { return typeSpec_; }

private:
  ProcComponentDef(ProcDecl decl, Attrs attrs,
      const std::optional<Name> &interfaceName,
      const std::optional<DeclTypeSpec> &typeSpec);
  const ProcDecl decl_;
  const Attrs attrs_;
  const std::optional<Name> interfaceName_;
  const std::optional<DeclTypeSpec> typeSpec_;
  friend std::ostream &operator<<(std::ostream &, const ProcComponentDef &);
};

class DerivedTypeDefBuilder;

// Definition of a derived type
class DerivedTypeDef {
public:
  const Name &name() const { return data_.name; }
  const std::optional<Name> &extends() const { return data_.extends; }
  const Attrs &attrs() const { return data_.attrs; }
  const TypeParamDefs &lenParams() const { return data_.lenParams; }
  const TypeParamDefs &kindParams() const { return data_.kindParams; }
  const std::list<DataComponentDef> &dataComponents() const {
    return data_.dataComps;
  }
  const std::list<ProcComponentDef> &procComponents() const {
    return data_.procComps;
  }

private:
  struct Data {
    Name name;
    std::optional<Name> extends;
    Attrs attrs;
    bool Private{false};
    bool sequence{false};
    TypeParamDefs lenParams;
    TypeParamDefs kindParams;
    std::list<DataComponentDef> dataComps;
    std::list<ProcComponentDef> procComps;
  };
  friend class DerivedTypeDefBuilder;
  explicit DerivedTypeDef(const Data &x);
  const Data data_;
  // TODO: type-bound procedures
  friend std::ostream &operator<<(std::ostream &, const DerivedTypeDef &);
};

class DerivedTypeDefBuilder {
public:
  DerivedTypeDefBuilder(const Name &name) { data_.name = name; }
  DerivedTypeDefBuilder() {}
  operator DerivedTypeDef() const { return DerivedTypeDef(data_); }
  DerivedTypeDefBuilder &name(const Name &x);
  DerivedTypeDefBuilder &extends(const Name &x);
  DerivedTypeDefBuilder &attr(const Attr &x);
  DerivedTypeDefBuilder &attrs(const Attrs &x);
  DerivedTypeDefBuilder &lenParam(const TypeParamDef &x);
  DerivedTypeDefBuilder &kindParam(const TypeParamDef &x);
  DerivedTypeDefBuilder &dataComponent(const DataComponentDef &x);
  DerivedTypeDefBuilder &procComponent(const ProcComponentDef &x);
  DerivedTypeDefBuilder &Private(bool x = true);
  DerivedTypeDefBuilder &sequence(bool x = true);

private:
  DerivedTypeDef::Data data_;
  friend class DerivedTypeDef;
};

using ParamValue = LenParamValue;

// Instantiation of a DerivedTypeDef with kind and len parameter values
class DerivedTypeSpec : public TypeSpec {
public:
  std::ostream &Output(std::ostream &o) const override { return o << *this; }
  DerivedTypeSpec(const Name &name) : name_{name} {}
  virtual ~DerivedTypeSpec() = default;
  DerivedTypeSpec &AddParamValue(const ParamValue &value) {
    paramValues_.push_back(std::make_pair(std::nullopt, value));
    return *this;
  }
  DerivedTypeSpec &AddParamValue(const Name &name, const ParamValue &value) {
    paramValues_.push_back(std::make_pair(name, value));
    return *this;
  }

  const std::list<std::pair<std::optional<Name>, ParamValue>> &paramValues() {
    return paramValues_;
  }

  // Provide access to the derived-type definition if is known
  const DerivedTypeDef *definition() {
    // TODO
    return 0;
  }

private:
  const Name name_;
  std::list<std::pair<std::optional<Name>, ParamValue>> paramValues_;
  friend std::ostream &operator<<(std::ostream &, const DerivedTypeSpec &);
};

}  // namespace Fortran::semantics

#endif  // FORTRAN_SEMANTICS_TYPE_H_
