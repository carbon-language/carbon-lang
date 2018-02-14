#ifndef FORTRAN_TYPE_H_
#define FORTRAN_TYPE_H_

#include "../parser/idioms.h"
#include "attr.h"
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

namespace Fortran {
namespace semantics {

using Name = std::string;

// TODO
class IntExpr {
public:
  virtual const IntExpr *clone() const { return new IntExpr{*this}; }
  virtual std::ostream &output(std::ostream &o) const { return o << "IntExpr"; }
};

// TODO
class IntConst : public IntExpr {
public:
  static const IntConst &make(int value);
  const IntExpr *clone() const override { return &make(value_); }
  bool operator==(const IntConst &x) const { return value_ == x.value_; }
  bool operator!=(const IntConst &x) const { return !operator==(x); }
  bool operator<(const IntConst &x) const { return value_ < x.value_; }
  std::ostream &output(std::ostream &o) const override {
    return o << this->value_;
  }

private:
  static std::unordered_map<int, IntConst> cache;
  IntConst(int value) : value_{value} {}
  const int value_;
};

// The value of a kind type parameter
class KindParamValue {
public:
  KindParamValue(int value = 0) : value_{IntConst::make(value)} {}
  bool operator==(const KindParamValue &x) const { return value_ == x.value_; }
  bool operator!=(const KindParamValue &x) const { return !operator==(x); }
  bool operator<(const KindParamValue &x) const { return value_ < x.value_; }

private:
  const IntConst &value_;
  friend std::ostream &operator<<(std::ostream &, const KindParamValue &);
};

// An array spec bound: an explicit integer expression or ASSUMED or DEFERRED
class Bound {
public:
  static const Bound ASSUMED;
  static const Bound DEFERRED;
  Bound(const IntExpr &expr) : category_{Explicit}, expr_{expr.clone()} {}
  bool isExplicit() const { return category_ == Explicit; }
  bool isAssumed() const { return category_ == Assumed; }
  bool isDeferred() const { return category_ == Deferred; }
  const IntExpr &getExplicit() const { return *expr_; }

private:
  enum Category { Explicit, Deferred, Assumed };
  Bound(Category category) : category_{category}, expr_{&IntConst::make(0)} {}
  const Category category_;
  const IntExpr *const expr_;
  friend std::ostream &operator<<(std::ostream &, const Bound &);
};

// The value of a len type parameter
using LenParamValue = Bound;

class IntrinsicTypeSpec;
class DerivedTypeSpec;
class DeclTypeSpec {
public:
  // intrinsic-type-spec or TYPE(intrinsic-type-spec)
  static DeclTypeSpec makeIntrinsic(
      const IntrinsicTypeSpec *intrinsicTypeSpec) {
    return DeclTypeSpec{Intrinsic, intrinsicTypeSpec};
  }
  // TYPE(derived-type-spec)
  static DeclTypeSpec makeTypeDerivedType(
      const DerivedTypeSpec *derivedTypeSpec) {
    return DeclTypeSpec{TypeDerived, nullptr, derivedTypeSpec};
  }
  // CLASS(derived-type-spec)
  static DeclTypeSpec makeClassDerivedType(
      const DerivedTypeSpec *derivedTypeSpec) {
    return DeclTypeSpec{ClassDerived, nullptr, derivedTypeSpec};
  }
  // TYPE(*)
  static DeclTypeSpec makeTypeStar() { return DeclTypeSpec{TypeStar}; }
  // CLASS(*)
  static DeclTypeSpec makeClassStar() { return DeclTypeSpec{ClassStar}; }

  enum Category { Intrinsic, TypeDerived, ClassDerived, TypeStar, ClassStar };
  Category category() const { return category_; }
  const IntrinsicTypeSpec *intrinsicTypeSpec() const {
    return intrinsicTypeSpec_;
  }
  const DerivedTypeSpec *derivedTypeSpec() const { return derivedTypeSpec_; }

private:
  DeclTypeSpec(Category category,
      const IntrinsicTypeSpec *intrinsicTypeSpec = nullptr,
      const DerivedTypeSpec *derivedTypeSpec = nullptr)
    : category_{category}, intrinsicTypeSpec_{intrinsicTypeSpec},
      derivedTypeSpec_{derivedTypeSpec} {}
  const Category category_;
  const IntrinsicTypeSpec *const intrinsicTypeSpec_;
  const DerivedTypeSpec *const derivedTypeSpec_;
  friend std::ostream &operator<<(std::ostream &, const DeclTypeSpec &);
};

// Root of the *TypeSpec hierarchy
class TypeSpec {
public:
  virtual std::ostream &output(std::ostream &o) const = 0;
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
  const T &make() { return make(defaultValue_); }
  const T &make(KindParamValue kind) {
    auto it = cache.find(kind);
    if (it == cache.end()) {
      it = cache.insert(std::make_pair(kind, T{kind})).first;
    }
    return it->second;
  }
  std::ostream &output(std::ostream &o, const T &x) {
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
  static const LogicalTypeSpec *make();
  static const LogicalTypeSpec *make(KindParamValue kind);
  std::ostream &output(std::ostream &o) const override { return o << *this; }

private:
  friend class KindedTypeHelper<LogicalTypeSpec>;
  static KindedTypeHelper<LogicalTypeSpec> helper;
  LogicalTypeSpec(KindParamValue kind) : IntrinsicTypeSpec(kind) {}
  friend std::ostream &operator<<(std::ostream &o, const LogicalTypeSpec &x);
};

// One unique instance of IntegerTypeSpec for each kind.
class IntegerTypeSpec : public NumericTypeSpec {
public:
  static const IntegerTypeSpec *make();
  static const IntegerTypeSpec *make(KindParamValue kind);
  std::ostream &output(std::ostream &o) const override { return o << *this; }

private:
  friend class KindedTypeHelper<IntegerTypeSpec>;
  static KindedTypeHelper<IntegerTypeSpec> helper;
  IntegerTypeSpec(KindParamValue kind) : NumericTypeSpec(kind) {}
  friend std::ostream &operator<<(std::ostream &o, const IntegerTypeSpec &x);
};

// One unique instance of RealTypeSpec for each kind.
class RealTypeSpec : public NumericTypeSpec {
public:
  static const RealTypeSpec *make();
  static const RealTypeSpec *make(KindParamValue kind);
  std::ostream &output(std::ostream &o) const override { return o << *this; }

private:
  friend class KindedTypeHelper<RealTypeSpec>;
  static KindedTypeHelper<RealTypeSpec> helper;
  RealTypeSpec(KindParamValue kind) : NumericTypeSpec(kind) {}
  friend std::ostream &operator<<(std::ostream &o, const RealTypeSpec &x);
};

// One unique instance of ComplexTypeSpec for each kind.
class ComplexTypeSpec : public NumericTypeSpec {
public:
  static const ComplexTypeSpec *make();
  static const ComplexTypeSpec *make(KindParamValue kind);
  std::ostream &output(std::ostream &o) const override { return o << *this; }

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
  std::ostream &output(std::ostream &o) const override { return o << *this; }

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
  static ShapeSpec makeExplicit(const Bound &lb, const Bound &ub) {
    return ShapeSpec(lb, ub);
  }
  // 1:ub
  static const ShapeSpec makeExplicit(const Bound &ub) {
    return makeExplicit(IntConst::make(1), ub);
  }
  // 1: or lb:
  static ShapeSpec makeAssumed(const Bound &lb = IntConst::make(1)) {
    return ShapeSpec(lb, Bound::DEFERRED);
  }
  // :
  static ShapeSpec makeDeferred() {
    return ShapeSpec(Bound::DEFERRED, Bound::DEFERRED);
  }
  // 1:* or lb:*
  static ShapeSpec makeImplied(const Bound &lb) {
    return ShapeSpec(lb, Bound::ASSUMED);
  }
  // ..
  static ShapeSpec makeAssumedRank() {
    return ShapeSpec(Bound::ASSUMED, Bound::ASSUMED);
  }

  bool isExplicit() const { return ub_.isExplicit(); }
  bool isDeferred() const { return lb_.isDeferred(); }

private:
  ShapeSpec(const Bound &lb, const Bound &ub) : lb_{lb}, ub_{ub} {}
  const Bound lb_;
  const Bound ub_;
  friend std::ostream &operator<<(std::ostream &, const ShapeSpec &);
};

using ComponentArraySpec = std::list<ShapeSpec>;

class DataComponentDef {
public:
  // TODO: character-length - should be in DeclTypeSpec (overrides what is
  // there)
  // TODO: coarray-spec
  // TODO: component-initialization
  DataComponentDef(
      const DeclTypeSpec &type, const Name &name, const Attrs &attrs)
    : DataComponentDef(type, name, attrs, ComponentArraySpec{}) {}
  DataComponentDef(const DeclTypeSpec &type, const Name &name,
      const Attrs &attrs, const ComponentArraySpec &arraySpec);

private:
  const DeclTypeSpec type_;
  const Name name_;
  const Attrs attrs_;
  const ComponentArraySpec arraySpec_;
  friend std::ostream &operator<<(std::ostream &, const DataComponentDef &);
};

class ProcDecl {
public:
  ProcDecl(const Name &name) : name_{name} {}
  // TODO: proc-pointer-init
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
  operator DerivedTypeDef() const { return DerivedTypeDef(data_); }
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

using KindParamValues = std::map<Name, KindParamValue>;
using LenParamValues = std::map<Name, LenParamValue>;

// Instantiation of a DerivedTypeDef with kind and len parameter values
class DerivedTypeSpec : public TypeSpec {
public:
  std::ostream &output(std::ostream &o) const override { return o << *this; }

private:
  const DerivedTypeDef def_;
  const KindParamValues kindParamValues_;
  const LenParamValues lenParamValues_;
  DerivedTypeSpec(DerivedTypeDef def, const KindParamValues &kindParamValues,
      const LenParamValues &lenParamValues);
  friend std::ostream &operator<<(std::ostream &, const DerivedTypeSpec &);
};

}  // namespace semantics
}  // namespace Fortran

#endif
