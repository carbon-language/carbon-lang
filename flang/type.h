#ifndef FORTRAN_TYPE_H_
#define FORTRAN_TYPE_H_

#include "attr.h"
#include "idioms.h"
#include <algorithm>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

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

Attributes:
The enum class Attr contains all possible attributes. DerivedTypeDef checks
that supplied attributes are among the allowed ones using checkAttrs().

*/

namespace Fortran {

using Name = std::string;

// TODO
class IntExpr {
public:
  virtual const IntExpr *clone() const { return new IntExpr{*this}; }
  virtual std::ostream &output(std::ostream &o) const { return o << "IntExpr"; }
};
std::ostream &operator<<(std::ostream &o, const IntExpr &x) {
  return x.output(o);
}

// TODO
class IntConst : public IntExpr {
public:
  static const IntConst ZERO;
  static const IntConst ONE;
  IntConst(int value) : value_{value} {}
  virtual const IntExpr *clone() const;
  bool operator==(const IntConst &x) const { return value_ == x.value_; }
  bool operator!=(const IntConst &x) const { return !operator==(x); }
  bool operator<(const IntConst &x) const { return value_ < x.value_; }
  virtual std::ostream &output(std::ostream &o) const {
    return o << this->value_;
  }

private:
  const int value_;
};

// The value of a kind type parameter
class KindParamValue {
public:
  KindParamValue(int value) : value_{value} {}
  bool operator==(const KindParamValue &x) const { return value_ == x.value_; }
  bool operator!=(const KindParamValue &x) const { return !operator==(x); }
  bool operator<(const KindParamValue &x) const { return value_ < x.value_; }

private:
  const IntConst value_;
  friend std::ostream &operator<<(std::ostream &, const KindParamValue &);
};

// The value of a len type parameter
class LenParamValue {
public:
  static const LenParamValue ASSUMED;
  static const LenParamValue DEFERRED;
  LenParamValue(const IntExpr &value) : category_{Expr}, value_{value} {}

private:
  enum Category { Assumed, Deferred, Expr };
  LenParamValue(Category category) : category_{category} {}
  const Category category_;
  const std::optional<const IntExpr> value_;
  friend std::ostream &operator<<(std::ostream &, const LenParamValue &);
};

// Root of the *TypeSpec hierarchy
class TypeSpec {
protected:
  TypeSpec() {}
  virtual ~TypeSpec() = 0;
};
TypeSpec::~TypeSpec() {}

class IntrinsicTypeSpec : public TypeSpec {
public:
  const KindParamValue &kind() const { return kind_; }

protected:
  IntrinsicTypeSpec(KindParamValue kind) : kind_{kind} {}
  virtual ~IntrinsicTypeSpec() = 0;
  const KindParamValue kind_;
};
IntrinsicTypeSpec::~IntrinsicTypeSpec() {}

class NumericTypeSpec : public IntrinsicTypeSpec {
protected:
  NumericTypeSpec(KindParamValue kind) : IntrinsicTypeSpec(kind) {}
  virtual ~NumericTypeSpec() = 0;
};
NumericTypeSpec::~NumericTypeSpec() {}

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
  static const LogicalTypeSpec &make() { return helper.make(); }
  static const LogicalTypeSpec &make(KindParamValue kind) {
    return helper.make(kind);
  }

private:
  friend class KindedTypeHelper<LogicalTypeSpec>;
  static KindedTypeHelper<LogicalTypeSpec> helper;
  LogicalTypeSpec(KindParamValue kind) : IntrinsicTypeSpec(kind) {}
  friend std::ostream &operator<<(std::ostream &o, const LogicalTypeSpec &x);
};

// One unique instance of IntegerTypeSpec for each kind.
class IntegerTypeSpec : public NumericTypeSpec {
public:
  static const IntegerTypeSpec &make() { return helper.make(); }
  static const IntegerTypeSpec &make(KindParamValue kind) {
    return helper.make(kind);
  }

private:
  friend class KindedTypeHelper<IntegerTypeSpec>;
  static KindedTypeHelper<IntegerTypeSpec> helper;
  IntegerTypeSpec(KindParamValue kind) : NumericTypeSpec(kind) {}
  friend std::ostream &operator<<(std::ostream &o, const IntegerTypeSpec &x);
};

// One unique instance of RealTypeSpec for each kind.
class RealTypeSpec : public NumericTypeSpec {
public:
  static const RealTypeSpec &make() { return helper.make(); }
  static const RealTypeSpec &make(KindParamValue kind) {
    return helper.make(kind);
  }

private:
  friend class KindedTypeHelper<RealTypeSpec>;
  static KindedTypeHelper<RealTypeSpec> helper;
  RealTypeSpec(KindParamValue kind) : NumericTypeSpec(kind) {}
  friend std::ostream &operator<<(std::ostream &o, const RealTypeSpec &x);
};

// One unique instance of ComplexTypeSpec for each kind.
class ComplexTypeSpec : public NumericTypeSpec {
public:
  static const ComplexTypeSpec &make() { return helper.make(); }
  static const ComplexTypeSpec &make(KindParamValue kind) {
    return helper.make(kind);
  }

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

using TypeParamDefs = std::vector<TypeParamDef>;

// Definition of a derived type
class DerivedTypeDef {
public:
  DerivedTypeDef(const Name &name, const Attrs &attrs = {},
      const TypeParamDefs &lenParams = {}, const TypeParamDefs &kindParams = {},
      bool private_ = false, bool sequence = false);
  const Name name_;
  const std::optional<Name> parent_ = {};
  const Attrs attrs_;
  const TypeParamDefs lenParams_;
  const TypeParamDefs kindParams_;
  const bool private_ = false;
  const bool sequence_ = false;
  // TODO: components
  // TODO: type-bound procedures
  friend std::ostream &operator<<(std::ostream &, const DerivedTypeDef &);
};

using KindParamValues = std::map<Name, KindParamValue>;
using LenParamValues = std::map<Name, LenParamValue>;

// Instantiation of a DerivedTypeDef with kind and len parameter values
class DerivedTypeSpec : public TypeSpec {
public:
  DerivedTypeSpec(DerivedTypeDef def, KindParamValues kindParamValues{},
      LenParamValues lenParamValues{});

private:
  const DerivedTypeDef def_;
  const KindParamValues kindParamValues_;
  const LenParamValues lenParamValues_;
  friend std::ostream &operator<<(std::ostream &, const DerivedTypeSpec &);
};

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
  const IntrinsicTypeSpec &intrinsicTypeSpec() const {
    return *intrinsicTypeSpec_;
  }
  const DerivedTypeSpec &derivedTypeSpec() const { return *derivedTypeSpec_; }

private:
  DeclTypeSpec(Category category,
      const IntrinsicTypeSpec *intrinsicTypeSpec = nullptr,
      const DerivedTypeSpec *derivedTypeSpec = nullptr)
    : category_{category}, intrinsicTypeSpec_{intrinsicTypeSpec},
      derivedTypeSpec_{derivedTypeSpec} {}
  const Category category_;
  const IntrinsicTypeSpec *const intrinsicTypeSpec_;
  const DerivedTypeSpec *const derivedTypeSpec_;
};

class DataComponentDef {
public:
  // component-array-spec
  // coarray-spec
  DataComponentDef(
      const DeclTypeSpec &type, const Name &name, const Attrs &attrs)
    : type_{type}, name_{name}, attrs_{attrs} {
    checkAttrs("DataComponentDef", attrs,
        Attrs{Attr::PUBLIC, Attr::PRIVATE, Attr::ALLOCATABLE, Attr::CONTIGUOUS,
            Attr::POINTER});
  }

private:
  const DeclTypeSpec type_;
  const Name name_;
  const Attrs attrs_;
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
  Bound(Category category) : category_{category}, expr_{&IntConst::ZERO} {}
  const Category category_;
  const IntExpr *const expr_;
  friend std::ostream &operator<<(std::ostream &, const Bound &);
};

class ShapeSpec {
public:
  // lb:ub
  static ShapeSpec makeExplicit(const Bound &lb, const Bound &ub) {
    return ShapeSpec(lb, ub);
  }
  // 1:ub
  static const ShapeSpec makeExplicit(const Bound &ub) {
    return makeExplicit(IntConst::ONE, ub);
  }
  // 1: or lb:
  static ShapeSpec makeAssumed(const Bound &lb = IntConst::ONE) {
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
  friend std::ostream &operator<<(std::ostream &, const ShapeSpec &);

private:
  ShapeSpec(const Bound &lb, const Bound &ub) : lb_{lb}, ub_{ub} {}
  const Bound lb_;
  const Bound ub_;
};

}  // namespace Fortran

#endif
