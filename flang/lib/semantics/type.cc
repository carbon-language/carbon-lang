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

#include "type.h"
#include "scope.h"
#include "symbol.h"
#include "../evaluate/fold.h"
#include "../parser/characters.h"
#include <ostream>
#include <sstream>

namespace Fortran::semantics {

DerivedTypeSpec::DerivedTypeSpec(const DerivedTypeSpec &that)
  : typeSymbol_{that.typeSymbol_}, scope_{that.scope_}, parameters_{
                                                            that.parameters_} {}

DerivedTypeSpec::DerivedTypeSpec(DerivedTypeSpec &&that)
  : typeSymbol_{that.typeSymbol_}, scope_{that.scope_}, parameters_{std::move(
                                                            that.parameters_)} {
}

void DerivedTypeSpec::set_scope(const Scope &scope) {
  CHECK(!scope_);
  ReplaceScope(scope);
}
void DerivedTypeSpec::ReplaceScope(const Scope &scope) {
  CHECK(scope.IsDerivedType());
  scope_ = &scope;
}

ParamValue &DerivedTypeSpec::AddParamValue(
    SourceName name, ParamValue &&value) {
  auto pair{parameters_.insert(std::make_pair(name, std::move(value)))};
  CHECK(pair.second);  // name was not already present
  return pair.first->second;
}

ParamValue *DerivedTypeSpec::FindParameter(SourceName target) {
  return const_cast<ParamValue *>(
      const_cast<const DerivedTypeSpec *>(this)->FindParameter(target));
}

std::string DerivedTypeSpec::AsFortran() const {
  std::stringstream ss;
  ss << typeSymbol_.name().ToString();
  if (!parameters_.empty()) {
    ss << '(';
    bool first = true;
    for (const auto &[name, value] : parameters_) {
      if (first) {
        first = false;
      } else {
        ss << ',';
      }
      ss << name.ToString() << '=' << value.AsFortran();
    }
    ss << ')';
  }
  return ss.str();
}

std::ostream &operator<<(std::ostream &o, const DerivedTypeSpec &x) {
  return o << x.AsFortran();
}

Bound::Bound(int bound) : expr_{bound} {}

std::ostream &operator<<(std::ostream &o, const Bound &x) {
  if (x.isAssumed()) {
    o << '*';
  } else if (x.isDeferred()) {
    o << ':';
  } else if (x.expr_) {
    o << x.expr_;
  } else {
    o << "<no-expr>";
  }
  return o;
}

std::ostream &operator<<(std::ostream &o, const ShapeSpec &x) {
  if (x.lb_.isAssumed()) {
    CHECK(x.ub_.isAssumed());
    o << "..";
  } else {
    if (!x.lb_.isDeferred()) {
      o << x.lb_;
    }
    o << ':';
    if (!x.ub_.isDeferred()) {
      o << x.ub_;
    }
  }
  return o;
}

bool ArraySpec::IsExplicitShape() const {
  return CheckAll([](const ShapeSpec &x) { return x.ubound().isExplicit(); });
}
bool ArraySpec::IsAssumedShape() const {
  return CheckAll([](const ShapeSpec &x) { return x.ubound().isDeferred(); });
}
bool ArraySpec::IsDeferredShape() const {
  return CheckAll([](const ShapeSpec &x) {
    return x.lbound().isDeferred() && x.ubound().isDeferred();
  });
}
bool ArraySpec::IsImpliedShape() const {
  return CheckAll([](const ShapeSpec &x) { return x.ubound().isAssumed(); });
}
bool ArraySpec::IsAssumedSize() const {
  return !empty() &&
      std::all_of(begin(), end() - 1,
          [](const ShapeSpec &x) { return x.ubound().isExplicit(); }) &&
      back().ubound().isAssumed();
}
bool ArraySpec::IsAssumedRank() const {
  return Rank() == 1 && front().lbound().isAssumed();
}

std::ostream &operator<<(std::ostream &os, const ArraySpec &arraySpec) {
  char sep{'('};
  for (auto &shape : arraySpec) {
    os << sep << shape;
    sep = ',';
  }
  if (sep == ',') {
    os << ')';
  }
  return os;
}

ParamValue::ParamValue(MaybeIntExpr &&expr, common::TypeParamAttr attr)
  : attr_{attr}, expr_{std::move(expr)} {}
ParamValue::ParamValue(SomeIntExpr &&expr, common::TypeParamAttr attr)
  : attr_{attr}, expr_{std::move(expr)} {}
ParamValue::ParamValue(
    common::ConstantSubscript value, common::TypeParamAttr attr)
  : ParamValue(
        SomeIntExpr{evaluate::Expr<evaluate::SubscriptInteger>{value}}, attr) {}

void ParamValue::SetExplicit(SomeIntExpr &&x) {
  category_ = Category::Explicit;
  expr_ = std::move(x);
}

std::string ParamValue::AsFortran() const {
  switch (category_) {
  case Category::Assumed: return "*";
  case Category::Deferred: return ":";
  case Category::Explicit:
    if (expr_) {
      std::stringstream ss;
      expr_->AsFortran(ss);
      return ss.str();
    } else {
      return "";
    }
  default: CRASH_NO_CASE;
  }
}

std::ostream &operator<<(std::ostream &o, const ParamValue &x) {
  return o << x.AsFortran();
}

IntrinsicTypeSpec::IntrinsicTypeSpec(TypeCategory category, KindExpr &&kind)
  : category_{category}, kind_{std::move(kind)} {
  CHECK(category != TypeCategory::Derived);
}

static std::string KindAsFortran(const KindExpr &kind) {
  std::stringstream ss;
  if (auto k{evaluate::ToInt64(kind)}) {
    ss << *k;  // emit unsuffixed kind code
  } else {
    kind.AsFortran(ss);
  }
  return ss.str();
}

std::string IntrinsicTypeSpec::AsFortran() const {
  return parser::ToUpperCaseLetters(common::EnumToString(category_)) + '(' +
      KindAsFortran(kind_) + ')';
}

std::ostream &operator<<(std::ostream &os, const IntrinsicTypeSpec &x) {
  return os << x.AsFortran();
}

std::string CharacterTypeSpec::AsFortran() const {
  return "CHARACTER(" + length_.AsFortran() + ',' + KindAsFortran(kind()) + ')';
}

std::ostream &operator<<(std::ostream &os, const CharacterTypeSpec &x) {
  return os << x.AsFortran();
}

DeclTypeSpec::DeclTypeSpec(NumericTypeSpec &&typeSpec)
  : category_{Numeric}, typeSpec_{std::move(typeSpec)} {}
DeclTypeSpec::DeclTypeSpec(LogicalTypeSpec &&typeSpec)
  : category_{Logical}, typeSpec_{std::move(typeSpec)} {}
DeclTypeSpec::DeclTypeSpec(const CharacterTypeSpec &typeSpec)
  : category_{Character}, typeSpec_{typeSpec} {}
DeclTypeSpec::DeclTypeSpec(CharacterTypeSpec &&typeSpec)
  : category_{Character}, typeSpec_{std::move(typeSpec)} {}
DeclTypeSpec::DeclTypeSpec(Category category, const DerivedTypeSpec &typeSpec)
  : category_{category}, typeSpec_{typeSpec} {
  CHECK(category == TypeDerived || category == ClassDerived);
}
DeclTypeSpec::DeclTypeSpec(Category category, DerivedTypeSpec &&typeSpec)
  : category_{category}, typeSpec_{std::move(typeSpec)} {
  CHECK(category == TypeDerived || category == ClassDerived);
}
DeclTypeSpec::DeclTypeSpec(Category category) : category_{category} {
  CHECK(category == TypeStar || category == ClassStar);
}
bool DeclTypeSpec::IsNumeric(TypeCategory tc) const {
  return category_ == Numeric && numericTypeSpec().category() == tc;
}
IntrinsicTypeSpec *DeclTypeSpec::AsIntrinsic() {
  return const_cast<IntrinsicTypeSpec *>(
      const_cast<const DeclTypeSpec *>(this)->AsIntrinsic());
}
const NumericTypeSpec &DeclTypeSpec::numericTypeSpec() const {
  CHECK(category_ == Numeric);
  return std::get<NumericTypeSpec>(typeSpec_);
}
const LogicalTypeSpec &DeclTypeSpec::logicalTypeSpec() const {
  CHECK(category_ == Logical);
  return std::get<LogicalTypeSpec>(typeSpec_);
}
bool DeclTypeSpec::operator==(const DeclTypeSpec &that) const {
  return category_ == that.category_ && typeSpec_ == that.typeSpec_;
}

std::string DeclTypeSpec::AsFortran() const {
  switch (category_) {
  case Numeric: return numericTypeSpec().AsFortran();
  case Logical: return logicalTypeSpec().AsFortran();
  case Character: return characterTypeSpec().AsFortran();
  case TypeDerived: return "TYPE(" + derivedTypeSpec().AsFortran() + ')';
  case ClassDerived: return "CLASS(" + derivedTypeSpec().AsFortran() + ')';
  case TypeStar: return "TYPE(*)";
  case ClassStar: return "CLASS(*)";
  default: CRASH_NO_CASE;
  }
}

std::ostream &operator<<(std::ostream &o, const DeclTypeSpec &x) {
  return o << x.AsFortran();
}

void ProcInterface::set_symbol(const Symbol &symbol) {
  CHECK(!type_);
  symbol_ = &symbol;
}
void ProcInterface::set_type(const DeclTypeSpec &type) {
  CHECK(!symbol_);
  type_ = &type;
}
}
