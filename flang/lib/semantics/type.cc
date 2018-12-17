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

#include "type.h"
#include "expression.h"
#include "scope.h"
#include "semantics.h"
#include "symbol.h"
#include "../evaluate/fold.h"
#include "../evaluate/tools.h"
#include "../evaluate/type.h"
#include "../parser/characters.h"

namespace Fortran::semantics {

void DerivedTypeSpec::set_scope(const Scope &scope) {
  CHECK(!scope_);
  CHECK(scope.kind() == Scope::Kind::DerivedType);
  scope_ = &scope;
}

void DerivedTypeSpec::AddParamValue(ParamValue &&value) {
  paramValues_.emplace_back(std::nullopt, std::move(value));
}
void DerivedTypeSpec::AddParamValue(
    const SourceName &name, ParamValue &&value) {
  paramValues_.emplace_back(name, std::move(value));
}

std::ostream &operator<<(std::ostream &o, const DerivedTypeSpec &x) {
  o << x.name().ToString();
  if (!x.paramValues_.empty()) {
    bool first = true;
    o << '(';
    for (const auto &[name, value] : x.paramValues_) {
      if (first) {
        first = false;
      } else {
        o << ',';
      }
      if (name) {
        o << name->ToString() << '=';
      }
      o << value;
    }
    o << ')';
  }
  return o;
}

Bound::Bound(int bound)
  : category_{Category::Explicit},
    expr_{evaluate::Expr<evaluate::SubscriptInteger>{bound}} {}

Bound Bound::Clone() const { return Bound(category_, MaybeExpr{expr_}); }

std::ostream &operator<<(std::ostream &o, const Bound &x) {
  if (x.isAssumed()) {
    o << '*';
  } else if (x.isDeferred()) {
    o << ':';
  } else if (x.expr_) {
    x.expr_->AsFortran(o);
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

ParamValue::ParamValue(MaybeExpr &&expr)
  : category_{Category::Explicit}, expr_{std::move(expr)} {}
ParamValue::ParamValue(std::int64_t value)
  : ParamValue(SomeExpr{evaluate::Expr<evaluate::SubscriptInteger>{value}}) {}

std::ostream &operator<<(std::ostream &o, const ParamValue &x) {
  if (x.isAssumed()) {
    o << '*';
  } else if (x.isDeferred()) {
    o << ':';
  } else if (!x.GetExplicit()) {
    o << "<no-expr>";
  } else {
    x.GetExplicit()->AsFortran(o);
  }
  return o;
}

IntrinsicTypeSpec::IntrinsicTypeSpec(TypeCategory category, int kind)
  : category_{category}, kind_{kind} {
  CHECK(category != TypeCategory::Derived);
  CHECK(kind > 0);
}

std::ostream &operator<<(std::ostream &os, const IntrinsicTypeSpec &x) {
  os << parser::ToUpperCaseLetters(common::EnumToString(x.category()));
  if (x.kind() != 0) {
    os << '(' << x.kind() << ')';
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const CharacterTypeSpec &x) {
  return os << "CHARACTER(" << x.length() << ',' << x.kind() << ')';
}

DeclTypeSpec::DeclTypeSpec(const NumericTypeSpec &typeSpec)
  : category_{Numeric}, typeSpec_{typeSpec} {}
DeclTypeSpec::DeclTypeSpec(const LogicalTypeSpec &typeSpec)
  : category_{Logical}, typeSpec_{typeSpec} {}
DeclTypeSpec::DeclTypeSpec(CharacterTypeSpec &typeSpec)
  : category_{Character}, typeSpec_{&typeSpec} {}
DeclTypeSpec::DeclTypeSpec(Category category, DerivedTypeSpec &typeSpec)
  : category_{category}, typeSpec_{&typeSpec} {
  CHECK(category == TypeDerived || category == ClassDerived);
}
DeclTypeSpec::DeclTypeSpec(Category category) : category_{category} {
  CHECK(category == TypeStar || category == ClassStar);
}
bool DeclTypeSpec::IsNumeric(TypeCategory tc) const {
  return category_ == Numeric && numericTypeSpec().category() == tc;
}
const IntrinsicTypeSpec *DeclTypeSpec::AsIntrinsic() const {
  switch (category_) {
  case Numeric: return &typeSpec_.numeric;
  case Logical: return &typeSpec_.logical;
  case Character: return typeSpec_.character;
  default: return nullptr;
  }
}
const NumericTypeSpec &DeclTypeSpec::numericTypeSpec() const {
  CHECK(category_ == Numeric);
  return typeSpec_.numeric;
}
const LogicalTypeSpec &DeclTypeSpec::logicalTypeSpec() const {
  CHECK(category_ == Logical);
  return typeSpec_.logical;
}
const CharacterTypeSpec &DeclTypeSpec::characterTypeSpec() const {
  CHECK(category_ == Character);
  return *typeSpec_.character;
}
DerivedTypeSpec &DeclTypeSpec::derivedTypeSpec() {
  CHECK(category_ == TypeDerived || category_ == ClassDerived);
  return *typeSpec_.derived;
}
const DerivedTypeSpec &DeclTypeSpec::derivedTypeSpec() const {
  CHECK(category_ == TypeDerived || category_ == ClassDerived);
  return *typeSpec_.derived;
}
bool DeclTypeSpec::operator==(const DeclTypeSpec &that) const {
  if (category_ != that.category_) {
    return false;
  }
  switch (category_) {
  case Numeric: return typeSpec_.numeric == that.typeSpec_.numeric;
  case Logical: return typeSpec_.logical == that.typeSpec_.logical;
  case Character: return typeSpec_.character == that.typeSpec_.character;
  case TypeDerived:
  case ClassDerived: return typeSpec_.derived == that.typeSpec_.derived;
  default: return true;
  }
}

std::ostream &operator<<(std::ostream &o, const DeclTypeSpec &x) {
  switch (x.category()) {
  case DeclTypeSpec::Numeric: return o << x.numericTypeSpec();
  case DeclTypeSpec::Logical: return o << x.logicalTypeSpec();
  case DeclTypeSpec::Character: return o << x.characterTypeSpec();
  case DeclTypeSpec::TypeDerived:
    return o << "TYPE(" << x.derivedTypeSpec() << ')';
  case DeclTypeSpec::ClassDerived:
    return o << "CLASS(" << x.derivedTypeSpec() << ')';
  case DeclTypeSpec::TypeStar: return o << "TYPE(*)";
  case DeclTypeSpec::ClassStar: return o << "CLASS(*)";
  default: CRASH_NO_CASE; return o;
  }
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
