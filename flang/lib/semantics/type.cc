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
#include "scope.h"
#include "symbol.h"
#include "../evaluate/type.h"
#include "../parser/characters.h"

namespace Fortran::semantics {

IntExpr::~IntExpr() {}

std::ostream &operator<<(std::ostream &o, const IntExpr &x) {
  return x.Output(o);
}
std::ostream &operator<<(std::ostream &o, const IntConst &x) {
  return o << x.value_;
}

std::unordered_map<std::uint64_t, IntConst> IntConst::cache;

const IntConst &IntConst::Make(std::uint64_t value) {
  auto it{cache.find(value)};
  if (it == cache.end()) {
    it = cache.insert({value, IntConst{value}}).first;
  }
  return it->second;
}

void DerivedTypeSpec::set_scope(const Scope &scope) {
  CHECK(!scope_);
  CHECK(scope.kind() == Scope::Kind::DerivedType);
  scope_ = &scope;
}

std::ostream &operator<<(std::ostream &o, const DerivedTypeSpec &x) {
  return o << "TYPE(" << x.name().ToString() << ')';
}

const Bound Bound::ASSUMED{Bound::Assumed};
const Bound Bound::DEFERRED{Bound::Deferred};

std::ostream &operator<<(std::ostream &o, const Bound &x) {
  if (x.isAssumed()) {
    o << '*';
  } else if (x.isDeferred()) {
    o << ':';
  } else {
    x.expr_->Output(o);
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

IntrinsicTypeSpec::IntrinsicTypeSpec(TypeCategory category, int kind)
  : category_{category}, kind_{kind ? kind : GetDefaultKind(category)} {
  CHECK(category != TypeCategory::Derived);
}

int IntrinsicTypeSpec::GetDefaultKind(TypeCategory category) {
  switch (category) {
  case TypeCategory::Character: return evaluate::DefaultCharacter::kind;
  //case TypeCategory::Complex: return evaluate::DefaultComplex::kind;
  case TypeCategory::Complex: return 4;  // TEMP to work around bug
  case TypeCategory::Integer: return evaluate::DefaultInteger::kind;
  case TypeCategory::Logical: return evaluate::DefaultLogical::kind;
  case TypeCategory::Real: return evaluate::DefaultReal::kind;
  default: CRASH_NO_CASE;
  }
}

std::ostream &operator<<(std::ostream &os, const IntrinsicTypeSpec &x) {
  os << parser::ToUpperCaseLetters(common::EnumToString(x.category()));
  if (x.kind() != 0) {
    os << '(' << x.kind() << ')';
  }
  return os;
}

DeclTypeSpec::DeclTypeSpec(const IntrinsicTypeSpec &intrinsic)
  : category_{Intrinsic}, typeSpec_{intrinsic} {}
DeclTypeSpec::DeclTypeSpec(Category category, DerivedTypeSpec &derived)
  : category_{category}, typeSpec_{&derived} {
  CHECK(category == TypeDerived || category == ClassDerived);
}
DeclTypeSpec::DeclTypeSpec(Category category) : category_{category} {
  CHECK(category == TypeStar || category == ClassStar);
}
const IntrinsicTypeSpec &DeclTypeSpec::intrinsicTypeSpec() const {
  CHECK(category_ == Intrinsic);
  return typeSpec_.intrinsic;
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
  case Intrinsic: return typeSpec_.intrinsic == that.typeSpec_.intrinsic;
  case TypeDerived:
  case ClassDerived: return typeSpec_.derived == that.typeSpec_.derived;
  default: return true;
  }
}

std::ostream &operator<<(std::ostream &o, const DeclTypeSpec &x) {
  switch (x.category()) {
  case DeclTypeSpec::Intrinsic: return o << x.intrinsicTypeSpec();
  case DeclTypeSpec::TypeDerived:
    return o << "TYPE(" << x.derivedTypeSpec().name().ToString() << ')';
  case DeclTypeSpec::ClassDerived:
    return o << "CLASS(" << x.derivedTypeSpec().name().ToString() << ')';
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
  type_ = type;
}

std::ostream &operator<<(std::ostream &o, const GenericSpec &x) {
  switch (x.kind()) {
  case GenericSpec::GENERIC_NAME: return o << x.genericName().ToString();
  case GenericSpec::OP_DEFINED:
    return o << '(' << x.definedOp().ToString() << ')';
  case GenericSpec::ASSIGNMENT: return o << "ASSIGNMENT(=)";
  case GenericSpec::READ_FORMATTED: return o << "READ(FORMATTED)";
  case GenericSpec::READ_UNFORMATTED: return o << "READ(UNFORMATTED)";
  case GenericSpec::WRITE_FORMATTED: return o << "WRITE(FORMATTED)";
  case GenericSpec::WRITE_UNFORMATTED: return o << "WRITE(UNFORMATTED)";
  case GenericSpec::OP_ADD: return o << "OPERATOR(+)";
  case GenericSpec::OP_CONCAT: return o << "OPERATOR(//)";
  case GenericSpec::OP_DIVIDE: return o << "OPERATOR(/)";
  case GenericSpec::OP_MULTIPLY: return o << "OPERATOR(*)";
  case GenericSpec::OP_POWER: return o << "OPERATOR(**)";
  case GenericSpec::OP_SUBTRACT: return o << "OPERATOR(-)";
  case GenericSpec::OP_AND: return o << "OPERATOR(.AND.)";
  case GenericSpec::OP_EQ: return o << "OPERATOR(.EQ.)";
  case GenericSpec::OP_EQV: return o << "OPERATOR(.EQV.)";
  case GenericSpec::OP_GE: return o << "OPERATOR(.GE.)";
  case GenericSpec::OP_GT: return o << "OPERATOR(.GT.)";
  case GenericSpec::OP_LE: return o << "OPERATOR(.LE.)";
  case GenericSpec::OP_LT: return o << "OPERATOR(.LT.)";
  case GenericSpec::OP_NE: return o << "OPERATOR(.NE.)";
  case GenericSpec::OP_NEQV: return o << "OPERATOR(.NEQV.)";
  case GenericSpec::OP_NOT: return o << "OPERATOR(.NOT.)";
  case GenericSpec::OP_OR: return o << "OPERATOR(.OR.)";
  case GenericSpec::OP_XOR: return o << "OPERATOR(.XOR.)";
  default: CRASH_NO_CASE;
  }
}

}  // namespace Fortran::semantics
