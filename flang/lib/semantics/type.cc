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
#include "attr.h"
#include <iostream>
#include <set>

namespace Fortran::semantics {

std::ostream &operator<<(std::ostream &o, const IntExpr &x) {
  return x.Output(o);
}
std::ostream &operator<<(std::ostream &o, const IntConst &x) {
  return o << x.value_;
}

std::unordered_map<std::uint64_t, IntConst> IntConst::cache;

std::ostream &operator<<(std::ostream &o, const KindParamValue &x) {
  return o << x.value_;
}

const IntConst &IntConst::Make(std::uint64_t value) {
  auto it = cache.find(value);
  if (it == cache.end()) {
    it = cache.insert({value, IntConst{value}}).first;
  }
  return it->second;
}

const LogicalTypeSpec &LogicalTypeSpec::Make() { return helper.Make(); }
const LogicalTypeSpec &LogicalTypeSpec::Make(KindParamValue kind) {
  return helper.Make(kind);
}
KindedTypeHelper<LogicalTypeSpec> LogicalTypeSpec::helper{"LOGICAL", 0};
std::ostream &operator<<(std::ostream &o, const LogicalTypeSpec &x) {
  return LogicalTypeSpec::helper.Output(o, x);
}

const IntegerTypeSpec &IntegerTypeSpec::Make() { return helper.Make(); }
const IntegerTypeSpec &IntegerTypeSpec::Make(KindParamValue kind) {
  return helper.Make(kind);
}
KindedTypeHelper<IntegerTypeSpec> IntegerTypeSpec::helper{"INTEGER", 0};
std::ostream &operator<<(std::ostream &o, const IntegerTypeSpec &x) {
  return IntegerTypeSpec::helper.Output(o, x);
}

const RealTypeSpec &RealTypeSpec::Make() { return helper.Make(); }
const RealTypeSpec &RealTypeSpec::Make(KindParamValue kind) {
  return helper.Make(kind);
}
KindedTypeHelper<RealTypeSpec> RealTypeSpec::helper{"REAL", 0};
std::ostream &operator<<(std::ostream &o, const RealTypeSpec &x) {
  return RealTypeSpec::helper.Output(o, x);
}

const ComplexTypeSpec &ComplexTypeSpec::Make() { return helper.Make(); }
const ComplexTypeSpec &ComplexTypeSpec::Make(KindParamValue kind) {
  return helper.Make(kind);
}
KindedTypeHelper<ComplexTypeSpec> ComplexTypeSpec::helper{"COMPLEX", 0};
std::ostream &operator<<(std::ostream &o, const ComplexTypeSpec &x) {
  return ComplexTypeSpec::helper.Output(o, x);
}

std::ostream &operator<<(std::ostream &o, const CharacterTypeSpec &x) {
  o << "CHARACTER(" << x.len_;
  if (x.kind_ != CharacterTypeSpec::DefaultKind) {
    o << ", " << x.kind_;
  }
  return o << ')';
}

std::ostream &operator<<(std::ostream &o, const DerivedTypeDef &x) {
  o << "TYPE";
  if (!x.data_.attrs.empty()) {
    o << ", " << x.data_.attrs;
  }
  o << " :: " << x.data_.name;
  if (x.data_.lenParams.size() > 0 || x.data_.kindParams.size() > 0) {
    o << '(';
    int n = 0;
    for (const auto &param : x.data_.lenParams) {
      if (n++) {
        o << ", ";
      }
      o << param.name();
    }
    for (auto param : x.data_.kindParams) {
      if (n++) {
        o << ", ";
      }
      o << param.name();
    }
    o << ')';
  }
  o << '\n';
  for (const auto &param : x.data_.lenParams) {
    o << "  " << param.type() << ", LEN :: " << param.name() << "\n";
  }
  for (const auto &param : x.data_.kindParams) {
    o << "  " << param.type() << ", KIND :: " << param.name() << "\n";
  }
  if (x.data_.Private) {
    o << "  PRIVATE\n";
  }
  if (x.data_.sequence) {
    o << "  SEQUENCE\n";
  }
  for (const auto &comp : x.data_.dataComps) {
    o << "  " << comp << "\n";
  }
  for (const auto &comp : x.data_.procComps) {
    o << "  " << comp << "\n";
  }
  return o << "END TYPE";
}

std::ostream &operator<<(std::ostream &o, const DerivedTypeSpec &x) {
  o << "TYPE(" << x.name_;
  if (!x.paramValues_.empty()) {
    o << '(';
    int n = 0;
    for (const auto &paramValue : x.paramValues_) {
      if (n++) {
        o << ", ";
      }
      if (paramValue.first) {
        o << *paramValue.first << '=';
      }
      o << paramValue.second;
    }
    o << ')';
  }
  o << ')';
  return o;
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

std::ostream &operator<<(std::ostream &o, const DataComponentDef &x) {
  o << x.type_;
  if (!x.attrs_.empty()) {
    o << ", " << x.attrs_;
  }
  o << " :: " << x.name_;
  if (!x.arraySpec_.empty()) {
    o << '(';
    int n = 0;
    for (ShapeSpec shape : x.arraySpec_) {
      if (n++) {
        o << ", ";
      }
      o << shape;
    }
    o << ')';
  }
  return o;
}

DataComponentDef::DataComponentDef(const DeclTypeSpec &type, const Name &name,
    const Attrs &attrs, const ArraySpec &arraySpec)
  : type_{type}, name_{name}, attrs_{attrs}, arraySpec_{arraySpec} {
  attrs.CheckValid({Attr::PUBLIC, Attr::PRIVATE, Attr::ALLOCATABLE,
      Attr::POINTER, Attr::CONTIGUOUS});
  if (attrs.HasAny({Attr::ALLOCATABLE, Attr::POINTER})) {
    for (const auto &shapeSpec : arraySpec) {
      CHECK(shapeSpec.isDeferred());
    }
  } else {
    for (const auto &shapeSpec : arraySpec) {
      CHECK(shapeSpec.isExplicit());
    }
  }
}

DeclTypeSpec::DeclTypeSpec(const DeclTypeSpec &that)
  : category_{that.category_}, intrinsicTypeSpec_{that.intrinsicTypeSpec_} {
  if (category_ == TypeDerived || category_ == ClassDerived) {
    derivedTypeSpec_ =
        std::make_unique<DerivedTypeSpec>(*that.derivedTypeSpec_);
  }
}

DeclTypeSpec &DeclTypeSpec::operator=(const DeclTypeSpec &that) {
  category_ = that.category_;
  intrinsicTypeSpec_ = that.intrinsicTypeSpec_;
  if (category_ == TypeDerived || category_ == ClassDerived) {
    derivedTypeSpec_ =
        std::make_unique<DerivedTypeSpec>(*that.derivedTypeSpec_);
  }
  return *this;
}

DeclTypeSpec::DeclTypeSpec(
    Category category, std::unique_ptr<DerivedTypeSpec> &&typeSpec)
  : category_{category}, intrinsicTypeSpec_{nullptr}, derivedTypeSpec_{
                                                          std::move(typeSpec)} {
  CHECK(category == TypeDerived || category == ClassDerived);
}

std::ostream &operator<<(std::ostream &o, const DeclTypeSpec &x) {
  // TODO: need CLASS(...) instead of TYPE() for ClassDerived
  switch (x.category_) {
  case DeclTypeSpec::Intrinsic: return x.intrinsicTypeSpec().Output(o);
  case DeclTypeSpec::TypeDerived: return o << x.derivedTypeSpec();
  case DeclTypeSpec::ClassDerived: return o << x.derivedTypeSpec();
  case DeclTypeSpec::TypeStar: return o << "TYPE(*)";
  case DeclTypeSpec::ClassStar: return o << "CLASS(*)";
  default: CRASH_NO_CASE;
  }
}

std::ostream &operator<<(std::ostream &o, const ProcDecl &x) {
  return o << x.name_;
}

ProcComponentDef::ProcComponentDef(ProcDecl decl, Attrs attrs,
    const std::optional<Name> &interfaceName,
    const std::optional<DeclTypeSpec> &typeSpec)
  : decl_{decl}, attrs_{attrs}, interfaceName_{interfaceName}, typeSpec_{
                                                                   typeSpec} {
  CHECK(attrs_.test(Attr::POINTER));
  attrs_.CheckValid(
      {Attr::PUBLIC, Attr::PRIVATE, Attr::NOPASS, Attr::POINTER, Attr::PASS});
  CHECK(!interfaceName || !typeSpec);  // can't both be defined
}
std::ostream &operator<<(std::ostream &o, const ProcComponentDef &x) {
  o << "PROCEDURE(";
  if (x.interfaceName_) {
    o << *x.interfaceName_;
  } else if (x.typeSpec_) {
    o << *x.typeSpec_;
  }
  o << "), " << x.attrs_ << " :: " << x.decl_;
  return o;
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
  default: CRASH_NO_CASE;
  }
}

DerivedTypeDef::DerivedTypeDef(const DerivedTypeDef::Data &data)
  : data_{data} {}

DerivedTypeDefBuilder &DerivedTypeDefBuilder::name(const Name &x) {
  data_.name = x;
  return *this;
}
DerivedTypeDefBuilder &DerivedTypeDefBuilder::extends(const Name &x) {
  data_.extends = x;
  return *this;
}
DerivedTypeDefBuilder &DerivedTypeDefBuilder::attr(const Attr &x) {
  // TODO: x.CheckValid({Attr::ABSTRACT, Attr::PUBLIC, Attr::PRIVATE,
  // Attr::BIND_C});
  data_.attrs.set(x);
  return *this;
}
DerivedTypeDefBuilder &DerivedTypeDefBuilder::attrs(const Attrs &x) {
  x.CheckValid({Attr::ABSTRACT, Attr::PUBLIC, Attr::PRIVATE, Attr::BIND_C});
  data_.attrs |= x;
  return *this;
}
DerivedTypeDefBuilder &DerivedTypeDefBuilder::lenParam(const TypeParamDef &x) {
  data_.lenParams.push_back(x);
  return *this;
}
DerivedTypeDefBuilder &DerivedTypeDefBuilder::kindParam(const TypeParamDef &x) {
  data_.kindParams.push_back(x);
  return *this;
}
DerivedTypeDefBuilder &DerivedTypeDefBuilder::dataComponent(
    const DataComponentDef &x) {
  data_.dataComps.push_back(x);
  return *this;
}
DerivedTypeDefBuilder &DerivedTypeDefBuilder::procComponent(
    const ProcComponentDef &x) {
  data_.procComps.push_back(x);
  return *this;
}
DerivedTypeDefBuilder &DerivedTypeDefBuilder::Private(bool x) {
  data_.Private = x;
  return *this;
}
DerivedTypeDefBuilder &DerivedTypeDefBuilder::sequence(bool x) {
  data_.sequence = x;
  return *this;
}

}  // namespace Fortran::semantics
