#include <iostream>

#include "type.h"

namespace Fortran {

// Check that values specified for param defs are valid: they must match the
// names of the params and any def that doesn't have a default value must have a
// value.
template<typename V>
static void checkParams(
    std::string kindOrLen, TypeParamDefs defs, std::map<Name, V> values) {
  std::set<Name> validNames{};
  for (TypeParamDef def : defs) {
    Name name = def.name();
    validNames.insert(name);
    if (!def.defaultValue() && values.find(name) == values.end()) {
      die("no value or default value for %s parameter '%s'", kindOrLen.c_str(),
          name.c_str());
    }
  }
  for (auto pair : values) {
    Name name = pair.first;
    if (validNames.find(name) == validNames.end()) {
      die("invalid %s parameter '%s'", kindOrLen.c_str(), name.c_str());
    }
  }
}

const IntConst IntConst::ZERO = IntConst{0};
const IntConst IntConst::ONE = IntConst{1};

const IntExpr *IntConst::clone() const {
  if (*this == ZERO) {
    return &ZERO;
  } else if (*this == ONE) {
    return &ONE;
  } else {
    return new IntConst{*this};
  }
}

std::ostream &operator<<(std::ostream &o, const KindParamValue &x) {
  return o << x.value_;
}

const LenParamValue LenParamValue::ASSUMED =
    LenParamValue(LenParamValue::Assumed);
const LenParamValue LenParamValue::DEFERRED =
    LenParamValue(LenParamValue::Deferred);

std::ostream &operator<<(std::ostream &o, const LenParamValue &x) {
  switch (x.category_) {
  case LenParamValue::Assumed: return o << '*';
  case LenParamValue::Deferred: return o << ':';
  case LenParamValue::Expr: return o << *x.value_;
  default: CRASH_NO_CASE;
  }
}

KindedTypeHelper<LogicalTypeSpec> LogicalTypeSpec::helper{"LOGICAL", 0};
std::ostream &operator<<(std::ostream &o, const LogicalTypeSpec &x) {
  return LogicalTypeSpec::helper.output(o, x);
}

KindedTypeHelper<IntegerTypeSpec> IntegerTypeSpec::helper{"INTEGER", 0};
std::ostream &operator<<(std::ostream &o, const IntegerTypeSpec &x) {
  return IntegerTypeSpec::helper.output(o, x);
}

KindedTypeHelper<RealTypeSpec> RealTypeSpec::helper{"REAL", 0};
std::ostream &operator<<(std::ostream &o, const RealTypeSpec &x) {
  return RealTypeSpec::helper.output(o, x);
}

KindedTypeHelper<ComplexTypeSpec> ComplexTypeSpec::helper{"COMPLEX", 0};
std::ostream &operator<<(std::ostream &o, const ComplexTypeSpec &x) {
  return ComplexTypeSpec::helper.output(o, x);
}

std::ostream &operator<<(std::ostream &o, const CharacterTypeSpec &x) {
  o << "CHARACTER(" << x.len_;
  if (x.kind_ != CharacterTypeSpec::DefaultKind) {
    o << ", " << x.kind_;
  }
  return o << ')';
}

DerivedTypeDef::DerivedTypeDef(const Name &name, const Attrs &attrs,
    const TypeParamDefs &lenParams, const TypeParamDefs &kindParams,
    bool private_, bool sequence)
  : name_{name}, attrs_{attrs}, lenParams_{lenParams},
    kindParams_{kindParams}, private_{private_}, sequence_{sequence} {
  checkAttrs("DerivedTypeDef", attrs,
      Attrs{Attr::ABSTRACT, Attr::PUBLIC, Attr::PRIVATE, Attr::BIND_C});
}

std::ostream &operator<<(std::ostream &o, const DerivedTypeDef &x) {
  o << "TYPE";
  for (auto attr : x.attrs_) {
    o << ", " << attr;
  }
  o << " :: " << x.name_;
  if (x.lenParams_.size() > 0 || x.kindParams_.size() > 0) {
    o << '(';
    int n = 0;
    for (auto param : x.lenParams_) {
      if (n++) o << ", ";
      o << param.name();
    }
    for (auto param : x.kindParams_) {
      if (n++) o << ", ";
      o << param.name();
    }
    o << ')';
  }
  o << '\n';
  for (auto param : x.lenParams_) {
    o << "  " << param.type() << ", LEN :: " << param.name() << "\n";
  }
  for (auto param : x.kindParams_) {
    o << "  " << param.type() << ", KIND :: " << param.name() << "\n";
  }
  if (x.private_) o << "  PRIVATE\n";
  if (x.sequence_) o << "  SEQUENCE\n";
  // components
  return o << "END TYPE\n";
}

DerivedTypeSpec::DerivedTypeSpec(DerivedTypeDef def,
    KindParamValues kindParamValues, LenParamValues lenParamValues)
  : def_{def}, kindParamValues_{kindParamValues}, lenParamValues_{
                                                      lenParamValues} {
  checkParams("kind", def.kindParams_, kindParamValues);
  checkParams("len", def.lenParams_, lenParamValues);
}

std::ostream &operator<<(std::ostream &o, const DerivedTypeSpec &x) {
  o << "TYPE(" << x.def_.name_;
  if (x.kindParamValues_.size() > 0 || x.lenParamValues_.size() > 0) {
    o << '(';
    int n = 0;
    for (auto pair : x.kindParamValues_) {
      if (n++) o << ", ";
      o << pair.first << '=' << pair.second;
    }
    for (auto pair : x.lenParamValues_) {
      if (n++) o << ", ";
      o << pair.first << '=' << pair.second;
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
    x.expr_->output(o);
  }
  return o;
}

std::ostream &operator<<(std::ostream &o, const ShapeSpec &x) {
  if (x.lb_.isAssumed()) {
    CHECK(x.ub_.isAssumed());
    o << "..";
  } else {
    if (!x.lb_.isDeferred()) o << x.lb_;
    o << ':';
    if (!x.ub_.isDeferred()) o << x.ub_;
  }
  return o;
}

}  // namespace Fortran

using namespace Fortran;

void testTypeSpec() {
  LogicalTypeSpec l1 = LogicalTypeSpec::make();
  LogicalTypeSpec l2 = LogicalTypeSpec::make(2);
  std::cout << l1 << "\n";
  std::cout << l2 << "\n";
  RealTypeSpec r1 = RealTypeSpec::make();
  RealTypeSpec r2 = RealTypeSpec::make(2);
  std::cout << r1 << "\n";
  std::cout << r2 << "\n";
  CharacterTypeSpec c1{LenParamValue::DEFERRED, 1};
  std::cout << c1 << "\n";
  CharacterTypeSpec c2{IntConst{10}};
  std::cout << c2 << "\n";

  IntegerTypeSpec i1 = IntegerTypeSpec::make();
  IntegerTypeSpec i2 = IntegerTypeSpec::make(2);
  TypeParamDef lenParam{"my_len", i2};
  TypeParamDef kindParam{"my_kind", i1};

  DerivedTypeDef def1{
    "my_name",
    {Attr::PRIVATE, Attr::BIND_C},
    TypeParamDefs{lenParam},
    TypeParamDefs{kindParam},
    sequence : true
  };

  LenParamValues lenParamValues{
      LenParamValues::value_type{"my_len", LenParamValue::ASSUMED},
  };
  KindParamValues kindParamValues{
      KindParamValues::value_type{"my_kind", KindParamValue{123}},
  };
  DerivedTypeSpec dt1{def1, kindParamValues, lenParamValues};
  std::cout << dt1 << "\n";
}

void testShapeSpec() {
  IntConst ten{10};
  const ShapeSpec s1{ShapeSpec::makeExplicit(ten)};
  std::cout << "explicit-shape-spec: " << s1 << "\n";
  ShapeSpec s2{ShapeSpec::makeExplicit(IntConst{2}, IntConst{8})};
  std::cout << "explicit-shape-spec: " << s2 << "\n";

  ShapeSpec s3{ShapeSpec::makeAssumed()};
  std::cout << "assumed-shape-spec:  " << s3 << "\n";
  ShapeSpec s4{ShapeSpec::makeAssumed(IntConst{2})};
  std::cout << "assumed-shape-spec:  " << s4 << "\n";

  ShapeSpec s5{ShapeSpec::makeDeferred()};
  std::cout << "deferred-shape-spec: " << s5 << "\n";

  ShapeSpec s6{ShapeSpec::makeImplied(IntConst{2})};
  std::cout << "implied-shape-spec:  " << s6 << "\n";

  ShapeSpec s7{ShapeSpec::makeAssumedRank()};
  std::cout << "assumed-rank-spec:  " << s7 << "\n";
}

int main() {
  testTypeSpec();
  testShapeSpec();
  return 0;
}
