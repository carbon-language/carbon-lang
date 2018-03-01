#include "type.h"
#include "attr.h"
#include <iostream>
#include <set>

namespace Fortran {
namespace semantics {

// Check that values specified for param defs are valid: they must match the
// names of the params and any def that doesn't have a default value must have a
// value.
template<typename V>
static void checkParams(
    std::string kindOrLen, TypeParamDefs defs, std::map<Name, V> values) {
  std::set<Name> validNames{};
  for (const TypeParamDef &def : defs) {
    Name name = def.name();
    validNames.insert(name);
    if (!def.defaultValue() && values.find(name) == values.end()) {
      parser::die("no value or default value for %s parameter '%s'",
          kindOrLen.c_str(), name.c_str());
    }
  }
  for (const auto &pair : values) {
    Name name = pair.first;
    if (validNames.find(name) == validNames.end()) {
      parser::die("invalid %s parameter '%s'", kindOrLen.c_str(), name.c_str());
    }
  }
}

std::ostream &operator<<(std::ostream &o, const IntExpr &x) {
  return x.Output(o);
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

const LogicalTypeSpec *LogicalTypeSpec::Make() { return &helper.Make(); }
const LogicalTypeSpec *LogicalTypeSpec::Make(KindParamValue kind) {
  return &helper.Make(kind);
}
KindedTypeHelper<LogicalTypeSpec> LogicalTypeSpec::helper{"LOGICAL", 0};
std::ostream &operator<<(std::ostream &o, const LogicalTypeSpec &x) {
  return LogicalTypeSpec::helper.Output(o, x);
}

const IntegerTypeSpec *IntegerTypeSpec::Make() { return &helper.Make(); }
const IntegerTypeSpec *IntegerTypeSpec::Make(KindParamValue kind) {
  return &helper.Make(kind);
}
KindedTypeHelper<IntegerTypeSpec> IntegerTypeSpec::helper{"INTEGER", 0};
std::ostream &operator<<(std::ostream &o, const IntegerTypeSpec &x) {
  return IntegerTypeSpec::helper.Output(o, x);
}

const RealTypeSpec *RealTypeSpec::Make() { return &helper.Make(); }
const RealTypeSpec *RealTypeSpec::Make(KindParamValue kind) {
  return &helper.Make(kind);
}
KindedTypeHelper<RealTypeSpec> RealTypeSpec::helper{"REAL", 0};
std::ostream &operator<<(std::ostream &o, const RealTypeSpec &x) {
  return RealTypeSpec::helper.Output(o, x);
}

const ComplexTypeSpec *ComplexTypeSpec::Make() { return &helper.Make(); }
const ComplexTypeSpec *ComplexTypeSpec::Make(KindParamValue kind) {
  return &helper.Make(kind);
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

DerivedTypeSpec::DerivedTypeSpec(DerivedTypeDef def,
    const KindParamValues &kindParamValues,
    const LenParamValues &lenParamValues)
  : def_{def}, kindParamValues_{kindParamValues}, lenParamValues_{
                                                      lenParamValues} {
  checkParams("kind", def.kindParams(), kindParamValues);
  checkParams("len", def.lenParams(), lenParamValues);
}

std::ostream &operator<<(std::ostream &o, const DerivedTypeSpec &x) {
  o << "TYPE(" << x.def_.name();
  if (x.kindParamValues_.size() > 0 || x.lenParamValues_.size() > 0) {
    o << '(';
    int n = 0;
    for (const auto &pair : x.kindParamValues_) {
      if (n++) {
        o << ", ";
      }
      o << pair.first << '=' << pair.second;
    }
    for (const auto &pair : x.lenParamValues_) {
      if (n++) {
        o << ", ";
      }
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
    const Attrs &attrs, const ComponentArraySpec &arraySpec)
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

std::ostream &operator<<(std::ostream &o, const DeclTypeSpec &x) {
  // TODO: need CLASS(...) instead of TYPE() for ClassDerived
  switch (x.category_) {
  case DeclTypeSpec::Intrinsic: return x.intrinsicTypeSpec_->Output(o);
  case DeclTypeSpec::TypeDerived: return o << *x.derivedTypeSpec_;
  case DeclTypeSpec::ClassDerived: return o << *x.derivedTypeSpec_;
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
  CHECK(attrs_.Has(Attr::POINTER));
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
  data_.attrs.Set(x);
  return *this;
}
DerivedTypeDefBuilder &DerivedTypeDefBuilder::attrs(const Attrs &x) {
  x.CheckValid({Attr::ABSTRACT, Attr::PUBLIC, Attr::PRIVATE, Attr::BIND_C});
  data_.attrs.Add(x);
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

}  // namespace semantics
}  // namespace Fortran

using namespace Fortran::semantics;

void testTypeSpec() {
  const LogicalTypeSpec *l1 = LogicalTypeSpec::Make();
  const LogicalTypeSpec *l2 = LogicalTypeSpec::Make(2);
  std::cout << *l1 << "\n";
  std::cout << *l2 << "\n";
  const RealTypeSpec *r1 = RealTypeSpec::Make();
  const RealTypeSpec *r2 = RealTypeSpec::Make(2);
  std::cout << *r1 << "\n";
  std::cout << *r2 << "\n";
  const CharacterTypeSpec c1{LenParamValue::DEFERRED, 1};
  std::cout << c1 << "\n";
  const CharacterTypeSpec c2{IntConst::Make(10)};
  std::cout << c2 << "\n";

  const IntegerTypeSpec *i1 = IntegerTypeSpec::Make();
  const IntegerTypeSpec *i2 = IntegerTypeSpec::Make(2);
  TypeParamDef lenParam{"my_len", *i2};
  TypeParamDef kindParam{"my_kind", *i1};

  DerivedTypeDef def1{DerivedTypeDefBuilder("my_name")
                          .attrs({Attr::PRIVATE, Attr::BIND_C})
                          .lenParam(lenParam)
                          .kindParam(kindParam)
                          .sequence()};
  // DerivedTypeDef def1{"my_name", {Attr::PRIVATE, Attr::BIND_C},
  //    TypeParamDefs{lenParam}, TypeParamDefs{kindParam}, false, true};

  LenParamValues lenParamValues{
      LenParamValues::value_type{"my_len", LenParamValue::ASSUMED},
  };
  KindParamValues kindParamValues{
      KindParamValues::value_type{"my_kind", KindParamValue{123}},
  };
  // DerivedTypeSpec dt1{def1, kindParamValues, lenParamValues};

  // DerivedTypeSpec dt1{DerivedTypeSpec::Builder{"my_name2"}
  //  .lenParamValue("my_len", LenParamValue::ASSUMED)
  //  .attrs({Attr::BIND_C}).lenParam(lenParam)};
  // std::cout << dt1 << "\n";
}

void testShapeSpec() {
  const IntConst &ten{IntConst::Make(10)};
  const ShapeSpec s1{ShapeSpec::MakeExplicit(ten)};
  std::cout << "explicit-shape-spec: " << s1 << "\n";
  ShapeSpec s2{ShapeSpec::MakeExplicit(IntConst::Make(2), IntConst::Make(8))};
  std::cout << "explicit-shape-spec: " << s2 << "\n";

  ShapeSpec s3{ShapeSpec::MakeAssumed()};
  std::cout << "assumed-shape-spec:  " << s3 << "\n";
  ShapeSpec s4{ShapeSpec::MakeAssumed(IntConst::Make(2))};
  std::cout << "assumed-shape-spec:  " << s4 << "\n";

  ShapeSpec s5{ShapeSpec::MakeDeferred()};
  std::cout << "deferred-shape-spec: " << s5 << "\n";

  ShapeSpec s6{ShapeSpec::MakeImplied(IntConst::Make(2))};
  std::cout << "implied-shape-spec:  " << s6 << "\n";

  ShapeSpec s7{ShapeSpec::MakeAssumedRank()};
  std::cout << "assumed-rank-spec:  " << s7 << "\n";
}

void testDataComponentDef() {
  DataComponentDef def1{
      DeclTypeSpec::MakeClassStar(), "foo", Attrs{Attr::PUBLIC}};
  std::cout << "data-component-def: " << def1 << "\n";
  DataComponentDef def2{DeclTypeSpec::MakeTypeStar(), "foo", Attrs{},
      ComponentArraySpec{ShapeSpec::MakeExplicit(IntConst::Make(10))}};
  std::cout << "data-component-def: " << def2 << "\n";
}

void testProcComponentDef() {
  ProcDecl decl{"foo"};
  ProcComponentDef def1{decl, Attrs{Attr::POINTER, Attr::PUBLIC, Attr::NOPASS}};
  std::cout << "proc-component-def: " << def1;
  ProcComponentDef def2{decl, Attrs{Attr::POINTER}, Name{"my_interface"}};
  std::cout << "proc-component-def: " << def2;
  ProcComponentDef def3{
      decl, Attrs{Attr::POINTER}, DeclTypeSpec::MakeTypeStar()};
  std::cout << "proc-component-def: " << def3;
}

#if 0
int main() {
  testTypeSpec();
  //testShapeSpec();
  //testProcComponentDef();
  //testDataComponentDef();
  return 0;
}
#endif
