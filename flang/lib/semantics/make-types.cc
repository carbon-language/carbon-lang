#include "attr.h"
#include "type.h"

#include "../parser/idioms.h"
#include "../parser/parse-tree.h"
#include "../parser/parse-tree-visitor.h"
#include <iostream>
#include <set>

namespace Fortran {
namespace semantics {

static KindParamValue GetKindParamValue(
    const std::optional<parser::KindSelector> &kind);
static Bound GetBound(const parser::SpecificationExpr &x);

class MakeTypesVisitor {
public:
  // Create an UnparseVisitor that emits the Fortran to this ostream.
  MakeTypesVisitor(std::ostream &out) : out_{out} {}

  // Default action for a parse tree node is to visit children.
  template<typename T> bool Pre(const T &x) { return true; }
  template<typename T> void Post(const T &) {}

  bool Pre(const parser::TypeDeclarationStmt &x) {
    return true;
  }
  void Post(const parser::TypeDeclarationStmt &x) {
    CHECK(declTypeSpec_ != nullptr);
    out_ << *declTypeSpec_ << "\n\n";
    delete declTypeSpec_;
    declTypeSpec_ = nullptr;
  }

  bool Pre(const parser::DerivedTypeDef &x) {
    builder_ = new DerivedTypeDefBuilder{};
    return true;
  }
  void Post(const parser::DerivedTypeDef &x) {
    DerivedTypeDef derivedType{*builder_};
    out_ << derivedType << "\n\n";
    delete builder_;
    builder_ = nullptr;
  }

  bool Pre(const parser::TypeAttrSpec::Extends &x) {
    builder_->extends(x.v);
    return false;
  }
  bool Pre(const parser::AccessSpec &x) {
    switch (x.v) {
    case parser::AccessSpec::Kind::Public: attrs_->Set(Attr::PUBLIC); break;
    case parser::AccessSpec::Kind::Private: attrs_->Set(Attr::PRIVATE); break;
    default: CRASH_NO_CASE;
    }
    return false;
  }
  bool Pre(const parser::TypeAttrSpec::BindC &x) {
    attrs_->Set(Attr::BIND_C);
    return false;
  }
  bool Pre(const parser::Abstract &x) {
    attrs_->Set(Attr::ABSTRACT);
    return false;
  }
  bool Pre(const parser::Allocatable &) {
    attrs_->Set(Attr::ALLOCATABLE);
    return false;
  }
  bool Pre(const parser::Asynchronous &) {
    attrs_->Set(Attr::ASYNCHRONOUS);
    return false;
  }
  bool Pre(const parser::Contiguous &) {
    attrs_->Set(Attr::CONTIGUOUS);
    return false;
  }
  bool Pre(const parser::External &) {
    attrs_->Set(Attr::EXTERNAL);
    return false;
  }
  bool Pre(const parser::Intrinsic &) {
    attrs_->Set(Attr::INTRINSIC);
    return false;
  }
  bool Pre(const parser::NoPass &) {
    attrs_->Set(Attr::NOPASS);
    return false;
  }
  bool Pre(const parser::Optional &) {
    attrs_->Set(Attr::OPTIONAL);
    return false;
  }
  bool Pre(const parser::Parameter &) {
    attrs_->Set(Attr::PARAMETER);
    return false;
  }
  bool Pre(const parser::Pass &) {
    attrs_->Set(Attr::PASS);
    return false;
  }
  bool Pre(const parser::Pointer &) {
    attrs_->Set(Attr::POINTER);
    return false;
  }
  bool Pre(const parser::Protected &) {
    attrs_->Set(Attr::PROTECTED);
    return false;
  }
  bool Pre(const parser::Save &) {
    attrs_->Set(Attr::SAVE);
    return false;
  }
  bool Pre(const parser::Target &) {
    attrs_->Set(Attr::TARGET);
    return false;
  }
  bool Pre(const parser::Value &) {
    attrs_->Set(Attr::VALUE);
    return false;
  }
  bool Pre(const parser::Volatile &) {
    attrs_->Set(Attr::VOLATILE);
    return false;
  }
  bool Pre(const parser::IntentSpec &x) {
    switch (x.v) {
    case parser::IntentSpec::Intent::In:
      attrs_->Set(Attr::INTENT_IN);
      break;
    case parser::IntentSpec::Intent::Out:
      attrs_->Set(Attr::INTENT_OUT);
      break;
    case parser::IntentSpec::Intent::InOut:
      attrs_->Set(Attr::INTENT_IN);
      attrs_->Set(Attr::INTENT_OUT);
      break;
    default: CRASH_NO_CASE;
    }
    return false;
  }

  bool Pre(const parser::PrivateStmt &x) {
    //TODO: could be in TypeBoundProcedurePart
    builder_->Private();
    return false;
  }
  bool Pre(const parser::SequenceStmt &x) {
    builder_->sequence();
    return false;
  }

  bool Pre(const parser::ProcComponentDefStmt &x) {
    CHECK(attrs_ == nullptr);
    attrs_ = new Attrs();
    return true;
  }
  void Post(const parser::ProcComponentDefStmt &x) {
    if (declTypeSpec_) {
      std::cerr << "ProcComponentDefStmt: " << *declTypeSpec_ << "\n";
      delete declTypeSpec_;
      declTypeSpec_ = nullptr;
    }
    delete attrs_;
    attrs_ = nullptr;
  }
  void Post(const parser::ProcDecl &x) {
    CHECK(attrs_ != nullptr);
    const auto &name = std::get<parser::Name>(x.t);
    //TODO: std::get<std::optional<ProcPointerInit>>(x.t)
    builder_->procComponent(ProcComponentDef(ProcDecl(name), *attrs_));
  }

  bool Pre(const parser::DataComponentDefStmt &x) {
    CHECK(attrs_ == nullptr);
    attrs_ = new Attrs();
    return true;
  }
  void Post(const parser::DataComponentDefStmt &x) {
    delete declTypeSpec_;
    declTypeSpec_ = nullptr;
    delete attrs_;
    attrs_ = nullptr;
    delete attrArraySpec_;
    attrArraySpec_ = nullptr;
  }

  void Post(const parser::ComponentAttrSpec &x) {
    if (attrArraySpec_ == nullptr) {
      attrArraySpec_ = arraySpec_;
      arraySpec_ = nullptr;
    }
  }

  void Post(const parser::ComponentDecl &x) {
    CHECK(declTypeSpec_ != nullptr);
    CHECK(attrs_ != nullptr);
    const auto &name = std::get<parser::Name>(x.t);
    // use the array spec in the decl if present
    const auto &arraySpec = arraySpec_ && !arraySpec_->empty()
        ? *arraySpec_
        : attrArraySpec_ != nullptr ? *attrArraySpec_ : ComponentArraySpec{};
    builder_->dataComponent(
        DataComponentDef(*declTypeSpec_, name, *attrs_, arraySpec));
    delete arraySpec_;
    arraySpec_ = nullptr;
  }

  bool Pre(const parser::ComponentArraySpec &x) {
    CHECK(arraySpec_ == nullptr);
    arraySpec_ = new std::list<ShapeSpec>();
    return true;
  }
  bool Pre(const parser::DeferredShapeSpecList &x) {
    for (int i = 0; i < x.v; ++i) {
      arraySpec_->push_back(ShapeSpec::MakeDeferred());
    }
    return false;
  }
  bool Pre(const parser::ExplicitShapeSpec &x) {
    const auto &lb = std::get<std::optional<parser::SpecificationExpr>>(x.t);
    const auto &ub = GetBound(std::get<parser::SpecificationExpr>(x.t));
    if (lb) {
      arraySpec_->push_back(ShapeSpec::MakeExplicit(GetBound(*lb), ub));
    } else {
      arraySpec_->push_back(ShapeSpec::MakeExplicit(ub));
    }
    return false;
  }

  bool Pre(const parser::DeclarationTypeSpec::ClassStar &x) {
    declTypeSpec_ = new DeclTypeSpec{semantics::DeclTypeSpec::MakeClassStar()};
    return false;
  }
  bool Pre(const parser::DeclarationTypeSpec::TypeStar &x) {
    declTypeSpec_ = new DeclTypeSpec{semantics::DeclTypeSpec::MakeTypeStar()};
    return false;
  }
  bool Pre(const parser::DeclarationTypeSpec::Type &x) {
    // TODO - need DerivedTypeSpec => need type lookup
    return true;
  }
  bool Pre(const parser::DeclarationTypeSpec::Class &x) {
    // TODO - need DerivedTypeSpec => need type lookup
    return true;
  }
  bool Pre(const parser::DeclarationTypeSpec::Record &x) {
    // TODO
    return true;
  }
  bool Pre(const parser::IntegerTypeSpec &x) {
    declTypeSpec_ = new DeclTypeSpec{DeclTypeSpec::MakeIntrinsic(
      IntegerTypeSpec::Make(GetKindParamValue(x.v)))};
    return false;
  }

  bool Pre(const parser::IntrinsicTypeSpec::Logical &x) {
    declTypeSpec_ = new DeclTypeSpec{DeclTypeSpec::MakeIntrinsic(
        LogicalTypeSpec::Make(GetKindParamValue(x.kind)))};
    return false;
  }

  bool Pre(const parser::IntrinsicTypeSpec::Real &x) {
    declTypeSpec_ = new DeclTypeSpec{DeclTypeSpec::MakeIntrinsic(
        RealTypeSpec::Make(GetKindParamValue(x.kind)))};
    return false;
  }

  bool Pre(const parser::IntrinsicTypeSpec::Complex &x) {
    declTypeSpec_ = new DeclTypeSpec{DeclTypeSpec::MakeIntrinsic(
        ComplexTypeSpec::Make(GetKindParamValue(x.kind)))};
    return false;
  }

  bool Pre(const parser::IntrinsicTypeSpec::DoublePrecision &x) {
    return false; // TODO
  }
  bool Pre(const parser::IntrinsicTypeSpec::Character &x) {
    return false; // TODO
  }
  bool Pre(const parser::IntrinsicTypeSpec::DoubleComplex &x) {
    return false; // TODO
  }
  bool Pre(const parser::IntrinsicTypeSpec::NCharacter &x) {
    return false; // TODO
  }

  bool Pre(const parser::DerivedTypeStmt &x) {
    CHECK(attrs_ == nullptr);
    attrs_ = new Attrs{};
    return true;
  }
  void Post(const parser::DerivedTypeStmt &x) {
    builder_->name(std::get<Name>(x.t));
    builder_->attrs(*attrs_);
    delete attrs_;
    attrs_ = nullptr;
  }

  void Post(const parser::Program &) {
    // ensure that all temps were deallocated
    CHECK(builder_ == nullptr);
    CHECK(declTypeSpec_ == nullptr);
    CHECK(attrs_ == nullptr);
    CHECK(arraySpec_ == nullptr);
    CHECK(attrArraySpec_ == nullptr);
  }

private:
  std::ostream &out_;
  DerivedTypeDefBuilder *builder_ = nullptr;
  DeclTypeSpec *declTypeSpec_ = nullptr;
  Attrs *attrs_ = nullptr;
  std::list<ShapeSpec> *arraySpec_ = nullptr;
  // attrArraySpec_ is used to save the component-array-spec that is part of
  // the component-attr-spec
  std::list<ShapeSpec> *attrArraySpec_ = nullptr;

};

void MakeTypes(std::ostream &out, const parser::Program &program) {
  MakeTypesVisitor visitor{out};
  parser::Walk(program, visitor);
}

static KindParamValue GetKindParamValue(
    const std::optional<parser::KindSelector> &kind) {
  if (!kind) {
    return KindParamValue();
  } else {
    const auto &lit =
        std::get<parser::LiteralConstant>(kind->v.thing.thing.thing->u);
    const auto &intlit = std::get<parser::IntLiteralConstant>(lit.u);
    return KindParamValue(std::get<std::uint64_t>(intlit.t));
  }
}

static const IntExpr *GetIntExpr(const parser::ScalarIntExpr &x) {
  const parser::Expr &expr = *x.thing.thing;
  if (std::holds_alternative<parser::LiteralConstant>(expr.u)) {
    const auto &lit = std::get<parser::LiteralConstant>(expr.u);
    if (std::holds_alternative<parser::IntLiteralConstant>(lit.u)) {
      const auto &intLit = std::get<parser::IntLiteralConstant>(lit.u);
      return &IntConst::Make(std::get<std::uint64_t>(intLit.t));
    }
  }
  std::cerr << "IntExpr:\n" << expr << "\n";
  return new semantics::IntExpr();  // TODO
}

static Bound GetBound(const parser::SpecificationExpr &x) {
  return Bound(*GetIntExpr(x.v));
}

}  // namespace semantics
}  // namespace Fortran
