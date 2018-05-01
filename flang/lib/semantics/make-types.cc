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

#include "make-types.h"
#include "attr.h"
#include "type.h"
#include "../parser/idioms.h"
#include "../parser/parse-tree-visitor.h"
#include "../parser/parse-tree.h"
#include <iostream>
#include <memory>
#include <set>

namespace Fortran::semantics {

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
    out_ << *declTypeSpec_ << "\n\n";
    declTypeSpec_.reset();
  }

  bool Pre(const parser::DerivedTypeDef &x) {
    CHECK(!builder_);
    builder_ = std::make_unique<DerivedTypeDefBuilder>();
    return true;
  }
  void Post(const parser::DerivedTypeDef &x) {
    DerivedTypeDef derivedType{*builder_};
    out_ << derivedType << "\n\n";
    builder_.reset();
  }

  bool Pre(const parser::TypeAttrSpec::Extends &x) {
    builder_->extends(x.v.ToString());
    return false;
  }
  bool Pre(const parser::AccessSpec &x) {
    switch (x.v) {
    case parser::AccessSpec::Kind::Public: attrs_->set(Attr::PUBLIC); break;
    case parser::AccessSpec::Kind::Private: attrs_->set(Attr::PRIVATE); break;
    default: CRASH_NO_CASE;
    }
    return false;
  }
  bool Pre(const parser::TypeAttrSpec::BindC &x) {
    attrs_->set(Attr::BIND_C);
    return false;
  }
  bool Pre(const parser::Abstract &x) {
    attrs_->set(Attr::ABSTRACT);
    return false;
  }
  bool Pre(const parser::Allocatable &) {
    attrs_->set(Attr::ALLOCATABLE);
    return false;
  }
  bool Pre(const parser::Asynchronous &) {
    attrs_->set(Attr::ASYNCHRONOUS);
    return false;
  }
  bool Pre(const parser::Contiguous &) {
    attrs_->set(Attr::CONTIGUOUS);
    return false;
  }
  bool Pre(const parser::External &) {
    attrs_->set(Attr::EXTERNAL);
    return false;
  }
  bool Pre(const parser::Intrinsic &) {
    attrs_->set(Attr::INTRINSIC);
    return false;
  }
  bool Pre(const parser::NoPass &) {
    attrs_->set(Attr::NOPASS);
    return false;
  }
  bool Pre(const parser::Optional &) {
    attrs_->set(Attr::OPTIONAL);
    return false;
  }
  bool Pre(const parser::Parameter &) {
    attrs_->set(Attr::PARAMETER);
    return false;
  }
  bool Pre(const parser::Pass &) {
    attrs_->set(Attr::PASS);
    return false;
  }
  bool Pre(const parser::Pointer &) {
    attrs_->set(Attr::POINTER);
    return false;
  }
  bool Pre(const parser::Protected &) {
    attrs_->set(Attr::PROTECTED);
    return false;
  }
  bool Pre(const parser::Save &) {
    attrs_->set(Attr::SAVE);
    return false;
  }
  bool Pre(const parser::Target &) {
    attrs_->set(Attr::TARGET);
    return false;
  }
  bool Pre(const parser::Value &) {
    attrs_->set(Attr::VALUE);
    return false;
  }
  bool Pre(const parser::Volatile &) {
    attrs_->set(Attr::VOLATILE);
    return false;
  }
  bool Pre(const parser::IntentSpec &x) {
    switch (x.v) {
    case parser::IntentSpec::Intent::In:
      attrs_->set(Attr::INTENT_IN);
      break;
    case parser::IntentSpec::Intent::Out:
      attrs_->set(Attr::INTENT_OUT);
      break;
    case parser::IntentSpec::Intent::InOut:
      attrs_->set(Attr::INTENT_IN);
      attrs_->set(Attr::INTENT_OUT);
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
    CHECK(!attrs_);
    attrs_ = std::make_unique<Attrs>();
    return true;
  }
  void Post(const parser::ProcComponentDefStmt &x) {
    if (declTypeSpec_) {
      std::cerr << "ProcComponentDefStmt: " << *declTypeSpec_ << "\n";
      declTypeSpec_.reset();
    }
    attrs_.reset();
  }
  void Post(const parser::ProcDecl &x) {
    const auto &name = std::get<parser::Name>(x.t);
    //TODO: std::get<std::optional<ProcPointerInit>>(x.t)
    builder_->procComponent(ProcComponentDef(ProcDecl(name.ToString()), *attrs_));
  }

  bool Pre(const parser::DataComponentDefStmt &x) {
    CHECK(!attrs_);
    attrs_ = std::make_unique<Attrs>();
    return true;
  }
  void Post(const parser::DataComponentDefStmt &x) {
    declTypeSpec_.reset();
    attrs_.reset();
    attrArraySpec_.reset();
  }

  void Post(const parser::ComponentAttrSpec &x) {
    if (!attrArraySpec_) {
      attrArraySpec_ = std::move(arraySpec_);
    }
  }

  void Post(const parser::ComponentDecl &x) {
    const auto &name = std::get<parser::Name>(x.t);
    // use the array spec in the decl if present
    const auto &arraySpec = arraySpec_ && !arraySpec_->empty()
        ? *arraySpec_
        : attrArraySpec_ ? *attrArraySpec_ : ArraySpec{};
    builder_->dataComponent(
        DataComponentDef(*declTypeSpec_, name.ToString(), *attrs_, arraySpec));
    arraySpec_.reset();
  }

  bool Pre(const parser::ComponentArraySpec &x) {
    CHECK(!arraySpec_);
    arraySpec_ = std::make_unique<std::list<ShapeSpec>>();
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
    CHECK(!declTypeSpec_);
    declTypeSpec_ =
        std::make_unique<DeclTypeSpec>(DeclTypeSpec::MakeClassStar());
    return false;
  }
  bool Pre(const parser::DeclarationTypeSpec::TypeStar &x) {
    CHECK(!declTypeSpec_);
    declTypeSpec_ =
        std::make_unique<DeclTypeSpec>(DeclTypeSpec::MakeTypeStar());
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
    CHECK(!declTypeSpec_);
    declTypeSpec_ = std::make_unique<DeclTypeSpec>(DeclTypeSpec::MakeIntrinsic(
        IntegerTypeSpec::Make(GetKindParamValue(x.v))));
    return false;
  }

  bool Pre(const parser::IntrinsicTypeSpec::Logical &x) {
    CHECK(!declTypeSpec_);
    declTypeSpec_ = std::make_unique<DeclTypeSpec>(DeclTypeSpec::MakeIntrinsic(
        LogicalTypeSpec::Make(GetKindParamValue(x.kind))));
    return false;
  }

  bool Pre(const parser::IntrinsicTypeSpec::Real &x) {
    CHECK(!declTypeSpec_);
    declTypeSpec_ = std::make_unique<DeclTypeSpec>(DeclTypeSpec::MakeIntrinsic(
        RealTypeSpec::Make(GetKindParamValue(x.kind))));
    return false;
  }

  bool Pre(const parser::IntrinsicTypeSpec::Complex &x) {
    CHECK(!declTypeSpec_);
    declTypeSpec_ = std::make_unique<DeclTypeSpec>(DeclTypeSpec::MakeIntrinsic(
        ComplexTypeSpec::Make(GetKindParamValue(x.kind))));
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
    CHECK(!attrs_);
    attrs_ = std::make_unique<Attrs>();
    return true;
  }
  void Post(const parser::DerivedTypeStmt &x) {
    builder_->name(std::get<parser::Name>(x.t).ToString());
    builder_->attrs(*attrs_);
    attrs_.reset();
  }

  void Post(const parser::Program &) {
    // ensure that all temps were deallocated
    CHECK(!builder_);
    CHECK(!declTypeSpec_);
    CHECK(!attrs_);
    CHECK(!arraySpec_);
    CHECK(!attrArraySpec_);
  }

private:
  std::ostream &out_;
  std::unique_ptr<DerivedTypeDefBuilder> builder_;
  std::unique_ptr<DeclTypeSpec> declTypeSpec_;
  std::unique_ptr<Attrs> attrs_;
  std::unique_ptr<std::list<ShapeSpec>> arraySpec_;
  // attrArraySpec_ is used to save the component-array-spec that is part of
  // the component-attr-spec
  std::unique_ptr<std::list<ShapeSpec>> attrArraySpec_;

};

void MakeTypes(
    const parser::Program &program, const parser::CookedSource &cookedSource) {
  void ResolveNames(const parser::Program &, const parser::CookedSource &);
  ResolveNames(program, cookedSource);
}

static KindParamValue GetKindParamValue(
    const std::optional<parser::KindSelector> &kind) {
  if (!kind) {
    return KindParamValue();
  } else if (std::holds_alternative<parser::ScalarIntConstantExpr>(kind->u)) {
    const auto &expr = std::get<parser::ScalarIntConstantExpr>(kind->u);
    const auto &lit =
        std::get<parser::LiteralConstant>(expr.thing.thing.thing->u);
    const auto &intlit = std::get<parser::IntLiteralConstant>(lit.u);
    return KindParamValue(std::get<std::uint64_t>(intlit.t));
  } else {
    // TODO: COMPLEX*16 means COMPLEX(KIND=8) (yes?); translate
    return KindParamValue(std::get<parser::KindSelector::StarSize>(kind->u).v);
  }
}

static const IntExpr *GetIntExpr(const parser::ScalarIntExpr &x) {
  return new IntExpr(x);
}

static Bound GetBound(const parser::SpecificationExpr &x) {
  return Bound(*GetIntExpr(x.v));
}

}
