// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

// Static declaration checking

#include "check-declarations.h"
#include "scope.h"
#include "semantics.h"
#include "symbol.h"
#include "tools.h"
#include "type.h"
#include "../evaluate/check-expression.h"

namespace Fortran::semantics {

class CheckHelper {
public:
  explicit CheckHelper(SemanticsContext &c) : context_{c} {}

  void Check() const { Check(context_.globalScope()); }
  void Check(const ParamValue &value) const {
    CheckSpecExpr(value.GetExplicit());
  }
  void Check(const Bound &bound) const { CheckSpecExpr(bound.GetExplicit()); }
  void Check(const ShapeSpec &spec) const {
    Check(spec.lbound());
    Check(spec.ubound());
  }
  void Check(const ArraySpec &shape) const {
    for (const auto &spec : shape) {
      Check(spec);
    }
  }
  void Check(const DeclTypeSpec &type) const {
    if (type.category() == DeclTypeSpec::Character) {
      Check(type.characterTypeSpec().length());
    } else if (const DerivedTypeSpec * spec{type.AsDerived()}) {
      for (const auto &parm : spec->parameters()) {
        Check(parm.second);
      }
    }
  }
  void Check(const Symbol &symbol) const {
    if (context_.HasError(symbol) || symbol.has<UseDetails>() ||
        symbol.has<HostAssocDetails>()) {
      return;
    }
    auto save{messages_.SetLocation(symbol.name())};
    context_.set_location(symbol.name());
    if (const DeclTypeSpec * type{symbol.GetType()}) {
      Check(*type);
    }
    if (IsAssumedLengthCharacterFunction(symbol)) {  // C723
      if (symbol.attrs().test(Attr::RECURSIVE)) {
        context_.Say(
            "An assumed-length CHARACTER(*) function cannot be RECURSIVE"_err_en_US);
      }
      if (symbol.Rank() > 0) {
        context_.Say(
            "An assumed-length CHARACTER(*) function cannot return an array"_err_en_US);
      }
      if (symbol.attrs().test(Attr::PURE)) {
        context_.Say(
            "An assumed-length CHARACTER(*) function cannot be PURE"_err_en_US);
      }
      if (symbol.attrs().test(Attr::ELEMENTAL)) {
        context_.Say(
            "An assumed-length CHARACTER(*) function cannot be ELEMENTAL"_err_en_US);
      }
      if (const Symbol * result{FindFunctionResult(symbol)}) {
        if (result->attrs().test(Attr::POINTER)) {
          context_.Say(
              "An assumed-length CHARACTER(*) function cannot return a POINTER"_err_en_US);
        }
      }
    }
    if (const auto *object{symbol.detailsIf<ObjectEntityDetails>()}) {
      Check(object->shape());
      Check(object->coshape());
    }
  }
  void Check(const Scope &scope) const {
    for (const auto &pair : scope) {
      Check(*pair.second);
    }
    for (const Scope &child : scope.children()) {
      Check(child);
    }
  }

private:
  template<typename A> void CheckSpecExpr(const A &x) const {
    evaluate::CheckSpecificationExpr(x, messages_);
  }

  SemanticsContext &context_;
  parser::ContextualMessages &messages_{context_.foldingContext().messages()};
};

void CheckDeclarations(SemanticsContext &context) {
  CheckHelper{context}.Check();
}
}
