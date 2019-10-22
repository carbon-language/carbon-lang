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
#include "../evaluate/fold.h"

namespace Fortran::semantics {

class CheckHelper {
public:
  explicit CheckHelper(SemanticsContext &c) : context_{c} {}

  void Check() { Check(context_.globalScope()); }
  void Check(const ParamValue &, bool canBeAssumed);
  void Check(Bound &bound) { CheckSpecExpr(bound.GetExplicit()); }
  void Check(ShapeSpec &spec) {
    Check(spec.lbound());
    Check(spec.ubound());
  }
  void Check(ArraySpec &);
  void Check(DeclTypeSpec &, bool canHaveAssumedTypeParameters);
  void Check(Symbol &);
  void Check(Scope &);

private:
  template<typename A> void CheckSpecExpr(A &x) {
    x = Fold(foldingContext_, std::move(x));
    evaluate::CheckSpecificationExpr(x, messages_, DEREF(scope_));
  }
  template<typename A> void CheckSpecExpr(const A &x) {
    evaluate::CheckSpecificationExpr(x, messages_, DEREF(scope_));
  }

  SemanticsContext &context_;
  evaluate::FoldingContext &foldingContext_{context_.foldingContext()};
  parser::ContextualMessages &messages_{foldingContext_.messages()};
  const Scope *scope_{nullptr};
};

void CheckHelper::Check(const ParamValue &value, bool canBeAssumed) {
  if (value.isAssumed()) {
    if (!canBeAssumed) {  // C795
      messages_.Say(
          "An assumed (*) type parameter may be used only for a dummy argument, associate name, or named constant"_err_en_US);
    }
  } else {
    CheckSpecExpr(value.GetExplicit());
  }
}

void CheckHelper::Check(ArraySpec &shape) {
  for (auto &spec : shape) {
    Check(spec);
  }
}

void CheckHelper::Check(DeclTypeSpec &type, bool canHaveAssumedTypeParameters) {
  if (type.category() == DeclTypeSpec::Character) {
    Check(type.characterTypeSpec().length(), canHaveAssumedTypeParameters);
  } else if (const DerivedTypeSpec * spec{type.AsDerived()}) {
    for (auto &parm : spec->parameters()) {
      Check(parm.second, canHaveAssumedTypeParameters);
    }
  }
}

void CheckHelper::Check(Symbol &symbol) {
  if (context_.HasError(symbol) || symbol.has<UseDetails>() ||
      symbol.has<HostAssocDetails>()) {
    return;
  }
  auto save{messages_.SetLocation(symbol.name())};
  context_.set_location(symbol.name());
  if (DeclTypeSpec * type{symbol.GetType()}) {
    bool canHaveAssumedParameter{IsNamedConstant(symbol) ||
        IsAssumedLengthCharacterFunction(symbol) ||
        symbol.test(Symbol::Flag::ParentComp)};
    if (const auto *object{symbol.detailsIf<ObjectEntityDetails>()}) {
      canHaveAssumedParameter |= object->isDummy() ||
          (object->isFuncResult() &&
              type->category() == DeclTypeSpec::Character);
    } else {
      canHaveAssumedParameter |= symbol.has<AssocEntityDetails>();
    }
    Check(*type, canHaveAssumedParameter);
  }
  if (IsAssumedLengthCharacterFunction(symbol)) {  // C723
    if (symbol.attrs().test(Attr::RECURSIVE)) {
      messages_.Say(
          "An assumed-length CHARACTER(*) function cannot be RECURSIVE"_err_en_US);
    }
    if (symbol.Rank() > 0) {
      messages_.Say(
          "An assumed-length CHARACTER(*) function cannot return an array"_err_en_US);
    }
    if (symbol.attrs().test(Attr::PURE)) {
      messages_.Say(
          "An assumed-length CHARACTER(*) function cannot be PURE"_err_en_US);
    }
    if (symbol.attrs().test(Attr::ELEMENTAL)) {
      messages_.Say(
          "An assumed-length CHARACTER(*) function cannot be ELEMENTAL"_err_en_US);
    }
    if (const Symbol * result{FindFunctionResult(symbol)}) {
      if (result->attrs().test(Attr::POINTER)) {
        messages_.Say(
            "An assumed-length CHARACTER(*) function cannot return a POINTER"_err_en_US);
      }
    }
  }
  if (auto *object{symbol.detailsIf<ObjectEntityDetails>()}) {
    Check(object->shape());
    Check(object->coshape());
    if (object->isDummy() && symbol.attrs().test(Attr::INTENT_OUT)) {
      if (FindUltimateComponent(symbol, [](const Symbol &symbol) {
            return IsCoarray(symbol) && IsAllocatable(symbol);
          })) {  // C846
        messages_.Say(
            "An INTENT(OUT) dummy argument may not be, or contain, an ALLOCATABLE coarray"_err_en_US);
      }
      if (IsOrContainsEventOrLockComponent(symbol)) {  // C847
        messages_.Say(
            "An INTENT(OUT) dummy argument may not be, or contain, EVENT_TYPE or LOCK_TYPE"_err_en_US);
      }
    }
  }
}

void CheckHelper::Check(Scope &scope) {
  scope_ = &scope;
  for (auto &pair : scope) {
    Check(*pair.second);
  }
  for (Scope &child : scope.children()) {
    Check(child);
  }
}

void CheckDeclarations(SemanticsContext &context) {
  CheckHelper{context}.Check();
}
}
