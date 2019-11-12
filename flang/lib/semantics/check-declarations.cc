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
#include "../evaluate/tools.h"

namespace Fortran::semantics {

class CheckHelper {
public:
  explicit CheckHelper(SemanticsContext &c) : context_{c} {}

  void Check() { Check(context_.globalScope()); }
  void Check(const ParamValue &, bool canBeAssumed);
  void Check(const Bound &bound) { CheckSpecExpr(bound.GetExplicit()); }
  void Check(const ShapeSpec &spec) {
    Check(spec.lbound());
    Check(spec.ubound());
  }
  void Check(const ArraySpec &);
  void Check(const DeclTypeSpec &, bool canHaveAssumedTypeParameters);
  void Check(const Symbol &);
  void Check(const Scope &);

private:
  template<typename A> void CheckSpecExpr(A &x) {
    x = Fold(foldingContext_, std::move(x));
    evaluate::CheckSpecificationExpr(x, messages_, DEREF(scope_));
  }
  template<typename A> void CheckSpecExpr(const A &x) {
    evaluate::CheckSpecificationExpr(x, messages_, DEREF(scope_));
  }
  void CheckValue(const Symbol &, const DerivedTypeSpec *);
  void CheckVolatile(
      const Symbol &, bool isAssociated, const DerivedTypeSpec *);

  SemanticsContext &context_;
  evaluate::FoldingContext &foldingContext_{context_.foldingContext()};
  parser::ContextualMessages &messages_{foldingContext_.messages()};
  const Scope *scope_{nullptr};
  // This symbol is the one attached to the innermost enclosing scope
  // that has a symbol.
  const Symbol *innermostSymbol_{nullptr};
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

void CheckHelper::Check(const ArraySpec &shape) {
  for (const auto &spec : shape) {
    Check(spec);
  }
}

void CheckHelper::Check(
    const DeclTypeSpec &type, bool canHaveAssumedTypeParameters) {
  if (type.category() == DeclTypeSpec::Character) {
    Check(type.characterTypeSpec().length(), canHaveAssumedTypeParameters);
  } else if (const DerivedTypeSpec * spec{type.AsDerived()}) {
    for (auto &parm : spec->parameters()) {
      Check(parm.second, canHaveAssumedTypeParameters);
    }
  }
}

void CheckHelper::Check(const Symbol &symbol) {
  if (context_.HasError(symbol)) {
    return;
  }
  const DeclTypeSpec *type{symbol.GetUltimate().GetType()};
  const DerivedTypeSpec *derived{type ? type->AsDerived() : nullptr};
  auto save{messages_.SetLocation(symbol.name())};
  context_.set_location(symbol.name());
  bool isAssociated{symbol.has<UseDetails>() || symbol.has<HostAssocDetails>()};
  if (symbol.attrs().test(Attr::VOLATILE)) {
    CheckVolatile(symbol, isAssociated, derived);
  }
  if (isAssociated) {
    return;  // only care about checking VOLATILE on associated symbols
  }
  bool inPure{innermostSymbol_ && IsPureProcedure(*innermostSymbol_)};
  if (inPure) {
    if (IsSaved(symbol)) {
      messages_.Say(
          "A PURE subprogram may not have a variable with the SAVE attribute"_err_en_US);
    }
    if (symbol.attrs().test(Attr::VOLATILE)) {
      messages_.Say(
          "A PURE subprogram may not have a variable with the VOLATILE attribute"_err_en_US);
    }
    if (IsProcedure(symbol) && !IsPureProcedure(symbol) && IsDummy(symbol)) {
      messages_.Say(
          "A dummy procedure of a PURE subprogram must be PURE"_err_en_US);
    }
    if (!IsDummy(symbol) && !IsFunctionResult(symbol)) {
      if (IsPolymorphicAllocatable(symbol)) {
        evaluate::SayWithDeclaration(messages_, &symbol,
            "Deallocation of polymorphic object '%s' is not permitted in a PURE subprogram"_err_en_US,
            symbol.name());
      } else if (derived) {
        if (auto bad{FindPolymorphicAllocatableUltimateComponent(*derived)}) {
          evaluate::SayWithDeclaration(messages_, &*bad,
              "Deallocation of polymorphic object '%s%s' is not permitted in a PURE subprogram"_err_en_US,
              symbol.name(), bad.BuildResultDesignatorName());
        }
      }
    }
  }
  bool inFunction{innermostSymbol_ && IsFunction(*innermostSymbol_)};
  if (type) {
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
    if (inPure && inFunction && IsFunctionResult(symbol)) {
      if (derived && HasImpureFinal(*derived)) {  // C1584
        messages_.Say(
            "Result of PURE function may not have an impure FINAL subroutine"_err_en_US);
      }
      if (type->IsPolymorphic() && IsAllocatable(symbol)) {  // C1585
        messages_.Say(
            "Result of PURE function may not be both polymorphic and ALLOCATABLE"_err_en_US);
      }
      if (derived) {
        if (auto bad{FindPolymorphicAllocatableUltimateComponent(*derived)}) {
          evaluate::SayWithDeclaration(messages_, &*bad,
              "Result of PURE function may not have polymorphic ALLOCATABLE ultimate component '%s'"_err_en_US,
              bad.BuildResultDesignatorName());
        }
      }
    }
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
      if (IsPointer(*result)) {
        messages_.Say(
            "An assumed-length CHARACTER(*) function cannot return a POINTER"_err_en_US);
      }
    }
  }
  if (auto *object{symbol.detailsIf<ObjectEntityDetails>()}) {
    Check(object->shape());
    Check(object->coshape());
    if (!object->coshape().empty()) {
      if (IsAllocatable(symbol)) {
        if (!object->coshape().IsDeferredShape()) {  // C827
          messages_.Say(
              "ALLOCATABLE coarray must have a deferred coshape"_err_en_US);
        }
      } else {
        if (!object->coshape().IsAssumedSize()) {  // C828
          messages_.Say(
              "Non-ALLOCATABLE coarray must have an explicit coshape"_err_en_US);
        }
      }
    }
    if (object->isDummy()) {
      if (symbol.attrs().test(Attr::INTENT_OUT)) {
        if (FindUltimateComponent(symbol, [](const Symbol &x) {
              return IsCoarray(x) && IsAllocatable(x);
            })) {  // C846
          messages_.Say(
              "An INTENT(OUT) dummy argument may not be, or contain, an ALLOCATABLE coarray"_err_en_US);
        }
        if (IsOrContainsEventOrLockComponent(symbol)) {  // C847
          messages_.Say(
              "An INTENT(OUT) dummy argument may not be, or contain, EVENT_TYPE or LOCK_TYPE"_err_en_US);
        }
      }
      if (inPure && !IsPointer(symbol) && !IsIntentIn(symbol) &&
          !symbol.attrs().test(Attr::VALUE)) {
        if (inFunction) {  // C1583
          messages_.Say(
              "non-POINTER dummy argument of PURE function must be INTENT(IN) or VALUE"_err_en_US);
        } else if (IsIntentOut(symbol)) {
          if (type && type->IsPolymorphic()) {  // C1588
            messages_.Say(
                "An INTENT(OUT) dummy argument of a PURE subroutine may not be polymorphic"_err_en_US);
          } else if (derived) {
            if (FindUltimateComponent(*derived, [](const Symbol &x) {
                  const DeclTypeSpec *type{x.GetType()};
                  return type && type->IsPolymorphic();
                })) {  // C1588
              messages_.Say(
                  "An INTENT(OUT) dummy argument of a PURE subroutine may not have a polymorphic ultimate component"_err_en_US);
            }
            if (HasImpureFinal(*derived)) {  // C1587
              messages_.Say(
                  "An INTENT(OUT) dummy argument of a PURE subroutine may not have an impure FINAL subroutine"_err_en_US);
            }
          }
        } else if (!IsIntentInOut(symbol)) {  // C1586
          messages_.Say(
              "non-POINTER dummy argument of PURE subroutine must have INTENT() or VALUE attribute"_err_en_US);
        }
      }
    }
  } else if (auto *proc{symbol.detailsIf<ProcEntityDetails>()}) {
    if (proc->isDummy()) {
      const Symbol *interface{proc->interface().symbol()};
      if (!symbol.attrs().test(Attr::INTRINSIC) &&
          (symbol.attrs().test(Attr::ELEMENTAL) ||
              (interface && !interface->attrs().test(Attr::INTRINSIC) &&
                  interface->attrs().test(Attr::ELEMENTAL)))) {
        // There's no explicit constraint or "shall" that we can find in the
        // standard for this check, but it seems to be implied in multiple
        // sites, and ELEMENTAL non-intrinsic actual arguments *are*
        // explicitly forbidden.  But we allow "PROCEDURE(SIN)::dummy"
        // because it is explicitly legal to *pass* the specific intrinsic
        // function SIN as an actual argument.
        messages_.Say("A dummy procedure may not be ELEMENTAL"_err_en_US);
      }
    }
  }
  if (symbol.attrs().test(Attr::VALUE)) {
    CheckValue(symbol, derived);
  }
  if (symbol.attrs().test(Attr::CONTIGUOUS) && IsPointer(symbol) &&
      symbol.Rank() == 0) {  // C830
    messages_.Say("CONTIGUOUS POINTER must be an array"_err_en_US);
  }
}

void CheckHelper::CheckValue(
    const Symbol &symbol, const DerivedTypeSpec *derived) {  // C863 - C865
  if (!IsDummy(symbol)) {
    messages_.Say(
        "VALUE attribute may apply only to a dummy argument"_err_en_US);
  }
  if (IsProcedure(symbol)) {
    messages_.Say(
        "VALUE attribute may apply only to a dummy data object"_err_en_US);
  }
  if (IsAssumedSizeArray(symbol)) {
    messages_.Say(
        "VALUE attribute may not apply to an assumed-size array"_err_en_US);
  }
  if (IsCoarray(symbol)) {
    messages_.Say("VALUE attribute may not apply to a coarray"_err_en_US);
  }
  if (IsAllocatable(symbol)) {
    messages_.Say("VALUE attribute may not apply to an ALLOCATABLE"_err_en_US);
  } else if (IsPointer(symbol)) {
    messages_.Say("VALUE attribute may not apply to a POINTER"_err_en_US);
  }
  if (IsIntentInOut(symbol)) {
    messages_.Say(
        "VALUE attribute may not apply to an INTENT(IN OUT) argument"_err_en_US);
  } else if (IsIntentOut(symbol)) {
    messages_.Say(
        "VALUE attribute may not apply to an INTENT(OUT) argument"_err_en_US);
  }
  if (symbol.attrs().test(Attr::VOLATILE)) {
    messages_.Say("VALUE attribute may not apply to a VOLATILE"_err_en_US);
  }
  if (innermostSymbol_ && IsBindCProcedure(*innermostSymbol_) &&
      IsOptional(symbol)) {
    messages_.Say(
        "VALUE attribute may not apply to an OPTIONAL in a BIND(C) procedure"_err_en_US);
  }
  if (derived) {
    if (FindCoarrayUltimateComponent(*derived)) {
      messages_.Say(
          "VALUE attribute may not apply to a type with a coarray ultimate component"_err_en_US);
    }
  }
}

void CheckHelper::CheckVolatile(const Symbol &symbol, bool isAssociated,
    const DerivedTypeSpec *derived) {  // C866 - C868
  if (IsIntentIn(symbol)) {
    messages_.Say(
        "VOLATILE attribute may not apply to an INTENT(IN) argument"_err_en_US);
  }
  if (IsProcedure(symbol)) {
    messages_.Say("VOLATILE attribute may apply only to a variable"_err_en_US);
  }
  if (isAssociated) {
    const Symbol &ultimate{symbol.GetUltimate()};
    if (IsCoarray(ultimate)) {
      messages_.Say(
          "VOLATILE attribute may not apply to a coarray accessed by USE or host association"_err_en_US);
    }
    if (derived) {
      if (FindCoarrayUltimateComponent(*derived)) {
        messages_.Say(
            "VOLATILE attribute may not apply to a type with a coarray ultimate component accessed by USE or host association"_err_en_US);
      }
    }
  }
}

void CheckHelper::Check(const Scope &scope) {
  scope_ = &scope;
  if (const Symbol * scopeSymbol{scope.symbol()}) {
    innermostSymbol_ = scopeSymbol;
  }
  for (const auto &pair : scope) {
    Check(*pair.second);
  }
  for (const Scope &child : scope.children()) {
    Check(child);
  }
}

void CheckDeclarations(SemanticsContext &context) {
  CheckHelper{context}.Check();
}
}
