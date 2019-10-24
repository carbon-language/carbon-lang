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

#include "check-call.h"
#include "scope.h"
#include "tools.h"
#include "../evaluate/characteristics.h"
#include "../evaluate/shape.h"
#include "../evaluate/tools.h"
#include "../parser/characters.h"
#include "../parser/message.h"
#include <map>
#include <string>

using namespace Fortran::parser::literals;
namespace characteristics = Fortran::evaluate::characteristics;

namespace Fortran::semantics {

static void CheckImplicitInterfaceArg(
    evaluate::ActualArgument &arg, parser::ContextualMessages &messages) {
  if (const auto &kw{arg.keyword}) {
    messages.Say(*kw,
        "Keyword '%s=' may not appear in a reference to a procedure with an implicit interface"_err_en_US,
        *kw);
  }
  if (auto type{arg.GetType()}) {
    if (type->IsAssumedType()) {
      messages.Say(
          "Assumed type argument requires an explicit interface"_err_en_US);
    } else if (type->IsPolymorphic()) {
      messages.Say(
          "Polymorphic argument requires an explicit interface"_err_en_US);
    } else if (type->category() == TypeCategory::Derived) {
      auto &derived{type->GetDerivedTypeSpec()};
      if (!derived.parameters().empty()) {
        messages.Say(
            "Parameterized derived type argument requires an explicit interface"_err_en_US);
      }
    }
  }
  if (const auto *expr{arg.UnwrapExpr()}) {
    if (auto named{evaluate::ExtractNamedEntity(*expr)}) {
      const Symbol &symbol{named->GetLastSymbol()};
      if (symbol.Corank() > 0) {
        messages.Say(
            "Coarray argument requires an explicit interface"_err_en_US);
      }
      if (const auto *details{symbol.detailsIf<ObjectEntityDetails>()}) {
        if (details->IsAssumedRank()) {
          messages.Say(
              "Assumed rank argument requires an explicit interface"_err_en_US);
        }
      }
      if (symbol.attrs().test(Attr::ASYNCHRONOUS)) {
        messages.Say(
            "ASYNCHRONOUS argument requires an explicit interface"_err_en_US);
      }
      if (symbol.attrs().test(Attr::VOLATILE)) {
        messages.Say(
            "VOLATILE argument requires an explicit interface"_err_en_US);
      }
    }
  }
}

// When scalar CHARACTER actual arguments are known to be short,
// we extend them on the right with spaces and a warning.
static void PadShortCharacterActual(evaluate::Expr<evaluate::SomeType> &actual,
    const characteristics::TypeAndShape &dummyType,
    const characteristics::TypeAndShape &actualType,
    parser::ContextualMessages &messages) {
  if (dummyType.type().category() == TypeCategory::Character &&
      actualType.type().category() == TypeCategory::Character &&
      dummyType.type().kind() == actualType.type().kind() &&
      GetRank(actualType.shape()) == 0) {
    if (auto dummyLEN{ToInt64(dummyType.LEN())}) {
      if (auto actualLEN{ToInt64(actualType.LEN())}) {
        if (*actualLEN < *dummyLEN) {
          messages.Say(
              "Actual length '%jd' is less than expected length '%jd'"_en_US,
              *actualLEN, *dummyLEN);
          auto converted{ConvertToType(dummyType.type(), std::move(actual))};
          CHECK(converted.has_value());
          actual = std::move(*converted);
        }
      }
    }
  }
}

static bool DefersSameTypeParameters(
    const DerivedTypeSpec &actual, const DerivedTypeSpec &dummy) {
  for (const auto &pair : actual.parameters()) {
    const ParamValue &actualValue{pair.second};
    const ParamValue *dummyValue{dummy.FindParameter(pair.first)};
    if (!dummyValue || (actualValue.isDeferred() != dummyValue->isDeferred())) {
      return false;
    }
  }
  return true;
}

static void CheckExplicitDataArg(const characteristics::DummyDataObject &dummy,
    const std::string &dummyName, evaluate::Expr<evaluate::SomeType> &actual,
    const characteristics::TypeAndShape &actualType, bool isElemental,
    evaluate::FoldingContext &context, const Scope *scope) {

  // Basic type & rank checking
  parser::ContextualMessages &messages{context.messages()};
  PadShortCharacterActual(actual, dummy.type, actualType, messages);
  bool typesCompatible{dummy.type.IsCompatibleWith(
      messages, actualType, "dummy argument", "actual argument", isElemental)};
  bool actualIsPolymorphic{actualType.type().IsPolymorphic()};
  bool dummyIsPolymorphic{dummy.type.type().IsPolymorphic()};
  bool actualIsCoindexed{ExtractCoarrayRef(actual).has_value()};
  bool actualIsAssumedSize{actualType.attrs().test(
      characteristics::TypeAndShape::Attr::AssumedSize)};
  bool dummyIsAssumedSize{dummy.type.attrs().test(
      characteristics::TypeAndShape::Attr::AssumedSize)};
  bool dummyIsAsynchronous{
      dummy.attrs.test(characteristics::DummyDataObject::Attr::Asynchronous)};
  bool dummyIsVolatile{
      dummy.attrs.test(characteristics::DummyDataObject::Attr::Volatile)};
  bool dummyIsValue{
      dummy.attrs.test(characteristics::DummyDataObject::Attr::Value)};

  if (actualIsPolymorphic && dummyIsPolymorphic &&
      actualIsCoindexed) {  // 15.5.2.4(2)
    messages.Say(
        "Coindexed polymorphic object may not be associated with a polymorphic %s"_err_en_US,
        dummyName);
  }
  if (actualIsPolymorphic && !dummyIsPolymorphic &&
      actualIsAssumedSize) {  // 15.5.2.4(2)
    messages.Say(
        "Assumed-size polymorphic array may not be associated with a monomorphic %s"_err_en_US,
        dummyName);
  }

  // Derived type actual argument checks
  const Symbol *actualFirstSymbol{evaluate::GetFirstSymbol(actual)};
  bool actualIsAsynchronous{
      actualFirstSymbol && actualFirstSymbol->attrs().test(Attr::ASYNCHRONOUS)};
  bool actualIsVolatile{
      actualFirstSymbol && actualFirstSymbol->attrs().test(Attr::VOLATILE)};
  if (!actualType.type().IsUnlimitedPolymorphic() &&
      actualType.type().category() == TypeCategory::Derived) {
    const auto &derived{actualType.type().GetDerivedTypeSpec()};
    if (dummy.type.type().IsAssumedType()) {
      if (!derived.parameters().empty()) {  // 15.5.2.4(2)
        messages.Say(
            "Actual argument associated with TYPE(*) %s may not have a parameterized derived type"_err_en_US,
            dummyName);
      }
      if (const Symbol *
          tbp{FindImmediateComponent(derived,
              [](const Symbol &symbol) {
                return symbol.has<ProcBindingDetails>();
              })}) {  // 15.5.2.4(2)
        if (auto *msg{messages.Say(
                "Actual argument associated with TYPE(*) %s may not have type-bound procedure '%s'"_err_en_US,
                dummyName, tbp->name())}) {
          msg->Attach(tbp->name(), "Declaration of type-bound procedure"_en_US);
        }
      }
      if (const Symbol *
          finalizer{FindImmediateComponent(derived,
              [](const Symbol &symbol) {
                return symbol.has<FinalProcDetails>();
              })}) {  // 15.5.2.4(2)
        if (auto *msg{messages.Say(
                "Actual argument associated with TYPE(*) %s may not have FINAL subroutine '%s'"_err_en_US,
                dummyName, finalizer->name())}) {
          msg->Attach(
              finalizer->name(), "Declaration of FINAL subroutine"_en_US);
        }
      }
    }
    UltimateComponentIterator ultimates{derived};
    if (actualIsCoindexed && dummy.intent != common::Intent::In &&
        !dummyIsValue) {
      if (auto iter{std::find_if(
              ultimates.begin(), ultimates.end(), [](const Symbol *component) {
                return IsAllocatable(DEREF(component));
              })}) {  // 15.5.2.4(6)
        if (auto *msg{messages.Say(
                "Coindexed actual argument with ALLOCATABLE ultimate component '%s' must be associated with a %s with VALUE or INTENT(IN) attributes"_err_en_US,
                iter.BuildResultDesignatorName(), dummyName)}) {
          msg->Attach(
              (*iter)->name(), "Declaration of ALLOCATABLE component"_en_US);
        }
      }
    }
    if (actualIsVolatile != dummyIsVolatile) {  // 15.5.2.4(22)
      if (auto iter{std::find_if(
              ultimates.begin(), ultimates.end(), [](const Symbol *component) {
                const auto *object{
                    DEREF(component).detailsIf<ObjectEntityDetails>()};
                return object && object->IsCoarray();
              })}) {
        if (auto *msg{messages.Say(
                "VOLATILE attribute must match for %s when actual argument has a coarray ultimate component '%s'"_err_en_US,
                dummyName, iter.BuildResultDesignatorName())}) {
          msg->Attach(
              (*iter)->name(), "Declaration of coarray component"_en_US);
        }
      }
    }
  }

  // Rank and shape checks
  const auto *actualLastSymbol{evaluate::GetLastSymbol(actual)};
  if (actualLastSymbol != nullptr) {
    actualLastSymbol = GetAssociationRoot(*actualLastSymbol);
  }
  const ObjectEntityDetails *actualLastObject{actualLastSymbol
          ? actualLastSymbol->GetUltimate().detailsIf<ObjectEntityDetails>()
          : nullptr};
  int actualRank{evaluate::GetRank(actualType.shape())};
  bool actualIsPointer{actualLastSymbol && IsPointer(*actualLastSymbol)};
  if (dummy.type.attrs().test(
          characteristics::TypeAndShape::Attr::AssumedShape)) {
    // 15.5.2.4(16)
    if (actualRank == 0) {
      messages.Say(
          "Scalar actual argument may not be associated with assumed-shape %s"_err_en_US,
          dummyName);
    }
    if (actualIsAssumedSize) {
      if (auto *msg{messages.Say(
              "Assumed-size array may not be associated with assumed-shape %s"_err_en_US,
              dummyName)}) {
        msg->Attach(actualLastSymbol->name(),
            "Declaration of assumed-size array actual argument"_en_US);
      }
    }
  } else if (actualRank == 0 && dummy.type.Rank() > 0) {
    // Actual is scalar, dummy is an array.  15.5.2.4(14), 15.5.2.11
    if (actualIsCoindexed) {
      messages.Say(
          "Coindexed scalar actual argument must be associated with a scalar %s"_err_en_US,
          dummyName);
    }
    if (actualLastSymbol && actualLastSymbol->Rank() == 0 &&
        !(dummy.type.type().IsAssumedType() && dummyIsAssumedSize)) {
      messages.Say(
          "Whole scalar actual argument may not be associated with a %s array"_err_en_US,
          dummyName);
    }
    if (actualIsPolymorphic) {
      messages.Say(
          "Polymorphic scalar may not be associated with a %s array"_err_en_US,
          dummyName);
    }
    if (actualIsPointer) {
      messages.Say(
          "Scalar POINTER target may not be associated with a %s array"_err_en_US,
          dummyName);
    }
    if (actualLastObject && actualLastObject->IsAssumedShape()) {
      messages.Say(
          "Element of assumed-shape array may not be associated with a %s array"_err_en_US,
          dummyName);
    }
  }
  if (actualLastObject && actualLastObject->IsCoarray() &&
      IsAllocatable(*actualLastSymbol) &&
      dummy.intent == common::Intent::Out) {  // C846
    messages.Say(
        "ALLOCATABLE coarray '%s' may not be associated with INTENT(OUT) %s"_err_en_US,
        actualLastSymbol->name(), dummyName);
  }

  // Definability
  const char *reason{nullptr};
  if (dummy.intent == common::Intent::Out) {
    reason = "INTENT(OUT)";
  } else if (dummy.intent == common::Intent::InOut) {
    reason = "INTENT(IN OUT)";
  } else if (dummyIsAsynchronous) {
    reason = "ASYNCHRONOUS";
  } else if (dummyIsVolatile) {
    reason = "VOLATILE";
  }
  if (reason != nullptr && scope != nullptr) {
    bool vectorSubscriptIsOk{isElemental || dummyIsValue};  // 15.5.2.4(21)
    std::unique_ptr<parser::Message> why{
        WhyNotModifiable(messages.at(), actual, *scope, vectorSubscriptIsOk)};
    if (why.get() != nullptr) {
      if (auto *msg{messages.Say(
              "Actual argument associated with %s %s must be definable"_err_en_US,
              reason, dummyName)}) {
        msg->Attach(std::move(why));
      }
    }
  }

  // Cases when temporaries might be needed but must not be permitted.
  bool dummyIsPointer{
      dummy.attrs.test(characteristics::DummyDataObject::Attr::Pointer)};
  if ((actualIsAsynchronous || actualIsVolatile) &&
      (dummyIsAsynchronous || dummyIsVolatile) && !dummyIsValue) {
    if (actualIsCoindexed) {  // C1538
      messages.Say(
          "Coindexed ASYNCHRONOUS or VOLATILE actual argument may not be associated with %s with ASYNCHRONOUS or VOLATILE attributes unless VALUE"_err_en_US,
          dummyName);
    }
    if (actualRank > 0 && !IsSimplyContiguous(actual, context.intrinsics())) {
      bool dummyIsContiguous{
          dummy.attrs.test(characteristics::DummyDataObject::Attr::Contiguous)};
      bool dummyIsAssumedRank{dummy.type.attrs().test(
          characteristics::TypeAndShape::Attr::AssumedRank)};
      bool dummyIsAssumedShape{dummy.type.attrs().test(
          characteristics::TypeAndShape::Attr::AssumedShape)};
      if (dummyIsContiguous ||
          !(dummyIsAssumedShape || dummyIsAssumedRank ||
              (actualIsPointer && dummyIsPointer))) {  // C1539 & C1540
        messages.Say(
            "ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous %s"_err_en_US,
            dummyName);
      }
    }
  }

  // 15.5.2.5 -- actual & dummy are both POINTER or both ALLOCATABLE
  bool actualIsAllocatable{
      actualLastSymbol && IsAllocatable(*actualLastSymbol)};
  bool dummyIsAllocatable{
      dummy.attrs.test(characteristics::DummyDataObject::Attr::Allocatable)};
  if ((actualIsPointer && dummyIsPointer) ||
      (actualIsAllocatable && dummyIsAllocatable)) {
    if (dummyIsPolymorphic != actualIsPolymorphic) {
      messages.Say(
          "If a POINTER or ALLOCATABLE dummy or actual argument is polymorphic, both must be so"_err_en_US);
    }
    bool actualIsUnlimited{actualType.type().IsUnlimitedPolymorphic()};
    bool dummyIsUnlimited{dummy.type.type().IsUnlimitedPolymorphic()};
    if (!actualIsUnlimited) {
      if (dummyIsUnlimited) {
        messages.Say(
            "If a POINTER or ALLOCATABLE dummy or actual argument is unlimited polymorphic, both must be so"_err_en_US);
      } else if (typesCompatible) {
        if (!actualType.type().IsTypeCompatibleWith(dummy.type.type())) {
          messages.Say(
              "POINTER or ALLOCATABLE dummy and actual arguments must have the same declared type"_err_en_US);
        }
        if (actualType.type().category() == TypeCategory::Derived &&
            !DefersSameTypeParameters(actualType.type().GetDerivedTypeSpec(),
                dummy.type.type().GetDerivedTypeSpec())) {
          messages.Say(
              "Dummy and actual arguments must defer the same type parameters when POINTER or ALLOCATABLE"_err_en_US);
        }
      }
    }
  }
}

static void CheckExplicitInterfaceArg(evaluate::ActualArgument &arg,
    const characteristics::DummyArgument &dummy,
    const characteristics::Procedure &proc, evaluate::FoldingContext &context,
    const Scope *scope) {
  auto &messages{context.messages()};
  std::string dummyName{"dummy argument"};
  if (!dummy.name.empty()) {
    dummyName += " '"s + parser::ToLowerCaseLetters(dummy.name) + "='";
  }
  std::visit(
      common::visitors{
          [&](const characteristics::DummyDataObject &object) {
            if (auto *expr{arg.UnwrapExpr()}) {
              if (auto type{characteristics::TypeAndShape::Characterize(
                      *expr, context)}) {
                bool isElemental{object.type.Rank() == 0 && proc.IsElemental()};
                CheckExplicitDataArg(object, dummyName, *expr, *type,
                    isElemental, context, scope);
              } else if (object.type.type().IsTypelessIntrinsicArgument() &&
                  std::holds_alternative<evaluate::BOZLiteralConstant>(
                      expr->u)) {
                // ok
              } else {
                messages.Say(
                    "Actual argument is not a variable or typed expression"_err_en_US);
              }
            } else if (const Symbol * assumed{arg.GetAssumedTypeDummy()}) {
              // An assumed-type dummy is being forwarded.
              if (!object.type.type().IsAssumedType()) {
                messages.Say(
                    "Assumed-type TYPE(*) '%s' may be associated only with an assumed-TYPE(*) %s"_err_en_US,
                    assumed->name(), dummyName);
              }
            } else {
              messages.Say(
                  "Actual argument is not an expression or variable"_err_en_US);
            }
          },
          [](const auto &) {
            // TODO check actual procedure compatibility
            // TODO check alternate return
          },
      },
      dummy.u);
}

static void RearrangeArguments(const characteristics::Procedure &proc,
    evaluate::ActualArguments &actuals, parser::ContextualMessages &messages) {
  CHECK(proc.HasExplicitInterface());
  if (actuals.size() < proc.dummyArguments.size()) {
    actuals.resize(proc.dummyArguments.size());
  } else if (actuals.size() > proc.dummyArguments.size()) {
    messages.Say(
        "Too many actual arguments (%zd) passed to procedure that expects only %zd"_err_en_US,
        actuals.size(), proc.dummyArguments.size());
  }
  std::map<std::string, evaluate::ActualArgument> kwArgs;
  for (auto &x : actuals) {
    if (x.has_value()) {
      if (x->keyword.has_value()) {
        auto emplaced{
            kwArgs.try_emplace(x->keyword->ToString(), std::move(*x))};
        if (!emplaced.second) {
          messages.Say(*x->keyword,
              "Argument keyword '%s=' appears on more than one effective argument in this procedure reference"_err_en_US,
              *x->keyword);
        }
        x.reset();
      }
    }
  }
  if (!kwArgs.empty()) {
    int index{0};
    for (const auto &dummy : proc.dummyArguments) {
      if (!dummy.name.empty()) {
        auto iter{kwArgs.find(dummy.name)};
        if (iter != kwArgs.end()) {
          evaluate::ActualArgument &x{iter->second};
          if (actuals[index].has_value()) {
            messages.Say(*x.keyword,
                "Keyword argument '%s=' has already been specified positionally (#%d) in this procedure reference"_err_en_US,
                *x.keyword, index + 1);
          } else {
            actuals[index] = std::move(x);
          }
          kwArgs.erase(iter);
        }
      }
      ++index;
    }
    for (auto &bad : kwArgs) {
      evaluate::ActualArgument &x{bad.second};
      messages.Say(*x.keyword,
          "Argument keyword '%s=' is not recognized for this procedure reference"_err_en_US,
          *x.keyword);
    }
  }
}

static parser::Messages CheckExplicitInterface(
    const characteristics::Procedure &proc, evaluate::ActualArguments &actuals,
    const evaluate::FoldingContext &context, const Scope *scope) {
  parser::Messages buffer;
  parser::ContextualMessages messages{context.messages().at(), &buffer};
  evaluate::FoldingContext localContext{context, messages};
  RearrangeArguments(proc, actuals, messages);
  if (buffer.empty()) {
    int index{0};
    for (auto &actual : actuals) {
      const auto &dummy{proc.dummyArguments.at(index++)};
      if (actual.has_value()) {
        CheckExplicitInterfaceArg(*actual, dummy, proc, localContext, scope);
      } else if (!dummy.IsOptional()) {
        if (dummy.name.empty()) {
          messages.Say(
              "Dummy argument #%d is not OPTIONAL and is not associated with "
              "an actual argument in this procedure reference"_err_en_US,
              index);
        } else {
          messages.Say("Dummy argument '%s=' (#%d) is not OPTIONAL and is not "
                       "associated with an actual argument in this procedure "
                       "reference"_err_en_US,
              dummy.name, index);
        }
      }
    }
  }
  return buffer;
}

parser::Messages CheckExplicitInterface(const characteristics::Procedure &proc,
    evaluate::ActualArguments &actuals, const evaluate::FoldingContext &context,
    const Scope &scope) {
  return CheckExplicitInterface(proc, actuals, context, &scope);
}

bool CheckInterfaceForGeneric(const characteristics::Procedure &proc,
    evaluate::ActualArguments &actuals,
    const evaluate::FoldingContext &context) {
  return CheckExplicitInterface(proc, actuals, context, nullptr).empty();
}

void CheckArguments(const characteristics::Procedure &proc,
    evaluate::ActualArguments &actuals, evaluate::FoldingContext &context,
    const Scope &scope, bool treatingExternalAsImplicit) {
  bool explicitInterface{proc.HasExplicitInterface()};
  if (explicitInterface) {
    auto buffer{CheckExplicitInterface(proc, actuals, context, scope)};
    if (treatingExternalAsImplicit && !buffer.empty()) {
      if (auto *msg{context.messages().Say(
              "Warning: if the procedure's interface were explicit, this reference would be in error:"_en_US)}) {
        buffer.AttachTo(*msg);
      }
    }
    if (auto *msgs{context.messages().messages()}) {
      msgs->Merge(std::move(buffer));
    }
  }
  if (!explicitInterface || treatingExternalAsImplicit) {
    for (auto &actual : actuals) {
      if (actual.has_value()) {
        CheckImplicitInterfaceArg(*actual, context.messages());
      }
    }
  }
}
}
