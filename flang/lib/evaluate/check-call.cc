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
#include "characteristics.h"
#include "shape.h"
#include "tools.h"
#include "../parser/message.h"
#include "../semantics/scope.h"
#include <map>
#include <string>

using namespace Fortran::parser::literals;

namespace Fortran::evaluate {

static void CheckImplicitInterfaceArg(
    ActualArgument &arg, parser::ContextualMessages &messages) {
  if (const auto &kw{arg.keyword}) {
    messages.Say(*kw,
        "Keyword '%s=' cannot appear in a reference to a procedure with an implicit interface"_err_en_US,
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
    if (auto named{ExtractNamedEntity(*expr)}) {
      const semantics::Symbol &symbol{named->GetLastSymbol()};
      if (symbol.Corank() > 0) {
        messages.Say(
            "Coarray argument requires an explicit interface"_err_en_US);
      }
      if (const auto *details{
              symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
        if (details->IsAssumedRank()) {
          messages.Say(
              "Assumed rank argument requires an explicit interface"_err_en_US);
        }
      }
      if (symbol.attrs().test(semantics::Attr::ASYNCHRONOUS)) {
        messages.Say(
            "ASYNCHRONOUS argument requires an explicit interface"_err_en_US);
      }
      if (symbol.attrs().test(semantics::Attr::VOLATILE)) {
        messages.Say(
            "VOLATILE argument requires an explicit interface"_err_en_US);
      }
    }
  }
}

struct TypeConcerns {
  const semantics::Symbol *typeBoundProcedure{nullptr};
  const semantics::Symbol *finalProcedure{nullptr};
  const semantics::Symbol *allocatable{nullptr};
  const semantics::Symbol *coarray{nullptr};
};

static void InspectType(
    const semantics::DerivedTypeSpec &derived, TypeConcerns &concerns) {
  if (const auto *scope{derived.typeSymbol().scope()}) {
    for (const auto &pair : *scope) {
      const semantics::Symbol &component{*pair.second};
      if (const auto *object{
              component.detailsIf<semantics::ObjectEntityDetails>()}) {
        if (component.attrs().test(semantics::Attr::ALLOCATABLE)) {
          concerns.allocatable = &component;
        }
        if (object->IsCoarray()) {
          concerns.coarray = &component;
        }
        if (component.flags().test(semantics::Symbol::Flag::ParentComp)) {
          if (const auto *type{object->type()}) {
            if (const auto *parent{type->AsDerived()}) {
              InspectType(*parent, concerns);
            }
          }
        }
      } else if (component.has<semantics::ProcBindingDetails>()) {
        concerns.typeBoundProcedure = &component;
      } else if (component.has<semantics::FinalProcDetails>()) {
        concerns.finalProcedure = &component;
      }
    }
  }
}

static void CheckExplicitDataArg(const characteristics::DummyDataObject &dummy,
    const Expr<SomeType> &actual,
    const characteristics::TypeAndShape &actualType,
    parser::ContextualMessages &messages) {
  dummy.type.IsCompatibleWith(messages, actualType);
  bool actualIsPolymorphic{actualType.type().IsPolymorphic()};
  bool dummyIsPolymorphic{dummy.type.type().IsPolymorphic()};
  bool actualIsCoindexed{ExtractCoarrayRef(actual).has_value()};
  bool actualIsAssumedSize{actualType.attrs().test(
      characteristics::TypeAndShape::Attr::AssumedSize)};
  bool dummyIsAssumedSize{dummy.type.attrs().test(
      characteristics::TypeAndShape::Attr::AssumedSize)};
  if (actualIsPolymorphic && dummyIsPolymorphic &&
      actualIsCoindexed) {  // 15.5.2.4(2)
    messages.Say(
        "Coindexed polymorphic object may not be associated with a polymorphic dummy argument"_err_en_US);
  }
  if (actualIsPolymorphic && !dummyIsPolymorphic &&
      actualIsAssumedSize) {  // 15.5.2.4(2)
    messages.Say(
        "Assumed-size polymorphic array may not be associated with a monomorphic dummy argument"_err_en_US);
  }
  if (!actualType.type().IsUnlimitedPolymorphic() &&
      actualType.type().category() == TypeCategory::Derived) {
    const auto &derived{actualType.type().GetDerivedTypeSpec()};
    TypeConcerns concerns;
    InspectType(derived, concerns);
    if (dummy.type.type().IsAssumedType()) {
      if (!derived.parameters().empty()) {  // 15.5.2.4(2)
        messages.Say(
            "Actual argument associated with TYPE(*) dummy argument may not have a parameterized derived type"_err_en_US);
      }
      if (concerns.typeBoundProcedure) {  // 15.5.2.4(2)
        if (auto *msg{messages.Say(
                "Actual argument associated with TYPE(*) dummy argument may not have type-bound procedures"_err_en_US)}) {
          msg->Attach(concerns.typeBoundProcedure->name(),
              "Declaration of type-bound procedure"_en_US);
        }
      }
      if (concerns.finalProcedure) {  // 15.5.2.4(2)
        if (auto *msg{messages.Say(
                "Actual argument associated with TYPE(*) dummy argument may not have FINAL procedures"_err_en_US)}) {
          msg->Attach(concerns.finalProcedure->name(),
              "Declaration of FINAL procedure"_en_US);
        }
      }
    }
    if (actualIsCoindexed && concerns.allocatable &&
        dummy.intent != common::Intent::In &&
        !dummy.attrs.test(characteristics::DummyDataObject::Attr::Value)) {
      // 15.5.2.4(6)
      if (auto *msg{messages.Say(
              "Coindexed actual argument with ALLOCATABLE ultimate component must be associated with a dummy argument with VALUE or INTENT(IN) attributes"_err_en_US)}) {
        msg->Attach(concerns.allocatable->name(),
            "Declaration of ALLOCATABLE component"_en_US);
      }
    }
  }
  const auto *actualLastSymbol{GetLastSymbol(actual)};
  const semantics::ObjectEntityDetails *actualLastObject{actualLastSymbol
          ? actualLastSymbol->detailsIf<semantics::ObjectEntityDetails>()
          : nullptr};
  int actualRank{GetRank(actualType.shape())};
  int dummyRank{GetRank(dummy.type.shape())};
  if (dummy.type.attrs().test(
          characteristics::TypeAndShape::Attr::AssumedShape)) {
    // 15.5.2.4(16)
    if (actualRank != dummyRank) {
      messages.Say(
          "Rank of actual argument (%d) differs from assumed-shape dummy argument (%d)"_err_en_US,
          actualRank, dummyRank);
    }
    if (actualIsAssumedSize) {
      if (auto *msg{messages.Say(
              "Assumed-size array cannot be associated with assumed-shape dummy argument"_err_en_US)}) {
        msg->Attach(actualLastSymbol->name(),
            "Declaration of assumed-size array actual argument"_en_US);
      }
    }
  } else if (actualRank == 0 && dummyRank > 0) {
    // Actual is scalar, dummy is an array.  15.5.2.4(14), 15.5.2.11
    if (actualIsCoindexed) {
      messages.Say(
          "Coindexed scalar actual argument must be associated with a scalar dummy argument"_err_en_US);
    }
    if (actualLastSymbol && actualLastSymbol->Rank() == 0 &&
        !(dummy.type.type().IsAssumedType() && dummyIsAssumedSize)) {
      messages.Say(
          "Whole scalar actual argument may not be associated with a dummy argument array"_err_en_US);
    }
    if (actualIsPolymorphic) {
      messages.Say(
          "Element of polymorphic array may not be associated with a dummy argument array"_err_en_US);
    }
    if (actualLastSymbol &&
        actualLastSymbol->attrs().test(semantics::Attr::POINTER)) {
      messages.Say(
          "Element of pointer array may not be associated with a dummy argument array"_err_en_US);
    }
    if (actualLastObject && actualLastObject->IsAssumedShape()) {
      messages.Say(
          "Element of assumed-shape array may not be associated with a dummy argument array"_err_en_US);
    }
  }
  // TODO pmk more here
}

static void CheckExplicitInterfaceArg(const ActualArgument &arg,
    const characteristics::DummyArgument &dummy, FoldingContext &context) {
  auto &messages{context.messages()};
  std::visit(
      common::visitors{
          [&](const characteristics::DummyDataObject &object) {
            if (const auto *expr{arg.UnwrapExpr()}) {
              if (auto type{characteristics::TypeAndShape::Characterize(
                      *expr, context)}) {
                CheckExplicitDataArg(object, *expr, *type, context.messages());
              } else if (object.type.type().IsTypelessIntrinsicArgument() &&
                  std::holds_alternative<BOZLiteralConstant>(expr->u)) {
                // ok
              } else {
                messages.Say(
                    "Actual argument is not a variable or typed expression"_err_en_US);
              }
            } else if (const semantics::Symbol *
                assumed{arg.GetAssumedTypeDummy()}) {
              // An assumed-type dummy is being forwarded.
              if (!object.type.type().IsAssumedType()) {
                messages.Say(
                    "Assumed-type TYPE(*) '%s' may be associated only with an assumed-TYPE(*) dummy argument"_err_en_US,
                    assumed->name());
              }
            } else {
              messages.Say(
                  "Actual argument is not an expression or variable"_err_en_US);
            }
          },
          [&](const characteristics::DummyProcedure &) {
            // TODO check effective procedure compatibility
          },
          [&](const characteristics::AlternateReturn &) {
            // TODO check alternate return
          },
      },
      dummy.u);
  return true;  // TODO: return false when error detected
}

static bool RearrangeArguments(const characteristics::Procedure &proc,
    ActualArguments &actuals, parser::ContextualMessages &messages) {
  CHECK(proc.HasExplicitInterface());
  if (actuals.size() < proc.dummyArguments.size()) {
    actuals.resize(proc.dummyArguments.size());
  } else if (actuals.size() > proc.dummyArguments.size()) {
    messages.Say(
        "Too many actual arguments (%zd) passed to procedure that expects only %zd"_err_en_US,
        actuals.size(), proc.dummyArguments.size());
    return false;
  }
  std::map<std::string, ActualArgument> kwArgs;
  for (auto &x : actuals) {
    if (x.has_value()) {
      if (x->keyword.has_value()) {
        auto emplaced{
            kwArgs.try_emplace(x->keyword->ToString(), std::move(*x))};
        if (!emplaced.second) {
          messages.Say(*x->keyword,
              "Argument keyword '%s=' appears on more than one effective argument in this procedure reference"_err_en_US,
              *x->keyword);
          return false;
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
          ActualArgument &x{iter->second};
          if (actuals[index].has_value()) {
            messages.Say(*x.keyword,
                "Keyword argument '%s=' has already been specified positionally (#%d) in this procedure reference"_err_en_US,
                *x.keyword, index + 1);
            return false;
          } else {
            actuals[index] = std::move(x);
          }
          kwArgs.erase(iter);
        }
      }
      ++index;
    }
    for (auto &bad : kwArgs) {
      ActualArgument &x{bad.second};
      messages.Say(*x.keyword,
          "Argument keyword '%s=' is not recognized for this procedure reference"_err_en_US,
          *x.keyword);
      return false;
    }
  }
  return true;
}

bool CheckExplicitInterface(const characteristics::Procedure &proc,
    ActualArguments &actuals, FoldingContext &context) {
  if (!RearrangeArguments(proc, actuals, context.messages())) {
    return false;
  }
  int index{0};
  for (auto &actual : actuals) {
    const auto &dummy{proc.dummyArguments[index++]};
    if (actual.has_value()) {
      if (!CheckExplicitInterfaceArg(*actual, dummy, context)) {
        return false;
      }
    } else if (!dummy.IsOptional()) {
      if (dummy.name.empty()) {
        context.messages().Say(
            "Dummy argument #%d is not OPTIONAL and is not associated with an "
            "effective argument in this procedure reference"_err_en_US,
            index);
      } else {
        context.messages().Say(
            "Dummy argument '%s' (#%d) is not OPTIONAL and is not associated "
            "with an effective argument in this procedure reference"_err_en_US,
            dummy.name, index);
      }
      return false;
    }
  }
  return true;
}

void CheckArguments(const characteristics::Procedure &proc,
    ActualArguments &actuals, FoldingContext &context,
    bool treatingExternalAsImplicit) {
  parser::Messages buffer;
  parser::ContextualMessages messages{context.messages().at(), &buffer};
  if (proc.HasExplicitInterface()) {
    FoldingContext localContext{context, messages};
    CheckExplicitInterface(proc, actuals, localContext);
  }
  if (!proc.HasExplicitInterface() || treatingExternalAsImplicit) {
    for (auto &actual : actuals) {
      if (actual.has_value()) {
        CheckImplicitInterfaceArg(*actual, messages);
      }
    }
  }
  if (!buffer.empty()) {
    if (treatingExternalAsImplicit) {
      if (auto *msg{context.messages().Say(
              "Warning: if the procedure's interface were explicit, this reference would be in error:"_en_US)}) {
        buffer.AttachTo(*msg);
      }
    } else if (auto *msgs{context.messages().messages()}) {
      msgs->Merge(std::move(buffer));
    }
  }
}
}
