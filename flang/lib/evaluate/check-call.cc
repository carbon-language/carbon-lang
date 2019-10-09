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

static bool CheckExplicitInterfaceArg(const ActualArgument &arg,
    const characteristics::DummyArgument &dummy, FoldingContext &context) {
  std::visit(
      common::visitors{
          [&](const characteristics::DummyDataObject &object) {
            if (const auto *expr{arg.UnwrapExpr()}) {
              if (auto type{characteristics::GetTypeAndShape(*expr, context)}) {
                object.type.IsCompatibleWith(context.messages(), *type);
              } else {
                // TODO
              }
            } else {
              // TODO
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
