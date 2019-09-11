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

namespace Fortran::semantics {

static void CheckSymbol(SemanticsContext &context, const Symbol &symbol) {
  if (context.HasError(symbol)) {
    return;
  }
  context.set_location(symbol.name());
  if (IsAssumedLengthCharacterFunction(symbol)) {  // C723
    if (symbol.attrs().test(Attr::RECURSIVE)) {
      context.Say(
          "An assumed-length CHARACTER(*) function cannot be RECURSIVE."_err_en_US);
    }
    if (symbol.Rank() > 0) {
      context.Say(
          "An assumed-length CHARACTER(*) function cannot return an array."_err_en_US);
    }
    if (symbol.attrs().test(Attr::PURE)) {
      context.Say(
          "An assumed-length CHARACTER(*) function cannot be PURE."_err_en_US);
    }
    if (symbol.attrs().test(Attr::ELEMENTAL)) {
      context.Say(
          "An assumed-length CHARACTER(*) function cannot be ELEMENTAL."_err_en_US);
    }
    if (const Symbol * result{FindFunctionResult(symbol)}) {
      if (result->attrs().test(Attr::POINTER)) {
        context.Say(
            "An assumed-length CHARACTER(*) function cannot return a POINTER."_err_en_US);
      }
    }
  }
}

static void CheckScope(SemanticsContext &context, const Scope &scope) {
  for (const auto &pair : scope) {
    CheckSymbol(context, *pair.second);
  }
  for (const Scope &child : scope.children()) {
    CheckScope(context, child);
  }
}

void CheckDeclarations(SemanticsContext &context) {
  CheckScope(context, context.globalScope());
}
}
