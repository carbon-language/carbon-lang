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

#include "check-nullify.h"
#include "expression.h"
#include "tools.h"
#include "../evaluate/expression.h"
#include "../parser/message.h"
#include "../parser/parse-tree.h"

namespace Fortran::semantics {

void NullifyChecker::Leave(const parser::NullifyStmt &nullifyStmt) {
  for (const parser::PointerObject &pointerObject : nullifyStmt.v) {
    std::visit(
        common::visitors{
            [&](const parser::Name &name) {
              auto const *symbol{name.symbol};
              if (context_.HasError(symbol)) {
                // already reported an error
              } else if (!IsVariableName(*symbol) && !IsProcName(*symbol)) {
                context_.Say(name.source,
                    "name in NULLIFY statement must be a variable or procedure pointer name"_err_en_US);
              } else if (!IsPointer(*symbol)) {  // C951
                context_.Say(name.source,
                    "name in NULLIFY statement must have the POINTER attribute"_err_en_US);
              }
            },
            [&](const parser::StructureComponent &structureComponent) {
              evaluate::ExpressionAnalyzer analyzer{context_};
              if (MaybeExpr checked{analyzer.Analyze(structureComponent)}) {
                if (!IsPointer(*structureComponent.component.symbol)) {  // C951
                  context_.Say(structureComponent.component.source,
                      "component in NULLIFY statement must have the POINTER attribute"_err_en_US);
                }
              }
            },
        },
        pointerObject.u);
  }
  // From 9.7.3.1(1)
  //   A pointer-object shall not depend on the value,
  //   bounds, or association status of another pointer-
  //   object in the same NULLIFY statement.
  // This restriction is the programmer's responsibilty.
  // Some dependencies can be found compile time or at
  // runtime, but for now we choose to skip such checks.
}
}  // namespace Fortran::semantics
