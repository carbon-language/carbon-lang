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

#include "check-return.h"
#include "semantics.h"
#include "tools.h"
#include "../parser/message.h"
#include "../parser/parse-tree.h"

namespace Fortran::semantics {

const Scope *FindContainingSubprogram(const Scope &start) {
  const Scope *scope{&start};
  while (!scope->IsGlobal()) {
    switch (scope->kind()) {
    case Scope::Kind::MainProgram:
    case Scope::Kind::Subprogram: return scope;
    default: scope = &scope->parent(); break;
    }
  }
  return nullptr;
}

void ReturnStmtChecker::Leave(const parser::ReturnStmt &returnStmt) {
  // R1542 Expression analysis validates the scalar-int-expr
  // C1574 The return-stmt shall be in the inclusive scope of a function or
  // subroutine subprogram.
  // C1575 The scalar-int-expr is allowed only in the inclusive scope of a
  // subroutine subprogram.
  const auto &scope{context_.FindScope(context_.location().value())};
  const auto *subprogramScope{FindContainingSubprogram(scope)};
  if (!subprogramScope) {
    context_.Say(
        "RETURN must in the inclusive scope of a SUBPROGRAM"_err_en_US);
    return;
  }
  if (returnStmt.v && subprogramScope->kind() == Scope::Kind::Subprogram) {
    if (IsFunction(*subprogramScope->GetSymbol())) {
      context_.Say(
          "RETURN with expression is only allowed in SUBROUTINE subprogram"_err_en_US);
    }
  }
}

}  // namespace Fortran::semantics
