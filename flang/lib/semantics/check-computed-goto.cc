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

#include "check-computed-goto.h"
#include "attr.h"
#include "scope.h"
#include "semantics.h"
#include "symbol.h"
#include "tools.h"
#include "type.h"
#include "../evaluate/traversal.h"
#include "../parser/message.h"
#include "../parser/parse-tree.h"

namespace Fortran::semantics {

void ComputedGotoStmtChecker::Leave(
    const parser::ComputedGotoStmt &computedGotoStmt) {
  // C1169 Labels have already been checked
  // R1158 Check for scalar-int-expr
  auto &expr{
      std::get<parser::ScalarIntExpr>(computedGotoStmt.t).thing.thing.value()};
  if (expr.typedExpr->v.Rank() > 0) {
    context_.messages().Say(expr.source,
        "Computed GOTO expression must be a scalar expression"_err_en_US);
  } else if (!ExprHasTypeCategory(
                 *expr.typedExpr, common::TypeCategory::Integer)) {
    context_.messages().Say(expr.source,
        "Computed GOTO expression must be an integer expression"_err_en_US);
  }
}

}  // namespace Fortran::semantics
