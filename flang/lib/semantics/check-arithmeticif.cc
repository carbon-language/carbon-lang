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

#include "check-arithmeticif.h"
#include "tools.h"
#include "../parser/message.h"
#include "../parser/parse-tree.h"

namespace Fortran::semantics {

bool IsNumericExpr(const SomeExpr &expr) {
  auto dynamicType{expr.GetType()};
  return dynamicType.has_value() &&
      common::IsNumericTypeCategory(dynamicType->category);
}

void ArithmeticIfStmtChecker::Leave(
    const parser::ArithmeticIfStmt &arithmeticIfStmt) {
  // Arithmetic IF statements have been removed from Fortran 2018.
  // The constraints and requirements here refer to the 2008 spec.
  // R853 Check for a scalar-numeric-expr
  // C849 that shall not be of type complex.
  auto &parsedExpr{std::get<parser::Expr>(arithmeticIfStmt.t)};
  if (const auto *expr{GetExpr(parsedExpr)}) {
    if (expr->Rank() > 0) {
      context_.Say(parsedExpr.source,
          "ARITHMETIC IF expression must be a scalar expression"_err_en_US);
    } else if (ExprHasTypeCategory(*expr, common::TypeCategory::Complex)) {
      context_.Say(parsedExpr.source,
          "ARITHMETIC IF expression must not be a COMPLEX expression"_err_en_US);
    } else if (!IsNumericExpr(*expr)) {
      context_.Say(parsedExpr.source,
          "ARITHMETIC IF expression must be a numeric expression"_err_en_US);
    }
  }
  // The labels have already been checked in resolve-labels.
  // TODO: Really?  Check that they are really branch target
  // statements and in the same inclusive scope.
}

}  // namespace Fortran::semantics
