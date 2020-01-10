//===-- lib/semantics/check-arithmeticif.cc -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-arithmeticif.h"
#include "tools.h"
#include "../parser/message.h"
#include "../parser/parse-tree.h"

namespace Fortran::semantics {

bool IsNumericExpr(const SomeExpr &expr) {
  auto dynamicType{expr.GetType()};
  return dynamicType && common::IsNumericTypeCategory(dynamicType->category());
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
