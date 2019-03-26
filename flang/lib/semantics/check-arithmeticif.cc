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

#include "attr.h"
#include "check-arithmeticif.h"
#include "scope.h"
#include "semantics.h"
#include "symbol.h"
#include "tools.h"
#include "type.h"
#include "../evaluate/traversal.h"
#include "../parser/message.h"
#include "../parser/parse-tree.h"

namespace Fortran::semantics {

class ArithmeticIfStmtContext {
public:
  ArithmeticIfStmtContext(SemanticsContext &context)
    : messages_{context.messages()} {}

  bool operator==(const ArithmeticIfStmtContext &x) const { return this == &x; }
  bool operator!=(const ArithmeticIfStmtContext &x) const { return this != &x; }

  // Arithmetic IF statements have been removed from Fortran 2018.
  // The constraints and requirements here refer to the 2008 spec.
  void Check(const parser::ArithmeticIfStmt &arithmeticIfStmt) {
    // R853 Check for a scalar-numeric-expr
    // C849 that shall not be of type complex.
    auto &expr{std::get<parser::Expr>(arithmeticIfStmt.t)};
    if (expr.typedExpr->v.Rank() > 0) {
      messages_.Say(expr.source,
          "ARITHMETIC IF statement must have a scalar numeric expression"_err_en_US);
    } else if (ExprHasTypeCategory(
                   *expr.typedExpr, common::TypeCategory::Complex)) {
      messages_.Say(expr.source,
          "ARITHMETIC IF statement must not have a COMPLEX expression"_err_en_US);
    } else if (!IsNumericExpr(*expr.typedExpr)) {
      messages_.Say(expr.source,
          "ARITHMETIC IF statement must have a numeric expression"_err_en_US);
    }
    // The labels have already been checked in resolve-labels.
    // TODO: Really?  Check that they are really branch target
    // statements and in the same inclusive scope.
  }

private:
  bool IsNumericExpr(const evaluate::GenericExprWrapper &expr) {
    auto dynamicType{expr.v.GetType()};
    return dynamicType.has_value() &&
        common::IsNumericTypeCategory(dynamicType->category);
  }
  parser::Messages &messages_;
  parser::CharBlock currentStatementSourcePosition_;
};

}  // namespace Fortran::semantics

namespace Fortran::semantics {

ArithmeticIfStmtChecker::ArithmeticIfStmtChecker(SemanticsContext &context)
  : context_{new ArithmeticIfStmtContext{context}} {}

ArithmeticIfStmtChecker::~ArithmeticIfStmtChecker() = default;

void ArithmeticIfStmtChecker::Leave(const parser::ArithmeticIfStmt &x) {
  context_.value().Check(x);
}

}  // namespace Fortran::semantics

template class Fortran::common::Indirection<
    Fortran::semantics::ArithmeticIfStmtContext>;
