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
#include "check-if-stmt.h"
#include "scope.h"
#include "semantics.h"
#include "symbol.h"
#include "tools.h"
#include "type.h"
#include "../evaluate/traversal.h"
#include "../parser/message.h"
#include "../parser/parse-tree.h"

namespace Fortran::semantics {

class IfStmtContext {
public:
  IfStmtContext(SemanticsContext &context) : messages_{context.messages()} {}

  bool operator==(const IfStmtContext &x) const { return this == &x; }
  bool operator!=(const IfStmtContext &x) const { return this != &x; }

  // TODO: remove after fixing the issues that gives rise to the warning
  template<class T> void suppress_unused_variable_warning(const T &) {}

  void Check(const parser::IfStmt &ifStmt) {
    // R1139 Check for a scalar logical expression
    auto &expr{
        std::get<parser::ScalarLogicalExpr>(ifStmt.t).thing.thing.value()};
    CheckScalarLogicalExpr(expr, messages_);
    // C1143 Check that the action stmt is not an if stmt
    auto &actionStmt{std::get<parser::ActionStmt>(ifStmt.t)};
    if (auto *actionIfStmt{
            std::get_if<common::Indirection<parser::IfStmt>>(&actionStmt.u)}) {
      // TODO: get the source position from the action stmt
      suppress_unused_variable_warning(actionIfStmt);
      messages_.Say(expr.source,
          "IF statement is not allowed"_err_en_US);
    }
  }

private:
  parser::Messages &messages_;
  parser::CharBlock currentStatementSourcePosition_;
};

}  // namespace Fortran::semantics

namespace Fortran::semantics {

IfStmtChecker::IfStmtChecker(SemanticsContext &context)
  : context_{new IfStmtContext{context}} {}

IfStmtChecker::~IfStmtChecker() = default;

void IfStmtChecker::Leave(const parser::IfStmt &x) {
  context_.value().Check(x);
}

}  // namespace Fortran::semantics

template class Fortran::common::Indirection<Fortran::semantics::IfStmtContext>;
