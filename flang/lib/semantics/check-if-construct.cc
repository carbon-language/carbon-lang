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

#include "check-if-construct.h"
#include "tools.h"
#include "../parser/message.h"
#include "../parser/parse-tree.h"

namespace Fortran::semantics {

void IfConstructChecker::Leave(const parser::IfConstruct &ifConstruct) {
  auto &ifThenStmt{
      std::get<parser::Statement<parser::IfThenStmt>>(ifConstruct.t).statement};
  auto &ifThenExpr{
      std::get<parser::ScalarLogicalExpr>(ifThenStmt.t).thing.thing.value()};
  // R1135 - IF scalar logical expr
  CheckScalarLogicalExpr(ifThenExpr, context_.messages());
  for (const auto &elseIfBlock :
      std::get<std::list<parser::IfConstruct::ElseIfBlock>>(ifConstruct.t)) {
    auto &elseIfStmt{
        std::get<parser::Statement<parser::ElseIfStmt>>(elseIfBlock.t)
            .statement};
    auto &elseIfExpr{
        std::get<parser::ScalarLogicalExpr>(elseIfStmt.t).thing.thing.value()};
    // R1136 - ELSE IF scalar logical expr
    CheckScalarLogicalExpr(elseIfExpr, context_.messages());
  }
  // R1137 The (optional) ELSE does not have an expression to check; ignore it.
}

}  // namespace Fortran::semantics
