// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

#include "parse-tree.h"
#include "user-state.h"
#include "../common/idioms.h"
#include "../common/indirection.h"
#include <algorithm>

namespace Fortran::parser {

// R867
ImportStmt::ImportStmt(common::ImportKind &&k, std::list<Name> &&n)
  : kind{k}, names(std::move(n)) {
  CHECK(kind == common::ImportKind::Default ||
      kind == common::ImportKind::Only || names.empty());
}

// R901 designator
bool Designator::EndsInBareName() const {
  return std::visit(
      common::visitors{[](const ObjectName &) { return true; },
          [](const DataRef &dr) {
            return std::holds_alternative<Name>(dr.u) ||
                std::holds_alternative<common::Indirection<StructureComponent>>(
                    dr.u);
          },
          [](const Substring &) { return false; }},
      u);
}

// R911 data-ref -> part-ref [% part-ref]...
DataRef::DataRef(std::list<PartRef> &&prl) : u{std::move(prl.front().name)} {
  for (bool first{true}; !prl.empty(); first = false, prl.pop_front()) {
    PartRef &pr{prl.front()};
    if (!first) {
      u = common::Indirection<StructureComponent>::Make(
          std::move(*this), std::move(pr.name));
    }
    if (!pr.subscripts.empty()) {
      u = common::Indirection<ArrayElement>::Make(
          std::move(*this), std::move(pr.subscripts));
    }
    if (pr.imageSelector.has_value()) {
      u = common::Indirection<CoindexedNamedObject>::Make(
          std::move(*this), std::move(*pr.imageSelector));
    }
  }
}

// R1001 - R1022 expression
Expr::Expr(Designator &&x)
  : u{common::Indirection<Designator>::Make(std::move(x))} {}
Expr::Expr(FunctionReference &&x)
  : u{common::Indirection<FunctionReference>::Make(std::move(x))} {}

static Designator MakeArrayElementRef(Name &name, std::list<Expr> &subscripts) {
  ArrayElement arrayElement{name, std::list<SectionSubscript>{}};
  for (Expr &expr : subscripts) {
    arrayElement.subscripts.push_back(SectionSubscript{
        Scalar{Integer{common::Indirection{std::move(expr)}}}});
  }
  return Designator{DataRef{common::Indirection{std::move(arrayElement)}}};
}

Designator FunctionReference::ConvertToArrayElementRef() {
  auto &name{std::get<parser::Name>(std::get<ProcedureDesignator>(v.t).u)};
  std::list<Expr> args;
  for (auto &arg : std::get<std::list<ActualArgSpec>>(v.t)) {
    std::visit(
        common::visitors{
            [&](common::Indirection<Expr> &y) {
              args.push_back(std::move(*y));
            },
            [&](common::Indirection<Variable> &y) {
              args.push_back(std::visit(
                  common::visitors{
                      [&](common::Indirection<Designator> &z) {
                        return Expr{std::move(*z)};
                      },
                      [&](common::Indirection<FunctionReference> &z) {
                        return Expr{std::move(*z)};
                      },
                  },
                  y->u));
            },
            [&](auto &) { CHECK(!"unexpected kind of ActualArg"); },
        },
        std::get<ActualArg>(arg.t).u);
  }
  return MakeArrayElementRef(name, args);
}

// R1544 stmt-function-stmt
// Convert this stmt-function-stmt to an array element assignment statement.
Statement<ActionStmt> StmtFunctionStmt::ConvertToAssignment() {
  auto &funcName{std::get<Name>(t)};
  auto &funcArgs{std::get<std::list<Name>>(t)};
  auto &funcExpr{std::get<Scalar<Expr>>(t).thing};
  std::list<Expr> subscripts;
  for (Name &arg : funcArgs) {
    subscripts.push_back(Expr{common::Indirection{Designator{arg}}});
  }
  auto variable{
      Variable{common::Indirection{MakeArrayElementRef(funcName, subscripts)}}};
  return Statement{std::nullopt,
      ActionStmt{common::Indirection{
          AssignmentStmt{std::move(variable), std::move(funcExpr)}}}};
}

std::ostream &operator<<(std::ostream &os, const Name &x) {
  return os << x.ToString();
}
std::ostream &operator<<(std::ostream &os, const CharBlock &x) {
  return os << x.ToString();
}

}  // namespace Fortran::parser
