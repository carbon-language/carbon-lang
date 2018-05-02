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
#include "idioms.h"
#include "indirection.h"
#include "user-state.h"
#include <algorithm>

namespace Fortran::parser {

// R867
ImportStmt::ImportStmt(Kind &&k, std::list<Name> &&n)
  : kind{k}, names(std::move(n)) {
  CHECK(kind == Kind::Default || kind == Kind::Only || names.empty());
}

// R901 designator
bool Designator::EndsInBareName() const {
  return std::visit(
      visitors{[](const ObjectName &) { return true; },
          [](const DataRef &dr) {
            return std::holds_alternative<Name>(dr.u) ||
                std::holds_alternative<Indirection<StructureComponent>>(dr.u);
          },
          [](const Substring &) { return false; }},
      u);
}

// R911 data-ref -> part-ref [% part-ref]...
DataRef::DataRef(std::list<PartRef> &&prl) : u{std::move(prl.front().name)} {
  for (bool first{true}; !prl.empty(); first = false, prl.pop_front()) {
    PartRef &pr{prl.front()};
    if (!first) {
      u = Indirection<StructureComponent>{std::move(*this), std::move(pr.name)};
    }
    if (!pr.subscripts.empty()) {
      u = Indirection<ArrayElement>{std::move(*this), std::move(pr.subscripts)};
    }
    if (pr.imageSelector.has_value()) {
      u = Indirection<CoindexedNamedObject>{
          std::move(*this), std::move(*pr.imageSelector)};
    }
  }
}

// R1001 - R1022 expression
Expr::Expr(Designator &&x) : u{Indirection<Designator>(std::move(x))} {}
Expr::Expr(FunctionReference &&x)
  : u{Indirection<FunctionReference>(std::move(x))} {}

static Designator MakeArrayElementRef(Name &name, std::list<Expr> &subscripts) {
  ArrayElement arrayElement{name, std::list<SectionSubscript>{}};
  for (Expr &expr : subscripts) {
    arrayElement.subscripts.push_back(
        SectionSubscript{Scalar{Integer{Indirection{std::move(expr)}}}});
  }
  return Designator{DataRef{Indirection{std::move(arrayElement)}}};
}

Designator FunctionReference::ConvertToArrayElementRef() {
  auto &name = std::get<parser::Name>(std::get<ProcedureDesignator>(v.t).u);
  std::list<Expr> args;
  for (auto &arg : std::get<std::list<ActualArgSpec>>(v.t)) {
    std::visit(
        visitors{
            [&](Indirection<Expr> &y) { args.push_back(std::move(*y)); },
            [&](Indirection<Variable> &y) {
              args.push_back(std::visit(
                  visitors{
                      [&](Indirection<Designator> &z) {
                        return Expr{std::move(*z)};
                      },
                      [&](Indirection<FunctionReference> &z) {
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
  auto &funcName = std::get<Name>(t);
  auto &funcArgs = std::get<std::list<Name>>(t);
  auto &funcExpr = std::get<Scalar<Expr>>(t).thing;
  std::list<Expr> subscripts;
  for (Name &arg : funcArgs) {
    subscripts.push_back(Expr{Indirection{Designator{arg}}});
  }
  auto &&variable =
      Variable{Indirection{MakeArrayElementRef(funcName, subscripts)}};
  return Statement{std::nullopt,
      ActionStmt{Indirection{
          AssignmentStmt{std::move(variable), std::move(funcExpr)}}}};
}

}  // namespace Fortran::parser
