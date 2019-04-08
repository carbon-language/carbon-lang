// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

// So "delete Expr;" calls an external destructor for its typedExpr.
namespace Fortran::evaluate {
struct GenericExprWrapper {
  ~GenericExprWrapper();
};
}

namespace Fortran::parser {

// R867
ImportStmt::ImportStmt(common::ImportKind &&k, std::list<Name> &&n)
  : kind{k}, names(std::move(n)) {
  CHECK(kind == common::ImportKind::Default ||
      kind == common::ImportKind::Only || names.empty());
}

// R873
CommonStmt::CommonStmt(std::optional<Name> &&name,
    std::list<CommonBlockObject> &&objects, std::list<Block> &&others) {
  blocks.emplace_front(std::move(name), std::move(objects));
  blocks.splice(blocks.end(), std::move(others));
}

// R901 designator
bool Designator::EndsInBareName() const {
  return std::visit(
      common::visitors{
          [](const ObjectName &) { return true; },
          [](const DataRef &dr) {
            return std::holds_alternative<Name>(dr.u) ||
                std::holds_alternative<common::Indirection<StructureComponent>>(
                    dr.u);
          },
          [](const Substring &) { return false; },
      },
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

bool DoConstruct::IsDoConcurrent() const {
  auto &doStmt{std::get<Statement<NonLabelDoStmt>>(t).statement};
  auto &control{std::get<std::optional<LoopControl>>(doStmt.t)};
  return control && std::holds_alternative<LoopControl::Concurrent>(control->u);
}

static Designator MakeArrayElementRef(
    const Name &name, std::list<Expr> &subscripts) {
  ArrayElement arrayElement{Name{name.source}, std::list<SectionSubscript>{}};
  for (Expr &expr : subscripts) {
    arrayElement.subscripts.push_back(SectionSubscript{
        Scalar{Integer{common::Indirection{std::move(expr)}}}});
  }
  return Designator{DataRef{common::Indirection{std::move(arrayElement)}}};
}

static std::optional<Expr> ActualArgToExpr(
    parser::CharBlock at, ActualArgSpec &arg) {
  return std::visit(
      common::visitors{
          [&](common::Indirection<Expr> &y) {
            return std::make_optional<Expr>(std::move(y.value()));
          },
          [&](common::Indirection<Variable> &y) {
            return std::visit(
                [&](auto &indirection) {
                  std::optional<Expr> result{std::move(indirection.value())};
                  result->source = at;
                  return result;
                },
                y.value().u);
          },
          [&](auto &) -> std::optional<Expr> { return std::nullopt; },
      },
      std::get<ActualArg>(arg.t).u);
}

Designator FunctionReference::ConvertToArrayElementRef() {
  auto &name{std::get<parser::Name>(std::get<ProcedureDesignator>(v.t).u)};
  std::list<Expr> args;
  for (auto &arg : std::get<std::list<ActualArgSpec>>(v.t)) {
    args.emplace_back(std::move(ActualArgToExpr(name.source, arg).value()));
  }
  return MakeArrayElementRef(name, args);
}

StructureConstructor FunctionReference::ConvertToStructureConstructor(
    const semantics::DerivedTypeSpec &derived) {
  Name name{std::get<parser::Name>(std::get<ProcedureDesignator>(v.t).u)};
  std::list<ComponentSpec> components;
  for (auto &arg : std::get<std::list<ActualArgSpec>>(v.t)) {
    std::optional<Keyword> keyword;
    if (auto &kw{std::get<std::optional<Keyword>>(arg.t)}) {
      keyword.emplace(Keyword{Name{kw->v}});
    }
    components.emplace_back(std::move(keyword),
        ComponentDataSource{ActualArgToExpr(name.source, arg).value()});
  }
  DerivedTypeSpec spec{std::move(name), std::list<TypeParamSpec>{}};
  spec.derivedTypeSpec = &derived;
  return StructureConstructor{std::move(spec), std::move(components)};
}

Substring ArrayElement::ConvertToSubstring() {
  auto iter{subscripts.begin()};
  CHECK(iter != subscripts.end());
  auto &triplet{std::get<SubscriptTriplet>(iter->u)};
  SubstringRange range{
      std::move(std::get<0>(triplet.t)), std::move(std::get<1>(triplet.t))};
  CHECK(!std::get<2>(triplet.t).has_value());
  CHECK(++iter == subscripts.end());
  return Substring{std::move(base), std::move(range)};
}

// R1544 stmt-function-stmt
// Convert this stmt-function-stmt to an array element assignment statement.
Statement<ActionStmt> StmtFunctionStmt::ConvertToAssignment() {
  auto &funcName{std::get<Name>(t)};
  auto &funcArgs{std::get<std::list<Name>>(t)};
  auto &funcExpr{std::get<Scalar<Expr>>(t).thing};
  std::list<Expr> subscripts;
  for (Name &arg : funcArgs) {
    subscripts.push_back(
        Expr{common::Indirection{Designator{Name{arg.source}}}});
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
}

template class std::unique_ptr<Fortran::evaluate::GenericExprWrapper>;
