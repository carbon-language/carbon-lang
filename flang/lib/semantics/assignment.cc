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

#include "assignment.h"
#include "expression.h"
#include "symbol.h"
#include "../common/idioms.h"
#include "../evaluate/expression.h"
#include "../evaluate/fold.h"
#include "../evaluate/tools.h"
#include "../parser/message.h"
#include "../parser/parse-tree-visitor.h"
#include "../parser/parse-tree.h"
#include <optional>
#include <set>

using namespace Fortran::parser::literals;

namespace Fortran::semantics {

using ControlExpr = evaluate::Expr<evaluate::SubscriptInteger>;
using MaskExpr = evaluate::Expr<evaluate::LogicalResult>;

// The context tracks some number of active FORALL statements/constructs
// and some number of active WHERE statements/constructs.  WHERE can nest
// in FORALL but not vice versa.  Pointer assignments are allowed in
// FORALL but not in WHERE.  These constraints are manifest in the grammar
// and don't need to be rechecked here, since they cannot appear in the
// parse tree.
struct Control {
  Symbol *name;
  ControlExpr lower, upper, step;
};

struct ForallContext {
  explicit ForallContext(const ForallContext *that) : outer{that} {}

  // TODO pmk: Is this needed?  Does semantics already track these kinds?
  std::optional<int> GetActiveIntKind(const parser::CharBlock &name) const {
    const auto iter{activeNames.find(name)};
    if (iter != activeNames.cend()) {
      return {integerKind};
    } else if (outer != nullptr) {
      return outer->GetActiveIntKind(name);
    } else {
      return std::nullopt;
    }
  }

  const ForallContext *outer{nullptr};
  std::optional<parser::CharBlock> constructName;
  int integerKind;
  std::vector<Control> control;
  std::optional<MaskExpr> maskExpr;
  std::set<parser::CharBlock> activeNames;
};

struct WhereContext {
  explicit WhereContext(MaskExpr &&x) : thisMaskExpr{std::move(x)} {}

  const WhereContext *outer{nullptr};
  const ForallContext *forall{nullptr};  // innermost FORALL
  std::optional<parser::CharBlock> constructName;
  MaskExpr thisMaskExpr;  // independent of outer WHERE, if any
  MaskExpr cumulativeMaskExpr{thisMaskExpr};
};

class AssignmentContext {
public:
  explicit AssignmentContext(
      SemanticsContext &c, parser::CharBlock at = parser::CharBlock{})
    : context_{c}, messages_{at, &c.messages()} {}
  AssignmentContext(const AssignmentContext &c, WhereContext &w)
    : context_{c.context_}, messages_{c.messages_}, where_{&w} {}
  AssignmentContext(const AssignmentContext &c, ForallContext &f)
    : context_{c.context_}, messages_{c.messages_}, forall_{&f} {}

  bool operator==(const AssignmentContext &x) const { return this == &x; }
  bool operator!=(const AssignmentContext &x) const { return this != &x; }

  void Analyze(const parser::AssignmentStmt &);
  void Analyze(const parser::PointerAssignmentStmt &);
  void Analyze(const parser::WhereStmt &);
  void Analyze(const parser::WhereConstruct &);
  void Analyze(const parser::ForallStmt &);
  void Analyze(const parser::ForallConstruct &);
  void Analyze(const parser::ConcurrentHeader &);

  template<typename A> void Analyze(const parser::Statement<A> &stmt) {
    const auto *saveLocation{context_.location()};
    context_.set_location(&stmt.source);
    Analyze(stmt.statement);
    context_.set_location(saveLocation);
  }
  template<typename A> void Analyze(const common::Indirection<A> &x) {
    Analyze(x.value());
  }
  template<typename... As> void Analyze(const std::variant<As...> &u) {
    std::visit([&](const auto &x) { Analyze(x); }, u);
  }

private:
  void Analyze(const parser::WhereBodyConstruct &constr) { Analyze(constr.u); }
  void Analyze(const parser::WhereConstruct::MaskedElsewhere &);
  void Analyze(const parser::WhereConstruct::Elsewhere &);
  void Analyze(const parser::ForallAssignmentStmt &stmt) { Analyze(stmt.u); }

  int GetIntegerKind(const std::optional<parser::IntegerTypeSpec> &);

  MaskExpr GetMask(const parser::LogicalExpr &, bool defaultValue = true) const;

  template<typename... A> parser::Message *Say(A... args) {
    return messages_.Say(args...);
  }

  SemanticsContext &context_;
  parser::ContextualMessages messages_;
  WhereContext *where_{nullptr};
  ForallContext *forall_{nullptr};
};

}  // namespace Fortran::semantics

template class Fortran::common::Indirection<
    Fortran::semantics::AssignmentContext>;

namespace Fortran::semantics {

void AssignmentContext::Analyze(const parser::AssignmentStmt &stmt) {
  if (forall_ != nullptr) {
    // TODO: Warn if some name in forall_->activeNames or its outer
    // contexts does not appear on LHS
  }
  // TODO: Fortran 2003 ALLOCATABLE assignment semantics (automatic
  // (re)allocation of LHS array when unallocated or nonconformable)
}

void AssignmentContext::Analyze(const parser::PointerAssignmentStmt &stmt) {
  CHECK(!where_);
  if (forall_ != nullptr) {
    // TODO: Warn if some name in forall_->activeNames or its outer
    // contexts does not appear on LHS
  }
}

void AssignmentContext::Analyze(const parser::WhereStmt &stmt) {
  WhereContext where{GetMask(std::get<parser::LogicalExpr>(stmt.t))};
  AssignmentContext nested{*this, where};
  nested.Analyze(std::get<parser::AssignmentStmt>(stmt.t));
}

// N.B. Construct name matching is checked during label resolution.
void AssignmentContext::Analyze(const parser::WhereConstruct &construct) {
  const auto &whereStmt{
      std::get<parser::Statement<parser::WhereConstructStmt>>(construct.t)};
  WhereContext where{
      GetMask(std::get<parser::LogicalExpr>(whereStmt.statement.t))};
  if (const auto &name{
          std::get<std::optional<parser::Name>>(whereStmt.statement.t)}) {
    where.constructName = name->source;
  }
  AssignmentContext nested{*this, where};
  for (const auto &x :
      std::get<std::list<parser::WhereBodyConstruct>>(construct.t)) {
    nested.Analyze(x);
  }
  for (const auto &x :
      std::get<std::list<parser::WhereConstruct::MaskedElsewhere>>(
          construct.t)) {
    nested.Analyze(x);
  }
  if (const auto &x{std::get<std::optional<parser::WhereConstruct::Elsewhere>>(
          construct.t)}) {
    nested.Analyze(*x);
  }
}

void AssignmentContext::Analyze(const parser::ForallStmt &stmt) {
  CHECK(!where_);
  ForallContext forall{forall_};
  AssignmentContext nested{*this, forall};
  nested.Analyze(
      std::get<common::Indirection<parser::ConcurrentHeader>>(stmt.t));
  const auto &assign{
      std::get<parser::UnlabeledStatement<parser::ForallAssignmentStmt>>(
          stmt.t)};
  auto restorer{nested.messages_.SetLocation(assign.source)};
  nested.Analyze(assign.statement);
}

// N.B. Construct name matching is checked during label resolution;
// index name distinction is checked during name resolution.
void AssignmentContext::Analyze(const parser::ForallConstruct &construct) {
  CHECK(!where_);
  ForallContext forall{forall_};
  AssignmentContext nested{*this, forall};
  const auto &forallStmt{
      std::get<parser::Statement<parser::ForallConstructStmt>>(construct.t)};
  context_.set_location(&forallStmt.source);
  nested.Analyze(std::get<common::Indirection<parser::ConcurrentHeader>>(
      forallStmt.statement.t));
  for (const auto &body :
      std::get<std::list<parser::ForallBodyConstruct>>(construct.t)) {
    nested.Analyze(body.u);
  }
}

void AssignmentContext::Analyze(
    const parser::WhereConstruct::MaskedElsewhere &elsewhere) {
  CHECK(where_ != nullptr);
  const auto &elsewhereStmt{
      std::get<parser::Statement<parser::MaskedElsewhereStmt>>(elsewhere.t)};
  context_.set_location(&elsewhereStmt.source);
  MaskExpr mask{
      GetMask(std::get<parser::LogicalExpr>(elsewhereStmt.statement.t))};
  MaskExpr copyCumulative{where_->cumulativeMaskExpr};
  MaskExpr notOldMask{evaluate::LogicalNegation(std::move(copyCumulative))};
  if (!evaluate::AreConformable(notOldMask, mask)) {
    Say(elsewhereStmt.source,
        "mask of ELSEWHERE statement is not conformable with "
        "the prior mask(s) in its WHERE construct"_err_en_US);
  }
  MaskExpr copyMask{mask};
  where_->cumulativeMaskExpr =
      evaluate::BinaryLogicalOperation(evaluate::LogicalOperator::Or,
          std::move(where_->cumulativeMaskExpr), std::move(copyMask));
  where_->thisMaskExpr = evaluate::BinaryLogicalOperation(
      evaluate::LogicalOperator::And, std::move(notOldMask), std::move(mask));
  if (where_->outer != nullptr &&
      !evaluate::AreConformable(
          where_->outer->thisMaskExpr, where_->thisMaskExpr)) {
    Say(elsewhereStmt.source,
        "effective mask of ELSEWHERE statement is not conformable "
        "with the mask of the surrounding WHERE construct"_err_en_US);
  }
  for (const auto &x :
      std::get<std::list<parser::WhereBodyConstruct>>(elsewhere.t)) {
    Analyze(x);
  }
}

void AssignmentContext::Analyze(
    const parser::WhereConstruct::Elsewhere &elsewhere) {
  CHECK(where_ != nullptr);
  MaskExpr copyCumulative{where_->cumulativeMaskExpr};
  where_->thisMaskExpr = evaluate::LogicalNegation(std::move(copyCumulative));
  for (const auto &x :
      std::get<std::list<parser::WhereBodyConstruct>>(elsewhere.t)) {
    Analyze(x);
  }
}

void AssignmentContext::Analyze(const parser::ConcurrentHeader &header) {
  CHECK(forall_ != nullptr);
  forall_->integerKind = GetIntegerKind(
      std::get<std::optional<parser::IntegerTypeSpec>>(header.t));
  for (const auto &control :
      std::get<std::list<parser::ConcurrentControl>>(header.t)) {
    const parser::CharBlock &name{std::get<parser::Name>(control.t).source};
    bool inserted{forall_->activeNames.insert(name).second};
    CHECK(inserted);
  }
}

int AssignmentContext::GetIntegerKind(
    const std::optional<parser::IntegerTypeSpec> &spec) {
  std::optional<parser::KindSelector> empty;
  evaluate::Expr<evaluate::SubscriptInteger> kind{AnalyzeKindSelector(
      context_, TypeCategory::Integer, spec ? spec->v : empty)};
  if (auto value{evaluate::ToInt64(kind)}) {
    return static_cast<int>(*value);
  } else {
    Say("Kind of INTEGER type must be a constant value"_err_en_US);
    return context_.defaultKinds().GetDefaultKind(TypeCategory::Integer);
  }
}

MaskExpr AssignmentContext::GetMask(
    const parser::LogicalExpr &expr, bool defaultValue) const {
  MaskExpr mask{defaultValue};
  if (auto maybeExpr{AnalyzeExpr(context_, expr)}) {
    auto *logical{
        std::get_if<evaluate::Expr<evaluate::SomeLogical>>(&maybeExpr->u)};
    CHECK(logical != nullptr);
    mask = evaluate::ConvertTo(mask, std::move(*logical));
  }
  return mask;
}

void AnalyzeConcurrentHeader(
    SemanticsContext &context, const parser::ConcurrentHeader &header) {
  AssignmentContext{context}.Analyze(header);
}

AssignmentChecker::~AssignmentChecker() = default;

AssignmentChecker::AssignmentChecker(SemanticsContext &context)
  : context_{new AssignmentContext{context}} {}
void AssignmentChecker::Enter(const parser::AssignmentStmt &x) {
  context_.value().Analyze(x);
}
void AssignmentChecker::Enter(const parser::PointerAssignmentStmt &x) {
  context_.value().Analyze(x);
}
void AssignmentChecker::Enter(const parser::WhereStmt &x) {
  context_.value().Analyze(x);
}
void AssignmentChecker::Enter(const parser::WhereConstruct &x) {
  context_.value().Analyze(x);
}
void AssignmentChecker::Enter(const parser::ForallStmt &x) {
  context_.value().Analyze(x);
}
void AssignmentChecker::Enter(const parser::ForallConstruct &x) {
  context_.value().Analyze(x);
}

namespace {
class Visitor {
public:
  Visitor(SemanticsContext &context) : context_{context} {}

  template<typename A> bool Pre(const A &) { return true /* visit children */; }
  template<typename A> void Post(const A &) {}

  bool Pre(const parser::Statement<parser::AssignmentStmt> &stmt) {
    AssignmentContext{context_, stmt.source}.Analyze(stmt.statement);
    return false;
  }
  bool Pre(const parser::Statement<parser::PointerAssignmentStmt> &stmt) {
    AssignmentContext{context_, stmt.source}.Analyze(stmt.statement);
    return false;
  }
  bool Pre(const parser::Statement<parser::WhereStmt> &stmt) {
    AssignmentContext{context_, stmt.source}.Analyze(stmt.statement);
    return false;
  }
  bool Pre(const parser::WhereConstruct &construct) {
    AssignmentContext{context_}.Analyze(construct);
    return false;
  }
  bool Pre(const parser::Statement<parser::ForallStmt> &stmt) {
    AssignmentContext{context_, stmt.source}.Analyze(stmt.statement);
    return false;
  }
  bool Pre(const parser::ForallConstruct &construct) {
    AssignmentContext{context_}.Analyze(construct);
    return false;
  }

private:
  SemanticsContext &context_;
};

}

}  // namespace Fortran::semantics
