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
#include "tools.h"
#include "../common/idioms.h"
#include "../common/restorer.h"
#include "../evaluate/characteristics.h"
#include "../evaluate/expression.h"
#include "../evaluate/fold.h"
#include "../evaluate/tools.h"
#include "../parser/message.h"
#include "../parser/parse-tree-visitor.h"
#include "../parser/parse-tree.h"
#include <optional>
#include <set>
#include <string>
#include <type_traits>

using namespace Fortran::parser::literals;

namespace Fortran::evaluate {

class PointerAssignmentChecker {
public:
  PointerAssignmentChecker(const Symbol *pointer, parser::CharBlock source,
      const std::string &description, const characteristics::TypeAndShape *type,
      parser::ContextualMessages &messages,
      const IntrinsicProcTable &intrinsics,
      const characteristics::Procedure *procedure, bool isContiguous)
    : pointer_{pointer}, source_{source}, description_{description},
      type_{type}, messages_{messages}, intrinsics_{intrinsics},
      procedure_{procedure}, isContiguous_{isContiguous} {}

  template<typename A> void Check(const A &) {
    // Catch-all case for really bad target expression
    Say("Target associated with %s must be a designator or a call to a pointer-valued function"_err_en_US,
        description_);
  }

  template<typename T> void Check(const Expr<T> &x) {
    std::visit([&](const auto &x) { Check(x); }, x.u);
  }
  void Check(const Expr<SomeType> &);
  void Check(const NullPointer &) {}  // P => NULL() without MOLD=; always OK

  template<typename T> void Check(const FunctionRef<T> &f) {
    std::string funcName;
    const auto *symbol{f.proc().GetSymbol()};
    if (symbol) {
      funcName = symbol->name().ToString();
    } else if (const auto *intrinsic{f.proc().GetSpecificIntrinsic()}) {
      funcName = intrinsic->name;
    }
    if (auto proc{
            characteristics::Procedure::Characterize(f.proc(), intrinsics_)}) {
      std::optional<parser::MessageFixedText> error;
      if (const auto &funcResult{proc->functionResult}) {  // C1025
        const auto *frProc{funcResult->IsProcedurePointer()};
        if (procedure_) {
          // Shouldn't be here in this function unless lhs
          // is an object pointer.
          error =
              "Procedure %s is associated with the result of a reference to function '%s' that does not return a procedure pointer"_err_en_US;
        } else if (frProc) {
          error =
              "Object %s is associated with the result of a reference to function '%s' that is a procedure pointer"_err_en_US;
        } else if (!funcResult->attrs.test(
                       characteristics::FunctionResult::Attr::Pointer)) {
          error =
              "%s is associated with the result of a reference to function '%s' that is a not a pointer"_err_en_US;
        } else if (isContiguous_ &&
            !funcResult->attrs.test(
                characteristics::FunctionResult::Attr::Contiguous)) {
          error =
              "CONTIGUOUS %s is associated with the result of reference to function '%s' that is not contiguous"_err_en_US;
        } else if (type_) {
          const auto *frTypeAndShape{funcResult->GetTypeAndShape()};
          CHECK(frTypeAndShape);
          if (!type_->IsCompatibleWith(messages_, *frTypeAndShape)) {
            error =
                "%s is associated with the result of a reference to function '%s' whose pointer result has an incompatible type or shape"_err_en_US;
          }
        }
      } else {
        error =
            "%s is associated with the non-existent result of reference to procedure"_err_en_US;
      }
      if (error) {
        auto save{common::ScopedSet(pointer_, symbol)};
        Say(*error, description_, funcName);
      }
    }
  }

  template<typename T> void Check(const Designator<T> &d) {
    const Symbol *last{d.GetLastSymbol()};
    const Symbol *base{d.GetBaseObject().symbol()};
    if (last && base) {
      std::optional<parser::MessageFixedText> error;
      if (procedure_) {
        // Shouldn't be here in this function unless lhs is an
        // object pointer.
        error =
            "In assignment to procedure %s, the target is not a procedure or procedure pointer"_err_en_US;
      } else if (!GetLastTarget(GetSymbolVector(d))) {  // C1025
        error =
            "In assignment to object %s, the target '%s' is not an object with POINTER or TARGET attributes"_err_en_US;
      } else if (auto rhsTypeAndShape{
                     characteristics::TypeAndShape::Characterize(*last)}) {
        if (!type_ || !type_->IsCompatibleWith(messages_, *rhsTypeAndShape)) {
          error =
              "%s associated with object '%s' with incompatible type or shape"_err_en_US;
        }
      }
      if (error) {
        auto save{common::ScopedSet(pointer_, last)};
        Say(*error, description_, last->name());
      }
    } else {
      // P => "character literal"(1:3)
      messages_.Say("Pointer target is not a named entity"_err_en_US);
    }
  }

  void Check(const ProcedureDesignator &);
  void Check(const ProcedureRef &);

private:
  // Target is a procedure
  void Check(parser::CharBlock rhsName, bool isCall,
      const characteristics::Procedure * = nullptr);

  template<typename... A> parser::Message *Say(A &&... x) {
    auto *msg{messages_.Say(std::forward<A>(x)...)};
    if (pointer_) {
      return AttachDeclaration(msg, pointer_);
    } else if (!source_.empty()) {
      msg->Attach(source_, "Declaration of %s"_en_US, description_);
    }
    return msg;
  }

  const Symbol *pointer_{nullptr};
  const parser::CharBlock source_;
  const std::string &description_;
  const characteristics::TypeAndShape *type_{nullptr};
  parser::ContextualMessages &messages_;
  const IntrinsicProcTable &intrinsics_;
  const characteristics::Procedure *procedure_{nullptr};
  bool isContiguous_{false};
};

void PointerAssignmentChecker::Check(const Expr<SomeType> &rhs) {
  if (HasVectorSubscript(rhs)) {  // C1025
    Say("An array section with a vector subscript may not be a pointer target"_err_en_US);
  } else if (ExtractCoarrayRef(rhs)) {  // C1026
    Say("A coindexed object may not be a pointer target"_err_en_US);
  } else {
    std::visit([&](const auto &x) { Check(x); }, rhs.u);
  }
}

// Common handling for procedure pointer right-hand sides
void PointerAssignmentChecker::Check(parser::CharBlock rhsName, bool isCall,
    const characteristics::Procedure *targetChars) {
  if (procedure_) {
    if (targetChars) {
      if (*procedure_ != *targetChars) {
        if (isCall) {
          Say("Procedure %s associated with result of reference to function '%s' that is an incompatible procedure pointer"_err_en_US,
              description_, rhsName);
        } else {
          Say("Procedure %s associated with incompatible procedure designator '%s'"_err_en_US,
              description_, rhsName);
        }
      }
    } else {
      Say("In assignment to procedure %s, the characteristics of the target procedure '%s' could not be determined"_err_en_US,
          description_, rhsName);
    }
  } else {
    Say("In assignment to object %s, the target '%s' is a procedure designator"_err_en_US,
        description_, rhsName);
  }
}

void PointerAssignmentChecker::Check(const ProcedureDesignator &d) {
  if (auto chars{characteristics::Procedure::Characterize(d, intrinsics_)}) {
    Check(d.GetName(), false, &*chars);
  } else {
    Check(d.GetName(), false);
  }
}

void PointerAssignmentChecker::Check(const ProcedureRef &ref) {
  const characteristics::Procedure *procedure{nullptr};
  auto chars{characteristics::Procedure::Characterize(ref, intrinsics_)};
  if (chars) {
    procedure = &*chars;
    if (chars->functionResult) {
      if (const auto *proc{chars->functionResult->IsProcedurePointer()}) {
        procedure = proc;
      }
    }
  }
  Check(ref.proc().GetName(), true, procedure);
}

void CheckPointerAssignment(parser::ContextualMessages &messages,
    const IntrinsicProcTable &intrinsics, const Symbol &lhs,
    const evaluate::Expr<evaluate::SomeType> &rhs) {
  // TODO: Acquire values of deferred type parameters &/or array bounds
  // from the RHS.
  if (!IsPointer(lhs)) {
    SayWithDeclaration(
        messages, &lhs, "'%s' is not a pointer"_err_en_US, lhs.name());
  } else {
    auto type{characteristics::TypeAndShape::Characterize(lhs)};
    auto proc{characteristics::Procedure::Characterize(lhs, intrinsics)};
    std::string description{"pointer '"s + lhs.name().ToString() + '\''};
    PointerAssignmentChecker{&lhs, lhs.name(), description,
        type ? &*type : nullptr, messages, intrinsics, proc ? &*proc : nullptr,
        lhs.attrs().test(semantics::Attr::CONTIGUOUS)}
        .Check(rhs);
  }
}

void CheckPointerAssignment(parser::ContextualMessages &messages,
    const IntrinsicProcTable &intrinsics, parser::CharBlock source,
    const std::string &description, const characteristics::DummyDataObject &lhs,
    const evaluate::Expr<evaluate::SomeType> &rhs) {
  PointerAssignmentChecker{nullptr, source, description, &lhs.type, messages,
      intrinsics, nullptr /* proc */,
      lhs.attrs.test(characteristics::DummyDataObject::Attr::Contiguous)}
      .Check(rhs);
}
}

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

  std::optional<int> GetActiveIntKind(const parser::CharBlock &name) const {
    const auto iter{activeNames.find(name)};
    if (iter != activeNames.cend()) {
      return {integerKind};
    } else if (outer) {
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
  const ForallContext *forall{nullptr};  // innermost enclosing FORALL
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

  void Analyze(const parser::AssignmentStmt &);
  void Analyze(const parser::PointerAssignmentStmt &);
  void Analyze(const parser::WhereStmt &);
  void Analyze(const parser::WhereConstruct &);
  void Analyze(const parser::ForallStmt &);
  void Analyze(const parser::ForallConstruct &);
  void Analyze(const parser::ConcurrentHeader &);

  template<typename A> void Analyze(const parser::Statement<A> &stmt) {
    std::optional<parser::CharBlock> saveLocation{context_.location()};
    context_.set_location(stmt.source);
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

  template<typename... A> parser::Message *Say(A &&... args) {
    return messages_.Say(std::forward<A>(args)...);
  }

  SemanticsContext &context_;
  parser::ContextualMessages messages_;
  WhereContext *where_{nullptr};
  ForallContext *forall_{nullptr};
};

void AssignmentContext::Analyze(const parser::AssignmentStmt &) {
  if (forall_) {
    // TODO: Warn if some name in forall_->activeNames or its outer
    // contexts does not appear on LHS
  }
  // TODO: Fortran 2003 ALLOCATABLE assignment semantics (automatic
  // (re)allocation of LHS array when unallocated or nonconformable)
}

void AssignmentContext::Analyze(const parser::PointerAssignmentStmt &) {
  CHECK(!where_);
  if (forall_) {
    // TODO: Warn if some name in forall_->activeNames or its outer
    // contexts does not appear on LHS
  }
  // TODO continue here, using CheckPointerAssignment()
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
  context_.set_location(forallStmt.source);
  nested.Analyze(std::get<common::Indirection<parser::ConcurrentHeader>>(
      forallStmt.statement.t));
  for (const auto &body :
      std::get<std::list<parser::ForallBodyConstruct>>(construct.t)) {
    nested.Analyze(body.u);
  }
}

void AssignmentContext::Analyze(
    const parser::WhereConstruct::MaskedElsewhere &elsewhere) {
  CHECK(where_);
  const auto &elsewhereStmt{
      std::get<parser::Statement<parser::MaskedElsewhereStmt>>(elsewhere.t)};
  context_.set_location(elsewhereStmt.source);
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
  if (where_->outer &&
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
  MaskExpr copyCumulative{DEREF(where_).cumulativeMaskExpr};
  where_->thisMaskExpr = evaluate::LogicalNegation(std::move(copyCumulative));
  for (const auto &x :
      std::get<std::list<parser::WhereBodyConstruct>>(elsewhere.t)) {
    Analyze(x);
  }
}

void AssignmentContext::Analyze(const parser::ConcurrentHeader &header) {
  DEREF(forall_).integerKind = GetIntegerKind(
      std::get<std::optional<parser::IntegerTypeSpec>>(header.t));
  for (const auto &control :
      std::get<std::list<parser::ConcurrentControl>>(header.t)) {
    const parser::Name &name{std::get<parser::Name>(control.t)};
    bool inserted{forall_->activeNames.insert(name.source).second};
    CHECK(inserted || context_.HasError(name));
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
    return context_.GetDefaultKind(TypeCategory::Integer);
  }
}

MaskExpr AssignmentContext::GetMask(
    const parser::LogicalExpr &expr, bool defaultValue) const {
  MaskExpr mask{defaultValue};
  if (auto maybeExpr{AnalyzeExpr(context_, expr)}) {
    auto *logical{
        std::get_if<evaluate::Expr<evaluate::SomeLogical>>(&maybeExpr->u)};
    CHECK(logical);
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
}
template class Fortran::common::Indirection<
    Fortran::semantics::AssignmentContext>;
