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

#include "check-do-concurrent.h"
#include "attr.h"
#include "scope.h"
#include "semantics.h"
#include "symbol.h"
#include "tools.h"
#include "type.h"
#include "../evaluate/traversal.h"
#include "../parser/message.h"
#include "../parser/parse-tree-visitor.h"

namespace Fortran::semantics {

using namespace parser::literals;

static bool isPure(const Attrs &attrs) {
  return attrs.test(Attr::PURE) ||
      (attrs.test(Attr::ELEMENTAL) && !attrs.test(Attr::IMPURE));
}
static bool isProcedure(const Symbol::Flags &flags) {
  return flags.test(Symbol::Flag::Function) ||
      flags.test(Symbol::Flag::Subroutine);
}

// 11.1.7.5 - enforce semantics constraints on a DO CONCURRENT loop body
class DoConcurrentEnforcement {
public:
  DoConcurrentEnforcement(parser::Messages &messages) : messages_{messages} {}
  std::set<parser::Label> labels() { return labels_; }
  std::set<parser::CharBlock> names() { return names_; }
  template<typename T> bool Pre(const T &) { return true; }
  template<typename T> void Post(const T &) {}
  template<typename T> bool Pre(const parser::Statement<T> &statement) {
    currentStatementSourcePosition_ = statement.source;
    if (statement.label.has_value()) {
      labels_.insert(*statement.label);
    }
    return true;
  }
  // C1167
  bool Pre(const parser::WhereConstructStmt &s) {
    addName(std::get<std::optional<parser::Name>>(s.t));
    return true;
  }
  bool Pre(const parser::ForallConstructStmt &s) {
    addName(std::get<std::optional<parser::Name>>(s.t));
    return true;
  }
  bool Pre(const parser::ChangeTeamStmt &s) {
    addName(std::get<std::optional<parser::Name>>(s.t));
    return true;
  }
  bool Pre(const parser::CriticalStmt &s) {
    addName(std::get<std::optional<parser::Name>>(s.t));
    return true;
  }
  bool Pre(const parser::LabelDoStmt &s) {
    addName(std::get<std::optional<parser::Name>>(s.t));
    return true;
  }
  bool Pre(const parser::NonLabelDoStmt &s) {
    addName(std::get<std::optional<parser::Name>>(s.t));
    return true;
  }
  bool Pre(const parser::IfThenStmt &s) {
    addName(std::get<std::optional<parser::Name>>(s.t));
    return true;
  }
  bool Pre(const parser::SelectCaseStmt &s) {
    addName(std::get<std::optional<parser::Name>>(s.t));
    return true;
  }
  bool Pre(const parser::SelectRankStmt &s) {
    addName(std::get<0>(s.t));
    return true;
  }
  bool Pre(const parser::SelectTypeStmt &s) {
    addName(std::get<0>(s.t));
    return true;
  }
  // C1136
  void Post(const parser::ReturnStmt &) {
    messages_.Say(currentStatementSourcePosition_,
        "RETURN not allowed in DO CONCURRENT"_err_en_US);
  }
  // C1137
  void NoImageControl() {
    messages_.Say(currentStatementSourcePosition_,
        "image control statement not allowed in DO CONCURRENT"_err_en_US);
  }
  void Post(const parser::SyncAllStmt &) { NoImageControl(); }
  void Post(const parser::SyncImagesStmt &) { NoImageControl(); }
  void Post(const parser::SyncMemoryStmt &) { NoImageControl(); }
  void Post(const parser::SyncTeamStmt &) { NoImageControl(); }
  void Post(const parser::ChangeTeamConstruct &) { NoImageControl(); }
  void Post(const parser::CriticalConstruct &) { NoImageControl(); }
  void Post(const parser::EventPostStmt &) { NoImageControl(); }
  void Post(const parser::EventWaitStmt &) { NoImageControl(); }
  void Post(const parser::FormTeamStmt &) { NoImageControl(); }
  void Post(const parser::LockStmt &) { NoImageControl(); }
  void Post(const parser::UnlockStmt &) { NoImageControl(); }
  void Post(const parser::StopStmt &) { NoImageControl(); }
  void Post(const parser::EndProgramStmt &) { NoImageControl(); }

  void Post(const parser::AllocateStmt &) {
    if (anyObjectIsCoarray()) {
      messages_.Say(currentStatementSourcePosition_,
          "ALLOCATE coarray not allowed in DO CONCURRENT"_err_en_US);
    }
  }
  void Post(const parser::DeallocateStmt &) {
    if (anyObjectIsCoarray()) {
      messages_.Say(currentStatementSourcePosition_,
          "DEALLOCATE coarray not allowed in DO CONCURRENT"_err_en_US);
    }
    // C1140: deallocation of polymorphic objects
    if (anyObjectIsPolymorphic()) {
      messages_.Say(currentStatementSourcePosition_,
          "DEALLOCATE polymorphic object(s) not allowed"
          " in DO CONCURRENT"_err_en_US);
    }
  }
  template<typename T> void Post(const parser::Statement<T> &) {
    if (EndTDeallocatesCoarray()) {
      messages_.Say(currentStatementSourcePosition_,
          "implicit deallocation of coarray not allowed"
          " in DO CONCURRENT"_err_en_US);
    }
  }
  // C1141: cannot call ieee_get_flag, ieee_[gs]et_halting_mode
  void Post(const parser::ProcedureDesignator &procedureDesignator) {
    if (auto *name{std::get_if<parser::Name>(&procedureDesignator.u)}) {
      // C1137: call move_alloc with coarray arguments
      if (name->source == "move_alloc") {
        if (anyObjectIsCoarray()) {
          messages_.Say(currentStatementSourcePosition_,
              "call to MOVE_ALLOC intrinsic in DO CONCURRENT with coarray"
              " argument(s) not allowed"_err_en_US);
        }
      }
      // C1139: call to impure procedure
      if (name->symbol && !isPure(name->symbol->attrs())) {
        messages_.Say(currentStatementSourcePosition_,
            "call to impure subroutine in DO CONCURRENT not allowed"_err_en_US);
      }
      if (name->symbol && fromScope(*name->symbol, "ieee_exceptions"s)) {
        if (name->source == "ieee_get_flag") {
          messages_.Say(currentStatementSourcePosition_,
              "IEEE_GET_FLAG not allowed in DO CONCURRENT"_err_en_US);
        } else if (name->source == "ieee_set_halting_mode") {
          messages_.Say(currentStatementSourcePosition_,
              "IEEE_SET_HALTING_MODE not allowed in DO CONCURRENT"_err_en_US);
        } else if (name->source == "ieee_get_halting_mode") {
          messages_.Say(currentStatementSourcePosition_,
              "IEEE_GET_HALTING_MODE not allowed in DO CONCURRENT"_err_en_US);
        }
      }
    } else {
      // C1139: this a procedure component
      auto &component{std::get<parser::ProcComponentRef>(procedureDesignator.u)
                          .v.thing.component};
      if (component.symbol && !isPure(component.symbol->attrs())) {
        messages_.Say(currentStatementSourcePosition_,
            "call to impure subroutine in DO CONCURRENT not allowed"_err_en_US);
      }
    }
  }

  // 11.1.7.5
  void Post(const parser::IoControlSpec &ioControlSpec) {
    if (auto *charExpr{
            std::get_if<parser::IoControlSpec::CharExpr>(&ioControlSpec.u)}) {
      if (std::get<parser::IoControlSpec::CharExpr::Kind>(charExpr->t) ==
          parser::IoControlSpec::CharExpr::Kind::Advance) {
        messages_.Say(currentStatementSourcePosition_,
            "ADVANCE specifier not allowed in DO CONCURRENT"_err_en_US);
      }
    }
  }

private:
  bool anyObjectIsCoarray() { return false; }  // FIXME placeholder
  bool anyObjectIsPolymorphic() { return false; }  // FIXME placeholder
  bool EndTDeallocatesCoarray() { return false; }  // FIXME placeholder
  bool fromScope(const Symbol &symbol, const std::string &moduleName) {
    if (symbol.GetUltimate().owner().IsModule() &&
        symbol.GetUltimate().owner().name().ToString() == moduleName) {
      return true;
    }
    return false;
  }
  void addName(const std::optional<parser::Name> &nm) {
    if (nm.has_value()) {
      names_.insert(nm.value().source);
    }
  }

  std::set<parser::CharBlock> names_;
  std::set<parser::Label> labels_;
  parser::CharBlock currentStatementSourcePosition_;
  parser::Messages &messages_;
};

class DoConcurrentLabelEnforce {
public:
  DoConcurrentLabelEnforce(parser::Messages &messages,
      std::set<parser::Label> &&labels, std::set<parser::CharBlock> &&names,
      parser::CharBlock doConcurrentSourcePosition)
    : messages_{messages}, labels_{labels}, names_{names},
      doConcurrentSourcePosition_{doConcurrentSourcePosition} {}
  template<typename T> bool Pre(const T &) { return true; }
  template<typename T> bool Pre(const parser::Statement<T> &statement) {
    currentStatementSourcePosition_ = statement.source;
    return true;
  }
  bool Pre(const parser::DoConstruct &) {
    ++do_depth_;
    return true;
  }
  template<typename T> void Post(const T &) {}

  // C1138: branch from within a DO CONCURRENT shall not target outside loop
  void Post(const parser::ExitStmt &exitStmt) { checkName(exitStmt.v); }
  void Post(const parser::GotoStmt &gotoStmt) { checkLabelUse(gotoStmt.v); }
  void Post(const parser::ComputedGotoStmt &computedGotoStmt) {
    for (auto &i : std::get<std::list<parser::Label>>(computedGotoStmt.t)) {
      checkLabelUse(i);
    }
  }
  void Post(const parser::ArithmeticIfStmt &arithmeticIfStmt) {
    checkLabelUse(std::get<1>(arithmeticIfStmt.t));
    checkLabelUse(std::get<2>(arithmeticIfStmt.t));
    checkLabelUse(std::get<3>(arithmeticIfStmt.t));
  }
  void Post(const parser::AssignStmt &assignStmt) {
    checkLabelUse(std::get<parser::Label>(assignStmt.t));
  }
  void Post(const parser::AssignedGotoStmt &assignedGotoStmt) {
    for (auto &i : std::get<std::list<parser::Label>>(assignedGotoStmt.t)) {
      checkLabelUse(i);
    }
  }
  void Post(const parser::AltReturnSpec &altReturnSpec) {
    checkLabelUse(altReturnSpec.v);
  }
  void Post(const parser::ErrLabel &errLabel) { checkLabelUse(errLabel.v); }
  void Post(const parser::EndLabel &endLabel) { checkLabelUse(endLabel.v); }
  void Post(const parser::EorLabel &eorLabel) { checkLabelUse(eorLabel.v); }
  void Post(const parser::DoConstruct &) { --do_depth_; }
  void checkName(const std::optional<parser::Name> &nm) {
    if (!nm.has_value()) {
      if (do_depth_ == 0) {
        messages_.Say(currentStatementSourcePosition_,
            "exit from DO CONCURRENT construct (%s)"_err_en_US,
            doConcurrentSourcePosition_);
      }
      // nesting of named constructs is assumed to have been previously checked
      // by the name/label resolution pass
    } else if (names_.find(nm.value().source) == names_.end()) {
      messages_.Say(currentStatementSourcePosition_,
          "exit from DO CONCURRENT construct (%s) to construct with name '%s'"_err_en_US,
          doConcurrentSourcePosition_, nm.value().source);
    }
  }
  void checkLabelUse(const parser::Label &labelUsed) {
    if (labels_.find(labelUsed) == labels_.end()) {
      messages_.Say(currentStatementSourcePosition_,
          "control flow escapes from DO CONCURRENT"_err_en_US);
    }
  }

private:
  parser::Messages &messages_;
  std::set<parser::Label> labels_;
  std::set<parser::CharBlock> names_;
  int do_depth_{0};
  parser::CharBlock currentStatementSourcePosition_{nullptr};
  parser::CharBlock doConcurrentSourcePosition_{nullptr};
};

using CS = std::vector<const Symbol *>;

struct GatherSymbols {
  CS symbols;
  template<typename T> constexpr bool Pre(const T &) { return true; }
  template<typename T> constexpr void Post(const T &) {}
  void Post(const parser::Name &name) { symbols.push_back(name.symbol); }
};

enum GatherWhichVariables { All, NotShared, Local };
static CS GatherVariables(const std::list<parser::LocalitySpec> &localitySpecs,
    GatherWhichVariables which) {
  CS symbols;
  for (auto &ls : localitySpecs) {
    auto names{std::visit(
        [=](const auto &x) {
          using T = std::decay_t<decltype(x)>;
          using namespace parser;
          if constexpr (!std::is_same_v<T, LocalitySpec::DefaultNone>) {
            if (which == GatherWhichVariables::All ||
                (which == GatherWhichVariables::NotShared &&
                    !std::is_same_v<T, LocalitySpec::Shared>) ||
                (which == GatherWhichVariables::Local &&
                    std::is_same_v<T, LocalitySpec::Local>)) {
              return x.v;
            }
          }
          return std::list<parser::Name>{};
        },
        ls.u)};
    for (const auto &name : names) {
      if (name.symbol) {
        symbols.push_back(name.symbol);
      }
    }
  }
  return symbols;
}

static CS GatherReferencesFromExpression(const parser::Expr &expression) {
  if (const auto *expr{GetExpr(expression)}) {
    struct CollectSymbols : public virtual evaluate::VisitorBase<CS> {
      using Result = CS;
      explicit CollectSymbols(int) {}
      void Handle(const Symbol *symbol) { result().push_back(symbol); }
    };
    return evaluate::Visitor<CollectSymbols>{0}.Traverse(*expr);
  } else {
    return {};
  }
}

// Find a canonical DO CONCURRENT and enforce semantics checks on its body
class DoConcurrentContext {
public:
  DoConcurrentContext(SemanticsContext &context)
    : messages_{context.messages()} {}

  bool operator==(const DoConcurrentContext &x) const { return this == &x; }

  void Check(const parser::DoConstruct &doConstruct) {
    auto &doStmt{
        std::get<parser::Statement<parser::NonLabelDoStmt>>(doConstruct.t)};
    auto &optionalLoopControl{
        std::get<std::optional<parser::LoopControl>>(doStmt.statement.t)};
    if (optionalLoopControl) {
      currentStatementSourcePosition_ = doStmt.source;
      if (auto *concurrent{std::get_if<parser::LoopControl::Concurrent>(
              &optionalLoopControl->u)}) {
        DoConcurrentEnforcement doConcurrentEnforcement{messages_};
        parser::Walk(
            std::get<parser::Block>(doConstruct.t), doConcurrentEnforcement);
        DoConcurrentLabelEnforce doConcurrentLabelEnforce{messages_,
            doConcurrentEnforcement.labels(), doConcurrentEnforcement.names(),
            currentStatementSourcePosition_};
        parser::Walk(
            std::get<parser::Block>(doConstruct.t), doConcurrentLabelEnforce);
        EnforceConcurrentLoopControl(*concurrent);
      }
    }
  }

private:
  bool InnermostEnclosingScope(const semantics::Symbol &symbol) const {
    // TODO - implement
    return true;
  }
  void CheckZeroOrOneDefaultNone(
      const std::list<parser::LocalitySpec> &localitySpecs) const {
    // C1127
    int count{0};
    for (auto &ls : localitySpecs) {
      if (std::holds_alternative<parser::LocalitySpec::DefaultNone>(ls.u)) {
        ++count;
        if (count > 1) {
          messages_.Say(currentStatementSourcePosition_,
              "only one DEFAULT(NONE) may appear"_err_en_US);
          return;
        }
      }
    }
  }
  void CheckScopingConstraints(const CS &symbols) const {
    // C1124
    for (auto *symbol : symbols) {
      if (!InnermostEnclosingScope(*symbol)) {
        messages_.Say(currentStatementSourcePosition_,
            "variable in locality-spec must be in innermost"
            " scoping unit"_err_en_US);
        return;
      }
    }
  }
  void CheckMaskIsPure(const parser::ScalarLogicalExpr &mask) const {
    // C1121 - procedures in mask must be pure
    // TODO - add the name of the impure procedure to the message
    CS references{GatherReferencesFromExpression(mask.thing.thing.value())};
    for (auto *r : references) {
      if (isProcedure(r->flags()) && !isPure(r->attrs())) {
        messages_.Say(currentStatementSourcePosition_,
            "concurrent-header mask expression cannot reference an impure"
            " procedure"_err_en_US);
        return;
      }
    }
  }
  void CheckNoCollisions(const CS &containerA, const CS &containerB,
      const parser::MessageFormattedText &errorMessage) const {
    for (auto *a : containerA) {
      for (auto *b : containerB) {
        if (a == b) {
          messages_.Say(currentStatementSourcePosition_, errorMessage);
          return;
        }
      }
    }
  }
  void HasNoReferences(
      const CS &indexNames, const parser::ScalarIntExpr &expression) const {
    CS references{
        GatherReferencesFromExpression(expression.thing.thing.value())};
    CheckNoCollisions(references, indexNames,
        "concurrent-control expression references index-name"_err_en_US);
  }
  void CheckMaskDoesNotReferenceLocal(
      const parser::ScalarLogicalExpr &mask, const CS &symbols) const {
    // C1129
    CheckNoCollisions(GatherReferencesFromExpression(mask.thing.thing.value()),
        symbols,
        "concurrent-header mask-expr references name"
        " in locality-spec"_err_en_US);
  }
  void CheckLocalAndLocalInitAttributes(const CS &symbols) const {
    // C1128
    // TODO - implement
  }
  void CheckDefaultNoneImpliesExplicitLocality(
      const std::list<parser::LocalitySpec> &localitySpecs) const {
    // C1130
    // TODO - implement
  }
  // check constraints [C1121 .. C1130]
  void EnforceConcurrentLoopControl(
      const parser::LoopControl::Concurrent &concurrent) const {
    auto &header{std::get<parser::ConcurrentHeader>(concurrent.t)};
    auto &mask{std::get<std::optional<parser::ScalarLogicalExpr>>(header.t)};
    if (mask.has_value()) {
      CheckMaskIsPure(*mask);
    }
    auto &controls{std::get<std::list<parser::ConcurrentControl>>(header.t)};
    CS indexNames;
    for (auto &c : controls) {
      auto &indexName{std::get<parser::Name>(c.t)};
      if (indexName.symbol) {
        indexNames.push_back(indexName.symbol);
      }
    }
    if (!indexNames.empty()) {
      for (auto &c : controls) {
        // C1123
        HasNoReferences(indexNames, std::get<1>(c.t));
        HasNoReferences(indexNames, std::get<2>(c.t));
        if (auto &expression{
                std::get<std::optional<parser::ScalarIntExpr>>(c.t)}) {
          HasNoReferences(indexNames, *expression);
        }
      }
    }
    auto &localitySpecs{
        std::get<std::list<parser::LocalitySpec>>(concurrent.t)};
    if (localitySpecs.empty()) {
      return;
    }
    auto variableNames{
        GatherVariables(localitySpecs, GatherWhichVariables::All)};
    CheckScopingConstraints(variableNames);
    CheckZeroOrOneDefaultNone(localitySpecs);
    CheckLocalAndLocalInitAttributes(
        GatherVariables(localitySpecs, GatherWhichVariables::NotShared));
    if (mask) {
      CheckMaskDoesNotReferenceLocal(
          *mask, GatherVariables(localitySpecs, GatherWhichVariables::Local));
    }
    CheckDefaultNoneImpliesExplicitLocality(localitySpecs);
  }

  parser::Messages &messages_;
  parser::CharBlock currentStatementSourcePosition_;
};

DoConcurrentChecker::DoConcurrentChecker(SemanticsContext &context)
  : context_{new DoConcurrentContext{context}} {}

DoConcurrentChecker::~DoConcurrentChecker() = default;

// DO loops must be canonicalized prior to calling
void DoConcurrentChecker::Leave(const parser::DoConstruct &x) {
  context_.value().Check(x);
}

}  // namespace Fortran::semantics

template class Fortran::common::Indirection<
    Fortran::semantics::DoConcurrentContext>;
