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

#include "resolve-labels.h"
#include "../common/enum-set.h"
#include "../parser/message.h"
#include "../parser/parse-tree-visitor.h"
#include <cctype>
#include <cstdarg>
#include <iostream>

namespace Fortran::semantics {

using namespace parser::literals;

ENUM_CLASS(TargetStatementEnum, Do, Branch, Format)
using LabeledStmtClassificationSet =
    common::EnumSet<TargetStatementEnum, TargetStatementEnum_enumSize>;

using IndexList = std::vector<std::pair<parser::CharBlock, parser::CharBlock>>;
// A ProxyForScope is an integral proxy for a Fortran scope. This is required
// because the parse tree does not actually have the scopes required.
using ProxyForScope = unsigned;
struct LabeledStatementInfoTuplePOD {
  LabeledStatementInfoTuplePOD(const ProxyForScope &_proxyForScope,
      const parser::CharBlock &_parserCharBlock,
      const LabeledStmtClassificationSet &_labeledStmtClassificationSet)
    : proxyForScope{_proxyForScope}, parserCharBlock{_parserCharBlock},
      labeledStmtClassificationSet{_labeledStmtClassificationSet} {}
  ProxyForScope proxyForScope;
  parser::CharBlock parserCharBlock;
  LabeledStmtClassificationSet labeledStmtClassificationSet;
};
using TargetStmtMap = std::map<parser::Label, LabeledStatementInfoTuplePOD>;
struct SourceStatementInfoTuplePOD {
  SourceStatementInfoTuplePOD(const parser::Label &_parserLabel,
      const ProxyForScope &_proxyForScope,
      const parser::CharBlock &_parserCharBlock)
    : parserLabel{_parserLabel}, proxyForScope{_proxyForScope},
      parserCharBlock{_parserCharBlock} {}
  parser::Label parserLabel;
  ProxyForScope proxyForScope;
  parser::CharBlock parserCharBlock;
};
using SourceStmtList = std::vector<SourceStatementInfoTuplePOD>;

const bool isStrictF18{false};  // FIXME - make a command-line option

bool HasScope(ProxyForScope scope) {
  if (scope != ProxyForScope{0u}) {
    return true;
  } else {
    return false;
  }
}

// F18:R1131
template<typename A>
constexpr bool IsLegalDoTerm(const parser::Statement<A> &) {
  if (std::is_same_v<A, common::Indirection<parser::EndDoStmt>> ||
      std::is_same_v<A, parser::EndDoStmt>) {
    return true;
  } else {
    return false;
  }
}
template<>
constexpr bool IsLegalDoTerm(
    const parser::Statement<parser::ActionStmt> &actionStmt) {
  if (std::holds_alternative<parser::ContinueStmt>(actionStmt.statement.u)) {
    // See F08:C816
    return true;
  } else if (isStrictF18) {
    return false;
  } else if (!(std::holds_alternative<
                   common::Indirection<parser::ArithmeticIfStmt>>(
                   actionStmt.statement.u) ||
                 std::holds_alternative<common::Indirection<parser::CycleStmt>>(
                     actionStmt.statement.u) ||
                 std::holds_alternative<common::Indirection<parser::ExitStmt>>(
                     actionStmt.statement.u) ||
                 std::holds_alternative<common::Indirection<parser::StopStmt>>(
                     actionStmt.statement.u) ||
                 std::holds_alternative<common::Indirection<parser::GotoStmt>>(
                     actionStmt.statement.u) ||
                 std::holds_alternative<
                     common::Indirection<parser::ReturnStmt>>(
                     actionStmt.statement.u))) {
    return true;
  } else {
    return false;
  }
}

template<typename A> constexpr bool IsFormat(const parser::Statement<A> &) {
  return std::is_same_v<A, common::Indirection<parser::FormatStmt>>;
}

template<typename A>
constexpr bool IsLegalBranchTarget(const parser::Statement<A> &) {
  return std::is_same_v<A, parser::AssociateStmt> ||
      std::is_same_v<A, parser::EndAssociateStmt> ||
      std::is_same_v<A, parser::IfThenStmt> ||
      std::is_same_v<A, parser::EndIfStmt> ||
      std::is_same_v<A, parser::SelectCaseStmt> ||
      std::is_same_v<A, parser::EndSelectStmt> ||
      std::is_same_v<A, parser::SelectRankStmt> ||
      std::is_same_v<A, parser::SelectTypeStmt> ||
      std::is_same_v<A, common::Indirection<parser::LabelDoStmt>> ||
      std::is_same_v<A, parser::NonLabelDoStmt> ||
      std::is_same_v<A, parser::EndDoStmt> ||
      std::is_same_v<A, common::Indirection<parser::EndDoStmt>> ||
      std::is_same_v<A, parser::BlockStmt> ||
      std::is_same_v<A, parser::EndBlockStmt> ||
      std::is_same_v<A, parser::CriticalStmt> ||
      std::is_same_v<A, parser::EndCriticalStmt> ||
      std::is_same_v<A, parser::ForallConstructStmt> ||
      std::is_same_v<A, parser::ForallStmt> ||
      std::is_same_v<A, parser::WhereConstructStmt> ||
      std::is_same_v<A, parser::EndFunctionStmt> ||
      std::is_same_v<A, parser::EndMpSubprogramStmt> ||
      std::is_same_v<A, parser::EndProgramStmt> ||
      std::is_same_v<A, parser::EndSubroutineStmt>;
}
template<>
constexpr bool IsLegalBranchTarget(
    const parser::Statement<parser::ActionStmt> &actionStmt) {
  if (!isStrictF18) {
    return true;
  } else if (
      !(std::holds_alternative<common::Indirection<parser::ArithmeticIfStmt>>(
            actionStmt.statement.u) ||
          std::holds_alternative<common::Indirection<parser::AssignStmt>>(
              actionStmt.statement.u) ||
          std::holds_alternative<common::Indirection<parser::AssignedGotoStmt>>(
              actionStmt.statement.u) ||
          std::holds_alternative<common::Indirection<parser::PauseStmt>>(
              actionStmt.statement.u))) {
    return true;
  } else {
    return false;
  }
}

template<typename A>
constexpr LabeledStmtClassificationSet constructBranchTargetFlags(
    const parser::Statement<A> &statement) {
  LabeledStmtClassificationSet labeledStmtClassificationSet{};
  if (IsLegalDoTerm(statement)) {
    labeledStmtClassificationSet.set(TargetStatementEnum::Do);
  }
  if (IsLegalBranchTarget(statement)) {
    labeledStmtClassificationSet.set(TargetStatementEnum::Branch);
  }
  if (IsFormat(statement)) {
    labeledStmtClassificationSet.set(TargetStatementEnum::Format);
  }
  return labeledStmtClassificationSet;
}

bool BothEqOrNone(const std::optional<parser::Name> &name_a,
    const std::optional<parser::Name> &name_b) {
  if (name_a.has_value()) {
    if (name_b.has_value()) {
      if (name_a->ToString() == name_b->ToString()) {
        return true;
      } else {
        return false;
      }
    } else {
      return false;
    }
  } else {
    if (!name_b.has_value()) {
      return true;
    } else {
      return false;
    }
  }
}

bool PresentAndEq(const std::optional<parser::Name> &name_a,
    const std::optional<parser::Name> &name_b) {
  if (!name_a.has_value()) {
    return true;
  } else if (name_b.has_value()) {
    if (name_a->ToString() == name_b->ToString()) {
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

struct UnitAnalysis {
  UnitAnalysis() { scopeModel.push_back(0); }
  UnitAnalysis(UnitAnalysis &&that)
    : doStmtSources{std::move(that.doStmtSources)},
      formatStmtSources{std::move(that.formatStmtSources)},
      otherStmtSources{std::move(that.otherStmtSources)},
      targetStmts{std::move(that.targetStmts)}, scopeModel{std::move(
                                                    that.scopeModel)} {}

  SourceStmtList doStmtSources;
  SourceStmtList formatStmtSources;
  SourceStmtList otherStmtSources;
  TargetStmtMap targetStmts;
  std::vector<ProxyForScope> scopeModel;
};

class ParseTreeAnalyzer {
public:
  ParseTreeAnalyzer() {}
  ParseTreeAnalyzer(ParseTreeAnalyzer &&that)
    : programUnits_{std::move(that.programUnits_)},
      errorHandler_{std::move(that.errorHandler_)}, currentPosition_{std::move(
                                                        that.currentPosition_)},
      constructNames_{std::move(that.constructNames_)} {}

  template<typename A> constexpr bool Pre(const A &) { return true; }
  template<typename A> constexpr void Post(const A &) {}

  template<typename A> bool Pre(const parser::Statement<A> &statement) {
    currentPosition_ = statement.source;
    if (statement.label.has_value())
      addTargetLabelDefinition(
          statement.label.value(), constructBranchTargetFlags(statement));
    return true;
  }

  // see 11.1.1
  bool Pre(const parser::ProgramUnit &) { return PushNewScope(); }
  bool Pre(const parser::AssociateConstruct &associateConstruct) {
    return pushConstructName(associateConstruct);
  }
  bool Pre(const parser::BlockConstruct &blockConstruct) {
    return pushConstructName(blockConstruct);
  }
  bool Pre(const parser::ChangeTeamConstruct &changeTeamConstruct) {
    return pushConstructName(changeTeamConstruct);
  }
  bool Pre(const parser::CriticalConstruct &criticalConstruct) {
    return pushConstructName(criticalConstruct);
  }
  bool Pre(const parser::DoConstruct &doConstruct) {
    return pushConstructName(doConstruct);
  }
  bool Pre(const parser::IfConstruct &ifConstruct) {
    return pushConstructName(ifConstruct);
  }
  bool Pre(const parser::IfConstruct::ElseIfBlock &) {
    return switchToNewScope();
  }
  bool Pre(const parser::IfConstruct::ElseBlock &) {
    return switchToNewScope();
  }
  bool Pre(const parser::CaseConstruct &caseConstruct) {
    return pushConstructName(caseConstruct);
  }
  bool Pre(const parser::CaseConstruct::Case &) { return switchToNewScope(); }
  bool Pre(const parser::SelectRankConstruct &selectRankConstruct) {
    return pushConstructName(selectRankConstruct);
  }
  bool Pre(const parser::SelectRankConstruct::RankCase &) {
    return switchToNewScope();
  }
  bool Pre(const parser::SelectTypeConstruct &selectTypeConstruct) {
    return pushConstructName(selectTypeConstruct);
  }
  bool Pre(const parser::SelectTypeConstruct::TypeCase &) {
    return switchToNewScope();
  }
  bool Pre(const parser::WhereConstruct &whereConstruct) {
    return pushConstructNameWithoutBlock(whereConstruct);
  }
  bool Pre(const parser::ForallConstruct &forallConstruct) {
    return pushConstructNameWithoutBlock(forallConstruct);
  }

  void Post(const parser::ProgramUnit &) { PopScope(); }
  void Post(const parser::AssociateConstruct &associateConstruct) {
    popConstructName(associateConstruct);
  }
  void Post(const parser::BlockConstruct &blockConstruct) {
    popConstructName(blockConstruct);
  }
  void Post(const parser::ChangeTeamConstruct &changeTeamConstruct) {
    popConstructName(changeTeamConstruct);
  }
  void Post(const parser::CriticalConstruct &criticalConstruct) {
    popConstructName(criticalConstruct);
  }
  void Post(const parser::DoConstruct &doConstruct) {
    popConstructName(doConstruct);
  }
  void Post(const parser::IfConstruct &ifConstruct) {
    popConstructName(ifConstruct);
  }
  void Post(const parser::CaseConstruct &caseConstruct) {
    popConstructName(caseConstruct);
  }
  void Post(const parser::SelectRankConstruct &selectRankConstruct) {
    popConstructName(selectRankConstruct);
  }
  void Post(const parser::SelectTypeConstruct &selectTypeConstruct) {
    popConstructName(selectTypeConstruct);
  }

  void Post(const parser::WhereConstruct &whereConstruct) {
    popConstructNameWithoutBlock(whereConstruct);
  }
  void Post(const parser::ForallConstruct &forallConstruct) {
    popConstructNameWithoutBlock(forallConstruct);
  }

  void Post(const parser::LabelDoStmt &labelDoStmt) {
    addLabelReferenceFromDoStmt(std::get<parser::Label>(labelDoStmt.t));
  }
  void Post(const parser::GotoStmt &gotoStmt) { addLabelReference(gotoStmt.v); }
  void Post(const parser::ComputedGotoStmt &computedGotoStmt) {
    addLabelReference(std::get<std::list<parser::Label>>(computedGotoStmt.t));
  }
  void Post(const parser::ArithmeticIfStmt &arithmeticIfStmt) {
    addLabelReference(std::get<1>(arithmeticIfStmt.t));
    addLabelReference(std::get<2>(arithmeticIfStmt.t));
    addLabelReference(std::get<3>(arithmeticIfStmt.t));
  }
  void Post(const parser::AssignStmt &assignStmt) {
    addLabelReference(std::get<parser::Label>(assignStmt.t));
  }
  void Post(const parser::AssignedGotoStmt &assignedGotoStmt) {
    addLabelReference(std::get<std::list<parser::Label>>(assignedGotoStmt.t));
  }
  void Post(const parser::AltReturnSpec &altReturnSpec) {
    addLabelReference(altReturnSpec.v);
  }

  void Post(const parser::ErrLabel &errLabel) { addLabelReference(errLabel.v); }
  void Post(const parser::EndLabel &endLabel) { addLabelReference(endLabel.v); }
  void Post(const parser::EorLabel &eorLabel) { addLabelReference(eorLabel.v); }
  void Post(const parser::Format &format) {
    // BUG: the label is saved as an IntLiteralConstant rather than a Label
#if 0
    if (const auto *P{std::get_if<parser::Label>(&format.u)}) {
      addLabelReferenceFromFormatStmt(*P);
    }
#else
    if (const auto *P{std::get_if<0>(&format.u)}) {
      addLabelReferenceFromFormatStmt(
          parser::Label{std::get<0>(std::get<parser::IntLiteralConstant>(
              std::get<parser::LiteralConstant>((*P->thing).u).u)
                                        .t)});
    }
#endif
  }
  void Post(const parser::CycleStmt &cycleStmt) {
    if (cycleStmt.v.has_value()) {
      CheckLabelContext("CYCLE", cycleStmt.v->ToString());
    }
  }
  void Post(const parser::ExitStmt &exitStmt) {
    if (exitStmt.v.has_value()) {
      CheckLabelContext("EXIT", exitStmt.v->ToString());
    }
  }

  const std::vector<UnitAnalysis> &programUnits() const {
    return programUnits_;
  }
  parser::Messages &errorHandler() { return errorHandler_; }

private:
  bool pushSubscope() {
    programUnits_.back().scopeModel.push_back(currentScope_);
    currentScope_ = programUnits_.back().scopeModel.size() - 1;
    return true;
  }
  bool PushNewScope() {
    programUnits_.emplace_back(UnitAnalysis{});
    return pushSubscope();
  }
  void PopScope() {
    currentScope_ = programUnits_.back().scopeModel[currentScope_];
  }
  bool switchToNewScope() {
    PopScope();
    return pushSubscope();
  }

  template<typename A> bool pushConstructName(const A &a) {
    const auto &optionalName{std::get<0>(std::get<0>(a.t).statement.t)};
    if (optionalName.has_value()) {
      constructNames_.emplace_back(optionalName->ToString());
    }
    return pushSubscope();
  }
  bool pushConstructName(const parser::BlockConstruct &blockConstruct) {
    const auto &optionalName{
        std::get<parser::Statement<parser::BlockStmt>>(blockConstruct.t)
            .statement.v};
    if (optionalName.has_value()) {
      constructNames_.emplace_back(optionalName->ToString());
    }
    return pushSubscope();
  }
  template<typename A> bool pushConstructNameWithoutBlock(const A &a) {
    const auto &optionalName{std::get<0>(std::get<0>(a.t).statement.t)};
    if (optionalName.has_value()) {
      constructNames_.emplace_back(optionalName->ToString());
    }
    return true;
  }

  template<typename A> void popConstructNameWithoutBlock(const A &a) {
    CheckName(a);
    popConstructNameIfPresent(a);
  }

  template<typename A> void popConstructNameIfPresent(const A &a) {
    const auto &optionalName{std::get<0>(std::get<0>(a.t).statement.t)};
    if (optionalName.has_value()) {
      constructNames_.pop_back();
    }
  }

  void popConstructNameIfPresent(const parser::BlockConstruct &blockConstruct) {
    const auto &optionalName{
        std::get<parser::Statement<parser::BlockStmt>>(blockConstruct.t)
            .statement.v};
    if (optionalName.has_value()) {
      constructNames_.pop_back();
    }
  }

  template<typename A> void popConstructName(const A &a) {
    CheckName(a);
    PopScope();
    popConstructNameIfPresent(a);
  }

  // C1144
  void popConstructName(const parser::CaseConstruct &caseConstruct) {
    CheckName(caseConstruct, "CASE");
    PopScope();
    popConstructNameIfPresent(caseConstruct);
  }

  // C1154, C1156
  void popConstructName(
      const parser::SelectRankConstruct &selectRankConstruct) {
    CheckName(selectRankConstruct, "RANK", "RANK ");
    PopScope();
    popConstructNameIfPresent(selectRankConstruct);
  }

  // C1165
  void popConstructName(
      const parser::SelectTypeConstruct &selectTypeConstruct) {
    CheckName(selectTypeConstruct, "TYPE", "TYPE ");
    PopScope();
    popConstructNameIfPresent(selectTypeConstruct);
  }

  // C1106
  void CheckName(const parser::AssociateConstruct &associateConstruct) {
    CheckName("ASSOCIATE", associateConstruct);
  }
  // C1117
  void CheckName(const parser::CriticalConstruct &criticalConstruct) {
    CheckName("CRITICAL", criticalConstruct);
  }
  // C1131
  void CheckName(const parser::DoConstruct &doConstruct) {
    CheckName("DO", doConstruct);
  }
  // C1035
  void CheckName(const parser::ForallConstruct &forallConstruct) {
    CheckName("FORALL", forallConstruct);
  }

  template<typename A>
  void CheckName(const char *const constructTag, const A &a) {
    if (!BothEqOrNone(
            std::get<std::optional<parser::Name>>(std::get<0>(a.t).statement.t),
            std::get<2>(a.t).statement.v)) {
      errorHandler_.Say(currentPosition_,
          parser::MessageFormattedText{
              "%s construct name mismatch"_err_en_US, constructTag});
    }
  }

  // C1109
  void CheckName(const parser::BlockConstruct &blockConstruct) {
    if (!BothEqOrNone(
            std::get<parser::Statement<parser::BlockStmt>>(blockConstruct.t)
                .statement.v,
            std::get<parser::Statement<parser::EndBlockStmt>>(blockConstruct.t)
                .statement.v)) {
      errorHandler_.Say(currentPosition_,
          parser::MessageFormattedText{
              "BLOCK construct name mismatch"_err_en_US});
    }
  }
  // C1112
  void CheckName(const parser::ChangeTeamConstruct &changeTeamConstruct) {
    if (!BothEqOrNone(std::get<std::optional<parser::Name>>(
                          std::get<parser::Statement<parser::ChangeTeamStmt>>(
                              changeTeamConstruct.t)
                              .statement.t),
            std::get<std::optional<parser::Name>>(
                std::get<parser::Statement<parser::EndChangeTeamStmt>>(
                    changeTeamConstruct.t)
                    .statement.t))) {
      errorHandler_.Say(currentPosition_,
          parser::MessageFormattedText{
              "CHANGE TEAM construct name mismatch"_err_en_US});
    }
  }

  // C1142
  void CheckName(const parser::IfConstruct &ifConstruct) {
    const auto &constructName{std::get<std::optional<parser::Name>>(
        std::get<parser::Statement<parser::IfThenStmt>>(ifConstruct.t)
            .statement.t)};
    if (!BothEqOrNone(constructName,
            std::get<parser::Statement<parser::EndIfStmt>>(ifConstruct.t)
                .statement.v)) {
      errorHandler_.Say(currentPosition_,
          parser::MessageFormattedText{"IF construct name mismatch"_err_en_US});
    }
    for (const auto &elseIfBlock :
        std::get<std::list<parser::IfConstruct::ElseIfBlock>>(ifConstruct.t)) {
      if (!PresentAndEq(
              std::get<std::optional<parser::Name>>(
                  std::get<parser::Statement<parser::ElseIfStmt>>(elseIfBlock.t)
                      .statement.t),
              constructName)) {
        errorHandler_.Say(currentPosition_,
            parser::MessageFormattedText{
                "ELSE IF statement name mismatch"_err_en_US});
      }
    }
    if (std::get<std::optional<parser::IfConstruct::ElseBlock>>(ifConstruct.t)
            .has_value()) {
      if (!PresentAndEq(
              std::get<parser::Statement<parser::ElseStmt>>(
                  std::get<std::optional<parser::IfConstruct::ElseBlock>>(
                      ifConstruct.t)
                      ->t

                  )
                  .statement.v,
              constructName)) {
        errorHandler_.Say(currentPosition_,
            parser::MessageFormattedText{
                "ELSE statement name mismatch"_err_en_US});
      }
    }
  }

  template<typename A>
  void CheckName(const A &a, const char *const selectTag,
      const char *const selectSubTag = "") {
    const auto &constructName{std::get<0>(std::get<0>(a.t).statement.t)};
    if (!BothEqOrNone(constructName, std::get<2>(a.t).statement.v)) {
      errorHandler_.Say(currentPosition_,
          parser::MessageFormattedText{
              "SELECT %s construct name mismatch"_err_en_US, selectTag});
    }
    for (const auto &subpart : std::get<1>(a.t)) {
      if (!PresentAndEq(std::get<std::optional<parser::Name>>(
                            std::get<0>(subpart.t).statement.t),
              constructName)) {
        errorHandler_.Say(currentPosition_,
            parser::MessageFormattedText{
                "%sCASE statement name mismatch"_err_en_US, selectSubTag});
      }
    }
  }

  // C1033
  void CheckName(const parser::WhereConstruct &whereConstruct) {
    const auto &constructName{std::get<std::optional<parser::Name>>(
        std::get<parser::Statement<parser::WhereConstructStmt>>(
            whereConstruct.t)
            .statement.t)};
    if (!BothEqOrNone(constructName,
            std::get<parser::Statement<parser::EndWhereStmt>>(whereConstruct.t)
                .statement.v)) {
      errorHandler_.Say(currentPosition_,
          parser::MessageFormattedText{
              "WHERE construct name mismatch"_err_en_US});
    }
    for (const auto &maskedElsewhere :
        std::get<std::list<parser::WhereConstruct::MaskedElsewhere>>(
            whereConstruct.t)) {
      if (!PresentAndEq(
              std::get<std::optional<parser::Name>>(
                  std::get<parser::Statement<parser::MaskedElsewhereStmt>>(
                      maskedElsewhere.t)
                      .statement.t),
              constructName)) {
        errorHandler_.Say(currentPosition_,
            parser::MessageFormattedText{
                "ELSEWHERE (<mask>) statement name mismatch"_err_en_US});
      }
    }
    if (std::get<std::optional<parser::WhereConstruct::Elsewhere>>(
            whereConstruct.t)
            .has_value()) {
      if (!PresentAndEq(
              std::get<parser::Statement<parser::ElsewhereStmt>>(
                  std::get<std::optional<parser::WhereConstruct::Elsewhere>>(
                      whereConstruct.t)
                      ->t)
                  .statement.v,
              constructName)) {
        errorHandler_.Say(currentPosition_,
            parser::MessageFormattedText{
                "ELSEWHERE statement name mismatch"_err_en_US});
      }
    }
  }

  // C1134, C1166
  template<typename A>
  void CheckLabelContext(const char *const stmtString, const A &constructName) {
    const auto I{std::find(
        constructNames_.crbegin(), constructNames_.crend(), constructName)};
    if (I == constructNames_.crend()) {
      errorHandler_.Say(currentPosition_,
          parser::MessageFormattedText{
              "%s construct-name '%s' is not in scope"_err_en_US, stmtString,
              constructName.c_str()});
    }
  }

  // 6.2.5, paragraph 2
  void CheckLabelInRange(parser::Label label) {
    if (label < 1 || label > 99999) {
      errorHandler_.Say(currentPosition_,
          parser::MessageFormattedText{
              "label '%" PRIu64 "' is out of range"_err_en_US, label});
    }
  }

  // 6.2.5., paragraph 2
  void addTargetLabelDefinition(parser::Label label,
      LabeledStmtClassificationSet labeledStmtClassificationSet) {
    CheckLabelInRange(label);
    const auto pair{
        programUnits_.back().targetStmts.emplace(std::make_pair(label,
            LabeledStatementInfoTuplePOD{currentScope_, currentPosition_,
                labeledStmtClassificationSet}))};
    if (!pair.second) {
      errorHandler_.Say(currentPosition_,
          parser::MessageFormattedText{
              "label '%" PRIu64 "' is not distinct"_err_en_US, label});
    }
  }

  void addLabelReferenceFromDoStmt(parser::Label label) {
    CheckLabelInRange(label);
    programUnits_.back().doStmtSources.emplace_back(

        label, currentScope_, currentPosition_);
  }

  void addLabelReferenceFromFormatStmt(parser::Label label) {
    CheckLabelInRange(label);
    programUnits_.back().formatStmtSources.emplace_back(

        label, currentScope_, currentPosition_);
  }

  void addLabelReference(parser::Label label) {
    CheckLabelInRange(label);
    programUnits_.back().otherStmtSources.emplace_back(

        label, currentScope_, currentPosition_);
  }

  void addLabelReference(const std::list<parser::Label> &labels) {
    for (const parser::Label &label : labels) {
      addLabelReference(label);
    }
  }

  std::vector<UnitAnalysis> programUnits_;
  parser::Messages errorHandler_;
  parser::CharBlock currentPosition_{nullptr};
  ProxyForScope currentScope_{0};
  std::vector<std::string> constructNames_;
};

bool InInclusiveScope(const std::vector<ProxyForScope> &scopes,
    ProxyForScope tail, ProxyForScope head) {
  for (; tail != head; tail = scopes[tail]) {
    if (!HasScope(tail)) {
      return false;
    }
  }
  return true;
}

ParseTreeAnalyzer LabelAnalysis(const parser::Program &program) {
  ParseTreeAnalyzer analysis;
  Walk(program, analysis);
  return analysis;
}

bool InBody(const parser::CharBlock &position,
    const std::pair<parser::CharBlock, parser::CharBlock> &pair) {
  if (position.begin() >= pair.first.begin()) {
    if (position.begin() < pair.second.end()) {
      return true;
    }
  }
  return false;
}

LabeledStatementInfoTuplePOD GetLabel(
    const TargetStmtMap &labels, const parser::Label &label) {
  const auto iter{labels.find(label)};
  if (iter == labels.cend()) {
    return {0u, nullptr, LabeledStmtClassificationSet{}};
  } else {
    return iter->second;
  }
}

// 11.1.7.3
void CheckBranchesIntoDoBody(const SourceStmtList &branches,
    const TargetStmtMap &labels, const std::vector<ProxyForScope> &scopes,
    const IndexList &loopBodies, parser::Messages &errorHandler) {
  for (const auto branch : branches) {
    const auto &label{branch.parserLabel};
    auto branchTarget{GetLabel(labels, label)};
    if (HasScope(branchTarget.proxyForScope)) {
      const auto &fromPosition{branch.parserCharBlock};
      const auto &toPosition{branchTarget.parserCharBlock};
      for (const auto body : loopBodies) {
        if (!InBody(fromPosition, body) && InBody(toPosition, body)) {
          if (isStrictF18) {
            errorHandler.Say(fromPosition,
                parser::MessageFormattedText{
                    "branch into '%s' from another scope"_err_en_US,
                    body.first.ToString().c_str()});
          } else {
            errorHandler.Say(fromPosition,
                parser::MessageFormattedText{
                    "branch into '%s' from another scope"_en_US,
                    body.first.ToString().c_str()});
          }
        }
      }
    }
  }
}

void CheckDoNesting(
    const IndexList &loopBodies, parser::Messages &errorHandler) {
  for (auto i1{loopBodies.cbegin()}; i1 != loopBodies.cend(); ++i1) {
    const auto &v1{*i1};
    for (auto i2{i1 + 1}; i2 != loopBodies.cend(); ++i2) {
      const auto &v2{*i2};
      if (v2.first.begin() < v1.second.end() &&
          v1.second.begin() < v2.second.begin()) {
        errorHandler.Say(v2.second,
            parser::MessageFormattedText{"'%s' doesn't properly nest"_err_en_US,
                v1.first.ToString().c_str()});
      }
    }
  }
}

parser::CharBlock SkipLabel(const parser::CharBlock &position) {
  const long maxPosition{position.end() - position.begin()};
  if (maxPosition && (position[0] >= '0') && (position[0] <= '9')) {
    long i{1l};
    for (; (i < maxPosition) && std::isdigit(position[i]); ++i) {
    }
    for (; (i < maxPosition) && std::isspace(position[i]); ++i) {
    }
    return parser::CharBlock{position.begin() + i, position.end()};
  }
  return position;
}

void CheckLabelDoConstraints(const SourceStmtList &dos,
    const SourceStmtList &branches, const TargetStmtMap &labels,
    const std::vector<ProxyForScope> &scopes, parser::Messages &errorHandler) {
  IndexList loopBodies;
  for (const auto stmt : dos) {
    const auto &label{stmt.parserLabel};
    const auto &scope{stmt.proxyForScope};
    const auto &position{stmt.parserCharBlock};
    auto doTarget{GetLabel(labels, label)};
    if (!HasScope(doTarget.proxyForScope)) {
      // C1133
      errorHandler.Say(position,
          parser::MessageFormattedText{
              "label '%" PRIu64 "' cannot be found"_err_en_US, label});
    } else if (doTarget.parserCharBlock.begin() < position.begin()) {
      // R1119
      errorHandler.Say(position,
          parser::MessageFormattedText{
              "label '%" PRIu64 "' doesn't lexically follow DO stmt"_err_en_US,
              label});
    } else if (!InInclusiveScope(scopes, scope, doTarget.proxyForScope)) {
      // C1133
      if (isStrictF18) {
        errorHandler.Say(position,
            parser::MessageFormattedText{
                "label '%" PRIu64 "' is not in scope"_err_en_US, label});
      } else {
        errorHandler.Say(position,
            parser::MessageFormattedText{
                "label '%" PRIu64 "' is not in scope"_en_US, label});
      }
    } else if (!doTarget.labeledStmtClassificationSet.test(
                   TargetStatementEnum::Do)) {
      errorHandler.Say(doTarget.parserCharBlock,
          parser::MessageFormattedText{
              "'%" PRIu64 "' invalid DO terminal statement"_err_en_US, label});
    } else {
      loopBodies.emplace_back(SkipLabel(position), doTarget.parserCharBlock);
    }
  }

  CheckBranchesIntoDoBody(branches, labels, scopes, loopBodies, errorHandler);
  CheckDoNesting(loopBodies, errorHandler);
}

// 6.2.5
void CheckScopeConstraints(const SourceStmtList &stmts,
    const TargetStmtMap &labels, const std::vector<ProxyForScope> &scopes,
    parser::Messages &errorHandler) {
  for (const auto stmt : stmts) {
    const auto &label{stmt.parserLabel};
    const auto &scope{stmt.proxyForScope};
    const auto &position{stmt.parserCharBlock};
    auto target{GetLabel(labels, label)};
    if (!HasScope(target.proxyForScope)) {
      errorHandler.Say(position,
          parser::MessageFormattedText{
              "label '%" PRIu64 "' was not found"_err_en_US, label});
    } else if (!InInclusiveScope(scopes, scope, target.proxyForScope)) {
      if (isStrictF18) {
        errorHandler.Say(position,
            parser::MessageFormattedText{
                "label '%" PRIu64 "' is not in scope"_err_en_US, label});
      } else {
        errorHandler.Say(position,
            parser::MessageFormattedText{
                "label '%" PRIu64 "' is not in scope"_en_US, label});
      }
    }
  }
}

void CheckBranchTargetConstraints(const SourceStmtList &stmts,
    const TargetStmtMap &labels, parser::Messages &errorHandler) {
  for (const auto stmt : stmts) {
    const auto &label{stmt.parserLabel};
    auto branchTarget{GetLabel(labels, label)};
    if (HasScope(branchTarget.proxyForScope)) {
      if (!branchTarget.labeledStmtClassificationSet.test(
              TargetStatementEnum::Branch)) {
        errorHandler.Say(branchTarget.parserCharBlock,
            parser::MessageFormattedText{
                "'%" PRIu64 "' not a branch target"_err_en_US, label});
      }
    }
  }
}

void CheckBranchConstraints(const SourceStmtList &branches,
    const TargetStmtMap &labels, const std::vector<ProxyForScope> &scopes,
    parser::Messages &errorHandler) {
  CheckScopeConstraints(branches, labels, scopes, errorHandler);
  CheckBranchTargetConstraints(branches, labels, errorHandler);
}

void CheckDataXferTargetConstraints(const SourceStmtList &stmts,
    const TargetStmtMap &labels, parser::Messages &errorHandler) {
  for (const auto stmt : stmts) {
    const auto &label{stmt.parserLabel};
    auto ioTarget{GetLabel(labels, label)};
    if (HasScope(ioTarget.proxyForScope)) {
      if (!ioTarget.labeledStmtClassificationSet.test(
              TargetStatementEnum::Format)) {
        errorHandler.Say(ioTarget.parserCharBlock,
            parser::MessageFormattedText{
                "'%" PRIu64 "' not a FORMAT"_err_en_US, label});
      }
    }
  }
}

void CheckDataTransferConstraints(const SourceStmtList &dataTransfers,
    const TargetStmtMap &labels, const std::vector<ProxyForScope> &scopes,
    parser::Messages &errorHandler) {
  CheckScopeConstraints(dataTransfers, labels, scopes, errorHandler);
  CheckDataXferTargetConstraints(dataTransfers, labels, errorHandler);
}

bool CheckConstraints(ParseTreeAnalyzer &&parseTreeAnalysis,
    const parser::CookedSource &cookedSource) {
  auto &errorHandler{parseTreeAnalysis.errorHandler()};
  for (const auto &programUnit : parseTreeAnalysis.programUnits()) {
    const auto &dos{programUnit.doStmtSources};
    const auto &branches{programUnit.otherStmtSources};
    const auto &labels{programUnit.targetStmts};
    const auto &scopes{programUnit.scopeModel};
    CheckLabelDoConstraints(dos, branches, labels, scopes, errorHandler);
    CheckBranchConstraints(branches, labels, scopes, errorHandler);
    const auto &dataTransfers{programUnit.formatStmtSources};
    CheckDataTransferConstraints(dataTransfers, labels, scopes, errorHandler);
  }
  if (!errorHandler.empty()) {
    errorHandler.Emit(std::cerr, cookedSource);
  }
  return !errorHandler.AnyFatalError();
}

bool ValidateLabels(
    const parser::Program &program, const parser::CookedSource &cookedSource) {
  return CheckConstraints(LabelAnalysis(program), cookedSource);
}

}  // namespace Fortran::semantics
