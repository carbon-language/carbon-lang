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
#include <cassert>
#include <cctype>
#include <cstdarg>
#include <iostream>

namespace {

using namespace Fortran;
using namespace parser::literals;

ENUM_CLASS(TargetStatementEnum, Do, Branch, Format)
using TargetStmtType = common::EnumSet<TargetStatementEnum, 8>;

using IndexList = std::vector<std::pair<parser::CharBlock, parser::CharBlock>>;
using ScopeProxy = unsigned;
using LabelStmtInfo = std::tuple<ScopeProxy, parser::CharBlock, TargetStmtType>;
using TargetStmtMap = std::map<parser::Label, LabelStmtInfo>;
using SourceStmtList =
    std::vector<std::tuple<parser::Label, ScopeProxy, parser::CharBlock>>;
using ErrorHandler = parser::Messages;

const bool isStrictF18{false};  // FIXME - make a command-line option

inline bool HasScope(ScopeProxy scope) { return scope != ScopeProxy{0u}; }

inline bool HasNoScope(ScopeProxy scope) { return !HasScope(scope); }

parser::Message &Report(ErrorHandler &eh, const parser::CharBlock &ip,
    parser::MessageFormattedText &&msg) {
  return eh.Say(parser::Message{ip, msg});
}

inline bool HasNoErrors(const ErrorHandler &eh) { return !eh.AnyFatalError(); }

/// \brief Is this a legal DO terminator?
/// Pattern match dependent on the standard we're enforcing
/// F18:R1131 (must be CONTINUE or END DO)
template<typename A>
constexpr bool IsLegalDoTerm(const parser::Statement<A> &) {
  return std::disjunction_v<
      std::is_same<A, common::Indirection<parser::EndDoStmt>>,
      std::is_same<A, parser::EndDoStmt>>;
}
template<>
constexpr bool IsLegalDoTerm(
    const parser::Statement<parser::ActionStmt> &actionStmt) {
  if (std::holds_alternative<parser::ContinueStmt>(actionStmt.statement.u)) {
    // See F08:C816
    return true;
  } else if (isStrictF18) {
    return false;
  } else {
    // Applies in F08 and earlier
    return !(
        std::holds_alternative<common::Indirection<parser::ArithmeticIfStmt>>(
            actionStmt.statement.u) ||
        std::holds_alternative<common::Indirection<parser::CycleStmt>>(
            actionStmt.statement.u) ||
        std::holds_alternative<common::Indirection<parser::ExitStmt>>(
            actionStmt.statement.u) ||
        std::holds_alternative<common::Indirection<parser::StopStmt>>(
            actionStmt.statement.u) ||
        std::holds_alternative<common::Indirection<parser::GotoStmt>>(
            actionStmt.statement.u) ||
        std::holds_alternative<common::Indirection<parser::ReturnStmt>>(
            actionStmt.statement.u));
  }
}

/// \brief Is this a FORMAT stmt?
/// Pattern match for FORMAT statement
template<typename A> constexpr bool IsFormat(const parser::Statement<A> &) {
  return std::is_same_v<A, common::Indirection<parser::FormatStmt>>;
}

/// \brief Is this a legal branch target?
/// Pattern match dependent on the standard we're enforcing
template<typename A>
constexpr bool IsLegalBranchTarget(const parser::Statement<A> &) {
  return std::disjunction_v<std::is_same<A, parser::AssociateStmt>,
      std::is_same<A, parser::EndAssociateStmt>,
      std::is_same<A, parser::IfThenStmt>, std::is_same<A, parser::EndIfStmt>,
      std::is_same<A, parser::SelectCaseStmt>,
      std::is_same<A, parser::EndSelectStmt>,
      std::is_same<A, parser::SelectRankStmt>,
      std::is_same<A, parser::SelectTypeStmt>,
      std::is_same<A, common::Indirection<parser::LabelDoStmt>>,
      std::is_same<A, parser::NonLabelDoStmt>,
      std::is_same<A, parser::EndDoStmt>,
      std::is_same<A, common::Indirection<parser::EndDoStmt>>,
      std::is_same<A, parser::BlockStmt>, std::is_same<A, parser::EndBlockStmt>,
      std::is_same<A, parser::CriticalStmt>,
      std::is_same<A, parser::EndCriticalStmt>,
      std::is_same<A, parser::ForallConstructStmt>,
      std::is_same<A, parser::ForallStmt>,
      std::is_same<A, parser::WhereConstructStmt>,
      std::is_same<A, parser::EndFunctionStmt>,
      std::is_same<A, parser::EndMpSubprogramStmt>,
      std::is_same<A, parser::EndProgramStmt>,
      std::is_same<A, parser::EndSubroutineStmt>>;
}
template<>
constexpr bool IsLegalBranchTarget(
    const parser::Statement<parser::ActionStmt> &actionStmt) {
  if (!isStrictF18) {
    return true;
  } else {
    // XXX: do we care to flag these as errors? If we want strict F18, these
    // statements should not even be present
    return !(
        std::holds_alternative<common::Indirection<parser::ArithmeticIfStmt>>(
            actionStmt.statement.u) ||
        std::holds_alternative<common::Indirection<parser::AssignStmt>>(
            actionStmt.statement.u) ||
        std::holds_alternative<common::Indirection<parser::AssignedGotoStmt>>(
            actionStmt.statement.u) ||
        std::holds_alternative<common::Indirection<parser::PauseStmt>>(
            actionStmt.statement.u));
  }
}

template<typename A>
constexpr TargetStmtType ConsTrgtFlags(const parser::Statement<A> &statement) {
  TargetStmtType targetStmtType{};
  if (IsLegalDoTerm(statement)) {
    targetStmtType.set(TargetStatementEnum::Do);
  }
  if (IsLegalBranchTarget(statement)) {
    targetStmtType.set(TargetStatementEnum::Branch);
  }
  if (IsFormat(statement)) {
    targetStmtType.set(TargetStatementEnum::Format);
  }
  return targetStmtType;
}

/// \brief \p opt1 and \p opt2 must be either present and identical or absent
/// \param a  an optional construct-name (opening statement)
/// \param b  an optional construct-name (ending statement)
template<typename A> inline bool BothEqOrNone(const A &a, const A &b) {
  return (a.has_value() == b.has_value())
      ? (a.has_value() ? (a.value().ToString() == b.value().ToString()) : true)
      : false;
}

/// \brief \p opt1 must either be absent or identical to \p b
/// \param a  an optional construct-name for an optional constraint
/// \param b  an optional construct-name (opening statement)
template<typename A> inline bool PresentAndEq(const A &a, const A &b) {
  return (!a.has_value()) ||
      (b.has_value() && (a.value().ToString() == b.value().ToString()));
}

/// \brief Iterates over parse tree, creates the analysis result
/// As a side-effect checks the constraints for the usages of
/// <i>construct-name</i>.
struct ParseTreeAnalyzer {
public:
  struct UnitAnalysis {
  public:
    SourceStmtList doStmtSources_;  ///< bases of label-do-stmts
    SourceStmtList
        formatStmtSources_;  ///< bases of all other stmts with labels
    SourceStmtList otherStmtSources_;  ///< bases of all other stmts with labels
    TargetStmtMap targetStmts_;  ///< unique map of labels to stmt info
    std::vector<ScopeProxy> scopeModel_;  ///< scope stack model

    UnitAnalysis() { scopeModel_.push_back(0); }
    UnitAnalysis(UnitAnalysis &&) = default;
    ~UnitAnalysis() = default;
    UnitAnalysis(const UnitAnalysis &) = delete;
    UnitAnalysis &operator=(const UnitAnalysis &) = delete;

    const SourceStmtList &GetLabelDos() const { return doStmtSources_; }
    const SourceStmtList &GetDataXfers() const { return formatStmtSources_; }
    const SourceStmtList &GetBranches() const { return otherStmtSources_; }
    const TargetStmtMap &GetLabels() const { return targetStmts_; }
    const std::vector<ScopeProxy> &GetScopes() const { return scopeModel_; }
  };

  ParseTreeAnalyzer() {}
  ~ParseTreeAnalyzer() = default;
  ParseTreeAnalyzer(ParseTreeAnalyzer &&) = default;
  ParseTreeAnalyzer(const ParseTreeAnalyzer &) = delete;
  ParseTreeAnalyzer &operator=(const ParseTreeAnalyzer &) = delete;

  // Default Pre() and Post()
  template<typename A> constexpr bool Pre(const A &) { return true; }
  template<typename A> constexpr void Post(const A &) {}

  // Specializations of Pre() and Post()

  /// \brief Generic handling of all statements
  template<typename A> bool Pre(const parser::Statement<A> &Stmt) {
    currentPosition_ = Stmt.source;
    if (Stmt.label.has_value())
      AddTrgt(Stmt.label.value(), ConsTrgtFlags(Stmt));
    return true;
  }

  //  Inclusive scopes (see 11.1.1)
  bool Pre(const parser::ProgramUnit &) { return PushNewScope(); }
  bool Pre(const parser::AssociateConstruct &associateConstruct) {
    return PushName(associateConstruct);
  }
  bool Pre(const parser::BlockConstruct &blockConstruct) {
    return PushName(blockConstruct);
  }
  bool Pre(const parser::ChangeTeamConstruct &changeTeamConstruct) {
    return PushName(changeTeamConstruct);
  }
  bool Pre(const parser::CriticalConstruct &criticalConstruct) {
    return PushName(criticalConstruct);
  }
  bool Pre(const parser::DoConstruct &doConstruct) {
    return PushName(doConstruct);
  }
  bool Pre(const parser::IfConstruct &ifConstruct) {
    return PushName(ifConstruct);
  }
  bool Pre(const parser::IfConstruct::ElseIfBlock &) { return SwScope(); }
  bool Pre(const parser::IfConstruct::ElseBlock &) { return SwScope(); }
  bool Pre(const parser::CaseConstruct &caseConstruct) {
    return PushName(caseConstruct);
  }
  bool Pre(const parser::CaseConstruct::Case &) { return SwScope(); }
  bool Pre(const parser::SelectRankConstruct &selectRankConstruct) {
    return PushName(selectRankConstruct);
  }
  bool Pre(const parser::SelectRankConstruct::RankCase &) { return SwScope(); }
  bool Pre(const parser::SelectTypeConstruct &selectTypeConstruct) {
    return PushName(selectTypeConstruct);
  }
  bool Pre(const parser::SelectTypeConstruct::TypeCase &) { return SwScope(); }
  bool Pre(const parser::WhereConstruct &whereConstruct) {
    return PushNonBlockName(whereConstruct);
  }
  bool Pre(const parser::ForallConstruct &forallConstruct) {
    return PushNonBlockName(forallConstruct);
  }

  void Post(const parser::ProgramUnit &) { PopScope(); }
  void Post(const parser::AssociateConstruct &associateConstruct) {
    PopName(associateConstruct);
  }
  void Post(const parser::BlockConstruct &blockConstruct) {
    PopName(blockConstruct);
  }
  void Post(const parser::ChangeTeamConstruct &changeTeamConstruct) {
    PopName(changeTeamConstruct);
  }
  void Post(const parser::CriticalConstruct &criticalConstruct) {
    PopName(criticalConstruct);
  }
  void Post(const parser::DoConstruct &doConstruct) { PopName(doConstruct); }
  void Post(const parser::IfConstruct &ifConstruct) { PopName(ifConstruct); }
  void Post(const parser::CaseConstruct &caseConstruct) {
    PopName(caseConstruct);
  }
  void Post(const parser::SelectRankConstruct &selectRankConstruct) {
    PopName(selectRankConstruct);
  }
  void Post(const parser::SelectTypeConstruct &selectTypeConstruct) {
    PopName(selectTypeConstruct);
  }

  //  Named constructs without block scope
  void Post(const parser::WhereConstruct &whereConstruct) {
    PopNonBlockConstructName(whereConstruct);
  }
  void Post(const parser::ForallConstruct &forallConstruct) {
    PopNonBlockConstructName(forallConstruct);
  }

  //  Statements with label references
  void Post(const parser::LabelDoStmt &labelDoStmt) {
    AddDoBase(std::get<parser::Label>(labelDoStmt.t));
  }
  void Post(const parser::GotoStmt &gotoStmt) { AddBase(gotoStmt.v); }
  void Post(const parser::ComputedGotoStmt &computedGotoStmt) {
    AddBase(std::get<std::list<parser::Label>>(computedGotoStmt.t));
  }
  void Post(const parser::ArithmeticIfStmt &arithmeticIfStmt) {
    AddBase(std::get<1>(arithmeticIfStmt.t));
    AddBase(std::get<2>(arithmeticIfStmt.t));
    AddBase(std::get<3>(arithmeticIfStmt.t));
  }
  void Post(const parser::AssignStmt &assignStmt) {
    AddBase(std::get<parser::Label>(assignStmt.t));
  }
  void Post(const parser::AssignedGotoStmt &assignedGotoStmt) {
    AddBase(std::get<std::list<parser::Label>>(assignedGotoStmt.t));
  }
  void Post(const parser::AltReturnSpec &altReturnSpec) {
    AddBase(altReturnSpec.v);
  }

  void Post(const parser::ErrLabel &errLabel) { AddBase(errLabel.v); }
  void Post(const parser::EndLabel &endLabel) { AddBase(endLabel.v); }
  void Post(const parser::EorLabel &eorLabel) { AddBase(eorLabel.v); }
  void Post(const parser::Format &format) {
    // BUG: the label is saved as an IntLiteralConstant rather than a Label
#if 0
    if (const auto *P{std::get_if<parser::Label>(&format.u)}) {
      AddFmtBase(*P);
    }
#else
    // FIXME: this is wrong, but extracts the label's value
    if (const auto *P{std::get_if<0>(&format.u)}) {
      AddFmtBase(parser::Label{std::get<0>(std::get<parser::IntLiteralConstant>(
          std::get<parser::LiteralConstant>((*P->thing).u).u)
                                               .t)});
    }
#endif
  }
  void Post(const parser::CycleStmt &cycleStmt) {
    if (cycleStmt.v.has_value()) {
      CheckLabelContext("CYCLE", cycleStmt.v.value().ToString());
    }
  }
  void Post(const parser::ExitStmt &exitStmt) {
    if (exitStmt.v.has_value()) {
      CheckLabelContext("EXIT", exitStmt.v.value().ToString());
    }
  }

  // Getters for the results
  const std::vector<UnitAnalysis> &GetProgramUnits() const {
    return programUnits_;
  }
  ErrorHandler &GetEH() { return eh; }

private:
  bool PushScope() {
    programUnits_.back().scopeModel_.push_back(currentScope_);
    currentScope_ = programUnits_.back().scopeModel_.size() - 1;
    return true;
  }
  bool PushNewScope() {
    programUnits_.emplace_back(UnitAnalysis{});
    return PushScope();
  }
  void PopScope() {
    currentScope_ = programUnits_.back().scopeModel_[currentScope_];
  }
  bool SwScope() {
    PopScope();
    return PushScope();
  }

  template<typename A> bool PushName(const A &a) {
    const auto &optionalName{std::get<0>(std::get<0>(a.t).statement.t)};
    if (optionalName.has_value()) {
      constructNames_.push_back(optionalName.value().ToString());
    }
    return PushScope();
  }
  bool PushName(const parser::BlockConstruct &blockConstruct) {
    const auto &optionalName{
        std::get<parser::Statement<parser::BlockStmt>>(blockConstruct.t)
            .statement.v};
    if (optionalName.has_value()) {
      constructNames_.push_back(optionalName.value().ToString());
    }
    return PushScope();
  }
  template<typename A> bool PushNonBlockName(const A &a) {
    const auto &optionalName{std::get<0>(std::get<0>(a.t).statement.t)};
    if (optionalName.has_value()) {
      constructNames_.push_back(optionalName.value().ToString());
    }
    return true;
  }

  template<typename A> void PopNonBlockConstructName(const A &a) {
    CheckName(a);
    SelectivePopBack(a);
  }

  template<typename A> void SelectivePopBack(const A &a) {
    const auto &optionalName{std::get<0>(std::get<0>(a.t).statement.t)};
    if (optionalName.has_value()) {
      constructNames_.pop_back();
    }
  }
  void SelectivePopBack(const parser::BlockConstruct &blockConstruct) {
    const auto &optionalName{
        std::get<parser::Statement<parser::BlockStmt>>(blockConstruct.t)
            .statement.v};
    if (optionalName.has_value()) {
      constructNames_.pop_back();
    }
  }

  /// \brief Check constraints and pop scope
  template<typename A> void PopName(const A &a) {
    CheckName(a);
    PopScope();
    SelectivePopBack(a);
  }

  /// \brief Check <i>case-construct-name</i> and pop the scope
  /// Constraint C1144 - opening and ending name must match if present, and
  /// <i>case-stmt</i> must either match or be unnamed
  void PopName(const parser::CaseConstruct &caseConstruct) {
    CheckName(caseConstruct, "CASE");
    PopScope();
    SelectivePopBack(caseConstruct);
  }

  /// \brief Check <i>select-rank-construct-name</i> and pop the scope
  /// Constraints C1154, C1156 - opening and ending name must match if present,
  /// and <i>select-rank-case-stmt</i> must either match or be unnamed
  void PopName(const parser::SelectRankConstruct &selectRankConstruct) {
    CheckName(selectRankConstruct, "RANK", "RANK ");
    PopScope();
    SelectivePopBack(selectRankConstruct);
  }

  /// \brief Check <i>select-construct-name</i> and pop the scope
  /// Constraint C1165 - opening and ending name must match if present, and
  /// <i>type-guard-stmt</i> must either match or be unnamed
  void PopName(const parser::SelectTypeConstruct &selectTypeConstruct) {
    CheckName(selectTypeConstruct, "TYPE", "TYPE ");
    PopScope();
    SelectivePopBack(selectTypeConstruct);
  }

  // -----------------------------------------------
  // CheckName - check constraints on construct-name
  // Case 1: construct name must be absent or specified & identical on END

  /// \brief Check <i>associate-construct-name</i>, constraint C1106
  void CheckName(const parser::AssociateConstruct &associateConstruct) {
    CheckName("ASSOCIATE", associateConstruct);
  }
  /// \brief Check <i>critical-construct-name</i>, constraint C1117
  void CheckName(const parser::CriticalConstruct &criticalConstruct) {
    CheckName("CRITICAL", criticalConstruct);
  }
  /// \brief Check <i>do-construct-name</i>, constraint C1131
  void CheckName(const parser::DoConstruct &doConstruct) {
    CheckName("DO", doConstruct);
  }
  /// \brief Check <i>forall-construct-name</i>, constraint C1035
  void CheckName(const parser::ForallConstruct &forallConstruct) {
    CheckName("FORALL", forallConstruct);
  }
  /// \brief Common code for ASSOCIATE, CRITICAL, DO, and FORALL
  template<typename A>
  void CheckName(const char *const constructTag, const A &a) {
    if (!BothEqOrNone(
            std::get<std::optional<parser::Name>>(std::get<0>(a.t).statement.t),
            std::get<2>(a.t).statement.v)) {
      Report(eh, currentPosition_,
          parser::MessageFormattedText{
              "%s construct name mismatch"_err_en_US, constructTag});
    }
  }

  /// \brief Check <i>do-construct-name</i>, constraint C1109
  void CheckName(const parser::BlockConstruct &blockConstruct) {
    if (!BothEqOrNone(
            std::get<parser::Statement<parser::BlockStmt>>(blockConstruct.t)
                .statement.v,
            std::get<parser::Statement<parser::EndBlockStmt>>(blockConstruct.t)
                .statement.v)) {
      Report(eh, currentPosition_,
          parser::MessageFormattedText{
              "BLOCK construct name mismatch"_err_en_US});
    }
  }
  /// \brief Check <i>team-cosntruct-name</i>, constraint C1112
  void CheckName(const parser::ChangeTeamConstruct &changeTeamConstruct) {
    if (!BothEqOrNone(std::get<std::optional<parser::Name>>(
                          std::get<parser::Statement<parser::ChangeTeamStmt>>(
                              changeTeamConstruct.t)
                              .statement.t),
            std::get<std::optional<parser::Name>>(
                std::get<parser::Statement<parser::EndChangeTeamStmt>>(
                    changeTeamConstruct.t)
                    .statement.t))) {
      Report(eh, currentPosition_,
          parser::MessageFormattedText{
              "CHANGE TEAM construct name mismatch"_err_en_US});
    }
  }

  // -----------------------------------------------
  // Case 2: same as case 1, but subblock statement construct-names are
  // optional but if they are specified their values must be identical

  /// \brief Check <i>if-construct-name</i>
  /// Constraint C1142 - opening and ending name must match if present, and
  /// <i>else-if-stmt</i> and <i>else-stmt</i> must either match or be unnamed
  void CheckName(const parser::IfConstruct &ifConstruct) {
    const auto &constructName{std::get<std::optional<parser::Name>>(
        std::get<parser::Statement<parser::IfThenStmt>>(ifConstruct.t)
            .statement.t)};
    if (!BothEqOrNone(constructName,
            std::get<parser::Statement<parser::EndIfStmt>>(ifConstruct.t)
                .statement.v)) {
      Report(eh, currentPosition_,
          parser::MessageFormattedText{"IF construct name mismatch"_err_en_US});
    }
    for (const auto &elseIfBlock :
        std::get<std::list<parser::IfConstruct::ElseIfBlock>>(ifConstruct.t)) {
      if (!PresentAndEq(
              std::get<std::optional<parser::Name>>(
                  std::get<parser::Statement<parser::ElseIfStmt>>(elseIfBlock.t)
                      .statement.t),
              constructName)) {
        Report(eh, currentPosition_,
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
                      .value()
                      .t)
                  .statement.v,
              constructName)) {
        Report(eh, currentPosition_,
            parser::MessageFormattedText{
                "ELSE statement name mismatch"_err_en_US});
      }
    }
  }
  /// \brief Common code for SELECT CASE, SELECT RANK, and SELECT TYPE
  template<typename A>
  void CheckName(const A &a, const char *const selectTag,
      const char *const selectSubTag = "") {
    const auto &constructName{std::get<0>(std::get<0>(a.t).statement.t)};
    if (!BothEqOrNone(constructName, std::get<2>(a.t).statement.v)) {
      Report(eh, currentPosition_,
          parser::MessageFormattedText{
              "SELECT %s construct name mismatch"_err_en_US, selectTag});
    }
    for (const auto &subpart : std::get<1>(a.t)) {
      if (!PresentAndEq(std::get<std::optional<parser::Name>>(
                            std::get<0>(subpart.t).statement.t),
              constructName)) {
        Report(eh, currentPosition_,
            parser::MessageFormattedText{
                "%sCASE statement name mismatch"_err_en_US, selectSubTag});
      }
    }
  }

  /// \brief Check <i>where-construct-name</i>
  /// Constraint C1033 - opening and ending name must match if present, and
  /// <i>masked-elsewhere-stmt</i> and <i>elsewhere-stmt</i> either match
  /// or be unnamed
  void CheckName(const parser::WhereConstruct &whereConstruct) {
    const auto &constructName{std::get<std::optional<parser::Name>>(
        std::get<parser::Statement<parser::WhereConstructStmt>>(
            whereConstruct.t)
            .statement.t)};
    if (!BothEqOrNone(constructName,
            std::get<parser::Statement<parser::EndWhereStmt>>(whereConstruct.t)
                .statement.v)) {
      Report(eh, currentPosition_,
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
        Report(eh, currentPosition_,
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
                      .value()
                      .t)
                  .statement.v,
              constructName)) {
        Report(eh, currentPosition_,
            parser::MessageFormattedText{
                "ELSEWHERE statement name mismatch"_err_en_US});
      }
    }
  }

  /// \brief Check constraint <i>construct-name</i> in scope (C1134 and C1166)
  /// \param SStr  a string to specify the statement, \c CYCLE or \c EXIT
  /// \param Label the name used by the \c CYCLE or \c EXIT
  template<typename A>
  void CheckLabelContext(const char *const stmtString, const A &constructName) {
    const auto I{std::find(
        constructNames_.crbegin(), constructNames_.crend(), constructName)};
    if (I == constructNames_.crend()) {
      Report(eh, currentPosition_,
          parser::MessageFormattedText{
              "%s construct-name '%s' is not in scope"_err_en_US, stmtString,
              constructName.c_str()});
    }
  }

  /// \brief Check label range
  /// Constraint per section 6.2.5, paragraph 2
  void CheckLabelInRange(parser::Label label) {
    if ((label < 1) || (label > 99999)) {
      // this is an error: labels must have a value 1 to 99999, inclusive
      Report(eh, currentPosition_,
          parser::MessageFormattedText{
              "label '%lu' is out of range"_err_en_US, label});
    }
  }

  /// \brief Add a labeled statement (label must be distinct)
  /// Constraint per section 6.2.5., paragraph 2
  void AddTrgt(parser::Label label, TargetStmtType targetStmtType) {
    CheckLabelInRange(label);
    const auto pair{programUnits_.back().targetStmts_.insert(
        {label, {currentScope_, currentPosition_, targetStmtType}})};
    if (!pair.second) {
      // this is an error: labels must be pairwise distinct
      Report(eh, currentPosition_,
          parser::MessageFormattedText{
              "label '%lu' is not distinct"_err_en_US, label});
    }
    // Don't enforce a limit to the cardinality of labels
  }

  /// \brief Reference to a labeled statement from a DO statement
  void AddDoBase(parser::Label label) {
    CheckLabelInRange(label);
    programUnits_.back().doStmtSources_.push_back(
        {label, currentScope_, currentPosition_});
  }

  /// \brief Reference to a labeled FORMAT statement
  void AddFmtBase(parser::Label label) {
    CheckLabelInRange(label);
    programUnits_.back().formatStmtSources_.push_back(
        {label, currentScope_, currentPosition_});
  }

  /// \brief Reference to a labeled statement as a (possible) branch
  void AddBase(parser::Label label) {
    CheckLabelInRange(label);
    programUnits_.back().otherStmtSources_.push_back(
        {label, currentScope_, currentPosition_});
  }

  /// \brief References to labeled statements as (possible) branches
  void AddBase(const std::list<parser::Label> &labels) {
    for (const parser::Label &label : labels) {
      AddBase(label);
    }
  }

  std::vector<UnitAnalysis> programUnits_;  ///< results for each program unit
  ErrorHandler eh;  ///< error handler, collects messages
  parser::CharBlock currentPosition_{
      nullptr};  ///< current location in parse tree
  ScopeProxy currentScope_{0};  ///< current scope in the model
  std::vector<std::string> constructNames_;
};

template<typename A, typename B>
bool InInclusiveScope(const A &scopes, B tail, const B &head) {
  assert(HasScope(head));
  assert(HasScope(tail));
  while (HasScope(tail) && (tail != head)) {
    tail = scopes[tail];
  }
  return tail == head;
}

ParseTreeAnalyzer LabelAnalysis(const parser::Program &program) {
  ParseTreeAnalyzer analysis;
  Walk(program, analysis);
  return analysis;
}

template<typename A, typename B>
inline bool InBody(const A &position, const B &pair) {
  assert(pair.first.begin() < pair.second.begin());
  return (position.begin() >= pair.first.begin()) &&
      (position.begin() < pair.second.end());
}

template<typename A, typename B>
LabelStmtInfo GetLabel(const A &labels, const B &label) {
  const auto iter{labels.find(label)};
  if (iter == labels.cend()) {
    return {0u, nullptr, TargetStmtType{}};
  } else {
    return iter->second;
  }
}

/// \brief Check branches into a <i>label-do-stmt</i>
/// Relates to 11.1.7.3, loop activation
template<typename A, typename B, typename C, typename D>
inline void CheckBranchesIntoDoBody(const A &branches, const B &labels,
    const C &scopes, const D &loopBodies, ErrorHandler &eh) {
  for (const auto branch : branches) {
    const auto &label{std::get<parser::Label>(branch)};
    auto branchTarget{GetLabel(labels, label)};
    if (HasScope(std::get<ScopeProxy>(branchTarget))) {
      const auto &fromPosition{std::get<parser::CharBlock>(branch)};
      const auto &toPosition{std::get<parser::CharBlock>(branchTarget)};
      for (const auto body : loopBodies) {
        if (!InBody(fromPosition, body) && InBody(toPosition, body)) {
          // this is an error: branch into labeled DO body
          if (isStrictF18) {
            Report(eh, fromPosition,
                parser::MessageFormattedText{
                    "branch into '%s' from another scope"_err_en_US,
                    body.first.ToString().c_str()});
          } else {
            Report(eh, fromPosition,
                parser::MessageFormattedText{
                    "branch into '%s' from another scope"_en_US,
                    body.first.ToString().c_str()});
          }
        }
      }
    }
  }
}

/// \brief Check that DO loops properly nest
template<typename A>
inline void CheckDoNesting(const A &loopBodies, ErrorHandler &eh) {
  for (auto i1{loopBodies.cbegin()}; i1 != loopBodies.cend(); ++i1) {
    const auto &v1{*i1};
    for (auto i2{i1 + 1}; i2 != loopBodies.cend(); ++i2) {
      const auto &v2{*i2};
      assert(v1.first.begin() != v2.first.begin());
      if ((v2.first.begin() < v1.second.end()) &&
          (v1.second.begin() < v2.second.begin())) {
        // this is an error: DOs do not properly nest
        Report(eh, v2.second,
            parser::MessageFormattedText{"'%s' doesn't properly nest"_err_en_US,
                v1.first.ToString().c_str()});
      }
    }
  }
}

/// \brief Advance \p Pos past any label and whitespace
/// Want the statement without its label for error messages, range checking
template<typename A> inline A SkipLabel(const A &position) {
  const long maxPosition{position.end() - position.begin()};
  if (maxPosition && (position[0] >= '0') && (position[0] <= '9')) {
    long i{1l};
    for (; (i < maxPosition) && std::isdigit(position[i]); ++i)
      ;
    for (; (i < maxPosition) && std::isspace(position[i]); ++i)
      ;
    return parser::CharBlock{position.begin() + i, position.end()};
  }
  return position;
}

/// \brief Check constraints on <i>label-do-stmt</i>
template<typename A, typename B, typename C>
inline void CheckLabelDoConstraints(const A &dos, const A &branches,
    const B &labels, const C &scopes, ErrorHandler &eh) {
  IndexList loopBodies;
  for (const auto stmt : dos) {
    const auto &label{std::get<parser::Label>(stmt)};
    const auto &scope{std::get<ScopeProxy>(stmt)};
    const auto &position{std::get<parser::CharBlock>(stmt)};
    auto doTarget{GetLabel(labels, label)};
    if (HasNoScope(std::get<ScopeProxy>(doTarget))) {
      // C1133: this is an error: label not found
      Report(eh, position,
          parser::MessageFormattedText{
              "label '%lu' cannot be found"_err_en_US, label});
    } else if (std::get<parser::CharBlock>(doTarget).begin() <
        position.begin()) {
      // R1119: this is an error: label does not follow DO
      Report(eh, position,
          parser::MessageFormattedText{
              "label '%lu' doesn't lexically follow DO stmt"_err_en_US, label});
    } else if (!InInclusiveScope(
                   scopes, scope, std::get<ScopeProxy>(doTarget))) {
      // C1133: this is an error: label is not in scope
      if (isStrictF18) {
        Report(eh, position,
            parser::MessageFormattedText{
                "label '%lu' is not in scope"_err_en_US, label});
      } else {
        Report(eh, position,
            parser::MessageFormattedText{
                "label '%lu' is not in scope"_en_US, label});
      }
    } else if ((std::get<TargetStmtType>(doTarget) &
                   TargetStmtType{TargetStatementEnum::Do})
                   .none()) {
      Report(eh, std::get<parser::CharBlock>(doTarget),
          parser::MessageFormattedText{
              "'%lu' invalid DO terminal statement"_err_en_US, label});
    } else {
      // save the loop body marks
      loopBodies.push_back(
          {SkipLabel(position), std::get<parser::CharBlock>(doTarget)});
    }
  }

  // check that nothing jumps into the block
  CheckBranchesIntoDoBody(branches, labels, scopes, loopBodies, eh);
  // check that do loops properly nest
  CheckDoNesting(loopBodies, eh);
}

/// \brief General constraint, control transfers within inclusive scope
/// See, for example, section 6.2.5.
template<typename A, typename B, typename C>
void CheckScopeConstraints(
    const A &stmts, const B &labels, const C &scopes, ErrorHandler &eh) {
  for (const auto stmt : stmts) {
    const auto &label{std::get<parser::Label>(stmt)};
    const auto &scope{std::get<ScopeProxy>(stmt)};
    const auto &position{std::get<parser::CharBlock>(stmt)};
    auto target{GetLabel(labels, label)};
    if (HasNoScope(std::get<ScopeProxy>(target))) {
      // this is an error: label not found
      Report(eh, position,
          parser::MessageFormattedText{
              "label '%lu' was not found"_err_en_US, label});
    } else if (!InInclusiveScope(scopes, scope, std::get<ScopeProxy>(target))) {
      // this is an error: label not in scope
      if (isStrictF18) {
        Report(eh, position,
            parser::MessageFormattedText{
                "label '%lu' is not in scope"_err_en_US, label});
      } else {
        Report(eh, position,
            parser::MessageFormattedText{
                "label '%lu' is not in scope"_en_US, label});
      }
    }
  }
}

template<typename A, typename B>
inline void CheckBranchTargetConstraints(
    const A &stmts, const B &labels, ErrorHandler &eh) {
  for (const auto stmt : stmts) {
    const auto &label{std::get<parser::Label>(stmt)};
    auto branchTarget{GetLabel(labels, label)};
    if (HasScope(std::get<ScopeProxy>(branchTarget))) {
      if ((std::get<TargetStmtType>(branchTarget) &
              TargetStmtType{TargetStatementEnum::Branch})
              .none()) {
        // this is an error: label statement is not a branch target
        Report(eh, std::get<parser::CharBlock>(branchTarget),
            parser::MessageFormattedText{
                "'%lu' not a branch target"_err_en_US, label});
      }
    }
  }
}

/// \brief Validate the constraints on branches
/// \param Analysis  the analysis result
template<typename A, typename B, typename C>
inline void CheckBranchConstraints(
    const A &branches, const B &labels, const C &scopes, ErrorHandler &eh) {
  CheckScopeConstraints(branches, labels, scopes, eh);
  CheckBranchTargetConstraints(branches, labels, eh);
}

template<typename A, typename B>
inline void CheckDataXferTargetConstraints(
    const A &stmts, const B &labels, ErrorHandler &eh) {
  for (const auto stmt : stmts) {
    const auto &label{std::get<parser::Label>(stmt)};
    auto ioTarget{GetLabel(labels, label)};
    if (HasScope(std::get<ScopeProxy>(ioTarget))) {
      if ((std::get<TargetStmtType>(ioTarget) &
              TargetStmtType{TargetStatementEnum::Format})
              .none()) {
        // this is an error: label not a FORMAT
        Report(eh, std::get<parser::CharBlock>(ioTarget),
            parser::MessageFormattedText{
                "'%lu' not a FORMAT"_err_en_US, label});
      }
    }
  }
}

/// \brief Validate that data transfers reference FORMATs in scope
/// \param Analysis  the analysis result
/// These label uses are disjoint from branching (control flow)
template<typename A, typename B, typename C>
inline void CheckDataTransferConstraints(const A &dataTransfers,
    const B &labels, const C &scopes, ErrorHandler &eh) {
  CheckScopeConstraints(dataTransfers, labels, scopes, eh);
  CheckDataXferTargetConstraints(dataTransfers, labels, eh);
}

/// \brief Validate label related constraints on the parse tree
/// \param analysis  the analysis results as run of the parse tree
/// \param cookedSrc cooked source for error report
/// \return true iff all the semantics checks passed
bool CheckConstraints(ParseTreeAnalyzer &&parseTreeAnalysis,
    const parser::CookedSource &cookedSource) {
  auto &eh{parseTreeAnalysis.GetEH()};
  for (const auto &programUnit : parseTreeAnalysis.GetProgramUnits()) {
    const auto &dos{programUnit.GetLabelDos()};
    const auto &branches{programUnit.GetBranches()};
    const auto &labels{programUnit.GetLabels()};
    const auto &scopes{programUnit.GetScopes()};
    CheckLabelDoConstraints(dos, branches, labels, scopes, eh);
    CheckBranchConstraints(branches, labels, scopes, eh);
    const auto &dataTransfers{programUnit.GetDataXfers()};
    CheckDataTransferConstraints(dataTransfers, labels, scopes, eh);
  }
  if (!eh.empty()) {
    eh.Emit(std::cerr, cookedSource);
  }
  return HasNoErrors(eh);
}

}  // namespace

namespace Fortran::semantics {

/// \brief Check the semantics of LABELs in the program
/// \return true iff the program's use of LABELs is semantically correct
bool ValidateLabels(
    const parser::Program &program, const parser::CookedSource &cookedSource) {
  return CheckConstraints(LabelAnalysis(program), cookedSource);
}

}  // namespace Fortran::semantics
