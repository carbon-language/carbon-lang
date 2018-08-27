/* -*- mode: c++; c-basic-offset: 2 -*- */
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
#include "../parser/message.h"
#include "../parser/parse-tree-visitor.h"
#include <cstdarg>
#include <iostream>
#include <cctype>
#include <cassert>

namespace {

using namespace Fortran;
using namespace parser::literals;
using ParseTree_t = parser::Program;
using CookedSource_t = parser::CookedSource;
using Index_t = parser::CharBlock;
using IndexList = std::vector<std::pair<Index_t, Index_t>>;
using Scope_t = unsigned;
using LblStmt_t = std::tuple<Scope_t, Index_t, unsigned>;
using ArcTrgt_t = std::map<parser::Label, LblStmt_t>;
using ArcBase_t = std::vector<std::tuple<parser::Label, Scope_t, Index_t>>;

const bool StrictF18 = false; // FIXME - make a command-line option

const unsigned DO_TERM_FLAG = 1u;
const unsigned BRANCH_TARGET_FLAG = 2u;
const unsigned FORMAT_STMT_FLAG = 4u;

// convenient package for error reporting
struct ErrorHandler {
public:
  explicit ErrorHandler(const parser::CookedSource& CookedSource)
    : cookedSource{CookedSource}, messages{parser::Messages()} {}
  ~ErrorHandler() = default;
  ErrorHandler(ErrorHandler&&) = default;
  ErrorHandler() = delete;
  ErrorHandler(const ErrorHandler&) = delete;
  ErrorHandler& operator=(const ErrorHandler&) = delete;

  parser::Message& Report(const parser::CharBlock& CB,
			  const parser::MessageFixedText& Fixed, ...) {
    va_list ap;
    va_start(ap, Fixed);
    parser::MessageFormattedText Msg{Fixed, ap};
    va_end(ap);
    return messages.Say(parser::Message{CB, Msg});
  }

  const parser::CookedSource& cookedSource;
  parser::Messages messages;
};

/// \brief Is this a legal DO terminator?
/// Pattern match dependent on the standard we're enforcing
template<typename A> constexpr bool IsLegalDoTerm(const parser::Statement<A>&) {
  return false;
}
// F18:R1131 (must be CONTINUE or END DO)
template<> constexpr bool IsLegalDoTerm(const parser::Statement<parser::
					EndDoStmt>&) {
  return true;
}
template<> constexpr bool IsLegalDoTerm(const parser::Statement<common::
					Indirection<parser::EndDoStmt>>&) {
  return true;
}
template<> constexpr bool IsLegalDoTerm(const parser::Statement<parser::
					ActionStmt>& A) {
  if (std::get_if<parser::ContinueStmt>(&A.statement.u)) {
    // See F08:C816 
    return true;
  }
  if (StrictF18)
    return false;
  // Applies in F08 and earlier
  const auto* P{&A.statement.u};
  return !(std::get_if<common::Indirection<parser::ArithmeticIfStmt>>(P) ||
	   std::get_if<common::Indirection<parser::CycleStmt>>(P) ||
	   std::get_if<common::Indirection<parser::ExitStmt>>(P) ||
	   std::get_if<common::Indirection<parser::StopStmt>>(P) ||
	   std::get_if<common::Indirection<parser::GotoStmt>>(P) ||
	   std::get_if<common::Indirection<parser::ReturnStmt>>(P));
}

/// \brief Is this a FORMAT stmt?
/// Pattern match for FORMAT statement
template<typename A> constexpr bool IsFormat(const parser::Statement<A>&) {
  return false;
}
template<> constexpr bool IsFormat(const parser::Statement<common::
				   Indirection<parser::FormatStmt>>&) {
  return true;
}

/// \brief Is this a legal branch target?
/// Pattern match dependent on the standard we're enforcing
template<typename A> constexpr bool IsLegalBranchTarget(const parser::
							Statement<A>&) {
  return false;
}
template<> constexpr bool IsLegalBranchTarget(const parser::Statement<parser::
					      ActionStmt>& A) {
  if (!StrictF18)
    return true;
  // XXX: do we care to flag these as errors? If we want strict F18, these
  // statements should not even be present
  const auto* P{&A.statement.u};
  return !(std::get_if<common::Indirection<parser::ArithmeticIfStmt>>(P) ||
	   std::get_if<common::Indirection<parser::AssignStmt>>(P) ||
	   std::get_if<common::Indirection<parser::AssignedGotoStmt>>(P) ||
	   std::get_if<common::Indirection<parser::PauseStmt>>(P));
}
#define Instantiate(TYPE)						\
  template<> constexpr bool IsLegalBranchTarget(const parser::		\
						Statement<TYPE>&) {	\
    return true;							\
  }
Instantiate(parser::AssociateStmt)
Instantiate(parser::EndAssociateStmt)
Instantiate(parser::IfThenStmt)
Instantiate(parser::EndIfStmt)
Instantiate(parser::SelectCaseStmt)
Instantiate(parser::EndSelectStmt)
Instantiate(parser::SelectRankStmt)
Instantiate(parser::SelectTypeStmt)
Instantiate(common::Indirection<parser::LabelDoStmt>)
Instantiate(parser::NonLabelDoStmt)
Instantiate(parser::EndDoStmt)
Instantiate(common::Indirection<parser::EndDoStmt>)
Instantiate(parser::BlockStmt)
Instantiate(parser::EndBlockStmt)
Instantiate(parser::CriticalStmt)
Instantiate(parser::EndCriticalStmt)
Instantiate(parser::ForallConstructStmt)
Instantiate(parser::ForallStmt)
Instantiate(parser::WhereConstructStmt)
Instantiate(parser::EndFunctionStmt)
Instantiate(parser::EndMpSubprogramStmt)
Instantiate(parser::EndProgramStmt)
Instantiate(parser::EndSubroutineStmt)
#undef Instantiate

template<typename A>
constexpr unsigned ConsTrgtFlags(const parser::Statement<A>& S) {
  unsigned Flags{0u};
  if (IsLegalDoTerm(S))
    Flags |= DO_TERM_FLAG;
  if (IsLegalBranchTarget(S))
    Flags |= BRANCH_TARGET_FLAG;
  if (IsFormat(S))
    Flags |= FORMAT_STMT_FLAG;
  return Flags;
}

/// \brief \p opt1 and \p opt2 must be either present and identical or absent
/// \param opt1  an optional construct-name (opening statement)
/// \param opt2  an optional construct-name (ending statement)
template<typename A> inline bool BothEqOrNone(const A& opt1, const A& opt2) {
  return (opt1.has_value() == opt2.has_value())
    ? (opt1.has_value() 
       ? (opt1.value().ToString() == opt2.value().ToString()) : true)
    : false;
}

/// \brief \p opt1 must either be absent or identical to \p opt2
/// \param opt1  an optional construct-name for an optional constraint
/// \param opt2  an optional construct-name (opening statement)
template<typename A> inline bool PresentAndEq(const A& opt1, const A& opt2) {
  return (!opt1.has_value()) ||
    (opt2.has_value() &&
     (opt1.value().ToString() == opt2.value().ToString()));
}

/// \brief Iterates over parse tree, creates the analysis result
/// As a side-effect checks the constraints for the usages of
/// <i>construct-name</i>.
struct ParseTreeAnalyzer {
public:
  struct UnitAnalysis {
  public:
    ArcBase_t DoArcBases;      ///< bases of label-do-stmts
    ArcBase_t FmtArcBases;     ///< bases of all other stmts with labels
    ArcBase_t ArcBases;        ///< bases of all other stmts with labels
    ArcTrgt_t ArcTrgts;        ///< unique map of labels to stmt info
    std::vector<Scope_t> Scopes; ///< scope stack model

    explicit UnitAnalysis() { Scopes.push_back(0); }
    UnitAnalysis(UnitAnalysis&&) = default;
    ~UnitAnalysis() = default;
    UnitAnalysis(const UnitAnalysis&) = delete;
    UnitAnalysis& operator=(const UnitAnalysis&) = delete;

    const ArcBase_t& GetLabelDos() const { return DoArcBases; }
    const ArcBase_t& GetDataXfers() const { return FmtArcBases; }
    const ArcBase_t& GetBranches() const { return ArcBases; }
    const ArcTrgt_t& GetLabels() const { return ArcTrgts; }
    const std::vector<Scope_t>& GetScopes() const { return Scopes; }
  };

  explicit ParseTreeAnalyzer(const parser::CookedSource& Src) : EH{Src} {}
  ~ParseTreeAnalyzer() = default;
  ParseTreeAnalyzer(ParseTreeAnalyzer&&) = default;
  ParseTreeAnalyzer() = delete;
  ParseTreeAnalyzer(const ParseTreeAnalyzer&) = delete;
  ParseTreeAnalyzer& operator=(const ParseTreeAnalyzer&) = delete;

  // Default Pre() and Post()
  template<typename A> constexpr bool Pre(const A&) { return true; }
  template<typename A> constexpr void Post(const A&) {}

  // Specializations of Pre() and Post()

  /// \brief Generic handling of all statements
  template<typename A> bool Pre(const parser::Statement<A>& Stmt) {
    Index = Stmt.source;
    if (Stmt.label.has_value())
      AddTrgt(Stmt.label.value(), ConsTrgtFlags(Stmt));
    return true;
  }

  //  Inclusive scopes (see 11.1.1)
  bool Pre(const parser::ProgramUnit&) { return PushNewScope(); }
  bool Pre(const parser::AssociateConstruct& A) { return PushName(A); }
  bool Pre(const parser::BlockConstruct& Blk) { return PushName(Blk); }
  bool Pre(const parser::ChangeTeamConstruct& Ctm) { return PushName(Ctm); }
  bool Pre(const parser::CriticalConstruct& Crit) { return PushName(Crit); }
  bool Pre(const parser::DoConstruct& Do) { return PushName(Do); }
  bool Pre(const parser::IfConstruct& If) { return PushName(If); }
  bool Pre(const parser::IfConstruct::ElseIfBlock&) { return SwScope(); }
  bool Pre(const parser::IfConstruct::ElseBlock&) { return SwScope(); }
  bool Pre(const parser::CaseConstruct& Case) { return PushName(Case); }
  bool Pre(const parser::CaseConstruct::Case&) { return SwScope(); }
  bool Pre(const parser::SelectRankConstruct& SRk) { return PushName(SRk); }
  bool Pre(const parser::SelectRankConstruct::RankCase&) { return SwScope(); }
  bool Pre(const parser::SelectTypeConstruct& STy) { return PushName(STy); }
  bool Pre(const parser::SelectTypeConstruct::TypeCase&) { return SwScope(); }
  bool Pre(const parser::WhereConstruct& W) { return PushNonBlockName(W); }
  bool Pre(const parser::ForallConstruct& F) { return PushNonBlockName(F); }

  void Post(const parser::ProgramUnit&) { PopScope(); }
  void Post(const parser::AssociateConstruct& A) { PopName(A); }
  void Post(const parser::BlockConstruct& Blk) { PopName(Blk); }
  void Post(const parser::ChangeTeamConstruct& Ctm) { PopName(Ctm); }
  void Post(const parser::CriticalConstruct& Crit) { PopName(Crit); }
  void Post(const parser::DoConstruct& Do) { PopName(Do); }
  void Post(const parser::IfConstruct& If) { PopName(If); }
  void Post(const parser::CaseConstruct& Case) { PopName(Case); }
  void Post(const parser::SelectRankConstruct& SelRk) { PopName(SelRk); }
  void Post(const parser::SelectTypeConstruct& SelTy) { PopName(SelTy); }

  //  Named constructs without block scope
  void Post(const parser::WhereConstruct& W) { PopNonBlockConstructName(W); }
  void Post(const parser::ForallConstruct& F) { PopNonBlockConstructName(F); }

  //  Statements with label references
  void Post(const parser::LabelDoStmt& Do) { AddDoBase(std::get<1>(Do.t)); }
  void Post(const parser::GotoStmt& Goto) { AddBase(Goto.v); }
  void Post(const parser::ComputedGotoStmt& C) { AddBase(std::get<0>(C.t)); }
  void Post(const parser::ArithmeticIfStmt& AIf) {
    AddBase(std::get<1>(AIf.t));
    AddBase(std::get<2>(AIf.t));
    AddBase(std::get<3>(AIf.t));
  }
  void Post(const parser::AssignStmt& Assn) { AddBase(std::get<0>(Assn.t)); }
  void Post(const parser::AssignedGotoStmt& A) { AddBase(std::get<1>(A.t)); }
  void Post(const parser::AltReturnSpec& ARS) { AddBase(ARS.v); }

  void Post(const parser::ErrLabel& Err) { AddBase(Err.v); }
  void Post(const parser::EndLabel& End) { AddBase(End.v); }
  void Post(const parser::EorLabel& Eor) { AddBase(Eor.v); }
  void Post(const parser::Format& Fmt) {
    // BUG: the label is saved as an IntLiteralConstant rather than a Label
#if 0
    if (const auto* P{std::get_if<parser::Label>(&Fmt.u)})
      AddFmtBase(*P);
#else
    // FIXME: this is wrong, but extracts the label's value
    if (const auto* P{std::get_if<0>(&Fmt.u)}) {
      parser::Label L{std::get<0>(std::get<parser::IntLiteralConstant>(std::get<parser::LiteralConstant>((*P->thing).u).u).t)};
      AddFmtBase(L);
    }
#endif
  }
  void Post(const parser::CycleStmt& Cycle) {
    if (Cycle.v.has_value())
      CheckLabelContext("CYCLE", Cycle.v.value().ToString());
  }
  void Post(const parser::ExitStmt& Exit) {
    if (Exit.v.has_value())
      CheckLabelContext("EXIT", Exit.v.value().ToString());
  }

  // Getters for the results
  const std::vector<UnitAnalysis>& GetProgramUnits() const { return PUnits; }
  ErrorHandler& GetEH() { return EH; }
  bool HasNoErrors() const { return NoErrors; }

private:
  bool PushScope() {
    PUnits.back().Scopes.push_back(CurrScope);
    CurrScope = PUnits.back().Scopes.size() - 1;
    return true;
  }
  bool PushNewScope() {
    PUnits.emplace_back(UnitAnalysis{});
    return PushScope();
  }
  void PopScope() { CurrScope = PUnits.back().Scopes[CurrScope]; }
  bool SwScope() { PopScope(); return PushScope(); }

  template<typename A> bool PushName(const A& X) {
    const auto& OptName{std::get<0>(std::get<0>(X.t).statement.t)};
    if (OptName.has_value())
      Names.push_back(OptName.value().ToString());
    return PushScope();
  }
  bool PushName(const parser::BlockConstruct& Blk) {
    const auto& OptName{std::get<0>(Blk.t).statement.v};
    if (OptName.has_value())
      Names.push_back(OptName.value().ToString());
    return PushScope();
  }
  template<typename A> bool PushNonBlockName(const A& X) {
    const auto& OptName{std::get<0>(std::get<0>(X.t).statement.t)};
    if (OptName.has_value())
      Names.push_back(OptName.value().ToString());
    return true;
  }

  template<typename A> void PopNonBlockConstructName(const A& X) {
    CheckName(X); SelectivePopBack(X);
  }

  template<typename A> void SelectivePopBack(const A& X) {
    const auto& OptName{std::get<0>(std::get<0>(X.t).statement.t)};
    if (OptName.has_value())
      Names.pop_back();
  }
  void SelectivePopBack(const parser::BlockConstruct& Blk) {
    const auto& OptName{std::get<0>(Blk.t).statement.v};
    if (OptName.has_value())
      Names.pop_back();
  }

  /// \brief Check constraints and pop scope
  template<typename A> void PopName(const A& V) {
    CheckName(V); PopScope(); SelectivePopBack(V);
  }

  /// \brief Check <i>case-construct-name</i> and pop the scope
  /// Constraint C1144 - opening and ending name must match if present, and
  /// <i>case-stmt</i> must either match or be unnamed
  void PopName(const parser::CaseConstruct& Case) {
    CheckName(Case, "CASE"); PopScope(); SelectivePopBack(Case);
  }

  /// \brief Check <i>select-rank-construct-name</i> and pop the scope
  /// Constraints C1154, C1156 - opening and ending name must match if present,
  /// and <i>select-rank-case-stmt</i> must either match or be unnamed
  void PopName(const parser::SelectRankConstruct& SelRk) {
    CheckName(SelRk, "RANK","RANK "); PopScope(); SelectivePopBack(SelRk);
  }

  /// \brief Check <i>select-construct-name</i> and pop the scope
  /// Constraint C1165 - opening and ending name must match if present, and
  /// <i>type-guard-stmt</i> must either match or be unnamed
  void PopName(const parser::SelectTypeConstruct& SelTy) {
    CheckName(SelTy, "TYPE", "TYPE "); PopScope(); SelectivePopBack(SelTy);
  }

  // -----------------------------------------------
  // CheckName - check constraints on construct-name
  // Case 1: construct name must be absent or specified & identical on END

  /// \brief Check <i>associate-construct-name</i>, constraint C1106
  void CheckName(const parser::AssociateConstruct& A) { ChkNm(A, "ASSOCIATE"); }
  /// \brief Check <i>critical-construct-name</i>, constraint C1117
  void CheckName(const parser::CriticalConstruct& C) { ChkNm(C, "CRITICAL"); }
  /// \brief Check <i>do-construct-name</i>, constraint C1131
  void CheckName(const parser::DoConstruct& Do) { ChkNm(Do, "DO"); }
  /// \brief Check <i>forall-construct-name</i>, constraint C1035
  void CheckName(const parser::ForallConstruct& F) { ChkNm(F, "FORALL"); }
  /// \brief Common code for ASSOCIATE, CRITICAL, DO, and FORALL
  template<typename A> void ChkNm(const A& V, const char *const Con) {
    if (!BothEqOrNone(std::get<0>(std::get<0>(V.t).statement.t),
		      std::get<2>(V.t).statement.v)) {
      EH.Report(Index, "%s construct name mismatch"_err_en_US, Con);
      NoErrors = false;
    }
  }
  
  /// \brief Check <i>do-construct-name</i>, constraint C1109
  void CheckName(const parser::BlockConstruct& B) {
    if (!BothEqOrNone(std::get<0>(B.t).statement.v,
		      std::get<3>(B.t).statement.v)) {
      EH.Report(Index, "BLOCK construct name mismatch"_err_en_US);
      NoErrors = false;
    }
  }
  /// \brief Check <i>team-cosntruct-name</i>, constraint C1112
  void CheckName(const parser::ChangeTeamConstruct& C) {
    if (!BothEqOrNone(std::get<0>(std::get<0>(C.t).statement.t),
		      std::get<1>(std::get<2>(C.t).statement.t))) {
      EH.Report(Index, "CHANGE TEAM construct name mismatch"_err_en_US);
      NoErrors = false;
    }
  }

  // -----------------------------------------------
  // Case 2: same as case 1, but subblock statement construct-names are 
  // optional but if they are specified their values must be identical

  /// \brief Check <i>if-construct-name</i>
  /// Constraint C1142 - opening and ending name must match if present, and
  /// <i>else-if-stmt</i> and <i>else-stmt</i> must either match or be unnamed
  void CheckName(const parser::IfConstruct& If) {
    const auto& Name{std::get<0>(std::get<0>(If.t).statement.t)};
    if (!BothEqOrNone(Name, std::get<4>(If.t).statement.v)) {
      EH.Report(Index, "IF construct name mismatch"_err_en_US);
      NoErrors = false;
    }
    for (const auto& ElseIfBlock : std::get<2>(If.t)) {
      const auto& E{std::get<0>(ElseIfBlock.t).statement.t};
      if (!PresentAndEq(std::get<1>(E), Name)) {
	EH.Report(Index, "ELSE IF statement name mismatch"_err_en_US);
	NoErrors = false;
      }
    }
    if (std::get<3>(If.t).has_value()) {
      const auto& E{std::get<3>(If.t).value().t};
      if (!PresentAndEq(std::get<0>(E).statement.v, Name)) {
	EH.Report(Index, "ELSE statement name mismatch"_err_en_US);
	NoErrors = false;
      }
    }
  }
  /// \brief Common code for SELECT CASE, SELECT RANK, and SELECT TYPE
  template<typename A> void CheckName(const A& Case, const char *const Sel1,
				      const char *const Sel2 = "") {
    const auto& Name{std::get<0>(std::get<0>(Case.t).statement.t)};
    if (!BothEqOrNone(Name, std::get<2>(Case.t).statement.v)) {
      EH.Report(Index, "SELECT %s construct name mismatch"_err_en_US, Sel1);
      NoErrors = false;
    }
    for (const auto& CS : std::get<1>(Case.t))
      if (!PresentAndEq(std::get<1>(std::get<0>(CS.t).statement.t), Name)) {
	EH.Report(Index, "%sCASE statement name mismatch"_err_en_US, Sel2);
	NoErrors = false;
      }
  }

  /// \brief Check <i>where-construct-name</i>
  /// Constraint C1033 - opening and ending name must match if present, and
  /// <i>masked-elsewhere-stmt</i> and <i>elsewhere-stmt</i> either match
  /// or be unnamed
  void CheckName(const parser::WhereConstruct& Where) {
    const auto& Name{std::get<0>(std::get<0>(Where.t).statement.t)};
    if (!BothEqOrNone(Name, std::get<4>(Where.t).statement.v)) {
      EH.Report(Index, "WHERE construct name mismatch"_err_en_US);
      NoErrors = false;
    }
    for (const auto& W : std::get<2>(Where.t))
      if (!PresentAndEq(std::get<1>(std::get<0>(W.t).statement.t), Name)) {
	EH.Report(Index,
		  "ELSEWHERE (<mask>) statement name mismatch"_err_en_US);
	NoErrors = false;
      }
    if (std::get<3>(Where.t).has_value()) {
      const auto& E{std::get<3>(Where.t).value().t};
      if (!PresentAndEq(std::get<0>(E).statement.v, Name)) {
	EH.Report(Index, "ELSEWHERE statement name mismatch"_err_en_US);
	NoErrors = false;
      }
    }
  }

  /// \brief Check constraint <i>construct-name</i> in scope (C1134 and C1166)
  /// \param SStr  a string to specify the statement, \c CYCLE or \c EXIT
  /// \param Label the name used by the \c CYCLE or \c EXIT
  template<typename A> void CheckLabelContext(const char* const SStr,
					      const A& Name) {
    auto E{Names.crend()};
    for (auto I{Names.crbegin()}; I != E; ++I) {
      if (*I == Name)
	return;
    }
    EH.Report(Index, "%s construct-name '%s' is not in scope"_err_en_US,
	      SStr, Name.c_str());
    NoErrors = false;
  }

  /// \brief Check label range
  /// Constraint per section 6.2.5, paragraph 2
  void LabelInRange(parser::Label Label) {
    if ((Label < 1) || (Label > 99999)) {
      // this is an error: labels must have a value 1 to 99999, inclusive
      EH.Report(Index, "label '%lu' is out of range"_err_en_US, Label);
      NoErrors = false;
    }
  }

  /// \brief Add a labeled statement (label must be distinct)
  /// Constraint per section 6.2.5., paragraph 2
  void AddTrgt(parser::Label Label, unsigned Flags) {
    LabelInRange(Label);
    const auto Pair{PUnits.back().ArcTrgts.insert({Label,
	    {CurrScope, Index, Flags}})};
    if (!Pair.second) {
      // this is an error: labels must be pairwise distinct
      EH.Report(Index, "label '%lu' is not distinct"_err_en_US, Label);
      NoErrors = false;
    }
    // Don't enforce a limit to the cardinality of labels
  }

  /// \brief Reference to a labeled statement from a DO statement
  void AddDoBase(parser::Label Label) {
    LabelInRange(Label);
    PUnits.back().DoArcBases.push_back({Label, CurrScope, Index});
  }

  /// \brief Reference to a labeled FORMAT statement
  void AddFmtBase(parser::Label Label) {
    LabelInRange(Label);
    PUnits.back().FmtArcBases.push_back({Label, CurrScope, Index});
  }

  /// \brief Reference to a labeled statement as a (possible) branch
  void AddBase(parser::Label Label) {
    LabelInRange(Label);
    PUnits.back().ArcBases.push_back({Label, CurrScope, Index});
  }

  /// \brief References to labeled statements as (possible) branches
  void AddBase(const std::list<parser::Label>& Labels) {
    for (const parser::Label& L : Labels)
      AddBase(L);
  }

  std::vector<UnitAnalysis> PUnits; ///< results for each program unit
  ErrorHandler EH;		///< error handler, collects messages
  Index_t Index{nullptr};	///< current location in parse tree
  Scope_t CurrScope{0};		///< current scope in the model
  bool NoErrors{true};		///< no semantic errors found?
  std::vector<std::string> Names;
};

template<typename A, typename B>
bool InInclusiveScope(const A& Scopes, B Tl, const B& Hd) {
  assert(Hd > 0);
  assert(Tl > 0);
  while (Tl && (Tl != Hd))
    Tl = Scopes[Tl];
  return Tl == Hd;
}

ParseTreeAnalyzer LabelAnalysis(const ParseTree_t& ParseTree,
				const CookedSource_t& Source) {
  ParseTreeAnalyzer Analysis{Source};
  Walk(ParseTree, Analysis);
  return Analysis;
}

template<typename A, typename B>
inline bool InBody(const A& CP, const B& Pair) {
  assert(Pair.first.begin() < Pair.second.begin());
  return (CP.begin() >= Pair.first.begin()) &&
    (CP.begin() < Pair.second.end());
}

template<typename A, typename B>
LblStmt_t GetLabel(const A& Labels, const B& Label) {
  const auto Iter{Labels.find(Label)};
  if (Iter == Labels.cend())
    return {0, 0, 0};
  return Iter->second;
}

/// \brief Check branches into a <i>label-do-stmt</i>
/// Relates to 11.1.7.3, loop activation
template<typename A, typename B, typename C, typename D>
inline bool CheckBranchesIntoDoBody(const A& Branches, const B& Labels,
				    const C& Scopes, const D& LoopBodies,
				    ErrorHandler& EH) {
  auto NoErrors{true};
  for (const auto Branch : Branches) {
    const auto& Label{std::get<0>(Branch)};
    auto Trgt{GetLabel(Labels, Label)};
    if (!std::get<0>(Trgt))
      continue;
    const auto& FmIdx{std::get<2>(Branch)};
    const auto& ToIdx{std::get<1>(Trgt)};
    for (const auto Body : LoopBodies) {
      if (!InBody(FmIdx, Body) && InBody(ToIdx, Body)) {
	// this is an error: branch into labeled DO body
	if (StrictF18) {
	  EH.Report(FmIdx, "branch into '%s' from another scope"_err_en_US,
		    Body.first.ToString().c_str());
	  NoErrors = false;
	} else {
	  EH.Report(FmIdx, "branch into '%s' from another scope"_en_US,
		    Body.first.ToString().c_str());
	}
      }
    }
  }
  return NoErrors;
}

/// \brief Check that DO loops properly nest
template<typename A>
inline bool CheckDoNesting(const A& LoopBodies, ErrorHandler& EH) {
  auto NoErrors{true};
  auto E{LoopBodies.cend()};
  for (auto I1{LoopBodies.cbegin()}; I1 != E; ++I1) {
    const auto& L1{*I1};
    for (auto I2{I1 + 1}; I2 != E; ++I2) {
      const auto& L2{*I2};
      assert(L1.first.begin() != L2.first.begin());
      if ((L2.first.begin() < L1.second.end()) &&
	  (L1.second.begin() < L2.second.begin())) {
	// this is an error: DOs do not properly nest
	EH.Report(L2.second, "'%s' doesn't properly nest"_err_en_US,
		  L1.first.ToString().c_str());
	NoErrors = false;
      }
    }
  }
  return NoErrors;
}

/// \brief Advance \p Pos past any label and whitespace
/// Want the statement without its label for error messages, range checking
template<typename A> inline A SkipLabel(const A& Pos) {
  const long Max{Pos.end() - Pos.begin()};
  if (Max && (Pos[0] >= '0') && (Pos[0] <= '9')) {
    long i{1l};
    for (;(i < Max) && std::isdigit(Pos[i]); ++i);
    for (;(i < Max) && std::isspace(Pos[i]); ++i);
    return Index_t{Pos.begin() + i, Pos.end()};
  }
  return Pos;
}

/// \brief Check constraints on <i>label-do-stmt</i>
template<typename A, typename B, typename C>
inline bool CheckLabelDoConstraints(const A& Dos, const A& Branches,
				    const B& Labels, const C& Scopes,
				    ErrorHandler& EH) {
  auto NoErrors{true};
  IndexList LoopBodies;
  for (const auto Stmt : Dos) {
    const auto& Label{std::get<0>(Stmt)};
    const auto& Scope{std::get<1>(Stmt)};
    const auto& Index{std::get<2>(Stmt)};
    auto Trgt{GetLabel(Labels, Label)};
    if (!std::get<0>(Trgt)) {
      // C1133: this is an error: label not found
      EH.Report(Index, "label '%lu' cannot be found"_err_en_US, Label);
      NoErrors = false;
      continue;
    }
    if (std::get<1>(Trgt).begin() < Index.begin()) {
      // R1119: this is an error: label does not follow DO
      EH.Report(Index, "label '%lu' doesn't lexically follow DO stmt"_err_en_US,
		Label);
      NoErrors = false;
      continue;
    }
    if (!InInclusiveScope(Scopes, Scope, std::get<0>(Trgt))) {
      // C1133: this is an error: label is not in scope
      if (StrictF18) {
	EH.Report(Index, "label '%lu' is not in scope"_err_en_US, Label);
	NoErrors = false;
      } else {
	EH.Report(Index, "label '%lu' is not in scope"_en_US, Label);
      }
      continue;
    }
    if (!(std::get<2>(Trgt) & DO_TERM_FLAG)) {
      EH.Report(std::get<Index_t>(Trgt),
		"'%lu' invalid DO terminal statement"_err_en_US, Label);
      NoErrors = false;
    }
    // save the loop body marks
    LoopBodies.push_back({SkipLabel(Index), std::get<1>(Trgt)});
  }
  
  if (NoErrors) {
    NoErrors =
      // check that nothing jumps into the block
      CheckBranchesIntoDoBody(Branches, Labels, Scopes, LoopBodies, EH) &
      // check that do loops properly nest
      CheckDoNesting(LoopBodies, EH);
  }
  return NoErrors;
}

/// \brief General constraint, control transfers within inclusive scope
/// See, for example, section 6.2.5.
template<typename A, typename B, typename C>
bool CheckScopeConstraints(const A& Stmts, const B& Labels,
			   const C& Scopes, ErrorHandler& EH) {
  auto NoErrors{true};
  for (const auto Stmt : Stmts) {
    const auto& Label{std::get<0>(Stmt)};
    const auto& Scope{std::get<1>(Stmt)};
    const auto& Index{std::get<2>(Stmt)};
    auto Trgt{GetLabel(Labels, Label)};
    if (!std::get<0>(Trgt)) {
      // this is an error: label not found
      EH.Report(Index, "label '%lu' was not found"_err_en_US, Label);
      NoErrors = false;
      continue;
    }
    if (!InInclusiveScope(Scopes, Scope, std::get<0>(Trgt))) {
      // this is an error: label not in scope
      if (StrictF18) {
	EH.Report(Index, "label '%lu' is not in scope"_err_en_US, Label);
	NoErrors = false;
      } else {
	EH.Report(Index, "label '%lu' is not in scope"_en_US, Label);
      }
    }
  }
  return NoErrors;
}

template<typename A, typename B>
inline bool CheckBranchTargetConstraints(const A& Stmts, const B& Labels,
					 ErrorHandler& EH) {
  auto NoErrors{true};
  for (const auto Stmt : Stmts) {
    const auto& Label{std::get<0>(Stmt)};
    auto Trgt{GetLabel(Labels, Label)};
    if (!std::get<0>(Trgt))
      continue;
    if (!(std::get<2>(Trgt) & BRANCH_TARGET_FLAG)) {
      // this is an error: label statement is not a branch target
      EH.Report(std::get<Index_t>(Trgt), "'%lu' not a branch target"_err_en_US,
		Label);
      NoErrors = false;
    }
  }
  return NoErrors;
}

/// \brief Validate the constraints on branches
/// \param Analysis  the analysis result
template<typename A, typename B, typename C>
inline bool CheckBranchConstraints(const A& Branches, const B& Labels,
				   const C& Scopes, ErrorHandler& EH) {
  return CheckScopeConstraints(Branches, Labels, Scopes, EH) &
    CheckBranchTargetConstraints(Branches, Labels, EH);
}

template<typename A, typename B>
inline bool CheckDataXferTargetConstraints(const A& Stmts, const B& Labels,
					   ErrorHandler& EH) {
  auto NoErrors{true};
  for (const auto Stmt : Stmts) {
    const auto& Label{std::get<0>(Stmt)};
    auto Trgt{GetLabel(Labels, Label)};
    if (!std::get<0>(Trgt))
      continue;
    if (!(std::get<2>(Trgt) & FORMAT_STMT_FLAG)) {
      // this is an error: label not a FORMAT
      EH.Report(std::get<Index_t>(Trgt), "'%lu' not a FORMAT"_err_en_US, Label);
      NoErrors = false;
    }
  }
  return NoErrors;
}

/// \brief Validate that data transfers reference FORMATs in scope
/// \param Analysis  the analysis result
/// These label uses are disjoint from branching (control flow)
template<typename A, typename B, typename C>
inline bool CheckDataTransferConstraints(const A& DataXfers, const B& Labels,
					 const C& Scopes, ErrorHandler& EH) {
  return CheckScopeConstraints(DataXfers, Labels, Scopes, EH) &
    CheckDataXferTargetConstraints(DataXfers, Labels, EH);
}

/// \brief Validate label related constraints on the parse tree
/// \param Analysis  the analysis results as run of the parse tree
/// \param EH        the error handler
/// \return true iff all the semantics checks passed
bool CheckConstraints(ParseTreeAnalyzer&& Analysis) {
  auto result{Analysis.HasNoErrors()};
  auto& EH{Analysis.GetEH()};
  for (const auto& A : Analysis.GetProgramUnits()) {
    const auto& Dos{A.GetLabelDos()};
    const auto& Branches{A.GetBranches()};
    const auto& DataXfers{A.GetDataXfers()};
    const auto& Labels{A.GetLabels()};
    const auto& Scopes{A.GetScopes()};
    result &= CheckLabelDoConstraints(Dos, Branches, Labels, Scopes, EH) &
      CheckBranchConstraints(Branches, Labels, Scopes, EH) &
      CheckDataTransferConstraints(DataXfers, Labels, Scopes, EH);
  }
  if (!EH.messages.empty())
    EH.messages.Emit(std::cerr, EH.cookedSource);
  return result;
}

} // <anonymous>

namespace Fortran::semantics {

/// \brief Check the semantics of LABELs in the program
/// \return true iff the program's use of LABELs is semantically correct
bool ValidateLabels(const parser::Program& ParseTree,
		    const parser::CookedSource& Source) {
  return CheckConstraints(LabelAnalysis(ParseTree, Source));
}

} // Fortran::semantics
