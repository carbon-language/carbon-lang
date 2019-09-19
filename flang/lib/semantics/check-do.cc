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

#include "check-do.h"
#include "attr.h"
#include "scope.h"
#include "semantics.h"
#include "symbol.h"
#include "tools.h"
#include "type.h"
#include "../common/template.h"
#include "../evaluate/expression.h"
#include "../evaluate/tools.h"
#include "../parser/message.h"
#include "../parser/parse-tree-visitor.h"

namespace Fortran::semantics {

using namespace parser::literals;

// Return the (possibly null)  name of the construct
template<typename A>
static const parser::Name *MaybeGetConstructName(const A &a) {
  return common::GetPtrFromOptional(std::get<0>(std::get<0>(a.t).statement.t));
}

static const parser::Name *MaybeGetConstructName(
    const parser::BlockConstruct &blockConstruct) {
  return common::GetPtrFromOptional(
      std::get<parser::Statement<parser::BlockStmt>>(blockConstruct.t)
          .statement.v);
}

// Return the (possibly null) name of the statement
template<typename A> static const parser::Name *MaybeGetStmtName(const A &a) {
  return common::GetPtrFromOptional(std::get<0>(a.t));
}

// 11.1.7.5 - enforce semantics constraints on a DO CONCURRENT loop body
class DoConcurrentBodyEnforce {
public:
  DoConcurrentBodyEnforce(SemanticsContext &context) : context_{context} {}
  std::set<parser::Label> labels() { return labels_; }
  std::set<SourceName> names() { return names_; }
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
  bool Pre(const parser::WhereConstruct &s) {
    AddName(MaybeGetConstructName(s));
    return true;
  }

  bool Pre(const parser::ForallConstruct &s) {
    AddName(MaybeGetConstructName(s));
    return true;
  }

  bool Pre(const parser::ChangeTeamConstruct &s) {
    AddName(MaybeGetConstructName(s));
    return true;
  }

  bool Pre(const parser::CriticalConstruct &s) {
    AddName(MaybeGetConstructName(s));
    return true;
  }

  bool Pre(const parser::LabelDoStmt &s) {
    AddName(MaybeGetStmtName(s));
    return true;
  }

  bool Pre(const parser::NonLabelDoStmt &s) {
    AddName(MaybeGetStmtName(s));
    return true;
  }

  bool Pre(const parser::IfThenStmt &s) {
    AddName(MaybeGetStmtName(s));
    return true;
  }

  bool Pre(const parser::SelectCaseStmt &s) {
    AddName(MaybeGetStmtName(s));
    return true;
  }

  bool Pre(const parser::SelectRankStmt &s) {
    AddName(MaybeGetStmtName(s));
    return true;
  }

  bool Pre(const parser::SelectTypeStmt &s) {
    AddName(MaybeGetStmtName(s));
    return true;
  }

  // C1136
  void Post(const parser::ReturnStmt &) {
    context_.Say(currentStatementSourcePosition_,
        "RETURN not allowed in DO CONCURRENT"_err_en_US);
  }

  // C1137
  void NoImageControl() {
    context_.Say(currentStatementSourcePosition_,
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
      context_.Say(currentStatementSourcePosition_,
          "ALLOCATE coarray not allowed in DO CONCURRENT"_err_en_US);
    }
  }

  void Post(const parser::DeallocateStmt &) {
    if (anyObjectIsCoarray()) {
      context_.Say(currentStatementSourcePosition_,
          "DEALLOCATE coarray not allowed in DO CONCURRENT"_err_en_US);
    }
    // C1140: deallocation of polymorphic objects
    if (anyObjectIsPolymorphic()) {
      context_.Say(currentStatementSourcePosition_,
          "DEALLOCATE polymorphic object(s) not allowed"
          " in DO CONCURRENT"_err_en_US);
    }
  }

  template<typename T> void Post(const parser::Statement<T> &) {
    if (EndTDeallocatesCoarray()) {
      context_.Say(currentStatementSourcePosition_,
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
          context_.Say(currentStatementSourcePosition_,
              "call to MOVE_ALLOC intrinsic in DO CONCURRENT with coarray"
              " argument(s) not allowed"_err_en_US);
        }
      }
      // C1139: call to impure procedure
      if (name->symbol && !IsPureProcedure(*name->symbol)) {
        context_.Say(currentStatementSourcePosition_,
            "call to impure procedure in DO CONCURRENT not allowed"_err_en_US);
      }
      if (name->symbol && fromScope(*name->symbol, "ieee_exceptions"s)) {
        if (name->source == "ieee_get_flag") {
          context_.Say(currentStatementSourcePosition_,
              "IEEE_GET_FLAG not allowed in DO CONCURRENT"_err_en_US);
        } else if (name->source == "ieee_set_halting_mode") {
          context_.Say(currentStatementSourcePosition_,
              "IEEE_SET_HALTING_MODE not allowed in DO CONCURRENT"_err_en_US);
        } else if (name->source == "ieee_get_halting_mode") {
          context_.Say(currentStatementSourcePosition_,
              "IEEE_GET_HALTING_MODE not allowed in DO CONCURRENT"_err_en_US);
        }
      }
    } else {
      // C1139: this a procedure component
      auto &component{std::get<parser::ProcComponentRef>(procedureDesignator.u)
                          .v.thing.component};
      if (component.symbol && !IsPureProcedure(*component.symbol)) {
        context_.Say(currentStatementSourcePosition_,
            "call to impure procedure in DO CONCURRENT not allowed"_err_en_US);
      }
    }
  }

  // 11.1.7.5
  void Post(const parser::IoControlSpec &ioControlSpec) {
    if (auto *charExpr{
            std::get_if<parser::IoControlSpec::CharExpr>(&ioControlSpec.u)}) {
      if (std::get<parser::IoControlSpec::CharExpr::Kind>(charExpr->t) ==
          parser::IoControlSpec::CharExpr::Kind::Advance) {
        context_.Say(currentStatementSourcePosition_,
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
        symbol.GetUltimate().owner().GetName().value().ToString() ==
            moduleName) {
      return true;
    }
    return false;
  }

  void AddName(const parser::Name *nm) {
    if (nm) {
      names_.insert(nm->source);
    }
  }

  std::set<parser::CharBlock> names_;
  std::set<parser::Label> labels_;
  parser::CharBlock currentStatementSourcePosition_;
  SemanticsContext &context_;
};  // class DoConcurrentBodyEnforce

class DoConcurrentLabelEnforce {
public:
  DoConcurrentLabelEnforce(SemanticsContext &context,
      std::set<parser::Label> &&labels, std::set<parser::CharBlock> &&names,
      parser::CharBlock doConcurrentSourcePosition)
    : context_{context}, labels_{labels}, names_{names},
      doConcurrentSourcePosition_{doConcurrentSourcePosition} {}
  template<typename T> bool Pre(const T &) { return true; }
  template<typename T> bool Pre(const parser::Statement<T> &statement) {
    currentStatementSourcePosition_ = statement.source;
    return true;
  }

  template<typename T> void Post(const T &) {}

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

  void checkLabelUse(const parser::Label &labelUsed) {
    if (labels_.find(labelUsed) == labels_.end()) {
      context_.Say(currentStatementSourcePosition_,
          "control flow escapes from DO CONCURRENT"_err_en_US);
    }
  }

private:
  SemanticsContext &context_;
  std::set<parser::Label> labels_;
  std::set<parser::CharBlock> names_;
  parser::CharBlock currentStatementSourcePosition_{nullptr};
  parser::CharBlock doConcurrentSourcePosition_{nullptr};
};  // class DoConcurrentLabelEnforce

// Class for enforcing C1130
class DoConcurrentVariableEnforce {
public:
  DoConcurrentVariableEnforce(
      SemanticsContext &context, parser::CharBlock doConcurrentSourcePosition)
    : context_{context},
      doConcurrentSourcePosition_{doConcurrentSourcePosition},
      blockScope_{context.FindScope(doConcurrentSourcePosition_)} {}

  template<typename T> bool Pre(const T &) { return true; }
  template<typename T> void Post(const T &) {}

  // Check to see if the name is a variable from an enclosing scope
  void Post(const parser::Name &name) {
    if (const Symbol * symbol{name.symbol}) {
      if (IsVariableName(*symbol)) {
        const Scope &variableScope{symbol->owner()};
        if (DoesScopeContain(&variableScope, blockScope_)) {
          context_.Say(name.source,
              "Variable '%s' from an enclosing scope referenced in a DO "
              "CONCURRENT with DEFAULT(NONE) must appear in a "
              "locality-spec"_err_en_US,
              name.source);
        }
      }
    }
  }

private:
  SemanticsContext &context_;
  parser::CharBlock doConcurrentSourcePosition_;
  const Scope &blockScope_;
};  // class DoConcurrentVariableEnforce

// Find a DO statement and enforce semantics checks on its body
class DoContext {
public:
  DoContext(SemanticsContext &context) : context_{context} {}

  void Check(const parser::DoConstruct &doConstruct) {
    if (doConstruct.IsDoConcurrent()) {
      CheckDoConcurrent(doConstruct);
      return;
    }
    if (doConstruct.IsDoNormal()) {
      CheckDoNormal(doConstruct);
      return;
    }
    // TODO: handle the other cases
  }

private:
  using Bounds = parser::LoopControl::Bounds;

  const Bounds &GetBounds(const parser::DoConstruct &doConstruct) {
    auto &loopControl{doConstruct.GetLoopControl().value()};
    return std::get<Bounds>(loopControl.u);
  }

  void SayBadDoControl(parser::CharBlock sourceLocation) {
    context_.Say(sourceLocation, "DO controls should be INTEGER"_err_en_US);
  }

  void CheckDoControl(const parser::CharBlock &sourceLocation, bool isReal) {
    const bool warn{context_.warnOnNonstandardUsage() ||
        context_.ShouldWarn(parser::LanguageFeature::RealDoControls)};
    if (isReal && !warn) {
      // No messages for the default case
    } else if (isReal && warn) {
      // TODO: Mark the following message as a warning when we have warnings
      context_.Say(sourceLocation, "DO controls should be INTEGER"_en_US);
    } else {
      SayBadDoControl(sourceLocation);
    }
  }

  void CheckDoVariable(const parser::ScalarName &scalarName) {
    const parser::CharBlock &sourceLocation{scalarName.thing.source};
    if (const Symbol * symbol{scalarName.thing.symbol}) {
      if (!IsVariableName(*symbol)) {
        context_.Say(
            sourceLocation, "DO control must be an INTEGER variable"_err_en_US);
      } else {
        const DeclTypeSpec *symType{symbol->GetType()};
        if (!symType) {
          SayBadDoControl(sourceLocation);
        } else {
          if (!symType->IsNumeric(TypeCategory::Integer)) {
            CheckDoControl(
                sourceLocation, symType->IsNumeric(TypeCategory::Real));
          }
        }
      }  // No messages for INTEGER
    }
  }

  // Semantic checks for the limit and step expressions
  void CheckDoExpression(const parser::ScalarExpr &scalarExpression) {
    if (const SomeExpr * expr{GetExpr(scalarExpression)}) {
      if (!ExprHasTypeCategory(*expr, TypeCategory::Integer)) {
        // No warnings or errors for type INTEGER
        const parser::CharBlock &loc{scalarExpression.thing.value().source};
        CheckDoControl(loc, ExprHasTypeCategory(*expr, TypeCategory::Real));
      }
    }
  }

  void CheckDoNormal(const parser::DoConstruct &doConstruct) {
    // C1120 extended by allowing REAL and DOUBLE PRECISION
    // Get the bounds, then check the variable, init, final, and step
    const Bounds &bounds{GetBounds(doConstruct)};
    CheckDoVariable(bounds.name);
    CheckDoExpression(bounds.lower);
    CheckDoExpression(bounds.upper);
    if (bounds.step.has_value()) {
      CheckDoExpression(bounds.step.value());
    }
  }

  void CheckDoConcurrent(const parser::DoConstruct &doConstruct) {
    auto &doStmt{
        std::get<parser::Statement<parser::NonLabelDoStmt>>(doConstruct.t)};
    currentStatementSourcePosition_ = doStmt.source;

    const parser::Block &block{std::get<parser::Block>(doConstruct.t)};
    DoConcurrentBodyEnforce doConcurrentBodyEnforce{context_};
    parser::Walk(block, doConcurrentBodyEnforce);

    DoConcurrentLabelEnforce doConcurrentLabelEnforce{context_,
        doConcurrentBodyEnforce.labels(), doConcurrentBodyEnforce.names(),
        currentStatementSourcePosition_};
    parser::Walk(block, doConcurrentLabelEnforce);

    const auto &loopControl{
        std::get<std::optional<parser::LoopControl>>(doStmt.statement.t)};
    const auto &concurrent{
        std::get<parser::LoopControl::Concurrent>(loopControl->u)};
    CheckConcurrentLoopControl(concurrent, block);
  }

  using SymbolSet = std::set<const Symbol *>;

  // Return a set of symbols whose names are in a Local locality-spec.  Look
  // the names up in the scope that encloses the DO construct to avoid getting
  // the local versions of them.  Then follow the host-, use-, and
  // construct-associations to get the root symbols
  SymbolSet GatherLocals(
      const std::list<parser::LocalitySpec> &localitySpecs) const {
    SymbolSet symbols;
    const Scope &parentScope{
        context_.FindScope(currentStatementSourcePosition_).parent()};
    // Loop through the LocalitySpec::Local locality-specs
    for (const auto &ls : localitySpecs) {
      if (const auto *names{std::get_if<parser::LocalitySpec::Local>(&ls.u)}) {
        // Loop through the names in the Local locality-spec getting their
        // symbols
        for (const parser::Name &name : names->v) {
          if (const Symbol * symbol{parentScope.FindSymbol(name.source)}) {
            if (const Symbol * root{GetAssociationRoot(*symbol)}) {
              symbols.insert(root);
            }
          }
        }
      }
    }
    return symbols;
  }

  static SymbolSet GatherSymbolsFromExpression(const parser::Expr &expression) {
    SymbolSet result;
    if (const auto *expr{GetExpr(expression)}) {
      for (const Symbol *symbol : evaluate::CollectSymbols(*expr)) {
        if (const Symbol * root{GetAssociationRoot(*symbol)}) {
          result.insert(root);
        }
      }
    }
    return result;
  }

  // C1121 - procedures in mask must be pure
  void CheckMaskIsPure(const parser::ScalarLogicalExpr &mask) const {
    SymbolSet references{GatherSymbolsFromExpression(mask.thing.thing.value())};
    for (const Symbol *ref : references) {
      if (IsProcedure(*ref) && !IsPureProcedure(*ref)) {
        const parser::CharBlock &name{ref->name()};
        context_
            .Say(currentStatementSourcePosition_,
                "concurrent-header mask expression cannot reference an impure"
                " procedure"_err_en_US)
            .Attach(name, "Declaration of impure procedure '%s'"_en_US, name);
        return;
      }
    }
  }

  void CheckNoCollisions(const SymbolSet &refs, const SymbolSet &uses,
      const parser::MessageFixedText &errorMessage,
      const parser::CharBlock &refPosition) const {
    for (const Symbol *ref : refs) {
      if (uses.find(ref) != uses.end()) {
        const parser::CharBlock &name{ref->name()};
        context_.Say(refPosition, errorMessage, name)
            .Attach(name, "Declaration of '%s'"_en_US, name);
        return;
      }
    }
  }

  void HasNoReferences(
      const SymbolSet &indexNames, const parser::ScalarIntExpr &expr) const {
    CheckNoCollisions(GatherSymbolsFromExpression(expr.thing.thing.value()),
        indexNames,
        "concurrent-control expression references index-name '%s'"_err_en_US,
        expr.thing.thing.value().source);
  }

  // C1129, names in local locality-specs can't be in mask expressions
  void CheckMaskDoesNotReferenceLocal(
      const parser::ScalarLogicalExpr &mask, const SymbolSet &localVars) const {
    CheckNoCollisions(GatherSymbolsFromExpression(mask.thing.thing.value()),
        localVars,
        "concurrent-header mask-expr references variable '%s'"
        " in LOCAL locality-spec"_err_en_US,
        mask.thing.thing.value().source);
  }

  // C1129, names in local locality-specs can't be in limit or step expressions
  void CheckExprDoesNotReferenceLocal(
      const parser::ScalarIntExpr &expr, const SymbolSet &localVars) const {
    CheckNoCollisions(GatherSymbolsFromExpression(expr.thing.thing.value()),
        localVars,
        "concurrent-header expression references variable '%s'"
        " in LOCAL locality-spec"_err_en_US,
        expr.thing.thing.value().source);
  }

  // C1130, default(none) locality requires names to be in locality-specs to be
  // used in the body of the DO loop
  void CheckDefaultNoneImpliesExplicitLocality(
      const std::list<parser::LocalitySpec> &localitySpecs,
      const parser::Block &block) const {
    bool hasDefaultNone{false};
    for (auto &ls : localitySpecs) {
      if (std::holds_alternative<parser::LocalitySpec::DefaultNone>(ls.u)) {
        if (hasDefaultNone) {
          // C1127, you can only have one DEFAULT(NONE)
          context_.Say(currentStatementSourcePosition_,
              "only one DEFAULT(NONE) may appear"_en_US);
          break;
        }
        hasDefaultNone = true;
      }
    }
    if (hasDefaultNone) {
      DoConcurrentVariableEnforce doConcurrentVariableEnforce{
          context_, currentStatementSourcePosition_};
      parser::Walk(block, doConcurrentVariableEnforce);
    }
  }

  // C1123, concurrent limit or step expressions can't reference index-names
  void CheckConcurrentHeader(const parser::ConcurrentHeader &header) const {
    auto &controls{std::get<std::list<parser::ConcurrentControl>>(header.t)};
    SymbolSet indexNames;
    for (const auto &c : controls) {
      const auto &indexName{std::get<parser::Name>(c.t)};
      if (indexName.symbol) {
        indexNames.insert(indexName.symbol);
      }
    }
    if (!indexNames.empty()) {
      for (const auto &c : controls) {
        HasNoReferences(indexNames, std::get<1>(c.t));
        HasNoReferences(indexNames, std::get<2>(c.t));
        if (const auto &expr{
                std::get<std::optional<parser::ScalarIntExpr>>(c.t)}) {
          HasNoReferences(indexNames, *expr);
        }
      }
    }
  }

  void CheckLocalitySpecs(const parser::LoopControl::Concurrent &concurrent,
      const parser::Block &block) const {
    const auto &header{std::get<parser::ConcurrentHeader>(concurrent.t)};
    const auto &controls{
        std::get<std::list<parser::ConcurrentControl>>(header.t)};
    const auto &localitySpecs{
        std::get<std::list<parser::LocalitySpec>>(concurrent.t)};
    if (!localitySpecs.empty()) {
      const SymbolSet &localVars{GatherLocals(localitySpecs)};
      for (const auto &c : controls) {
        CheckExprDoesNotReferenceLocal(std::get<1>(c.t), localVars);
        CheckExprDoesNotReferenceLocal(std::get<2>(c.t), localVars);
        if (const auto &expr{
                std::get<std::optional<parser::ScalarIntExpr>>(c.t)}) {
          CheckExprDoesNotReferenceLocal(*expr, localVars);
        }
      }
      if (const auto &mask{
              std::get<std::optional<parser::ScalarLogicalExpr>>(header.t)}) {
        CheckMaskDoesNotReferenceLocal(*mask, localVars);
      }
      CheckDefaultNoneImpliesExplicitLocality(localitySpecs, block);
    }
  }

  // check constraints [C1121 .. C1130]
  void CheckConcurrentLoopControl(
      const parser::LoopControl::Concurrent &concurrent,
      const parser::Block &block) const {

    const auto &header{std::get<parser::ConcurrentHeader>(concurrent.t)};
    const auto &mask{
        std::get<std::optional<parser::ScalarLogicalExpr>>(header.t)};
    if (mask.has_value()) {
      CheckMaskIsPure(*mask);
    }
    CheckConcurrentHeader(header);
    CheckLocalitySpecs(concurrent, block);
  }

  SemanticsContext &context_;
  parser::CharBlock currentStatementSourcePosition_;
};  // class DoContext

// DO loops must be canonicalized prior to calling
void DoChecker::Leave(const parser::DoConstruct &x) {
  DoContext doContext{context_};
  doContext.Check(x);
}

// Return the (possibly null) name of the ConstructNode
static const parser::Name *MaybeGetNodeName(const ConstructNode &construct) {
  return std::visit(
      [&](const auto &x) { return MaybeGetConstructName(*x); }, construct);
}

template<typename A> static parser::CharBlock GetConstructPosition(const A &a) {
  return std::get<0>(a.t).source;
}

static parser::CharBlock GetNodePosition(const ConstructNode &construct) {
  return std::visit(
      [&](const auto &x) { return GetConstructPosition(*x); }, construct);
}

void DoChecker::SayBadLeave(StmtType stmtType, const char *enclosingStmtName,
    const ConstructNode &construct) const {
  context_
      .Say("%s must not leave a %s statement"_err_en_US, EnumToString(stmtType),
          enclosingStmtName)
      .Attach(GetNodePosition(construct), "The construct that was left"_en_US);
}

static const parser::DoConstruct *MaybeGetDoConstruct(
    const ConstructNode &construct) {
  if (const auto *doNode{
          std::get_if<const parser::DoConstruct *>(&construct)}) {
    return *doNode;
  } else {
    return nullptr;
  }
}

static bool ConstructIsDoConcurrent(const ConstructNode &construct) {
  const parser::DoConstruct *doConstruct{MaybeGetDoConstruct(construct)};
  return doConstruct && doConstruct->IsDoConcurrent();
}

// Check that CYCLE and EXIT statements do not cause flow of control to
// leave DO CONCURRENT, CRITICAL, or CHANGE TEAM constructs.
void DoChecker::CheckForBadLeave(
    StmtType stmtType, const ConstructNode &construct) const {
  std::visit(
      common::visitors{
          [&](const parser::DoConstruct *doConstructPtr) {
            if (doConstructPtr->IsDoConcurrent()) {
              // C1135 and C1167
              SayBadLeave(stmtType, "DO CONCURRENT", construct);
            }
          },
          [&](const parser::CriticalConstruct *) {
            // C1135 and C1168
            SayBadLeave(stmtType, "CRITICAL", construct);
          },
          [&](const parser::ChangeTeamConstruct *) {
            // C1135 and C1168
            SayBadLeave(stmtType, "CHANGE TEAM", construct);
          },
          [](const auto *) {},
      },
      construct);
}

static bool StmtMatchesConstruct(const parser::Name *stmtName,
    StmtType stmtType, const parser::Name *constructName,
    const ConstructNode &construct) {
  bool inDoConstruct{MaybeGetDoConstruct(construct) != nullptr};
  if (stmtName == nullptr) {
    return inDoConstruct;  // Unlabeled statements match all DO constructs
  } else if (constructName && constructName->source == stmtName->source) {
    return stmtType == StmtType::EXIT || inDoConstruct;
  } else {
    return false;
  }
}

// C1167 Can't EXIT from a DO CONCURRENT
void DoChecker::CheckDoConcurrentExit(
    StmtType stmtType, const ConstructNode &construct) const {
  if (stmtType == StmtType::EXIT && ConstructIsDoConcurrent(construct)) {
    SayBadLeave(StmtType::EXIT, "DO CONCURRENT", construct);
  }
}

// Check nesting violations for a CYCLE or EXIT statement.  Loop up the nesting
// levels looking for a construct that matches the CYCLE or EXIT statment.  At
// every construct, check for a violation.  If we find a match without finding
// a violation, the check is complete.
void DoChecker::CheckNesting(
    StmtType stmtType, const parser::Name *stmtName) const {
  const ConstructStack &stack{context_.constructStack()};
  for (auto iter{stack.cend()}; iter-- != stack.cbegin();) {
    const ConstructNode &construct{*iter};
    const parser::Name *constructName{MaybeGetNodeName(construct)};
    if (StmtMatchesConstruct(stmtName, stmtType, constructName, construct)) {
      CheckDoConcurrentExit(stmtType, construct);
      return;  // We got a match, so we're finished checking
    }
    CheckForBadLeave(stmtType, construct);
  }

  // We haven't found a match in the enclosing constructs
  if (stmtType == StmtType::EXIT) {
    context_.Say("No matching construct for EXIT statement"_err_en_US);
  } else {
    context_.Say("No matching DO construct for CYCLE statement"_err_en_US);
  }
}

// C1135
void DoChecker::Enter(const parser::CycleStmt &cycleStmt) {
  CheckNesting(StmtType::CYCLE, common::GetPtrFromOptional(cycleStmt.v));
}

// C1167 and C1168
void DoChecker::Enter(const parser::ExitStmt &exitStmt) {
  CheckNesting(StmtType::EXIT, common::GetPtrFromOptional(exitStmt.v));
}

}  // namespace Fortran::semantics
