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
#include "../evaluate/expression.h"
#include "../evaluate/traversal.h"
#include "../parser/message.h"
#include "../parser/parse-tree-visitor.h"

namespace Fortran::semantics {

using namespace parser::literals;

// 11.1.7.5 - enforce semantics constraints on a DO CONCURRENT loop body
class DoConcurrentBodyEnforce {
public:
  DoConcurrentBodyEnforce(SemanticsContext &context) : context_{context} {}
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
            "call to impure subroutine in DO CONCURRENT not allowed"_err_en_US);
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

  void addName(const std::optional<parser::Name> &nm) {
    if (nm.has_value()) {
      names_.insert(nm.value().source);
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
        context_.Say(currentStatementSourcePosition_,
            "exit from DO CONCURRENT construct (%s)"_err_en_US,
            doConcurrentSourcePosition_);
      }
      // nesting of named constructs is assumed to have been previously checked
      // by the name/label resolution pass
    } else if (names_.find(nm.value().source) == names_.end()) {
      context_.Say(currentStatementSourcePosition_,
          "exit from DO CONCURRENT construct (%s) to construct with name '%s'"_err_en_US,
          doConcurrentSourcePosition_, nm.value().source);
    }
  }

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
  int do_depth_{0};
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
    if (const auto *expr{GetExpr(expression)}) {
      struct CollectSymbols : public virtual evaluate::VisitorBase<SymbolSet> {
        explicit CollectSymbols(int) {}
        void Handle(const Symbol *symbol) {
          if (const Symbol * root{GetAssociationRoot(*symbol)}) {
            result().insert(root);
          }
        }
      };
      return evaluate::Visitor<CollectSymbols>{0}.Traverse(*expr);
    } else {
      return {};
    }
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

}  // namespace Fortran::semantics
