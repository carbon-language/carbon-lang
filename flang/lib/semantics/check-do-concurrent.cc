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

#include "check-do-concurrent.h"
#include "attr.h"
#include "scope.h"
#include "symbol.h"
#include "type.h"
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
  template<typename T> bool Pre(const T &) { return true; }
  template<typename T> void Post(const T &) {}
  template<typename T> bool Pre(const parser::Statement<T> &statement) {
    currentStatementSourcePosition_ = statement.source;
    if (statement.label.has_value()) {
      labels_.insert(*statement.label);
    }
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
      if (name->ToString() == "move_alloc"s) {
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
        if (name->ToString() == "ieee_get_flag"s) {
          messages_.Say(currentStatementSourcePosition_,
              "IEEE_GET_FLAG not allowed in DO CONCURRENT"_err_en_US);
        } else if (name->ToString() == "ieee_set_halting_mode"s) {
          messages_.Say(currentStatementSourcePosition_,
              "IEEE_SET_HALTING_MODE not allowed in DO CONCURRENT"_err_en_US);
        } else if (name->ToString() == "ieee_get_halting_mode"s) {
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

  std::set<parser::Label> labels_;
  parser::CharBlock currentStatementSourcePosition_;
  parser::Messages &messages_;
};

class DoConcurrentLabelEnforce {
public:
  DoConcurrentLabelEnforce(
      parser::Messages &messages, std::set<parser::Label> &&labels)
    : messages_{messages}, labels_{labels} {}
  template<typename T> bool Pre(const T &) { return true; }
  template<typename T> bool Pre(const parser::Statement<T> &statement) {
    currentStatementSourcePosition_ = statement.source;
    return true;
  }
  template<typename T> void Post(const T &) {}

  // C1138: branch from within a DO CONCURRENT shall not target outside loop
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
      messages_.Say(currentStatementSourcePosition_,
          "control flow escapes from DO CONCURRENT"_err_en_US);
    }
  }

private:
  parser::Messages &messages_;
  std::set<parser::Label> labels_;
  parser::CharBlock currentStatementSourcePosition_{nullptr};
};

using CS = std::vector<Symbol>;

struct GatherSymbols {
  CS symbols;
  template<typename T> constexpr bool Pre(const T &) { return true; }
  template<typename T> constexpr void Post(const T &) {}
  void Post(const parser::Name &name) { symbols.push_back(*name.symbol); }
};

static bool IntegerVariable(const Symbol &variable) {
  return variable.GetType()->category() == semantics::DeclTypeSpec::Intrinsic &&
      variable.GetType()->intrinsicTypeSpec().category() ==
      common::TypeCategory::Integer;
}
static CS GatherAllVariableNames(
    const std::list<parser::LocalitySpec> &localitySpecs) {
  CS names;
  for (auto &ls : localitySpecs) {
    std::visit(common::visitors{[](auto &) {},
                   [&](const parser::LocalitySpec::Local &local) {
                     for (auto &v : local.v) {
                       CHECK(v.symbol);
                       names.emplace_back(*v.symbol);
                     }
                   },
                   [&](const parser::LocalitySpec::LocalInit &localInit) {
                     for (auto &v : localInit.v) {
                       CHECK(v.symbol);
                       names.emplace_back(*v.symbol);
                     }
                   },
                   [&](const parser::LocalitySpec::Shared &shared) {
                     for (auto &v : shared.v) {
                       CHECK(v.symbol);
                       names.emplace_back(*v.symbol);
                     }
                   }},
        ls.u);
  }
  return names;
}
static CS GatherNotSharedVariableNames(
    const std::list<parser::LocalitySpec> &localitySpecs) {
  CS names;
  for (auto &ls : localitySpecs) {
    std::visit(common::visitors{[](auto &) {},
                   [&](const parser::LocalitySpec::Local &local) {
                     for (auto &v : local.v) {
                       CHECK(v.symbol);
                       names.emplace_back(*v.symbol);
                     }
                   },
                   [&](const parser::LocalitySpec::LocalInit &localInit) {
                     for (auto &v : localInit.v) {
                       CHECK(v.symbol);
                       names.emplace_back(*v.symbol);
                     }
                   }},
        ls.u);
  }
  return names;
}
static CS GatherLocalVariableNames(
    const std::list<parser::LocalitySpec> &localitySpecs) {
  CS names;
  for (auto &ls : localitySpecs) {
    std::visit(common::visitors{[](auto &) {},
                   [&](const parser::LocalitySpec::Local &local) {
                     for (auto &v : local.v) {
                       CHECK(v.symbol);
                       names.emplace_back(*v.symbol);
                     }
                   }},
        ls.u);
  }
  return names;
}
static CS GatherReferencesFromExpression(const parser::Expr &expression) {
  GatherSymbols gatherSymbols;
  parser::Walk(expression, gatherSymbols);
  return gatherSymbols.symbols;
}

// Find a canonical DO CONCURRENT and enforce semantics checks on its body
class FindDoConcurrentLoops {
public:
  FindDoConcurrentLoops(parser::Messages &messages) : messages_{messages} {}
  template<typename T> constexpr bool Pre(const T &) { return true; }
  template<typename T> constexpr void Post(const T &) {}
  void Post(const parser::DoConstruct &doConstruct) {
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
        DoConcurrentLabelEnforce doConcurrentLabelEnforce{
            messages_, doConcurrentEnforcement.labels()};
        parser::Walk(
            std::get<parser::Block>(doConstruct.t), doConcurrentLabelEnforce);
        EnforceConcurrentLoopControl(*concurrent);
      } else if (auto *loopBounds{
                     std::get_if<parser::LoopBounds<parser::ScalarIntExpr>>(
                         &optionalLoopControl->u)}) {
        // C1120 - FIXME? may be checked before we get here
        auto *doVariable{loopBounds->name.thing.thing.symbol};
        CHECK(doVariable);
        currentStatementSourcePosition_ = loopBounds->name.thing.thing.source;
        if (!IntegerVariable(*doVariable)) {
          // warning only: older Fortrans allowed floating-point do-variables
          messages_.Say(currentStatementSourcePosition_,
              "do-variable must have INTEGER type"_en_US);
        }
      } else {
        // C1006 - FIXME? may be checked before we get here
        auto &logicalExpr{
            std::get<parser::ScalarLogicalExpr>(optionalLoopControl->u)
                .thing.thing};
        if (!ExpressionHasTypeCategory(
                *logicalExpr->typedExpr, common::TypeCategory::Logical)) {
          messages_.Say(currentStatementSourcePosition_,
              "DO WHERE must have LOGICAL expression"_err_en_US);
        }
      }
    }
  }

private:
  bool ExpressionHasTypeCategory(const evaluate::GenericExprWrapper &expr,
      const common::TypeCategory &type) {
    // TODO - implement
    return false;
  }
  bool InnermostEnclosingScope(const semantics::Symbol &symbol) const {
    // TODO - implement
    return false;
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
    for (auto &symbol : symbols) {
      if (!InnermostEnclosingScope(symbol)) {
        messages_.Say(currentStatementSourcePosition_,
            "variable in locality-spec must be in innermost"
            " scoping unit"_err_en_US);
        return;
      }
    }
  }
  void CheckMaskIsPure(const parser::ScalarLogicalExpr &mask) const {
    // C1121 - procedures in mask must be pure
    CS references{GatherReferencesFromExpression(*mask.thing.thing)};
    for (auto &r : references) {
      if (isProcedure(r.flags()) && !isPure(r.attrs())) {
        messages_.Say(currentStatementSourcePosition_,
            "concurrent-header mask expression cannot reference impure"
            " procedure"_err_en_US);
        return;
      }
    }
  }
  void CheckNoCollisions(const CS &containerA, const CS &containerB,
      const parser::MessageFormattedText &errorMessage) const {
    for (auto &a : containerA) {
      for (auto &b : containerB) {
        if (a == b) {
          messages_.Say(currentStatementSourcePosition_, errorMessage);
          return;
        }
      }
    }
  }
  void HasNoReferences(
      const CS &indexNames, const parser::ScalarIntExpr &expression) const {
    CS references{GatherReferencesFromExpression(*expression.thing.thing)};
    CheckNoCollisions(references, indexNames,
        "concurrent-control expression references index-name"_err_en_US);
  }
  void CheckNoDuplicates(const CS &symbols) const {
    // C1126
    CheckNoCollisions(symbols, symbols,
        "name appears more than once in concurrent-locality"_err_en_US);
  }
  void CheckMaskDoesNotReferenceLocal(
      const parser::ScalarLogicalExpr &mask, const CS &symbols) const {
    // C1129
    CheckNoCollisions(GatherReferencesFromExpression(*mask.thing.thing),
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
      // C1122 - FIXME? may be checked somewhere else before we get here
      if (!indexName.symbol) {
        continue;  // XXX - this shouldn't be needed
      }
      CHECK(indexName.symbol);
      indexNames.push_back(*indexName.symbol);
      if (!IntegerVariable(*indexName.symbol)) {
        messages_.Say(
            indexName.source, "index-name must have INTEGER type"_err_en_US);
        return;
      }
    }
    if (!indexNames.empty()) {
      for (auto &c : controls) {
        // C1123
        HasNoReferences(indexNames, std::get<1>(c.t));
        HasNoReferences(indexNames, std::get<2>(c.t));
        auto &expression{std::get<std::optional<parser::ScalarIntExpr>>(c.t)};
        if (expression) {
          HasNoReferences(indexNames, *expression);
        }
      }
    }
    auto &localitySpecs{
        std::get<std::list<parser::LocalitySpec>>(concurrent.t)};
    if (localitySpecs.empty()) {
      return;
    }
    auto variableNames{GatherAllVariableNames(localitySpecs)};
    CheckScopingConstraints(variableNames);
    // C1125
    CheckNoCollisions(indexNames, variableNames,
        "name in concurrent-locality also appears in index-names"_err_en_US);
    CheckNoDuplicates(variableNames);
    CheckZeroOrOneDefaultNone(localitySpecs);
    CheckLocalAndLocalInitAttributes(
        GatherNotSharedVariableNames(localitySpecs));
    if (mask) {
      CheckMaskDoesNotReferenceLocal(
          *mask, GatherLocalVariableNames(localitySpecs));
    }
    CheckDefaultNoneImpliesExplicitLocality(localitySpecs);
  }

  parser::Messages &messages_;
  parser::CharBlock currentStatementSourcePosition_;
};

// DO loops must be canonicalized prior to calling
void CheckDoConcurrentConstraints(
    parser::Messages &messages, const parser::Program &program) {
  FindDoConcurrentLoops findDoConcurrentLoops{messages};
  Walk(program, findDoConcurrentLoops);
}

}  // namespace Fortran::semantics
