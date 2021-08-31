// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/action.h"

#include <iterator>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/function_definition.h"
#include "executable_semantics/common/arena.h"
#include "executable_semantics/interpreter/stack.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

namespace Carbon {

using llvm::cast;

void Action::Print(llvm::raw_ostream& out) const {
  switch (Tag()) {
    case Action::Kind::LValAction:
      out << *cast<LValAction>(*this).Exp();
      break;
    case Action::Kind::ExpressionAction:
      out << *cast<ExpressionAction>(*this).Exp();
      break;
    case Action::Kind::PatternAction:
      out << *cast<PatternAction>(*this).Pat();
      break;
    case Action::Kind::StatementAction:
      cast<StatementAction>(*this).Stmt()->PrintDepth(1, out);
      break;
  }
  out << "<" << pos << ">";
  if (results.size() > 0) {
    out << "(";
    llvm::ListSeparator sep;
    for (auto& result : results) {
      out << sep;
      if (result) {
        out << *result;
      }
    }
    out << ")";
  }
}

void Action::PrintList(const Stack<Ptr<Action>>& ls, llvm::raw_ostream& out) {
  llvm::ListSeparator sep(" :: ");
  for (const auto& action : ls) {
    out << sep << *action;
  }
}

template <typename StatementType>
static auto MakeConcreteStatementAction(Ptr<const Statement> stmt)
    -> Ptr<ConcreteStatementAction<StatementType>> {
  Ptr<const StatementType> concrete_stmt(cast<const StatementType>(stmt.Get()));
  return global_arena->New<ConcreteStatementAction<StatementType>>(
      concrete_stmt);
}

auto StatementAction::Make(Ptr<const Statement> stmt) -> Ptr<StatementAction> {
  switch (stmt->Tag()) {
    case Statement::Kind::ExpressionStatement:
      return MakeConcreteStatementAction<ExpressionStatement>(stmt);
    case Statement::Kind::Assign:
      return MakeConcreteStatementAction<Assign>(stmt);
    case Statement::Kind::VariableDefinition:
      return MakeConcreteStatementAction<VariableDefinition>(stmt);
    case Statement::Kind::If:
      return MakeConcreteStatementAction<If>(stmt);
    case Statement::Kind::Return:
      return MakeConcreteStatementAction<Return>(stmt);
    case Statement::Kind::Sequence:
      return MakeConcreteStatementAction<Sequence>(stmt);
    case Statement::Kind::Block:
      return MakeConcreteStatementAction<Block>(stmt);
    case Statement::Kind::While:
      return MakeConcreteStatementAction<While>(stmt);
    case Statement::Kind::Break:
      return MakeConcreteStatementAction<Break>(stmt);
    case Statement::Kind::Continue:
      return MakeConcreteStatementAction<Continue>(stmt);
    case Statement::Kind::Match:
      return MakeConcreteStatementAction<Match>(stmt);
    case Statement::Kind::Continuation:
      return MakeConcreteStatementAction<Continuation>(stmt);
    case Statement::Kind::Run:
      return MakeConcreteStatementAction<Run>(stmt);
    case Statement::Kind::Await:
      return MakeConcreteStatementAction<Await>(stmt);
  }
}

}  // namespace Carbon
