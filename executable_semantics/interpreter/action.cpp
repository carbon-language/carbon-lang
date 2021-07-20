// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/action.h"

#include <iostream>
#include <iterator>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/function_definition.h"
#include "executable_semantics/interpreter/stack.h"

namespace Carbon {

auto Action::MakeLValAction(const Expression* e) -> Action* {
  auto* act = new Action();
  act->value = LValAction({.exp = e});
  return act;
}

auto Action::MakeExpressionAction(const Expression* e) -> Action* {
  auto* act = new Action();
  act->value = ExpressionAction({.exp = e});
  return act;
}

auto Action::MakeStatementAction(const Statement* s) -> Action* {
  auto* act = new Action();
  act->value = StatementAction({.stmt = s});
  return act;
}

auto Action::MakeValAction(const Value* v) -> Action* {
  auto* act = new Action();
  act->value = ValAction({.val = v});
  return act;
}

auto Action::GetLValAction() const -> const LValAction& {
  return std::get<LValAction>(value);
}

auto Action::GetExpressionAction() const -> const ExpressionAction& {
  return std::get<ExpressionAction>(value);
}

auto Action::GetStatementAction() const -> const StatementAction& {
  return std::get<StatementAction>(value);
}

auto Action::GetValAction() const -> const ValAction& {
  return std::get<ValAction>(value);
}

void Action::Print(std::ostream& out) {
  switch (tag()) {
    case ActionKind::LValAction:
      PrintExp(GetLValAction().exp);
      break;
    case ActionKind::ExpressionAction:
      PrintExp(GetExpressionAction().exp);
      break;
    case ActionKind::StatementAction:
      PrintStatement(GetStatementAction().stmt, 1);
      break;
    case ActionKind::ValAction:
      PrintValue(GetValAction().val, out);
      break;
  }
  out << "<" << pos << ">";
  if (results.size() > 0) {
    out << "(";
    for (auto& result : results) {
      if (result) {
        PrintValue(result, out);
      }
      out << ",";
    }
    out << ")";
  }
}

void Action::PrintList(Stack<Action*> ls, std::ostream& out) {
  if (!ls.IsEmpty()) {
    ls.Pop()->Print(out);
    if (!ls.IsEmpty()) {
      out << " :: ";
      PrintList(ls, out);
    }
  }
}

}  // namespace Carbon
