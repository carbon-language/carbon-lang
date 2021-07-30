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

namespace Carbon {

auto Action::MakeLValAction(const Expression* e) -> Action* {
  auto* act = global_arena->New<Action>();
  act->value = LValAction({.exp = e});
  return act;
}

auto Action::MakeExpressionAction(const Expression* e) -> Action* {
  auto* act = global_arena->New<Action>();
  act->value = ExpressionAction({.exp = e});
  return act;
}

auto Action::MakeStatementAction(const Statement* s) -> Action* {
  auto* act = global_arena->New<Action>();
  act->value = StatementAction({.stmt = s});
  return act;
}

auto Action::MakeValAction(const Value* v) -> Action* {
  auto* act = global_arena->New<Action>();
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

void Action::Print(llvm::raw_ostream& out) const {
  switch (tag()) {
    case ActionKind::LValAction:
      out << *GetLValAction().exp;
      break;
    case ActionKind::ExpressionAction:
      out << *GetExpressionAction().exp;
      break;
    case ActionKind::StatementAction:
      GetStatementAction().stmt->PrintDepth(1, out);
      break;
    case ActionKind::ValAction:
      out << *GetValAction().val;
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

void Action::PrintList(const Stack<Action*>& ls, llvm::raw_ostream& out) {
  llvm::ListSeparator sep(" :: ");
  for (const auto& action : ls) {
    out << sep << *action;
  }
}

}  // namespace Carbon
