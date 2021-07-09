// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <iterator>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/function_definition.h"
#include "executable_semantics/interpreter/interpreter.h"
#include "executable_semantics/interpreter/stack.h"
#include "executable_semantics/interpreter/typecheck.h"

namespace Carbon {

void PrintAct(Action* act, std::ostream& out) {
  switch (act->tag) {
    case ActionKind::LValAction:
    case ActionKind::ExpressionAction:
      PrintExp(act->u.exp);
      break;
    case ActionKind::StatementAction:
      PrintStatement(act->u.stmt, 1);
      break;
    case ActionKind::ValAction:
      PrintValue(act->u.val, out);
      break;
  }
  out << "<" << act->pos << ">";
  if (act->results.size() > 0) {
    out << "(";
    for (auto& result : act->results) {
      if (result) {
        PrintValue(result, out);
      }
      out << ",";
    }
    out << ")";
  }
}

void PrintActList(Stack<Action*> ls, std::ostream& out) {
  if (!ls.IsEmpty()) {
    PrintAct(ls.Pop(), out);
    if (!ls.IsEmpty()) {
      out << " :: ";
      PrintActList(ls, out);
    }
  }
}

auto MakeExpAct(const Expression* e) -> Action* {
  auto* act = new Action();
  act->tag = ActionKind::ExpressionAction;
  act->u.exp = e;
  act->pos = 0;
  return act;
}

auto MakeLvalAct(const Expression* e) -> Action* {
  auto* act = new Action();
  act->tag = ActionKind::LValAction;
  act->u.exp = e;
  act->pos = 0;
  return act;
}

auto MakeStmtAct(const Statement* s) -> Action* {
  auto* act = new Action();
  act->tag = ActionKind::StatementAction;
  act->u.stmt = s;
  act->pos = 0;
  return act;
}

auto MakeValAct(const Value* v) -> Action* {
  auto* act = new Action();
  act->tag = ActionKind::ValAction;
  act->u.val = v;
  act->pos = 0;
  return act;
}

}  // namespace Carbon
