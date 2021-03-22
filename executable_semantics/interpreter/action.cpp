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
    case ActionKind::DeleteTmpAction:
      std::cout << "delete_tmp(" << act->u.delete_tmp << ")";
      break;
    case ActionKind::ExpToLValAction:
      out << "exp=>lval";
      break;
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

auto MakeExpAct(Expression* e) -> Action* {
  auto* act = new Action();
  act->tag = ActionKind::ExpressionAction;
  act->u.exp = e;
  act->pos = -1;
  return act;
}

auto MakeLvalAct(Expression* e) -> Action* {
  auto* act = new Action();
  act->tag = ActionKind::LValAction;
  act->u.exp = e;
  act->pos = -1;
  return act;
}

auto MakeStmtAct(Statement* s) -> Action* {
  auto* act = new Action();
  act->tag = ActionKind::StatementAction;
  act->u.stmt = s;
  act->pos = -1;
  return act;
}

auto MakeValAct(Value* v) -> Action* {
  auto* act = new Action();
  act->tag = ActionKind::ValAction;
  act->u.val = v;
  act->pos = -1;
  return act;
}

auto MakeExpToLvalAct() -> Action* {
  auto* act = new Action();
  act->tag = ActionKind::ExpToLValAction;
  act->pos = -1;
  return act;
}

auto MakeDeleteAct(Address a) -> Action* {
  auto* act = new Action();
  act->tag = ActionKind::DeleteTmpAction;
  act->pos = -1;
  act->u.delete_tmp = a;
  return act;
}

// Returns a new action object that is a copy of the given action.
auto CopyAction(Action* action) -> Action* {
  Action* copy = new Action();
  copy->tag = action->tag;
  copy->pos = action->pos;
  copy->results = action->results;
  switch (action->tag) {
    case ActionKind::LValAction:
    case ActionKind::ExpressionAction:
      copy->u.exp = action->u.exp;
      break;
    case ActionKind::StatementAction:
      copy->u.stmt = action->u.stmt;
      break;
    case ActionKind::ValAction:
      copy->u.val = action->u.val;
      break;
    case ActionKind::ExpToLValAction:
      // nothing to copy
      break;
    case ActionKind::DeleteTmpAction:
      copy->u.delete_tmp = action->u.delete_tmp;
      break;
  }
  return copy;
}

}  // namespace Carbon
