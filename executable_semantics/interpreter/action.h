// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_ACTION_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_ACTION_H_

#include <iostream>
#include <vector>

#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/statement.h"
#include "executable_semantics/interpreter/stack.h"
#include "executable_semantics/interpreter/value.h"

namespace Carbon {

enum class ActionKind {
  LValAction,
  ExpressionAction,
  StatementAction,
  ValAction,
  ExpToLValAction,
  DeleteTmpAction
};

struct Action {
  ActionKind tag;
  union {
    const Expression* exp;  // for LValAction and ExpressionAction
    const Statement* stmt;
    const Value* val;  // for finished actions with a value (ValAction)
    Address delete_tmp;
  } u;
  int pos;  // position or state of the action, starts at 0 and goes up to
  // the number of subexpressions.
  // pos indicates how many of the entries in the following`results` vector
  // are filled in. For each i < pos, results[i] contains a pointer to a Value.
  std::vector<const Value*> results;  // results from subexpression
};

void PrintAct(Action* act, std::ostream& out);
void PrintActList(Stack<Action*> ls, std::ostream& out);
auto MakeExpAct(const Expression* e) -> Action*;
auto MakeLvalAct(const Expression* e) -> Action*;
auto MakeStmtAct(const Statement* s) -> Action*;
auto MakeValAct(const Value* v) -> Action*;
auto MakeExpToLvalAct() -> Action*;
auto MakeDeleteAct(Address a) -> Action*;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_ACTION_H_
