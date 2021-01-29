// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_ACTION_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_ACTION_H_

#include <iostream>
#include <vector>

#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/statement.h"
#include "executable_semantics/interpreter/cons_list.h"
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
    Expression* exp;  // for LValAction and ExpressionAction
    Statement* stmt;
    Value* val;  // for finished actions with a value (ValAction)
    Address delete_tmp;
  } u;
  int pos;                      // position or state of the action
  std::vector<Value*> results;  // results from subexpression
};

void PrintAct(Action* act, std::ostream& out);
void PrintActList(Cons<Action*>* ls, std::ostream& out);
auto MakeExpAct(Expression* e) -> Action*;
auto MakeLvalAct(Expression* e) -> Action*;
auto MakeStmtAct(Statement* s) -> Action*;
auto MakeValAct(Value* v) -> Action*;
auto MakeExpToLvalAct() -> Action*;
auto MakeDeleteAct(Address a) -> Action*;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_ACTION_H_
