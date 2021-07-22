// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_ACTION_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_ACTION_H_

#include <vector>

#include "common/ostream.h"
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
};

struct LValAction {
  static constexpr ActionKind Kind = ActionKind::LValAction;
  const Expression* exp;
};

struct ExpressionAction {
  static constexpr ActionKind Kind = ActionKind::ExpressionAction;
  const Expression* exp;
};

struct StatementAction {
  static constexpr ActionKind Kind = ActionKind::StatementAction;
  const Statement* stmt;
};

struct ValAction {
  static constexpr ActionKind Kind = ActionKind::ValAction;
  const Value* val;
};

struct Action {
  static auto MakeLValAction(const Expression* e) -> Action*;
  static auto MakeExpressionAction(const Expression* e) -> Action*;
  static auto MakeStatementAction(const Statement* s) -> Action*;
  static auto MakeValAction(const Value* v) -> Action*;

  static void PrintList(const Stack<Action*>& ls, llvm::raw_ostream& out);

  auto GetLValAction() const -> const LValAction&;
  auto GetExpressionAction() const -> const ExpressionAction&;
  auto GetStatementAction() const -> const StatementAction&;
  auto GetValAction() const -> const ValAction&;

  void Print(llvm::raw_ostream& out) const;

  inline auto tag() const -> ActionKind {
    return std::visit([](const auto& t) { return t.Kind; }, value);
  }

  // The position or state of the action. Starts at 0 and goes up to the number
  // of subexpressions.
  //
  // pos indicates how many of the entries in the following `results` vector
  // will be filled in the next time this action is active.
  // For each i < pos, results[i] contains a pointer to a Value.
  int pos = 0;

  // Results from a subexpression.
  std::vector<const Value*> results;

 private:
  std::variant<LValAction, ExpressionAction, StatementAction, ValAction> value;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_ACTION_H_
