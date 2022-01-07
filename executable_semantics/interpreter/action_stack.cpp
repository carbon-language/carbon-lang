// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/action_stack.h"

#include "executable_semantics/interpreter/action.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

namespace Carbon {

void ActionStack::Print(llvm::raw_ostream& out) const {
  llvm::ListSeparator sep(" :: ");
  for (const std::unique_ptr<Action>& action : todo_) {
    out << sep << *action;
  }
}

void ActionStack::Start(std::unique_ptr<Action> action, Scope scope) {
  result_ = std::nullopt;
  todo_ = {};
  todo_.Push(std::make_unique<ScopeAction>(std::move(scope)));
  todo_.Push(std::move(action));
}

auto ActionStack::CurrentScope() const -> Scope& {
  for (const std::unique_ptr<Action>& action : todo_) {
    if (action->scope().has_value()) {
      return *action->scope();
    }
  }
  FATAL() << "No current scope";
}

void ActionStack::FinishAction() {
  std::unique_ptr<Action> act = todo_.Pop();
  switch (act->kind()) {
    case Action::Kind::ExpressionAction:
    case Action::Kind::LValAction:
    case Action::Kind::PatternAction:
      FATAL() << "This kind of action must produce a result.";
    case Action::Kind::ScopeAction:
      FATAL() << "ScopeAction at top of stack";
    case Action::Kind::StatementAction:
      PopScopes();
      CHECK(!IsEmpty());
  }
}

void ActionStack::FinishAction(Nonnull<const Value*> result) {
  std::unique_ptr<Action> act = todo_.Pop();
  switch (act->kind()) {
    case Action::Kind::StatementAction:
      FATAL() << "Statements cannot produce results.";
    case Action::Kind::ScopeAction:
      FATAL() << "ScopeAction at top of stack";
    case Action::Kind::ExpressionAction:
    case Action::Kind::LValAction:
    case Action::Kind::PatternAction:
      PopScopes();
      SetResult(result);
  }
}

void ActionStack::Spawn(std::unique_ptr<Action> child) {
  Action& action = *todo_.Top();
  action.set_pos(action.pos() + 1);
  todo_.Push(std::move(child));
}

void ActionStack::Spawn(std::unique_ptr<Action> child, Scope scope) {
  Action& action = *todo_.Top();
  action.set_pos(action.pos() + 1);
  todo_.Push(std::make_unique<ScopeAction>(std::move(scope)));
  todo_.Push(std::move(child));
}

void ActionStack::RunAgain() {
  Action& action = *todo_.Top();
  action.set_pos(action.pos() + 1);
}

void ActionStack::UnwindTo(Nonnull<const Statement*> ast_node) {
  while (true) {
    if (const auto* statement_action =
            llvm::dyn_cast<StatementAction>(todo_.Top().get());
        statement_action != nullptr &&
        &statement_action->statement() == ast_node) {
      break;
    }
    todo_.Pop();
  }
}

void ActionStack::UnwindPast(Nonnull<const Statement*> ast_node) {
  UnwindTo(ast_node);
  todo_.Pop();
  PopScopes();
}

void ActionStack::UnwindPast(Nonnull<const Statement*> ast_node,
                             Nonnull<const Value*> result) {
  UnwindPast(ast_node);
  SetResult(result);
}

void ActionStack::Resume(Nonnull<const ContinuationValue*> continuation) {
  Action& action = *todo_.Top();
  action.set_pos(action.pos() + 1);
  continuation->stack().RestoreTo(todo_);
}

static auto IsRunAction(const Action& action) -> bool {
  const auto* statement = llvm::dyn_cast<StatementAction>(&action);
  return statement != nullptr && llvm::isa<Run>(statement->statement());
}

void ActionStack::Suspend() {
  // Pause the current continuation
  todo_.Pop();
  std::vector<std::unique_ptr<Action>> paused;
  while (!IsRunAction(*todo_.Top())) {
    paused.push_back(todo_.Pop());
  }
  const auto& continuation =
      llvm::cast<const ContinuationValue>(*todo_.Top()->results()[0]);
  // Update the continuation with the paused stack.
  continuation.stack().StoreReversed(std::move(paused));
}

void ActionStack::PopScopes() {
  while (!todo_.IsEmpty() && llvm::isa<ScopeAction>(*todo_.Top())) {
    todo_.Pop();
  }
}

void ActionStack::SetResult(Nonnull<const Value*> result) {
  if (todo_.IsEmpty()) {
    result_ = result;
  } else {
    todo_.Top()->AddResult(result);
  }
}

}  // namespace Carbon
