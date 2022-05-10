// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/action_stack.h"

#include "explorer/interpreter/action.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"

namespace Carbon {

void ActionStack::Print(llvm::raw_ostream& out) const {
  llvm::ListSeparator sep(" :: ");
  for (const std::unique_ptr<Action>& action : todo_) {
    out << sep << *action;
  }
}

void ActionStack::PrintScopes(llvm::raw_ostream& out) const {
  llvm::ListSeparator sep(" :: ");
  for (const std::unique_ptr<Action>& action : todo_) {
    if (action->scope().has_value()) {
      out << sep << *action->scope();
    }
  }
  if (globals_.has_value()) {
    out << sep << *globals_;
  }
  // TODO: should we print constants as well?
}

void ActionStack::Start(std::unique_ptr<Action> action) {
  result_ = std::nullopt;
  CHECK(todo_.IsEmpty());
  todo_.Push(std::move(action));
}

void ActionStack::Initialize(ValueNodeView value_node,
                             Nonnull<const Value*> value) {
  for (const std::unique_ptr<Action>& action : todo_) {
    if (action->scope().has_value()) {
      action->scope()->Initialize(value_node, value);
      return;
    }
  }
  globals_->Initialize(value_node, value);
}

auto ActionStack::ValueOfNode(ValueNodeView value_node,
                              SourceLocation source_loc) const
    -> ErrorOr<Nonnull<const Value*>> {
  std::optional<const Value*> value = (phase_ == Phase::CompileTime)
                                          ? value_node.symbolic_identity()
                                          : value_node.constant_value();
  if (value.has_value()) {
    return *value;
  }
  for (const std::unique_ptr<Action>& action : todo_) {
    // TODO: have static name resolution identify the scope of value_node
    // as an AstNode, and then perform lookup _only_ on the Action associated
    // with that node. This will help keep unwanted dynamic-scoping behavior
    // from sneaking in.
    if (action->scope().has_value()) {
      std::optional<Nonnull<const Value*>> result =
          action->scope()->Get(value_node);
      if (result.has_value()) {
        return *result;
      }
    }
  }
  if (globals_.has_value()) {
    std::optional<Nonnull<const Value*>> result = globals_->Get(value_node);
    if (result.has_value()) {
      return *result;
    }
  }
  // TODO: Move these errors to compile time and explain them more clearly.
  return FATAL_RUNTIME_ERROR(source_loc)
         << "could not find `" << value_node.base() << "`";
}

void ActionStack::MergeScope(RuntimeScope scope) {
  for (const std::unique_ptr<Action>& action : todo_) {
    if (action->scope().has_value()) {
      action->scope()->Merge(std::move(scope));
      return;
    }
  }
  if (globals_.has_value()) {
    globals_->Merge(std::move(scope));
    return;
  }
  FATAL() << "No current scope";
}

void ActionStack::InitializeFragment(ContinuationValue::StackFragment& fragment,
                                     Nonnull<const Statement*> body) {
  std::vector<Nonnull<const RuntimeScope*>> scopes;
  for (const std::unique_ptr<Action>& action : todo_) {
    if (action->scope().has_value()) {
      scopes.push_back(&*action->scope());
    }
  }
  // We don't capture globals_ or constants_ because they're global.

  std::vector<std::unique_ptr<Action>> reversed_todo;
  reversed_todo.push_back(std::make_unique<StatementAction>(body));
  reversed_todo.push_back(
      std::make_unique<ScopeAction>(RuntimeScope::Capture(scopes)));
  fragment.StoreReversed(std::move(reversed_todo));
}

auto ActionStack::FinishAction() -> ErrorOr<Success> {
  std::unique_ptr<Action> act = todo_.Pop();
  switch (act->kind()) {
    case Action::Kind::ExpressionAction:
    case Action::Kind::LValAction:
    case Action::Kind::PatternAction:
      FATAL() << "This kind of action must produce a result: " << *act;
    case Action::Kind::ScopeAction:
      FATAL() << "ScopeAction at top of stack";
    case Action::Kind::StatementAction:
    case Action::Kind::DeclarationAction:
      PopScopes();
  }
  return Success();
}

auto ActionStack::FinishAction(Nonnull<const Value*> result)
    -> ErrorOr<Success> {
  std::unique_ptr<Action> act = todo_.Pop();
  switch (act->kind()) {
    case Action::Kind::StatementAction:
    case Action::Kind::DeclarationAction:
      FATAL() << "This kind of Action cannot produce results: " << *act;
    case Action::Kind::ScopeAction:
      FATAL() << "ScopeAction at top of stack";
    case Action::Kind::ExpressionAction:
    case Action::Kind::LValAction:
    case Action::Kind::PatternAction:
      PopScopes();
      SetResult(result);
  }
  return Success();
}

auto ActionStack::Spawn(std::unique_ptr<Action> child) -> ErrorOr<Success> {
  Action& action = *todo_.Top();
  action.set_pos(action.pos() + 1);
  todo_.Push(std::move(child));
  return Success();
}

auto ActionStack::Spawn(std::unique_ptr<Action> child, RuntimeScope scope)
    -> ErrorOr<Success> {
  Action& action = *todo_.Top();
  action.set_pos(action.pos() + 1);
  todo_.Push(std::make_unique<ScopeAction>(std::move(scope)));
  todo_.Push(std::move(child));
  return Success();
}

auto ActionStack::RunAgain() -> ErrorOr<Success> {
  Action& action = *todo_.Top();
  action.set_pos(action.pos() + 1);
  return Success();
}

auto ActionStack::UnwindTo(Nonnull<const Statement*> ast_node)
    -> ErrorOr<Success> {
  while (true) {
    if (const auto* statement_action =
            llvm::dyn_cast<StatementAction>(todo_.Top().get());
        statement_action != nullptr &&
        &statement_action->statement() == ast_node) {
      break;
    }
    todo_.Pop();
  }
  return Success();
}

auto ActionStack::UnwindPast(Nonnull<const Statement*> ast_node)
    -> ErrorOr<Success> {
  RETURN_IF_ERROR(UnwindTo(ast_node));
  todo_.Pop();
  PopScopes();
  return Success();
}

auto ActionStack::UnwindPast(Nonnull<const Statement*> ast_node,
                             Nonnull<const Value*> result) -> ErrorOr<Success> {
  RETURN_IF_ERROR(UnwindPast(ast_node));
  SetResult(result);
  return Success();
}

auto ActionStack::Resume(Nonnull<const ContinuationValue*> continuation)
    -> ErrorOr<Success> {
  Action& action = *todo_.Top();
  action.set_pos(action.pos() + 1);
  continuation->stack().RestoreTo(todo_);
  return Success();
}

static auto IsRunAction(const Action& action) -> bool {
  const auto* statement = llvm::dyn_cast<StatementAction>(&action);
  return statement != nullptr && llvm::isa<Run>(statement->statement());
}

auto ActionStack::Suspend() -> ErrorOr<Success> {
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
  return Success();
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
