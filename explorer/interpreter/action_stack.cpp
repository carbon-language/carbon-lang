// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/action_stack.h"

#include "common/error.h"
#include "explorer/interpreter/action.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"

namespace Carbon {

void ActionStack::Print(llvm::raw_ostream& out) const {
  llvm::ListSeparator sep(" ## ");
  for (const std::unique_ptr<Action>& action : todo_) {
    out << sep << *action;
  }
}

// OBSOLETE
void ActionStack::PrintScopes(llvm::raw_ostream& out) const {
  llvm::ListSeparator sep(" ## ");
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
  CARBON_CHECK(todo_.IsEmpty());
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
  std::optional<const Value*> constant_value = value_node.constant_value();
  if (constant_value.has_value()) {
    return *constant_value;
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
  // We don't know the value of this node, but at compile time we may still be
  // able to form a symbolic value for it. For example, in
  //
  //   fn F[T:! Type](x: T) {}
  //
  // ... we don't know the value of `T` but can still symbolically evaluate it
  // to a `VariableType`. At runtime we need actual values.
  if (phase_ == Phase::CompileTime) {
    std::optional<const Value*> symbolic_identity =
        value_node.symbolic_identity();
    if (symbolic_identity.has_value()) {
      return *symbolic_identity;
    }
  }
  // TODO: Move these errors to compile time and explain them more clearly.
  return ProgramError(source_loc)
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
  CARBON_FATAL() << "No current scope";
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

namespace {
// The way in which FinishAction should be called for a particular kind of
// action.
enum class FinishActionKind {
  // FinishAction should not be passed a value.
  NoValue,
  // FinishAction should be passed a value.
  Value,
  // FinishAction should not be called. The Action needs custom handling.
  NeverCalled,
};
}  // namespace

static auto FinishActionKindFor(Action::Kind kind) -> FinishActionKind {
  switch (kind) {
    case Action::Kind::ExpressionAction:
    case Action::Kind::WitnessAction:
    case Action::Kind::LValAction:
    case Action::Kind::PatternAction:
      return FinishActionKind::Value;
    case Action::Kind::StatementAction:
    case Action::Kind::DeclarationAction:
    case Action::Kind::RecursiveAction:
      return FinishActionKind::NoValue;
    case Action::Kind::ScopeAction:
    case Action::Kind::CleanUpAction:
      return FinishActionKind::NeverCalled;
  }
}

auto ActionStack::FinishAction() -> ErrorOr<Success> {
  std::stack<std::unique_ptr<Action>> scopes_to_destroy;
  std::unique_ptr<Action> act = todo_.Pop();
  switch (FinishActionKindFor(act->kind())) {
    case FinishActionKind::Value:
      CARBON_FATAL() << "This kind of action must produce a result: " << *act;
    case FinishActionKind::NeverCalled:
      CARBON_FATAL() << "Should not call FinishAction for: " << *act;
    case FinishActionKind::NoValue:
      PopScopes(scopes_to_destroy);
      break;
  }
  PushCleanUpAction(std::move(act));
  PushCleanUpActions(std::move(scopes_to_destroy));
  return Success();
}

auto ActionStack::FinishAction(Nonnull<const Value*> result)
    -> ErrorOr<Success> {
  std::stack<std::unique_ptr<Action>> scopes_to_destroy;
  std::unique_ptr<Action> act = todo_.Pop();
  switch (FinishActionKindFor(act->kind())) {
    case FinishActionKind::NoValue:
      CARBON_FATAL() << "This kind of action cannot produce results: " << *act;
    case FinishActionKind::NeverCalled:
      CARBON_FATAL() << "Should not call FinishAction for: " << *act;
    case FinishActionKind::Value:
      PopScopes(scopes_to_destroy);
      SetResult(result);
      break;
  }
  PushCleanUpAction(std::move(act));
  PushCleanUpActions(std::move(scopes_to_destroy));
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

auto ActionStack::ReplaceWith(std::unique_ptr<Action> replacement)
    -> ErrorOr<Success> {
  std::unique_ptr<Action> old = todo_.Pop();
  CARBON_CHECK(FinishActionKindFor(old->kind()) ==
               FinishActionKindFor(replacement->kind()))
      << "Can't replace action " << *old << " with " << *replacement;
  todo_.Push(std::move(replacement));
  return Success();
}

auto ActionStack::RunAgain() -> ErrorOr<Success> {
  Action& action = *todo_.Top();
  action.set_pos(action.pos() + 1);
  return Success();
}

auto ActionStack::UnwindToWithCaptureScopesToDestroy(
    Nonnull<const Statement*> ast_node) -> std::stack<std::unique_ptr<Action>> {
  std::stack<std::unique_ptr<Action>> scopes_to_destroy;
  while (true) {
    if (const auto* statement_action =
            llvm::dyn_cast<StatementAction>(todo_.Top().get());
        statement_action != nullptr &&
        &statement_action->statement() == ast_node) {
      break;
    }
    auto item = todo_.Pop();
    auto& scope = item->scope();
    if (scope && item->kind() != Action::Kind::CleanUpAction) {
      std::unique_ptr<Action> cleanup_action =
          std::make_unique<CleanupAction>(std::move(*scope));
      scopes_to_destroy.push(std::move(cleanup_action));
    }
  }
  return scopes_to_destroy;
}

auto ActionStack::UnwindTo(Nonnull<const Statement*> ast_node)
    -> ErrorOr<Success> {
  std::stack<std::unique_ptr<Action>> scopes_to_destroy =
      UnwindToWithCaptureScopesToDestroy(ast_node);
  PushCleanUpActions(std::move(scopes_to_destroy));
  return Success();
}

auto ActionStack::UnwindPast(Nonnull<const Statement*> ast_node)
    -> ErrorOr<Success> {
  std::stack<std::unique_ptr<Action>> scopes_to_destroy =
      UnwindPastWithCaptureScopesToDestroy(ast_node);
  PushCleanUpActions(std::move(scopes_to_destroy));

  return Success();
}

auto ActionStack::UnwindPastWithCaptureScopesToDestroy(
    Nonnull<const Statement*> ast_node) -> std::stack<std::unique_ptr<Action>> {
  std::stack<std::unique_ptr<Action>> scopes_to_destroy =
      UnwindToWithCaptureScopesToDestroy(ast_node);
  auto item = todo_.Pop();
  scopes_to_destroy.push(std::move(item));
  PopScopes(scopes_to_destroy);
  return scopes_to_destroy;
}

auto ActionStack::UnwindPast(Nonnull<const Statement*> ast_node,
                             Nonnull<const Value*> result) -> ErrorOr<Success> {
  std::stack<std::unique_ptr<Action>> scopes_to_destroy =
      UnwindPastWithCaptureScopesToDestroy(ast_node);
  SetResult(result);
  PushCleanUpActions(std::move(scopes_to_destroy));
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

void ActionStack::PopScopes(
    std::stack<std::unique_ptr<Action>>& cleanup_stack) {
  while (!todo_.IsEmpty() && llvm::isa<ScopeAction>(*todo_.Top())) {
    auto act = todo_.Pop();
    if (act->scope()) {
      if ((*act->scope()).DestructionState() <
          RuntimeScope::State::CleanUpped) {
        (*act->scope()).TransitState();
        cleanup_stack.push(std::move(act));
      }
    }
  }
}

void ActionStack::SetResult(Nonnull<const Value*> result) {
  if (todo_.IsEmpty()) {
    result_ = result;
  } else {
    todo_.Top()->AddResult(result);
  }
}

void ActionStack::PushCleanUpActions(
    std::stack<std::unique_ptr<Action>> actions) {
  while (!actions.empty()) {
    auto& act = actions.top();
    if (act->scope()) {
      std::unique_ptr<Action> cleanup_action =
          std::make_unique<CleanupAction>(std::move(*act->scope()));
      todo_.Push(std::move(cleanup_action));
    }
    actions.pop();
  }
}

void ActionStack::PushCleanUpAction(std::unique_ptr<Action> act) {
  auto& scope = act->scope();
  if (scope && act->kind() != Action::Kind::CleanUpAction) {
    std::unique_ptr<Action> cleanup_action =
        std::make_unique<CleanupAction>(std::move(*scope));
    todo_.Push(std::move(cleanup_action));
  }
}

}  // namespace Carbon
