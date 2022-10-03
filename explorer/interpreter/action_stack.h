// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_ACTION_STACK_H_
#define CARBON_EXPLORER_INTERPRETER_ACTION_STACK_H_

#include <memory>
#include <optional>
#include <stack>

#include "common/ostream.h"
#include "explorer/ast/statement.h"
#include "explorer/interpreter/action.h"
#include "explorer/interpreter/value.h"

namespace Carbon {

// Selects between compile-time and run-time behavior.
enum class Phase { CompileTime, RunTime };

// The stack of Actions currently being executed by the interpreter.
class ActionStack {
 public:
  // Constructs an empty compile-time ActionStack.
  ActionStack() : phase_(Phase::CompileTime) {}

  // Constructs an empty run-time ActionStack that allocates global variables
  // on `heap`.
  explicit ActionStack(Nonnull<HeapAllocationInterface*> heap)
      : globals_(RuntimeScope(heap)), phase_(Phase::RunTime) {}

  void Print(llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

  // TODO: consider unifying with Print.
  void PrintScopes(llvm::raw_ostream& out) const;

  // Starts execution with `action` at the top of the stack. Cannot be called
  // when IsEmpty() is false.
  void Start(std::unique_ptr<Action> action);

  // True if the stack is empty.
  auto IsEmpty() const -> bool { return todo_.IsEmpty(); }

  // The Action currently at the top of the stack. This will never be a
  // ScopeAction.
  auto CurrentAction() -> Action& { return *todo_.Top(); }

  // Allocates storage for `value_node`, and initializes it to `value`.
  void Initialize(ValueNodeView value_node, Nonnull<const Value*> value);

  // Returns the value bound to `value_node`. If `value_node` is a local
  // variable, this will be an LValue.
  auto ValueOfNode(ValueNodeView value_node, SourceLocation source_loc) const
      -> ErrorOr<Nonnull<const Value*>>;

  // Merges `scope` into the innermost scope currently on the stack.
  void MergeScope(RuntimeScope scope);

  // Initializes `fragment` so that, when resumed, it begins execution of
  // `body`.
  void InitializeFragment(ContinuationValue::StackFragment& fragment,
                          Nonnull<const Statement*> body);

  // The result produced by the `action` argument of the most recent
  // Start call. Cannot be called if IsEmpty() is false, or if `action`
  // was an action that doesn't produce results.
  auto result() const -> Nonnull<const Value*> { return *result_; }

  // The following methods, called "transition methods", update the state of
  // the ActionStack and/or the current Action to reflect the effects of
  // executing a step of that Action. Execution of an Action step should always
  // invoke exactly one transition method, as the very last operation. This is a
  // matter of safety as well as convention: most transition methods modify the
  // state of the current action, and some of them destroy it. To help enforce
  // this requirement, we have a convention of making these methods return an
  // ErrorOr<Success> even when a method can't actually fail, and calling the
  // methods as part of return statements, e.g. `return todo_.FinishAction()`.

  // Finishes execution of the current Action. If `result` is specified, it
  // represents the result of that Action.
  auto FinishAction() -> ErrorOr<Success>;
  auto FinishAction(Nonnull<const Value*> result) -> ErrorOr<Success>;

  // Advances the current action one step, and push `child` onto the stack.
  // If `scope` is specified, `child` will be executed in that scope.
  auto Spawn(std::unique_ptr<Action> child) -> ErrorOr<Success>;
  auto Spawn(std::unique_ptr<Action> child, RuntimeScope scope)
      -> ErrorOr<Success>;
  // Replace the current action with another action that produces the same kind
  // of result and run it next.
  auto ReplaceWith(std::unique_ptr<Action> child) -> ErrorOr<Success>;

  // Start a new recursive action.
  auto BeginRecursiveAction() {
    todo_.Push(std::make_unique<RecursiveAction>());
  }

  // Advances the current action one step.
  auto RunAgain() -> ErrorOr<Success>;

  // Unwinds Actions from the stack until the StatementAction associated with
  // `ast_node` is at the top of the stack.
  auto UnwindTo(Nonnull<const Statement*> ast_node) -> ErrorOr<Success>;

  // Unwinds Actions from the stack until the StatementAction associated with
  // `ast_node` has been removed from the stack. If `result` is specified,
  // it represents the result of that Action (StatementActions normally cannot
  // produce results, but the body of a function can).
  auto UnwindPast(Nonnull<const Statement*> ast_node) -> ErrorOr<Success>;
  auto UnwindPast(Nonnull<const Statement*> ast_node,
                  Nonnull<const Value*> result) -> ErrorOr<Success>;

  // Resumes execution of a suspended continuation.
  auto Resume(Nonnull<const ContinuationValue*> continuation)
      -> ErrorOr<Success>;

  // Suspends execution of the currently-executing continuation.
  auto Suspend() -> ErrorOr<Success>;

  void Pop() { todo_.Pop(); }

 private:
  // Pop any ScopeActions from the top of the stack, propagating results as
  // needed, to restore the invariant that todo_.Top() is not a ScopeAction.
  // Store the popped scope action into cleanup_stack, so that the destructor
  // can be called for the variables
  void PopScopes(std::stack<std::unique_ptr<Action>>& cleanup_stack);

  // Set `result` as the result of the Action most recently removed from the
  // stack.
  void SetResult(Nonnull<const Value*> result);

  auto UnwindToWithCaptureScopesToDestroy(Nonnull<const Statement*> ast_node)
      -> std::stack<std::unique_ptr<Action>>;

  auto UnwindPastWithCaptureScopesToDestroy(Nonnull<const Statement*> ast_node)
      -> std::stack<std::unique_ptr<Action>>;

  // Create CleanUpActions for all actions
  void PushCleanUpActions(std::stack<std::unique_ptr<Action>> actions);

  // Create and push a CleanUpAction on the stack
  void PushCleanUpAction(std::unique_ptr<Action> act);

  // TODO: consider defining a non-nullable unique_ptr-like type to use here.
  Stack<std::unique_ptr<Action>> todo_;
  std::optional<Nonnull<const Value*>> result_;
  std::optional<RuntimeScope> globals_;
  Phase phase_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_ACTION_STACK_H_
