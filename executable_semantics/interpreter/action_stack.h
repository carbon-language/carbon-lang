// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_ACTION_STACK_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_ACTION_STACK_H_

#include <memory>
#include <optional>

#include "common/ostream.h"
#include "executable_semantics/ast/statement.h"
#include "executable_semantics/interpreter/action.h"
#include "executable_semantics/interpreter/value.h"

namespace Carbon {

// The stack of Actions currently being executed by the interpreter.
class ActionStack {
 public:
  // Constructs an empty ActionStack
  ActionStack() = default;

  void Print(llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

  // Starts execution with `action` at the top of the stack, in the given scope.
  // `action` must be an `ExpressionAction` or `PatternAction`.
  void Start(std::unique_ptr<Action> action, Scope scope);

  // True if the stack is empty.
  auto IsEmpty() const -> bool { return todo_.IsEmpty(); }

  // The result produced by the `action` argument of the most recent
  // `Start` call. *this must be empty, signifying that the action has been
  // fully executed.
  auto result() const -> Nonnull<const Value*> { return *result_; }

  // The Action currently at the top of the stack. This will never be a
  // ScopeAction.
  auto CurrentAction() -> Action& { return *todo_.Top(); }

  // The scope that should be used to resolve name lookups in the current
  // action.
  auto CurrentScope() const -> Scope&;

  // The following methods, called "transition methods", update the state of
  // the ActionStack and/or the current Action to reflect the effects of
  // executing a step of that Action. Execution of an Action step should always
  // invoke exactly one transition method, as the very last operation. This is a
  // matter of safety as well as convention: most transition methods modify the
  // state of the current action, and some of them destroy it. To help enforce
  // this requirement, we have a convention of calling these methods as part of
  // return statements, e.g. `return todo_.FinishAction()`, even though they
  // return void.

  // Finishes execution of the current Action. If `result` is specified, it
  // represents the result of that Action.
  void FinishAction();
  void FinishAction(Nonnull<const Value*> result);

  // Advances the current action one step, and push `child` onto the stack.
  // If `scope` is specified, `child` will be executed in that scope.
  void Spawn(std::unique_ptr<Action> child);
  void Spawn(std::unique_ptr<Action> child, Scope scope);

  // Advances the current action one step.
  void RunAgain();

  // Unwinds Actions from the stack until the StatementAction associated with
  // `ast_node` is at the top of the stack.
  void UnwindTo(Nonnull<const Statement*> ast_node);

  // Unwinds Actions from the stack until the StatementAction associated with
  // `ast_node` has been removed from the stack. If `result` is specified,
  // it represents the result of that Action (StatementActions normally cannot
  // produce results, but the body of a function can).
  void UnwindPast(Nonnull<const Statement*> ast_node);
  void UnwindPast(Nonnull<const Statement*> ast_node,
                  Nonnull<const Value*> result);

  // Resumes execution of a suspended continuation.
  void Resume(Nonnull<const ContinuationValue*> continuation);

  // Suspends execution of the currently-executing continuation.
  void Suspend();

 private:
  // Pop any ScopeActions from the top of the stack, propagating results as
  // needed, to restore the invariant that todo_.Top() is not a ScopeAction.
  void PopScopes();

  // Set `result` as the result of the Action most recently removed from the
  // stack.
  void SetResult(Nonnull<const Value*> result);

  // TODO: consider defining a non-nullable unique_ptr-like type to use here.
  Stack<std::unique_ptr<Action>> todo_;
  std::optional<Nonnull<const Value*>> result_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_ACTION_STACK_H_
