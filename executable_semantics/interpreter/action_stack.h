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

  // TODO: consider unifying with Print.
  void PrintScopes(llvm::raw_ostream& out) const;

  // Sets the heap that variables will be allocated on. Cannot be called at
  // run time, or when IsEmpty() is false, and marks the start of run time.
  void SetHeap(Nonnull<HeapAllocationInterface*> heap) {
    CHECK(todo_.IsEmpty());
    CHECK(!globals_.has_value());
    globals_ = RuntimeScope(heap);
  }

  // Starts execution with `action` at the top of the stack. Cannot be called
  // when IsEmpty() is false.
  void Start(std::unique_ptr<Action> action);

  // True if the stack is empty.
  auto IsEmpty() const -> bool { return todo_.IsEmpty(); }

  // The Action currently at the top of the stack. This will never be a
  // ScopeAction.
  auto CurrentAction() -> Action& { return *todo_.Top(); }

  // Allocates storage for `named_entity`, and initializes it to `value`.
  void Initialize(NamedEntityView named_entity, Nonnull<const Value*> value);

  // Returns the value bound to `named_entity`. If `named_entity` is a local
  // variable, this will be an LValue.
  auto ValueOfName(NamedEntityView named_entity,
                   SourceLocation source_loc) const -> Nonnull<const Value*>;

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
  void Spawn(std::unique_ptr<Action> child, RuntimeScope scope);

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
  std::optional<RuntimeScope> globals_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_ACTION_STACK_H_
