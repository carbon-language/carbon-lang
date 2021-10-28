// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_INTERPRETER_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_INTERPRETER_H_

#include <optional>
#include <utility>
#include <vector>

#include "common/ostream.h"
#include "executable_semantics/ast/declaration.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/pattern.h"
#include "executable_semantics/interpreter/action.h"
#include "executable_semantics/interpreter/heap.h"
#include "executable_semantics/interpreter/stack.h"
#include "executable_semantics/interpreter/value.h"
#include "llvm/ADT/ArrayRef.h"

namespace Carbon {

class Interpreter {
 public:
  explicit Interpreter(Nonnull<Arena*> arena, bool trace)
      : arena_(arena), globals_(arena), heap_(arena), trace_(trace) {}

  // Interpret the whole program.
  auto InterpProgram(llvm::ArrayRef<Nonnull<Declaration*>> fs,
                     Nonnull<const Expression*> call_main) -> int;

  // Interpret an expression at compile-time.
  auto InterpExp(Env values, Nonnull<const Expression*> e)
      -> Nonnull<const Value*>;

  // Interpret a pattern at compile-time.
  auto InterpPattern(Env values, Nonnull<const Pattern*> p)
      -> Nonnull<const Value*>;

  // Attempts to match `v` against the pattern `p`. If matching succeeds,
  // returns the bindings of pattern variables to their matched values.
  auto PatternMatch(Nonnull<const Value*> p, Nonnull<const Value*> v,
                    SourceLocation source_loc) -> std::optional<Env>;

  // Support TypeChecker allocating values on the heap.
  auto AllocateValue(Nonnull<const Value*> v) -> Address {
    return heap_.AllocateValue(v);
  }

  void InitEnv(const Declaration& d, Env* env);
  void PrintEnv(Env values, llvm::raw_ostream& out);

 private:
  // State transition functions
  //
  // The `Step*` family of functions implement state transitions in the
  // interpreter by executing a step of the Action at the top of the todo stack,
  // and then returning a Transition that specifies how `state.stack` should be
  // updated. `Transition` is a variant of several "transition types"
  // representing the different kinds of state transition.

  // Transition type which indicates that the current Action is now done.
  struct Done {
    // The value computed by the Action. Should always be nullopt for Statement
    // Actions, and never null for any other kind of Action.
    std::optional<Nonnull<const Value*>> result;
  };

  // Transition type which spawns a new Action on the todo stack above the
  // current Action, and increments the current Action's position counter.
  struct Spawn {
    Nonnull<Action*> child;
  };

  // Transition type which spawns a new Action that replaces the current action
  // on the todo stack.
  struct Delegate {
    Nonnull<Action*> delegate;
  };

  // Transition type which keeps the current Action at the top of the stack,
  // and increments its position counter.
  struct RunAgain {};

  // Transition type which unwinds the `todo` stack down to `ast_node`. If
  // `unwind_ast_node` is true, it also unwinds `ast_node`; in that case,
  // `result` is will be treated as the result of that StatementAction if set.
  struct Unwind {
    Nonnull<const Statement*> ast_node;
    bool unwind_ast_node;
    std::optional<Nonnull<const Value*>> result;
  };

  // Transition type which removes the current action from the top of the todo
  // stack, then creates a new stack frame which calls the specified function
  // with the specified arguments.
  struct CallFunction {
    Nonnull<const FunctionDeclaration*> function;
    Nonnull<const Value*> args;
    SourceLocation source_loc;
  };

  // Transition type which does nothing.
  //
  // TODO(geoffromer): This is a temporary placeholder during refactoring. All
  // uses of this type should be replaced with meaningful transitions.
  struct ManualTransition {};

  using Transition = std::variant<Done, Spawn, Delegate, RunAgain, Unwind,
                                  CallFunction, ManualTransition>;

  // Visitor which implements the behavior associated with each transition type.
  class DoTransition;
  friend class DoTransition;

  void Step();

  // State transitions for expressions.
  auto StepExp() -> Transition;
  // State transitions for lvalues.
  auto StepLvalue() -> Transition;
  // State transitions for patterns.
  auto StepPattern() -> Transition;
  // State transition for statements.
  auto StepStmt() -> Transition;

  void InitGlobals(llvm::ArrayRef<Nonnull<Declaration*>> fs);
  auto CurrentScope() -> Scope&;
  auto CurrentEnv() -> Env;
  auto GetFromEnv(SourceLocation source_loc, const std::string& name)
      -> Address;

  auto PopAndDeallocateScope() -> Nonnull<Action*>;

  auto CreateTuple(Nonnull<Action*> act, Nonnull<const Expression*> exp)
      -> Nonnull<const Value*>;
  auto CreateStruct(const std::vector<FieldInitializer>& fields,
                    const std::vector<Nonnull<const Value*>>& values)
      -> Nonnull<const Value*>;

  auto EvalPrim(Operator op, const std::vector<Nonnull<const Value*>>& args,
                SourceLocation source_loc) -> Nonnull<const Value*>;

  void PatternAssignment(Nonnull<const Value*> pat, Nonnull<const Value*> val,
                         SourceLocation source_loc);

  // Returns the result of converting `value` to type `destination_type`.
  auto Convert(Nonnull<const Value*> value,
               Nonnull<const Value*> destination_type) const
      -> Nonnull<const Value*>;

  void PrintState(llvm::raw_ostream& out);

  // Runs `action` in a scope consisting of `values`, and returns the result.
  // `action` must produce a result. In other words, it must not be a
  // StatementAction or ScopeAction.
  //
  // TODO: consider whether to use this->trace_ rather than a separate
  // trace_steps parameter.
  auto ExecuteAction(Nonnull<Action*> action, Env values, bool trace_steps)
      -> Nonnull<const Value*>;

  Nonnull<Arena*> arena_;

  // Globally-defined entities, such as functions, structs, or choices.
  Env globals_;

  Stack<Nonnull<Action*>> todo_;
  Heap heap_;

  bool trace_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_INTERPRETER_H_
