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
#include "executable_semantics/interpreter/frame.h"
#include "executable_semantics/interpreter/heap.h"
#include "executable_semantics/interpreter/stack.h"
#include "executable_semantics/interpreter/value.h"

namespace Carbon {

using Env = Dictionary<std::string, Address>;

class Interpreter {
 public:
  explicit Interpreter(Nonnull<Arena*> arena)
      : arena(arena), globals(arena), heap(arena) {}

  // Interpret the whole program.
  auto InterpProgram(const std::vector<Nonnull<const Declaration*>>& fs) -> int;

  // Interpret an expression at compile-time.
  auto InterpExp(Env values, Nonnull<const Expression*> e)
      -> Nonnull<const Value*>;

  // Interpret a pattern at compile-time.
  auto InterpPattern(Env values, Nonnull<const Pattern*> p)
      -> Nonnull<const Value*>;

  // Attempts to match `v` against the pattern `p`. If matching succeeds,
  // returns the bindings of pattern variables to their matched values.
  auto PatternMatch(Nonnull<const Value*> p, Nonnull<const Value*> v,
                    SourceLocation loc) -> std::optional<Env>;

  // Support TypeChecker allocating values on the heap.
  auto AllocateValue(Nonnull<const Value*> v) -> Address {
    return heap.AllocateValue(v);
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

  // Transition type which unwinds the `todo` and `scopes` stacks until it
  // reaches a specified Action lower in the stack.
  struct UnwindTo {
    const Nonnull<Action*> new_top;
  };

  // Transition type which unwinds the entire current stack frame, and returns
  // a specified value to the caller.
  struct UnwindFunctionCall {
    Nonnull<const Value*> return_val;
  };

  // Transition type which removes the current action from the top of the todo
  // stack, then creates a new stack frame which calls the specified function
  // with the specified arguments.
  struct CallFunction {
    Nonnull<const FunctionValue*> function;
    Nonnull<const Value*> args;
    SourceLocation loc;
  };

  // Transition type which does nothing.
  //
  // TODO(geoffromer): This is a temporary placeholder during refactoring. All
  // uses of this type should be replaced with meaningful transitions.
  struct ManualTransition {};

  using Transition =
      std::variant<Done, Spawn, Delegate, RunAgain, UnwindTo,
                   UnwindFunctionCall, CallFunction, ManualTransition>;

  // Visitor which implements the behavior associated with each transition type.
  class DoTransition;

  void Step();

  // State transitions for expressions.
  auto StepExp() -> Transition;
  // State transitions for lvalues.
  auto StepLvalue() -> Transition;
  // State transitions for patterns.
  auto StepPattern() -> Transition;
  // State transition for statements.
  auto StepStmt() -> Transition;

  void InitGlobals(const std::vector<Nonnull<const Declaration*>>& fs);
  auto CurrentEnv() -> Env;
  auto GetFromEnv(SourceLocation loc, const std::string& name) -> Address;

  void DeallocateScope(Nonnull<Scope*> scope);
  void DeallocateLocals(Nonnull<Frame*> frame);

  auto CreateTuple(Nonnull<Action*> act, Nonnull<const Expression*> exp)
      -> Nonnull<const Value*>;

  auto EvalPrim(Operator op, const std::vector<Nonnull<const Value*>>& args,
                SourceLocation loc) -> Nonnull<const Value*>;

  void PatternAssignment(Nonnull<const Value*> pat, Nonnull<const Value*> val,
                         SourceLocation loc);

  void PrintState(llvm::raw_ostream& out);

  Nonnull<Arena*> arena;

  // Globally-defined entities, such as functions, structs, or choices.
  Env globals;

  Stack<Nonnull<Frame*>> stack;
  Heap heap;
  std::optional<Nonnull<const Value*>> program_value;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_INTERPRETER_H_
