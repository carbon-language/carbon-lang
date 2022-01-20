// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_INTERPRETER_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_INTERPRETER_H_

#include <optional>
#include <utility>
#include <vector>

#include "common/ostream.h"
#include "executable_semantics/ast/ast.h"
#include "executable_semantics/ast/declaration.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/pattern.h"
#include "executable_semantics/interpreter/action.h"
#include "executable_semantics/interpreter/action_stack.h"
#include "executable_semantics/interpreter/heap.h"
#include "executable_semantics/interpreter/value.h"
#include "llvm/ADT/ArrayRef.h"

namespace Carbon {

class Interpreter {
 public:
  explicit Interpreter(Nonnull<Arena*> arena, bool trace)
      : arena_(arena), heap_(arena), trace_(trace) {}

  // Interpret the whole program.
  auto InterpProgram(const AST& ast) -> int;

  // Interpret an expression at compile-time.
  auto InterpExp(Nonnull<const Expression*> e) -> Nonnull<const Value*>;

  // Interpret a pattern at compile-time.
  auto InterpPattern(Nonnull<const Pattern*> p) -> Nonnull<const Value*>;

  // Attempts to match `v` against the pattern `p`, returning whether matching
  // is successful. If it is, populates **bindings with the variables bound by
  // the match; `bindings` should only be nullopt in contexts where `p`
  // is not permitted to bind variables. **bindings may be modified even if the
  // match is unsuccessful, so it should typically be created for the
  // PatternMatch call and then merged into an existing scope on success.
  [[nodiscard]] auto PatternMatch(
      Nonnull<const Value*> p, Nonnull<const Value*> v,
      SourceLocation source_loc, std::optional<Nonnull<RuntimeScope*>> bindings)
      -> bool;

  // Support TypeChecker allocating values on the heap.
  auto AllocateValue(Nonnull<const Value*> v) -> AllocationId {
    return heap_.AllocateValue(v);
  }

 private:
  void Step();

  // State transitions for expressions.
  void StepExp();
  // State transitions for lvalues.
  void StepLvalue();
  // State transitions for patterns.
  void StepPattern();
  // State transition for statements.
  void StepStmt();
  // State transition for declarations.
  void StepDeclaration();

  // Calls Step() repeatedly until there are no steps left to execute. Produces
  // trace output if trace_steps is true.
  void RunAllSteps(bool trace_steps);

  auto CreateStruct(const std::vector<FieldInitializer>& fields,
                    const std::vector<Nonnull<const Value*>>& values)
      -> Nonnull<const Value*>;

  auto EvalPrim(Operator op, const std::vector<Nonnull<const Value*>>& args,
                SourceLocation source_loc) -> Nonnull<const Value*>;

  // Returns the result of converting `value` to type `destination_type`.
  auto Convert(Nonnull<const Value*> value,
               Nonnull<const Value*> destination_type) const
      -> Nonnull<const Value*>;

  void PrintState(llvm::raw_ostream& out);

  // Runs `action` in an environment where the given constants are defined, and
  // returns the result. `action` must produce a result. In other words, it must
  // not be a StatementAction, ScopeAction, or DeclarationAction. Can only be
  // called at compile time (before InterpProgram), and while `todo_` is empty.
  auto RunCompileTimeAction(std::unique_ptr<Action> action)
      -> Nonnull<const Value*>;

  Nonnull<Arena*> arena_;

  Heap heap_;
  ActionStack todo_;

  // The underlying states of continuation values. All StackFragments created
  // during execution are tracked here, in order to safely deallocate the
  // contents of any non-completed continuations at the end of execution.
  std::vector<Nonnull<ContinuationValue::StackFragment*>> stack_fragments_;

  bool trace_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_INTERPRETER_H_
