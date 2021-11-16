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
#include "executable_semantics/interpreter/action_stack.h"
#include "executable_semantics/interpreter/heap.h"
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
  auto AllocateValue(Nonnull<const Value*> v) -> AllocationId {
    return heap_.AllocateValue(v);
  }

  void InitEnv(const Declaration& d, Env* env);
  void PrintEnv(Env values, llvm::raw_ostream& out);

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

  void InitGlobals(llvm::ArrayRef<Nonnull<Declaration*>> fs);
  auto CurrentEnv() -> Env;
  auto GetFromEnv(SourceLocation source_loc, const std::string& name)
      -> Address;

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
  auto ExecuteAction(std::unique_ptr<Action> action, Env values,
                     bool trace_steps) -> Nonnull<const Value*>;

  Nonnull<Arena*> arena_;

  // Globally-defined entities, such as functions, structs, or choices.
  Env globals_;

  ActionStack todo_;
  Heap heap_;

  // The underlying states of continuation values. All StackFragments created
  // during execution are tracked here, in order to safely deallocate the
  // contents of any non-completed continuations at the end of execution.
  std::vector<Nonnull<ContinuationValue::StackFragment*>> stack_fragments_;

  bool trace_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_INTERPRETER_H_
