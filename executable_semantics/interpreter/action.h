// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_ACTION_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_ACTION_H_

#include <vector>

#include "common/ostream.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/pattern.h"
#include "executable_semantics/ast/statement.h"
#include "executable_semantics/interpreter/dictionary.h"
#include "executable_semantics/interpreter/heap_allocation_interface.h"
#include "executable_semantics/interpreter/stack.h"
#include "executable_semantics/interpreter/value.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

using Env = Dictionary<std::string, AllocationId>;

// A Scope represents the name lookup environment associated with an Action,
// including any variables that are local to that action. Local variables
// will be deallocated from the Carbon Heap when the Scope is destroyed.
class Scope {
 public:
  // Constructs a Scope whose name environment is `values`, containing the local
  // variables in `locals`. The elements of `locals` must also be keys in
  // `values`, and their values must be allocated in `heap`.
  Scope(Env values, std::vector<std::string> locals,
        Nonnull<HeapAllocationInterface*> heap)
      : values_(values), locals_(std::move(locals)), heap_(heap) {}

  // Equivalent to `Scope(values, {}, heap)`.
  Scope(Env values, Nonnull<HeapAllocationInterface*> heap)
      : Scope(values, std::vector<std::string>(), heap) {}

  // Moving a Scope transfers ownership of its local variables.
  Scope(Scope&&) noexcept;
  auto operator=(Scope&&) noexcept -> Scope&;

  ~Scope();

  // Binds `name` to the value of `allocation` in `heap`, and takes
  // ownership of it.
  void AddLocal(const std::string& name, AllocationId allocation) {
    values_.Set(name, allocation);
    locals_.push_back(name);
  }

  auto values() const -> Env { return values_; }

 private:
  Env values_;
  std::vector<std::string> locals_;
  Nonnull<HeapAllocationInterface*> heap_;
};

// An Action represents the current state of a self-contained computation,
// usually associated with some AST node, such as evaluation of an expression or
// execution of a statement. Execution of an action is divided into a series of
// steps, and the `pos` field typically counts the number of steps executed.
//
// They should be destroyed as soon as they are done executing, in order to
// clean up the associated Carbon scope, and consequently they should not be
// allocated on an Arena. Actions are typically owned by the ActionStack.
//
// The actual behavior of an Action step is defined by Interpreter::Step, not by
// Action or its subclasses.
// TODO: consider moving this logic to a virtual method `Step`.
class Action {
 public:
  enum class Kind {
    LValAction,
    ExpressionAction,
    PatternAction,
    StatementAction,
    DeclarationAction,
    ScopeAction,
  };

  Action(const Value&) = delete;
  auto operator=(const Value&) -> Action& = delete;

  virtual ~Action() = default;

  void Print(llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto kind() const -> Kind { return kind_; }

  // The position or state of the action. Starts at 0 and is typically
  // incremented after each step.
  auto pos() const -> int { return pos_; }
  void set_pos(int pos) { this->pos_ = pos; }

  // The results of any Actions spawned by this Action.
  auto results() const -> const std::vector<Nonnull<const Value*>>& {
    return results_;
  }

  // Appends `result` to `results`.
  void AddResult(Nonnull<const Value*> result) { results_.push_back(result); }

  // Resets this Action to its initial state.
  void Clear() {
    CHECK(!scope_.has_value());
    pos_ = 0;
    results_.clear();
  }

  // Associates this action with a new scope, with initial state `scope`.
  // Values that are local to this scope will be deallocated when this
  // Action is completed or unwound. Can only be called once on a given
  // Action.
  void StartScope(Scope scope) {
    CHECK(!scope_.has_value());
    scope_ = std::move(scope);
  }

  // Returns the scope associated with this Action, if any.
  auto scope() -> std::optional<Scope>& { return scope_; }

 protected:
  // Constructs an Action. `kind` must be the enumerator corresponding to the
  // most-derived type being constructed.
  explicit Action(Kind kind) : kind_(kind) {}

 private:
  int pos_ = 0;
  std::vector<Nonnull<const Value*>> results_;
  std::optional<Scope> scope_;

  const Kind kind_;
};

// An Action which implements evaluation of an Expression to produce an
// LValue.
class LValAction : public Action {
 public:
  explicit LValAction(Nonnull<const Expression*> expression)
      : Action(Kind::LValAction), expression_(expression) {}

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::LValAction;
  }

  // The Expression this Action evaluates.
  auto expression() const -> const Expression& { return *expression_; }

 private:
  Nonnull<const Expression*> expression_;
};

// An Action which implements evaluation of an Expression to produce an
// rvalue. The result is expressed as a Value.
class ExpressionAction : public Action {
 public:
  explicit ExpressionAction(Nonnull<const Expression*> expression)
      : Action(Kind::ExpressionAction), expression_(expression) {}

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::ExpressionAction;
  }

  // The Expression this Action evaluates.
  auto expression() const -> const Expression& { return *expression_; }

 private:
  Nonnull<const Expression*> expression_;
};

// An Action which implements evaluation of a Pattern. The result is expressed
// as a Value.
class PatternAction : public Action {
 public:
  explicit PatternAction(Nonnull<const Pattern*> pattern)
      : Action(Kind::PatternAction), pattern_(pattern) {}

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::PatternAction;
  }

  // The Pattern this Action evaluates.
  auto pattern() const -> const Pattern& { return *pattern_; }

 private:
  Nonnull<const Pattern*> pattern_;
};

// An Action which implements execution of a Statement. Does not produce a
// result.
class StatementAction : public Action {
 public:
  explicit StatementAction(Nonnull<const Statement*> statement)
      : Action(Kind::StatementAction), statement_(statement) {}

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::StatementAction;
  }

  // The Statement this Action executes.
  auto statement() const -> const Statement& { return *statement_; }

 private:
  Nonnull<const Statement*> statement_;
};

// Action which implements the run-time effects of executing a Declaration.
// Does not produce a result.
class DeclarationAction : public Action {
 public:
  explicit DeclarationAction(Nonnull<const Declaration*> declaration)
      : Action(Kind::DeclarationAction), declaration_(declaration) {}

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::DeclarationAction;
  }

  // The Declaration this Action executes.
  auto declaration() const -> const Declaration& { return *declaration_; }

 private:
  Nonnull<const Declaration*> declaration_;
};

// Action which does nothing except introduce a new scope into the action
// stack. This is useful when a distinct scope doesn't otherwise have an
// Action it can naturally be associated with. ScopeActions are not associated
// with AST nodes.
class ScopeAction : public Action {
 public:
  explicit ScopeAction(Scope scope) : Action(Kind::ScopeAction) {
    StartScope(std::move(scope));
  }

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::ScopeAction;
  }
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_ACTION_H_
