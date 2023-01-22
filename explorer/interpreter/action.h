// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_ACTION_H_
#define CARBON_EXPLORER_INTERPRETER_ACTION_H_

#include <list>
#include <map>
#include <tuple>
#include <vector>

#include "common/ostream.h"
#include "explorer/ast/expression.h"
#include "explorer/ast/pattern.h"
#include "explorer/ast/statement.h"
#include "explorer/interpreter/dictionary.h"
#include "explorer/interpreter/heap_allocation_interface.h"
#include "explorer/interpreter/stack.h"
#include "explorer/interpreter/value.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

// A RuntimeScope manages and provides access to the storage for names that are
// not compile-time constants.
class RuntimeScope {
 public:
  // Returns a RuntimeScope whose Get() operation for a given name returns the
  // storage owned by the first entry in `scopes` that defines that name. This
  // behavior is closely analogous to a `[&]` capture in C++, hence the name.
  // `scopes` must contain at least one entry, and all entries must be backed
  // by the same Heap.
  static auto Capture(const std::vector<Nonnull<const RuntimeScope*>>& scopes)
      -> RuntimeScope;

  // Constructs a RuntimeScope that allocates storage in `heap`.
  explicit RuntimeScope(Nonnull<HeapAllocationInterface*> heap) : heap_(heap) {}

  // Moving a RuntimeScope transfers ownership of its allocations.
  RuntimeScope(RuntimeScope&&) noexcept;
  auto operator=(RuntimeScope&&) noexcept -> RuntimeScope&;

  // Deallocates any allocations in this scope from `heap`.
  ~RuntimeScope();

  void Print(llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

  // Binds `value` as the value of `value_node`.
  void Bind(ValueNodeView value_node, Nonnull<const Value*> value);

  // Allocates storage for `value_node` in `heap`, and initializes it with
  // `value`.
  // TODO: Update existing callers to use Bind instead, where appropriate.
  void Initialize(ValueNodeView value_node, Nonnull<const Value*> value);

  // Transfers the names and allocations from `other` into *this. The two
  // scopes must not define the same name, and must be backed by the same Heap.
  void Merge(RuntimeScope other);

  // Returns the local storage for value_node, if it has storage local to
  // this scope.
  auto Get(ValueNodeView value_node) const
      -> std::optional<Nonnull<const LValue*>>;

  // Returns the local values in created order
  auto allocations() const -> const std::vector<AllocationId>& {
    return allocations_;
  }

 private:
  llvm::MapVector<ValueNodeView, Nonnull<const LValue*>,
                  std::map<ValueNodeView, unsigned>>
      locals_;
  std::vector<AllocationId> allocations_;
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
    WitnessAction,
    StatementAction,
    DeclarationAction,
    ScopeAction,
    RecursiveAction,
    CleanUpAction,
    DestroyAction
  };

  Action(const Value&) = delete;
  auto operator=(const Value&) -> Action& = delete;

  virtual ~Action() = default;

  void Print(llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

  // Resets this Action to its initial state.
  void Clear() {
    CARBON_CHECK(!scope_.has_value());
    pos_ = 0;
    results_.clear();
  }

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
  void ReplaceResult(std::size_t index, Nonnull<const Value*> value) {
    CARBON_CHECK(index < results_.size());
    results_[index] = value;
  }
  // Appends `result` to `results`.
  void AddResult(Nonnull<const Value*> result) { results_.push_back(result); }

  // Returns the scope associated with this Action, if any.
  auto scope() -> std::optional<RuntimeScope>& { return scope_; }
  auto scope() const -> const std::optional<RuntimeScope>& { return scope_; }

  // Associates this action with a new scope, with initial state `scope`.
  // Values that are local to this scope will be deallocated when this
  // Action is completed or unwound. Can only be called once on a given
  // Action.
  void StartScope(RuntimeScope scope) {
    CARBON_CHECK(!scope_.has_value());
    scope_ = std::move(scope);
  }

 protected:
  // Constructs an Action. `kind` must be the enumerator corresponding to the
  // most-derived type being constructed.
  explicit Action(Kind kind) : kind_(kind) {}

 private:
  int pos_ = 0;
  std::vector<Nonnull<const Value*>> results_;
  std::optional<RuntimeScope> scope_;

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

// An Action which implements evaluation of a Witness to resolve it in the
// local context.
class WitnessAction : public Action {
 public:
  explicit WitnessAction(Nonnull<const Witness*> witness)
      : Action(Kind::WitnessAction), witness_(witness) {}

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::WitnessAction;
  }

  // The Witness this Action resolves.
  auto witness() const -> Nonnull<const Witness*> { return witness_; }

 private:
  Nonnull<const Witness*> witness_;
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

// An Action which implements destroying all local allocations in a scope.
class CleanUpAction : public Action {
 public:
  explicit CleanUpAction(RuntimeScope scope)
      : Action(Kind::CleanUpAction),
        allocations_count_(scope.allocations().size()) {
    StartScope(std::move(scope));
  }

  auto allocations_count() const -> int { return allocations_count_; }

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::CleanUpAction;
  }

 private:
  int allocations_count_;
};

// An Action which implements destroying a single value, including all nested
// values.
class DestroyAction : public Action {
 public:
  // lvalue: Address of the object to be destroyed
  // value:  The value to be destroyed
  //         In most cases the lvalue address points to value
  //         In the case that the member of a class is to be destroyed,
  //         the lvalue points to the address of the class object
  //         and the value is the member of the class
  explicit DestroyAction(Nonnull<const LValue*> lvalue,
                         Nonnull<const Value*> value)
      : Action(Kind::DestroyAction), lvalue_(lvalue), value_(value) {}

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::DestroyAction;
  }

  auto lvalue() const -> Nonnull<const LValue*> { return lvalue_; }

  auto value() const -> Nonnull<const Value*> { return value_; }

 private:
  Nonnull<const LValue*> lvalue_;
  Nonnull<const Value*> value_;
};

// Action which does nothing except introduce a new scope into the action
// stack. This is useful when a distinct scope doesn't otherwise have an
// Action it can naturally be associated with. ScopeActions are not associated
// with AST nodes.
class ScopeAction : public Action {
 public:
  explicit ScopeAction(RuntimeScope scope) : Action(Kind::ScopeAction) {
    StartScope(std::move(scope));
  }

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::ScopeAction;
  }
};

// Action which contains another action and does nothing further once that
// action completes. This action therefore acts as a marker on the action stack
// that indicates that the interpreter should stop when the inner action has
// finished, and holds the result of that inner action. This is useful to allow
// a sequence of steps for an action to be run immediately rather than as part
// of the normal step queue.
//
// Should be avoided where possible.
class RecursiveAction : public Action {
 public:
  explicit RecursiveAction() : Action(Kind::RecursiveAction) {}

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::RecursiveAction;
  }
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_ACTION_H_
