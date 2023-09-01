// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_ACTION_H_
#define CARBON_EXPLORER_INTERPRETER_ACTION_H_

#include <list>
#include <map>
#include <optional>
#include <tuple>
#include <vector>

#include "common/check.h"
#include "common/ostream.h"
#include "explorer/ast/address.h"
#include "explorer/ast/expression.h"
#include "explorer/ast/pattern.h"
#include "explorer/ast/statement.h"
#include "explorer/ast/value.h"
#include "explorer/base/source_location.h"
#include "explorer/interpreter/dictionary.h"
#include "explorer/interpreter/heap_allocation_interface.h"
#include "explorer/interpreter/stack.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

// A RuntimeScope manages and provides access to the storage for names that are
// not compile-time constants.
class RuntimeScope : public Printable<RuntimeScope> {
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

  void Print(llvm::raw_ostream& out) const;

  // Allocates storage for `value_node` in `heap`, and initializes it with
  // `value`.
  auto Initialize(ValueNodeView value_node, Nonnull<const Value*> value)
      -> Nonnull<const LocationValue*>;

  // Bind allocation lifetime to scope. Should only be called with unowned
  // allocations to avoid a double free.
  void BindLifetimeToScope(Address address);

  // Binds location `address` of a reference value to `value_node` without
  // allocating local storage.
  void Bind(ValueNodeView value_node, Address address);

  // Binds location `address` of a reference value to `value_node` without
  // allocating local storage, and pins the value, making it immutable.
  void BindAndPin(ValueNodeView value_node, Address address);

  // Binds unlocated `value` to `value_node` without allocating local storage.
  // TODO: BindValue should pin the lifetime of `value` and make sure it isn't
  // mutated.
  void BindValue(ValueNodeView value_node, Nonnull<const Value*> value);

  // Transfers the names and allocations from `other` into *this. The two
  // scopes must not define the same name, and must be backed by the same Heap.
  void Merge(RuntimeScope other);

  // Given node `value_node`, returns:
  // - its `LocationValue*` if bound to a reference expression in this scope,
  // - a `Value*` if bound to a value expression in this scope, or
  // - `nullptr` if not bound.
  auto Get(ValueNodeView value_node, SourceLocation source_loc) const
      -> ErrorOr<std::optional<Nonnull<const Value*>>>;

  // Returns the local values with allocation in created order.
  auto allocations() const -> const std::vector<AllocationId>& {
    return allocations_;
  }

 private:
  llvm::MapVector<ValueNodeView, Nonnull<const Value*>,
                  std::map<ValueNodeView, unsigned>>
      locals_;
  llvm::DenseSet<const AstNode*> bound_values_;
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
class Action : public Printable<Action> {
 public:
  enum class Kind {
    LocationAction,
    ValueExpressionAction,
    ExpressionAction,
    WitnessAction,
    StatementAction,
    DeclarationAction,
    ScopeAction,
    RecursiveAction,
    CleanUpAction,
    DestroyAction,
    TypeInstantiationAction
  };

  Action(const Value&) = delete;
  auto operator=(const Value&) -> Action& = delete;

  virtual ~Action() = default;

  void Print(llvm::raw_ostream& out) const;

  // Resets this Action to its initial state.
  void Clear() {
    CARBON_CHECK(!scope_.has_value());
    pos_ = 0;
    results_.clear();
  }

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto kind() const -> Kind { return kind_; }

  auto kind_string() const -> std::string_view;

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

  auto source_loc() const -> std::optional<SourceLocation> {
    return source_loc_;
  }

 protected:
  // Constructs an Action. `kind` must be the enumerator corresponding to the
  // most-derived type being constructed.
  explicit Action(std::optional<SourceLocation> source_loc, Kind kind)
      : source_loc_(source_loc), kind_(kind) {}
  std::optional<SourceLocation> source_loc_;

 private:
  int pos_ = 0;
  std::vector<Nonnull<const Value*>> results_;
  std::optional<RuntimeScope> scope_;

  const Kind kind_;
};

// An Action which implements evaluation of an Expression to produce an
// LocationValue.
class LocationAction : public Action {
 public:
  explicit LocationAction(Nonnull<const Expression*> expression)
      : Action(expression->source_loc(), Kind::LocationAction),
        expression_(expression) {}

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::LocationAction;
  }

  // The Expression this Action evaluates.
  auto expression() const -> const Expression& { return *expression_; }

 private:
  Nonnull<const Expression*> expression_;
};

// An Action which implements evaluation of an Expression to produce a `Value*`.
class ValueExpressionAction : public Action {
 public:
  explicit ValueExpressionAction(
      Nonnull<const Expression*> expression,
      std::optional<AllocationId> initialized_location = std::nullopt)
      : Action(expression->source_loc(), Kind::ValueExpressionAction),
        expression_(expression),
        location_received_(initialized_location) {}

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::ValueExpressionAction;
  }

  // The Expression this Action evaluates.
  auto expression() const -> const Expression& { return *expression_; }

  // The location provided for the initializing expression, if any.
  auto location_received() const -> std::optional<AllocationId> {
    return location_received_;
  }

 private:
  Nonnull<const Expression*> expression_;
  std::optional<AllocationId> location_received_;
};

// An Action which implements evaluation of a reference Expression to produce an
// `ReferenceExpressionValue*`. The `preserve_nested_categories` flag can be
// used to preserve values as `ReferenceExpressionValue` in nested value types,
// such as tuples.
class ExpressionAction : public Action {
 public:
  ExpressionAction(
      Nonnull<const Expression*> expression, bool preserve_nested_categories,
      std::optional<AllocationId> initialized_location = std::nullopt)
      : Action(expression->source_loc(), Kind::ExpressionAction),
        expression_(expression),
        location_received_(initialized_location),
        preserve_nested_categories_(preserve_nested_categories) {}

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::ExpressionAction;
  }

  // The Expression this Action evaluates.
  auto expression() const -> const Expression& { return *expression_; }

  // Returns whether direct descendent actions should preserve values as
  // `ReferenceExpressionValue*`s.
  auto preserve_nested_categories() const -> bool {
    return preserve_nested_categories_;
  }

  // The location provided for the initializing expression, if any.
  auto location_received() const -> std::optional<AllocationId> {
    return location_received_;
  }

 private:
  Nonnull<const Expression*> expression_;
  std::optional<AllocationId> location_received_;
  bool preserve_nested_categories_;
};

// An Action which implements the Instantiation of Type. The result is expressed
// as a Value.
class TypeInstantiationAction : public Action {
 public:
  explicit TypeInstantiationAction(Nonnull<const Value*> type,
                                   SourceLocation source_loc)
      : Action(source_loc, Kind::TypeInstantiationAction),
        type_(type),
        source_loc_(source_loc) {}

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::TypeInstantiationAction;
  }

  auto type() const -> Nonnull<const Value*> { return type_; }
  auto source_loc() const -> SourceLocation { return source_loc_; }

 private:
  Nonnull<const Value*> type_;
  SourceLocation source_loc_;
};

// An Action which implements evaluation of a Witness to resolve it in the
// local context.
class WitnessAction : public Action {
 public:
  explicit WitnessAction(Nonnull<const Witness*> witness,
                         SourceLocation source_loc)
      : Action(source_loc, Kind::WitnessAction), witness_(witness) {}

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::WitnessAction;
  }

  auto source_loc() -> SourceLocation {
    CARBON_CHECK(source_loc_);
    return *source_loc_;
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
  explicit StatementAction(Nonnull<const Statement*> statement,
                           std::optional<AllocationId> location_received)
      : Action(statement->source_loc(), Kind::StatementAction),
        statement_(statement),
        location_received_(location_received) {}

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::StatementAction;
  }

  // The Statement this Action executes.
  auto statement() const -> const Statement& { return *statement_; }

  // The location provided for the initializing expression, if any.
  auto location_received() const -> std::optional<AllocationId> {
    return location_received_;
  }

  // Sets the location provided to an initializing expression.
  auto set_location_created(AllocationId location_created) {
    CARBON_CHECK(!location_created_) << "location created set twice";
    location_created_ = location_created;
  }
  // Returns the location provided to an initializing expression, if any.
  auto location_created() const -> std::optional<AllocationId> {
    return location_created_;
  }

 private:
  Nonnull<const Statement*> statement_;
  std::optional<AllocationId> location_received_;
  std::optional<AllocationId> location_created_;
};

// Action which implements the run-time effects of executing a Declaration.
// Does not produce a result.
class DeclarationAction : public Action {
 public:
  explicit DeclarationAction(Nonnull<const Declaration*> declaration)
      : Action(declaration->source_loc(), Kind::DeclarationAction),
        declaration_(declaration) {}

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
  explicit CleanUpAction(RuntimeScope scope, SourceLocation source_loc)
      : Action(source_loc, Kind::CleanUpAction),
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
  // location: Location of the object to be destroyed
  // value:    The value to be destroyed
  //           In most cases the location address points to value
  //           In the case that the member of a class is to be destroyed,
  //           the location points to the address of the class object
  //           and the value is the member of the class
  explicit DestroyAction(Nonnull<const LocationValue*> location,
                         Nonnull<const Value*> value)
      : Action(std::nullopt, Kind::DestroyAction),
        location_(location),
        value_(value) {}

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::DestroyAction;
  }

  auto location() const -> Nonnull<const LocationValue*> { return location_; }

  auto value() const -> Nonnull<const Value*> { return value_; }

 private:
  Nonnull<const LocationValue*> location_;
  Nonnull<const Value*> value_;
};

// Action which does nothing except introduce a new scope into the action
// stack. This is useful when a distinct scope doesn't otherwise have an
// Action it can naturally be associated with. ScopeActions are not associated
// with AST nodes.
class ScopeAction : public Action {
 public:
  explicit ScopeAction(RuntimeScope scope)
      : Action(std::nullopt, Kind::ScopeAction) {
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
  explicit RecursiveAction() : Action(std::nullopt, Kind::RecursiveAction) {}

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::RecursiveAction;
  }
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_ACTION_H_
