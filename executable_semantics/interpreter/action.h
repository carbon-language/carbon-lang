// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_ACTION_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_ACTION_H_

#include <map>
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

// A DynamicScope manages and provides access to the storage for names that are
// not compile-time constants.
class DynamicScope {
 public:
  // Returns a DynamicScope whose Get() operation for a given name returns the
  // storage owned by the first entry in `scopes` that defines that name. This
  // behavior is closely analogous to a `[&]` capture in C++, hence the name.
  // `scopes` must contain at least one entry, and all entries must be backed
  // by the same Heap.
  static auto Capture(const std::vector<Nonnull<const DynamicScope*>>& scopes)
      -> DynamicScope;

  // Constructs a DynamicScope that allocates storage in `heap`.
  explicit DynamicScope(Nonnull<HeapAllocationInterface*> heap) : heap_(heap) {}

  // Moving a DynamicScope transfers ownership of its allocations.
  DynamicScope(DynamicScope&&) noexcept;
  auto operator=(DynamicScope&&) noexcept -> DynamicScope&;

  // Deallocates any allocations in this scope from `heap`.
  ~DynamicScope();

  void Print(llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

  // Allocates storage for `named_entity` in `heap`, and initializes it with
  // `value`.
  void Initialize(NamedEntityView named_entity, Nonnull<const Value*> value);

  // Transfers the names and allocations from `other` into *this. The two
  // scopes must not define the same name, and must be backed by the same Heap.
  void Merge(DynamicScope other);

  // Returns the local storage for named_entity, if it has storage local to
  // this scope.
  auto Get(NamedEntityView named_entity) const
      -> std::optional<Nonnull<const LValue*>>;

 private:
  std::map<NamedEntityView, Nonnull<const LValue*>> locals_;
  std::vector<AllocationId> allocations_;
  Nonnull<HeapAllocationInterface*> heap_;
};

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

  void AddResult(Nonnull<const Value*> result) { results_.push_back(result); }

  void Clear() {
    CHECK(!scope_.has_value());
    pos_ = 0;
    results_.clear();
  }

  // Associates this action with a new scope, with initial state `scope`.
  // Values that are local to this scope will be deallocated when this
  // Action is completed or unwound. Can only be called once on a given
  // Action.
  void StartScope(DynamicScope scope) {
    CHECK(!scope_.has_value());
    scope_ = std::move(scope);
  }

  // Returns the scope associated with this Action, if any.
  auto scope() -> std::optional<DynamicScope>& { return scope_; }

  static void PrintList(const Stack<Nonnull<Action*>>& ls,
                        llvm::raw_ostream& out);

  void Print(llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto kind() const -> Kind { return kind_; }

  // The position or state of the action. Starts at 0 and goes up to the number
  // of subexpressions.
  //
  // pos indicates how many of the entries in the following `results` vector
  // will be filled in the next time this action is active.
  // For each i < pos, results[i] contains a pointer to a Value.
  auto pos() const -> int { return pos_; }

  void set_pos(int pos) { this->pos_ = pos; }

  // Results from a subexpression.
  auto results() const -> const std::vector<Nonnull<const Value*>>& {
    return results_;
  }

  virtual ~Action() = default;

 protected:
  // Constructs an Action. `kind` must be the enumerator corresponding to the
  // most-derived type being constructed.
  explicit Action(Kind kind) : kind_(kind) {}

 private:
  int pos_ = 0;
  std::vector<Nonnull<const Value*>> results_;
  std::optional<DynamicScope> scope_;

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
  explicit ScopeAction(DynamicScope scope) : Action(Kind::ScopeAction) {
    StartScope(std::move(scope));
  }

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::ScopeAction;
  }
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_ACTION_H_
