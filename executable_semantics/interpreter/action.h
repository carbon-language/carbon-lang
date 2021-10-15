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
#include "executable_semantics/interpreter/stack.h"
#include "executable_semantics/interpreter/value.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

class Action {
 public:
  enum class Kind {
    LValAction,
    ExpressionAction,
    PatternAction,
    StatementAction,
  };

  Action(const Value&) = delete;
  Action& operator=(const Value&) = delete;

  void AddResult(Nonnull<const Value*> result) { results_.push_back(result); }

  void Clear() {
    pos_ = 0;
    results_.clear();
  }

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

 protected:
  // Constructs an Action. `kind` must be the enumerator corresponding to the
  // most-derived type being constructed.
  explicit Action(Kind kind) : kind_(kind) {}

 private:
  int pos_ = 0;
  std::vector<Nonnull<const Value*>> results_;

  const Kind kind_;
};

class LValAction : public Action {
 public:
  explicit LValAction(Nonnull<const Expression*> expression)
      : Action(Kind::LValAction), expression_(expression) {}

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::LValAction;
  }

  auto expression() const -> const Expression& { return *expression_; }

 private:
  Nonnull<const Expression*> expression_;
};

class ExpressionAction : public Action {
 public:
  explicit ExpressionAction(Nonnull<const Expression*> expression)
      : Action(Kind::ExpressionAction), expression_(expression) {}

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::ExpressionAction;
  }

  auto expression() const -> const Expression& { return *expression_; }

 private:
  Nonnull<const Expression*> expression_;
};

class PatternAction : public Action {
 public:
  explicit PatternAction(Nonnull<const Pattern*> pattern)
      : Action(Kind::PatternAction), pattern_(pattern) {}

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::PatternAction;
  }

  auto pattern() const -> const Pattern& { return *pattern_; }

 private:
  Nonnull<const Pattern*> pattern_;
};

class StatementAction : public Action {
 public:
  explicit StatementAction(Nonnull<const Statement*> statement)
      : Action(Kind::StatementAction), statement_(statement) {}

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::StatementAction;
  }

  auto statement() const -> const Statement& { return *statement_; }

 private:
  Nonnull<const Statement*> statement_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_ACTION_H_
