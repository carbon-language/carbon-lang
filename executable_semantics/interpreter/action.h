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
  // Constructs an Action. `tag` must be the enumerator corresponding to the
  // most-derived type being constructed.
  explicit Action(Kind kind) : kind_(kind) {}

 private:
  int pos_ = 0;
  std::vector<Nonnull<const Value*>> results_;

  const Kind kind_;
};

class LValAction : public Action {
 public:
  explicit LValAction(Nonnull<const Expression*> exp)
      : Action(Kind::LValAction), exp(exp) {}

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::LValAction;
  }

  auto Exp() const -> Nonnull<const Expression*> { return exp; }

 private:
  Nonnull<const Expression*> exp;
};

class ExpressionAction : public Action {
 public:
  explicit ExpressionAction(Nonnull<const Expression*> exp)
      : Action(Kind::ExpressionAction), exp(exp) {}

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::ExpressionAction;
  }

  auto Exp() const -> Nonnull<const Expression*> { return exp; }

 private:
  Nonnull<const Expression*> exp;
};

class PatternAction : public Action {
 public:
  explicit PatternAction(Nonnull<const Pattern*> pat)
      : Action(Kind::PatternAction), pat(pat) {}

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::PatternAction;
  }

  auto Pat() const -> Nonnull<const Pattern*> { return pat; }

 private:
  Nonnull<const Pattern*> pat;
};

class StatementAction : public Action {
 public:
  explicit StatementAction(Nonnull<const Statement*> stmt)
      : Action(Kind::StatementAction), stmt(stmt) {}

  static auto classof(const Action* action) -> bool {
    return action->kind() == Kind::StatementAction;
  }

  auto Stmt() const -> Nonnull<const Statement*> { return stmt; }

 private:
  Nonnull<const Statement*> stmt;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_ACTION_H_
