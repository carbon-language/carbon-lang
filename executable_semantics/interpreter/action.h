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

  // The position or state of the action. Starts at 0 and goes up to the number
  // of subexpressions.
  //
  // pos indicates how many of the entries in the following `results` vector
  // will be filled in the next time this action is active.
  // For each i < pos, results[i] contains a pointer to a Value.
  auto Pos() const -> int { return pos; }

  // Results from a subexpression.
  auto Results() const -> const std::vector<Nonnull<const Value*>>& {
    return results;
  }

  void SetPos(int pos) { this->pos = pos; }

  void AddResult(Nonnull<const Value*> result) { results.push_back(result); }

  void Clear() {
    pos = 0;
    results.clear();
  }

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto Tag() const -> Kind { return kind; }

  static void PrintList(const Stack<Nonnull<Action*>>& ls,
                        llvm::raw_ostream& out);

  void Print(llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

 protected:
  // Constructs an Action. `tag` must be the enumerator corresponding to the
  // most-derived type being constructed.
  explicit Action(Kind kind) : kind(kind) {}

 private:
  int pos = 0;
  std::vector<Nonnull<const Value*>> results;

  const Kind kind;
};

class LValAction : public Action {
 public:
  explicit LValAction(Nonnull<const Expression*> exp)
      : Action(Kind::LValAction), exp(exp) {}

  static auto classof(const Action* action) -> bool {
    return action->Tag() == Kind::LValAction;
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
    return action->Tag() == Kind::ExpressionAction;
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
    return action->Tag() == Kind::PatternAction;
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
    return action->Tag() == Kind::StatementAction;
  }

  auto Stmt() const -> Nonnull<const Statement*> { return stmt; }

 private:
  Nonnull<const Statement*> stmt;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_ACTION_H_
