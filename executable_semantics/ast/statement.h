// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_STATEMENT_H_
#define EXECUTABLE_SEMANTICS_AST_STATEMENT_H_

#include <list>

#include "common/ostream.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/pattern.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

enum class StatementKind {
  ExpressionStatement,
  Assign,
  VariableDefinition,
  If,
  Return,
  Sequence,
  Block,
  While,
  Break,
  Continue,
  Match,
  Continuation,  // Create a first-class continuation.
  Run,           // Run a continuation to the next await or until it finishes..
  Await,         // Pause execution of the continuation.
};

struct Statement {
  void Print(llvm::raw_ostream& out) const { PrintDepth(-1, out); }
  void PrintDepth(int depth, llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

  inline auto tag() const -> StatementKind {
    return std::visit([](const auto& t) { return t.Kind; }, value);
  }

  int line_num;
};

struct ExpressionStatement {
  static constexpr StatementKind Kind = StatementKind::ExpressionStatement;
  const Expression* exp;
};

struct Assign {
  static constexpr StatementKind Kind = StatementKind::Assign;
  const Expression* lhs;
  const Expression* rhs;
};

struct VariableDefinition {
  static constexpr StatementKind Kind = StatementKind::VariableDefinition;
  const Pattern* pat;
  const Expression* init;
};

struct If {
  static constexpr StatementKind Kind = StatementKind::If;
  const Expression* cond;
  const Statement* then_stmt;
  const Statement* else_stmt;
};

struct Return {
  static constexpr StatementKind Kind = StatementKind::Return;
  const Expression* exp;
  bool is_omitted_exp;
};

struct Sequence {
  static constexpr StatementKind Kind = StatementKind::Sequence;
  const Statement* stmt;
  const Statement* next;
};

struct Block {
  static constexpr StatementKind Kind = StatementKind::Block;
  const Statement* stmt;
};

struct While {
  static constexpr StatementKind Kind = StatementKind::While;
  const Expression* cond;
  const Statement* body;
};

struct Break {
  static constexpr StatementKind Kind = StatementKind::Break;
};

struct Continue {
  static constexpr StatementKind Kind = StatementKind::Continue;
};

struct Match {
  static constexpr StatementKind Kind = StatementKind::Match;
  const Expression* exp;
  std::list<std::pair<const Pattern*, const Statement*>>* clauses;
};

// A continuation statement.
//
//     __continuation <continuation_variable> {
//       <body>
//     }
struct Continuation {
  static constexpr StatementKind Kind = StatementKind::Continuation;
  std::string continuation_variable;
  const Statement* body;
};

// A run statement.
//
//     __run <argument>;
struct Run {
  static constexpr StatementKind Kind = StatementKind::Run;
  const Expression* argument;
};

// An await statement.
//
//    __await;
struct Await {
  static constexpr StatementKind Kind = StatementKind::Await;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_STATEMENT_H_
