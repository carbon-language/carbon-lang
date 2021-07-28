// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_STATEMENT_H_
#define EXECUTABLE_SEMANTICS_AST_STATEMENT_H_

#include <list>

#include "common/ostream.h"
#include "executable_semantics/ast/expression.h"
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

struct Statement;

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
  const Expression* pat;
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
  std::list<std::pair<const Expression*, const Statement*>>* clauses;
};

struct Continuation {
  static constexpr StatementKind Kind = StatementKind::Continuation;
  std::string continuation_variable;
  const Statement* body;
};

struct Run {
  static constexpr StatementKind Kind = StatementKind::Run;
  const Expression* argument;
};

struct Await {
  static constexpr StatementKind Kind = StatementKind::Await;
};

struct Statement {
  // Constructors
  static auto MakeExpressionStatement(int line_num, const Expression* exp)
      -> const Statement*;
  static auto MakeAssign(int line_num, const Expression* lhs,
                         const Expression* rhs) -> const Statement*;
  static auto MakeVariableDefinition(int line_num, const Expression* pat,
                                     const Expression* init)
      -> const Statement*;
  static auto MakeIf(int line_num, const Expression* cond,
                     const Statement* then_stmt, const Statement* else_stmt)
      -> const Statement*;
  static auto MakeReturn(int line_num, const Expression* e) -> const Statement*;
  static auto MakeSequence(int line_num, const Statement* s1,
                           const Statement* s2) -> const Statement*;
  static auto MakeBlock(int line_num, const Statement* s) -> const Statement*;
  static auto MakeWhile(int line_num, const Expression* cond,
                        const Statement* body) -> const Statement*;
  static auto MakeBreak(int line_num) -> const Statement*;
  static auto MakeContinue(int line_num) -> const Statement*;
  static auto MakeMatch(
      int line_num, const Expression* exp,
      std::list<std::pair<const Expression*, const Statement*>>* clauses)
      -> const Statement*;
  // Returns an AST node for a continuation statement give its line number and
  // contituent parts.
  //
  //     __continuation <continuation_variable> {
  //       <body>
  //     }
  static auto MakeContinuation(int line_num, std::string continuation_variable,
                               const Statement* body) -> const Statement*;
  // Returns an AST node for a run statement give its line number and argument.
  //
  //     __run <argument>;
  static auto MakeRun(int line_num, const Expression* argument)
      -> const Statement*;
  // Returns an AST node for an await statement give its line number.
  //
  //     __await;
  static auto MakeAwait(int line_num) -> const Statement*;

  auto GetExpressionStatement() const -> const ExpressionStatement&;
  auto GetAssign() const -> const Assign&;
  auto GetVariableDefinition() const -> const VariableDefinition&;
  auto GetIf() const -> const If&;
  auto GetReturn() const -> const Return&;
  auto GetSequence() const -> const Sequence&;
  auto GetBlock() const -> const Block&;
  auto GetWhile() const -> const While&;
  auto GetBreak() const -> const Break&;
  auto GetContinue() const -> const Continue&;
  auto GetMatch() const -> const Match&;
  auto GetContinuation() const -> const Continuation&;
  auto GetRun() const -> const Run&;
  auto GetAwait() const -> const Await&;

  void Print(llvm::raw_ostream& out) const { PrintDepth(-1, out); }
  void PrintDepth(int depth, llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::outs()); }

  inline auto tag() const -> StatementKind {
    return std::visit([](const auto& t) { return t.Kind; }, value);
  }

  int line_num;

 private:
  std::variant<ExpressionStatement, Assign, VariableDefinition, If, Return,
               Sequence, Block, While, Break, Continue, Match, Continuation,
               Run, Await>
      value;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_STATEMENT_H_
