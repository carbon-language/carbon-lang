// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_STATEMENT_H_
#define EXECUTABLE_SEMANTICS_AST_STATEMENT_H_

#include <list>

#include "executable_semantics/ast/expression.h"

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
  int line_num;
  StatementKind tag;

  union {
    const Expression* exp;

    struct {
      const Expression* lhs;
      const Expression* rhs;
    } assign;

    struct {
      const Expression* pat;
      const Expression* init;
    } variable_definition;

    struct {
      const Expression* cond;
      const Statement* then_stmt;
      const Statement* else_stmt;
    } if_stmt;

    const Expression* return_stmt;

    struct {
      const Statement* stmt;
      const Statement* next;
    } sequence;

    struct {
      const Statement* stmt;
    } block;

    struct {
      const Expression* cond;
      const Statement* body;
    } while_stmt;

    struct {
      const Expression* exp;
      std::list<std::pair<const Expression*, const Statement*>>* clauses;
    } match_stmt;

    struct {
      std::string* continuation_variable;
      const Statement* body;
    } continuation;

    struct {
      const Expression* argument;
    } run;

  } u;
};

auto MakeExpStmt(int line_num, const Expression* exp) -> const Statement*;
auto MakeAssign(int line_num, const Expression* lhs, const Expression* rhs)
    -> const Statement*;
auto MakeVarDef(int line_num, const Expression* pat, const Expression* init)
    -> const Statement*;
auto MakeIf(int line_num, const Expression* cond, const Statement* then_stmt,
            const Statement* else_stmt) -> const Statement*;
auto MakeReturn(int line_num, const Expression* e) -> const Statement*;
auto MakeSeq(int line_num, const Statement* s1, const Statement* s2)
    -> const Statement*;
auto MakeBlock(int line_num, const Statement* s) -> const Statement*;
auto MakeWhile(int line_num, const Expression* cond, const Statement* body)
    -> const Statement*;
auto MakeBreak(int line_num) -> const Statement*;
auto MakeContinue(int line_num) -> const Statement*;
auto MakeMatch(
    int line_num, const Expression* exp,
    std::list<std::pair<const Expression*, const Statement*>>* clauses)
    -> const Statement*;
// Returns an AST node for a continuation statement give its line number and
// contituent parts.
//
//     __continuation <continuation_variable> {
//       <body>
//     }
auto MakeContinuationStatement(int line_num, std::string continuation_variable,
                               const Statement* body) -> const Statement*;
// Returns an AST node for a run statement give its line number and argument.
//
//     __run <argument>;
auto MakeRun(int line_num, const Expression* argument) -> const Statement*;
// Returns an AST node for an await statement give its line number.
//
//     __await;
auto MakeAwait(int line_num) -> const Statement*;

void PrintStatement(const Statement*, int);

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_STATEMENT_H_
