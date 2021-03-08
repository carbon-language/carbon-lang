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
  Delimit,
  Yield,
  Resume
};

struct Statement {
  int line_num;
  StatementKind tag;

  union {
    Expression* exp;

    struct {
      Expression* lhs;
      Expression* rhs;
    } assign;

    struct {
      Expression* pat;
      Expression* init;
    } variable_definition;

    struct {
      Expression* cond;
      Statement* then_stmt;
      Statement* else_stmt;
    } if_stmt;

    Expression* return_stmt;

    struct {
      Statement* stmt;
      Statement* next;
    } sequence;

    struct {
      Statement* stmt;
    } block;

    struct {
      Expression* cond;
      Statement* body;
    } while_stmt;

    struct {
      Expression* exp;
      std::list<std::pair<Expression*, Statement*>>* clauses;
    } match_stmt;

    struct {
      Statement* body;
      std::string* yield_variable;
      std::string* continuation;
      Statement* handler;
    } delimit_stmt;

    struct {
      Expression* exp;
    } yield_stmt;

    struct {
      Expression* exp;
    } resume_stmt;

  } u;
};

auto MakeExpStmt(int line_num, Expression* exp) -> Statement*;
auto MakeAssign(int line_num, Expression* lhs, Expression* rhs) -> Statement*;
auto MakeVarDef(int line_num, Expression* pat, Expression* init) -> Statement*;
auto MakeIf(int line_num, Expression* cond, Statement* then_stmt,
            Statement* else_stmt) -> Statement*;
auto MakeReturn(int line_num, Expression* e) -> Statement*;
auto MakeSeq(int line_num, Statement* s1, Statement* s2) -> Statement*;
auto MakeBlock(int line_num, Statement* s) -> Statement*;
auto MakeWhile(int line_num, Expression* cond, Statement* body) -> Statement*;
auto MakeBreak(int line_num) -> Statement*;
auto MakeContinue(int line_num) -> Statement*;
auto MakeMatch(int line_num, Expression* exp,
               std::list<std::pair<Expression*, Statement*>>* clauses)
    -> Statement*;
auto MakeDelimitStmt(int line_num, Statement*, std::string, std::string,
                     Statement*) -> Statement*;
auto MakeYieldStmt(int line_num, Expression*) -> Statement*;
auto MakeResumeStmt(int line_num, Expression*) -> Statement*;

void PrintStatement(Statement*, int);

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_STATEMENT_H_
