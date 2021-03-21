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
  Delimit,  // An experimental "try" for delimited continuations.
  Yield,    // Pause the current continuation, return to the enclosing delimit.
  Resume,   // Restart a continuation.
  Continuation,  // Create a first-class continuation.
  Run,           // Run a continuation to the next await or until it finishes..
  Await,         // Pause execution of the continuation.
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
      std::string* continuation_variable;
      Statement* handler;
    } delimit_stmt;

    struct {
      Expression* operand;
    } yield_stmt;

    struct {
      Expression* operand;
    } resume_stmt;

    struct {
      std::string* continuation_variable;
      Statement* body;
    } continuation;

    struct {
      Expression* argument;
    } run;

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
// Returns a delimit statement's AST node, given its source
// location and constituent parts.
//
//      __delimit {
//         <body>
//      }
//      __catch ( <yieldedValueName>, <continuationName> ) {
//        <handler>
//      }
//
auto MakeDelimitStatement(int source_location, Statement* body,
                          std::string yield_variable,
                          std::string continuation_variable, Statement* handler)
    -> Statement*;
// Returns an AST node for a yield stament given an expression
// that produces the yielded value.
auto MakeYieldStatement(int line_num, Expression* operand) -> Statement*;
// Returns an AST node for a resume statement given an expression
// that produces a continuation.
auto MakeResumeStatement(int line_num, Expression* operand) -> Statement*;
// Returns an AST node for a continuation statement give its line number and
// contituent parts.
//
//     __continuation <continuation_variable> {
//       <body>
//     }
auto MakeContinuation(int line_num, std::string continuation_variable,
                      Statement* body) -> Statement*;
// Returns an AST node for a run statement give its line number and argument.
//
//     __run <argument>;
auto MakeRun(int line_num, Expression* argument) -> Statement*;
// Returns an AST node for an await statement give its line number.
//
//     __await;
auto MakeAwait(int line_num) -> Statement*;

void PrintStatement(Statement*, int);

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_STATEMENT_H_
