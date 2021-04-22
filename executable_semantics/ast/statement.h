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

struct Statement;

struct Assignment {
  const Expression* lhs;
  const Expression* rhs;
};

struct VariableDefinition {
  const Expression* pat;
  const Expression* init;
};

struct IfStatement {
  const Expression* cond;
  const Statement* then_stmt;
  const Statement* else_stmt;
};

struct Sequence {
  const Statement* stmt;
  const Statement* next;
};

struct Block {
  const Statement* stmt;
};

struct While {
  const Expression* cond;
  const Statement* body;
};

struct Match {
  const Expression* exp;
  std::list<std::pair<const Expression*, const Statement*>>* clauses;
};

struct Continuation {
  std::string* continuation_variable;
  const Statement* body;
};

struct Run {
  const Expression* argument;
};

struct Statement {
  // TODO: change Statement to a class and make all members private
  int line_num;
  StatementKind tag;

  const Expression* GetExpression() const;
  Assignment GetAssign() const;
  VariableDefinition GetVariableDefinition() const;
  IfStatement GetIf() const;
  const Expression* GetReturn() const;
  Sequence GetSequence() const;
  Block GetBlock() const;
  While GetWhile() const;
  Match GetMatch() const;
  Continuation GetContinuation() const;
  Run GetRun() const;

 private:
  union {
    const Expression* exp;
    Assignment assign;
    VariableDefinition variable_definition;
    IfStatement if_stmt;
    const Expression* return_stmt;
    Sequence sequence;
    Block block;
    While while_stmt;
    Match match_stmt;
    Continuation continuation;
    Run run;
  } u;

  // TODO: replace these constructors functions with real constructors
  //
  // RANT: The following long list of friend declarations is an
  // example of a problem in the design of C++. It is so focused on
  // classes and objects that it fails for modular procedural
  // programming. There are better ways to control access, for
  // example, going back to the module system of in CLU programming
  // language in the 1970's. -Jeremy
  friend auto MakeExpStmt(int line_num, const Expression* exp)
      -> const Statement*;
  friend auto MakeAssign(int line_num, const Expression* lhs,
                         const Expression* rhs) -> const Statement*;
  friend auto MakeVarDef(int line_num, const Expression* pat,
                         const Expression* init) -> const Statement*;
  friend auto MakeIf(int line_num, const Expression* cond,
                     const Statement* then_stmt, const Statement* else_stmt)
      -> const Statement*;
  friend auto MakeReturn(int line_num, const Expression* e) -> const Statement*;
  friend auto MakeSeq(int line_num, const Statement* s1, const Statement* s2)
      -> const Statement*;
  friend auto MakeBlock(int line_num, const Statement* s) -> const Statement*;
  friend auto MakeWhile(int line_num, const Expression* cond,
                        const Statement* body) -> const Statement*;
  friend auto MakeBreak(int line_num) -> const Statement*;
  friend auto MakeContinue(int line_num) -> const Statement*;
  friend auto MakeMatch(
      int line_num, const Expression* exp,
      std::list<std::pair<const Expression*, const Statement*>>* clauses)
      -> const Statement*;
  friend auto MakeContinuationStatement(int line_num,
                                        std::string continuation_variable,
                                        const Statement* body)
      -> const Statement*;
  friend auto MakeRun(int line_num, const Expression* argument)
      -> const Statement*;
  friend auto MakeAwait(int line_num) -> const Statement*;
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
