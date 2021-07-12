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

struct Assign {
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
  auto tag() const -> StatementKind;

  // Constructors
  static auto MakeExpStmt(int line_num, const Expression* exp)
      -> const Statement*;
  static auto MakeAssign(int line_num, const Expression* lhs,
                         const Expression* rhs) -> const Statement*;
  static auto MakeVarDef(int line_num, const Expression* pat,
                         const Expression* init) -> const Statement*;
  static auto MakeIf(int line_num, const Expression* cond,
                     const Statement* then_stmt, const Statement* else_stmt)
      -> const Statement*;
  static auto MakeReturn(int line_num, const Expression* e) -> const Statement*;
  static auto MakeSeq(int line_num, const Statement* s1, const Statement* s2)
      -> const Statement*;
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

  // Access to the alternatives
  const Expression* GetExpression() const;
  Assign GetAssign() const;
  VariableDefinition GetVariableDefinition() const;
  IfStatement GetIf() const;
  const Expression* GetReturn() const;
  Sequence GetSequence() const;
  Block GetBlock() const;
  While GetWhile() const;
  Match GetMatch() const;
  Continuation GetContinuation() const;
  Run GetRun() const;

  int line_num;

 private:
  /*
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
   */
  std::variant<Assign> value;
};

void PrintStatement(const Statement*, int);

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_STATEMENT_H_
