// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/statement.h"

#include <iostream>

#include "executable_semantics/common/check.h"

namespace Carbon {

const Expression* Statement::GetExpression() const {
  CHECK(tag == StatementKind::ExpressionStatement);
  return u.exp;
}

Assignment Statement::GetAssign() const {
  CHECK(tag == StatementKind::Assign);
  return u.assign;
}

VariableDefinition Statement::GetVariableDefinition() const {
  CHECK(tag == StatementKind::VariableDefinition);
  return u.variable_definition;
}

IfStatement Statement::GetIf() const {
  CHECK(tag == StatementKind::If);
  return u.if_stmt;
}

const Expression* Statement::GetReturn() const {
  CHECK(tag == StatementKind::Return);
  return u.return_stmt;
}

Sequence Statement::GetSequence() const {
  CHECK(tag == StatementKind::Sequence);
  return u.sequence;
}

Block Statement::GetBlock() const {
  CHECK(tag == StatementKind::Block);
  return u.block;
}

While Statement::GetWhile() const {
  CHECK(tag == StatementKind::While);
  return u.while_stmt;
}

Match Statement::GetMatch() const {
  CHECK(tag == StatementKind::Match);
  return u.match_stmt;
}

Continuation Statement::GetContinuation() const {
  CHECK(tag == StatementKind::Continuation);
  return u.continuation;
}

Run Statement::GetRun() const {
  CHECK(tag == StatementKind::Run);
  return u.run;
}

auto Statement::MakeExpStmt(int line_num, const Expression* exp)
    -> const Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = StatementKind::ExpressionStatement;
  s->u.exp = exp;
  return s;
}

auto Statement::MakeAssign(int line_num, const Expression* lhs,
                           const Expression* rhs) -> const Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = StatementKind::Assign;
  s->u.assign.lhs = lhs;
  s->u.assign.rhs = rhs;
  return s;
}

auto Statement::MakeVarDef(int line_num, const Expression* pat,
                           const Expression* init) -> const Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = StatementKind::VariableDefinition;
  s->u.variable_definition.pat = pat;
  s->u.variable_definition.init = init;
  return s;
}

auto Statement::MakeIf(int line_num, const Expression* cond,
                       const Statement* then_stmt, const Statement* else_stmt)
    -> const Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = StatementKind::If;
  s->u.if_stmt.cond = cond;
  s->u.if_stmt.then_stmt = then_stmt;
  s->u.if_stmt.else_stmt = else_stmt;
  return s;
}

auto Statement::MakeWhile(int line_num, const Expression* cond,
                          const Statement* body) -> const Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = StatementKind::While;
  s->u.while_stmt.cond = cond;
  s->u.while_stmt.body = body;
  return s;
}

auto Statement::MakeBreak(int line_num) -> const Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = StatementKind::Break;
  return s;
}

auto Statement::MakeContinue(int line_num) -> const Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = StatementKind::Continue;
  return s;
}

auto Statement::MakeReturn(int line_num, const Expression* e)
    -> const Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = StatementKind::Return;
  s->u.return_stmt = e;
  return s;
}

auto Statement::MakeSeq(int line_num, const Statement* s1, const Statement* s2)
    -> const Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = StatementKind::Sequence;
  s->u.sequence.stmt = s1;
  s->u.sequence.next = s2;
  return s;
}

auto Statement::MakeBlock(int line_num, const Statement* stmt)
    -> const Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = StatementKind::Block;
  s->u.block.stmt = stmt;
  return s;
}

auto Statement::MakeMatch(
    int line_num, const Expression* exp,
    std::list<std::pair<const Expression*, const Statement*>>* clauses)
    -> const Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = StatementKind::Match;
  s->u.match_stmt.exp = exp;
  s->u.match_stmt.clauses = clauses;
  return s;
}

// Returns an AST node for a continuation statement give its line number and
// parts.
auto Statement::MakeContinuation(int line_num,
                                 std::string continuation_variable,
                                 const Statement* body) -> const Statement* {
  auto* continuation = new Statement();
  continuation->line_num = line_num;
  continuation->tag = StatementKind::Continuation;
  continuation->u.continuation.continuation_variable =
      new std::string(continuation_variable);
  continuation->u.continuation.body = body;
  return continuation;
}

// Returns an AST node for a run statement give its line number and argument.
auto Statement::MakeRun(int line_num, const Expression* argument)
    -> const Statement* {
  auto* run = new Statement();
  run->line_num = line_num;
  run->tag = StatementKind::Run;
  run->u.run.argument = argument;
  return run;
}

// Returns an AST node for an await statement give its line number.
auto Statement::MakeAwait(int line_num) -> const Statement* {
  auto* await = new Statement();
  await->line_num = line_num;
  await->tag = StatementKind::Await;
  return await;
}

void PrintStatement(const Statement* s, int depth) {
  if (!s) {
    return;
  }
  if (depth == 0) {
    std::cout << " ... ";
    return;
  }
  switch (s->tag) {
    case StatementKind::Match:
      std::cout << "match (";
      PrintExp(s->GetMatch().exp);
      std::cout << ") {";
      if (depth < 0 || depth > 1) {
        std::cout << std::endl;
        for (auto& clause : *s->GetMatch().clauses) {
          std::cout << "case ";
          PrintExp(clause.first);
          std::cout << " =>" << std::endl;
          PrintStatement(clause.second, depth - 1);
          std::cout << std::endl;
        }
      } else {
        std::cout << "...";
      }
      std::cout << "}";
      break;
    case StatementKind::While:
      std::cout << "while (";
      PrintExp(s->GetWhile().cond);
      std::cout << ")" << std::endl;
      PrintStatement(s->GetWhile().body, depth - 1);
      break;
    case StatementKind::Break:
      std::cout << "break;";
      break;
    case StatementKind::Continue:
      std::cout << "continue;";
      break;
    case StatementKind::VariableDefinition:
      std::cout << "var ";
      PrintExp(s->GetVariableDefinition().pat);
      std::cout << " = ";
      PrintExp(s->GetVariableDefinition().init);
      std::cout << ";";
      break;
    case StatementKind::ExpressionStatement:
      PrintExp(s->GetExpression());
      std::cout << ";";
      break;
    case StatementKind::Assign:
      PrintExp(s->GetAssign().lhs);
      std::cout << " = ";
      PrintExp(s->GetAssign().rhs);
      std::cout << ";";
      break;
    case StatementKind::If:
      std::cout << "if (";
      PrintExp(s->GetIf().cond);
      std::cout << ")" << std::endl;
      PrintStatement(s->GetIf().then_stmt, depth - 1);
      std::cout << std::endl << "else" << std::endl;
      PrintStatement(s->GetIf().else_stmt, depth - 1);
      break;
    case StatementKind::Return:
      std::cout << "return ";
      PrintExp(s->GetReturn());
      std::cout << ";";
      break;
    case StatementKind::Sequence:
      PrintStatement(s->GetSequence().stmt, depth);
      if (depth < 0 || depth > 1) {
        std::cout << std::endl;
      } else {
        std::cout << " ";
      }
      PrintStatement(s->GetSequence().next, depth - 1);
      break;
    case StatementKind::Block:
      std::cout << "{";
      if (depth < 0 || depth > 1) {
        std::cout << std::endl;
      }
      PrintStatement(s->GetBlock().stmt, depth);
      if (depth < 0 || depth > 1) {
        std::cout << std::endl;
      }
      std::cout << "}";
      if (depth < 0 || depth > 1) {
        std::cout << std::endl;
      }
      break;
    case StatementKind::Continuation:
      std::cout << "continuation "
                << *s->GetContinuation().continuation_variable << " ";
      if (depth < 0 || depth > 1) {
        std::cout << std::endl;
      }
      PrintStatement(s->GetContinuation().body, depth - 1);
      if (depth < 0 || depth > 1) {
        std::cout << std::endl;
      }
      break;
    case StatementKind::Run:
      std::cout << "run ";
      PrintExp(s->GetRun().argument);
      std::cout << ";";
      break;
    case StatementKind::Await:
      std::cout << "await;";
      break;
  }
}
}  // namespace Carbon
