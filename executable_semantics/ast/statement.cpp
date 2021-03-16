// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/statement.h"

#include <iostream>

namespace Carbon {

auto MakeExpStmt(int line_num, Expression* exp) -> Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = StatementKind::ExpressionStatement;
  s->u.exp = exp;
  return s;
}

auto MakeAssign(int line_num, Expression* lhs, Expression* rhs) -> Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = StatementKind::Assign;
  s->u.assign.lhs = lhs;
  s->u.assign.rhs = rhs;
  return s;
}

auto MakeVarDef(int line_num, Expression* pat, Expression* init) -> Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = StatementKind::VariableDefinition;
  s->u.variable_definition.pat = pat;
  s->u.variable_definition.init = init;
  return s;
}

auto MakeIf(int line_num, Expression* cond, Statement* then_stmt,
            Statement* else_stmt) -> Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = StatementKind::If;
  s->u.if_stmt.cond = cond;
  s->u.if_stmt.then_stmt = then_stmt;
  s->u.if_stmt.else_stmt = else_stmt;
  return s;
}

auto MakeWhile(int line_num, Expression* cond, Statement* body) -> Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = StatementKind::While;
  s->u.while_stmt.cond = cond;
  s->u.while_stmt.body = body;
  return s;
}

auto MakeBreak(int line_num) -> Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = StatementKind::Break;
  return s;
}

auto MakeContinue(int line_num) -> Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = StatementKind::Continue;
  return s;
}

auto MakeReturn(int line_num, Expression* e) -> Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = StatementKind::Return;
  s->u.return_stmt = e;
  return s;
}

auto MakeSeq(int line_num, Statement* s1, Statement* s2) -> Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = StatementKind::Sequence;
  s->u.sequence.stmt = s1;
  s->u.sequence.next = s2;
  return s;
}

auto MakeBlock(int line_num, Statement* stmt) -> Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = StatementKind::Block;
  s->u.block.stmt = stmt;
  return s;
}

auto MakeMatch(int line_num, Expression* exp,
               std::list<std::pair<Expression*, Statement*>>* clauses)
    -> Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = StatementKind::Match;
  s->u.match_stmt.exp = exp;
  s->u.match_stmt.clauses = clauses;
  return s;
}

// Returns an AST node for a delimit statement, given the source
// line number, the body statement, the variable name for the yielded value,
// the variable name for the captured continuation, and the handler
// statement.
auto MakeDelimitStmt(int line_num, Statement* body, std::string yield,
                     std::string cont, Statement* handler) -> Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = StatementKind::Delimit;
  s->u.delimit_stmt.body = body;
  s->u.delimit_stmt.yield_variable = new std::string(yield);
  s->u.delimit_stmt.continuation = new std::string(cont);
  s->u.delimit_stmt.handler = handler;
  return s;
}

// Returns an AST node for a yield stament given an expression
// that produces the yielded value.
auto MakeYieldStmt(int line_num, Expression* exp) -> Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = StatementKind::Yield;
  s->u.yield_stmt.exp = exp;
  return s;
}

// Returns an AST node for a resume statement given an expression
// that produces a continuation.
auto MakeResumeStmt(int line_num, Expression* exp) -> Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->tag = StatementKind::Resume;
  s->u.resume_stmt.exp = exp;
  return s;
}

void PrintStatement(Statement* s, int depth) {
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
      PrintExp(s->u.match_stmt.exp);
      std::cout << ") {";
      if (depth < 0 || depth > 1) {
        std::cout << std::endl;
        for (auto& clause : *s->u.match_stmt.clauses) {
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
      PrintExp(s->u.while_stmt.cond);
      std::cout << ")" << std::endl;
      PrintStatement(s->u.while_stmt.body, depth - 1);
      break;
    case StatementKind::Break:
      std::cout << "break;";
      break;
    case StatementKind::Continue:
      std::cout << "continue;";
      break;
    case StatementKind::VariableDefinition:
      std::cout << "var ";
      PrintExp(s->u.variable_definition.pat);
      std::cout << " = ";
      PrintExp(s->u.variable_definition.init);
      std::cout << ";";
      break;
    case StatementKind::ExpressionStatement:
      PrintExp(s->u.exp);
      std::cout << ";";
      break;
    case StatementKind::Assign:
      PrintExp(s->u.assign.lhs);
      std::cout << " = ";
      PrintExp(s->u.assign.rhs);
      std::cout << ";";
      break;
    case StatementKind::If:
      std::cout << "if (";
      PrintExp(s->u.if_stmt.cond);
      std::cout << ")" << std::endl;
      PrintStatement(s->u.if_stmt.then_stmt, depth - 1);
      std::cout << std::endl << "else" << std::endl;
      PrintStatement(s->u.if_stmt.else_stmt, depth - 1);
      break;
    case StatementKind::Return:
      std::cout << "return ";
      PrintExp(s->u.return_stmt);
      std::cout << ";";
      break;
    case StatementKind::Sequence:
      PrintStatement(s->u.sequence.stmt, depth);
      if (depth < 0 || depth > 1) {
        std::cout << std::endl;
      }
      PrintStatement(s->u.sequence.next, depth - 1);
      break;
    case StatementKind::Block:
      std::cout << "{" << std::endl;
      PrintStatement(s->u.block.stmt, depth - 1);
      std::cout << std::endl << "}" << std::endl;
      break;
    case StatementKind::Delimit:
      std::cout << "delimit" << std::endl;
      PrintStatement(s->u.delimit_stmt.body, depth - 1);
      std::cout << std::endl
                << "with (" << *s->u.delimit_stmt.yield_variable << ", "
                << *s->u.delimit_stmt.continuation << ")" << std::endl;
      PrintStatement(s->u.delimit_stmt.handler, depth - 1);
      std::cout << std::endl;
      break;
    case StatementKind::Yield:
      std::cout << "yield ";
      PrintExp(s->u.yield_stmt.exp);
      std::cout << ";";
      break;
    case StatementKind::Resume:
      std::cout << "resume ";
      PrintExp(s->u.resume_stmt.exp);
      std::cout << ";";
      break;
  }
}
}  // namespace Carbon
