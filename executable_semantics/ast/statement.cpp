// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/statement.h"

#include <iostream>

#include "common/check.h"

namespace Carbon {

auto Statement::GetExpressionStatement() const -> const ExpressionStatement& {
  return std::get<ExpressionStatement>(value);
}

auto Statement::GetAssign() const -> const Assign& {
  return std::get<Assign>(value);
}

auto Statement::GetVariableDefinition() const -> const VariableDefinition& {
  return std::get<VariableDefinition>(value);
}

auto Statement::GetIf() const -> const If& { return std::get<If>(value); }

auto Statement::GetReturn() const -> const Return& {
  return std::get<Return>(value);
}

auto Statement::GetSequence() const -> const Sequence& {
  return std::get<Sequence>(value);
}

auto Statement::GetBlock() const -> const Block& {
  return std::get<Block>(value);
}

auto Statement::GetWhile() const -> const While& {
  return std::get<While>(value);
}

auto Statement::GetBreak() const -> const Break& {
  return std::get<Break>(value);
}

auto Statement::GetContinue() const -> const Continue& {
  return std::get<Continue>(value);
}

auto Statement::GetMatch() const -> const Match& {
  return std::get<Match>(value);
}

auto Statement::GetContinuation() const -> const Continuation& {
  return std::get<Continuation>(value);
}

auto Statement::GetRun() const -> const Run& { return std::get<Run>(value); }

auto Statement::GetAwait() const -> const Await& {
  return std::get<Await>(value);
}

auto Statement::MakeExpressionStatement(int line_num, const Expression* exp)
    -> const Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->value = ExpressionStatement({.exp = exp});
  return s;
}

auto Statement::MakeAssign(int line_num, const Expression* lhs,
                           const Expression* rhs) -> const Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->value = Assign({.lhs = lhs, .rhs = rhs});
  return s;
}

auto Statement::MakeVariableDefinition(int line_num, const Expression* pat,
                                       const Expression* init)
    -> const Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->value = VariableDefinition({.pat = pat, .init = init});
  return s;
}

auto Statement::MakeIf(int line_num, const Expression* cond,
                       const Statement* then_stmt, const Statement* else_stmt)
    -> const Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->value = If({.cond = cond, .then_stmt = then_stmt, .else_stmt = else_stmt});
  return s;
}

auto Statement::MakeWhile(int line_num, const Expression* cond,
                          const Statement* body) -> const Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->value = While({.cond = cond, .body = body});
  return s;
}

auto Statement::MakeBreak(int line_num) -> const Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->value = Break();
  return s;
}

auto Statement::MakeContinue(int line_num) -> const Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->value = Continue();
  return s;
}

auto Statement::MakeReturn(int line_num, const Expression* e)
    -> const Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->value = Return({.exp = e});
  return s;
}

auto Statement::MakeSequence(int line_num, const Statement* s1,
                             const Statement* s2) -> const Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->value = Sequence({.stmt = s1, .next = s2});
  return s;
}

auto Statement::MakeBlock(int line_num, const Statement* stmt)
    -> const Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->value = Block({.stmt = stmt});
  return s;
}

auto Statement::MakeMatch(
    int line_num, const Expression* exp,
    std::list<std::pair<const Expression*, const Statement*>>* clauses)
    -> const Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->value = Match({.exp = exp, .clauses = clauses});
  return s;
}

// Returns an AST node for a continuation statement give its line number and
// parts.
auto Statement::MakeContinuation(int line_num,
                                 std::string continuation_variable,
                                 const Statement* body) -> const Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->value =
      Continuation({.continuation_variable = std::move(continuation_variable),
                    .body = body});
  return s;
}

// Returns an AST node for a run statement give its line number and argument.
auto Statement::MakeRun(int line_num, const Expression* argument)
    -> const Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->value = Run({.argument = argument});
  return s;
}

// Returns an AST node for an await statement give its line number.
auto Statement::MakeAwait(int line_num) -> const Statement* {
  auto* s = new Statement();
  s->line_num = line_num;
  s->value = Await();
  return s;
}

void PrintStatement(const Statement* s, int depth) {
  if (!s) {
    return;
  }
  if (depth == 0) {
    std::cout << " ... ";
    return;
  }
  switch (s->tag()) {
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
      PrintExp(s->GetExpressionStatement().exp);
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
      PrintExp(s->GetReturn().exp);
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
      std::cout << "continuation " << s->GetContinuation().continuation_variable
                << " ";
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
