// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/statement.h"

#include "common/check.h"
#include "executable_semantics/common/arena.h"

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
  auto* s = global_arena->New<Statement>();
  s->line_num = line_num;
  s->value = ExpressionStatement({.exp = exp});
  return s;
}

auto Statement::MakeAssign(int line_num, const Expression* lhs,
                           const Expression* rhs) -> const Statement* {
  auto* s = global_arena->New<Statement>();
  s->line_num = line_num;
  s->value = Assign({.lhs = lhs, .rhs = rhs});
  return s;
}

auto Statement::MakeVariableDefinition(int line_num, const Pattern* pat,
                                       const Expression* init)
    -> const Statement* {
  auto* s = global_arena->New<Statement>();
  s->line_num = line_num;
  s->value = VariableDefinition({.pat = pat, .init = init});
  return s;
}

auto Statement::MakeIf(int line_num, const Expression* cond,
                       const Statement* then_stmt, const Statement* else_stmt)
    -> const Statement* {
  auto* s = global_arena->New<Statement>();
  s->line_num = line_num;
  s->value = If({.cond = cond, .then_stmt = then_stmt, .else_stmt = else_stmt});
  return s;
}

auto Statement::MakeWhile(int line_num, const Expression* cond,
                          const Statement* body) -> const Statement* {
  auto* s = global_arena->New<Statement>();
  s->line_num = line_num;
  s->value = While({.cond = cond, .body = body});
  return s;
}

auto Statement::MakeBreak(int line_num) -> const Statement* {
  auto* s = global_arena->New<Statement>();
  s->line_num = line_num;
  s->value = Break();
  return s;
}

auto Statement::MakeContinue(int line_num) -> const Statement* {
  auto* s = global_arena->New<Statement>();
  s->line_num = line_num;
  s->value = Continue();
  return s;
}

auto Statement::MakeReturn(int line_num, const Expression* exp,
                           bool is_omitted_exp) -> const Statement* {
  auto* s = global_arena->New<Statement>();
  s->line_num = line_num;
  if (exp == nullptr) {
    CHECK(is_omitted_exp);
    exp = global_arena->New<TupleLiteral>(line_num);
  }
  s->value = Return({.exp = exp, .is_omitted_exp = is_omitted_exp});
  return s;
}

auto Statement::MakeSequence(int line_num, const Statement* s1,
                             const Statement* s2) -> const Statement* {
  auto* s = global_arena->New<Statement>();
  s->line_num = line_num;
  s->value = Sequence({.stmt = s1, .next = s2});
  return s;
}

auto Statement::MakeBlock(int line_num, const Statement* stmt)
    -> const Statement* {
  auto* s = global_arena->New<Statement>();
  s->line_num = line_num;
  s->value = Block({.stmt = stmt});
  return s;
}

auto Statement::MakeMatch(
    int line_num, const Expression* exp,
    std::list<std::pair<const Pattern*, const Statement*>>* clauses)
    -> const Statement* {
  auto* s = global_arena->New<Statement>();
  s->line_num = line_num;
  s->value = Match({.exp = exp, .clauses = clauses});
  return s;
}

// Returns an AST node for a continuation statement give its line number and
// parts.
auto Statement::MakeContinuation(int line_num,
                                 std::string continuation_variable,
                                 const Statement* body) -> const Statement* {
  auto* s = global_arena->New<Statement>();
  s->line_num = line_num;
  s->value =
      Continuation({.continuation_variable = std::move(continuation_variable),
                    .body = body});
  return s;
}

// Returns an AST node for a run statement give its line number and argument.
auto Statement::MakeRun(int line_num, const Expression* argument)
    -> const Statement* {
  auto* s = global_arena->New<Statement>();
  s->line_num = line_num;
  s->value = Run({.argument = argument});
  return s;
}

// Returns an AST node for an await statement give its line number.
auto Statement::MakeAwait(int line_num) -> const Statement* {
  auto* s = global_arena->New<Statement>();
  s->line_num = line_num;
  s->value = Await();
  return s;
}

void Statement::PrintDepth(int depth, llvm::raw_ostream& out) const {
  if (depth == 0) {
    out << " ... ";
    return;
  }
  switch (tag()) {
    case StatementKind::Match:
      out << "match (" << *GetMatch().exp << ") {";
      if (depth < 0 || depth > 1) {
        out << "\n";
        for (auto& clause : *GetMatch().clauses) {
          out << "case " << *clause.first << " =>\n";
          clause.second->PrintDepth(depth - 1, out);
          out << "\n";
        }
      } else {
        out << "...";
      }
      out << "}";
      break;
    case StatementKind::While:
      out << "while (" << *GetWhile().cond << ")\n";
      GetWhile().body->PrintDepth(depth - 1, out);
      break;
    case StatementKind::Break:
      out << "break;";
      break;
    case StatementKind::Continue:
      out << "continue;";
      break;
    case StatementKind::VariableDefinition:
      out << "var " << *GetVariableDefinition().pat << " = "
          << *GetVariableDefinition().init << ";";
      break;
    case StatementKind::ExpressionStatement:
      out << *GetExpressionStatement().exp << ";";
      break;
    case StatementKind::Assign:
      out << *GetAssign().lhs << " = " << *GetAssign().rhs << ";";
      break;
    case StatementKind::If:
      out << "if (" << *GetIf().cond << ")\n";
      GetIf().then_stmt->PrintDepth(depth - 1, out);
      if (GetIf().else_stmt) {
        out << "\nelse\n";
        GetIf().else_stmt->PrintDepth(depth - 1, out);
      }
      break;
    case StatementKind::Return:
      if (GetReturn().is_omitted_exp) {
        out << "return;";
      } else {
        out << "return " << *GetReturn().exp << ";";
      }
      break;
    case StatementKind::Sequence:
      GetSequence().stmt->PrintDepth(depth, out);
      if (depth < 0 || depth > 1) {
        out << "\n";
      } else {
        out << " ";
      }
      if (GetSequence().next) {
        GetSequence().next->PrintDepth(depth - 1, out);
      }
      break;
    case StatementKind::Block:
      out << "{";
      if (depth < 0 || depth > 1) {
        out << "\n";
      }
      if (GetBlock().stmt) {
        GetBlock().stmt->PrintDepth(depth, out);
        if (depth < 0 || depth > 1) {
          out << "\n";
        }
      }
      out << "}";
      if (depth < 0 || depth > 1) {
        out << "\n";
      }
      break;
    case StatementKind::Continuation:
      out << "continuation " << GetContinuation().continuation_variable << " ";
      if (depth < 0 || depth > 1) {
        out << "\n";
      }
      GetContinuation().body->PrintDepth(depth - 1, out);
      if (depth < 0 || depth > 1) {
        out << "\n";
      }
      break;
    case StatementKind::Run:
      out << "run " << *GetRun().argument << ";";
      break;
    case StatementKind::Await:
      out << "await;";
      break;
  }
}

}  // namespace Carbon
