// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/statement.h"

#include "common/check.h"
#include "executable_semantics/common/arena.h"
#include "llvm/Support/Casting.h"

namespace Carbon {

using llvm::cast;

Statement::~Statement() = default;

void Statement::PrintDepth(int depth, llvm::raw_ostream& out) const {
  if (depth == 0) {
    out << " ... ";
    return;
  }
  switch (kind()) {
    case StatementKind::Match: {
      const auto& match = cast<Match>(*this);
      out << "match (" << match.expression() << ") {";
      if (depth < 0 || depth > 1) {
        out << "\n";
        for (auto& clause : match.clauses()) {
          out << "case " << clause.pattern() << " =>\n";
          clause.statement().PrintDepth(depth - 1, out);
          out << "\n";
        }
      } else {
        out << "...";
      }
      out << "}";
      break;
    }
    case StatementKind::While: {
      const auto& while_stmt = cast<While>(*this);
      out << "while (" << while_stmt.condition() << ")\n";
      while_stmt.body().PrintDepth(depth - 1, out);
      break;
    }
    case StatementKind::Break:
      out << "break;";
      break;
    case StatementKind::Continue:
      out << "continue;";
      break;
    case StatementKind::VariableDefinition: {
      const auto& var = cast<VariableDefinition>(*this);
      out << "var " << var.pattern() << " = " << var.init() << ";";
      break;
    }
    case StatementKind::ExpressionStatement:
      out << cast<ExpressionStatement>(*this).expression() << ";";
      break;
    case StatementKind::Assign: {
      const auto& assign = cast<Assign>(*this);
      out << assign.lhs() << " = " << assign.rhs() << ";";
      break;
    }
    case StatementKind::If: {
      const auto& if_stmt = cast<If>(*this);
      out << "if (" << if_stmt.condition() << ")\n";
      if_stmt.then_block().PrintDepth(depth - 1, out);
      if (if_stmt.else_block()) {
        out << "\nelse\n";
        (*if_stmt.else_block())->PrintDepth(depth - 1, out);
      }
      break;
    }
    case StatementKind::Return: {
      const auto& ret = cast<Return>(*this);
      if (ret.is_omitted_expression()) {
        out << "return;";
      } else {
        out << "return " << ret.expression() << ";";
      }
      break;
    }
    case StatementKind::Block: {
      const auto& block = cast<Block>(*this);
      out << "{";
      if (depth < 0 || depth > 1) {
        out << "\n";
      }
      for (const auto* statement : block.statements()) {
        statement->PrintDepth(depth, out);
        if (depth < 0 || depth > 1) {
          out << "\n";
        }
      }
      out << "}";
      if (depth < 0 || depth > 1) {
        out << "\n";
      }
      break;
    }
    case StatementKind::Continuation: {
      const auto& cont = cast<Continuation>(*this);
      out << "continuation " << cont.continuation_variable() << " ";
      if (depth < 0 || depth > 1) {
        out << "\n";
      }
      cont.body().PrintDepth(depth - 1, out);
      if (depth < 0 || depth > 1) {
        out << "\n";
      }
      break;
    }
    case StatementKind::Run:
      out << "run " << cast<Run>(*this).argument() << ";";
      break;
    case StatementKind::Await:
      out << "await;";
      break;
  }
}

}  // namespace Carbon
