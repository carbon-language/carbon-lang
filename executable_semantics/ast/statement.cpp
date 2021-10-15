// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/statement.h"

#include "common/check.h"
#include "executable_semantics/common/arena.h"
#include "llvm/Support/Casting.h"

namespace Carbon {

using llvm::cast;

void Statement::PrintDepth(int depth, llvm::raw_ostream& out) const {
  if (depth == 0) {
    out << " ... ";
    return;
  }
  switch (kind()) {
    case Kind::Match: {
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
    case Kind::While: {
      const auto& while_stmt = cast<While>(*this);
      out << "while (" << while_stmt.condition() << ")\n";
      while_stmt.body().PrintDepth(depth - 1, out);
      break;
    }
    case Kind::Break:
      out << "break;";
      break;
    case Kind::Continue:
      out << "continue;";
      break;
    case Kind::VariableDefinition: {
      const auto& var = cast<VariableDefinition>(*this);
      out << "var " << var.pattern() << " = " << var.init() << ";";
      break;
    }
    case Kind::ExpressionStatement:
      out << cast<ExpressionStatement>(*this).expression() << ";";
      break;
    case Kind::Assign: {
      const auto& assign = cast<Assign>(*this);
      out << assign.lhs() << " = " << assign.rhs() << ";";
      break;
    }
    case Kind::If: {
      const auto& if_stmt = cast<If>(*this);
      out << "if (" << if_stmt.condition() << ")\n";
      if_stmt.then_statement().PrintDepth(depth - 1, out);
      if (if_stmt.else_statement()) {
        out << "\nelse\n";
        (*if_stmt.else_statement())->PrintDepth(depth - 1, out);
      }
      break;
    }
    case Kind::Return: {
      const auto& ret = cast<Return>(*this);
      if (ret.is_omitted_expression()) {
        out << "return;";
      } else {
        out << "return " << ret.expression() << ";";
      }
      break;
    }
    case Kind::Sequence: {
      const auto& seq = cast<Sequence>(*this);
      seq.statement().PrintDepth(depth, out);
      if (depth < 0 || depth > 1) {
        out << "\n";
      } else {
        out << " ";
      }
      if (seq.next()) {
        (*seq.next())->PrintDepth(depth - 1, out);
      }
      break;
    }
    case Kind::Block: {
      const auto& block = cast<Block>(*this);
      out << "{";
      if (depth < 0 || depth > 1) {
        out << "\n";
      }
      if (block.statement()) {
        (*block.statement())->PrintDepth(depth, out);
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
    case Kind::Continuation: {
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
    case Kind::Run:
      out << "run " << cast<Run>(*this).argument() << ";";
      break;
    case Kind::Await:
      out << "await;";
      break;
  }
}

}  // namespace Carbon
