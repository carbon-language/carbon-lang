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
  switch (Tag()) {
    case Kind::Match: {
      const auto& match = cast<Match>(*this);
      out << "match (" << *match.Exp() << ") {";
      if (depth < 0 || depth > 1) {
        out << "\n";
        for (auto& clause : *match.Clauses()) {
          out << "case " << *clause.first << " =>\n";
          clause.second->PrintDepth(depth - 1, out);
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
      out << "while (" << *while_stmt.Cond() << ")\n";
      while_stmt.Body()->PrintDepth(depth - 1, out);
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
      out << "var " << *var.Pat() << " = " << *var.Init() << ";";
      break;
    }
    case Kind::ExpressionStatement:
      out << *cast<ExpressionStatement>(*this).Exp() << ";";
      break;
    case Kind::Assign: {
      const auto& assign = cast<Assign>(*this);
      out << *assign.Lhs() << " = " << *assign.Rhs() << ";";
      break;
    }
    case Kind::If: {
      const auto& if_stmt = cast<If>(*this);
      out << "if (" << *if_stmt.Cond() << ")\n";
      if_stmt.ThenStmt()->PrintDepth(depth - 1, out);
      if (if_stmt.ElseStmt()) {
        out << "\nelse\n";
        if_stmt.ElseStmt()->PrintDepth(depth - 1, out);
      }
      break;
    }
    case Kind::Return: {
      const auto& ret = cast<Return>(*this);
      if (ret.IsOmittedExp()) {
        out << "return;";
      } else {
        out << "return " << *ret.Exp() << ";";
      }
      break;
    }
    case Kind::Sequence: {
      const auto& seq = cast<Sequence>(*this);
      seq.Stmt()->PrintDepth(depth, out);
      if (depth < 0 || depth > 1) {
        out << "\n";
      } else {
        out << " ";
      }
      if (seq.Next()) {
        seq.Next()->PrintDepth(depth - 1, out);
      }
      break;
    }
    case Kind::Block: {
      const auto& block = cast<Block>(*this);
      out << "{";
      if (depth < 0 || depth > 1) {
        out << "\n";
      }
      if (block.Stmt()) {
        block.Stmt()->PrintDepth(depth, out);
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
      out << "continuation " << cont.ContinuationVariable() << " ";
      if (depth < 0 || depth > 1) {
        out << "\n";
      }
      cont.Body()->PrintDepth(depth - 1, out);
      if (depth < 0 || depth > 1) {
        out << "\n";
      }
      break;
    }
    case Kind::Run:
      out << "run " << *cast<Run>(*this).Argument() << ";";
      break;
    case Kind::Await:
      out << "await;";
      break;
  }
}

}  // namespace Carbon
