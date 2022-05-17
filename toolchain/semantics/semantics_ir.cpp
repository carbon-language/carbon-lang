// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_ir.h"

#include "common/check.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/semantics/nodes/expression_statement.h"

namespace Carbon {

void SemanticsIR::Print(llvm::raw_ostream& out,
                        Semantics::Declaration decl) const {
  switch (decl.kind()) {
    case Semantics::DeclarationKind::Function:
      Print(out, declarations_.Get<Semantics::Function>(decl));
      return;
    case Semantics::DeclarationKind::Invalid:
      CARBON_FATAL() << "Invalid declaration type";
  }
}

void SemanticsIR::Print(llvm::raw_ostream& out, ParseTree::Node node) const {
  out << parse_tree_->GetNodeText(node);
}

void SemanticsIR::Print(llvm::raw_ostream& out,
                        Semantics::DeclaredName name) const {
  Print(out, name.node());
}

void SemanticsIR::Print(llvm::raw_ostream& out,
                        Semantics::Expression expr) const {
  switch (expr.kind()) {
    case Semantics::ExpressionKind::InfixOperator:
      Print(out, expressions_.Get<Semantics::InfixOperator>(expr));
      return;
    case Semantics::ExpressionKind::Literal:
      Print(out, expressions_.Get<Semantics::Literal>(expr));
      return;
    case Semantics::ExpressionKind::Invalid:
      CARBON_FATAL() << "Invalid expression type";
  }
}

void SemanticsIR::Print(llvm::raw_ostream& out,
                        Semantics::ExpressionStatement expr) const {
  Print(out, expr.expression());
}

void SemanticsIR::Print(llvm::raw_ostream& out,
                        Semantics::Function function) const {
  out << "fn ";
  Print(out, function.name());
  out << "(";
  llvm::ListSeparator sep;
  for (const auto& param : function.params()) {
    out << sep;
    Print(out, param);
  }
  out << ")";
  if (function.return_expr()) {
    out << " -> ";
    Print(out, *function.return_expr());
  }
  Print(out, function.body());
}

void SemanticsIR::Print(llvm::raw_ostream& out,
                        Semantics::InfixOperator op) const {
  Print(out, op.lhs());
  out << " ";
  Print(out, op.node());
  out << " ";
  Print(out, op.rhs());
}

void SemanticsIR::Print(llvm::raw_ostream& out,
                        Semantics::Literal literal) const {
  Print(out, literal.node());
}

void SemanticsIR::Print(llvm::raw_ostream& out,
                        Semantics::PatternBinding binding) const {
  Print(out, binding.name());
  out << ": ";
  Print(out, binding.type());
}

void SemanticsIR::Print(llvm::raw_ostream& out, Semantics::Return ret) const {
  out << "return";
  if (ret.expression()) {
    out << " ";
    Print(out, *ret.expression());
  }
}

void SemanticsIR::Print(llvm::raw_ostream& out,
                        Semantics::Statement stmt) const {
  switch (stmt.kind()) {
    case Semantics::StatementKind::ExpressionStatement:
      Print(out, statements_.Get<Semantics::ExpressionStatement>(stmt));
      return;
    case Semantics::StatementKind::Return:
      Print(out, statements_.Get<Semantics::Return>(stmt));
      return;
    case Semantics::StatementKind::Invalid:
      CARBON_FATAL() << "Invalid expression type";
  }
  out << ";";
}

void SemanticsIR::Print(llvm::raw_ostream& out,
                        Semantics::StatementBlock block) const {
  out << " { ";
  for (const auto& statement : block.nodes()) {
    Print(out, statement);
    out << "; ";
  }
  out << "}";
}

}  // namespace Carbon
