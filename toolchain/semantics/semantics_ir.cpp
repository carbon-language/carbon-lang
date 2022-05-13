// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_ir.h"

#include "common/check.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/lexer/tokenized_buffer.h"

namespace Carbon {

void SemanticsIR::AddFunction(Semantics::DeclarationBlock& block,
                              Semantics::Function function) {
  int32_t index = functions_.size();
  functions_.push_back(function);
  block.add_named_node(
      parse_tree_->GetNodeText(function.name().node()),
      Semantics::Declaration(Semantics::DeclarationKind::Function, index));
}

auto SemanticsIR::AddReturn(Semantics::Return ret) -> Semantics::Statement {
  int32_t index = returns_.size();
  returns_.push_back(ret);
  return Semantics::Statement(Semantics::StatementKind::Return, index);
}

void SemanticsIR::Print(llvm::raw_ostream& out,
                        Semantics::Declaration decl) const {
  switch (decl.kind_) {
    case Semantics::DeclarationKind::Function:
      Print(out, functions_[decl.index_]);
      return;
    case Semantics::DeclarationKind::Invalid:
      CARBON_FATAL() << "Invalid node type";
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
  Print(out, expr.literal());
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
  out << ";";
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

}  // namespace Carbon
