// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_ir.h"

#include "common/check.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/lexer/tokenized_buffer.h"

namespace Carbon {

void SemanticsIR::Block::Add(llvm::StringRef name, Node named_entity) {
  nodes_.push_back(named_entity);
  name_lookup_.insert({name, named_entity});
}

void SemanticsIR::AddFunction(Block& block, Semantics::Function function) {
  int32_t index = functions_.size();
  functions_.push_back(function);
  block.Add(parse_tree_->GetNodeText(function.name().node()),
            Node(Node::Kind::Function, index));
}

void SemanticsIR::Print(llvm::raw_ostream& out, Node node) const {
  switch (node.kind_) {
    case Node::Kind::Function:
      Print(out, functions_[node.index_]);
      return;
    case Node::Kind::Invalid:
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
