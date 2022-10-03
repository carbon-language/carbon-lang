// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_ir.h"

#include "common/check.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/lexer/tokenized_buffer.h"

namespace Carbon {

auto SemanticsIR::Print(llvm::raw_ostream& out) const -> void {
  PrintBlock(out, 0, root_block());
  out << "\n";
}

auto SemanticsIR::PrintBlock(llvm::raw_ostream& out, int indent,
                             llvm::ArrayRef<Semantics::NodeRef> node_refs) const
    -> void {
  out << "{\n";
  int child_indent = indent + 2;
  for (const auto& node_ref : node_refs) {
    out.indent(child_indent);
    Print(out, child_indent, node_ref);
    out << ",\n";
  }
  out.indent(indent);
  out << "}";
}

auto SemanticsIR::Print(llvm::raw_ostream& out, int indent,
                        Semantics::NodeRef node_ref) const -> void {
  switch (node_ref.kind()) {
    case Semantics::NodeKind::BinaryOperator:
      nodes_.Get<Semantics::BinaryOperator>(node_ref).Print(out);
      return;
    case Semantics::NodeKind::Function:
      nodes_.Get<Semantics::Function>(node_ref).Print(
          out, indent,
          [&](int block_indent, llvm::ArrayRef<Semantics::NodeRef> block) {
            PrintBlock(out, block_indent, block);
          });
      return;
    case Semantics::NodeKind::IntegerLiteral:
      nodes_.Get<Semantics::IntegerLiteral>(node_ref).Print(out);
      return;
    case Semantics::NodeKind::Return:
      nodes_.Get<Semantics::Return>(node_ref).Print(out);
      return;
    case Semantics::NodeKind::SetName:
      nodes_.Get<Semantics::SetName>(node_ref).Print(out);
      return;
    case Semantics::NodeKind::Invalid:
      CARBON_FATAL() << "Invalid NodeRef kind";
  }
}

}  // namespace Carbon
