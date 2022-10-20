// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_ir.h"

#include "common/check.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

static auto PrintArgs(llvm::raw_ostream& /*out*/, int unused_no_args[0]) {}

static auto PrintArgs(llvm::raw_ostream& out, SemanticsNodeId one_node) {
  out << one_node;
}

static auto PrintArgs(llvm::raw_ostream& out, SemanticsTwoNodeIds two_nodes) {
  out << two_nodes.nodes[0] << ", " << two_nodes.nodes[1];
}

static auto PrintArgs(llvm::raw_ostream& out,
                      SemanticsIdentifierId identifier) {
  out << identifier;
}

static auto PrintArgs(llvm::raw_ostream& out,
                      SemanticsIntegerLiteralId identifier) {
  out << identifier;
}

auto SemanticsIR::Print(llvm::raw_ostream& out) const -> void {
  out << "identifiers = {\n";
  for (int32_t i = 0; i < static_cast<int32_t>(identifiers_.size()); ++i) {
    out.indent(2);
    out << SemanticsIdentifierId(i) << " = \"" << identifiers_[i] << "\";\n";
  }
  out << "},\n";

  out << "integer_literals = {\n";
  for (int32_t i = 0; i < static_cast<int32_t>(integer_literals_.size()); ++i) {
    out.indent(2);
    out << SemanticsIntegerLiteralId(i) << " = " << integer_literals_[i]
        << ";\n";
  }
  out << "},\n";

  out << "nodes = {\n";
  int indent = 2;
  for (int32_t i = 0; i < static_cast<int32_t>(nodes_.size()); ++i) {
    SemanticsNode node = nodes_[i];

    // Adjust indent for block contents.
    switch (node.kind_) {
      case SemanticsNodeKind::CodeBlockStart():
      case SemanticsNodeKind::FunctionDefinitionStart():
        out.indent(indent);
        indent += 2;
        break;
      case SemanticsNodeKind::CodeBlockEnd():
      case SemanticsNodeKind::FunctionDefinitionEnd():
        indent -= 2;
        out.indent(indent);
        break;
      default:
        // No indentation change.
        out.indent(indent);
        break;
    }

    out << SemanticsNodeId(i) << " = " << node.kind() << "(";
    switch (node.kind_) {
#define CARBON_SEMANTICS_NODE_KIND(Name, Args) \
  case SemanticsNodeKind::Name():              \
    PrintArgs(out, node.one_of_args_.Args);    \
    break;
#include "toolchain/semantics/semantics_node_kind.def"
    }
    out << ");\n";
  }
  out << "}\n";
}

}  // namespace Carbon
