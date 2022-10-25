// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_ir.h"

#include "common/check.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/semantics/semantics_node.h"
#include "toolchain/semantics/semantics_parse_tree_handler.h"

namespace Carbon {

auto SemanticsIR::Build(const TokenizedBuffer& tokens,
                        const ParseTree& parse_tree) -> void {
  SemanticsParseTreeHandler(tokens, parse_tree, *this).Build();
}

auto SemanticsIR::BuildBuiltins() -> void {
  // TODO: Implement.
}

auto SemanticsIR::Print(llvm::raw_ostream& out) const -> void {
  constexpr int Indent = 2;

  out << "identifiers = {\n";
  for (int32_t i = 0; i < static_cast<int32_t>(identifiers_.size()); ++i) {
    out.indent(Indent);
    out << SemanticsIdentifierId(i) << " = \"" << identifiers_[i] << "\";\n";
  }
  out << "},\n";

  out << "integer_literals = {\n";
  for (int32_t i = 0; i < static_cast<int32_t>(integer_literals_.size()); ++i) {
    out.indent(Indent);
    out << SemanticsIntegerLiteralId(i) << " = " << integer_literals_[i]
        << ";\n";
  }
  out << "},\n";

  out << "node_blocks = {\n";
  for (int32_t i = 0; i < static_cast<int32_t>(node_blocks_.size()); ++i) {
    out.indent(Indent);
    out << SemanticsNodeBlockId(i) << " = {\n";

    const auto& node_block = node_blocks_[i];
    for (int32_t i = 0; i < static_cast<int32_t>(node_block.size()); ++i) {
      out.indent(2 * Indent);
      out << SemanticsNodeId(i) << " = " << node_block[i] << ";\n";
    }

    out.indent(Indent);
    out << "},\n";
  }
  out << "}\n";
}

}  // namespace Carbon
