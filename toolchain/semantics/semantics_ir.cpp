// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_ir.h"

#include "common/check.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/semantics/semantics_builtin_kind.h"
#include "toolchain/semantics/semantics_node.h"
#include "toolchain/semantics/semantics_parse_tree_handler.h"

namespace Carbon {

auto SemanticsIR::BuildBuiltins() -> void {
  CARBON_CHECK(node_blocks_.empty())
      << "BuildBuiltins must be called before blocks are added.";

  auto block_id = AddNodeBlock();

  auto builtin_type_type = AddNode(
      block_id, SemanticsNode::MakeBuiltin(SemanticsBuiltinKind::TypeType(),
                                           SemanticsNodeId(0)));
  builtins_[SemanticsBuiltinKind::TypeType().AsInt()] = builtin_type_type;
  CARBON_CHECK(builtin_type_type.id == 0)
      << "TypeType's type must be self-referential.";
  AddNode(block_id, SemanticsNode::MakeBindName(AddIdentifier("Type"),
                                                builtin_type_type));

  builtins_[SemanticsBuiltinKind::Int32().AsInt()] =
      AddNode(block_id, SemanticsNode::MakeBuiltin(
                            SemanticsBuiltinKind::Int32(), builtin_type_type));

  CARBON_CHECK(node_blocks_.size() == 1)
      << "BuildBuiltins should only produce 1 block, actual: "
      << node_blocks_.size();
}

auto SemanticsIR::Build(const TokenizedBuffer& tokens,
                        const ParseTree& parse_tree) -> void {
  SemanticsParseTreeHandler(tokens, parse_tree, *this).Build();
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
