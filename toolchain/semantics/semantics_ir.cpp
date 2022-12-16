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

auto SemanticsIR::MakeBuiltinIR() -> SemanticsIR {
  SemanticsIR semantics;
  auto block_id = semantics.AddNodeBlock();
  semantics.nodes_.reserve(SemanticsBuiltinKind::ValidCount);

  constexpr int32_t TypeOfTypeType = 0;
  auto type_type = semantics.AddNode(
      block_id, SemanticsNode::MakeBuiltin(SemanticsBuiltinKind::TypeType(),
                                           SemanticsNodeId(TypeOfTypeType)));
  CARBON_CHECK(type_type.index == TypeOfTypeType)
      << "TypeType's type must be self-referential.";

  constexpr int32_t TypeOfInvalidType = 1;
  auto invalid_type = semantics.AddNode(
      block_id, SemanticsNode::MakeBuiltin(SemanticsBuiltinKind::InvalidType(),
                                           SemanticsNodeId(TypeOfInvalidType)));
  CARBON_CHECK(invalid_type.index == TypeOfInvalidType)
      << "InvalidType's type must be self-referential.";

  semantics.AddNode(
      block_id, SemanticsNode::MakeBuiltin(SemanticsBuiltinKind::IntegerType(),
                                           type_type));

  semantics.AddNode(block_id, SemanticsNode::MakeBuiltin(
                                  SemanticsBuiltinKind::RealType(), type_type));

  CARBON_CHECK(semantics.node_blocks_.size() == 1)
      << "BuildBuiltins should only produce 1 block, actual: "
      << semantics.node_blocks_.size();
  return semantics;
}

auto SemanticsIR::MakeFromParseTree(const SemanticsIR& builtin_ir,
                                    const TokenizedBuffer& tokens,
                                    const ParseTree& parse_tree,
                                    DiagnosticConsumer& consumer,
                                    llvm::raw_ostream* vlog_stream)
    -> SemanticsIR {
  SemanticsIR semantics(builtin_ir, FromBuiltins::Constructor);

  TokenizedBuffer::TokenLocationTranslator translator(
      &tokens, /*last_line_lexed_to_column=*/nullptr);
  TokenDiagnosticEmitter emitter(translator, consumer);
  SemanticsParseTreeHandler(tokens, emitter, parse_tree, semantics, vlog_stream)
      .Build();
  return semantics;
}

SemanticsIR::SemanticsIR(const SemanticsIR& builtins,
                         FromBuiltins /*from_builtins*/)
    : cross_reference_irs_({&builtins, this}) {
  nodes_.resize_for_overwrite(SemanticsBuiltinKind::ValidCount);
  static constexpr auto BuiltinIR = SemanticsCrossReferenceIRId(0);
  for (int i = 0; i < SemanticsBuiltinKind::ValidCount; ++i) {
    // We can reuse the type node ID because the offsets of cross-references
    // will be the same in this IR.
    auto type = builtins.nodes_[i].type();
    nodes_[i] =
        SemanticsNode::MakeCrossReference(type, BuiltinIR, SemanticsNodeId(i));
  }
}

auto SemanticsIR::Print(llvm::raw_ostream& out) const -> void {
  constexpr int Indent = 2;

  out << "cross_reference_irs.size == " << cross_reference_irs_.size() << ",\n";

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

  out << "nodes = {\n";
  for (int32_t i = 0; i < static_cast<int32_t>(nodes_.size()); ++i) {
    out.indent(Indent);
    out << SemanticsNodeId(i) << " = " << nodes_[i] << ";\n";
  }
  out << "},\n";

  out << "node_blocks = {\n";
  for (int32_t i = 0; i < static_cast<int32_t>(node_blocks_.size()); ++i) {
    out.indent(Indent);
    out << SemanticsNodeBlockId(i) << " = {\n";

    const auto& node_block = node_blocks_[i];
    for (int32_t i = 0; i < static_cast<int32_t>(node_block.size()); ++i) {
      out.indent(2 * Indent);
      out << node_block[i] << ";\n";
    }

    out.indent(Indent);
    out << "},\n";
  }
  out << "}\n";
}

}  // namespace Carbon
