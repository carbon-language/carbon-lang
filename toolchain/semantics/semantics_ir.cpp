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
  static constexpr auto BuiltinIR = SemanticsCrossReferenceIRId(0);
  auto block_id = semantics.AddNodeBlock();
  semantics.cross_references_.resize_for_overwrite(
      SemanticsBuiltinKind::ValidCount);

  constexpr int32_t TypeOfTypeType = 0;
  auto type_type = semantics.AddNode(
      block_id, SemanticsNode::MakeBuiltin(SemanticsBuiltinKind::TypeType(),
                                           SemanticsNodeId(TypeOfTypeType)));
  semantics.cross_references_[SemanticsBuiltinKind::TypeType().AsInt()] =
      SemanticsCrossReference(BuiltinIR, block_id, type_type);
  CARBON_CHECK(type_type.id == TypeOfTypeType)
      << "TypeType's type must be self-referential.";

  constexpr int32_t TypeOfInvalidType = 1;
  auto invalid_type = semantics.AddNode(
      block_id, SemanticsNode::MakeBuiltin(SemanticsBuiltinKind::InvalidType(),
                                           SemanticsNodeId(TypeOfInvalidType)));
  semantics.cross_references_[SemanticsBuiltinKind::InvalidType().AsInt()] =
      SemanticsCrossReference(BuiltinIR, block_id, invalid_type);
  CARBON_CHECK(invalid_type.id == TypeOfInvalidType)
      << "InvalidType's type must be self-referential.";

  auto integer_literal_type = semantics.AddNode(
      block_id, SemanticsNode::MakeBuiltin(
                    SemanticsBuiltinKind::IntegerLiteralType(), type_type));
  semantics
      .cross_references_[SemanticsBuiltinKind::IntegerLiteralType().AsInt()] =
      SemanticsCrossReference(BuiltinIR, block_id, integer_literal_type);

  auto real_literal_type = semantics.AddNode(
      block_id, SemanticsNode::MakeBuiltin(
                    SemanticsBuiltinKind::RealLiteralType(), type_type));
  semantics.cross_references_[SemanticsBuiltinKind::RealLiteralType().AsInt()] =
      SemanticsCrossReference(BuiltinIR, block_id, real_literal_type);

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
  SemanticsIR semantics(builtin_ir);

  TokenizedBuffer::TokenLocationTranslator translator(
      &tokens, /*last_line_lexed_to_column=*/nullptr);
  TokenDiagnosticEmitter emitter(translator, consumer);
  SemanticsParseTreeHandler(tokens, emitter, parse_tree, semantics, vlog_stream)
      .Build();
  return semantics;
}

auto SemanticsIR::Print(llvm::raw_ostream& out) const -> void {
  constexpr int Indent = 2;

  out << "cross_reference_irs.size == " << cross_reference_irs_.size() << ",\n";

  out << "cross_references = {\n";
  for (int32_t i = 0; i < static_cast<int32_t>(cross_references_.size()); ++i) {
    out.indent(Indent);
    out << SemanticsNodeId::MakeCrossReference(i) << " = "
        << cross_references_[i] << ";\n";
  }
  out << "},\n";

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
