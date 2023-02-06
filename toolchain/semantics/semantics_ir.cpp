// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_ir.h"

#include "common/check.h"
#include "toolchain/parser/parse_tree_node_location_translator.h"
#include "toolchain/semantics/semantics_builtin_kind.h"
#include "toolchain/semantics/semantics_node.h"
#include "toolchain/semantics/semantics_parse_tree_handler.h"

namespace Carbon {

auto SemanticsIR::MakeBuiltinIR() -> SemanticsIR {
  SemanticsIR semantics(/*builtin_ir=*/nullptr);
  auto block_id = semantics.AddNodeBlock();
  semantics.nodes_.reserve(SemanticsBuiltinKind::ValidCount);

  constexpr int32_t TypeOfTypeType = 0;
  auto type_type = semantics.AddNode(
      block_id, SemanticsNode::MakeBuiltin(SemanticsBuiltinKind::TypeType,
                                           SemanticsNodeId(TypeOfTypeType)));
  CARBON_CHECK(type_type.index == TypeOfTypeType)
      << "TypeType's type must be self-referential.";

  constexpr int32_t TypeOfInvalidType = 1;
  auto invalid_type = semantics.AddNode(
      block_id, SemanticsNode::MakeBuiltin(SemanticsBuiltinKind::InvalidType,
                                           SemanticsNodeId(TypeOfInvalidType)));
  CARBON_CHECK(invalid_type.index == TypeOfInvalidType)
      << "InvalidType's type must be self-referential.";

  semantics.AddNode(
      block_id,
      SemanticsNode::MakeBuiltin(SemanticsBuiltinKind::IntegerType, type_type));

  semantics.AddNode(block_id, SemanticsNode::MakeBuiltin(
                                  SemanticsBuiltinKind::RealType, type_type));

  semantics.AddNode(block_id, SemanticsNode::MakeBuiltin(
                                  SemanticsBuiltinKind::EmptyTuple, type_type));

  CARBON_CHECK(semantics.node_blocks_.size() == 2)
      << "BuildBuiltins should produce 2 blocks, actual: "
      << semantics.node_blocks_.size();
  return semantics;
}

auto SemanticsIR::MakeFromParseTree(const SemanticsIR& builtin_ir,
                                    const TokenizedBuffer& tokens,
                                    const ParseTree& parse_tree,
                                    DiagnosticConsumer& consumer,
                                    llvm::raw_ostream* vlog_stream)
    -> SemanticsIR {
  SemanticsIR semantics(&builtin_ir);

  // Copy builtins over.
  semantics.nodes_.resize_for_overwrite(SemanticsBuiltinKind::ValidCount);
  static constexpr auto BuiltinIR = SemanticsCrossReferenceIRId(0);
  for (int i = 0; i < SemanticsBuiltinKind::ValidCount; ++i) {
    // We can reuse the type node ID because the offsets of cross-references
    // will be the same in this IR.
    auto type = builtin_ir.nodes_[i].type();
    semantics.nodes_[i] =
        SemanticsNode::MakeCrossReference(type, BuiltinIR, SemanticsNodeId(i));
  }

  ParseTreeNodeLocationTranslator translator(&tokens, &parse_tree);
  ErrorTrackingDiagnosticConsumer err_tracker(consumer);
  DiagnosticEmitter<ParseTree::Node> emitter(translator, err_tracker);
  SemanticsParseTreeHandler(tokens, emitter, parse_tree, semantics, vlog_stream)
      .Build();
  semantics.has_errors_ = err_tracker.seen_error();
  return semantics;
}

static constexpr int Indent = 2;

template <typename T>
static auto PrintList(llvm::raw_ostream& out, llvm::StringLiteral name,
                      const llvm::SmallVector<T>& list) {
  out << name << ": [\n";
  for (const auto& element : list) {
    out.indent(Indent);
    out << element << ",\n";
  }
  out << "]\n";
}

auto SemanticsIR::Print(llvm::raw_ostream& out) const -> void {
  out << "cross_reference_irs_size: " << cross_reference_irs_.size() << "\n";

  PrintList(out, "calls", calls_);
  PrintList(out, "callables", callables_);
  PrintList(out, "integer_literals", integer_literals_);
  PrintList(out, "strings", strings_);
  PrintList(out, "nodes", nodes_);

  out << "node_blocks: [\n";
  for (const auto& node_block : node_blocks_) {
    out.indent(Indent);
    out << "[\n";

    for (const auto& node : node_block) {
      out.indent(2 * Indent);
      out << node << ",\n";
    }
    out.indent(Indent);
    out << "],\n";
  }
  out << "]\n";
}

}  // namespace Carbon
