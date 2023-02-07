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

#define CARBON_SEMANTICS_BUILTIN_KIND(Name, Type)                      \
  semantics.AddNode(                                                   \
      block_id, SemanticsNode::MakeBuiltin(SemanticsBuiltinKind::Name, \
                                           SemanticsNodeId::Builtin##Type));
#include "toolchain/semantics/semantics_builtin_kind.def"

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

auto SemanticsIR::Print(llvm::raw_ostream& out) const -> void {
  constexpr int Indent = 2;

  out << "cross_reference_irs_size: " << cross_reference_irs_.size() << "\n";

  out << "callables: [\n";
  for (auto callable : callables_) {
    out.indent(Indent);
    out << callable << "\n";
  }
  out << "]\n";

  out << "integer_literals: [\n";
  for (const auto& integer_literal : integer_literals_) {
    out.indent(Indent);
    out << integer_literal << ",\n";
  }
  out << "]\n";

  out << "strings: [\n";
  for (const auto& string : strings_) {
    out.indent(Indent);
    out << string << ",\n";
  }
  out << "]\n";

  out << "nodes: [\n";
  for (const auto& node : nodes_) {
    out.indent(Indent);
    out << node << ",\n";
  }
  out << "]\n";

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
