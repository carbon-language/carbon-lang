// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_ir.h"

#include "common/check.h"
#include "toolchain/common/pretty_stack_trace_function.h"
#include "toolchain/parser/parse_tree_node_location_translator.h"
#include "toolchain/semantics/semantics_builtin_kind.h"
#include "toolchain/semantics/semantics_context.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

auto SemanticsIR::MakeBuiltinIR() -> SemanticsIR {
  SemanticsIR semantics(/*builtin_ir=*/nullptr);
  semantics.nodes_.reserve(SemanticsBuiltinKind::ValidCount);

#define CARBON_SEMANTICS_BUILTIN_KIND(Name, Type, ...)     \
  semantics.nodes_.push_back(SemanticsNode::Builtin::Make( \
      SemanticsBuiltinKind::Name, SemanticsNodeId::Builtin##Type));
#include "toolchain/semantics/semantics_builtin_kind.def"

  CARBON_CHECK(semantics.node_blocks_.size() == 1)
      << "BuildBuiltins should only have the empty block, actual: "
      << semantics.node_blocks_.size();
  CARBON_CHECK(semantics.nodes_.size() == SemanticsBuiltinKind::ValidCount)
      << "BuildBuiltins should produce " << SemanticsBuiltinKind::ValidCount
      << " nodes, actual: " << semantics.nodes_.size();
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
    auto type = builtin_ir.nodes_[i].type_id();
    semantics.nodes_[i] = SemanticsNode::CrossReference::Make(
        type, BuiltinIR, SemanticsNodeId(i));
  }

  ParseTreeNodeLocationTranslator translator(&tokens, &parse_tree);
  ErrorTrackingDiagnosticConsumer err_tracker(consumer);
  DiagnosticEmitter<ParseTree::Node> emitter(translator, err_tracker);

  SemanticsContext context(tokens, emitter, parse_tree, semantics, vlog_stream);
  PrettyStackTraceFunction context_dumper(
      [&](llvm::raw_ostream& output) { context.PrintForStackDump(output); });

  // Add a block for the ParseTree.
  context.node_block_stack().Push();
  context.PushScope();

  // Loops over all nodes in the tree. On some errors, this may return early,
  // for example if an unrecoverable state is encountered.
  for (auto parse_node : parse_tree.postorder()) {
    switch (auto parse_kind = parse_tree.node_kind(parse_node)) {
#define CARBON_PARSE_NODE_KIND(Name)                   \
  case ParseNodeKind::Name: {                          \
    if (!SemanticsHandle##Name(context, parse_node)) { \
      semantics.has_errors_ = true;                    \
      return semantics;                                \
    }                                                  \
    break;                                             \
  }
#include "toolchain/parser/parse_node_kind.def"
    }
  }

  // Pop information for the file-level scope.
  semantics.top_node_block_id_ = context.node_block_stack().Pop();
  context.PopScope();

  context.VerifyOnFinish();

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

auto SemanticsIR::Print(llvm::raw_ostream& out, bool include_builtins) const
    -> void {
  out << "cross_reference_irs_size: " << cross_reference_irs_.size() << "\n";

  PrintList(out, "calls", calls_);
  PrintList(out, "callables", callables_);
  PrintList(out, "integer_literals", integer_literals_);
  PrintList(out, "real_literals", real_literals_);
  PrintList(out, "strings", strings_);

  out << "nodes: [\n";
  for (int i = include_builtins ? 0 : SemanticsBuiltinKind::ValidCount;
       i < static_cast<int>(nodes_.size()); ++i) {
    const auto& element = nodes_[i];
    out.indent(Indent);
    out << element << ",\n";
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

auto SemanticsIR::StringifyNode(SemanticsNodeId node_id) -> std::string {
  std::string str;
  llvm::raw_string_ostream out(str);

  struct Step {
    // The node to print.
    SemanticsNodeId node_id;
    // The index into node_id to print. Not used by all types.
    int index = 0;
  };
  llvm::SmallVector<Step> steps = {{.node_id = node_id}};

  while (!steps.empty()) {
    auto step = steps.pop_back_val();

    // Invalid node IDs will use the default invalid printing.
    if (!step.node_id.is_valid()) {
      out << step.node_id;
      continue;
    }

    // Builtins have designated labels.
    if (step.node_id.index < SemanticsBuiltinKind::ValidCount) {
      out << SemanticsBuiltinKind::FromInt(step.node_id.index).label();
      continue;
    }

    auto node = GetNode(step.node_id);
    switch (node.kind()) {
      case SemanticsNodeKind::StructType: {
        auto refs = GetNodeBlock(node.GetAsStructType().second);
        if (step.index == 0) {
          out << "{";
        } else if (step.index < static_cast<int>(refs.size())) {
          out << ", ";
        } else {
          out << "}";
          break;
        }

        steps.push_back({.node_id = step.node_id, .index = step.index + 1});
        steps.push_back({.node_id = refs[step.index]});
        break;
      }
      case SemanticsNodeKind::StructTypeField: {
        out << "." << GetString(node.GetAsStructTypeField()) << ": ";
        steps.push_back({.node_id = node.type_id()});
        break;
      }
      case SemanticsNodeKind::Assign:
      case SemanticsNodeKind::BinaryOperatorAdd:
      case SemanticsNodeKind::BindName:
      case SemanticsNodeKind::Builtin:
      case SemanticsNodeKind::Call:
      case SemanticsNodeKind::CodeBlock:
      case SemanticsNodeKind::CrossReference:
      case SemanticsNodeKind::FunctionDeclaration:
      case SemanticsNodeKind::FunctionDefinition:
      case SemanticsNodeKind::IntegerLiteral:
      case SemanticsNodeKind::RealLiteral:
      case SemanticsNodeKind::Return:
      case SemanticsNodeKind::ReturnExpression:
      case SemanticsNodeKind::StringLiteral:
      case SemanticsNodeKind::StructMemberAccess:
      case SemanticsNodeKind::StructValue:
      case SemanticsNodeKind::StubReference:
      case SemanticsNodeKind::VarStorage:
        // We don't need to handle stringification for nodes that don't show up
        // in errors, but make it clear what's going on so that it's clearer
        // when stringification is needed.
        out << "<cannot stringify " << step.node_id << ">";
        break;
      case SemanticsNodeKind::Invalid:
        llvm_unreachable("SemanticsNodeKind::Invalid is never used.");
    }
  }

  return str;
}

}  // namespace Carbon
