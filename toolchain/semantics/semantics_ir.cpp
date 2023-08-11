// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_ir.h"

#include "common/check.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SaveAndRestore.h"
#include "toolchain/common/pretty_stack_trace_function.h"
#include "toolchain/parser/parse_tree_node_location_translator.h"
#include "toolchain/semantics/semantics_builtin_kind.h"
#include "toolchain/semantics/semantics_context.h"
#include "toolchain/semantics/semantics_node.h"
#include "toolchain/semantics/semantics_node_kind.h"

namespace Carbon {

auto SemanticsIR::MakeBuiltinIR() -> SemanticsIR {
  SemanticsIR semantics_ir(/*builtin_ir=*/nullptr);
  semantics_ir.nodes_.reserve(SemanticsBuiltinKind::ValidCount);

  // Error uses a self-referential type so that it's not accidentally treated as
  // a normal type. Every other builtin is a type, including the
  // self-referential TypeType.
#define CARBON_SEMANTICS_BUILTIN_KIND(Name, ...)                \
  semantics_ir.nodes_.push_back(SemanticsNode::Builtin::Make(   \
      SemanticsBuiltinKind::Name,                               \
      SemanticsBuiltinKind::Name == SemanticsBuiltinKind::Error \
          ? SemanticsTypeId::Error                              \
          : SemanticsTypeId::TypeType));
#include "toolchain/semantics/semantics_builtin_kind.def"

  CARBON_CHECK(semantics_ir.node_blocks_.size() == 1)
      << "BuildBuiltins should only have the empty block, actual: "
      << semantics_ir.node_blocks_.size();
  CARBON_CHECK(semantics_ir.nodes_.size() == SemanticsBuiltinKind::ValidCount)
      << "BuildBuiltins should produce " << SemanticsBuiltinKind::ValidCount
      << " nodes, actual: " << semantics_ir.nodes_.size();
  return semantics_ir;
}

auto SemanticsIR::MakeFromParseTree(const SemanticsIR& builtin_ir,
                                    const TokenizedBuffer& tokens,
                                    const ParseTree& parse_tree,
                                    DiagnosticConsumer& consumer,
                                    llvm::raw_ostream* vlog_stream)
    -> SemanticsIR {
  SemanticsIR semantics_ir(&builtin_ir);

  // Copy builtins over.
  semantics_ir.nodes_.resize_for_overwrite(SemanticsBuiltinKind::ValidCount);
  static constexpr auto BuiltinIR = SemanticsCrossReferenceIRId(0);
  for (int i = 0; i < SemanticsBuiltinKind::ValidCount; ++i) {
    // We can reuse the type node ID because the offsets of cross-references
    // will be the same in this IR.
    auto type = builtin_ir.nodes_[i].type_id();
    semantics_ir.nodes_[i] = SemanticsNode::CrossReference::Make(
        type, BuiltinIR, SemanticsNodeId(i));
  }

  ParseTreeNodeLocationTranslator translator(&tokens, &parse_tree);
  ErrorTrackingDiagnosticConsumer err_tracker(consumer);
  DiagnosticEmitter<ParseTree::Node> emitter(translator, err_tracker);

  SemanticsContext context(tokens, emitter, parse_tree, semantics_ir,
                           vlog_stream);
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
      semantics_ir.has_errors_ = true;                 \
      return semantics_ir;                             \
    }                                                  \
    break;                                             \
  }
#include "toolchain/parser/parse_node_kind.def"
    }
  }

  // Pop information for the file-level scope.
  semantics_ir.top_node_block_id_ = context.node_block_stack().Pop();
  context.PopScope();

  context.VerifyOnFinish();

  semantics_ir.has_errors_ = err_tracker.seen_error();

#ifndef NDEBUG
  if (auto verify = semantics_ir.Verify(); !verify.ok()) {
    CARBON_FATAL() << semantics_ir
                   << "Built invalid semantics IR: " << verify.error() << "\n";
  }
#endif

  return semantics_ir;
}

auto SemanticsIR::Verify() const -> ErrorOr<Success> {
  // Invariants don't necessarily hold for invalid IR.
  if (has_errors_) {
    return Success();
  }

  // Check that every code block has a terminator sequence that appears at the
  // end of the block.
  for (const SemanticsFunction& function : functions_) {
    for (SemanticsNodeBlockId block_id : function.body_block_ids) {
      SemanticsTerminatorKind prior_kind =
          SemanticsTerminatorKind::NotTerminator;
      for (SemanticsNodeId node_id : GetNodeBlock(block_id)) {
        SemanticsTerminatorKind node_kind =
            GetNode(node_id).kind().terminator_kind();
        if (prior_kind == SemanticsTerminatorKind::Terminator) {
          return Error(llvm::formatv("Node {0} in block {1} follows terminator",
                                     node_id, block_id));
        }
        if (prior_kind > node_kind) {
          return Error(
              llvm::formatv("Non-terminator node {0} in block {1} follows "
                            "terminator sequence",
                            node_id, block_id));
        }
        prior_kind = node_kind;
      }
      if (prior_kind != SemanticsTerminatorKind::Terminator) {
        return Error(llvm::formatv("No terminator in block {0}", block_id));
      }
    }
  }

  // TODO: Check that a node only references other nodes that are either global
  // or that dominate it.
  return Success();
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

template <typename T>
static auto PrintBlock(llvm::raw_ostream& out, llvm::StringLiteral block_name,
                       const llvm::SmallVector<T>& blocks) {
  out << block_name << ": [\n";
  for (const auto& block : blocks) {
    out.indent(Indent);
    out << "[\n";

    for (const auto& node : block) {
      out.indent(2 * Indent);
      out << node << ",\n";
    }
    out.indent(Indent);
    out << "],\n";
  }
  out << "]\n";
}

auto SemanticsIR::Print(llvm::raw_ostream& out, bool include_builtins) const
    -> void {
  out << "cross_reference_irs_size: " << cross_reference_irs_.size() << "\n";

  PrintList(out, "functions", functions_);
  PrintList(out, "integer_literals", integer_literals_);
  PrintList(out, "real_literals", real_literals_);
  PrintList(out, "strings", strings_);
  PrintList(out, "types", types_);

  PrintBlock(out, "type_blocks", type_blocks_);

  out << "nodes: [\n";
  for (int i = include_builtins ? 0 : SemanticsBuiltinKind::ValidCount;
       i < static_cast<int>(nodes_.size()); ++i) {
    const auto& element = nodes_[i];
    out.indent(Indent);
    out << element << ",\n";
  }
  out << "]\n";

  PrintBlock(out, "node_blocks", node_blocks_);
}

// Map a node kind representing a type into an integer describing the
// precedence of that type's syntax. Higher numbers correspond to higher
// precedence.
static auto GetTypePrecedence(SemanticsNodeKind kind) -> int {
  switch (kind) {
    case SemanticsNodeKind::Builtin:
    case SemanticsNodeKind::StructType:
    case SemanticsNodeKind::TupleType:
      return 0;
    case SemanticsNodeKind::ConstType:
      return -1;
    case SemanticsNodeKind::PointerType:
      return -2;

    case SemanticsNodeKind::CrossReference:
      // TODO: Once we support stringification of cross-references, we'll need
      // to determine the precedence of the target of the cross-reference. For
      // now, all cross-references refer to builtin types from the prelude.
      return 0;

    case SemanticsNodeKind::AddressOf:
    case SemanticsNodeKind::Assign:
    case SemanticsNodeKind::BinaryOperatorAdd:
    case SemanticsNodeKind::BindName:
    case SemanticsNodeKind::BlockArg:
    case SemanticsNodeKind::BoolLiteral:
    case SemanticsNodeKind::Branch:
    case SemanticsNodeKind::BranchIf:
    case SemanticsNodeKind::BranchWithArg:
    case SemanticsNodeKind::Call:
    case SemanticsNodeKind::Dereference:
    case SemanticsNodeKind::FunctionDeclaration:
    case SemanticsNodeKind::Index:
    case SemanticsNodeKind::InitializeFrom:
    case SemanticsNodeKind::IntegerLiteral:
    case SemanticsNodeKind::Invalid:
    case SemanticsNodeKind::MaterializeTemporary:
    case SemanticsNodeKind::Namespace:
    case SemanticsNodeKind::RealLiteral:
    case SemanticsNodeKind::Return:
    case SemanticsNodeKind::ReturnExpression:
    case SemanticsNodeKind::StringLiteral:
    case SemanticsNodeKind::StructMemberAccess:
    case SemanticsNodeKind::StructTypeField:
    case SemanticsNodeKind::StructValue:
    case SemanticsNodeKind::StubReference:
    case SemanticsNodeKind::TupleValue:
    case SemanticsNodeKind::UnaryOperatorNot:
    case SemanticsNodeKind::ValueBinding:
    case SemanticsNodeKind::VarStorage:
      CARBON_FATAL() << "GetTypePrecedence for non-type node kind " << kind;
  }
}

auto SemanticsIR::StringifyType(SemanticsTypeId type_id,
                                bool in_type_context) const -> std::string {
  std::string str;
  llvm::raw_string_ostream out(str);

  struct Step {
    // The node to print.
    SemanticsNodeId node_id;
    // The index into node_id to print. Not used by all types.
    int index = 0;

    auto Next() const -> Step {
      return {.node_id = node_id, .index = index + 1};
    }
  };
  auto outer_node_id = GetTypeAllowBuiltinTypes(type_id);
  llvm::SmallVector<Step> steps = {{.node_id = outer_node_id}};

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
      case SemanticsNodeKind::ConstType: {
        if (step.index == 0) {
          out << "const ";

          // Add parentheses if required.
          auto inner_type_node_id =
              GetTypeAllowBuiltinTypes(node.GetAsConstType());
          if (GetTypePrecedence(GetNode(inner_type_node_id).kind()) <
              GetTypePrecedence(node.kind())) {
            out << "(";
            steps.push_back(step.Next());
          }

          steps.push_back({.node_id = inner_type_node_id});
        } else if (step.index == 1) {
          out << ")";
        }
        break;
      }
      case SemanticsNodeKind::PointerType: {
        if (step.index == 0) {
          steps.push_back(step.Next());
          steps.push_back(
              {.node_id = GetTypeAllowBuiltinTypes(node.GetAsPointerType())});
        } else if (step.index == 1) {
          out << "*";
        }
        break;
      }
      case SemanticsNodeKind::StructType: {
        auto refs = GetNodeBlock(node.GetAsStructType());
        if (refs.empty()) {
          out << "{}";
          break;
        } else if (step.index == 0) {
          out << "{";
        } else if (step.index < static_cast<int>(refs.size())) {
          out << ", ";
        } else {
          out << "}";
          break;
        }

        steps.push_back(step.Next());
        steps.push_back({.node_id = refs[step.index]});
        break;
      }
      case SemanticsNodeKind::StructTypeField: {
        auto [name_id, type_id] = node.GetAsStructTypeField();
        out << "." << GetString(name_id) << ": ";
        steps.push_back({.node_id = GetTypeAllowBuiltinTypes(type_id)});
        break;
      }
      case SemanticsNodeKind::TupleType: {
        auto refs = GetTypeBlock(node.GetAsTupleType());
        if (refs.empty()) {
          out << "()";
          break;
        } else if (step.index == 0) {
          out << "(";
        } else if (step.index < static_cast<int>(refs.size())) {
          out << ", ";
        } else {
          // A tuple of one element has a comma to disambiguate from an
          // expression.
          if (step.index == 1) {
            out << ",";
          }
          out << ")";
          break;
        }
        steps.push_back(step.Next());
        steps.push_back(
            {.node_id = GetTypeAllowBuiltinTypes(refs[step.index])});
        break;
      }
      case SemanticsNodeKind::AddressOf:
      case SemanticsNodeKind::Assign:
      case SemanticsNodeKind::BinaryOperatorAdd:
      case SemanticsNodeKind::BindName:
      case SemanticsNodeKind::BlockArg:
      case SemanticsNodeKind::BoolLiteral:
      case SemanticsNodeKind::Branch:
      case SemanticsNodeKind::BranchIf:
      case SemanticsNodeKind::BranchWithArg:
      case SemanticsNodeKind::Builtin:
      case SemanticsNodeKind::Call:
      case SemanticsNodeKind::CrossReference:
      case SemanticsNodeKind::Dereference:
      case SemanticsNodeKind::FunctionDeclaration:
      case SemanticsNodeKind::Index:
      case SemanticsNodeKind::InitializeFrom:
      case SemanticsNodeKind::IntegerLiteral:
      case SemanticsNodeKind::MaterializeTemporary:
      case SemanticsNodeKind::Namespace:
      case SemanticsNodeKind::RealLiteral:
      case SemanticsNodeKind::Return:
      case SemanticsNodeKind::ReturnExpression:
      case SemanticsNodeKind::StringLiteral:
      case SemanticsNodeKind::StructMemberAccess:
      case SemanticsNodeKind::StructValue:
      case SemanticsNodeKind::StubReference:
      case SemanticsNodeKind::TupleValue:
      case SemanticsNodeKind::UnaryOperatorNot:
      case SemanticsNodeKind::ValueBinding:
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

  // For `{}` or any tuple type, we've printed a non-type expression, so add a
  // conversion to type `type` if it's not implied by the context.
  if (!in_type_context) {
    auto outer_node = GetNode(outer_node_id);
    if (outer_node.kind() == SemanticsNodeKind::TupleType ||
        (outer_node.kind() == SemanticsNodeKind::StructType &&
         GetNodeBlock(outer_node.GetAsStructType()).empty())) {
      out << " as type";
    }
  }

  return str;
}

auto GetSemanticsExpressionCategory(const SemanticsIR& semantics_ir,
                                    SemanticsNodeId node_id)
    -> SemanticsExpressionCategory {
  const SemanticsIR* ir = &semantics_ir;
  while (true) {
    auto node = ir->GetNode(node_id);
    switch (node.kind()) {
      case SemanticsNodeKind::Invalid:
      case SemanticsNodeKind::Assign:
      case SemanticsNodeKind::Branch:
      case SemanticsNodeKind::BranchIf:
      case SemanticsNodeKind::BranchWithArg:
      case SemanticsNodeKind::FunctionDeclaration:
      case SemanticsNodeKind::Namespace:
      case SemanticsNodeKind::Return:
      case SemanticsNodeKind::ReturnExpression:
      case SemanticsNodeKind::StructTypeField:
        return SemanticsExpressionCategory::NotExpression;

      case SemanticsNodeKind::CrossReference: {
        auto [xref_id, xref_node_id] = node.GetAsCrossReference();
        ir = &semantics_ir.GetCrossReferenceIR(xref_id);
        node_id = xref_node_id;
        continue;
      }

      case SemanticsNodeKind::Call:
        // TODO: This should eventually be Initializing.
        return SemanticsExpressionCategory::Value;

      case SemanticsNodeKind::BindName: {
        auto [name_id, value_id] = node.GetAsBindName();
        node_id = value_id;
        continue;
      }

      case SemanticsNodeKind::AddressOf:
      case SemanticsNodeKind::BinaryOperatorAdd:
      case SemanticsNodeKind::BlockArg:
      case SemanticsNodeKind::BoolLiteral:
      case SemanticsNodeKind::Builtin:
      case SemanticsNodeKind::ConstType:
      case SemanticsNodeKind::IntegerLiteral:
      case SemanticsNodeKind::PointerType:
      case SemanticsNodeKind::RealLiteral:
      case SemanticsNodeKind::StringLiteral:
      case SemanticsNodeKind::StructType:
      case SemanticsNodeKind::TupleType:
      case SemanticsNodeKind::UnaryOperatorNot:
      case SemanticsNodeKind::ValueBinding:
        return SemanticsExpressionCategory::Value;

      case SemanticsNodeKind::StructMemberAccess: {
        auto [base_id, member_index] = node.GetAsStructMemberAccess();
        node_id = base_id;
        continue;
      }

      case SemanticsNodeKind::Index: {
        auto [base_id, index_id] = node.GetAsIndex();
        node_id = base_id;
        continue;
      }

      case SemanticsNodeKind::StubReference: {
        node_id = node.GetAsStubReference();
        continue;
      }

      case SemanticsNodeKind::StructValue:
      case SemanticsNodeKind::TupleValue:
        // TODO: Eventually these will depend on the context in which the value
        // is used, and could be either Value or Initializing. We may want
        // different node kinds for a struct/tuple initializer versus a
        // struct/tuple value construction.
        return SemanticsExpressionCategory::Value;

      case SemanticsNodeKind::InitializeFrom:
        return SemanticsExpressionCategory::Initializing;

      case SemanticsNodeKind::Dereference:
      case SemanticsNodeKind::VarStorage:
        return SemanticsExpressionCategory::DurableReference;

      case SemanticsNodeKind::MaterializeTemporary:
        return SemanticsExpressionCategory::EphemeralReference;
    }
  }
}

}  // namespace Carbon
