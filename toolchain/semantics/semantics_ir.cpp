// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_ir.h"

#include "common/check.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/base/pretty_stack_trace_function.h"
#include "toolchain/parser/parse_tree_node_location_translator.h"
#include "toolchain/semantics/semantics_builtin_kind.h"
#include "toolchain/semantics/semantics_context.h"
#include "toolchain/semantics/semantics_node.h"
#include "toolchain/semantics/semantics_node_kind.h"

namespace Carbon::SemIR {

auto File::MakeBuiltinIR() -> File {
  File semantics_ir(/*builtin_ir=*/nullptr);
  semantics_ir.nodes_.reserve(BuiltinKind::ValidCount);

  // Error uses a self-referential type so that it's not accidentally treated as
  // a normal type. Every other builtin is a type, including the
  // self-referential TypeType.
#define CARBON_SEMANTICS_BUILTIN_KIND(Name, ...)                 \
  semantics_ir.nodes_.push_back(Node::Builtin::Make(             \
      BuiltinKind::Name, BuiltinKind::Name == BuiltinKind::Error \
                             ? TypeId::Error                     \
                             : TypeId::TypeType));
#include "toolchain/semantics/semantics_builtin_kind.def"

  CARBON_CHECK(semantics_ir.node_blocks_.size() == 1)
      << "BuildBuiltins should only have the empty block, actual: "
      << semantics_ir.node_blocks_.size();
  CARBON_CHECK(semantics_ir.nodes_.size() == BuiltinKind::ValidCount)
      << "BuildBuiltins should produce " << BuiltinKind::ValidCount
      << " nodes, actual: " << semantics_ir.nodes_.size();
  return semantics_ir;
}

auto File::MakeFromParseTree(const File& builtin_ir,
                             const TokenizedBuffer& tokens,
                             const ParseTree& parse_tree,
                             DiagnosticConsumer& consumer,
                             llvm::raw_ostream* vlog_stream) -> File {
  File semantics_ir(&builtin_ir);

  // Copy builtins over.
  semantics_ir.nodes_.resize_for_overwrite(BuiltinKind::ValidCount);
  static constexpr auto BuiltinIR = CrossReferenceIRId(0);
  for (int i : llvm::seq(BuiltinKind::ValidCount)) {
    // We can reuse the type node ID because the offsets of cross-references
    // will be the same in this IR.
    auto type = builtin_ir.nodes_[i].type_id();
    semantics_ir.nodes_[i] =
        Node::CrossReference::Make(type, BuiltinIR, NodeId(i));
  }

  ParseTreeNodeLocationTranslator translator(&tokens, &parse_tree);
  ErrorTrackingDiagnosticConsumer err_tracker(consumer);
  DiagnosticEmitter<ParseTree::Node> emitter(translator, err_tracker);

  Check::Context context(tokens, emitter, parse_tree, semantics_ir,
                         vlog_stream);
  PrettyStackTraceFunction context_dumper(
      [&](llvm::raw_ostream& output) { context.PrintForStackDump(output); });

  // Add a block for the ParseTree.
  context.node_block_stack().Push();
  context.PushScope();

  // Loops over all nodes in the tree. On some errors, this may return early,
  // for example if an unrecoverable state is encountered.
  for (auto parse_node : parse_tree.postorder()) {
    // clang warns on unhandled enum values; clang-tidy is incorrect here.
    // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
    switch (auto parse_kind = parse_tree.node_kind(parse_node)) {
#define CARBON_PARSE_NODE_KIND(Name)                 \
  case ParseNodeKind::Name: {                        \
    if (!Check::Handle##Name(context, parse_node)) { \
      semantics_ir.has_errors_ = true;               \
      return semantics_ir;                           \
    }                                                \
    break;                                           \
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

auto File::Verify() const -> ErrorOr<Success> {
  // Invariants don't necessarily hold for invalid IR.
  if (has_errors_) {
    return Success();
  }

  // Check that every code block has a terminator sequence that appears at the
  // end of the block.
  for (const Function& function : functions_) {
    for (NodeBlockId block_id : function.body_block_ids) {
      TerminatorKind prior_kind = TerminatorKind::NotTerminator;
      for (NodeId node_id : GetNodeBlock(block_id)) {
        TerminatorKind node_kind = GetNode(node_id).kind().terminator_kind();
        if (prior_kind == TerminatorKind::Terminator) {
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
      if (prior_kind != TerminatorKind::Terminator) {
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

auto File::Print(llvm::raw_ostream& out, bool include_builtins) const -> void {
  out << "cross_reference_irs_size: " << cross_reference_irs_.size() << "\n";

  PrintList(out, "functions", functions_);
  PrintList(out, "integer_literals", integer_literals_);
  PrintList(out, "real_literals", real_literals_);
  PrintList(out, "strings", strings_);
  PrintList(out, "types", types_);

  PrintBlock(out, "type_blocks", type_blocks_);

  out << "nodes: [\n";
  for (int i = include_builtins ? 0 : BuiltinKind::ValidCount;
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
static auto GetTypePrecedence(NodeKind kind) -> int {
  // clang warns on unhandled enum values; clang-tidy is incorrect here.
  // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
  switch (kind) {
    case NodeKind::ArrayType:
    case NodeKind::Builtin:
    case NodeKind::StructType:
    case NodeKind::TupleType:
      return 0;
    case NodeKind::ConstType:
      return -1;
    case NodeKind::PointerType:
      return -2;

    case NodeKind::CrossReference:
      // TODO: Once we support stringification of cross-references, we'll need
      // to determine the precedence of the target of the cross-reference. For
      // now, all cross-references refer to builtin types from the prelude.
      return 0;

    case NodeKind::AddressOf:
    case NodeKind::ArrayIndex:
    case NodeKind::ArrayValue:
    case NodeKind::Assign:
    case NodeKind::BinaryOperatorAdd:
    case NodeKind::BindValue:
    case NodeKind::BlockArg:
    case NodeKind::BoolLiteral:
    case NodeKind::Branch:
    case NodeKind::BranchIf:
    case NodeKind::BranchWithArg:
    case NodeKind::Call:
    case NodeKind::Dereference:
    case NodeKind::FunctionDeclaration:
    case NodeKind::IntegerLiteral:
    case NodeKind::Invalid:
    case NodeKind::MaterializeTemporary:
    case NodeKind::Namespace:
    case NodeKind::NoOp:
    case NodeKind::Parameter:
    case NodeKind::RealLiteral:
    case NodeKind::Return:
    case NodeKind::ReturnExpression:
    case NodeKind::StringLiteral:
    case NodeKind::StructAccess:
    case NodeKind::StructTypeField:
    case NodeKind::StructValue:
    case NodeKind::StubReference:
    case NodeKind::TupleIndex:
    case NodeKind::TupleValue:
    case NodeKind::UnaryOperatorNot:
    case NodeKind::VarStorage:
      CARBON_FATAL() << "GetTypePrecedence for non-type node kind " << kind;
  }
}

auto File::StringifyType(TypeId type_id, bool in_type_context) const
    -> std::string {
  std::string str;
  llvm::raw_string_ostream out(str);

  struct Step {
    // The node to print.
    NodeId node_id;
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
    if (step.node_id.index < BuiltinKind::ValidCount) {
      out << BuiltinKind::FromInt(step.node_id.index).label();
      continue;
    }

    auto node = GetNode(step.node_id);
    // clang warns on unhandled enum values; clang-tidy is incorrect here.
    // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
    switch (node.kind()) {
      case NodeKind::ArrayType: {
        auto [bound_id, type_id] = node.GetAsArrayType();
        if (step.index == 0) {
          out << "[";
          steps.push_back(step.Next());
          steps.push_back({.node_id = GetTypeAllowBuiltinTypes(type_id)});
        } else if (step.index == 1) {
          out << "; " << GetArrayBoundValue(bound_id) << "]";
        }
        break;
      }
      case NodeKind::ConstType: {
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
      case NodeKind::PointerType: {
        if (step.index == 0) {
          steps.push_back(step.Next());
          steps.push_back(
              {.node_id = GetTypeAllowBuiltinTypes(node.GetAsPointerType())});
        } else if (step.index == 1) {
          out << "*";
        }
        break;
      }
      case NodeKind::StructType: {
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
      case NodeKind::StructTypeField: {
        auto [name_id, type_id] = node.GetAsStructTypeField();
        out << "." << GetString(name_id) << ": ";
        steps.push_back({.node_id = GetTypeAllowBuiltinTypes(type_id)});
        break;
      }
      case NodeKind::TupleType: {
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
      case NodeKind::AddressOf:
      case NodeKind::ArrayIndex:
      case NodeKind::ArrayValue:
      case NodeKind::Assign:
      case NodeKind::BinaryOperatorAdd:
      case NodeKind::BindValue:
      case NodeKind::BlockArg:
      case NodeKind::BoolLiteral:
      case NodeKind::Branch:
      case NodeKind::BranchIf:
      case NodeKind::BranchWithArg:
      case NodeKind::Builtin:
      case NodeKind::Call:
      case NodeKind::CrossReference:
      case NodeKind::Dereference:
      case NodeKind::FunctionDeclaration:
      case NodeKind::IntegerLiteral:
      case NodeKind::MaterializeTemporary:
      case NodeKind::Namespace:
      case NodeKind::NoOp:
      case NodeKind::Parameter:
      case NodeKind::RealLiteral:
      case NodeKind::Return:
      case NodeKind::ReturnExpression:
      case NodeKind::StringLiteral:
      case NodeKind::StructAccess:
      case NodeKind::StructValue:
      case NodeKind::StubReference:
      case NodeKind::TupleIndex:
      case NodeKind::TupleValue:
      case NodeKind::UnaryOperatorNot:
      case NodeKind::VarStorage:
        // We don't need to handle stringification for nodes that don't show up
        // in errors, but make it clear what's going on so that it's clearer
        // when stringification is needed.
        out << "<cannot stringify " << step.node_id << ">";
        break;
      case NodeKind::Invalid:
        llvm_unreachable("NodeKind::Invalid is never used.");
    }
  }

  // For `{}` or any tuple type, we've printed a non-type expression, so add a
  // conversion to type `type` if it's not implied by the context.
  if (!in_type_context) {
    auto outer_node = GetNode(outer_node_id);
    if (outer_node.kind() == NodeKind::TupleType ||
        (outer_node.kind() == NodeKind::StructType &&
         GetNodeBlock(outer_node.GetAsStructType()).empty())) {
      out << " as type";
    }
  }

  return str;
}

auto GetExpressionCategory(const File& file, NodeId node_id)
    -> ExpressionCategory {
  const File* ir = &file;
  while (true) {
    auto node = ir->GetNode(node_id);
    // clang warns on unhandled enum values; clang-tidy is incorrect here.
    // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
    switch (node.kind()) {
      case NodeKind::Invalid:
      case NodeKind::Assign:
      case NodeKind::Branch:
      case NodeKind::BranchIf:
      case NodeKind::BranchWithArg:
      case NodeKind::FunctionDeclaration:
      case NodeKind::Namespace:
      case NodeKind::NoOp:
      case NodeKind::Return:
      case NodeKind::ReturnExpression:
      case NodeKind::StructTypeField:
        return ExpressionCategory::NotExpression;

      case NodeKind::CrossReference: {
        auto [xref_id, xref_node_id] = node.GetAsCrossReference();
        ir = &ir->GetCrossReferenceIR(xref_id);
        node_id = xref_node_id;
        continue;
      }

      case NodeKind::AddressOf:
      case NodeKind::ArrayType:
      case NodeKind::BinaryOperatorAdd:
      case NodeKind::BindValue:
      case NodeKind::BlockArg:
      case NodeKind::BoolLiteral:
      case NodeKind::Builtin:
      case NodeKind::ConstType:
      case NodeKind::IntegerLiteral:
      case NodeKind::Parameter:
      case NodeKind::PointerType:
      case NodeKind::RealLiteral:
      case NodeKind::StringLiteral:
      case NodeKind::StructType:
      case NodeKind::TupleType:
      case NodeKind::UnaryOperatorNot:
        return ExpressionCategory::Value;

      case NodeKind::ArrayIndex: {
        auto [base_id, index_id] = node.GetAsArrayIndex();
        node_id = base_id;
        continue;
      }

      case NodeKind::StructAccess: {
        auto [base_id, member_index] = node.GetAsStructAccess();
        node_id = base_id;
        continue;
      }

      case NodeKind::TupleIndex: {
        auto [base_id, index_id] = node.GetAsTupleIndex();
        node_id = base_id;
        continue;
      }

      case NodeKind::StubReference: {
        node_id = node.GetAsStubReference();
        continue;
      }

      case NodeKind::ArrayValue:
      case NodeKind::StructValue:
      case NodeKind::TupleValue:
        // TODO: Eventually these will depend on the context in which the value
        // is used, and could be either Value or Initializing. We may want
        // different node kinds for a struct/tuple initializer versus a
        // struct/tuple value construction.
        return ExpressionCategory::Value;

      case NodeKind::Call:
        return ExpressionCategory::Initializing;

      case NodeKind::Dereference:
      case NodeKind::VarStorage:
        return ExpressionCategory::DurableReference;

      case NodeKind::MaterializeTemporary:
        return ExpressionCategory::EphemeralReference;
    }
  }
}

auto GetValueRepresentation(const File& file, TypeId type_id)
    -> ValueRepresentation {
  const File* ir = &file;
  NodeId node_id = ir->GetTypeAllowBuiltinTypes(type_id);
  while (true) {
    auto node = ir->GetNode(node_id);
    // clang warns on unhandled enum values; clang-tidy is incorrect here.
    // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
    switch (node.kind()) {
      case NodeKind::AddressOf:
      case NodeKind::ArrayIndex:
      case NodeKind::ArrayValue:
      case NodeKind::Assign:
      case NodeKind::BinaryOperatorAdd:
      case NodeKind::BindValue:
      case NodeKind::BlockArg:
      case NodeKind::BoolLiteral:
      case NodeKind::Branch:
      case NodeKind::BranchIf:
      case NodeKind::BranchWithArg:
      case NodeKind::Call:
      case NodeKind::Dereference:
      case NodeKind::FunctionDeclaration:
      case NodeKind::IntegerLiteral:
      case NodeKind::Invalid:
      case NodeKind::MaterializeTemporary:
      case NodeKind::Namespace:
      case NodeKind::NoOp:
      case NodeKind::Parameter:
      case NodeKind::RealLiteral:
      case NodeKind::Return:
      case NodeKind::ReturnExpression:
      case NodeKind::StringLiteral:
      case NodeKind::StructAccess:
      case NodeKind::StructTypeField:
      case NodeKind::StructValue:
      case NodeKind::TupleIndex:
      case NodeKind::TupleValue:
      case NodeKind::UnaryOperatorNot:
      case NodeKind::VarStorage:
        CARBON_FATAL() << "Type refers to non-type node " << node;

      case NodeKind::CrossReference: {
        auto [xref_id, xref_node_id] = node.GetAsCrossReference();
        ir = &ir->GetCrossReferenceIR(xref_id);
        node_id = xref_node_id;
        continue;
      }

      case NodeKind::StubReference: {
        node_id = node.GetAsStubReference();
        continue;
      }

      case NodeKind::ArrayType:
        // For arrays, it's convenient to always use a pointer representation,
        // even when the array has zero or one element, in order to support
        // indexing.
        return {.kind = ValueRepresentation::Pointer, .type = type_id};

      case NodeKind::StructType: {
        const auto& fields = ir->GetNodeBlock(node.GetAsStructType());
        if (fields.empty()) {
          // An empty struct has an empty representation.
          return {.kind = ValueRepresentation::None, .type = TypeId::Invalid};
        }
        if (fields.size() == 1) {
          // A struct with one field has the same representation as its field.
          auto [field_name_id, field_type_id] =
              ir->GetNode(fields.front()).GetAsStructTypeField();
          node_id = ir->GetTypeAllowBuiltinTypes(field_type_id);
          continue;
        }
        // For any other struct, use a pointer representation.
        return {.kind = ValueRepresentation::Pointer, .type = type_id};
      }

      case NodeKind::TupleType: {
        const auto& elements = ir->GetTypeBlock(node.GetAsTupleType());
        if (elements.empty()) {
          // An empty tuple has an empty representation.
          return {.kind = ValueRepresentation::None, .type = TypeId::Invalid};
        }
        if (elements.size() == 1) {
          // A one-tuple has the same representation as its sole element.
          node_id = ir->GetTypeAllowBuiltinTypes(elements.front());
          continue;
        }
        // For any other tuple, use a pointer representation.
        return {.kind = ValueRepresentation::Pointer, .type = type_id};
      }

      case NodeKind::Builtin:
        // clang warns on unhandled enum values; clang-tidy is incorrect here.
        // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
        switch (node.GetAsBuiltin()) {
          case BuiltinKind::TypeType:
          case BuiltinKind::Error:
          case BuiltinKind::Invalid:
            return {.kind = ValueRepresentation::None, .type = TypeId::Invalid};
          case BuiltinKind::BoolType:
          case BuiltinKind::IntegerType:
          case BuiltinKind::FloatingPointType:
            return {.kind = ValueRepresentation::Copy, .type = type_id};
          case BuiltinKind::StringType:
            // TODO: Decide on string value semantics. This should probably be a
            // custom value representation carrying a pointer and size or
            // similar.
            return {.kind = ValueRepresentation::Pointer, .type = type_id};
        }

      case NodeKind::PointerType:
        return {.kind = ValueRepresentation::Copy, .type = type_id};

      case NodeKind::ConstType:
        node_id = ir->GetTypeAllowBuiltinTypes(node.GetAsConstType());
        continue;
    }
  }
}

auto GetInitializingRepresentation(const File& file, TypeId type_id)
    -> InitializingRepresentation {
  auto value_rep = GetValueRepresentation(file, type_id);
  switch (value_rep.kind) {
    case ValueRepresentation::None:
      return {.kind = InitializingRepresentation::None};

    case ValueRepresentation::Copy:
      // TODO: Use in-place initialization for types that have non-trivial
      // destructive move.
      return {.kind = InitializingRepresentation::ByCopy};

    case ValueRepresentation::Pointer:
    case ValueRepresentation::Custom:
      return {.kind = InitializingRepresentation::InPlace};
  }
}

}  // namespace Carbon::SemIR
