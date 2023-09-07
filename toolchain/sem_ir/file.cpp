// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/file.h"

#include "common/check.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/sem_ir/builtin_kind.h"
#include "toolchain/sem_ir/node.h"
#include "toolchain/sem_ir/node_kind.h"

namespace Carbon::SemIR {

File::File()
    // Builtins are always the first IR, even when self-referential.
    : filename_("<builtins>"),
      cross_reference_irs_({this}),
      // Default entry for NodeBlockId::Empty.
      node_blocks_(1) {
  nodes_.reserve(BuiltinKind::ValidCount);

  // Error uses a self-referential type so that it's not accidentally treated as
  // a normal type. Every other builtin is a type, including the
  // self-referential TypeType.
#define CARBON_SEMANTICS_BUILTIN_KIND(Name, ...)                               \
  nodes_.push_back(Node::Builtin::Make(BuiltinKind::Name,                      \
                                       BuiltinKind::Name == BuiltinKind::Error \
                                           ? TypeId::Error                     \
                                           : TypeId::TypeType));
#include "toolchain/sem_ir/builtin_kind.def"

  CARBON_CHECK(nodes_.size() == BuiltinKind::ValidCount)
      << "Builtins should produce " << BuiltinKind::ValidCount
      << " nodes, actual: " << nodes_.size();
}

File::File(std::string filename, const File* builtins)
    // Builtins are always the first IR.
    : filename_(std::move(filename)),
      cross_reference_irs_({builtins}),
      // Default entry for NodeBlockId::Empty.
      node_blocks_(1) {
  CARBON_CHECK(builtins != nullptr);
  CARBON_CHECK(builtins->cross_reference_irs_[0] == builtins)
      << "Not called with builtins!";

  // Copy builtins over.
  nodes_.reserve(BuiltinKind::ValidCount);
  static constexpr auto BuiltinIR = CrossReferenceIRId(0);
  for (auto [i, node] : llvm::enumerate(builtins->nodes_)) {
    // We can reuse builtin type IDs because they're special-cased values.
    nodes_.push_back(Node::CrossReference::Make(node.type_id(), BuiltinIR,
                                                SemIR::NodeId(i)));
  }
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

static constexpr int BaseIndent = 4;
static constexpr int IndentStep = 2;

// Define PrintList for ArrayRef.
template <typename T, typename PrintT =
                          std::function<void(llvm::raw_ostream&, const T& val)>>
static auto PrintList(
    llvm::raw_ostream& out, llvm::StringLiteral name, llvm::ArrayRef<T> list,
    PrintT print = [](llvm::raw_ostream& out, const T& val) { out << val; }) {
  out.indent(BaseIndent);
  out << name << ": [\n";
  for (const auto& element : list) {
    out.indent(BaseIndent + IndentStep);
    print(out, element);
    out << ",\n";
  }
  out.indent(BaseIndent);
  out << "]\n";
}

// Adapt PrintList for a vector.
template <typename T, typename PrintT =
                          std::function<void(llvm::raw_ostream&, const T& val)>>
static auto PrintList(
    llvm::raw_ostream& out, llvm::StringLiteral name,
    const llvm::SmallVector<T>& list,
    PrintT print = [](llvm::raw_ostream& out, const T& val) { out << val; }) {
  PrintList(out, name, llvm::ArrayRef(list), print);
}

// PrintBlock is only used for vectors.
template <typename T>
static auto PrintBlock(llvm::raw_ostream& out, llvm::StringLiteral block_name,
                       const llvm::SmallVector<T>& blocks) {
  out.indent(BaseIndent);
  out << block_name << ": [\n";
  for (const auto& block : blocks) {
    out.indent(BaseIndent + IndentStep);
    out << "[\n";

    for (const auto& node : block) {
      out.indent(BaseIndent + 2 * IndentStep);
      out << node << ",\n";
    }
    out.indent(BaseIndent + IndentStep);
    out << "],\n";
  }
  out.indent(BaseIndent);
  out << "]\n";
}

auto File::Print(llvm::raw_ostream& out, bool include_builtins) const -> void {
  out << "- filename: " << filename_ << "\n"
      << "  sem_ir:\n"
      << "  - cross_reference_irs_size: " << cross_reference_irs_.size()
      << "\n";

  PrintList(out, "functions", functions_);
  // Integer literals are an APInt, and default to a signed print, but the
  // ZExtValue print is correct.
  PrintList(out, "integer_literals", integer_literals_,
            [](llvm::raw_ostream& out, const llvm::APInt& val) {
              val.print(out, /*isSigned=*/false);
            });
  PrintList(out, "real_literals", real_literals_);
  PrintList(out, "strings", strings_);
  PrintList(out, "types", types_);
  PrintBlock(out, "type_blocks", type_blocks_);

  llvm::ArrayRef nodes = nodes_;
  if (!include_builtins) {
    nodes = nodes.drop_front(BuiltinKind::ValidCount);
  }
  PrintList(out, "nodes", nodes);

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
    case NodeKind::ArrayInit:
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
    case NodeKind::InitializeFrom:
    case NodeKind::IntegerLiteral:
    case NodeKind::Invalid:
    case NodeKind::Namespace:
    case NodeKind::NoOp:
    case NodeKind::Parameter:
    case NodeKind::RealLiteral:
    case NodeKind::Return:
    case NodeKind::ReturnExpression:
    case NodeKind::StringLiteral:
    case NodeKind::StructAccess:
    case NodeKind::StructTypeField:
    case NodeKind::StructLiteral:
    case NodeKind::StubReference:
    case NodeKind::Temporary:
    case NodeKind::TemporaryStorage:
    case NodeKind::TupleIndex:
    case NodeKind::TupleLiteral:
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
      case NodeKind::ArrayInit:
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
      case NodeKind::InitializeFrom:
      case NodeKind::IntegerLiteral:
      case NodeKind::Namespace:
      case NodeKind::NoOp:
      case NodeKind::Parameter:
      case NodeKind::RealLiteral:
      case NodeKind::Return:
      case NodeKind::ReturnExpression:
      case NodeKind::StringLiteral:
      case NodeKind::StructAccess:
      case NodeKind::StructLiteral:
      case NodeKind::StubReference:
      case NodeKind::Temporary:
      case NodeKind::TemporaryStorage:
      case NodeKind::TupleIndex:
      case NodeKind::TupleLiteral:
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

      case NodeKind::StructLiteral:
      case NodeKind::TupleLiteral:
        // TODO: Eventually these will depend on the context in which the value
        // is used, and could be either Value or Initializing. We may want
        // different node kinds for a struct/tuple initializer versus a
        // struct/tuple value construction.
        return ExpressionCategory::Value;

      case NodeKind::ArrayInit:
      case NodeKind::Call:
      case NodeKind::InitializeFrom:
        return ExpressionCategory::Initializing;

      case NodeKind::Dereference:
      case NodeKind::VarStorage:
        return ExpressionCategory::DurableReference;

      case NodeKind::Temporary:
      case NodeKind::TemporaryStorage:
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
      case NodeKind::ArrayInit:
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
      case NodeKind::InitializeFrom:
      case NodeKind::IntegerLiteral:
      case NodeKind::Invalid:
      case NodeKind::Namespace:
      case NodeKind::NoOp:
      case NodeKind::Parameter:
      case NodeKind::RealLiteral:
      case NodeKind::Return:
      case NodeKind::ReturnExpression:
      case NodeKind::StringLiteral:
      case NodeKind::StructAccess:
      case NodeKind::StructTypeField:
      case NodeKind::StructLiteral:
      case NodeKind::Temporary:
      case NodeKind::TemporaryStorage:
      case NodeKind::TupleIndex:
      case NodeKind::TupleLiteral:
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
