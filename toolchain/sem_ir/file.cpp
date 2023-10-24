// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/file.h"

#include "common/check.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/base/value_store.h"
#include "toolchain/sem_ir/builtin_kind.h"
#include "toolchain/sem_ir/node.h"
#include "toolchain/sem_ir/node_kind.h"

namespace Carbon::SemIR {

auto ValueRepresentation::Print(llvm::raw_ostream& out) const -> void {
  out << "{kind: ";
  switch (kind) {
    case Unknown:
      out << "unknown";
      break;
    case None:
      out << "none";
      break;
    case Copy:
      out << "copy";
      break;
    case Pointer:
      out << "pointer";
      break;
    case Custom:
      out << "custom";
      break;
  }
  out << ", type: " << type_id << "}";
}

auto TypeInfo::Print(llvm::raw_ostream& out) const -> void {
  out << "{node: " << node_id << ", value_rep: " << value_representation << "}";
}

File::File(SharedValueStores& value_stores)
    : value_stores_(&value_stores),
      filename_("<builtins>"),
      // Builtins are always the first IR, even when self-referential.
      cross_reference_irs_({this}),
      // Default entry for NodeBlockId::Empty.
      node_blocks_(1) {
  nodes_.reserve(BuiltinKind::ValidCount);

  // Error uses a self-referential type so that it's not accidentally treated as
  // a normal type. Every other builtin is a type, including the
  // self-referential TypeType.
#define CARBON_SEM_IR_BUILTIN_KIND(Name, ...)                      \
  nodes_.push_back(Builtin{BuiltinKind::Name == BuiltinKind::Error \
                               ? TypeId::Error                     \
                               : TypeId::TypeType,                 \
                           BuiltinKind::Name});
#include "toolchain/sem_ir/builtin_kind.def"

  CARBON_CHECK(nodes_.size() == BuiltinKind::ValidCount)
      << "Builtins should produce " << BuiltinKind::ValidCount
      << " nodes, actual: " << nodes_.size();
}

File::File(SharedValueStores& value_stores, std::string filename,
           const File* builtins)
    : value_stores_(&value_stores),
      filename_(std::move(filename)),
      // Builtins are always the first IR.
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
    nodes_.push_back(
        CrossReference{node.type_id(), BuiltinIR, SemIR::NodeId(i)});
  }
}

auto File::Verify() const -> ErrorOr<Success> {
  // Invariants don't necessarily hold for invalid IR.
  if (has_errors_) {
    return Success();
  }

  // Check that every code block has a terminator sequence that appears at the
  // end of the block.
  for (const Function& function : functions_.array_ref()) {
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

// Prints a list of elements.
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

  PrintList(out, "functions", functions_.array_ref());
  PrintList(out, "classes", classes_.array_ref());
  PrintList(out, "types", types_.array_ref());
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
    case ArrayType::Kind:
    case Builtin::Kind:
    case ClassDeclaration::Kind:
    case NameReference::Kind:
    case StructType::Kind:
    case TupleType::Kind:
    case ClassType::Kind:
      return 0;
    case ConstType::Kind:
      return -1;
    case PointerType::Kind:
      return -2;

    case CrossReference::Kind:
      // TODO: Once we support stringification of cross-references, we'll need
      // to determine the precedence of the target of the cross-reference. For
      // now, all cross-references refer to builtin types from the prelude.
      return 0;

    case AddressOf::Kind:
    case ArrayIndex::Kind:
    case ArrayInit::Kind:
    case Assign::Kind:
    case BinaryOperatorAdd::Kind:
    case BindName::Kind:
    case BindValue::Kind:
    case BlockArg::Kind:
    case BoolLiteral::Kind:
    case Branch::Kind:
    case BranchIf::Kind:
    case BranchWithArg::Kind:
    case Call::Kind:
    case Dereference::Kind:
    case FunctionDeclaration::Kind:
    case InitializeFrom::Kind:
    case IntegerLiteral::Kind:
    case Namespace::Kind:
    case NoOp::Kind:
    case Parameter::Kind:
    case RealLiteral::Kind:
    case Return::Kind:
    case ReturnExpression::Kind:
    case SpliceBlock::Kind:
    case StringLiteral::Kind:
    case StructAccess::Kind:
    case StructTypeField::Kind:
    case StructLiteral::Kind:
    case StructInit::Kind:
    case StructValue::Kind:
    case Temporary::Kind:
    case TemporaryStorage::Kind:
    case TupleAccess::Kind:
    case TupleIndex::Kind:
    case TupleLiteral::Kind:
    case TupleInit::Kind:
    case TupleValue::Kind:
    case UnaryOperatorNot::Kind:
    case ValueAsReference::Kind:
    case VarStorage::Kind:
      CARBON_FATAL() << "GetTypePrecedence for non-type node kind " << kind;
  }
}

auto File::StringifyType(TypeId type_id, bool in_type_context) const
    -> std::string {
  return StringifyTypeExpression(GetTypeAllowBuiltinTypes(type_id),
                                 in_type_context);
}

auto File::StringifyTypeExpression(NodeId outer_node_id,
                                   bool in_type_context) const -> std::string {
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
  llvm::SmallVector<Step> steps = {{.node_id = outer_node_id}};

  while (!steps.empty()) {
    auto step = steps.pop_back_val();
    if (!step.node_id.is_valid()) {
      out << "<invalid type>";
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
      case ArrayType::Kind: {
        auto array = node.As<ArrayType>();
        if (step.index == 0) {
          out << "[";
          steps.push_back(step.Next());
          steps.push_back(
              {.node_id = GetTypeAllowBuiltinTypes(array.element_type_id)});
        } else if (step.index == 1) {
          out << "; " << GetArrayBoundValue(array.bound_id) << "]";
        }
        break;
      }
      case ClassType::Kind: {
        auto class_name_id =
            classes().Get(node.As<ClassType>().class_id).name_id;
        out << strings().Get(class_name_id);
        break;
      }
      case ConstType::Kind: {
        if (step.index == 0) {
          out << "const ";

          // Add parentheses if required.
          auto inner_type_node_id =
              GetTypeAllowBuiltinTypes(node.As<ConstType>().inner_id);
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
      case NameReference::Kind: {
        out << strings().Get(node.As<NameReference>().name_id);
        break;
      }
      case PointerType::Kind: {
        if (step.index == 0) {
          steps.push_back(step.Next());
          steps.push_back({.node_id = GetTypeAllowBuiltinTypes(
                               node.As<PointerType>().pointee_id)});
        } else if (step.index == 1) {
          out << "*";
        }
        break;
      }
      case StructType::Kind: {
        auto refs = GetNodeBlock(node.As<StructType>().fields_id);
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
      case StructTypeField::Kind: {
        auto field = node.As<StructTypeField>();
        out << "." << strings().Get(field.name_id) << ": ";
        steps.push_back(
            {.node_id = GetTypeAllowBuiltinTypes(field.field_type_id)});
        break;
      }
      case TupleType::Kind: {
        auto refs = GetTypeBlock(node.As<TupleType>().elements_id);
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
      case AddressOf::Kind:
      case ArrayIndex::Kind:
      case ArrayInit::Kind:
      case Assign::Kind:
      case BinaryOperatorAdd::Kind:
      case BindName::Kind:
      case BindValue::Kind:
      case BlockArg::Kind:
      case BoolLiteral::Kind:
      case Branch::Kind:
      case BranchIf::Kind:
      case BranchWithArg::Kind:
      case Builtin::Kind:
      case Call::Kind:
      case ClassDeclaration::Kind:
      case CrossReference::Kind:
      case Dereference::Kind:
      case FunctionDeclaration::Kind:
      case InitializeFrom::Kind:
      case IntegerLiteral::Kind:
      case Namespace::Kind:
      case NoOp::Kind:
      case Parameter::Kind:
      case RealLiteral::Kind:
      case Return::Kind:
      case ReturnExpression::Kind:
      case SpliceBlock::Kind:
      case StringLiteral::Kind:
      case StructAccess::Kind:
      case StructLiteral::Kind:
      case StructInit::Kind:
      case StructValue::Kind:
      case Temporary::Kind:
      case TemporaryStorage::Kind:
      case TupleAccess::Kind:
      case TupleIndex::Kind:
      case TupleLiteral::Kind:
      case TupleInit::Kind:
      case TupleValue::Kind:
      case UnaryOperatorNot::Kind:
      case ValueAsReference::Kind:
      case VarStorage::Kind:
        // We don't need to handle stringification for nodes that don't show up
        // in errors, but make it clear what's going on so that it's clearer
        // when stringification is needed.
        out << "<cannot stringify " << step.node_id << ">";
        break;
    }
  }

  // For `{}` or any tuple type, we've printed a non-type expression, so add a
  // conversion to type `type` if it's not implied by the context.
  if (!in_type_context) {
    auto outer_node = GetNode(outer_node_id);
    if (outer_node.Is<TupleType>() ||
        (outer_node.Is<StructType>() &&
         GetNodeBlock(outer_node.As<StructType>().fields_id).empty())) {
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
      case Assign::Kind:
      case Branch::Kind:
      case BranchIf::Kind:
      case BranchWithArg::Kind:
      case ClassDeclaration::Kind:
      case FunctionDeclaration::Kind:
      case Namespace::Kind:
      case NoOp::Kind:
      case Return::Kind:
      case ReturnExpression::Kind:
      case StructTypeField::Kind:
        return ExpressionCategory::NotExpression;

      case CrossReference::Kind: {
        auto xref = node.As<CrossReference>();
        ir = &ir->GetCrossReferenceIR(xref.ir_id);
        node_id = xref.node_id;
        continue;
      }

      case NameReference::Kind: {
        node_id = node.As<NameReference>().value_id;
        continue;
      }

      case AddressOf::Kind:
      case ArrayType::Kind:
      case BinaryOperatorAdd::Kind:
      case BindValue::Kind:
      case BlockArg::Kind:
      case BoolLiteral::Kind:
      case ClassType::Kind:
      case ConstType::Kind:
      case IntegerLiteral::Kind:
      case Parameter::Kind:
      case PointerType::Kind:
      case RealLiteral::Kind:
      case StringLiteral::Kind:
      case StructValue::Kind:
      case StructType::Kind:
      case TupleValue::Kind:
      case TupleType::Kind:
      case UnaryOperatorNot::Kind:
        return ExpressionCategory::Value;

      case Builtin::Kind: {
        if (node.As<Builtin>().builtin_kind == BuiltinKind::Error) {
          return ExpressionCategory::Error;
        }
        return ExpressionCategory::Value;
      }

      case BindName::Kind: {
        node_id = node.As<BindName>().value_id;
        continue;
      }

      case ArrayIndex::Kind: {
        node_id = node.As<ArrayIndex>().array_id;
        continue;
      }

      case StructAccess::Kind: {
        node_id = node.As<StructAccess>().struct_id;
        continue;
      }

      case TupleAccess::Kind: {
        node_id = node.As<TupleAccess>().tuple_id;
        continue;
      }

      case TupleIndex::Kind: {
        node_id = node.As<TupleIndex>().tuple_id;
        continue;
      }

      case SpliceBlock::Kind: {
        node_id = node.As<SpliceBlock>().result_id;
        continue;
      }

      case StructLiteral::Kind:
      case TupleLiteral::Kind:
        return ExpressionCategory::Mixed;

      case ArrayInit::Kind:
      case Call::Kind:
      case InitializeFrom::Kind:
      case StructInit::Kind:
      case TupleInit::Kind:
        return ExpressionCategory::Initializing;

      case Dereference::Kind:
      case VarStorage::Kind:
        return ExpressionCategory::DurableReference;

      case Temporary::Kind:
      case TemporaryStorage::Kind:
      case ValueAsReference::Kind:
        return ExpressionCategory::EphemeralReference;
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

    case ValueRepresentation::Unknown:
      CARBON_FATAL()
          << "Attempting to perform initialization of incomplete type";
  }
}

}  // namespace Carbon::SemIR
