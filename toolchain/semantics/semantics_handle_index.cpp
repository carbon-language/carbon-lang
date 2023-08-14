// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_builtin_kind.h"
#include "toolchain/semantics/semantics_context.h"
#include "toolchain/semantics/semantics_node.h"
#include "toolchain/semantics/semantics_node_kind.h"

namespace Carbon {

auto SemanticsHandleIndexExpressionStart(SemanticsContext& /*context*/,
                                         ParseTree::Node /*parse_node*/)
    -> bool {
  // Leave the expression on the stack for IndexExpression.
  return true;
}

auto SemanticsHandleIndexExpression(SemanticsContext& context,
                                    ParseTree::Node parse_node) -> bool {
  auto index_node_id = context.node_stack().PopExpression();
  auto index_node = context.semantics_ir().GetNode(index_node_id);
  auto name_node_id = context.node_stack().PopExpression();
  auto name_node = context.semantics_ir().GetNode(name_node_id);
  auto name_type_id =
      context.semantics_ir().GetTypeAllowBuiltinTypes(name_node.type_id());
  auto name_type_node = context.semantics_ir().GetNode(name_type_id);

  if (name_type_node.kind() == SemanticsNodeKind::ArrayType) {
    auto [bound_id, type_id] = name_type_node.GetAsArrayType();
    if (index_node.kind() == SemanticsNodeKind::IntegerLiteral) {
      const auto& index_val = context.semantics_ir().GetIntegerLiteral(
          index_node.GetAsIntegerLiteral());
      if (index_val.uge(context.semantics_ir().GetArrayBoundValue(bound_id))) {
        context.DiagnoseOutOfBounds(parse_node, index_val, name_node);
      } else {
        context.AddNodeAndPush(
            parse_node, SemanticsNode::ArrayIndex::Make(
                            parse_node, type_id, name_node_id, index_node_id));
        return true;
      }
    } else if (SemanticsBuiltinKind::FromInt(
                   context.semantics_ir()
                       .GetTypeAllowBuiltinTypes(index_node.type_id())
                       .index) ==
               Internal::SemanticsBuiltinKindRawEnum::IntegerType) {
      context.AddNodeAndPush(
          parse_node, SemanticsNode::ArrayIndex::Make(
                          parse_node, type_id, name_node_id, index_node_id));
      return true;
    } else if (name_type_id != SemanticsNodeId::BuiltinError) {
      context.DiagnoseUndeterministicType(parse_node);
    }
  } else if (name_type_node.kind() == SemanticsNodeKind::TupleType &&
             index_node.kind() == SemanticsNodeKind::IntegerLiteral) {
    const auto& index_val = context.semantics_ir().GetIntegerLiteral(
        index_node.GetAsIntegerLiteral());
    auto type_block =
        context.semantics_ir().GetTypeBlock(name_type_node.GetAsTupleType());

    if (index_val.uge(static_cast<uint64_t>(type_block.size()))) {
      context.DiagnoseOutOfBounds(parse_node, index_val, name_node);
    } else {
      context.AddNodeAndPush(
          parse_node, SemanticsNode::TupleIndex::Make(
                          parse_node, type_block[index_val.getZExtValue()],
                          name_node_id, index_node_id));
      return true;
    }
  } else if (index_node.kind() != SemanticsNodeKind::IntegerLiteral) {
    context.DiagnoseUndeterministicType(parse_node);
  } else if ((name_type_node.kind() != SemanticsNodeKind::TupleType ||
              name_type_node.kind() != SemanticsNodeKind::ArrayType) &&
             name_type_id != SemanticsNodeId::BuiltinError) {
    CARBON_DIAGNOSTIC(InvalidIndexExpression, Error,
                      "Invalid index expression.");
    context.emitter().Emit(parse_node, InvalidIndexExpression);
  }

  context.node_stack().Push(parse_node, SemanticsNodeId::BuiltinError);
  return true;
}

}  // namespace Carbon
