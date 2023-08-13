// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>

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

  if (name_type_node.kind() == SemanticsNodeKind::TupleType &&
      index_node.kind() == SemanticsNodeKind::IntegerLiteral) {
    const auto& index_val = context.semantics_ir().GetIntegerLiteral(
        index_node.GetAsIntegerLiteral());
    auto type_block =
        context.semantics_ir().GetTypeBlock(name_type_node.GetAsTupleType());

    if (index_val.uge(static_cast<uint64_t>(type_block.size()))) {
      CARBON_DIAGNOSTIC(OutOfBoundsAccess, Error,
                        "Index `{0}` is past the end of `{1}`.", llvm::APSInt,
                        std::string);
      context.emitter().Emit(
          parse_node, OutOfBoundsAccess,
          llvm::APSInt(index_val, /*isUnsigned=*/true),
          context.semantics_ir().StringifyType(name_node.type_id()));
    } else {
      context.AddNodeAndPush(
          parse_node, SemanticsNode::TupleIndex::Make(
                          parse_node, type_block[index_val.getZExtValue()],
                          name_node_id, index_node_id));
      return true;
    }
  } else if (index_node.kind() != SemanticsNodeKind::IntegerLiteral) {
    CARBON_DIAGNOSTIC(NondeterministicType, Error,
                      "Type cannot be determined at compile time.");
    context.emitter().Emit(parse_node, NondeterministicType);
  } else if (name_type_node.kind() != SemanticsNodeKind::TupleType &&
             name_type_id != SemanticsNodeId::BuiltinError) {
    CARBON_DIAGNOSTIC(InvalidIndexExpression, Error,
                      "Invalid index expression.");
    context.emitter().Emit(parse_node, InvalidIndexExpression);
  }

  context.node_stack().Push(parse_node, SemanticsNodeId::BuiltinError);
  return true;
}

}  // namespace Carbon
