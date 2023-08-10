// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"
#include "toolchain/semantics/semantics_node.h"
#include "toolchain/semantics/semantics_node_kind.h"

namespace Carbon {

auto SemanticsHandleArrayExpressionStart(SemanticsContext& /*context*/,
                                         ParseTree::Node /*parse_node*/)
    -> bool {
  return true;
}

auto SemanticsHandleArrayExpressionSemi(SemanticsContext& /*context*/,
                                        ParseTree::Node /*parse_node*/)
    -> bool {
  return true;
}

auto SemanticsHandleArrayExpression(SemanticsContext& context,
                                    ParseTree::Node parse_node) -> bool {
  // TODO: Handle array type with undefined bound.
  auto bound_node_id = context.node_stack().PopExpression();
  auto element_type_node_id = context.node_stack().PopExpression();
  auto bound_node = context.semantics_ir().GetNode(bound_node_id);
  if (bound_node.kind() == SemanticsNodeKind::IntegerLiteral) {
    auto bound_value = context.semantics_ir().GetIntegerLiteral(
        bound_node.GetAsIntegerLiteral());
    if (!bound_value.isNegative()) {
      context.AddNodeAndPush(
          parse_node, SemanticsNode::ArrayType::Make(
                          parse_node, SemanticsTypeId::TypeType, bound_node_id,
                          context.CanonicalizeType(element_type_node_id)));
      return true;
    }
  }
  CARBON_DIAGNOSTIC(InvalidArrayExpression, Error, "Invalid array expression.");
  context.emitter().Emit(parse_node, InvalidArrayExpression);
  context.node_stack().Push(parse_node, SemanticsNodeId::BuiltinError);
  return true;
}

}  // namespace Carbon
