// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/semantics/semantics_context.h"
#include "toolchain/semantics/semantics_node.h"
#include "toolchain/semantics/semantics_node_kind.h"

namespace Carbon {

auto SemanticsHandleArrayExpressionStart(SemanticsContext& /*context*/,
                                         ParseTree::Node /*parse_node*/)
    -> bool {
  return true;
}

auto SemanticsHandleArrayExpressionSemi(SemanticsContext& context,
                                        ParseTree::Node parse_node) -> bool {
  context.node_stack().Push(parse_node);
  return true;
}

auto SemanticsHandleArrayExpression(SemanticsContext& context,
                                    ParseTree::Node parse_node) -> bool {
  // TODO: Handle array type with undefined bound.
  if (context.parse_tree().node_kind(context.node_stack().PeekParseNode()) ==
      ParseNodeKind::ArrayExpressionSemi) {
    context.node_stack().PopAndIgnore();
    context.node_stack().PopAndIgnore();
    return context.TODO(parse_node, "HandleArrayExpressionWithoutBounds");
  }

  auto bound_node_id = context.node_stack().PopExpression();
  context.node_stack()
      .PopAndDiscardSoloParseNode<ParseNodeKind::ArrayExpressionSemi>();
  auto element_type_node_id = context.node_stack().PopExpression();
  auto bound_node = context.semantics_ir().GetNode(bound_node_id);
  if (bound_node.kind() == SemanticsNodeKind::IntegerLiteral) {
    auto bound_value = context.semantics_ir().GetIntegerLiteral(
        bound_node.GetAsIntegerLiteral());
    if (!bound_value.isNegative()) {
      context.AddNodeAndPush(
          parse_node,
          SemanticsNode::ArrayType::Make(
              parse_node, SemanticsTypeId::TypeType, bound_node_id,
              context.ExpressionAsType(parse_node, element_type_node_id)));
      return true;
    }
  }
  CARBON_DIAGNOSTIC(InvalidArrayExpression, Error, "Invalid array expression.");
  context.emitter().Emit(parse_node, InvalidArrayExpression);
  context.node_stack().Push(parse_node, SemanticsNodeId::BuiltinError);
  return true;
}

}  // namespace Carbon
