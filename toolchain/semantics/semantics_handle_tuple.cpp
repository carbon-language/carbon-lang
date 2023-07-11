// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"

namespace Carbon {

auto SemanticsHandleTupleLiteralComma(SemanticsContext& context,
                                      ParseTree::Node /*parse_node*/) -> bool {
  context.ParamOrArgComma(true);
  return true;
}

auto SemanticsHandleTupleFieldType(SemanticsContext& context,
                                   ParseTree::Node parse_node) -> bool {
  auto [type_node, type_id] =
      context.node_stack().PopWithParseNode<SemanticsNodeId>();
  SemanticsTypeId cast_type_id = context.ExpressionAsType(type_node, type_id);

  context.AddNode(
      SemanticsNode::TupleTypeField::Make(parse_node, cast_type_id, type_id));
  context.node_stack().Push(parse_node);
  return true;
}

auto SemanticsHandleTupleFieldUnknown(SemanticsContext& context,
                                      ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleTupleFieldUnknown");
}

auto SemanticsHandleTupleFieldValue(SemanticsContext& context,
                                    ParseTree::Node parse_node) -> bool {
  auto [value_parse_node, value_node_id] =
      context.node_stack().PopWithParseNode<SemanticsNodeId>();
  // Store the name for the type.
  auto type_block_id = context.args_type_info_stack().PeekForAdd();
  context.semantics_ir().AddNode(
      type_block_id,
      SemanticsNode::TupleTypeField::Make(
          parse_node, context.semantics_ir().GetNode(value_node_id).type_id(),
          value_node_id));

  // Push the value back on the stack as an argument.
  context.node_stack().Push(parse_node, value_node_id);
  return true;
}

auto SemanticsHandleTupleLiteral(SemanticsContext& context,
                                 ParseTree::Node parse_node) -> bool {
  auto refs_id = context.ParamOrArgEnd(
      /*for_args=*/true, ParseNodeKind::ParenExpressionOrTupleLiteralStart);

  context.PopScope();
  context.node_stack().PopAndDiscardSoloParseNode(
      ParseNodeKind::ParenExpressionOrTupleLiteralStart);
  auto type_block_id = context.args_type_info_stack().Pop();

  auto type_id = context.CanonicalizeTupleType(parse_node, type_block_id);

  auto value_id = context.AddNode(
      SemanticsNode::TupleValue::Make(parse_node, type_id, refs_id));
  context.node_stack().Push(parse_node, value_id);
  return true;
}

auto SemanticsHandleParenExpressionOrTupleLiteralStart(
    SemanticsContext& context, ParseTree::Node parse_node) -> bool {
  context.PushScope();
  context.node_stack().Push(parse_node);
  // At this point we aren't sure whether this will be a value or type literal,
  // so we push onto args irrespective. It just won't be used for a type
  // literal.
  context.args_type_info_stack().Push();
  context.ParamOrArgStart();
  return true;
}

auto SemanticsHandleTupleTypeLiteral(SemanticsContext& context,
                                     ParseTree::Node parse_node) -> bool {
  auto refs_id = context.ParamOrArgEnd(
      /*for_args=*/false, ParseNodeKind::ParenExpressionOrTupleLiteralStart);

  context.PopScope();
  context.node_stack().PopAndDiscardSoloParseNode(
      ParseNodeKind::ParenExpressionOrTupleLiteralStart);
  // This is only used for value literals.
  context.args_type_info_stack().Pop();

  CARBON_CHECK(refs_id != SemanticsNodeBlockId::Empty)
      << "() is handled by TupleLiteral.";

  auto type_id = context.CanonicalizeTupleType(parse_node, refs_id);
  context.node_stack().Push(parse_node,
                            context.semantics_ir().GetType(type_id));
  return true;
}

}  // namespace Carbon
