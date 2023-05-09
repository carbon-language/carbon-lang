// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"

namespace Carbon {

auto SemanticsHandleStructComma(SemanticsContext& context,
                                ParseTree::Node /*parse_node*/) -> bool {
  context.ParamOrArgComma(
      /*for_args=*/context.parse_tree().node_kind(
          context.node_stack().PeekParseNode()) !=
      ParseNodeKind::StructFieldType);
  return true;
}

auto SemanticsHandleStructFieldDesignator(SemanticsContext& context,
                                          ParseTree::Node /*parse_node*/)
    -> bool {
  // This leaves the designated name on top because the `.` isn't interesting.
  CARBON_CHECK(
      context.parse_tree().node_kind(context.node_stack().PeekParseNode()) ==
      ParseNodeKind::DesignatedName);
  return true;
}

auto SemanticsHandleStructFieldType(SemanticsContext& context,
                                    ParseTree::Node parse_node) -> bool {
  auto [type_node, type_id] = context.node_stack().PopForParseNodeAndNodeId();
  SemanticsNodeId cast_type_id = context.ImplicitAsRequired(
      type_node, type_id, SemanticsNodeId::BuiltinTypeType);

  auto [name_node, name_id] = context.node_stack().PopForParseNodeAndNameId(
      ParseNodeKind::DesignatedName);

  context.AddNode(
      SemanticsNode::StructTypeField::Make(name_node, cast_type_id, name_id));
  context.node_stack().Push(parse_node);
  return true;
}

auto SemanticsHandleStructFieldUnknown(SemanticsContext& context,
                                       ParseTree::Node parse_node) -> bool {
  return context.TODO(parse_node, "HandleStructFieldUnknown");
}

auto SemanticsHandleStructFieldValue(SemanticsContext& context,
                                     ParseTree::Node parse_node) -> bool {
  auto [value_parse_node, value_node_id] =
      context.node_stack().PopForParseNodeAndNodeId();
  auto [_, name_id] = context.node_stack().PopForParseNodeAndNameId(
      ParseNodeKind::DesignatedName);

  // Store the name for the type.
  auto type_block_id = context.args_type_info_stack().PeekForAdd();
  context.semantics().AddNode(
      type_block_id,
      SemanticsNode::StructTypeField::Make(
          parse_node, context.semantics().GetNode(value_node_id).type_id(),
          name_id));

  // Push the value back on the stack as an argument.
  context.node_stack().Push(parse_node, value_node_id);
  return true;
}

auto SemanticsHandleStructLiteral(SemanticsContext& context,
                                  ParseTree::Node parse_node) -> bool {
  auto [ir_id, refs_id] = context.ParamOrArgEnd(
      /*for_args=*/true, ParseNodeKind::StructLiteralOrStructTypeLiteralStart);

  context.PopScope();
  context.node_stack().PopAndDiscardSoloParseNode(
      ParseNodeKind::StructLiteralOrStructTypeLiteralStart);
  auto type_block_id = context.args_type_info_stack().Pop();

  // Special-case `{}`.
  if (refs_id == SemanticsNodeBlockId::Empty) {
    context.node_stack().Push(parse_node, SemanticsNodeId::BuiltinEmptyStruct);
    return true;
  }

  // Construct a type for the literal. Each field is one node, so ir_id and
  // refs_id match.
  auto refs = context.semantics().GetNodeBlock(refs_id);
  auto type_id = context.AddNode(SemanticsNode::StructType::Make(
      parse_node, type_block_id, type_block_id));

  auto value_id = context.AddNode(
      SemanticsNode::StructValue::Make(parse_node, type_id, ir_id, refs_id));
  context.node_stack().Push(parse_node, value_id);
  return true;
}

auto SemanticsHandleStructLiteralOrStructTypeLiteralStart(
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

auto SemanticsHandleStructTypeLiteral(SemanticsContext& context,
                                      ParseTree::Node parse_node) -> bool {
  auto [ir_id, refs_id] = context.ParamOrArgEnd(
      /*for_args=*/false, ParseNodeKind::StructLiteralOrStructTypeLiteralStart);

  context.PopScope();
  context.node_stack().PopAndDiscardSoloParseNode(
      ParseNodeKind::StructLiteralOrStructTypeLiteralStart);
  // This is only used for value literals.
  context.args_type_info_stack().Pop();

  CARBON_CHECK(refs_id != SemanticsNodeBlockId::Empty)
      << "{} is handled by StructLiteral.";

  auto type_id = context.AddNode(
      SemanticsNode::StructType::Make(parse_node, ir_id, refs_id));
  context.node_stack().Push(parse_node, type_id);
  return true;
}

}  // namespace Carbon
