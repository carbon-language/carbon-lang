// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"

namespace Carbon::Check {

auto HandleStructComma(Context& context, ParseTree::Node /*parse_node*/)
    -> bool {
  context.ParamOrArgComma(
      /*for_args=*/context.parse_tree().node_kind(
          context.node_stack().PeekParseNode()) !=
      ParseNodeKind::StructFieldType);
  return true;
}

auto HandleStructFieldDesignator(Context& context,
                                 ParseTree::Node /*parse_node*/) -> bool {
  // This leaves the designated name on top because the `.` isn't interesting.
  CARBON_CHECK(
      context.parse_tree().node_kind(context.node_stack().PeekParseNode()) ==
      ParseNodeKind::Name);
  return true;
}

auto HandleStructFieldType(Context& context, ParseTree::Node parse_node)
    -> bool {
  auto [type_node, type_id] = context.node_stack().PopExpressionWithParseNode();
  SemIR::TypeId cast_type_id = context.ExpressionAsType(type_node, type_id);

  auto [name_node, name_id] =
      context.node_stack().PopWithParseNode<ParseNodeKind::Name>();

  context.AddNodeAndPush(parse_node, SemIR::Node::StructTypeField::Make(
                                         name_node, name_id, cast_type_id));
  return true;
}

auto HandleStructFieldUnknown(Context& context, ParseTree::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleStructFieldUnknown");
}

auto HandleStructFieldValue(Context& context, ParseTree::Node parse_node)
    -> bool {
  auto [value_parse_node, value_node_id] =
      context.node_stack().PopExpressionWithParseNode();
  SemIR::StringId name_id = context.node_stack().Pop<ParseNodeKind::Name>();

  // Store the name for the type.
  auto type_block_id = context.args_type_info_stack().PeekForAdd();
  context.semantics_ir().AddNode(
      type_block_id,
      SemIR::Node::StructTypeField::Make(
          parse_node, name_id,
          context.semantics_ir().GetNode(value_node_id).type_id()));

  // Push the value back on the stack as an argument.
  context.node_stack().Push(parse_node, value_node_id);
  return true;
}

auto HandleStructLiteral(Context& context, ParseTree::Node parse_node) -> bool {
  auto refs_id = context.ParamOrArgEnd(
      /*for_args=*/true, ParseNodeKind::StructLiteralOrStructTypeLiteralStart);

  context.PopScope();
  context.node_stack()
      .PopAndDiscardSoloParseNode<
          ParseNodeKind::StructLiteralOrStructTypeLiteralStart>();
  auto type_block_id = context.args_type_info_stack().Pop();

  auto type_id = context.CanonicalizeStructType(parse_node, type_block_id);

  auto value_id = context.AddNode(
      SemIR::Node::StructValue::Make(parse_node, type_id, refs_id));
  context.node_stack().Push(parse_node, value_id);
  return true;
}

auto HandleStructLiteralOrStructTypeLiteralStart(Context& context,
                                                 ParseTree::Node parse_node)
    -> bool {
  context.PushScope();
  context.node_stack().Push(parse_node);
  // At this point we aren't sure whether this will be a value or type literal,
  // so we push onto args irrespective. It just won't be used for a type
  // literal.
  context.args_type_info_stack().Push();
  context.ParamOrArgStart();
  return true;
}

auto HandleStructTypeLiteral(Context& context, ParseTree::Node parse_node)
    -> bool {
  auto refs_id = context.ParamOrArgEnd(
      /*for_args=*/false, ParseNodeKind::StructLiteralOrStructTypeLiteralStart);

  context.PopScope();
  context.node_stack()
      .PopAndDiscardSoloParseNode<
          ParseNodeKind::StructLiteralOrStructTypeLiteralStart>();
  // This is only used for value literals.
  context.args_type_info_stack().Pop();

  CARBON_CHECK(refs_id != SemIR::NodeBlockId::Empty)
      << "{} is handled by StructLiteral.";

  context.AddNodeAndPush(parse_node,
                         SemIR::Node::StructType::Make(
                             parse_node, SemIR::TypeId::TypeType, refs_id));
  return true;
}

}  // namespace Carbon::Check
