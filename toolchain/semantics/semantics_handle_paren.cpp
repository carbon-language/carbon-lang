// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"

namespace Carbon {

auto SemanticsHandleParenExpression(SemanticsContext& context,
                                    ParseTree::Node parse_node) -> bool {
  auto value_id = context.node_stack().PopExpression();
  // ParamOrArgStart was called for tuple handling; clean up the ParamOrArg
  // support for non-tuple cases.
  context.ParamOrArgEnd(
      /*for_args=*/true, ParseNodeKind::ParenExpressionOrTupleLiteralStart);
  context.node_stack()
      .PopAndDiscardSoloParseNode<
          ParseNodeKind::ParenExpressionOrTupleLiteralStart>();
  context.node_stack().Push(parse_node, value_id);
  return true;
}

auto SemanticsHandleTupleLiteralComma(SemanticsContext& context,
                                      ParseTree::Node /*parse_node*/) -> bool {
  context.ParamOrArgComma(/*for_args=*/true);
  return true;
}

auto SemanticsHandleTupleLiteral(SemanticsContext& context,
                                 ParseTree::Node parse_node) -> bool {
  auto refs_id = context.ParamOrArgEnd(
      /*for_args=*/true, ParseNodeKind::ParenExpressionOrTupleLiteralStart);

  context.node_stack()
      .PopAndDiscardSoloParseNode<
          ParseNodeKind::ParenExpressionOrTupleLiteralStart>();
  const auto& node_block = context.semantics_ir().GetNodeBlock(refs_id);
  llvm::SmallVector<SemanticsTypeId> type_ids;
  type_ids.reserve(node_block.size());
  for (auto node : node_block) {
    type_ids.push_back(context.semantics_ir().GetNode(node).type_id());
  }
  auto type_id = context.CanonicalizeTupleType(parse_node, type_ids);

  auto value_id = context.AddNode(
      SemanticsNode::TupleValue::Make(parse_node, type_id, refs_id));
  context.node_stack().Push(parse_node, value_id);
  return true;
}

auto SemanticsHandleParenExpressionOrTupleLiteralStart(
    SemanticsContext& context, ParseTree::Node parse_node) -> bool {
  context.node_stack().Push(parse_node);
  context.ParamOrArgStart();
  return true;
}

}  // namespace Carbon
