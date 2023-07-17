// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"

namespace Carbon {

auto SemanticsHandleParenExpression(SemanticsContext& context,
                                    ParseTree::Node parse_node) -> bool {
  auto value_id = context.node_stack().PopExpression();
  context.node_stack()
      .PopAndDiscardSoloParseNode<
          ParseNodeKind::ParenExpressionOrTupleLiteralStart>();
  auto value_id = context.node_stack().Pop<SemanticsNodeId>();
  // ParamOrArgStart was called for tuple handling; clean up the ParamOrArg
  // support for non-tuple cases.
  context.ParamOrArgEnd(
      /*for_args=*/true, ParseNodeKind::ParenExpressionOrTupleLiteralStart);
  context.node_stack().PopAndDiscardSoloParseNode(
      ParseNodeKind::ParenExpressionOrTupleLiteralStart);
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
  llvm::SmallVector<SemanticsNodeId> node_blocks;
  auto refs_id = context.ParamOrArgEnd(
      /*for_args=*/true, ParseNodeKind::ParenExpressionOrTupleLiteralStart);

  context.node_stack().PopAndDiscardSoloParseNode(
      ParseNodeKind::ParenExpressionOrTupleLiteralStart);
  if (node_blocks.empty()) {
    node_blocks = context.semantics_ir().GetNodeBlock(refs_id);
  }
  llvm::SmallVector<SemanticsTypeId> type_ids;
  for (auto node : node_blocks) {
    auto type_id = context.semantics_ir().GetNode(node).type_id();
    type_ids.push_back(type_id);
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
