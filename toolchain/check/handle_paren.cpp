// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleParenExpression(Context& context, Parse::Node parse_node) -> bool {
  auto value_id = context.node_stack().PopExpression();
  // ParamOrArgStart was called for tuple handling; clean up the ParamOrArg
  // support for non-tuple cases.
  context.ParamOrArgEnd(Parse::NodeKind::ParenExpressionOrTupleLiteralStart);
  context.node_stack()
      .PopAndDiscardSoloParseNode<
          Parse::NodeKind::ParenExpressionOrTupleLiteralStart>();
  context.node_stack().Push(parse_node, value_id);
  return true;
}

auto HandleParenExpressionOrTupleLiteralStart(Context& context,
                                              Parse::Node parse_node) -> bool {
  context.node_stack().Push(parse_node);
  context.ParamOrArgStart();
  return true;
}

auto HandleTupleLiteralComma(Context& context, Parse::Node /*parse_node*/)
    -> bool {
  context.ParamOrArgComma();
  return true;
}

auto HandleTupleLiteral(Context& context, Parse::Node parse_node) -> bool {
  auto refs_id = context.ParamOrArgEnd(
      Parse::NodeKind::ParenExpressionOrTupleLiteralStart);

  context.node_stack()
      .PopAndDiscardSoloParseNode<
          Parse::NodeKind::ParenExpressionOrTupleLiteralStart>();
  const auto& node_block = context.semantics_ir().GetNodeBlock(refs_id);
  llvm::SmallVector<SemIR::TypeId> type_ids;
  type_ids.reserve(node_block.size());
  for (auto node : node_block) {
    type_ids.push_back(context.semantics_ir().GetNode(node).type_id());
  }
  auto type_id = context.CanonicalizeTupleType(parse_node, std::move(type_ids));

  auto value_id =
      context.AddNode(SemIR::TupleLiteral(parse_node, type_id, refs_id));
  context.node_stack().Push(parse_node, value_id);
  return true;
}

}  // namespace Carbon::Check
