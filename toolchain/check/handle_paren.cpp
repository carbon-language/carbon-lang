// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleParenExpression(Context& context, Parse::Lamp parse_node) -> bool {
  auto value_id = context.lamp_stack().PopExpression();
  // ParamOrArgStart was called for tuple handling; clean up the ParamOrArg
  // support for non-tuple cases.
  context.ParamOrArgEnd(Parse::LampKind::ParenExpressionOrTupleLiteralStart);
  context.lamp_stack()
      .PopAndDiscardSoloParseNode<
          Parse::LampKind::ParenExpressionOrTupleLiteralStart>();
  context.lamp_stack().Push(parse_node, value_id);
  return true;
}

auto HandleParenExpressionOrTupleLiteralStart(Context& context,
                                              Parse::Lamp parse_node) -> bool {
  context.lamp_stack().Push(parse_node);
  context.ParamOrArgStart();
  return true;
}

auto HandleTupleLiteralComma(Context& context, Parse::Lamp /*parse_node*/)
    -> bool {
  context.ParamOrArgComma();
  return true;
}

auto HandleTupleLiteral(Context& context, Parse::Lamp parse_node) -> bool {
  auto refs_id = context.ParamOrArgEnd(
      Parse::LampKind::ParenExpressionOrTupleLiteralStart);

  context.lamp_stack()
      .PopAndDiscardSoloParseNode<
          Parse::LampKind::ParenExpressionOrTupleLiteralStart>();
  const auto& inst_block = context.inst_blocks().Get(refs_id);
  llvm::SmallVector<SemIR::TypeId> type_ids;
  type_ids.reserve(inst_block.size());
  for (auto node : inst_block) {
    type_ids.push_back(context.insts().Get(node).type_id());
  }
  auto type_id = context.CanonicalizeTupleType(parse_node, std::move(type_ids));

  auto value_id =
      context.AddNode(SemIR::TupleLiteral{parse_node, type_id, refs_id});
  context.lamp_stack().Push(parse_node, value_id);
  return true;
}

}  // namespace Carbon::Check
