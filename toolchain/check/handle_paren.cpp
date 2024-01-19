// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleExprOpenParen(Context& context, Parse::ExprOpenParenId parse_node)
    -> bool {
  context.node_stack().Push(parse_node);
  context.ParamOrArgStart();
  return true;
}

auto HandleParenExpr(Context& context, Parse::ParenExprId parse_node) -> bool {
  auto value_id = context.node_stack().PopExpr();
  // ParamOrArgStart was called for tuple handling; clean up the ParamOrArg
  // support for non-tuple cases.
  context.ParamOrArgEnd(Parse::NodeKind::ExprOpenParen);
  context.node_stack()
      .PopAndDiscardSoloParseNode<Parse::NodeKind::ExprOpenParen>();
  context.node_stack().Push(parse_node, value_id);
  return true;
}

auto HandleTupleLiteralComma(Context& context,
                             Parse::TupleLiteralCommaId /*parse_node*/)
    -> bool {
  context.ParamOrArgComma();
  return true;
}

auto HandleTupleLiteral(Context& context, Parse::TupleLiteralId parse_node)
    -> bool {
  auto refs_id = context.ParamOrArgEnd(Parse::NodeKind::ExprOpenParen);

  context.node_stack()
      .PopAndDiscardSoloParseNode<Parse::NodeKind::ExprOpenParen>();
  const auto& inst_block = context.inst_blocks().Get(refs_id);
  llvm::SmallVector<SemIR::TypeId> type_ids;
  type_ids.reserve(inst_block.size());
  for (auto inst : inst_block) {
    type_ids.push_back(context.insts().Get(inst).type_id());
  }
  auto type_id = context.GetTupleType(type_ids);

  auto value_id =
      context.AddInst({parse_node, SemIR::TupleLiteral{type_id, refs_id}});
  context.node_stack().Push(parse_node, value_id);
  return true;
}

}  // namespace Carbon::Check
